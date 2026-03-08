#!/usr/bin/env python3
"""
Simple HTTP server for Qwen3-TTS.

Usage:
    python server.py [--host 0.0.0.0] [--port 8888] [--model PATH]

    Then tunnel with:
        cloudflared tunnel --url http://localhost:8888
"""

import argparse
import base64
import io
import json
import math
import os
import re
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_tts import Qwen3TTSModel


# ---------------------------------------------------------------------------
# Performance statistics tracker
# ---------------------------------------------------------------------------
class PerfStats:
    """Thread-safe accumulator for timing measurements with percentile stats."""

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
        self._times: list[float] = []  # in ms

    def record(self, elapsed_ms: float):
        with self._lock:
            self._times.append(elapsed_ms)

    def count(self) -> int:
        with self._lock:
            return len(self._times)

    def summary(self) -> str | None:
        with self._lock:
            if not self._times:
                return None
            n = len(self._times)
            arr = sorted(self._times)
            mean = sum(arr) / n
            mn, mx = arr[0], arr[-1]
            # standard deviation
            if n > 1:
                variance = sum((x - mean) ** 2 for x in arr) / (n - 1)
                std = math.sqrt(variance)
            else:
                std = 0.0

            def percentile(p):
                k = (p / 100) * (n - 1)
                f = int(k)
                c = f + 1 if f + 1 < n else f
                d = k - f
                return arr[f] + d * (arr[c] - arr[f])

            p50 = percentile(50)
            p75 = percentile(75)
            p90 = percentile(90)
            p95 = percentile(95)
            p99 = percentile(99)

            lines = [
                f"--- [{self.name}] Performance Stats ---",
                f"  Calls:  {n}",
                f"  Avg:    {mean:,.2f} ms",
                f"  Min:    {mn:,.2f} ms",
                f"  Max:    {mx:,.2f} ms",
                f"  StdDev: {std:,.2f} ms",
                f"  P50:    {p50:,.2f} ms",
                f"  P75:    {p75:,.2f} ms",
                f"  P90:    {p90:,.2f} ms",
                f"  P95:    {p95:,.2f} ms",
                f"  P99:    {p99:,.2f} ms",
                f"---",
            ]
            return "\n".join(lines)

    def to_dict(self) -> dict:
        with self._lock:
            if not self._times:
                return {"name": self.name, "count": 0}
            n = len(self._times)
            arr = sorted(self._times)
            mean = sum(arr) / n
            if n > 1:
                variance = sum((x - mean) ** 2 for x in arr) / (n - 1)
                std = math.sqrt(variance)
            else:
                std = 0.0

            def percentile(p):
                k = (p / 100) * (n - 1)
                f = int(k)
                c = f + 1 if f + 1 < n else f
                d = k - f
                return arr[f] + d * (arr[c] - arr[f])

            return {
                "name": self.name,
                "count": n,
                "avg_ms": round(mean, 2),
                "min_ms": round(arr[0], 2),
                "max_ms": round(arr[-1], 2),
                "std_ms": round(std, 2),
                "p50_ms": round(percentile(50), 2),
                "p75_ms": round(percentile(75), 2),
                "p90_ms": round(percentile(90), 2),
                "p95_ms": round(percentile(95), 2),
                "p99_ms": round(percentile(99), 2),
            }


# Per-operation trackers
PERF_GENERATE = PerfStats("generate")
PERF_STREAM_TOTAL = PerfStats("stream_total")
PERF_STREAM_FIRST = PerfStats("stream_first_chunk")
PERF_TRANSLATE = PerfStats("translate")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STYLES = ["neutral", "bright", "calm", "angry"]
LANGS = ["ja", "en", "ch"]

# language code -> model language value for generate
LANG_TO_MODEL = {"ja": "Japanese", "en": "English", "ch": "Chinese"}

# full language name (scenario JSON) -> language code
LANG_NAME_TO_CODE = {
    "japanese": "ja", "english": "en", "chinese": "ch",
    "ja": "ja", "en": "en", "ch": "ch",
}

# language code -> full name for translation prompts
TRANSLATE_LANG_MAP = {"ja": "Japanese", "en": "English", "ch": "Chinese"}

REF_TEXTS = {
    "ja": {
        "neutral": (
            "本日のトップニュースです。"
            "新しい再生可能エネルギー技術が発表されました。"
            "この技術は、発電効率を高めながらコストを削減できると期待されています。"
            "関係者によりますと、今年中に実証実験を開始し、"
            "来年春の本格導入を目指しているとのことです。"
        ),
        "bright": (
            "みなさん、こんにちは！"
            "今日はとても嬉しいお知らせがあります！"
            "これまで努力してきたことが、ついに大きな成果につながりました。"
            "みなさんの応援があったからこそです。"
            "これからも一緒に、もっと素敵な未来を作っていきましょう！"
        ),
        "calm": (
            "少しだけ、深呼吸をしてみましょう。"
            "これまでの歩みを、ゆっくり振り返ってみてください。"
            "焦らなくても大丈夫です。"
            "一歩ずつ、確実に前へ進めばいいのです。"
            "今できることに、静かに集中してみましょう。"
        ),
        "angry": (
            "この状況は、到底受け入れられるものではありません。"
            "何度も問題を指摘してきたにもかかわらず、"
            "改善はまったく見られませんでした。"
            "私たちは、曖昧な説明ではなく、"
            "明確な回答と、直ちに具体的な対応を求めます。"
        ),
    },
    "en": {
        "neutral": (
            "Good evening. Today's top story: Researchers have announced a breakthrough "
            "in renewable energy technology. The new system is expected to improve efficiency "
            "while reducing costs for consumers. Officials say the project will begin testing "
            "later this year, with full deployment planned for next spring."
        ),
        "bright": (
            "Hello everyone! I'm so excited to share this wonderful news with you today. "
            "We've reached an important milestone, and it's all thanks to your incredible support. "
            "This achievement opens the door to new opportunities, and I truly can't wait to see "
            "what we accomplish next together!"
        ),
        "calm": (
            "Let's take a quiet moment to reflect on the journey we've taken so far. "
            "Step by step, we have learned, adapted, and grown stronger. "
            "There is no need to rush. Progress comes steadily, with patience and care. "
            "Breathe deeply, and focus on one small task at a time."
        ),
        "angry": (
            "This situation is completely unacceptable. We have raised this issue "
            "again and again, yet nothing has changed. People are tired of empty promises "
            "and repeated delays. We demand clear answers, and we expect immediate action "
            "to fix this problem once and for all."
        ),
    },
    "ch": {
        "neutral": "请给我一杯水，谢谢。",
        "bright": "请给我一杯水，谢谢。",
        "calm": "请给我一杯水，谢谢。",
        "angry": "请给我一杯水，谢谢。",
    },
}

REFERENCES_DIR = Path("./references")

# ---------------------------------------------------------------------------
# Globals (set in main)
# ---------------------------------------------------------------------------
TTS: Qwen3TTSModel = None
MODEL_KIND: str = ""

# Translation model (Qwen3-8B-Instruct)
TRANSLATOR_MODEL = None
TRANSLATOR_TOKENIZER = None

# prompt cache:  key = "{username}/{lang}/{style}"  ->  list[VoiceClonePromptItem]
PROMPT_CACHE: dict = {}

# reference paths: key -> wav file path  (discovered at startup, built lazily)
REFERENCE_PATHS: dict = {}

# ---------------------------------------------------------------------------
# Prompt cache helpers
# ---------------------------------------------------------------------------

def _cache_key(username: str, lang: str, style: str) -> str:
    return f"{username}/{lang}/{style}"


def _build_prompt(wav_path: str, lang: str, style: str):
    """Build voice_clone_prompt from a wav file and return it."""
    ref_text = REF_TEXTS[lang][style]
    items = TTS.create_voice_clone_prompt(
        ref_audio=str(wav_path),
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    return items


def load_all_references():
    """Scan ./references/ and register wav paths (prompt is built lazily on first use)."""
    if not REFERENCES_DIR.exists():
        REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
        print("[references] Created ./references/ directory.")
        return

    count = 0
    for user_dir in sorted(REFERENCES_DIR.iterdir()):
        if not user_dir.is_dir():
            continue
        username = user_dir.name
        for lang in LANGS:
            lang_dir = user_dir / lang
            if not lang_dir.is_dir():
                continue
            for style in STYLES:
                wav_path = lang_dir / f"{style}.wav"
                if wav_path.exists():
                    key = _cache_key(username, lang, style)
                    REFERENCE_PATHS[key] = str(wav_path)
                    count += 1

    print(f"[references] Found {count} reference(s) in {REFERENCES_DIR}")


def get_prompt(username: str, lang: str, style: str):
    """Return cached prompt, building it lazily from REFERENCE_PATHS if needed."""
    key = _cache_key(username, lang, style)
    if key in PROMPT_CACHE:
        return PROMPT_CACHE[key]
    wav_path = REFERENCE_PATHS.get(key)
    if wav_path is None:
        return None
    print(f"[cache] Building prompt for {key} (first use) ...")
    t0 = time.time()
    items = _build_prompt(wav_path, lang, style)
    PROMPT_CACHE[key] = items
    elapsed = time.time() - t0
    print(f"[cache] {key} OK ({elapsed:.2f}s)")
    return items


def save_and_cache_reference(username: str, lang: str, style: str, wav_bytes: bytes):
    """Save wav to disk and (re)build prompt cache entry."""
    lang_dir = REFERENCES_DIR / username / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    wav_path = lang_dir / f"{style}.wav"
    wav_path.write_bytes(wav_bytes)
    print(f"[upload] Saved {wav_path} ({len(wav_bytes)} bytes)")

    key = _cache_key(username, lang, style)
    items = _build_prompt(str(wav_path), lang, style)
    PROMPT_CACHE[key] = items
    print(f"[cache] {key} updated")


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_text(src: str, inlng: str, outlng: str) -> str:
    """Translate src text from inlng to outlng using the translator model."""
    if TRANSLATOR_MODEL is None:
        raise RuntimeError("Translation model not loaded (use --translate-model)")

    in_name = TRANSLATE_LANG_MAP[inlng]
    out_name = TRANSLATE_LANG_MAP[outlng]

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a professional translator. "
                f"Translate the following {in_name} text into natural, fluent {out_name}. "
                f"Output ONLY the translated text, no explanations."
            ),
        },
        {"role": "user", "content": src},
    ]

    text_input = TRANSLATOR_TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,  # disable Qwen3 thinking mode
    )

    # find actual device of first parameter
    device = next(TRANSLATOR_MODEL.parameters()).device
    inputs = TRANSLATOR_TOKENIZER([text_input], return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = TRANSLATOR_MODEL.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    # strip the prompt tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    result = TRANSLATOR_TOKENIZER.decode(generated, skip_special_tokens=True).strip()

    # safety: strip any leftover <think>...</think> tags
    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()

    return result


# ---------------------------------------------------------------------------
# HTML — loaded from server.html at startup
# ---------------------------------------------------------------------------
INDEX_HTML: str = ""


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
class TTSHandler(BaseHTTPRequestHandler):

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_wav(self, wav: np.ndarray, sr: int):
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        data = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw)

    # --- routes ---

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", ""):
            self._send_html(INDEX_HTML)
        elif path == "/api/info":
            self._handle_info()
        elif path == "/api/user_status":
            qs = parse_qs(parsed.query)
            username = (qs.get("username", [""])[0]).strip()
            self._handle_user_status(username)
        elif path == "/api/reference_audio":
            qs = parse_qs(parsed.query)
            self._handle_reference_audio(qs)
        elif path == "/api/translate":
            qs = parse_qs(parsed.query)
            self._handle_translate(qs)
        elif path == "/api/stats":
            self._handle_stats()
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/upload_reference":
                self._handle_upload_reference()
            elif path == "/api/generate":
                self._handle_generate()
            elif path == "/api/generate_stream":
                self._handle_generate_stream()
            elif path == "/api/generate_scenario_stream":
                self._handle_generate_scenario_stream()
            else:
                self.send_error(404)
        except Exception as e:
            self._send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    # --- handlers ---

    def _handle_info(self):
        all_keys = set(PROMPT_CACHE.keys()) | set(REFERENCE_PATHS.keys())
        self._send_json({
            "model_kind": MODEL_KIND,
            "translator_available": TRANSLATOR_MODEL is not None,
            "cached_keys": sorted(all_keys),
        })

    def _handle_reference_audio(self, qs):
        username = (qs.get("username", [""])[0]).strip()
        lang = (qs.get("lang", [""])[0]).strip()
        style = (qs.get("style", [""])[0]).strip()

        if not username or not re.match(r'^[a-zA-Z0-9_-]+$', username):
            self.send_error(400, "Invalid username")
            return
        if lang not in LANGS or style not in STYLES:
            self.send_error(400, "Invalid lang or style")
            return

        wav_path = REFERENCES_DIR / username / lang / f"{style}.wav"
        if not wav_path.exists():
            self.send_error(404, "Reference audio not found")
            return

        data = wav_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_user_status(self, username):
        if not username:
            self._send_json({"error": "username required"}, 400)
            return
        styles = {}
        for lang in LANGS:
            styles[lang] = {}
            for style in STYLES:
                key = _cache_key(username, lang, style)
                styles[lang][style] = key in PROMPT_CACHE or key in REFERENCE_PATHS
        self._send_json({"username": username, "styles": styles})

    def _handle_translate(self, qs):
        src = (qs.get("src", [""])[0]).strip()
        inlng = (qs.get("inlng", [""])[0]).strip()
        outlng = (qs.get("outlng", [""])[0]).strip()

        print(f"[translate] REQ  {inlng}->{outlng}  src={src!r}")

        if not src:
            print(f"[translate] ERR  empty src")
            self._send_json({"error": "src is required"}, 400)
            return
        if inlng not in LANGS:
            print(f"[translate] ERR  invalid inlng={inlng!r}")
            self._send_json({"error": f"Invalid inlng: {inlng}"}, 400)
            return
        if outlng not in LANGS:
            print(f"[translate] ERR  invalid outlng={outlng!r}")
            self._send_json({"error": f"Invalid outlng: {outlng}"}, 400)
            return
        if inlng == outlng:
            print(f"[translate] SKIP same lang ({inlng})")
            self._send_json({"translated": src, "inlng": inlng, "outlng": outlng})
            return

        try:
            t0 = time.time()
            result = translate_text(src, inlng, outlng)
            elapsed = time.time() - t0
            PERF_TRANSLATE.record(elapsed * 1000)
            print(f"[translate] OK   {elapsed:.2f}s  {inlng}->{outlng}")
            print(f"[translate]  IN:  {src!r}")
            print(f"[translate]  OUT: {result!r}")
            self._send_json({"translated": result, "inlng": inlng, "outlng": outlng})
        except Exception as e:
            print(f"[translate] FAIL {type(e).__name__}: {e}")
            self._send_json({"error": f"Translation failed: {e}"}, 500)

    def _handle_stats(self):
        self._send_json({
            "generate": PERF_GENERATE.to_dict(),
            "stream_total": PERF_STREAM_TOTAL.to_dict(),
            "stream_first_chunk": PERF_STREAM_FIRST.to_dict(),
            "translate": PERF_TRANSLATE.to_dict(),
        })

    def _handle_upload_reference(self):
        body = self._read_json_body()
        username = body.get("username", "").strip()
        lang = body.get("lang", "").strip()
        style = body.get("style", "").strip()
        wav_b64 = body.get("wav_base64", "")

        if not username or not re.match(r'^[a-zA-Z0-9_-]+$', username):
            self._send_json({"error": "Invalid username"}, 400)
            return
        if lang not in LANGS:
            self._send_json({"error": f"Invalid lang: {lang}"}, 400)
            return
        if style not in STYLES:
            self._send_json({"error": f"Invalid style: {style}"}, 400)
            return
        if not wav_b64:
            self._send_json({"error": "No audio data"}, 400)
            return

        wav_bytes = base64.b64decode(wav_b64)
        print(f"[upload]   REQ  user={username} lang={lang} style={style} size={len(wav_bytes)} bytes")
        save_and_cache_reference(username, lang, style, wav_bytes)
        print(f"[upload]   OK   key={_cache_key(username, lang, style)}")
        self._send_json({"ok": True, "key": _cache_key(username, lang, style)})

    def _handle_generate(self):
        body = self._read_json_body()
        username = body["username"]
        style = body["style"]
        ref_lang = body.get("ref_lang", "ja")
        out_lang = body.get("out_lang", "ja")
        text = body["text"]

        print(f"[generate] REQ  user={username} style={style} ref={ref_lang} out={out_lang}")
        print(f"[generate]  TEXT: {text!r}")

        if ref_lang not in LANGS:
            print(f"[generate] ERR  invalid ref_lang={ref_lang!r}")
            self._send_json({"error": f"Invalid ref_lang: {ref_lang}"}, 400)
            return
        if out_lang not in LANGS:
            print(f"[generate] ERR  invalid out_lang={out_lang!r}")
            self._send_json({"error": f"Invalid out_lang: {out_lang}"}, 400)
            return

        key = _cache_key(username, ref_lang, style)
        prompt_items = get_prompt(username, ref_lang, style)
        if prompt_items is None:
            print(f"[generate] ERR  prompt not found: {key}")
            self._send_json(
                {"error": f"Prompt not found: {key}. Upload reference audio first."},
                404,
            )
            return

        model_language = LANG_TO_MODEL[out_lang]

        print(f"[generate] START model_lang={model_language} prompt_key={key}")
        t0 = time.time()
        wavs, sr = TTS.generate_voice_clone(
            text=text,
            language=model_language,
            voice_clone_prompt=prompt_items,
        )
        elapsed = time.time() - t0
        elapsed_ms = elapsed * 1000
        PERF_GENERATE.record(elapsed_ms)
        duration = len(wavs[0]) / sr
        print(f"[generate] OK   {elapsed:.2f}s  audio={duration:.1f}s ({sr}Hz, {len(wavs[0])} samples)")
        summary = PERF_GENERATE.summary()
        if summary:
            print(summary)
        self._send_wav(wavs[0], sr)

    def _handle_generate_stream(self):
        body = self._read_json_body()
        username = body["username"]
        style = body["style"]
        ref_lang = body.get("ref_lang", "ja")
        out_lang = body.get("out_lang", "ja")
        text = body["text"]

        print(f"[stream]   REQ  user={username} style={style} ref={ref_lang} out={out_lang}")
        print(f"[stream]    TEXT: {text!r}")

        if ref_lang not in LANGS:
            self._send_json({"error": f"Invalid ref_lang: {ref_lang}"}, 400)
            return
        if out_lang not in LANGS:
            self._send_json({"error": f"Invalid out_lang: {out_lang}"}, 400)
            return

        key = _cache_key(username, ref_lang, style)
        prompt_items = get_prompt(username, ref_lang, style)
        if prompt_items is None:
            self._send_json(
                {"error": f"Prompt not found: {key}. Upload reference audio first."},
                404,
            )
            return

        model_language = LANG_TO_MODEL[out_lang]

        # Send headers immediately so client can start reading chunks
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("X-Sample-Rate", "24000")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        print(f"[stream]   START model_lang={model_language} prompt_key={key}")
        t0 = time.time()
        t_first = None
        chunk_count = 0
        total_samples = 0

        try:
            for chunk, sr in TTS.stream_generate_voice_clone(
                text=text,
                language=model_language,
                voice_clone_prompt=prompt_items,
                emit_every_frames=8,
                decode_window_frames=80,
                overlap_samples=0,
            ):
                if t_first is None:
                    t_first = time.time()
                    print(f"[stream]   FIRST CHUNK in {t_first - t0:.2f}s  ({len(chunk)} samples)")

                # Convert float32 -> int16 PCM bytes
                pcm_int16 = np.clip(chunk, -1.0, 1.0)
                pcm_int16 = (pcm_int16 * 32767).astype(np.int16)
                self.wfile.write(pcm_int16.tobytes())
                self.wfile.flush()
                chunk_count += 1
                total_samples += len(chunk)
        except BrokenPipeError:
            print(f"[stream]   Client disconnected after {chunk_count} chunks")
            return

        elapsed = time.time() - t0
        elapsed_ms = elapsed * 1000
        PERF_STREAM_TOTAL.record(elapsed_ms)
        if t_first is not None:
            first_ms = (t_first - t0) * 1000
            PERF_STREAM_FIRST.record(first_ms)
        duration = total_samples / 24000
        first_latency = (t_first - t0) if t_first else elapsed
        print(f"[stream]   DONE {elapsed:.2f}s  first={first_latency:.2f}s  chunks={chunk_count}  audio={duration:.1f}s ({total_samples} samples)")
        for stats in (PERF_STREAM_TOTAL, PERF_STREAM_FIRST):
            summary = stats.summary()
            if summary:
                print(summary)

    def _handle_generate_scenario_stream(self):
        body = self._read_json_body()
        scenario = body.get("scenario", [])

        if not scenario or not isinstance(scenario, list):
            self._send_json({"error": "scenario array is required"}, 400)
            return

        print(f"[scenario] REQ  {len(scenario)} entries")

        # Validate and resolve all entries first
        resolved = []
        for i, entry in enumerate(scenario):
            user_id = entry.get("userId", "").strip()
            style = entry.get("style", "").strip()
            lang_out_raw = entry.get("langOut", "").strip().lower()
            lang_in_raw = entry.get("langIn", "").strip().lower()
            text = entry.get("text", "")  # preserve spaces for quit

            # "quit" style = silence entry: count half-width spaces × 30ms
            if style == "quit":
                space_count = text.count(" ")
                silence_ms = space_count * 30
                silence_samples = int(24000 * silence_ms / 1000)
                resolved.append({
                    "index": i,
                    "silence_samples": silence_samples,
                    "silence_ms": silence_ms,
                })
                continue

            text = text.strip()
            lang_out = LANG_NAME_TO_CODE.get(lang_out_raw)
            lang_in = LANG_NAME_TO_CODE.get(lang_in_raw)

            if not user_id or not style or not lang_out or not lang_in or not text:
                self._send_json(
                    {"error": f"Entry {i}: invalid fields (userId={user_id}, style={style}, langOut={lang_out_raw}, langIn={lang_in_raw})"},
                    400,
                )
                return
            if style not in STYLES:
                self._send_json({"error": f"Entry {i}: unknown style '{style}'"}, 400)
                return

            key = _cache_key(user_id, lang_out, style)
            prompt_items = get_prompt(user_id, lang_out, style)
            if prompt_items is None:
                self._send_json(
                    {"error": f"Entry {i}: prompt not found: {key}. Upload reference audio first."},
                    404,
                )
                return

            resolved.append({
                "index": i,
                "user_id": user_id,
                "style": style,
                "lang_out": lang_out,
                "lang_in": lang_in,
                "text": text,
                "prompt_items": prompt_items,
            })

        # Send headers — stream PCM int16 data with JSON progress lines in X-Progress header
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("X-Sample-Rate", "24000")
        self.send_header("X-Scenario-Count", str(len(resolved)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        t0 = time.time()
        total_samples = 0

        try:
            for item in resolved:
                idx = item["index"]

                # Handle silence (quit) entries
                if "silence_samples" in item:
                    n_samples = item["silence_samples"]
                    if n_samples > 0:
                        silence_pcm = np.zeros(n_samples, dtype=np.int16)
                        self.wfile.write(silence_pcm.tobytes())
                        self.wfile.flush()
                        total_samples += n_samples
                    print(f"[scenario] #{idx} silence {item['silence_ms']}ms ({n_samples} samples)")
                    continue

                gen_text = item["text"]

                # Translate if input language differs from output language
                if item["lang_in"] != item["lang_out"] and TRANSLATOR_MODEL is not None:
                    print(f"[scenario] #{idx} translating {item['lang_in']}->{item['lang_out']}")
                    gen_text = translate_text(gen_text, item["lang_in"], item["lang_out"])
                    print(f"[scenario] #{idx} translated: {gen_text!r}")

                model_language = LANG_TO_MODEL[item["lang_out"]]
                print(f"[scenario] #{idx} generating user={item['user_id']} style={item['style']} lang={model_language}")
                print(f"[scenario] #{idx}  TEXT: {gen_text!r}")

                for chunk, sr in TTS.stream_generate_voice_clone(
                    text=gen_text,
                    language=model_language,
                    voice_clone_prompt=item["prompt_items"],
                    emit_every_frames=8,
                    decode_window_frames=80,
                    overlap_samples=0,
                ):
                    pcm_int16 = np.clip(chunk, -1.0, 1.0)
                    pcm_int16 = (pcm_int16 * 32767).astype(np.int16)
                    self.wfile.write(pcm_int16.tobytes())
                    self.wfile.flush()
                    total_samples += len(chunk)

                print(f"[scenario] #{idx} done")

        except BrokenPipeError:
            print(f"[scenario] Client disconnected")
            return

        elapsed = time.time() - t0
        duration = total_samples / 24000
        print(f"[scenario] ALL DONE {elapsed:.2f}s  audio={duration:.1f}s ({total_samples} samples)")

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global TTS, MODEL_KIND, INDEX_HTML, TRANSLATOR_MODEL, TRANSLATOR_TOKENIZER

    # Load HTML from server.html (same directory as server.py)
    html_path = Path(__file__).parent / "server.html"
    INDEX_HTML = html_path.read_text(encoding="utf-8")
    print(f"[html] Loaded {html_path}")

    parser = argparse.ArgumentParser(description="Qwen3-TTS HTTP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8888, help="Port (default: 8888)")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="Model path or HF repo id")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention")
    parser.add_argument("--translate-model", default="Qwen/Qwen3-8B",
                        help="Translation model (default: Qwen/Qwen3-8B). Set to 'none' to disable.")
    parser.add_argument("--translate-device", default=None,
                        help="Device for translator (default: same as --device)")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    attn = None if args.no_flash_attn else "flash_attention_2"

    print(f"Loading model: {args.model} on {args.device} ({args.dtype}) ...")
    TTS = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype_map[args.dtype],
        attn_implementation=attn,
    )
    MODEL_KIND = getattr(TTS.model, "tts_model_type", "base")
    print(f"Model loaded. type={MODEL_KIND}")

    # Load translation model
    if args.translate_model and args.translate_model.lower() != "none":
        tr_device = args.translate_device or args.device
        print(f"Loading translator: {args.translate_model} on {tr_device} ...")
        TRANSLATOR_TOKENIZER = AutoTokenizer.from_pretrained(args.translate_model)
        TRANSLATOR_MODEL = AutoModelForCausalLM.from_pretrained(
            args.translate_model,
            torch_dtype=dtype_map[args.dtype],
            device_map=tr_device,
        )
        TRANSLATOR_MODEL.eval()
        print("Translator loaded.")
    else:
        print("Translation model disabled.")

    print("Scanning existing references...")
    load_all_references()

    server = HTTPServer((args.host, args.port), TTSHandler)
    print(f"Serving on http://{args.host}:{args.port}")
    print(f"  cloudflared tunnel --url http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
