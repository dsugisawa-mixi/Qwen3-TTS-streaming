#!/usr/bin/env python3
"""
Voice Design streaming server for Qwen3-TTS.

Flow:
  1. VoiceDesign model generates emotional reference audio from control instructions
     for each (style x language) combination (4 styles x 3 langs = 12 refs)
  2. VoiceDesign model is unloaded (GPU memory freed)
  3. Base model loads and builds ICL clone prompts from designed audio
  4. Streaming TTS uses designed voice prompts

Usage:
    python server-design.py [--port 8889]

    Then tunnel with:
        cloudflared tunnel --url http://localhost:8889
"""

import argparse
import io
import json
import math
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem


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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STYLES = ["neutral", "bright", "calm", "angry"]
LANGS = ["ja", "en", "ch"]

# language code -> model language value
LANG_TO_MODEL = {"ja": "Japanese", "en": "English", "ch": "Chinese"}

# Texts spoken by VoiceDesign model when generating emotional reference audio.
# Also used as ref_text for ICL-mode clone prompt (style reference).
DESIGN_REF_TEXTS = {
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


# Base character traits shared across all styles
_BASE_INSTRUCTION = """\
gender: Female.
age: Young adult to middle-aged adult.
clarity: Highly articulate and distinct pronunciation.
fluency: Very fluent speech with no hesitations.
accent: British English.
texture: Bright and clear vocal texture."""

# Per-style overrides for emotion, tone, speed, volume, pitch, personality
STYLE_INSTRUCTIONS = {
    "neutral": _BASE_INSTRUCTION + """
pitch: Medium female pitch, steady and measured.
speed: Moderate pace, clear and professional.
volume: Moderate and consistent.
emotion: Calm and composed, matter-of-fact delivery.
tone: Authoritative, professional, and informative.
personality: Steady, reliable, and trustworthy.""",

    "bright": _BASE_INSTRUCTION + """
pitch: Medium female pitch with significant upward inflections for emphasis and excitement.
speed: Fast-paced delivery with deliberate pauses for dramatic effect.
volume: Loud and projecting, increasing notably during moments of praise and announcements.
emotion: Enthusiastic and excited, especially when complimenting.
tone: Upbeat, authoritative, and performative.
personality: Confident, extroverted, and engaging.""",

    "calm": _BASE_INSTRUCTION + """
pitch: Medium female pitch, gentle and soothing with minimal inflection.
speed: Slow and relaxed pace, unhurried delivery.
volume: Soft and quiet, intimate and warm.
emotion: Peaceful and serene, reassuring.
tone: Gentle, warm, and meditative.
personality: Patient, nurturing, and contemplative.""",

    "angry": _BASE_INSTRUCTION + """
pitch: Medium female pitch with sharp, forceful downward inflections.
speed: Intense and clipped, with aggressive emphasis on key words.
volume: Very loud and forceful, almost shouting at peak moments.
emotion: Angry and frustrated, seething with indignation.
tone: Confrontational, demanding, and fierce.
personality: Assertive, combative, and unyielding.""",
}

DESIGNED_DIR = Path("./designed_references")


# ---------------------------------------------------------------------------
# Globals (set in main)
# ---------------------------------------------------------------------------
TTS: Qwen3TTSModel = None
MODEL_KIND: str = ""

# The active style instructions (for /api/info)
ACTIVE_STYLE_INSTRUCTIONS: dict = {}

# Clone prompts: key = "{lang}/{style}" -> list[VoiceClonePromptItem]
# Each list contains 1 item: designed voice ICL prompt
CLONE_PROMPTS: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt_key(lang: str, style: str) -> str:
    return f"{lang}/{style}"


def _designed_wav_path(lang: str, style: str) -> Path:
    """Path to VoiceDesign-generated emotional reference audio."""
    return DESIGNED_DIR / lang / f"{style}.wav"


# ---------------------------------------------------------------------------
# Voice Design (startup-time reference generation)
# ---------------------------------------------------------------------------

def _instructions_fingerprint(style_instructions: dict) -> str:
    """Build a deterministic string from all style instructions for change detection."""
    parts = []
    for style in sorted(style_instructions.keys()):
        parts.append(f"[{style}]\n{style_instructions[style].strip()}")
    return "\n\n".join(parts)


def design_voice(design_model_path: str, device: str, dtype, attn,
                 style_instructions: dict, force: bool = False):
    """Load VoiceDesign model, generate reference audio per (style x lang), save, unload."""
    DESIGNED_DIR.mkdir(parents=True, exist_ok=True)

    instruct_file = DESIGNED_DIR / "instruct.txt"
    fingerprint = _instructions_fingerprint(style_instructions)

    # Check if we can skip the design step
    all_exist = all(
        _designed_wav_path(lang, style).exists()
        for lang in LANGS for style in STYLES
    )
    instruct_match = (
        instruct_file.exists()
        and instruct_file.read_text(encoding="utf-8").strip() == fingerprint.strip()
    )

    if all_exist and instruct_match and not force:
        total = len(LANGS) * len(STYLES)
        print(f"[design] All {total} designed references found (instructions unchanged). Skipping VoiceDesign.")
        return

    print(f"[design] Loading VoiceDesign model: {design_model_path} ...")
    tts_design = Qwen3TTSModel.from_pretrained(
        design_model_path,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn,
    )

    generated = 0
    total = len(LANGS) * len(STYLES)
    for lang in LANGS:
        lang_dir = DESIGNED_DIR / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        for style in STYLES:
            wav_path = _designed_wav_path(lang, style)
            # Skip existing if instructions unchanged and not forced
            if wav_path.exists() and instruct_match and not force:
                generated += 1
                print(f"[design] [{generated}/{total}] {lang}/{style} — exists, skip")
                continue

            ref_text = DESIGN_REF_TEXTS[lang][style]
            model_lang = LANG_TO_MODEL[lang]
            instruct = style_instructions[style]
            generated += 1
            print(f"[design] [{generated}/{total}] Generating {lang}/{style} ({model_lang}) ...")
            t0 = time.time()
            wavs, sr = tts_design.generate_voice_design(
                text=ref_text,
                instruct=instruct,
                language=model_lang,
            )
            elapsed = time.time() - t0
            sf.write(str(wav_path), wavs[0], sr)
            duration = len(wavs[0]) / sr
            print(f"[design]   Saved {wav_path} ({elapsed:.2f}s, {duration:.1f}s audio, {sr}Hz)")

    # Save fingerprint for change detection on next restart
    instruct_file.write_text(fingerprint, encoding="utf-8")

    # Unload VoiceDesign model to free GPU memory
    del tts_design
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[design] VoiceDesign model unloaded. GPU memory freed.")


def build_clone_prompts():
    """Build ICL clone prompts from designed reference audio only (no speaker embedding)."""
    built = 0
    total = len(LANGS) * len(STYLES)
    for lang in LANGS:
        for style in STYLES:
            designed_wav = _designed_wav_path(lang, style)
            key = _prompt_key(lang, style)

            if not designed_wav.exists():
                print(f"[prompt] WARNING: designed ref {designed_wav} not found, skipping {key}")
                continue

            design_ref_text = DESIGN_REF_TEXTS[lang][style]

            print(f"[prompt] Building ICL prompt for {key} ...")
            t0 = time.time()

            prompt_items = TTS.create_voice_clone_prompt(
                ref_audio=str(designed_wav),
                ref_text=design_ref_text,
                x_vector_only_mode=False,
            )
            CLONE_PROMPTS[key] = prompt_items

            elapsed = time.time() - t0
            built += 1
            print(f"[prompt] {key} OK ({elapsed:.2f}s)  [{built}/{total}]")


# ---------------------------------------------------------------------------
# HTML — loaded from server-design.html at startup
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
        elif path == "/api/reference_audio":
            qs = parse_qs(parsed.query)
            self._handle_reference_audio(qs)
        elif path == "/api/stats":
            self._handle_stats()
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/generate":
                self._handle_generate()
            elif path == "/api/generate_stream":
                self._handle_generate_stream()
            else:
                self.send_error(404)
        except Exception as e:
            self._send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    # --- handlers ---

    def _handle_info(self):
        self._send_json({
            "model_kind": MODEL_KIND,
            "style_instructions": ACTIVE_STYLE_INSTRUCTIONS,
            "styles": STYLES,
            "langs": LANGS,
            "cached_keys": sorted(CLONE_PROMPTS.keys()),
        })

    def _handle_reference_audio(self, qs):
        lang = (qs.get("lang", [""])[0]).strip()
        style = (qs.get("style", [""])[0]).strip()
        if lang not in LANGS:
            self.send_error(400, "Invalid lang")
            return
        if style not in STYLES:
            self.send_error(400, "Invalid style")
            return
        wav_path = _designed_wav_path(lang, style)
        if not wav_path.exists():
            self.send_error(404, "Reference audio not found")
            return
        data = wav_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_stats(self):
        self._send_json({
            "generate": PERF_GENERATE.to_dict(),
            "stream_total": PERF_STREAM_TOTAL.to_dict(),
            "stream_first_chunk": PERF_STREAM_FIRST.to_dict(),
        })

    def _handle_generate(self):
        body = self._read_json_body()
        style = body.get("style", "neutral")
        out_lang = body.get("out_lang", "en")
        text = body["text"]

        print(f"[generate] REQ  style={style} out={out_lang}")
        print(f"[generate]  TEXT: {text!r}")

        if style not in STYLES:
            self._send_json({"error": f"Invalid style: {style}"}, 400)
            return
        if out_lang not in LANGS:
            self._send_json({"error": f"Invalid out_lang: {out_lang}"}, 400)
            return

        key = _prompt_key(out_lang, style)
        prompt_items = CLONE_PROMPTS.get(key)
        if prompt_items is None:
            self._send_json(
                {"error": f"No clone prompt for {key}. Check speaker refs and designed refs."},
                404,
            )
            return

        model_language = LANG_TO_MODEL[out_lang]

        print(f"[generate] START model_lang={model_language} key={key}")
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
        style = body.get("style", "neutral")
        out_lang = body.get("out_lang", "en")
        text = body["text"]

        print(f"[stream]   REQ  style={style} out={out_lang}")
        print(f"[stream]    TEXT: {text!r}")

        if style not in STYLES:
            self._send_json({"error": f"Invalid style: {style}"}, 400)
            return
        if out_lang not in LANGS:
            self._send_json({"error": f"Invalid out_lang: {out_lang}"}, 400)
            return

        key = _prompt_key(out_lang, style)
        prompt_items = CLONE_PROMPTS.get(key)
        if prompt_items is None:
            self._send_json(
                {"error": f"No clone prompt for {key}. Check speaker refs and designed refs."},
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

        print(f"[stream]   START model_lang={model_language} key={key}")
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

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global TTS, MODEL_KIND, INDEX_HTML, ACTIVE_STYLE_INSTRUCTIONS

    # Load HTML from server-design.html
    html_path = Path(__file__).parent / "server-design.html"
    INDEX_HTML = html_path.read_text(encoding="utf-8")
    print(f"[html] Loaded {html_path}")

    parser = argparse.ArgumentParser(description="Qwen3-TTS Voice Design Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8889, help="Port (default: 8889)")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="Base model for streaming (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)")
    parser.add_argument("--design-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                        help="VoiceDesign model for reference generation. Set to 'none' to skip. "
                             "(default: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention")
    parser.add_argument("--force-redesign", action="store_true",
                        help="Force regeneration of designed reference audio")
    args = parser.parse_args()

    ACTIVE_STYLE_INSTRUCTIONS = STYLE_INSTRUCTIONS

    print(f"[config] {len(ACTIVE_STYLE_INSTRUCTIONS)} style instructions:")
    for style, instruct in ACTIVE_STYLE_INSTRUCTIONS.items():
        line_count = len(instruct.strip().splitlines())
        print(f"  [{style}] {line_count} lines")
    print(f"[config] {len(STYLES)} styles x {len(LANGS)} langs = {len(STYLES)*len(LANGS)} combinations")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    attn = None if args.no_flash_attn else "flash_attention_2"

    # Step 1: Design emotional voice references (VoiceDesign model)
    if args.design_model and args.design_model.lower() != "none":
        design_voice(
            args.design_model, args.device, dtype_map[args.dtype], attn,
            ACTIVE_STYLE_INSTRUCTIONS, force=args.force_redesign,
        )
    else:
        print("[design] VoiceDesign model disabled (--design-model none). Using existing designed refs.")

    # Verify designed references exist
    designed_missing = [
        f"{lang}/{style}"
        for lang in LANGS for style in STYLES
        if not _designed_wav_path(lang, style).exists()
    ]
    if designed_missing:
        print(f"[design] WARNING: Missing designed refs: {designed_missing}")
        print(f"[design] Run with a --design-model to generate them.")

    # Step 2: Load Base model for streaming
    print(f"Loading Base model: {args.model} on {args.device} ({args.dtype}) ...")
    TTS = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype_map[args.dtype],
        attn_implementation=attn,
    )
    MODEL_KIND = getattr(TTS.model, "tts_model_type", "base")
    print(f"Base model loaded. type={MODEL_KIND}")

    # Step 3: Build ICL prompts from designed voice references
    print("Building ICL prompts from designed references ...")
    build_clone_prompts()
    if not CLONE_PROMPTS:
        print("[prompt] WARNING: No prompts built. TTS will not work.")
    else:
        print(f"[prompt] Ready: {len(CLONE_PROMPTS)} prompts — {sorted(CLONE_PROMPTS.keys())}")

    # Step 4: Serve
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
