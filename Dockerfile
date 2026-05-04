FROM --platform=linux/amd64 nvidia/cuda:13.0.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        wget git build-essential libsox-dev sox \
    && rm -rf /var/lib/apt/lists/*

# Use bash for all subsequent RUN steps so `source` and `conda activate` work
SHELL ["/bin/bash", "-c"]

# --- Miniconda ---
ENV CONDA_DIR=/opt/conda
RUN wget -qO /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# --- conda env: qwen3-tts (Python 3.12) ---
RUN conda create -y -n qwen3-tts python=3.12.* pip \
        --override-channels -c conda-forge \
    && conda clean -afy

# --- PyTorch 2.9.1 (CUDA 13.0) ---
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate qwen3-tts \
    && python --version && which python && which pip && python -m pip --version \
    && python -m pip install --no-cache-dir \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130

# --- Flash Attention 2 (prebuilt wheel for torch 2.9 / cu130) ---
# Dao-AILab公式リリースには cu13 wheel が無いため、サードパーティ製の prebuild wheel を利用:
#   https://github.com/mjun0812/flash-attention-prebuild-wheels
# conda-forge python rejects bare "linux_x86_64" tag, so rename wheel to manylinux_2_28
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate qwen3-tts \
    && wget -qO /tmp/flash_attn-2.8.3+cu130torch2.9-cp312-cp312-manylinux_2_28_x86_64.whl \
        https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.8/flash_attn-2.8.3%2Bcu130torch2.9-cp312-cp312-linux_x86_64.whl \
    && python -m pip install --no-cache-dir /tmp/flash_attn-2.8.3+cu130torch2.9-cp312-cp312-manylinux_2_28_x86_64.whl \
    && rm /tmp/flash_attn-*.whl

# --- qwen_tts package + deps ---
WORKDIR /app
COPY pyproject.toml MANIFEST.in README.md ./
COPY qwen_tts/ ./qwen_tts/
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate qwen3-tts \
    && python -m pip install --no-cache-dir .

# --- Server files ---
COPY server-design.py server-design.html ./
COPY pron_dict.json ./

EXPOSE 8889

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "qwen3-tts"]
CMD ["python", "./server-design.py"]

