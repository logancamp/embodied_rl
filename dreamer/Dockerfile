# DreamerV3 — CUDA Docker image for RTX 3070
# Base: official PyTorch image with CUDA 11.8 (compatible with RTX 3070)
# PyTorch 2.0+ with CUDA 11.8 is the sweet spot for Ampere GPUs

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Required for Atari ROM display/rendering
    libsdl2-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /dreamer

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Register Atari ROMs ────────────────────────────────────────────────────────
RUN autorom --accept-license

# ── Copy project files ─────────────────────────────────────────────────────────
COPY . .

# ── Default command ────────────────────────────────────────────────────────────
CMD ["python", "train.py", "--configs", "configs/base.yaml", "configs/s_size.yaml"]