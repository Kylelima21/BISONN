# BISONN Plugin — BioCLIP 2.5 + Linear SVM bird mobbing classifier
#
# Base: NVIDIA PyTorch 25.08 — CUDA 13.0, PyTorch 2.8, Python 3.12
# Supports Thor (Blackwell sm_110) and DGX Spark (sm_121).
# Generic PyPI torch lacks Blackwell kernels — use this base, not pip torch.
#
# Previous bases and their problems:
#   24.06-py3 (CUDA 12.4): max sm_90 — silently falls back to CPU on Blackwell
#   25.04-py3 (CUDA 12.9): no sm_110 cubins — fails on Thor nodes
#   25.08-py3 (CUDA 13.0): sm_110 + sm_120/sm_121 — works on both ✓

FROM nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /app
COPY requirements.txt .

# CRITICAL: Freeze base image packages so pip cannot replace them.
# The NVIDIA base ships torch, torchvision, and numpy compiled for
# Blackwell GPUs. pip install open_clip etc. will try to pull generic
# PyPI versions that LACK GPU kernels or break ABI compatibility.
#
# Three things MUST be frozen:
#   torch       — generic PyPI torch lacks Blackwell sm_110/sm_121 kernels
#   torchvision — must match the NVIDIA torch build
#   numpy       — base ships 1.26.4; upgrading to 2.x breaks torch.from_numpy()
#
# Note: 25.08-py3 does NOT include torchaudio — don't try to freeze it.
RUN pip install --no-cache-dir --upgrade pip && \
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)") && \
    TV_VER=$(python3 -c "import torchvision; print(torchvision.__version__)") && \
    NP_VER=$(python3 -c "import numpy; print(numpy.__version__)") && \
    echo "Freezing base-image stack: torch==${TORCH_VER} torchvision==${TV_VER} numpy==${NP_VER}" && \
    printf "torch==${TORCH_VER}\ntorchvision==${TV_VER}\nnumpy==${NP_VER}\n" > /tmp/constraints.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt -r requirements.txt

# Fix OpenCV: the base image ships opencv compiled against a different numpy.
# pip uninstall alone leaves stale .so files — the rm -rf is essential.
# Use -c /tmp/constraints.txt here too to prevent numpy upgrade.
RUN pip uninstall -y opencv-python opencv-python-headless 2>/dev/null; \
    rm -rf /usr/local/lib/python3.*/dist-packages/cv2* && \
    pip install --no-cache-dir -c /tmp/constraints.txt opencv-python-headless>=4.8.0

# Pre-download BioCLIP 2.5 weights at build time (edge nodes may lack internet).
# This pulls ~3.5GB of safetensors into the HuggingFace cache, baked into the image.
# At runtime, open_clip will find them in the cache and skip the download.
RUN mkdir -p /hf_cache && \
    HF_HOME=/hf_cache python3 -c \
    "import open_clip; open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2.5-vith14')"

# Offline mode at runtime — no surprise downloads
ENV HF_HOME=/hf_cache
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Bake classifier weights (small, 1.7MB — changes rarely)
COPY models/ /app/models/

# Copy app code LAST (small, changes often — layer ordering matters)
COPY app.py .

ENTRYPOINT ["python3", "/app/app.py"]
