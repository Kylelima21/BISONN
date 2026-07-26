# BioCLIP 2.5 — CPU-only on host venv (CUDA_VISIBLE_DEVICES='')
# The host venv's PyPI torch lacks Blackwell sm_110 kernels and hangs on
# any CUDA call. For development we run CPU-only. GPU access is only needed
# later for the deployed plugin (Phase 5: pluginctl --selector resource.gpu=true).
#
# Usage:
#   CUDA_VISIBLE_DEVICES='' python3 app.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # force CPU — before torch import

import open_clip
import torch
import time

print("Loading BioCLIP 2.5 Huge (ViT-H/14, 1024-dim)...")
t0 = time.time()
model, _, preprocess = open_clip.create_model_and_transforms(
    "hf-hub:imageomics/bioclip-2.5-vith14"
)
print(f"Model loaded in {time.time()-t0:.1f}s")

print("cuda available:", torch.cuda.is_available(), "(expected False — CPU dev mode)")
print("embedding dim:", model.text_projection.shape[1])

# Quick CPU inference test
from PIL import Image
img = Image.new('RGB', (224, 224), color='green')
t1 = time.time()
img_tensor = preprocess(img).unsqueeze(0)
emb = model.encode_image(img_tensor)
print(f"CPU inference: {time.time()-t1:.1f}s")
print(f"Embedding shape: {emb.shape}, norm: {emb.norm(dim=-1).item():.4f}")
print("SUCCESS — BioCLIP 2.5 ready for CPU development")
