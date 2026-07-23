# Run INSIDE the NVIDIA container, not the host venv:
#   sudo docker run --rm --gpus all -v ~/BISONN:/app -w /app \
#     nvcr.io/nvidia/pytorch:25.08-py3 python3 /app/app.py
import open_clip
import torch

# BioCLIP 2.5 Huge — ViT-H/14, 1024-dim embeddings, ~1B params
model, _, preprocess = open_clip.create_model_and_transforms(
    "hf-hub:imageomics/bioclip-2.5-vith14"
)

print("Model loaded:", "BioCLIP 2.5 Huge")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("CPU only")
