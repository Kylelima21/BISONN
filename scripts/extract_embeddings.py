#!/usr/bin/env python3
"""
Phase 2: Extract BioCLIP 2.5 embeddings for all labeled BISONN images.

Loads the unified manifest, encodes every image with BioCLIP 2.5 Huge
(ViT-H/14, 1024-dim) on CPU, and saves the result as an EmbeddingBundle
(.npz with ordered IDs, L2-normalized features, and a producer manifest).

Reuses the EmbeddingBundle pattern from the Sage 2026 BioCLIP workshop
(sage-summer-2026-bioclip-main/notebooks/embedding_bundles.py).

Usage:
  CUDA_VISIBLE_DEVICES='' python3 scripts/extract_embeddings.py

Output:
  data/embeddings_bisonn.npz   — image embeddings + IDs + manifest
  data/labels_bisonn.npy       — integer labels (0=mobbing, 1=none)
  data/label_names.json        — class name mapping
"""
import os

# Force CPU before torch import — PyPI torch on Thor (Blackwell sm_110) hangs on CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make sibling modules importable
sys.path.insert(0, str(Path(__file__).parent))
from embedding_bundles import EmbeddingBundle, producer_manifest


# ── Configuration ──────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest_unified.csv"
OUTPUT_BUNDLE = DATA_DIR / "embeddings_bisonn.npz"
OUTPUT_LABELS = DATA_DIR / "labels_bisonn.npy"
OUTPUT_LABEL_NAMES = DATA_DIR / "label_names.json"

MODEL_NAME = "BioCLIP 2.5"
MODEL_REPO_ID = "imageomics/bioclip-2.5-vith14"
BATCH_SIZE = 4  # small batches for CPU memory — ViT-H/14 is ~1B params

# Class index mapping (sorted alphabetically for determinism)
LABEL_NAMES = ["mobbing", "none"]
LABEL_TO_IDX = {name: idx for idx, name in enumerate(LABEL_NAMES)}


# ── Helpers ─────────────────────────────────────────────────────────────

def load_manifest(path: Path):
    """Read the unified manifest CSV. Returns (ids, labels, image_paths)."""
    ids = []
    labels = []
    image_paths = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = Path(row["local_path"])
            if not img_path.is_file():
                print(f"  WARNING: missing file, skipping: {img_path}")
                continue

            # Use filename as the image ID (unique within the dataset)
            image_id = row["filename"]
            label_name = row["class"]

            if label_name not in LABEL_TO_IDX:
                print(f"  WARNING: unknown class '{label_name}', skipping: {image_id}")
                continue

            ids.append(image_id)
            labels.append(LABEL_TO_IDX[label_name])
            image_paths.append(img_path)

    return ids, np.array(labels, dtype=np.int64), image_paths


def encode_images(model, preprocess, image_paths, batch_size, device):
    """Encode a list of image paths into a (N, D) float32 numpy array."""
    all_features = []
    total = len(image_paths)
    t0 = time.time()

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = []
            for p in batch_paths:
                with Image.open(p) as img:
                    tensors.append(preprocess(img.convert("RGB")))

            batch = torch.stack(tensors).to(device)
            features = model.encode_image(batch, normalize=True)
            all_features.append(features.float().cpu())
            done = min(start + batch_size, total)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"  Encoded {done:,}/{total:,} images "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                end="\r",
            )

    print()
    return torch.cat(all_features).numpy()


def build_manifest():
    """Build the producer manifest for this embedding bundle."""
    return producer_manifest(
        bundle_id="bisonn:bioclip-2.5-vith14:fp32",
        model_name=MODEL_NAME,
        repo_id=MODEL_REPO_ID,
        revision="main",  # pinned via HF download cache
        precision="FP32",
        evidence_type="dataset-derived embeddings",
        framework="PyTorch",
        framework_version=torch.__version__,
        preprocessing="OpenCLIP evaluation transform from pinned model config",
        quantization=None,
        dataset={
            "name": "BISONN",
            "description": "Bird behavior binary classifier (mobbing vs none)",
            "images": "1690 labeled bird images",
            "sources": "iNaturalist + Wikimedia Commons + personal photos",
        },
        backend="CPU (forced via CUDA_VISIBLE_DEVICES='')",
        export=None,
        delegation=None,
        runtime={"device_type": "cpu", "batch_size": BATCH_SIZE},
    )


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BISONN Phase 2 — BioCLIP Embedding Extraction")
    print("=" * 60)
    print(f"Device: CPU (CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '')}')")
    print()

    # 1. Load manifest
    print(f"Loading manifest: {MANIFEST_PATH}")
    ids, labels, image_paths = load_manifest(MANIFEST_PATH)
    print(f"  {len(ids)} images loaded")
    class_counts = {name: int(np.sum(labels == idx)) for name, idx in LABEL_TO_IDX.items()}
    print(f"  Class distribution: {class_counts}")
    print()

    # 2. Load model
    import open_clip
    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        f"hf-hub:{MODEL_REPO_ID}"
    )
    model.eval()
    device = torch.device("cpu")
    model = model.to(device)
    emb_dim = model.text_projection.shape[1]
    print(f"  Loaded in {time.time()-t0:.1f}s, embedding dim = {emb_dim}")
    print()

    # 3. Encode all images
    print("Encoding images (CPU)...")
    features = encode_images(model, preprocess, image_paths, BATCH_SIZE, device)
    assert features.shape == (len(ids), emb_dim), (
        f"Expected shape ({len(ids)}, {emb_dim}), got {features.shape}"
    )
    print(f"  Done. Embeddings: {features.shape}")
    print()

    # 4. Build and save EmbeddingBundle
    print("Building EmbeddingBundle with L2-normalized features...")
    manifest = build_manifest()
    bundle = EmbeddingBundle.create(
        ids=ids,
        features=features,
        manifest=manifest,
    )
    bundle.save(OUTPUT_BUNDLE)
    print(f"  Saved: {OUTPUT_BUNDLE}")
    print(f"  Bundle shape: {bundle.features.shape}")
    print(f"  IDs: {len(bundle.ids)}")
    print()

    # 5. Save labels and label names
    np.save(OUTPUT_LABELS, labels)
    print(f"  Saved labels: {OUTPUT_LABELS}")
    with open(OUTPUT_LABEL_NAMES, "w") as f:
        json.dump({"labels": LABEL_NAMES, "mapping": LABEL_TO_IDX}, f, indent=2)
    print(f"  Saved label names: {OUTPUT_LABEL_NAMES}")
    print()

    # 6. Verify round-trip
    print("Verifying bundle round-trip...")
    loaded = EmbeddingBundle.load(OUTPUT_BUNDLE, expected_ids=ids)
    # Re-normalization on load introduces ~1e-8 float32 drift; use allclose.
    assert np.allclose(loaded.features, bundle.features, atol=1e-6), "Feature mismatch"
    assert loaded.manifest["rows"] == len(ids)
    print(f"  OK — {loaded.manifest['rows']} rows, {loaded.manifest['dimension']}-dim")
    print()
    print("=" * 60)
    print("Phase 2 complete. Next: extract text embeddings for zero-shot baseline.")
    print("=" * 60)


if __name__ == "__main__":
    main()
