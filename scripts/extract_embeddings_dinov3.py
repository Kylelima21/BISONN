#!/usr/bin/env python3
"""
Phase 2b: Extract DINOv3 embeddings for all labeled BISONN images.

Same pattern as extract_embeddings.py but uses timm + DINOv3 instead of
open_clip + BioCLIP. DINOv3 has no text encoder — this script only extracts
image embeddings. Accepts the model size as an argument.

Usage:
  CUDA_VISIBLE_DEVICES='' python3 scripts/extract_embeddings_dinov3.py small
  CUDA_VISIBLE_DEVICES='' python3 scripts/extract_embeddings_dinov3.py large

Output:
  data/embeddings_dinov3_{size}.npz   — EmbeddingBundle (IDs + L2-normalized features + manifest)
  data/labels_dinov3_{size}.npy       — integer labels (symlink to existing labels_bisonn.npy)
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_bundles import EmbeddingBundle, producer_manifest


# ── Configuration ──────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest_unified.csv"
LABELS_PATH = DATA_DIR / "labels_bisonn.npy"
LABEL_NAMES_PATH = DATA_DIR / "label_names.json"

BATCH_SIZE = 4

LABEL_NAMES = ["mobbing", "none"]
LABEL_TO_IDX = {name: idx for idx, name in enumerate(LABEL_NAMES)}

DINOV3_MODELS = {
    "small": {
        "timm_name": "vit_small_patch16_dinov3_qkvb",
        "hf_repo": "timm/vit_small_patch16_dinov3_qkvb.lvd1689m",
        "embedding_dim": 384,
    },
    "large": {
        "timm_name": "vit_large_patch16_dinov3_qkvb",
        "hf_repo": "timm/vit_large_patch16_dinov3_qkvb.lvd1689m",
        "embedding_dim": 1024,
    },
}


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
            image_id = row["filename"]
            label_name = row["class"]
            if label_name not in LABEL_TO_IDX:
                print(f"  WARNING: unknown class '{label_name}', skipping: {image_id}")
                continue
            ids.append(image_id)
            labels.append(LABEL_TO_IDX[label_name])
            image_paths.append(img_path)

    return ids, np.array(labels, dtype=np.int64), image_paths


def encode_images(model, transforms, image_paths, batch_size):
    """Encode images into (N, D) float32 numpy array using timm model."""
    all_features = []
    total = len(image_paths)
    t0 = time.time()

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = []
            for p in batch_paths:
                with Image.open(p) as img:
                    tensors.append(transforms(img.convert("RGB")))

            batch = torch.stack(tensors)
            features = model.forward_head(
                model.forward_features(batch), pre_logits=True
            )
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


def build_manifest(size_key, model_cfg, emb_dim):
    """Build the producer manifest for this embedding bundle."""
    import timm
    return producer_manifest(
        bundle_id=f"bisonn:dinov3-{size_key}:fp32",
        model_name=f"DINOv3 {size_key.capitalize()}",
        repo_id=model_cfg["hf_repo"],
        revision="main",
        precision="FP32",
        evidence_type="dataset-derived embeddings",
        framework="timm",
        framework_version=timm.__version__,
        preprocessing="timm.data transforms (ImageNet norm, 256x256, bicubic)",
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
    if len(sys.argv) < 2 or sys.argv[1] not in DINOV3_MODELS:
        print(f"Usage: {sys.argv[0]} <small|large>")
        print(f"Available: {', '.join(DINOV3_MODELS.keys())}")
        sys.exit(1)

    size_key = sys.argv[1]
    model_cfg = DINOV3_MODELS[size_key]
    output_bundle = DATA_DIR / f"embeddings_dinov3_{size_key}.npz"

    print("=" * 60)
    print(f"BISONN Phase 2b — DINOv3 {size_key.capitalize()} Embedding Extraction")
    print("=" * 60)
    print(f"Model: {model_cfg['timm_name']}")
    print(f"HF repo: {model_cfg['hf_repo']}")
    print(f"Expected embedding dim: {model_cfg['embedding_dim']}")
    print(f"Device: CPU (CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '')}')")
    print()

    # 1. Load manifest
    print(f"Loading manifest: {MANIFEST_PATH}")
    ids, labels, image_paths = load_manifest(MANIFEST_PATH)
    print(f"  {len(ids)} images loaded")
    class_counts = {name: int(np.sum(labels == idx)) for name, idx in LABEL_TO_IDX.items()}
    print(f"  Class distribution: {class_counts}")
    print()

    # 2. Load model via timm
    import timm
    print(f"Loading DINOv3 {size_key}...")
    t0 = time.time()
    model = timm.create_model(
        model_cfg["timm_name"],
        pretrained=True,
        num_classes=0,  # feature extraction only
    )
    model.eval()
    emb_dim = model.num_features
    print(f"  Loaded in {time.time()-t0:.1f}s, embedding dim = {emb_dim}")
    assert emb_dim == model_cfg["embedding_dim"], (
        f"Expected {model_cfg['embedding_dim']}, got {emb_dim}"
    )

    # 3. Get preprocessing transforms
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    print(f"  Transforms: input_size={data_config['input_size']}")
    print()

    # 4. Encode all images
    print("Encoding images (CPU)...")
    features = encode_images(model, transforms, image_paths, BATCH_SIZE)
    assert features.shape == (len(ids), emb_dim), (
        f"Expected shape ({len(ids)}, {emb_dim}), got {features.shape}"
    )
    print(f"  Done. Embeddings: {features.shape}")
    print()

    # 5. Build and save EmbeddingBundle
    print("Building EmbeddingBundle with L2-normalized features...")
    manifest = build_manifest(size_key, model_cfg, emb_dim)
    bundle = EmbeddingBundle.create(
        ids=ids,
        features=features,
        manifest=manifest,
    )
    bundle.save(output_bundle)
    print(f"  Saved: {output_bundle}")
    print(f"  Bundle shape: {bundle.features.shape}")
    print()

    # 6. Reuse existing labels (same images, same order)
    print("Labels reused from Phase 2 (same image order)")
    print(f"  Labels at: {LABELS_PATH}")
    print()

    # 7. Verify round-trip
    print("Verifying bundle round-trip...")
    loaded = EmbeddingBundle.load(output_bundle, expected_ids=ids)
    assert np.allclose(loaded.features, bundle.features, atol=1e-6), "Feature mismatch"
    assert loaded.manifest["rows"] == len(ids)
    print(f"  OK — {loaded.manifest['rows']} rows, {loaded.manifest['dimension']}-dim")
    print()
    print("=" * 60)
    print(f"Phase 2b complete (DINOv3 {size_key}).")
    print("=" * 60)


if __name__ == "__main__":
    main()
