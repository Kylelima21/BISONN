#!/usr/bin/env python3
"""
Phase 2b: Extract BioCLIP 2.5 text embeddings for zero-shot baseline.

Encodes behavior-description prompts for the two BISONN classes
(mobbing vs. none) using BioCLIP 2.5's text encoder. Each class gets
multiple prompts; their text embeddings are saved individually so that
downstream zero-shot scoring can either use individual prompts or
average them into class prototypes.

Saves a separate .npz for image-scoring and a .npy of class assignments.

Usage:
  CUDA_VISIBLE_DEVICES='' python3 scripts/extract_text_embeddings.py

Output:
  data/text_embeddings_bisonn.npz  — (prompt_ids, features, class_names)
  data/behavior_prompts.json       — the prompt text and class mapping
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from embedding_bundles import EmbeddingBundle, producer_manifest


# ── Configuration ──────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_BUNDLE = DATA_DIR / "text_embeddings_bisonn.npz"
OUTPUT_PROMPTS = DATA_DIR / "behavior_prompts.json"

MODEL_NAME = "BioCLIP 2.5"
MODEL_REPO_ID = "imageomics/bioclip-2.5-vith14"

# ── Behavior prompts ───────────────────────────────────────────────────
# BioCLIP was trained on photo captions, so "a photo of..." framing works.
# Multiple prompts per class let us build ensemble prototypes later.

PROMPTS = [
    # ── mobbing ──
    ("mobbing", "a photo of birds mobbing a predator"),
    ("mobbing", "a photo of a flock of birds harassing a threat"),
    ("mobbing", "a photo of small birds attacking a larger bird"),
    ("mobbing", "a photo of crows mobbing a hawk"),
    ("mobbing", "a photo of songbirds mobbing an owl"),
    ("mobbing", "a photo of birds aggressively harassing a raptor"),
    ("mobbing", "a photo of birds gathering around and calling at a predator"),
    ("mobbing", "a photo of birds dive-bombing a larger bird"),

    # ── none ──
    ("none", "a photo of a solitary bird perched"),
    ("none", "a photo of a bird flying with no interaction"),
    ("none", "a photo of a single bird foraging"),
    ("none", "a photo of a bird standing alone"),
    ("none", "a photo of a bird at rest with no visible interaction"),
    ("none", "a photo of a bird perched on a branch alone"),
    ("none", "a photo of a bird in flight with no other birds nearby"),
    ("none", "a photo of a calm bird with no aggressive behavior"),
]


def main():
    print("=" * 60)
    print("BISONN Phase 2b — Text Embedding Extraction (Zero-Shot Prompts)")
    print("=" * 60)
    print(f"Device: CPU (CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '')}')")
    print()

    import open_clip
    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    model, _, _ = open_clip.create_model_and_transforms(
        f"hf-hub:{MODEL_REPO_ID}"
    )
    model.eval()
    model = model.to("cpu")
    tokenizer = open_clip.get_tokenizer(f"hf-hub:{MODEL_REPO_ID}")
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print()

    # Encode all prompts
    class_names = [cls for cls, _ in PROMPTS]
    prompt_texts = [txt for _, txt in PROMPTS]

    print(f"Encoding {len(PROMPTS)} behavior prompts...")
    with torch.inference_mode():
        tokens = tokenizer(prompt_texts)  # (N, context_length)
        # normalize=True returns unit-length vectors for cosine similarity
        text_features = model.encode_text(tokens, normalize=True)
    text_features = text_features.float().cpu().numpy()

    print(f"  Text embeddings: {text_features.shape}")
    print()

    # Build prompt IDs
    prompt_ids = [f"{cls}_{i:03d}" for i, (cls, _) in enumerate(PROMPTS)]

    # Save .npz EmbeddingBundle
    manifest = producer_manifest(
        bundle_id="bisonn:bioclip-2.5-vith14:text-prompts",
        model_name=MODEL_NAME,
        repo_id=MODEL_REPO_ID,
        revision="main",
        precision="FP32",
        evidence_type="zero-shot text prototypes",
        framework="PyTorch",
        framework_version=torch.__version__,
        preprocessing="OpenCLIP tokenizer + text encoder, normalize=True",
        quantization=None,
        dataset={
            "name": "BISONN behavior prompts",
            "description": "Hand-authored behavior descriptions for zero-shot baseline",
            "prompts_per_class": {
                cls: sum(1 for c, _ in PROMPTS if c == cls)
                for cls in set(class_names)
            },
        },
        backend="CPU",
        export=None,
        delegation=None,
        runtime={"device_type": "cpu"},
    )

    bundle = EmbeddingBundle.create(
        ids=prompt_ids,
        features=text_features,
        manifest=manifest,
    )
    bundle.save(OUTPUT_BUNDLE)
    print(f"Saved text embedding bundle: {OUTPUT_BUNDLE}")
    print(f"  Shape: {bundle.features.shape}, IDs: {len(bundle.ids)}")
    print()

    # Save prompt metadata (prompts + class mapping)
    prompt_records = [
        {"id": prompt_ids[i], "class": class_names[i], "text": prompt_texts[i]}
        for i in range(len(PROMPTS))
    ]
    with open(OUTPUT_PROMPTS, "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "repo_id": MODEL_REPO_ID,
                "prompts": prompt_records,
                "classes": sorted(set(class_names)),
            },
            f,
            indent=2,
        )
    print(f"Saved prompt metadata: {OUTPUT_PROMPTS}")
    print()

    # Verify round-trip
    loaded = EmbeddingBundle.load(OUTPUT_BUNDLE, expected_ids=prompt_ids)
    assert np.array_equal(loaded.features, bundle.features)
    print("Round-trip verification: OK")
    print()
    print("=" * 60)
    print("Phase 2b complete. Ready for zero-shot baseline + head training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
