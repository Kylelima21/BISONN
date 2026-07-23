#!/usr/bin/env python3
"""
Flag non-living-bird images in the 'none' folder for removal.

Uses BioCLIP 2.5 Huge zero-shot classification to score each image against
prompts for living birds vs. non-bird / dead / nest / feather prompts.
Images scoring low on "living bird" and high on other prompts are flagged.

Outputs a list of flagged filenames for review. Does NOT delete anything
by default — use --execute to remove flagged files.

Usage:
  CUDA_VISIBLE_DEVICES='' python3 scripts/clean_none_folder.py [--execute] [--threshold 0.15]
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU before torch import

import open_clip
import torch
import numpy as np
from PIL import Image

# ── Prompts ──────────────────────────────────────────────────────────────

# Positive prompts — what we WANT to keep (living birds)
LIVING_BIRD_PROMPTS = [
    "a photo of a living bird",
    "a photo of a live bird perched",
    "a photo of a live bird flying",
    "a photo of a live bird standing",
    "a photo of a live bird on the ground",
    "a photo of a bird in nature",
    "a photo of a wild bird",
    "a photo of a living bird in its habitat",
]

# Negative prompts — what we want to REMOVE
NON_BIRD_PROMPTS = [
    "a photo of a feather",
    "a photo of a nest without birds",
    "a photo of an empty nest",
    "a photo of bird bones",
    "a photo of a skeleton",
    "a photo of a dead bird",
    "a photo of a feather on the ground",
    "a photo of an egg",
    "a photo of bird droppings",
    "a photo of a track or footprint",
    "a photo of a habitat without birds",
    "a photo of a landscape with no birds",
    "a photo of a plant or flower",
    "a photo of an insect",
    "a photo of a mammal",
    "a photo of a rock or stone",
    "a photo of a tree without birds",
]


def classify_image(model, preprocess, tokenizer, img_path, pos_text_embs, neg_text_embs):
    """Score an image against positive (living bird) and negative prompts.

    Returns (max_pos_score, max_neg_score, best_neg_prompt).
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return 0.0, 1.0, f"could not open: {e}"

    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        img_emb = model.encode_image(img_tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        # Cosine similarity with positive prompts
        pos_sims = (img_emb @ pos_text_embs.T).squeeze(0)
        max_pos = pos_sims.max().item()

        # Cosine similarity with negative prompts
        neg_sims = (img_emb @ neg_text_embs.T).squeeze(0)
        max_neg = neg_sims.max().item()
        best_neg_idx = neg_sims.argmax().item()

    return max_pos, max_neg, best_neg_idx


def main():
    parser = argparse.ArgumentParser(description="Flag non-living-bird images in none/ folder")
    parser.add_argument("--folder", type=str,
                        default="/home/kylelima21/BISONN/data/labeled/none",
                        help="Path to the none/ image folder")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Flag if neg_score > pos_score + threshold (default: 0.15)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete flagged files (default: dry run)")
    parser.add_argument("--model", type=str, default="hf-hub:imageomics/bioclip-2.5-vith14",
                        help="OpenCLIP model name")
    args = parser.parse_args()

    folder = Path(args.folder)
    images = sorted(folder.glob("*"))
    print(f"Scanning {len(images)} images in {folder}")
    print(f"Model: {args.model}")
    print(f"Threshold: flag if neg_score > pos_score + {args.threshold}")
    print(f"Mode: {'DELETE flagged' if args.execute else 'DRY RUN (no deletion)'}")
    print()

    # Load model
    print("Loading model...", flush=True)
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(args.model)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

    # Encode text prompts
    with torch.no_grad():
        pos_tokens = tokenizer(LIVING_BIRD_PROMPTS)
        pos_text_embs = model.encode_text(pos_tokens)
        pos_text_embs = pos_text_embs / pos_text_embs.norm(dim=-1, keepdim=True)

        neg_tokens = tokenizer(NON_BIRD_PROMPTS)
        neg_text_embs = model.encode_text(neg_tokens)
        neg_text_embs = neg_text_embs / neg_text_embs.norm(dim=-1, keepdim=True)

    # Classify each image
    flagged = []
    kept = 0
    t1 = time.time()

    for i, img_path in enumerate(images, 1):
        max_pos, max_neg, best_neg_idx = classify_image(
            model, preprocess, tokenizer, img_path, pos_text_embs, neg_text_embs
        )

        # Flag if the best negative prompt scores significantly higher than
        # the best positive prompt
        is_flagged = (max_neg > max_pos + args.threshold) or (max_pos < 0.10)

        status = "FLAG" if is_flagged else "ok"
        best_neg = NON_BIRD_PROMPTS[best_neg_idx] if is_flagged else ""

        if is_flagged:
            flagged.append((img_path.name, max_pos, max_neg, best_neg))
            print(f"  [{i}/{len(images)}] FLAG: {img_path.name} "
                  f"(pos={max_pos:.3f} neg={max_neg:.3f} match=\"{best_neg}\")")
        else:
            kept += 1

        if i % 50 == 0:
            elapsed = time.time() - t1
            eta = elapsed / i * (len(images) - i)
            print(f"  ... {i}/{len(images)} scanned, {len(flagged)} flagged, "
                  f"ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t1
    print()
    print("=" * 60)
    print(f"Scan complete: {len(images)} images in {elapsed:.0f}s")
    print(f"  Kept: {kept}")
    print(f"  Flagged: {len(flagged)}")
    print()

    if not flagged:
        print("Nothing to remove.")
        return

    print("Flagged images:")
    for name, pos, neg, match in flagged:
        print(f"  {name}  (pos={pos:.3f} neg={neg:.3f} match=\"{match}\")")
    print()

    if args.execute:
        print("Deleting flagged files...")
        for name, _, _, _ in flagged:
            filepath = folder / name
            filepath.unlink()
            print(f"  deleted: {name}")
        print(f"Deleted {len(flagged)} files. Remaining: {len(list(folder.glob('*')))}")
    else:
        print("DRY RUN — no files deleted. Re-run with --execute to remove them.")

    print("=" * 60)


if __name__ == "__main__":
    main()
