#!/usr/bin/env python3
"""
Phase 3d: Confusion matrix grid for every backbone x head combination.

Produces a single figure with all confusion matrices (test set):
  - 3 backbones (BioCLIP 2.5, DINOv3 Large, DINOv3 Small)
  - 3 supervised heads each (Logistic Reg, Linear SVM, kNN)
  - 1 zero-shot (BioCLIP 2.5 only)
  = 10 subplots in a 4x3 grid (zero-shot gets the extra slot)

Three-way split: 75% train / 15% validation / 15% test (seed=42, stratified).
Confusion matrices are on the held-out test set.

Usage:
  CUDA_VISIBLE_DEVICES='' python3 scripts/plot_all_confusion_matrices.py

Output:
  data/results/all_confusion_matrices.png
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_bundles import EmbeddingBundle

# ── Configuration ──────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"

LABEL_NAMES = ["mobbing", "none"]
RANDOM_SEED = 42
VAL_SIZE = 0.15 / 0.85
TEST_SIZE = 0.15

BACKBONES = [
    {
        "name": "BioCLIP 2.5",
        "bundle": DATA_DIR / "embeddings_bisonn.npz",
    },
    {
        "name": "DINOv3 Large",
        "bundle": DATA_DIR / "embeddings_dinov3_large.npz",
    },
    {
        "name": "DINOv3 Small",
        "bundle": DATA_DIR / "embeddings_dinov3_small.npz",
    },
]

HEADS = ["Logistic Reg", "Linear SVM", "kNN (k=5)"]


def plot_cm(ax, y_true, y_pred, title):
    """Draw a single 2x2 confusion matrix on the given axes."""
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues", vmin=0)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(LABEL_NAMES, fontsize=9)
    ax.set_yticklabels(LABEL_NAMES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")


def main():
    print("=" * 60)
    print("BISONN — All Confusion Matrices (75/15/15 split, test set)")
    print("=" * 60)

    labels = np.load(DATA_DIR / "labels_bisonn.npy")

    all_cms = []  # (row, col, title, y_true, y_pred)

    # ── Zero-shot (BioCLIP 2.5 only) ────────────────────────────────────
    txt_bundle_path = DATA_DIR / "text_embeddings_bisonn.npz"
    prompts_path = DATA_DIR / "behavior_prompts.json"
    if txt_bundle_path.exists() and prompts_path.exists():
        print("Zero-shot (BioCLIP 2.5)...")
        bioclip_bundle = EmbeddingBundle.load(
            DATA_DIR / "embeddings_bisonn.npz"
        )
        txt_bundle = EmbeddingBundle.load(txt_bundle_path)
        with open(prompts_path) as f:
            prompts_data = json.load(f)

        X_full = bioclip_bundle.features
        y_full = labels

        mobbing_idx = [
            i for i, p in enumerate(prompts_data["prompts"])
            if p["class"] == "mobbing"
        ]
        none_idx = [
            i for i, p in enumerate(prompts_data["prompts"])
            if p["class"] == "none"
        ]

        mobbing_proto = txt_bundle.features[mobbing_idx].mean(axis=0)
        none_proto = txt_bundle.features[none_idx].mean(axis=0)
        mobbing_proto /= np.linalg.norm(mobbing_proto)
        none_proto /= np.linalg.norm(none_proto)

        # Best-of-prompts scheme (matches train_and_evaluate.py zero-shot)
        all_scores = X_full @ txt_bundle.features.T
        mobbing_max = all_scores[:, mobbing_idx].max(axis=1)
        none_max = all_scores[:, none_idx].max(axis=1)
        zs_preds = (none_max > mobbing_max).astype(int)

        all_cms.append((0, 0, "BioCLIP 2.5\nZero-shot", y_full, zs_preds))

    # ── Supervised heads for each backbone (test set) ───────────────────
    for bb_row, bb in enumerate(BACKBONES):
        bundle_path = bb["bundle"]
        if not bundle_path.exists():
            print(f"  SKIP {bb['name']} — {bundle_path} not found")
            continue

        print(f"Processing {bb['name']}...")
        bundle = EmbeddingBundle.load(bundle_path)
        X = bundle.features
        y = labels

        # Three-way split: 75/15/15
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=VAL_SIZE, stratify=y_trainval,
            random_state=RANDOM_SEED,
        )

        # Logistic regression
        lr = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED,
        )
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        all_cms.append((
            bb_row,
            1,
            f"{bb['name']}\nLogistic Reg",
            y_test, lr_pred,
        ))

        # Linear SVM
        svm = SVC(
            kernel="linear", class_weight="balanced", random_state=RANDOM_SEED,
        )
        svm.fit(X_train, y_train)
        svm_pred = svm.predict(X_test)
        all_cms.append((bb_row, 2, f"{bb['name']}\nLinear SVM", y_test, svm_pred))

        # kNN
        knn = KNeighborsClassifier(
            n_neighbors=5, metric="cosine", weights="distance",
        )
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        all_cms.append((bb_row, 3, f"{bb['name']}\nkNN (k=5)", y_test, knn_pred))

    # ── Plot the grid ───────────────────────────────────────────────────
    n_rows = 3
    n_cols = 4  # zero-shot | logistic | svm | knn

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    fig.suptitle(
        "BISONN — Confusion Matrices for All Backbone x Head Combinations (test set, 75/15/15 split)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    for row, col, title, y_true, y_pred in all_cms:
        plot_cm(axes[row][col], y_true, y_pred, title)

    # Hide empty subplots (DINOv3 row 0, col 0)
    axes[1][0].set_visible(False)
    axes[2][0].set_visible(False)

    # Colorbars — one per visible subplot
    for row, col, _, y_true, y_pred in all_cms:
        cm = confusion_matrix(y_true, y_pred)
        im = axes[row][col].images[0]
        fig.colorbar(im, ax=axes[row][col], fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = RESULTS_DIR / "all_confusion_matrices.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
