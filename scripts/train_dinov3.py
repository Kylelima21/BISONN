#!/usr/bin/env python3
"""
Phase 3b: Train classification heads on DINOv3 embeddings.

Runs the same supervised methods (logistic regression, linear SVM, kNN) as
train_and_evaluate.py but on DINOv3 embeddings instead of BioCLIP.
No zero-shot (DINOv3 has no text encoder). MLP skipped (torch CPU issue).

Three-way split: 75% train / 15% validation / 15% test.
Train on train, select best head on validation, report final metrics on test.

Usage:
  CUDA_VISIBLE_DEVICES='' python3 scripts/train_dinov3.py small
  CUDA_VISIBLE_DEVICES='' python3 scripts/train_dinov3.py large

Output:
  data/models/dinov3_{size}_svm.joblib
  data/models/dinov3_{size}_logistic.joblib
  data/results/evaluation_report_dinov3_{size}.txt
  data/results/best_model_confusion_dinov3_{size}.png
  data/results/comparison_bar_dinov3_{size}.png
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import sys
import time
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_bundles import EmbeddingBundle


# ── Configuration ──────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = DATA_DIR / "models"

LABELS_PATH = DATA_DIR / "labels_bisonn.npy"

RANDOM_SEED = 42
VAL_SIZE = 0.15 / 0.85  # 15% of total → fraction of the 85% remainder
TEST_SIZE = 0.15         # 15% of total, held out for final evaluation
LABEL_NAMES = ["mobbing", "none"]


# ── Helpers (shared logic, duplicated from train_and_evaluate.py) ──────

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mobbing_precision": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "mobbing_recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "mobbing_f1": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "none_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "none_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "none_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def format_metrics(m):
    return (
        f"acc={m['accuracy']:.3f}  "
        f"macroF1={m['macro_f1']:.3f}  "
        f"mob P/R/F1={m['mobbing_precision']:.3f}/"
        f"{m['mobbing_recall']:.3f}/{m['mobbing_f1']:.3f}  "
        f"none P/R/F1={m['none_precision']:.3f}/"
        f"{m['none_recall']:.3f}/{m['none_f1']:.3f}"
    )


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(LABEL_NAMES); ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_comparison(method_names, metrics, path, title=""):
    n = len(method_names)
    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, [m["accuracy"] for m in metrics], width, label="Accuracy", color="#2196F3")
    bars2 = ax.bar(x + width/2, [m["macro_f1"] for m in metrics], width, label="Macro-F1", color="#FF9800")
    ax.set_ylabel("Score"); ax.set_title(title or "Method Comparison")
    ax.set_xticks(x); ax.set_xticklabels(method_names, rotation=15, ha="right")
    ax.legend(); ax.set_ylim(0, 1.05)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x()+bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("small", "large"):
        print(f"Usage: {sys.argv[0]} <small|large>")
        sys.exit(1)

    size_key = sys.argv[1]
    backbone_name = f"DINOv3 {size_key.capitalize()}"
    bundle_path = DATA_DIR / f"embeddings_dinov3_{size_key}.npz"

    print("=" * 60)
    print(f"BISONN Phase 3b — {backbone_name} Classification Head Training")
    print("=" * 60)
    print(f"Device: CPU")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    # 1. Load embeddings
    print(f"Loading embeddings: {bundle_path}")
    bundle = EmbeddingBundle.load(bundle_path)
    labels = np.load(LABELS_PATH)
    X = bundle.features
    y = labels
    print(f"  Embeddings: {X.shape}")
    print(f"  Labels: mobbing={np.sum(y==0)}, none={np.sum(y==1)}")
    print()

    # 2. Stratified three-way split: 75% train / 15% validation / 15% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, stratify=y_trainval,
        random_state=RANDOM_SEED,
    )
    print(f"Stratified split (75/15/15):")
    print(f"  Train:      {len(y_train)} (mobbing={np.sum(y_train==0)}, none={np.sum(y_train==1)})")
    print(f"  Validation: {len(y_val)} (mobbing={np.sum(y_val==0)}, none={np.sum(y_val==1)})")
    print(f"  Test:       {len(y_test)} (mobbing={np.sum(y_test==0)}, none={np.sum(y_test==1)})")
    print()

    method_names = []
    val_metrics = {}
    val_preds = {}
    models = {}

    # Logistic regression — train on train, evaluate on val
    print("[1/3] Logistic regression (class-weighted)...")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=RANDOM_SEED)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_val)
    lr_m = compute_metrics(y_val, lr_preds)
    print(f"  {format_metrics(lr_m)}")
    method_names.append("Logistic Reg"); val_metrics["logistic"] = lr_m
    val_preds["logistic"] = lr_preds; models["logistic"] = lr

    # Linear SVM — train on train, evaluate on val
    print("[2/3] Linear SVM (class-weighted)...")
    svm = SVC(kernel="linear", class_weight="balanced", random_state=RANDOM_SEED)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_val)
    svm_m = compute_metrics(y_val, svm_preds)
    print(f"  {format_metrics(svm_m)}")
    method_names.append("Linear SVM"); val_metrics["svm"] = svm_m
    val_preds["svm"] = svm_preds; models["svm"] = svm

    # kNN — train on train, evaluate on val
    print("[3/3] kNN (k=5, cosine, distance-weighted)...")
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_val)
    knn_m = compute_metrics(y_val, knn_preds)
    print(f"  {format_metrics(knn_m)}")
    method_names.append("kNN (k=5)"); val_metrics["knn"] = knn_m
    val_preds["knn"] = knn_preds; models["knn"] = knn

    # Summary — validation metrics
    print()
    print("=" * 80)
    print("VALIDATION SET METRICS (model selection)")
    print(f"{'Method':<16} {'Accuracy':>8} {'MacroF1':>8} "
          f"{'Mob P':>6} {'Mob R':>6} {'Mob F1':>6} "
          f"{'None P':>6} {'None R':>6} {'None F1':>6}")
    print("-" * 80)
    for name in method_names:
        key = {"Logistic Reg": "logistic", "Linear SVM": "svm", "kNN (k=5)": "knn"}[name]
        m = val_metrics[key]
        print(f"{name:<16} {m['accuracy']:>8.3f} {m['macro_f1']:>8.3f} "
              f"{m['mobbing_precision']:>6.3f} {m['mobbing_recall']:>6.3f} {m['mobbing_f1']:>6.3f} "
              f"{m['none_precision']:>6.3f} {m['none_recall']:>6.3f} {m['none_f1']:>6.3f}")
    print("=" * 80)

    # Find best on val
    keys = ["logistic", "svm", "knn"]
    best_idx = np.argmax([val_metrics[k]["macro_f1"] for k in keys])
    best_key = keys[best_idx]
    best_name = method_names[best_idx]
    best_val_m = val_metrics[best_key]
    print(f"\nBest model (by val macro-F1): {best_name} (val macroF1={best_val_m['macro_f1']:.3f})")

    # Final evaluation on held-out test set
    print(f"\n{'=' * 80}")
    print("TEST SET METRICS (held-out, final evaluation)")
    print(f"{'=' * 80}")

    test_metrics = {}
    test_preds = {}
    for name, key in zip(method_names, keys):
        preds = models[key].predict(X_test)
        m = compute_metrics(y_test, preds)
        test_metrics[key] = m
        test_preds[key] = preds
        print(f"  {name:<16} acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}  "
              f"mob P/R/F1={m['mobbing_precision']:.3f}/"
              f"{m['mobbing_recall']:.3f}/{m['mobbing_f1']:.3f}  "
              f"none P/R/F1={m['none_precision']:.3f}/"
              f"{m['none_recall']:.3f}/{m['none_f1']:.3f}")

    best_test_m = test_metrics[best_key]
    print(f"\nBest model: {best_name}")
    print(f"  Validation macroF1: {best_val_m['macro_f1']:.3f}")
    print(f"  Test macroF1:        {best_test_m['macro_f1']:.3f}")

    # Save artifacts
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Confusion matrix for best model (test set)
    best_cm = RESULTS_DIR / f"best_model_confusion_dinov3_{size_key}.png"
    plot_confusion(y_test, test_preds[best_key],
                   f"Confusion (test) — {backbone_name} {best_name}", best_cm)
    print(f"  Confusion matrix: {best_cm}")

    # Comparison bar chart — validation metrics
    comp_path = RESULTS_DIR / f"comparison_bar_dinov3_{size_key}.png"
    plot_comparison(method_names, [val_metrics[k] for k in keys], comp_path,
                    f"{backbone_name} — Method Comparison (val)")
    print(f"  Comparison chart: {comp_path}")

    # Save best sklearn model (retrained on train+val for deployment)
    if best_key == "logistic":
        deploy = LogisticRegression(max_iter=2000, class_weight="balanced",
                                    solver="lbfgs", random_state=RANDOM_SEED)
    elif best_key == "svm":
        deploy = SVC(kernel="linear", class_weight="balanced", random_state=RANDOM_SEED)
    else:
        deploy = KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")
    deploy.fit(X_trainval, y_trainval)
    joblib.dump(deploy, MODELS_DIR / f"dinov3_{size_key}_best.joblib")
    print(f"  Saved best model (retrained on train+val): dinov3_{size_key}_best.joblib")

    # Also save logistic + svm regardless
    lr_deploy = LogisticRegression(max_iter=2000, class_weight="balanced",
                                   solver="lbfgs", random_state=RANDOM_SEED)
    lr_deploy.fit(X_trainval, y_trainval)
    joblib.dump(lr_deploy, MODELS_DIR / f"dinov3_{size_key}_logistic.joblib")
    svm_deploy = SVC(kernel="linear", class_weight="balanced", random_state=RANDOM_SEED)
    svm_deploy.fit(X_trainval, y_trainval)
    joblib.dump(svm_deploy, MODELS_DIR / f"dinov3_{size_key}_svm.joblib")
    print(f"  Models saved to: {MODELS_DIR}")

    # Full report
    report_path = RESULTS_DIR / f"evaluation_report_dinov3_{size_key}.txt"
    with open(report_path, "w") as f:
        f.write(f"BISONN Phase 3b — {backbone_name} Classification Head Evaluation\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Backbone: {backbone_name} (timm, frozen)\n")
        f.write(f"Embedding dim: {X.shape[1]}\n")
        f.write(f"Dataset: 1690 images (101 mobbing + 1589 none)\n")
        f.write(f"Split: 75/15/15 stratified (train/val/test, seed={RANDOM_SEED})\n")
        f.write(f"  Train: {len(y_train)} (mobbing={np.sum(y_train==0)}, none={np.sum(y_train==1)})\n")
        f.write(f"  Val:   {len(y_val)} (mobbing={np.sum(y_val==0)}, none={np.sum(y_val==1)})\n")
        f.write(f"  Test:  {len(y_test)} (mobbing={np.sum(y_test==0)}, none={np.sum(y_test==1)})\n")
        f.write(f"Class weighting: balanced (inverse frequency)\n\n")

        f.write("VALIDATION SET METRICS (model selection)\n")
        f.write(f"{'Method':<16} {'Accuracy':>8} {'MacroF1':>8} "
                f"{'Mob P':>6} {'Mob R':>6} {'Mob F1':>6} "
                f"{'None P':>6} {'None R':>6} {'None F1':>6}\n")
        f.write("-" * 80 + "\n")
        for name in method_names:
            key = {"Logistic Reg": "logistic", "Linear SVM": "svm", "kNN (k=5)": "knn"}[name]
            m = val_metrics[key]
            f.write(f"{name:<16} {m['accuracy']:>8.3f} {m['macro_f1']:>8.3f} "
                    f"{m['mobbing_precision']:>6.3f} {m['mobbing_recall']:>6.3f} {m['mobbing_f1']:>6.3f} "
                    f"{m['none_precision']:>6.3f} {m['none_recall']:>6.3f} {m['none_f1']:>6.3f}\n")
        f.write("-" * 80 + "\n")
        f.write(f"\nBest (by val macroF1): {best_name} (val macroF1={best_val_m['macro_f1']:.3f})\n\n")

        f.write("TEST SET METRICS (final, held-out)\n")
        f.write(f"{'Method':<16} {'Accuracy':>8} {'MacroF1':>8} "
                f"{'Mob P':>6} {'Mob R':>6} {'Mob F1':>6} "
                f"{'None P':>6} {'None R':>6} {'None F1':>6}\n")
        f.write("-" * 80 + "\n")
        for name, key in zip(method_names, keys):
            m = test_metrics[key]
            f.write(f"{name:<16} {m['accuracy']:>8.3f} {m['macro_f1']:>8.3f} "
                    f"{m['mobbing_precision']:>6.3f} {m['mobbing_recall']:>6.3f} {m['mobbing_f1']:>6.3f} "
                    f"{m['none_precision']:>6.3f} {m['none_recall']:>6.3f} {m['none_f1']:>6.3f}\n")
        f.write("-" * 80 + "\n")
        f.write(f"\nBest model: {best_name} "
                f"(val macroF1={best_val_m['macro_f1']:.3f}, "
                f"test macroF1={best_test_m['macro_f1']:.3f})\n")

        # Confusion matrices (test set)
        for name, key in zip(method_names, keys):
            preds = test_preds[key]
            cm = confusion_matrix(y_test, preds)
            f.write(f"\n{name} confusion matrix (test, rows=actual, cols=predicted):\n")
            f.write(f"  {'':>10} {'mobbing':>8} {'none':>8}\n")
            f.write(f"  {'mobbing':>10} {cm[0,0]:>8} {cm[0,1]:>8}\n")
            f.write(f"  {'none':>10} {cm[1,0]:>8} {cm[1,1]:>8}\n")
            f.write(f"\n{name} — classification report (test set):\n")
            f.write(classification_report(y_test, preds, target_names=LABEL_NAMES, zero_division=0))

    print(f"  Evaluation report: {report_path}")
    print()
    print("=" * 60)
    print(f"Phase 3b complete ({backbone_name}).")
    print("=" * 60)


if __name__ == "__main__":
    main()
