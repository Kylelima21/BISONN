#!/usr/bin/env python3
"""
Phase 3: Train and evaluate classification heads on frozen BioCLIP 2.5 embeddings.

Implements the workshop's evaluation pattern (per-method accuracy, macro-F1,
confusion matrix) adapted for BISONN's binary behavior task (mobbing vs none).

Methods:
  1. Zero-shot retrieval (cosine sim to text prototypes, no training)
  2. Logistic regression (linear probe, class-weighted)
  3. Linear SVM (class-weighted)
  4. kNN (cosine metric, class-weighted voting)
  5. Small MLP (2-layer, class-weighted cross-entropy)

All supervised methods use stratified 80/20 train/test split.
Class imbalance (1:16 mobbing:none) handled via class_weight='balanced'
in sklearn and weighted CrossEntropyLoss in PyTorch.

Usage:
  CUDA_VISIBLE_DEVICES='' python3 scripts/train_and_evaluate.py

Output:
  data/results/evaluation_report.txt    — comparison table + per-method metrics
  data/results/best_model_confusion.png  — confusion matrix for best model
  data/results/comparison_bar.png        — bar chart comparing all methods
  data/models/<best>.joblib              — saved best sklearn model
  data/models/mlp_weights.pt             — saved MLP state dict (if trained)
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import sys
import time
from pathlib import Path

import numpy as np
import joblib
# NOTE: torch is NOT imported at module level — torch 2.13+cu130 on Blackwell
# hangs on CUDA init even with CUDA_VISIBLE_DEVICES=''. The MLP method (optional)
# imports torch lazily only if explicitly enabled via BISONN_ENABLE_MLP=1.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_bundles import EmbeddingBundle


# ── Configuration ──────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = DATA_DIR / "models"

IMG_BUNDLE = DATA_DIR / "embeddings_bisonn.npz"
TXT_BUNDLE = DATA_DIR / "text_embeddings_bisonn.npz"
LABELS_PATH = DATA_DIR / "labels_bisonn.npy"
LABEL_NAMES_PATH = DATA_DIR / "label_names.json"
PROMPTS_PATH = DATA_DIR / "behavior_prompts.json"

RANDOM_SEED = 42
TEST_SIZE = 0.2

LABEL_NAMES = ["mobbing", "none"]


# ── Helpers ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Return a dict of standard binary classification metrics."""
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
    """One-line summary for the comparison table."""
    return (
        f"acc={m['accuracy']:.3f}  "
        f"macroF1={m['macro_f1']:.3f}  "
        f"mob P/R/F1={m['mobbing_precision']:.3f}/"
        f"{m['mobbing_recall']:.3f}/{m['mobbing_f1']:.3f}  "
        f"none P/R/F1={m['none_precision']:.3f}/"
        f"{m['none_recall']:.3f}/{m['none_f1']:.3f}"
    )


def plot_confusion(y_true, y_pred, title, path):
    """Save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(method_names, metrics, path):
    """Bar chart comparing accuracy and macro-F1 across methods."""
    n = len(method_names)
    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    accs = [m["accuracy"] for m in metrics]
    f1s = [m["macro_f1"] for m in metrics]
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#2196F3")
    bars2 = ax.bar(x + width / 2, f1s, width, label="Macro-F1", color="#FF9800")
    ax.set_ylabel("Score")
    ax.set_title("BISONN Phase 3 — Method Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=20, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Method 1: Zero-shot retrieval ───────────────────────────────────────

def run_zero_shot(img_features, txt_features, prompts_data, y_true):
    """Cosine similarity to averaged text prototypes. No training."""
    print("\n[1/5] Zero-shot retrieval (text prototypes)...")

    mobbing_idx = [i for i, p in enumerate(prompts_data["prompts"]) if p["class"] == "mobbing"]
    none_idx = [i for i, p in enumerate(prompts_data["prompts"]) if p["class"] == "none"]

    # Average prompts per class, re-normalize (features are already unit-length)
    mobbing_proto = txt_features[mobbing_idx].mean(axis=0)
    none_proto = txt_features[none_idx].mean(axis=0)
    mobbing_proto /= np.linalg.norm(mobbing_proto)
    none_proto /= np.linalg.norm(none_proto)

    # Cosine sim = dot product (both sides normalized)
    mobbing_scores = img_features @ mobbing_proto
    none_scores = img_features @ none_proto

    # Also try best-of-prompts (max score across individual prompts)
    all_scores = img_features @ txt_features.T  # (N, 16)
    mobbing_max = all_scores[:, mobbing_idx].max(axis=1)
    none_max = all_scores[:, none_idx].max(axis=1)

    # Compare three voting schemes
    schemes = {
        "avg-prototype": (none_scores > mobbing_scores).astype(int),
        "best-of-prompts": (none_max > mobbing_max).astype(int),
    }

    best_name = None
    best_f1 = -1
    best_preds = None
    best_metrics = None

    for name, preds in schemes.items():
        m = compute_metrics(y_true, preds)
        print(f"  {name}: {format_metrics(m)}")
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_name = name
            best_preds = preds
            best_metrics = m

    print(f"  Best: {best_name}")
    return best_metrics, best_preds


# ── Method 2: Logistic regression ──────────────────────────────────────

def run_logistic_regression(X_train, y_train, X_test, y_test):
    print("\n[2/5] Logistic regression (linear probe, class-weighted)...")
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    m = compute_metrics(y_test, y_pred)
    print(f"  {format_metrics(m)}")
    return m, y_pred, clf


# ── Method 3: Linear SVM ────────────────────────────────────────────────

def run_svm(X_train, y_train, X_test, y_test):
    print("\n[3/5] Linear SVM (class-weighted)...")
    clf = SVC(
        kernel="linear",
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    m = compute_metrics(y_test, y_pred)
    print(f"  {format_metrics(m)}")
    return m, y_pred, clf


# ── Method 4: kNN ───────────────────────────────────────────────────────

def run_knn(X_train, y_train, X_test, y_test, k=5):
    print(f"\n[4/5] kNN (k={k}, cosine, distance-weighted)...")
    clf = KNeighborsClassifier(
        n_neighbors=k,
        metric="cosine",
        weights="distance",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    m = compute_metrics(y_test, y_pred)
    print(f"  {format_metrics(m)}")
    return m, y_pred, clf


# ── Method 5: Small MLP ─────────────────────────────────────────────────


def run_mlp(X_train, y_train, X_test, y_test, epochs=50, lr=1e-3, hidden_dim=256):
    """Train a 2-layer MLP. Skipped unless BISONN_ENABLE_MLP=1 is set.

    torch 2.13+cu130 on Blackwell hangs on CUDA init even with
    CUDA_VISIBLE_DEVICES=''. The MLP is optional (PLAN.md Phase 3D) —
    run it inside an NVIDIA container (nvcr.io/nvidia/pytorch:25.08-py3)
    by setting BISONN_ENABLE_MLP=1.
    """
    print("\n[5/5] MLP (2-layer, class-weighted cross-entropy)...")
    if os.environ.get("BISONN_ENABLE_MLP") != "1":
        print("  SKIPPED — set BISONN_ENABLE_MLP=1 to enable")
        print("  (requires torch build that works on this host or an NVIDIA container)")
        return None, None, None

    import torch
    import torch.nn as nn

    X_tr = torch.as_tensor(X_train, dtype=torch.float32)
    y_tr = torch.as_tensor(y_train, dtype=torch.long)
    X_te = torch.as_tensor(X_test, dtype=torch.float32)

    # Class weights: inverse frequency
    n_mob = (y_train == 0).sum()
    n_none = (y_train == 1).sum()
    w_mobbing = n_none / (n_mob + n_none) * 2
    w_none = n_mob / (n_mob + n_none) * 2
    class_weights = torch.as_tensor([w_mobbing, w_none], dtype=torch.float32)
    print(f"  Class weights: mobbing={w_mobbing:.3f}, none={w_none:.3f}")

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, 2),
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            acc = (logits.argmax(dim=1) == y_tr).float().mean().item()
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, train_acc={acc:.3f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te).argmax(dim=1).numpy()
    m = compute_metrics(y_test, y_pred)
    print(f"  {format_metrics(m)}")
    return m, y_pred, model


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BISONN Phase 3 — Train and Evaluate Classification Heads")
    print("=" * 60)
    print(f"Device: CPU (CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '')}')")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    # 1. Load artifacts
    print("Loading artifacts...")
    img_bundle = EmbeddingBundle.load(IMG_BUNDLE)
    txt_bundle = EmbeddingBundle.load(TXT_BUNDLE)
    labels = np.load(LABELS_PATH)
    with open(PROMPTS_PATH) as f:
        prompts_data = json.load(f)

    X = img_bundle.features
    y = labels
    print(f"  Image embeddings: {X.shape}")
    print(f"  Labels: mobbing={np.sum(y==0)}, none={np.sum(y==1)}")
    print()

    # 2. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED,
    )
    print(f"Stratified split ({1-TEST_SIZE:.0%}/{TEST_SIZE:.0%}):")
    print(f"  Train: {len(y_train)} (mobbing={np.sum(y_train==0)}, none={np.sum(y_train==1)})")
    print(f"  Test:  {len(y_test)} (mobbing={np.sum(y_test==0)}, none={np.sum(y_test==1)})")
    print()

    # 3. Run all methods
    method_names = []
    all_metrics = []
    all_preds = {}

    # 1. Zero-shot (uses full dataset, no train/test split)
    zs_metrics, zs_preds = run_zero_shot(X, txt_bundle.features, prompts_data, y)
    method_names.append("Zero-shot")
    all_metrics.append(zs_metrics)
    all_preds["zero_shot"] = zs_preds

    # 2-5: Supervised methods (use train/test split)
    lr_metrics, lr_preds, lr_model = run_logistic_regression(X_train, y_train, X_test, y_test)
    method_names.append("Logistic Reg")
    all_metrics.append(lr_metrics)
    all_preds["logistic"] = lr_preds

    svm_metrics, svm_preds, svm_model = run_svm(X_train, y_train, X_test, y_test)
    method_names.append("Linear SVM")
    all_metrics.append(svm_metrics)
    all_preds["svm"] = svm_preds

    knn_metrics, knn_preds, knn_model = run_knn(X_train, y_train, X_test, y_test, k=5)
    method_names.append("kNN (k=5)")
    all_metrics.append(knn_metrics)
    all_preds["knn"] = knn_preds

    # 5. MLP (optional — torch 2.13+cu130 hangs on CPU matmul due to Blackwell
    #    CUDA init. Skipped on this host; can be re-enabled in an NVIDIA container.)
    mlp_result = run_mlp(X_train, y_train, X_test, y_test)
    if mlp_result[0] is not None:
        mlp_metrics, mlp_preds, mlp_model = mlp_result
        method_names.append("MLP (256)")
        all_metrics.append(mlp_metrics)
        all_preds["mlp"] = mlp_preds
    else:
        print("  (MLP skipped — unavailable on this host's torch build)")

    # 4. Summary table
    print()
    print("=" * 80)
    print(f"{'Method':<16} {'Accuracy':>8} {'MacroF1':>8} "
          f"{'Mob P':>6} {'Mob R':>6} {'Mob F1':>6} "
          f"{'None P':>6} {'None R':>6} {'None F1':>6}")
    print("-" * 80)
    for name, m in zip(method_names, all_metrics):
        print(f"{name:<16} {m['accuracy']:>8.3f} {m['macro_f1']:>8.3f} "
              f"{m['mobbing_precision']:>6.3f} {m['mobbing_recall']:>6.3f} {m['mobbing_f1']:>6.3f} "
              f"{m['none_precision']:>6.3f} {m['none_recall']:>6.3f} {m['none_f1']:>6.3f}")
    print("=" * 80)

    # Find best model by macro-F1 (among supervised only)
    supervised_metrics = all_metrics[1:]  # exclude zero-shot
    supervised_names = method_names[1:]
    best_idx = np.argmax([m["macro_f1"] for m in supervised_metrics])
    best_name = supervised_names[best_idx]
    best_metrics = supervised_metrics[best_idx]
    print(f"\nBest supervised model (by macro-F1): {best_name} "
          f"(macroF1={best_metrics['macro_f1']:.3f})")

    # 6. Save artifacts
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Map supervised method names to their prediction keys and models
    sklearn_keys = ["logistic", "svm", "knn"]
    sklearn_models = {"logistic": lr_model, "svm": svm_model, "knn": knn_model}

    # Confusion matrix for best supervised model
    best_key = None
    for i, name in enumerate(supervised_names):
        if i == best_idx:
            if name in ("Logistic Reg",):
                best_key = "logistic"
            elif name == "Linear SVM":
                best_key = "svm"
            elif name.startswith("kNN"):
                best_key = "knn"
            elif name.startswith("MLP"):
                best_key = "mlp"
            break
    best_preds = all_preds[best_key]
    best_cm_path = RESULTS_DIR / "best_model_confusion.png"
    plot_confusion(y_test, best_preds, f"Confusion Matrix — {best_name}", best_cm_path)
    print(f"  Confusion matrix: {best_cm_path}")

    # Comparison bar chart
    comp_path = RESULTS_DIR / "comparison_bar.png"
    plot_comparison(method_names, all_metrics, comp_path)
    print(f"  Comparison chart: {comp_path}")

    # Save best sklearn model
    sklearn_models = {"logistic": lr_model, "svm": svm_model, "knn": knn_model}
    if best_key in sklearn_models:
        model_path = MODELS_DIR / f"{best_key}.joblib"
        joblib.dump(sklearn_models[best_key], model_path)
        print(f"  Saved model: {model_path}")
    elif best_key == "mlp":
        import torch as _torch
        model_path = MODELS_DIR / "mlp_weights.pt"
        _torch.save(mlp_model.state_dict(), model_path)  # type: ignore[union-attr]

        print(f"  Saved model: {model_path}")

    # Also save logistic regression regardless (likely deploy choice — simplest)
    if best_key != "logistic":
        joblib.dump(lr_model, MODELS_DIR / "logistic.joblib")
        print(f"  Also saved logistic: {MODELS_DIR / 'logistic.joblib'}")

    # Full evaluation report
    report_path = RESULTS_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("BISONN Phase 3 — Classification Head Evaluation\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Model: BioCLIP 2.5 Huge (ViT-H/14, 1024-dim, frozen)\n")
        f.write(f"Dataset: 1690 images (101 mobbing + 1589 none)\n")
        f.write(f"Split: 80/20 stratified (seed={RANDOM_SEED})\n")
        f.write(f"Class imbalance: 1:16 (mobbing:none)\n")
        f.write(f"Class weighting: balanced (inverse frequency)\n\n")
        f.write(f"{'Method':<16} {'Accuracy':>8} {'MacroF1':>8} "
                f"{'Mob P':>6} {'Mob R':>6} {'Mob F1':>6} "
                f"{'None P':>6} {'None R':>6} {'None F1':>6}\n")
        f.write("-" * 80 + "\n")
        for name, m in zip(method_names, all_metrics):
            f.write(f"{name:<16} {m['accuracy']:>8.3f} {m['macro_f1']:>8.3f} "
                    f"{m['mobbing_precision']:>6.3f} {m['mobbing_recall']:>6.3f} {m['mobbing_f1']:>6.3f} "
                    f"{m['none_precision']:>6.3f} {m['none_recall']:>6.3f} {m['none_f1']:>6.3f}\n")
        f.write("-" * 80 + "\n")
        f.write(f"\nBest supervised: {best_name} (macroF1={best_metrics['macro_f1']:.3f})\n\n")

        # Per-method confusion matrices
        for name, key in zip(method_names, ["zero_shot", "logistic", "svm", "knn", "mlp"]):
            preds = all_preds[key]
            if key == "zero_shot":
                y_true_full = y
            else:
                y_true_full = y_test
            cm = confusion_matrix(y_true_full, preds)
            f.write(f"\n{name} confusion matrix (rows=actual, cols=predicted):\n")
            f.write(f"  {'':>10} {'mobbing':>8} {'none':>8}\n")
            f.write(f"  {'mobbing':>10} {cm[0,0]:>8} {cm[0,1]:>8}\n")
            f.write(f"  {'none':>10} {cm[1,0]:>8} {cm[1,1]:>8}\n")

        # Full classification reports for supervised methods
        for name, key in zip(supervised_names, ["logistic", "svm", "knn", "mlp"]):
            preds = all_preds[key]
            f.write(f"\n{name} — full classification report:\n")
            f.write(classification_report(y_test, preds, target_names=LABEL_NAMES, zero_division=0))

    print(f"  Evaluation report: {report_path}")
    print()
    print("=" * 60)
    print("Phase 3 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
