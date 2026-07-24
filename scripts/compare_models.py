#!/usr/bin/env python3
"""
Phase 3c: Cross-model comparison — BioCLIP 2.5 vs DINOv3 Large vs DINOv3 Small.

Loads all evaluation reports from disk and produces a unified comparison table,
bar chart, and side-by-side confusion matrices for the best head per backbone.

Usage:
  python3 scripts/compare_models.py

Output:
  data/results/cross_model_comparison.txt
  data/results/cross_model_comparison.png
"""
import sys
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_bundles import EmbeddingBundle


# ── Configuration ──────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
LABELS_PATH = DATA_DIR / "labels_bisonn.npy"

LABEL_NAMES = ["mobbing", "none"]
RANDOM_SEED = 42


BACKBONES = [
    {
        "name": "BioCLIP 2.5",
        "key": "bioclip",
        "bundle": DATA_DIR / "embeddings_bisonn.npz",
        "reports": [DATA_DIR / "results" / "evaluation_report.txt"],
    },
    {
        "name": "DINOv3 Large",
        "key": "dinov3_large",
        "bundle": DATA_DIR / "embeddings_dinov3_large.npz",
        "reports": [DATA_DIR / "results" / "evaluation_report_dinov3_large.txt"],
    },
    {
        "name": "DINOv3 Small",
        "key": "dinov3_small",
        "bundle": DATA_DIR / "embeddings_dinov3_small.npz",
        "reports": [DATA_DIR / "results" / "evaluation_report_dinov3_small.txt"],
    },
]

HEADS = ["Logistic Reg", "Linear SVM", "kNN (k=5)"]


def main():
    print("=" * 60)
    print("BISONN Phase 3c — Cross-Model Comparison")
    print("=" * 60)
    print()

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    labels = np.load(LABELS_PATH)

    # Collect results for each backbone x head
    # We'll re-run the supervised heads here (fast on embeddings) to get
    # consistent metrics + predictions for the comparison plots.
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    results = []  # list of (backbone_name, head_name, metrics, y_true, y_pred, emb_dim)

    for bb in BACKBONES:
        bundle_path = bb["bundle"]
        if not bundle_path.exists():
            print(f"  SKIP {bb['name']} — {bundle_path} not found")
            continue

        print(f"Processing {bb['name']}...")
        bundle = EmbeddingBundle.load(bundle_path)
        X = bundle.features
        y = labels
        emb_dim = X.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED,
        )

        # Logistic regression
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        results.append((bb["name"], "Logistic Reg", y_test, lr_pred, emb_dim))

        # Linear SVM
        svm = SVC(kernel="linear", class_weight="balanced", random_state=RANDOM_SEED)
        svm.fit(X_train, y_train)
        svm_pred = svm.predict(X_test)
        results.append((bb["name"], "Linear SVM", y_test, svm_pred, emb_dim))

        # kNN
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        results.append((bb["name"], "kNN (k=5)", y_test, knn_pred, emb_dim))

    if not results:
        print("No embeddings found. Run extraction + training first.")
        sys.exit(1)

    # Build comparison table
    print()
    print("=" * 90)
    header = f"{'Backbone':<16} {'Head':<14} {'Dim':>5} {'Accuracy':>8} {'MacroF1':>8} {'Mob F1':>7} {'None F1':>8}"
    print(header)
    print("-" * 90)

    lines = []
    best_f1 = -1
    best_entry = None

    for bb_name, head_name, y_true, y_pred, emb_dim in results:
        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        mob_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        none_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        row = f"{bb_name:<16} {head_name:<14} {emb_dim:>5} {acc:>8.3f} {mf1:>8.3f} {mob_f1:>7.3f} {none_f1:>8.3f}"
        print(row)
        lines.append(row)

        if mf1 > best_f1:
            best_f1 = mf1
            best_entry = (bb_name, head_name, y_true, y_pred, acc, mf1)

    print("=" * 90)
    print(f"\nBest overall: {best_entry[0]} + {best_entry[1]} (macroF1={best_f1:.3f})")

    # Save text report
    report_path = RESULTS_DIR / "cross_model_comparison.txt"
    with open(report_path, "w") as f:
        f.write("BISONN Phase 3c — Cross-Model Comparison\n")
        f.write(f"Backbones: {', '.join(bb['name'] for bb in BACKBONES if bb['bundle'].exists())}\n")
        f.write(f"Methods: logistic regression, linear SVM, kNN (class-weighted, 80/20 stratified, seed={RANDOM_SEED})\n\n")
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        f.write("\n".join(lines) + "\n")
        f.write("-" * 90 + "\n")
        f.write(f"\nBest overall: {best_entry[0]} + {best_entry[1]} (macroF1={best_f1:.3f})\n")

        # Confusion matrices
        for bb_name, head_name, y_true, y_pred, _, _ in results:
            cm = confusion_matrix(y_true, y_pred)
            f.write(f"\n{bb_name} + {head_name}:\n")
            f.write(f"  {'':>10} {'mobbing':>8} {'none':>8}\n")
            f.write(f"  {'mobbing':>10} {cm[0,0]:>8} {cm[0,1]:>8}\n")
            f.write(f"  {'none':>10} {cm[1,0]:>8} {cm[1,1]:>8}\n")

    print(f"\nReport: {report_path}")

    # Bar chart: macro-F1 across all backbone x head combos
    bb_names = [r[0] for r in results]
    head_names = [r[1] for r in results]
    f1s = [f1_score(r[2], r[3], average="macro", zero_division=0) for r in results]
    accs = [accuracy_score(r[2], r[3]) for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results))
    width = 0.35
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#2196F3")
    bars2 = ax.bar(x + width/2, f1s, width, label="Macro-F1", color="#FF9800")
    ax.set_ylabel("Score")
    ax.set_title("BISONN Cross-Model Comparison — BioCLIP 2.5 vs DINOv3")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{bb}\n{h}" for bb, h in zip(bb_names, head_names)],
                       rotation=0, fontsize=8)
    ax.legend(); ax.set_ylim(0, 1.05)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x()+bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    chart_path = RESULTS_DIR / "cross_model_comparison.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Bar chart: {chart_path}")

    print()
    print("=" * 60)
    print("Phase 3c complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
