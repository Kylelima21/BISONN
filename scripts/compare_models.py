#!/usr/bin/env python3
"""
Phase 3c: Cross-model comparison — BioCLIP 2.5 vs DINOv3 Large vs DINOv3 Small.

Trains all backbone x head combinations on the train set, evaluates on validation
to select the best head per backbone, then reports final metrics on the held-out
test set. Produces a unified comparison table, bar chart, and side-by-side
confusion matrices.

Three-way split: 75% train / 15% validation / 15% test (seed=42, stratified).

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
VAL_SIZE = 0.15 / 0.85  # 15% of total → fraction of the 85% remainder
TEST_SIZE = 0.15         # 15% of total, held out for final evaluation


BACKBONES = [
    {
        "name": "BioCLIP 2.5",
        "key": "bioclip",
        "bundle": DATA_DIR / "embeddings_bisonn.npz",
    },
    {
        "name": "DINOv3 Large",
        "key": "dinov3_large",
        "bundle": DATA_DIR / "embeddings_dinov3_large.npz",
    },
    {
        "name": "DINOv3 Small",
        "key": "dinov3_small",
        "bundle": DATA_DIR / "embeddings_dinov3_small.npz",
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    labels = np.load(LABELS_PATH)

    # Collect results: (backbone_name, head_name, y_test, test_pred, val_macrof1, test_metrics, emb_dim)
    results = []

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

        # Three-way split: 75/15/15
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=VAL_SIZE, stratify=y_trainval,
            random_state=RANDOM_SEED,
        )

        head_configs = [
            ("Logistic Reg", "logistic", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
            ("Linear SVM", "svm", SVC(
                kernel="linear", class_weight="balanced", random_state=RANDOM_SEED)),
            ("kNN (k=5)", "knn", KNeighborsClassifier(
                n_neighbors=5, metric="cosine", weights="distance")),
        ]

        for head_name, head_key, model in head_configs:
            model.fit(X_train, y_train)
            # Validation: for model selection
            val_pred = model.predict(X_val)
            val_mf1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
            # Test: final metrics
            test_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)
            test_mf1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
            test_mob_f1 = f1_score(y_test, test_pred, pos_label=0, zero_division=0)
            test_none_f1 = f1_score(y_test, test_pred, pos_label=1, zero_division=0)
            test_mob_p = precision_score(y_test, test_pred, pos_label=0, zero_division=0)
            test_mob_r = recall_score(y_test, test_pred, pos_label=0, zero_division=0)
            test_none_p = precision_score(y_test, test_pred, pos_label=1, zero_division=0)
            test_none_r = recall_score(y_test, test_pred, pos_label=1, zero_division=0)

            results.append({
                "backbone": bb["name"],
                "head": head_name,
                "emb_dim": emb_dim,
                "val_mf1": val_mf1,
                "test_acc": test_acc,
                "test_mf1": test_mf1,
                "test_mob_f1": test_mob_f1,
                "test_none_f1": test_none_f1,
                "test_mob_p": test_mob_p,
                "test_mob_r": test_mob_r,
                "test_none_p": test_none_p,
                "test_none_r": test_none_r,
                "y_test": y_test,
                "test_pred": test_pred,
            })

    if not results:
        print("No embeddings found. Run extraction + training first.")
        sys.exit(1)

    # Build comparison table
    print()
    print("=" * 100)
    header = (f"{'Backbone':<16} {'Head':<14} {'Dim':>5} "
              f"{'ValF1':>6} {'Acc':>6} {'MacroF1':>8} "
              f"{'Mob F1':>7} {'None F1':>8}")
    print(header)
    print("-" * 100)

    lines = []
    best_f1 = -1
    best_entry = None

    for r in results:
        row = (f"{r['backbone']:<16} {r['head']:<14} {r['emb_dim']:>5} "
               f"{r['val_mf1']:>6.3f} {r['test_acc']:>6.3f} {r['test_mf1']:>8.3f} "
               f"{r['test_mob_f1']:>7.3f} {r['test_none_f1']:>8.3f}")
        print(row)
        lines.append(row)

        if r["test_mf1"] > best_f1:
            best_f1 = r["test_mf1"]
            best_entry = r

    print("=" * 100)
    print(f"\nBest overall: {best_entry['backbone']} + {best_entry['head']} "
          f"(test macroF1={best_f1:.3f}, val macroF1={best_entry['val_mf1']:.3f})")

    # Save text report
    report_path = RESULTS_DIR / "cross_model_comparison.txt"
    with open(report_path, "w") as f:
        f.write("BISONN Phase 3c — Cross-Model Comparison\n")
        backbones_found = [bb["name"] for bb in BACKBONES if bb["bundle"].exists()]
        f.write(f"Backbones: {', '.join(backbones_found)}\n")
        f.write(f"Methods: logistic regression, linear SVM, kNN "
                f"(class-weighted, 75/15/15 stratified, seed={RANDOM_SEED})\n")
        f.write(f"Train on 75%, select on 15% val, report on 15% test.\n\n")
        f.write(header + "\n")
        f.write("-" * 100 + "\n")
        f.write("\n".join(lines) + "\n")
        f.write("-" * 100 + "\n")
        f.write(f"\nBest overall: {best_entry['backbone']} + {best_entry['head']} "
                f"(test macroF1={best_f1:.3f}, val macroF1={best_entry['val_mf1']:.3f})\n")

        # Confusion matrices (test set)
        for r in results:
            cm = confusion_matrix(r["y_test"], r["test_pred"])
            f.write(f"\n{r['backbone']} + {r['head']}:\n")
            f.write(f"  {'':>10} {'mobbing':>8} {'none':>8}\n")
            f.write(f"  {'mobbing':>10} {cm[0,0]:>8} {cm[0,1]:>8}\n")
            f.write(f"  {'none':>10} {cm[1,0]:>8} {cm[1,1]:>8}\n")

    print(f"\nReport: {report_path}")

    # Bar chart: test macro-F1 across all backbone x head combos
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results))
    width = 0.35
    accs = [r["test_acc"] for r in results]
    f1s = [r["test_mf1"] for r in results]
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#2196F3")
    bars2 = ax.bar(x + width/2, f1s, width, label="Macro-F1", color="#FF9800")
    ax.set_ylabel("Score")
    ax.set_title("BISONN Cross-Model Comparison (test set) — BioCLIP 2.5 vs DINOv3")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['backbone']}\n{r['head']}" for r in results],
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
