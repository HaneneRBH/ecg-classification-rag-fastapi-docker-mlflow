"""
Layer 2 — Evaluate a trained ECG 1D-CNN checkpoint.

Loads the best checkpoint, runs it on the test set (fold 10), and produces:
    - figures/confusion_matrix.png    — multi-label confusion (one per class)
    - figures/auroc_per_class.png     — bar chart of AUROC per super-class
    - results/evaluation_report.txt   — human-readable text report

    python src/evaluate.py \
        --data-root "C:/path/to/ptb-xl" \
        --ckpt checkpoints/ecg_1dcnn_best.keras

CPU only. Windows / Spyder safe.
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "data"))
sys.path.insert(0, os.path.join(_ROOT, "src", "model"))


def evaluate(args):
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import (
        roc_auc_score, confusion_matrix, classification_report
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from loader import load_ptbxl, load_signals, split_by_fold, SUPER_CLASSES
    from preprocess import preprocess

    # --- Load data -----------------------------------------------------------
    print(f"Loading PTB-XL from: {args.data_root}")
    df, _ = load_ptbxl(args.data_root, sampling_rate=args.sr)
    max_rec = args.max_records
    signals, labels = load_signals(df, max_records=max_rec)
    signals = preprocess(signals, fs=args.sr)
    splits = split_by_fold(df, signals, labels)

    X_test = splits["test"]["signals"]
    y_test = splits["test"]["labels"]
    print(f"Test set: {X_test.shape[0]} ECGs")

    # --- Load model ----------------------------------------------------------
    ckpt_path = args.ckpt
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = keras.models.load_model(ckpt_path)
    print(f"Loaded: {ckpt_path} ({model.count_params():,} params)")

    # --- Predict -------------------------------------------------------------
    y_pred = model.predict(X_test, batch_size=64)
    y_pred_bin = (y_pred > 0.5).astype(np.int32)

    # --- Create output dirs --------------------------------------------------
    fig_dir = os.path.join(_ROOT, "figures")
    res_dir = os.path.join(_ROOT, "results")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # --- 1. Confusion matrices (one per class) -------------------------------
    n_classes = len(SUPER_CLASSES)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    for i, (sc, ax) in enumerate(zip(SUPER_CLASSES, axes)):
        cm = confusion_matrix(y_test[:, i], y_pred_bin[:, i], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(sc, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"])
        # annotate cells
        for r in range(2):
            for c in range(2):
                color = "white" if cm[r, c] > cm.max() / 2 else "black"
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color=color, fontsize=14)
    fig.suptitle("Confusion Matrices (per super-class, threshold 0.5)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    cm_path = os.path.join(fig_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {cm_path}")
    plt.close(fig)

    # --- 2. AUROC bar chart --------------------------------------------------
    aurocs = []
    for i, sc in enumerate(SUPER_CLASSES):
        try:
            aurocs.append(roc_auc_score(y_test[:, i], y_pred[:, i]))
        except ValueError:
            aurocs.append(0.0)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors = ["#2ecc71" if a >= 0.85 else "#f39c12" if a >= 0.75 else "#e74c3c"
              for a in aurocs]
    bars = ax2.barh(SUPER_CLASSES, aurocs, color=colors, edgecolor="white")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("AUROC")
    ax2.set_title("Test AUROC per Super-Class")
    for bar, val in zip(bars, aurocs):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=11)
    ax2.axvline(x=np.mean(aurocs), color="gray", linestyle="--", alpha=0.7,
                label=f"macro: {np.mean(aurocs):.3f}")
    ax2.legend(loc="lower right")
    plt.tight_layout()
    auroc_path = os.path.join(fig_dir, "auroc_per_class.png")
    fig2.savefig(auroc_path, dpi=150)
    print(f"[saved] {auroc_path}")
    plt.close(fig2)

    # --- 3. Text report ------------------------------------------------------
    report_lines = []
    report_lines.append("=" * 72)
    report_lines.append(" ECG 1D-CNN — TEST SET EVALUATION REPORT")
    report_lines.append("=" * 72)
    report_lines.append("")
    report_lines.append(f"  Checkpoint       : {os.path.basename(ckpt_path)}")
    report_lines.append(f"  Parameters       : {model.count_params():,}")
    report_lines.append(f"  Test ECGs        : {X_test.shape[0]}")
    report_lines.append(f"  Sampling rate    : {args.sr} Hz")
    report_lines.append(f"  Macro AUROC      : {np.mean(aurocs):.4f}")
    report_lines.append("")
    report_lines.append("-" * 72)
    report_lines.append(f"  {'Class':8s}  {'AUROC':>8s}  {'Positives':>10s}  {'Total':>8s}")
    report_lines.append("-" * 72)
    for i, sc in enumerate(SUPER_CLASSES):
        n_pos = int(y_test[:, i].sum())
        report_lines.append(
            f"  {sc:8s}  {aurocs[i]:8.4f}  {n_pos:10d}  {X_test.shape[0]:8d}"
        )
    report_lines.append("-" * 72)
    report_lines.append("")
    report_lines.append("Per-class classification report (threshold 0.5):")
    report_lines.append("")
    for i, sc in enumerate(SUPER_CLASSES):
        cr = classification_report(
            y_test[:, i], y_pred_bin[:, i],
            target_names=[f"not {sc}", sc], zero_division=0
        )
        report_lines.append(f"--- {sc} ---")
        report_lines.append(cr)

    report_path = os.path.join(res_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"[saved] {report_path}")

    print(f"\nDone. Figures in {fig_dir}/, report in {res_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained ECG 1D-CNN.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--ckpt", default=os.path.join(_ROOT, "checkpoints", "ecg_1dcnn_best.keras"))
    p.add_argument("--sr", type=int, default=100, choices=[100, 500])
    p.add_argument("--max-records", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
