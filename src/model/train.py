"""
Layer 2 — Training loop for the ECG 1D-CNN classifier.

Loads the PTB-XL dataset, preprocesses the signals, trains the 1D-CNN,
saves the best checkpoint (by validation AUROC), and plots training curves.

    python src/model/train.py \
        --data-root "C:/path/to/ptb-xl" \
        --epochs 30 --batch-size 64

    # then inspect the runs:
    mlflow ui          # opens http://localhost:5000

What this script does, step by step:
    1. Load PTB-XL via the Layer-1 loader (labels + signals + fold split).
    2. Preprocess: bandpass filter + z-score normalisation.
    3. Build the 1D-CNN model (from classifier.py).
    4. Compile with binary cross-entropy (multi-label) + AdamW.
    5. Train with:
       - EarlyStopping (patience 7, monitors val_auc)
       - ReduceLROnPlateau (patience 3)
       - ModelCheckpoint (saves best val_auc)
    6. Evaluate on the test set: per-class AUROC and overall macro AUROC.
    7. Save training curves (loss + AUC) as a PNG.
    8. Log everything to MLflow (Layer 6): hyperparameters, per-epoch metrics,
       final test scores, and the model itself. MLflow is optional — if it is
       not installed, training proceeds normally and only the local files are
       written.

CPU only. Windows / Spyder safe (if __name__ guard).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Add project paths for sibling imports.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "data"))
sys.path.insert(0, os.path.join(_ROOT, "src", "model"))


# --------------------------------------------------------------------------
# Layer 6 — MLflow experiment tracking (optional)
# --------------------------------------------------------------------------

def _mlflow_available():
    """Return the mlflow module, or None if it is not installed."""
    try:
        import mlflow
        return mlflow
    except ImportError:
        return None


def _mlflow_start(mlflow, args, run_name=None):
    """Start an MLflow run and log the hyperparameters.

    Recent MLflow versions deprecated the plain-directory ("file store")
    backend, so we track into a local SQLite database instead. It is a single
    file, needs no server, and is the backend MLflow now recommends.
    The location can be overridden with the MLFLOW_TRACKING_URI variable.
    """
    uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "sqlite:///" + os.path.join(_ROOT, "mlflow.db").replace("\\", "/"),
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("ecg-1dcnn-ptbxl")
    run = mlflow.start_run(run_name=run_name)
    mlflow.log_params({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "sampling_rate": args.sr,
        "max_records": args.max_records or "all",
        "optimizer": "AdamW",
        "loss": "binary_crossentropy",
        "architecture": "1D-CNN (3 conv blocks)",
    })
    return run


def _mlflow_log_history(mlflow, history):
    """Log per-epoch metrics so MLflow can plot the curves."""
    for epoch in range(len(history.history["loss"])):
        metrics = {}
        for key, values in history.history.items():
            if epoch < len(values):
                # MLflow metric names cannot contain some characters.
                metrics[key.replace("@", "_")] = float(values[epoch])
        mlflow.log_metrics(metrics, step=epoch)


def train(args):
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import roc_auc_score

    from loader import load_ptbxl, load_signals, split_by_fold, SUPER_CLASSES
    from preprocess import preprocess
    from classifier import build_model

    # --- 0. MLflow (optional) ------------------------------------------------
    mlflow = _mlflow_available() if not args.no_mlflow else None
    if mlflow is not None:
        try:
            _mlflow_start(mlflow, args)
            print("[mlflow] tracking enabled -> mlflow.db")
        except Exception as exc:
            # Experiment tracking is a convenience: never let it stop a run.
            print(f"[mlflow] disabled ({type(exc).__name__}: {str(exc)[:80]})")
            mlflow = None
    else:
        print("[mlflow] not tracking (module absent or --no-mlflow)")

    # --- 1. Load data --------------------------------------------------------
    print(f"Loading PTB-XL from: {args.data_root}")
    df, _ = load_ptbxl(args.data_root, sampling_rate=args.sr)

    max_rec = args.max_records  # None = all
    print(f"Loading signals (max_records={max_rec})...")
    signals, labels = load_signals(df, max_records=max_rec)
    print(f"  signals: {signals.shape}, labels: {labels.shape}")

    # --- 2. Preprocess -------------------------------------------------------
    print("Preprocessing (bandpass + normalisation)...")
    signals = preprocess(signals, fs=args.sr)

    # --- 3. Split ------------------------------------------------------------
    splits = split_by_fold(df, signals, labels)
    X_train, y_train = splits["train"]["signals"], splits["train"]["labels"]
    X_val,   y_val   = splits["val"]["signals"],   splits["val"]["labels"]
    X_test,  y_test  = splits["test"]["signals"],  splits["test"]["labels"]
    print(f"  train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}")

    # --- 4. Build model ------------------------------------------------------
    input_shape = (X_train.shape[1], X_train.shape[2])  # (1000, 12)
    model = build_model(input_shape=input_shape, num_classes=len(SUPER_CLASSES))
    model.summary()

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc", multi_label=True)],
    )

    # --- 5. Callbacks --------------------------------------------------------
    os.makedirs(os.path.join(_ROOT, "checkpoints"), exist_ok=True)
    ckpt_path = os.path.join(_ROOT, "checkpoints", "ecg_1dcnn_best.keras")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=7,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", patience=3,
            factor=0.5, min_lr=1e-6, verbose=1,
        ),
    ]

    # --- 6. Train ------------------------------------------------------------
    print(f"\nTraining: {args.epochs} epochs, batch {args.batch_size}, lr {args.lr}")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"Training done in {elapsed / 60:.1f} min")

    # --- 7. Evaluate on test set ---------------------------------------------
    print("\n=== Test Set Evaluation ===")
    y_pred = model.predict(X_test, batch_size=args.batch_size)

    try:
        macro_auc = roc_auc_score(y_test, y_pred, average="macro")
        print(f"Macro AUROC: {macro_auc:.4f}")
        for i, sc in enumerate(SUPER_CLASSES):
            if y_test[:, i].sum() > 0:
                auc_i = roc_auc_score(y_test[:, i], y_pred[:, i])
                print(f"  {sc:5s}: AUROC {auc_i:.4f}")
            else:
                print(f"  {sc:5s}: no positive samples in test set")
    except ValueError as e:
        print(f"AUROC computation skipped ({e})")
        macro_auc = None

    # --- 8. Save metrics -----------------------------------------------------
    metrics = {
        "epochs_run": len(history.history["loss"]),
        "train_time_min": round(elapsed / 60, 1),
        "best_val_auc": float(max(history.history.get("val_auc", [0]))),
        "test_macro_auroc": float(macro_auc) if macro_auc else None,
        "params": model.count_params(),
    }
    os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)
    # --- MLflow: log final metrics and the model ----------------------------
    if mlflow is not None:
        _mlflow_log_history(mlflow, history)
        mlflow.log_metrics({
            "test_macro_auroc": float(macro_auc) if macro_auc else 0.0,
            "best_val_auc": metrics["best_val_auc"],
            "train_time_min": metrics["train_time_min"],
        })
        for i, sc in enumerate(SUPER_CLASSES):
            if y_test[:, i].sum() > 0:
                mlflow.log_metric(f"test_auroc_{sc}", float(roc_auc_score(y_test[:, i], y_pred[:, i])))
        try:
            mlflow.keras.log_model(model, "model")
        except Exception as exc:
            print(f"[mlflow] model logging skipped: {type(exc).__name__}")

    metrics_path = os.path.join(_ROOT, "results", "ecg_1dcnn_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # --- 9. Plot training curves --------------------------------------------
    _plot_curves(history, os.path.join(_ROOT, "figures"))
    _save_text_report(metrics, SUPER_CLASSES, y_test, y_pred,
                      os.path.join(_ROOT, "results"))

    # --- MLflow: attach the generated files and close the run ---------------
    if mlflow is not None:
        for f in ["figures/training_curves.png",
                  "results/ecg_1dcnn_metrics.json",
                  "results/evaluation_report.txt"]:
            path = os.path.join(_ROOT, f)
            if os.path.isfile(path):
                mlflow.log_artifact(path)
        mlflow.end_run()
        print("[mlflow] run closed. View with: mlflow ui")


def _save_text_report(metrics, class_names, y_true, y_pred, out_dir):
    """Save a human-readable text report (like per_device_metrics.txt)."""
    from sklearn.metrics import roc_auc_score
    import json

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "evaluation_report.txt")
    lines = []
    lines.append("=" * 72)
    lines.append(" ECG 1D-CNN — EVALUATION REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  Total params      : {metrics['params']:,}")
    lines.append(f"  Epochs run        : {metrics['epochs_run']}")
    lines.append(f"  Training time     : {metrics['train_time_min']} min")
    lines.append(f"  Best val AUROC    : {metrics['best_val_auc']:.4f}")
    lines.append(f"  Test macro AUROC  : {metrics['test_macro_auroc']:.4f}")
    lines.append("")
    lines.append("-" * 72)
    lines.append(f"  {'Class':8s}  {'AUROC':>8s}  {'Positives':>10s}  {'Total':>8s}")
    lines.append("-" * 72)
    for i, sc in enumerate(class_names):
        n_pos = int(y_true[:, i].sum())
        n_tot = len(y_true)
        try:
            auc_i = roc_auc_score(y_true[:, i], y_pred[:, i])
            lines.append(f"  {sc:8s}  {auc_i:8.4f}  {n_pos:10d}  {n_tot:8d}")
        except ValueError:
            lines.append(f"  {sc:8s}  {'N/A':>8s}  {n_pos:10d}  {n_tot:8d}")
    lines.append("-" * 72)
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Evaluation report saved to {path}")


def _plot_curves(history, out_dir):
    """Save loss and AUC curves as a PNG."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for saving
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"], label="train")
    ax1.plot(history.history["val_loss"], label="val")
    ax1.set_title("Loss (binary cross-entropy)")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["auc"], label="train")
    ax2.plot(history.history["val_auc"], label="val")
    ax2.set_title("AUROC")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("ECG 1D-CNN Training Curves", fontsize=13)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    print(f"Training curves saved to {path}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Train the ECG 1D-CNN classifier.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--sr", type=int, default=100, choices=[100, 500])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-records", type=int, default=None,
                    help="Limit records for quick test (default: all).")
    p.add_argument("--no-mlflow", action="store_true",
                    help="Disable MLflow tracking even if it is installed.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
