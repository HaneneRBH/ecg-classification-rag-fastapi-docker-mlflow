"""
Bridge between Layer 2 (classifier) and Layer 3 (grounded clinical report).

Takes one ECG recording, runs the trained 1D-CNN to get per-class
probabilities, then asks the RAG layer for a grounded clinical report:

        ECG signal  ->  probabilities (5 classes)  ->  grounded report

This is the whole pipeline in one call, and the function the FastAPI layer
will wrap.

    python src/pipeline/predict_and_report.py \
        --data-root "C:/path/to/ptb-xl" --ecg-id 1 --no-llm

    python src/pipeline/predict_and_report.py \
        --record "C:/path/to/ptb-xl/records100/00000/00001_lr" --no-llm

Preprocessing is taken straight from the Layer-1 pipeline (bandpass filter +
z-score normalisation), so it matches training exactly.

CPU only. Heavy imports (TensorFlow) are deferred into the functions that
need them, so importing this module stays cheap and the report side can be
reused on its own.

DISCLAIMER: research and demonstration only, not for clinical diagnosis.
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "data"))
sys.path.insert(0, os.path.join(_ROOT, "src", "report"))

# Light imports (no TensorFlow): the report layer and the class list.
from explain import explain_prediction  # noqa: E402
from rules import load_rules, SUPER_CLASSES  # noqa: E402

DEFAULT_CKPT = os.path.join(_ROOT, "checkpoints", "ecg_1dcnn_best.keras")


def load_model(ckpt_path=DEFAULT_CKPT):
    """Load the trained Keras classifier, tolerating version differences.

    Checkpoint portability across TensorFlow/Keras versions is a real MLOps
    problem: a .keras file written by Keras 2 may fail to load under a slightly
    different build. We therefore try three routes, in order of convenience:

        1. the given path as-is (native .keras);
        2. the same name with a .h5 extension (legacy HDF5, more portable);
        3. rebuild the architecture from classifier.py and load weights only
           (.weights.h5) — the most robust route, since only the numbers move.

    Returns
    -------
    model : keras.Model
    """
    from tensorflow import keras

    base, _ = os.path.splitext(ckpt_path)
    candidates = [
        ckpt_path,          # e.g. ecg_1dcnn_best.keras
        base + ".h5",       # e.g. ecg_1dcnn_best.h5
    ]

    errors = []
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            return keras.models.load_model(path)
        except Exception as exc:
            errors.append(f"{os.path.basename(path)}: {type(exc).__name__}")

    # Route 3: rebuild the architecture, then load weights only.
    weights_path = base + ".weights.h5"
    if os.path.isfile(weights_path):
        try:
            sys.path.insert(0, os.path.join(_ROOT, "src", "model"))
            from classifier import build_model
            model = build_model(input_shape=(1000, 12), num_classes=5)
            model.load_weights(weights_path)
            return model
        except Exception as exc:
            errors.append(f"{os.path.basename(weights_path)}: {type(exc).__name__}: {exc}")

    if not errors:
        raise FileNotFoundError(
            f"No checkpoint found near: {ckpt_path}\n"
            "Train the model first: python src/model/train.py --data-root ..."
        )
    raise RuntimeError("Could not load any checkpoint. Tried: " + "; ".join(errors))


def load_one_ecg(record_path, fs=100):
    """Load and preprocess a single ECG recording from a WFDB path.

    Parameters
    ----------
    record_path : str
        Path to the record WITHOUT extension (wfdb reads .hea + .dat).
        Example: ".../records100/00000/00001_lr"
    fs : int
        Sampling frequency of the record (100 or 500 Hz).

    Returns
    -------
    signal : np.ndarray, shape (1, n_samples, 12), float32 — ready for the model.
    """
    import numpy as np
    import wfdb
    from preprocess import preprocess

    record = wfdb.rdrecord(record_path)
    raw = record.p_signal.astype(np.float32)        # (n_samples, 12)
    batch = raw[np.newaxis, ...]                     # (1, n_samples, 12)
    return preprocess(batch, fs=fs)


def find_record_path(data_root, ecg_id, sr=100):
    """Resolve the WFDB path of one ECG from its ecg_id, via the database CSV."""
    from loader import load_ptbxl

    df, _ = load_ptbxl(data_root, sampling_rate=sr)
    if ecg_id not in df.index:
        raise KeyError(f"ecg_id {ecg_id} not found in ptbxl_database.csv")
    return df.loc[ecg_id, "signal_path"], df.loc[ecg_id]


def predict(model, signal):
    """Run the classifier on one preprocessed ECG.

    Returns
    -------
    probs : list of float, length 5 — sigmoid outputs in SUPER_CLASSES order.
    """
    preds = model.predict(signal, verbose=0)
    return [float(p) for p in preds[0]]


def predict_and_report(record_path=None, data_root=None, ecg_id=None,
                       ckpt_path=DEFAULT_CKPT, sr=100, threshold=0.5,
                       use_llm=True, model=None, rules=None):
    """Full pipeline: ECG -> probabilities -> grounded clinical report.

    Provide either `record_path` (direct WFDB path) or `data_root` + `ecg_id`.

    Returns
    -------
    dict with the report, the probability table and the record metadata.
    """
    meta = {}

    # --- Resolve the record ------------------------------------------------
    if record_path is None:
        if data_root is None or ecg_id is None:
            raise ValueError("Give record_path, or data_root + ecg_id.")
        record_path, row = find_record_path(data_root, ecg_id, sr=sr)
        meta = {
            "ecg_id": int(ecg_id),
            "age": None if row.get("age") is None else float(row["age"]),
            "sex": "male" if row.get("sex") == 1 else "female",
            "true_labels": [sc for sc in SUPER_CLASSES if row.get(sc) == 1],
        }

    # --- Load, preprocess, predict ----------------------------------------
    signal = load_one_ecg(record_path, fs=sr)
    m = model if model is not None else load_model(ckpt_path)
    probs = predict(m, signal)

    # --- Generate the grounded report -------------------------------------
    out = explain_prediction(probs, threshold=threshold,
                             use_llm=use_llm, rules=rules)

    out["record"] = os.path.basename(record_path)
    if meta:
        out["metadata"] = meta
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="ECG signal -> class probabilities -> grounded clinical report."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--record", help="WFDB record path (no extension).")
    src.add_argument("--ecg-id", type=int, help="ecg_id from ptbxl_database.csv.")
    p.add_argument("--data-root", help="PTB-XL root (required with --ecg-id).")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--sr", type=int, default=100, choices=[100, 500])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no-llm", action="store_true",
                    help="Skip Ollama, use the deterministic template only.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.ecg_id is not None and not args.data_root:
        raise SystemExit("--ecg-id requires --data-root")

    out = predict_and_report(
        record_path=args.record,
        data_root=args.data_root,
        ecg_id=args.ecg_id,
        ckpt_path=args.ckpt,
        sr=args.sr,
        threshold=args.threshold,
        use_llm=not args.no_llm,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))
