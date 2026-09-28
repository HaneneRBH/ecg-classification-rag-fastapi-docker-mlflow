"""
Layer 1 — PTB-XL data loader.

Reads the PTB-XL dataset (WFDB format), maps each ECG to its diagnostic
super-class(es), and splits into train / val / test by the recommended
stratified folds (1-8 / 9 / 10).

    python src/data/loader.py --data-root /path/to/ptb-xl --show

What this script does, step by step:
    1. Read ptbxl_database.csv  → one row per ECG, with scp_codes (a dict).
    2. Read scp_statements.csv  → maps each SCP code to a diagnostic super-class.
    3. For each ECG, aggregate the SCP codes into a multi-hot vector over the
       five super-classes: NORM, MI, STTC, CD, HYP.
    4. Load the 100 Hz signals via wfdb (shape: 1000 samples × 12 leads).
    5. Split by strat_fold: folds 1-8 = train, fold 9 = val, fold 10 = test.

CPU only. Windows / Spyder safe (if __name__ guard, num_workers=0).
"""

import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd
import wfdb

# The five diagnostic super-classes used for classification.
SUPER_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_ptbxl(data_root, sampling_rate=100):
    """Load the full PTB-XL dataset and return (df, scp_df).

    Parameters
    ----------
    data_root : str
        Path to the extracted PTB-XL folder (contains ptbxl_database.csv).
    sampling_rate : int
        100 or 500 Hz. We default to 100 Hz for fast CPU prototyping.

    Returns
    -------
    df : pd.DataFrame
        One row per ECG. Columns include ecg_id, patient_id, age, sex,
        scp_codes (dict), strat_fold, filename_lr/hr, and the computed
        multi-hot super-class columns (NORM, MI, STTC, CD, HYP).
    scp_df : pd.DataFrame
        The SCP statements table (maps SCP codes → super-classes).
    """
    # --- 1. Read the main database -------------------------------------------
    db_path = os.path.join(data_root, "ptbxl_database.csv")
    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"ptbxl_database.csv not found in {data_root}.\n"
            "Make sure data_root points to the extracted PTB-XL folder."
        )
    df = pd.read_csv(db_path, index_col="ecg_id")
    # scp_codes is stored as a string repr of a dict → parse it.
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    # --- 2. Read the SCP statements table ------------------------------------
    scp_path = os.path.join(data_root, "scp_statements.csv")
    scp_df = pd.read_csv(scp_path, index_col=0)
    # Keep only diagnostic statements (the ones that have a super-class).
    scp_df = scp_df[scp_df["diagnostic"] == 1.0]

    # --- 3. Map each ECG → multi-hot super-class vector ----------------------
    def _aggregate_superclass(scp_dict):
        """Return a dict {super_class: max_likelihood} for one ECG."""
        result = {sc: 0.0 for sc in SUPER_CLASSES}
        for code, likelihood in scp_dict.items():
            if code in scp_df.index:
                sc = scp_df.loc[code, "diagnostic_class"]
                if sc in result:
                    result[sc] = max(result[sc], likelihood)
        return result

    agg = df["scp_codes"].apply(_aggregate_superclass)
    for sc in SUPER_CLASSES:
        # Binary: 1 if likelihood > 0, else 0.
        df[sc] = agg.apply(lambda d, c=sc: 1 if d[c] > 0 else 0)

    # --- 4. Record the signal file path for the chosen sampling rate ---------
    if sampling_rate == 100:
        df["signal_path"] = df["filename_lr"].apply(
            lambda f: os.path.join(data_root, f)
        )
    else:
        df["signal_path"] = df["filename_hr"].apply(
            lambda f: os.path.join(data_root, f)
        )

    return df, scp_df


def load_signals(df, max_records=None):
    """Load the raw ECG signals from disk.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'signal_path' column (set by load_ptbxl).
    max_records : int or None
        If set, only load the first N records (useful for quick tests).

    Returns
    -------
    signals : np.ndarray, shape (N, n_samples, 12), float32
        The raw 12-lead ECG signals.
    labels : np.ndarray, shape (N, 5), int
        Multi-hot labels for the five super-classes.
    """
    paths = df["signal_path"].values
    if max_records is not None:
        paths = paths[:max_records]
        df = df.iloc[:max_records]

    signals = []
    for p in paths:
        record = wfdb.rdrecord(p)
        signals.append(record.p_signal)  # (n_samples, 12)

    signals = np.array(signals, dtype=np.float32)
    labels = df[SUPER_CLASSES].values.astype(np.int32)

    return signals, labels


def split_by_fold(df, signals, labels):
    """Split into train / val / test by the recommended PTB-XL folds.

    Folds 1-8 → train, fold 9 → val, fold 10 → test.
    All ECGs of the same patient stay in the same fold (no leakage).

    Returns
    -------
    dict with keys 'train', 'val', 'test', each a dict of
    {'signals': np.ndarray, 'labels': np.ndarray}.
    """
    folds = df["strat_fold"].values
    if len(folds) != len(signals):
        folds = folds[: len(signals)]

    train_mask = folds <= 8
    val_mask = folds == 9
    test_mask = folds == 10

    return {
        "train": {"signals": signals[train_mask], "labels": labels[train_mask]},
        "val":   {"signals": signals[val_mask],   "labels": labels[val_mask]},
        "test":  {"signals": signals[test_mask],  "labels": labels[test_mask]},
    }


def summarise(df):
    """Print a summary of the dataset."""
    n = len(df)
    print(f"Total ECGs   : {n}")
    print(f"Patients     : {df['patient_id'].nunique()}")
    print(f"Age (median) : {df['age'].median():.0f}")
    print(f"Sex          : {(df['sex']==1).sum()} male / {(df['sex']==0).sum()} female")
    print()
    print("Super-class distribution:")
    for sc in SUPER_CLASSES:
        count = df[sc].sum()
        print(f"  {sc:5s}: {count:6d}  ({100*count/n:.1f}%)")
    print()
    for fold_id in [10, 9]:
        tag = "test" if fold_id == 10 else "val"
        n_f = (df["strat_fold"] == fold_id).sum()
        print(f"  fold {fold_id} ({tag:5s}): {n_f} ECGs")
    n_train = (df["strat_fold"] <= 8).sum()
    print(f"  folds 1-8 (train): {n_train} ECGs")


def parse_args():
    p = argparse.ArgumentParser(description="Load and inspect the PTB-XL dataset.")
    p.add_argument("--data-root", required=True,
                    help="Path to the extracted PTB-XL folder.")
    p.add_argument("--sr", type=int, default=100, choices=[100, 500],
                    help="Sampling rate (default 100 Hz).")
    p.add_argument("--show", action="store_true",
                    help="Plot the first ECG after loading.")
    p.add_argument("--max-records", type=int, default=None,
                    help="Only load this many records (for quick testing).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading PTB-XL from: {args.data_root}")
    print(f"Sampling rate: {args.sr} Hz")
    print()

    df, scp_df = load_ptbxl(args.data_root, sampling_rate=args.sr)
    summarise(df)

    print(f"\nLoading signals (max_records={args.max_records})...")
    signals, labels = load_signals(df, max_records=args.max_records)
    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape : {labels.shape}")

    splits = split_by_fold(df, signals, labels)
    for name, data in splits.items():
        print(f"  {name:5s}: signals {data['signals'].shape}, labels {data['labels'].shape}")

    if args.show:
        import matplotlib.pyplot as plt
        record = wfdb.rdrecord(df["signal_path"].iloc[0])
        wfdb.plot_wfdb(record=record,
                       title=f"PTB-XL - ECG {df.index[0]} (12 leads, {args.sr} Hz)",
                       figsize=(10, 12))
        plt.show()
