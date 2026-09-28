"""
Layer 1 — ECG signal preprocessing.

Applies two standard operations to the raw 12-lead signals before they
enter the classifier:

    1. Bandpass filter (0.5 – 40 Hz) — removes baseline wander (< 0.5 Hz,
       caused by breathing and electrode drift) and high-frequency noise
       (> 40 Hz, mostly power-line interference and muscle artefacts).
    2. Per-lead z-score normalisation — centres each lead to zero mean and
       unit variance, so the model sees consistent scale across patients
       and leads.

    python src/data/preprocess.py --data-root /path/to/ptb-xl --show

CPU only. Windows / Spyder safe.
"""

import argparse
import os
import sys

import numpy as np

# Add project root to path for sibling imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def bandpass_filter(signals, fs=100, lowcut=0.5, highcut=40.0, order=3):
    """Apply a Butterworth bandpass filter to each ECG signal.

    Parameters
    ----------
    signals : np.ndarray, shape (N, n_samples, 12)
    fs : int — sampling frequency (Hz).
    lowcut, highcut : float — passband edges (Hz).
    order : int — filter order.

    Returns
    -------
    filtered : np.ndarray, same shape as input.
    """
    from scipy.signal import butter, sosfiltfilt

    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    filtered = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        for lead in range(signals.shape[2]):
            filtered[i, :, lead] = sosfiltfilt(sos, signals[i, :, lead])
    return filtered


def normalise(signals):
    """Per-lead z-score normalisation: (x - mean) / std.

    Applied independently to each lead of each ECG, so every lead has
    zero mean and unit variance.

    Parameters
    ----------
    signals : np.ndarray, shape (N, n_samples, 12)

    Returns
    -------
    normed : np.ndarray, same shape.
    """
    normed = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        for lead in range(signals.shape[2]):
            x = signals[i, :, lead]
            mu = x.mean()
            sigma = x.std()
            if sigma < 1e-8:
                normed[i, :, lead] = x - mu
            else:
                normed[i, :, lead] = (x - mu) / sigma
    return normed


def preprocess(signals, fs=100):
    """Full preprocessing pipeline: bandpass filter + normalise.

    Parameters
    ----------
    signals : np.ndarray, shape (N, n_samples, 12)
    fs : int — sampling frequency.

    Returns
    -------
    processed : np.ndarray, same shape, float32.
    """
    filtered = bandpass_filter(signals, fs=fs)
    normed = normalise(filtered)
    return normed.astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess PTB-XL signals.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--sr", type=int, default=100, choices=[100, 500])
    p.add_argument("--max-records", type=int, default=50,
                    help="Records to load for the demo (default 50).")
    p.add_argument("--show", action="store_true",
                    help="Plot raw vs. filtered vs. normalised for one ECG.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from loader import load_ptbxl, load_signals

    print(f"Loading {args.max_records} records at {args.sr} Hz...")
    df, _ = load_ptbxl(args.data_root, sampling_rate=args.sr)
    signals, labels = load_signals(df, max_records=args.max_records)
    print(f"Raw signals: {signals.shape}")

    processed = preprocess(signals, fs=args.sr)
    print(f"Processed  : {processed.shape}")
    print(f"Mean (lead 0, ECG 0): {processed[0, :, 0].mean():.4f}")
    print(f"Std  (lead 0, ECG 0): {processed[0, :, 0].std():.4f}")

    if args.show:
        import matplotlib.pyplot as plt

        ecg_idx = 0
        lead = 1  # lead II is typically the clearest
        t = np.arange(signals.shape[1]) / args.sr

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t, signals[ecg_idx, :, lead], linewidth=0.7)
        axes[0].set_title("Raw signal (lead II)")
        axes[0].set_ylabel("mV")

        filtered = bandpass_filter(signals[ecg_idx:ecg_idx+1], fs=args.sr)
        axes[1].plot(t, filtered[0, :, lead], linewidth=0.7, color="tab:orange")
        axes[1].set_title("After bandpass filter (0.5 - 40 Hz)")
        axes[1].set_ylabel("mV")

        axes[2].plot(t, processed[ecg_idx, :, lead], linewidth=0.7, color="tab:green")
        axes[2].set_title("After normalisation (z-score)")
        axes[2].set_ylabel("z")
        axes[2].set_xlabel("Time (s)")

        fig.suptitle("ECG preprocessing pipeline", fontsize=13)
        plt.tight_layout()
        plt.show()
