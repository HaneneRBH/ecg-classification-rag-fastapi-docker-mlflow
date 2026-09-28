"""
Layer 2 — 1D-CNN classifier for 12-lead ECG signals (TensorFlow / Keras).

Architecture overview:
    Input: (1000, 12) — 10 seconds × 12 leads at 100 Hz.

    Three convolutional blocks, each:
        Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm → ReLU → MaxPool1D → Dropout

    The filters widen (32 → 64 → 128) while MaxPool shrinks the time axis,
    so the network progressively captures longer-range patterns (individual
    waves → full QRS complexes → inter-beat relationships).

    After the conv blocks:
        GlobalAveragePooling1D — collapses the time axis into one vector
        Dense(64) → ReLU → Dropout
        Dense(5, sigmoid) — multi-label output (one sigmoid per super-class)

    Sigmoid (not softmax) because an ECG can carry MULTIPLE diagnoses
    at once (e.g. MI + STTC). The loss is binary cross-entropy per class.

Why this architecture:
    - 1D convolutions operate directly on the time series (no spectrogram
      needed), which is the standard for ECG classification.
    - Small enough to train on CPU in reasonable time (~1.56 M parameters).
    - BatchNorm + Dropout for regularisation on a moderately sized dataset.

CPU only. Windows / Spyder safe.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape=(1000, 12), num_classes=5):
    """Build a 1D-CNN for multi-label ECG classification.

    Parameters
    ----------
    input_shape : tuple
        (n_samples, n_leads). Default (1000, 12) for 100 Hz, 10 s.
    num_classes : int
        Number of output classes (default 5 super-classes).

    Returns
    -------
    model : keras.Model
    """
    inputs = keras.Input(shape=input_shape, name="ecg_input")

    # --- Block 1: 32 filters, kernel 7 ---------------------------------
    x = layers.Conv1D(32, kernel_size=7, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(32, kernel_size=7, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool1D(pool_size=2)(x)    # 1000 → 500
    x = layers.Dropout(0.2)(x)

    # --- Block 2: 64 filters, kernel 5 ---------------------------------
    x = layers.Conv1D(64, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool1D(pool_size=2)(x)    # 500 → 250
    x = layers.Dropout(0.2)(x)

    # --- Block 3: 128 filters, kernel 3 --------------------------------
    x = layers.Conv1D(128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool1D(pool_size=2)(x)    # 250 → 125
    x = layers.Dropout(0.3)(x)

    # --- Head -----------------------------------------------------------
    x = layers.GlobalAveragePooling1D()(x)  # 125×128 → 128
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name="ecg_1dcnn")
    return model


def print_summary(model):
    """Print model summary and parameter count."""
    model.summary()
    total = model.count_params()
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    print(f"\nTotal params    : {total:,}")
    print(f"Trainable params: {trainable:,}")


if __name__ == "__main__":
    model = build_model()
    print_summary(model)
