"""
model.py — CNN + LSTM model definition.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    LSTM, Dense, Dropout,
    TimeDistributed, BatchNormalization
)
from tensorflow.keras.optimizers import Adam


def build_model(seq_len=8, img_size=64, num_classes=2, lr=0.0003):
    """
    Builds a TimeDistributed CNN + LSTM model for video classification.

    Architecture:
        TimeDistributed CNN (3 blocks) — extracts spatial features per frame
        GlobalAveragePooling2D         — reduces spatial dims to a feature vector
        LSTM(64)                       — learns temporal patterns across frames
        Dense(32) → Dense(num_classes) — classification head

    Args:
        seq_len     (int): Number of input frames per video.
        img_size    (int): Height and width of each frame.
        num_classes (int): Number of output classes.
        lr        (float): Adam learning rate.

    Returns:
        Compiled tf.keras.Model.
    """
    model = Sequential([
        # ── Block 1 ───────────────────────────────────────────
        TimeDistributed(
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            input_shape=(seq_len, img_size, img_size, 3)
        ),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),

        # ── Block 2 ───────────────────────────────────────────
        TimeDistributed(Conv2D(64, (3, 3), activation="relu", padding="same")),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),

        # ── Block 3 ───────────────────────────────────────────
        TimeDistributed(Conv2D(128, (3, 3), activation="relu", padding="same")),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),

        # ── Spatial pooling ───────────────────────────────────
        TimeDistributed(GlobalAveragePooling2D()),

        # ── Temporal learning ─────────────────────────────────
        LSTM(64, return_sequences=False),
        Dropout(0.5),

        # ── Classification head ───────────────────────────────
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=lr),
        metrics=["accuracy"],
    )
    return model
