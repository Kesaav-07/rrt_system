"""
ai/train_bilstm.py
------------------
Train the Bidirectional LSTM model for RRT deterioration forecasting.

The model learns to predict future vital signs (T+4h, T+8h) from an
8-step sequence of 6 vital parameters.

Inputs  : (batch, 8, 6)  — [RR, SpO2, HR, SBP, Temperature, AVPU]
Outputs : (batch, 12)    — [RR_T4, SpO2_T4, HR_T4, SBP_T4, Temp_T4, AVPU_T4,
                             RR_T8, SpO2_T8, HR_T8, SBP_T8, Temp_T8, AVPU_T8]

Run:
    python ai/train_bilstm.py
"""

from __future__ import annotations

import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BILSTM_MODEL_PATH,
    SCALER_X_PATH,
    SCALER_Y_PATH,
    SEQUENCE_LENGTH,
    N_FEATURES,
    FEATURE_COLS,
    TARGET_COLS,
    VITAL_HISTORY_FILE,
    TRAINED_DIR,
    VITAL_RANGES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------------------------------------------------------------------------
# Synthetic data generation (used when real history is sparse)
# ---------------------------------------------------------------------------

def _generate_synthetic_vitals(n_patients: int = 200, steps_per_patient: int = 40) -> pd.DataFrame:
    """
    Generate synthetic patient vital histories for training.

    Args:
        n_patients:        Number of synthetic patients.
        steps_per_patient: Number of time steps per patient.

    Returns:
        DataFrame with columns matching FEATURE_COLS plus patient_id and recorded_at.
    """
    rng = np.random.default_rng(42)
    rows: list[dict] = []

    for pid in range(n_patients):
        # Baseline vitals for this patient
        rr_base   = rng.uniform(12, 22)
        spo2_base = rng.uniform(93, 99)
        hr_base   = rng.uniform(60, 100)
        sbp_base  = rng.uniform(100, 140)
        temp_base = rng.uniform(36.0, 37.5)
        avpu_base = 0.0

        for t in range(steps_per_patient):
            drift = t / steps_per_patient
            rr   = float(np.clip(rr_base   + rng.normal(0, 1.5) + drift * rng.choice([-2, 0, 2]), *VITAL_RANGES["RR"]))
            spo2 = float(np.clip(spo2_base + rng.normal(0, 0.5) - drift * rng.uniform(0, 1),      *VITAL_RANGES["SpO2"]))
            hr   = float(np.clip(hr_base   + rng.normal(0, 3.0) + drift * rng.choice([-5, 0, 5]), *VITAL_RANGES["HR"]))
            sbp  = float(np.clip(sbp_base  + rng.normal(0, 4.0) + drift * rng.choice([-8, 0, 8]), *VITAL_RANGES["SBP"]))
            temp = float(np.clip(temp_base + rng.normal(0, 0.1) + drift * rng.uniform(0, 0.3),    *VITAL_RANGES["Temperature"]))
            avpu = float(int(np.clip(round(avpu_base + rng.choice([0, 0, 0, 1]) * (drift > 0.7)), 0, 3)))

            rows.append({
                "patient_id":  f"SYN{pid:04d}",
                "recorded_at": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=t),
                "RR":          rr,
                "SpO2":        spo2,
                "HR":          hr,
                "SBP":         sbp,
                "Temperature": temp,
                "AVPU":        avpu,
            })

    return pd.DataFrame(rows)


def _load_or_generate_history() -> pd.DataFrame:
    """Load real vital history or fall back to synthetic data."""
    col_map = {
        "respiratory_rate": "RR",
        "spo2":             "SpO2",
        "heart_rate":       "HR",
        "systolic_bp":      "SBP",
        "temperature":      "Temperature",
        "avpu_encoded":     "AVPU",
    }

    if os.path.exists(VITAL_HISTORY_FILE):
        try:
            df = pd.read_csv(VITAL_HISTORY_FILE)
            df.rename(columns=col_map, inplace=True)
            missing = [c for c in FEATURE_COLS if c not in df.columns]
            if not missing and len(df) >= SEQUENCE_LENGTH * 10:
                logger.info(f"Loaded real history: {len(df)} rows from {VITAL_HISTORY_FILE}")
                return df
        except Exception as exc:
            logger.warning(f"Could not load vital history: {exc}")

    logger.info("Generating synthetic training data …")
    return _generate_synthetic_vitals()


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def _build_sequences(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays for sequence-to-sequence training.

    For each patient, slide a window of SEQUENCE_LENGTH steps and predict
    the vitals at +4 and +8 steps ahead.

    Args:
        df: DataFrame with FEATURE_COLS columns (and patient_id).

    Returns:
        X: (N, SEQUENCE_LENGTH, N_FEATURES)
        y: (N, 12)  — 6 features × 2 horizons
    """
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    horizon_4 = 4
    horizon_8 = 8

    for pid, group in df.groupby("patient_id"):
        grp = group[FEATURE_COLS].dropna().reset_index(drop=True)
        vals = grp.values.astype(np.float32)

        for i in range(len(vals) - SEQUENCE_LENGTH - horizon_8):
            seq   = vals[i : i + SEQUENCE_LENGTH]
            t4    = vals[i + SEQUENCE_LENGTH + horizon_4 - 1]
            t8    = vals[i + SEQUENCE_LENGTH + horizon_8 - 1]
            X_list.append(seq)
            y_list.append(np.concatenate([t4, t8]))

    if not X_list:
        raise ValueError("Insufficient data to build training sequences.")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_model(sequence_len: int, n_features: int, n_outputs: int):
    """
    Build the Bidirectional LSTM model.

    Args:
        sequence_len: Number of time steps.
        n_features:   Number of input features.
        n_outputs:    Number of output predictions.

    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf  # type: ignore

    inp = tf.keras.Input(shape=(sequence_len, n_features), name="vitals_sequence")

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True), name="bilstm_1"
    )(inp)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=False), name="bilstm_2"
    )(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(x)
    out = tf.keras.layers.Dense(n_outputs, name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ---------------------------------------------------------------------------
# Public training API
# ---------------------------------------------------------------------------

def train(epochs: int = 30, batch_size: int = 64, val_split: float = 0.15) -> dict:
    """
    Train the BiLSTM model and save artifacts to TRAINED_DIR.

    Args:
        epochs:     Training epochs.
        batch_size: Mini-batch size.
        val_split:  Fraction of data reserved for validation.

    Returns:
        Dict with val_loss and val_mae from the final epoch.
    """
    import joblib  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    os.makedirs(TRAINED_DIR, exist_ok=True)

    logger.info("Loading/generating training data …")
    df = _load_or_generate_history()

    logger.info("Building sequences …")
    X_raw, y_raw = _build_sequences(df)
    logger.info(f"Sequences: X={X_raw.shape}, y={y_raw.shape}")

    # Flatten for scaler fit
    T, F = SEQUENCE_LENGTH, N_FEATURES
    X_flat = X_raw.reshape(-1, T * F)

    scaler_x = StandardScaler()
    X_scaled_flat = scaler_x.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(-1, T, F)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_raw)

    # Shuffle
    idx = np.random.permutation(len(X_scaled))
    X_scaled, y_scaled = X_scaled[idx], y_scaled[idx]

    # Build & train
    model = _build_model(T, F, len(TARGET_COLS))
    model.summary(print_fn=logger.info)

    callbacks = [
        __import__("tensorflow").keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        __import__("tensorflow").keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    history = model.fit(
        X_scaled, y_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1,
    )

    # Persist
    model.save(BILSTM_MODEL_PATH)
    joblib.dump(scaler_x, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    val_loss = float(history.history["val_loss"][-1])
    val_mae  = float(history.history["val_mae"][-1])
    logger.info(f"Training complete — val_loss={val_loss:.4f}, val_mae={val_mae:.4f}")
    logger.info(f"Model saved to {BILSTM_MODEL_PATH}")

    return {"val_loss": val_loss, "val_mae": val_mae}


if __name__ == "__main__":
    metrics = train()
    print(f"Done. val_loss={metrics['val_loss']:.4f}, val_mae={metrics['val_mae']:.4f}")
