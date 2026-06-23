"""
ai/predict_bilstm.py
--------------------
Bi-LSTM inference module for the RRT Surveillance System.

Design:
    1. Load Bi-LSTM model + scalers (lazy, single load per process).
    2. Accept an 8-step sequence of 6 vitals as input.
    3. Predict future vitals at T+4h and T+8h.
    4. Pass predicted vitals through rrt_calculator.py to obtain future RRT scores.
    5. Return a structured result dict.

IMPORTANT:
    The model predicts FUTURE VITALS, not RRT directly.
    RRT score calculation always goes through ai/rrt_calculator.py.
"""

from __future__ import annotations

import os
import sys
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BILSTM_MODEL_PATH,
    SCALER_X_PATH,
    SCALER_Y_PATH,
    SEQUENCE_LENGTH,
    N_FEATURES,
    FEATURE_COLS,
    TARGET_COLS,
    AVPU_DECODING,
    VITAL_RANGES,
)
from ai.rrt_calculator import calculate_rrt_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_model    = None
_scaler_x = None
_scaler_y = None


def _load_artifacts() -> bool:
    """
    Load model and scalers if not already loaded.

    Returns:
        True if artifacts loaded successfully, False otherwise.
    """
    global _model, _scaler_x, _scaler_y

    if _model is not None:
        return True

    if not os.path.exists(BILSTM_MODEL_PATH):
        logger.warning(
            f"BiLSTM model not found at {BILSTM_MODEL_PATH}. "
            "Run python ai/train_bilstm.py to train the model."
        )
        return False

    try:
        import joblib
        import tensorflow as tf  # type: ignore

        logger.info(f"Loading BiLSTM model from {BILSTM_MODEL_PATH} …")
        _model    = tf.keras.models.load_model(BILSTM_MODEL_PATH)
        _scaler_x = joblib.load(SCALER_X_PATH)
        _scaler_y = joblib.load(SCALER_Y_PATH)
        logger.info("BiLSTM model and scalers loaded successfully.")
        return True

    except Exception as exc:
        logger.error(f"Failed to load model artifacts: {exc}")
        return False


def model_is_ready() -> bool:
    """Return True if the model file exists and can be loaded."""
    return _load_artifacts()


# ---------------------------------------------------------------------------
# Vital clamping
# ---------------------------------------------------------------------------
_CLAMP: dict = {
    "RR":          (6,   45),
    "SpO2":        (70,  100),
    "HR":          (35,  220),
    "SBP":         (55,  230),
    "Temperature": (34.0, 42.0),
    "AVPU":        (0,   3),
}


def _clamp_vitals(vitals: dict) -> dict:
    """Clamp predicted vitals to physiologically plausible ranges."""
    out: dict = {}
    for key, val in vitals.items():
        lo, hi = _clamp.get(key, (-1e9, 1e9)) if False else _CLAMP.get(key, (val, val))
        out[key] = float(np.clip(val, lo, hi))
    return out


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def predict_from_sequence(sequence: list[list[float]]) -> dict:
    """
    Run inference on an 8-step vital sequence.

    Args:
        sequence: List of 8 time-steps, each step is a list of 6 floats
                  in order [RR, SpO2, HR, SBP, Temperature, AVPU].

    Returns:
        Dict with keys:
            current_vitals  : dict of last input step's vitals
            current_rrt     : {score, components, category, label}
            t4_vitals       : predicted vitals at +4h
            t4_rrt          : {score, components, category, label}
            t8_vitals       : predicted vitals at +8h
            t8_rrt          : {score, components, category, label}
            model_available : bool

    Raises:
        ValueError: If sequence shape is incorrect.
    """
    # Validate
    if len(sequence) != SEQUENCE_LENGTH:
        raise ValueError(
            f"sequence must have {SEQUENCE_LENGTH} time-steps, got {len(sequence)}"
        )
    for i, step in enumerate(sequence):
        if len(step) != N_FEATURES:
            raise ValueError(
                f"step[{i}] must have {N_FEATURES} features, got {len(step)}"
            )

    X = np.array(sequence, dtype=np.float32)   # (8, 6)

    # Current vitals from last step
    current_vitals: dict = {
        feat: float(X[-1, j]) for j, feat in enumerate(FEATURE_COLS)
    }
    current_avpu_str = AVPU_DECODING.get(
        int(round(current_vitals.get("AVPU", 0))), "Alert"
    )

    curr_score, curr_comp, curr_cat, curr_label = calculate_rrt_score(
        rr=current_vitals["RR"],
        spo2=current_vitals["SpO2"],
        hr=current_vitals["HR"],
        sbp=current_vitals["SBP"],
        temperature=current_vitals["Temperature"],
        avpu=current_avpu_str,
    )

    # Attempt model inference
    loaded = _load_artifacts()

    if loaded:
        N, T, F = 1, SEQUENCE_LENGTH, N_FEATURES
        X_flat   = X.reshape(1, T * F)
        X_scaled = _scaler_x.transform(X_flat).reshape(1, T, F)
        y_scaled = _model.predict(X_scaled, verbose=0)         # (1, 12)
        y_pred   = _scaler_y.inverse_transform(y_scaled)[0]   # (12,)

        raw: dict = {t: float(y_pred[i]) for i, t in enumerate(TARGET_COLS)}
    else:
        # Fallback: carry current vitals forward with slight perturbation
        rng = np.random.default_rng(42)
        raw = {
            "RR_T4":   current_vitals["RR"]   + rng.normal(0, 1.0),
            "SpO2_T4": current_vitals["SpO2"] + rng.normal(0, 0.5),
            "HR_T4":   current_vitals["HR"]   + rng.normal(0, 2.0),
            "SBP_T4":  current_vitals["SBP"]  + rng.normal(0, 3.0),
            "Temp_T4": current_vitals["Temperature"] + rng.normal(0, 0.1),
            "AVPU_T4": current_vitals["AVPU"],
            "RR_T8":   current_vitals["RR"]   + rng.normal(0, 2.0),
            "SpO2_T8": current_vitals["SpO2"] + rng.normal(0, 1.0),
            "HR_T8":   current_vitals["HR"]   + rng.normal(0, 4.0),
            "SBP_T8":  current_vitals["SBP"]  + rng.normal(0, 5.0),
            "Temp_T8": current_vitals["Temperature"] + rng.normal(0, 0.2),
            "AVPU_T8": current_vitals["AVPU"],
        }

    # T+4 vitals
    t4_vitals: dict = {
        "RR":          float(np.clip(raw["RR_T4"],   6,   45)),
        "SpO2":        float(np.clip(raw["SpO2_T4"], 70,  100)),
        "HR":          float(np.clip(raw["HR_T4"],   35,  220)),
        "SBP":         float(np.clip(raw["SBP_T4"],  55,  230)),
        "Temperature": float(np.clip(raw["Temp_T4"], 34.0, 42.0)),
        "AVPU":        float(np.clip(round(raw["AVPU_T4"]), 0, 3)),
    }
    t4_avpu_str = AVPU_DECODING.get(int(t4_vitals["AVPU"]), "Alert")
    t4_score, t4_comp, t4_cat, t4_label = calculate_rrt_score(
        rr=t4_vitals["RR"], spo2=t4_vitals["SpO2"], hr=t4_vitals["HR"],
        sbp=t4_vitals["SBP"], temperature=t4_vitals["Temperature"],
        avpu=t4_avpu_str,
    )

    # T+8 vitals
    t8_vitals: dict = {
        "RR":          float(np.clip(raw["RR_T8"],   6,   45)),
        "SpO2":        float(np.clip(raw["SpO2_T8"], 70,  100)),
        "HR":          float(np.clip(raw["HR_T8"],   35,  220)),
        "SBP":         float(np.clip(raw["SBP_T8"],  55,  230)),
        "Temperature": float(np.clip(raw["Temp_T8"], 34.0, 42.0)),
        "AVPU":        float(np.clip(round(raw["AVPU_T8"]), 0, 3)),
    }
    t8_avpu_str = AVPU_DECODING.get(int(t8_vitals["AVPU"]), "Alert")
    t8_score, t8_comp, t8_cat, t8_label = calculate_rrt_score(
        rr=t8_vitals["RR"], spo2=t8_vitals["SpO2"], hr=t8_vitals["HR"],
        sbp=t8_vitals["SBP"], temperature=t8_vitals["Temperature"],
        avpu=t8_avpu_str,
    )

    return {
        "current_vitals": {k: round(v, 2) for k, v in current_vitals.items()},
        "current_rrt": {
            "score": curr_score,
            "components": curr_comp,
            "category": curr_cat,
            "label": curr_label,
        },
        "t4_vitals": {k: round(v, 2) for k, v in t4_vitals.items()},
        "t4_rrt": {
            "score": t4_score,
            "components": t4_comp,
            "category": t4_cat,
            "label": t4_label,
        },
        "t8_vitals": {k: round(v, 2) for k, v in t8_vitals.items()},
        "t8_rrt": {
            "score": t8_score,
            "components": t8_comp,
            "category": t8_cat,
            "label": t8_label,
        },
        "model_available": loaded,
    }


def predict_from_current_vitals(vitals: dict) -> dict:
    """
    Build a synthetic 8-step history from a single set of current vitals
    and run prediction.

    Useful for new patients with no prior history in vital_history.csv.

    Args:
        vitals: Dict with keys RR, SpO2, HR, SBP, Temperature, AVPU.
                AVPU may be a string or numeric encoding.

    Returns:
        Same structure as predict_from_sequence().
    """
    rng = np.random.default_rng(seed=0)

    avpu_raw = vitals.get("AVPU", 0)
    from config import AVPU_ENCODING
    avpu_num = (
        AVPU_ENCODING.get(str(avpu_raw).strip().capitalize(), 0)
        if isinstance(avpu_raw, str) else int(avpu_raw)
    )

    current_vals = np.array(
        [
            float(vitals["RR"]),
            float(vitals["SpO2"]),
            float(vitals["HR"]),
            float(vitals["SBP"]),
            float(vitals["Temperature"]),
            float(avpu_num),
        ],
        dtype=np.float32,
    )

    noise_scale = np.array([0.4, 0.15, 1.0, 1.5, 0.05, 0.0], dtype=np.float32)
    clip_lo = np.array([6, 70, 35, 55, 34.0, 0], dtype=np.float32)
    clip_hi = np.array([45, 100, 220, 230, 42.0, 3], dtype=np.float32)

    sequence: list[list[float]] = []
    for step in range(SEQUENCE_LENGTH):
        noise  = rng.normal(0, noise_scale)
        factor = (SEQUENCE_LENGTH - step) * 0.15
        past   = np.clip(current_vals + noise * factor, clip_lo, clip_hi)
        sequence.append(past.tolist())

    return predict_from_sequence(sequence)


def predict_from_history_df(history_df, n_steps: int = SEQUENCE_LENGTH) -> dict | None:
    """
    Run prediction from a patient's actual vital history DataFrame.

    Selects the last n_steps rows from history_df as the input sequence.

    Args:
        history_df: DataFrame with columns matching FEATURE_COLS
                    (heart_rate → HR, respiratory_rate → RR, etc.).
                    Must contain at least n_steps rows.
        n_steps:    Number of steps to use (default = SEQUENCE_LENGTH = 8).

    Returns:
        Prediction dict or None if insufficient history.
    """
    col_map = {
        "respiratory_rate": "RR",
        "spo2":             "SpO2",
        "heart_rate":       "HR",
        "systolic_bp":      "SBP",
        "temperature_f":    "Temperature",
        "avpu_encoded":     "AVPU",
        # identity mappings (in case already named correctly)
        "RR":          "RR",
        "SpO2":        "SpO2",
        "HR":          "HR",
        "SBP":         "SBP",
        "Temperature": "Temperature",
        "AVPU":        "AVPU",
    }

    if len(history_df) < n_steps:
        return None

    tail = history_df.tail(n_steps).copy()
    sequence: list[list[float]] = []

    for _, row in tail.iterrows():
        step: list[float] = []
        for feat in FEATURE_COLS:
            # Try direct name first, then mapped names
            val = None
            for src_col, dst_feat in col_map.items():
                if dst_feat == feat and src_col in row.index:
                    val = float(row[src_col])
                    break
            step.append(val if val is not None else 0.0)
        sequence.append(step)

    return predict_from_sequence(sequence)
