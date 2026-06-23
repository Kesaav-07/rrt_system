"""
realtime.py
-----------
Real-time patient monitoring module for the RRT BiLSTM Surveillance System.

Responsibilities:
    - Simulate bedside monitoring via vectorized vital sign perturbation.
    - Append records to vital_history.csv (append-only log).
    - Generate BiLSTM predictions for each patient's 4h and 8h RRT scores.
    - Update live_future_records.csv with current and predicted scores.
    - Maintain realtime_state.json for connection status tracking.

Key design decisions:
    - Vectorized operations (no per-patient Python loops for performance).
    - BiLSTM uses actual history when ≥8 readings exist; current vitals otherwise.
    - All RRT scoring done exclusively through ai/rrt_calculator.py.
    - No Random Forest model anywhere in this codebase.
"""

from __future__ import annotations

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR,
    LIVE_RECORDS_FILE,
    VITAL_HISTORY_FILE,
    REALTIME_STATE_FILE,
    AVPU_ENCODING,
    AVPU_DECODING,
    AVPU_OPTIONS,
    REFRESH_INTERVAL_SECONDS,
    RECONNECTING_MULTIPLIER,
    OFFLINE_MULTIPLIER,
    SEQUENCE_LENGTH,
    FEATURE_COLS,
)
from ai.rrt_calculator import calculate_rrt_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection status labels
# ---------------------------------------------------------------------------
STATUS_DISPLAY: dict[str, tuple[str, str]] = {
    "live":         ("🟢 Live",                              "#2E8B57"),
    "reconnecting": ("🟠 Reconnecting…",                     "#E8A33D"),
    "offline":      ("🔴 Offline — showing last known data", "#D7263D"),
    "unknown":      ("⚪ Waiting for first update…",         "#5A7184"),
}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat(timespec="seconds")


def _load_state() -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(REALTIME_STATE_FILE):
        return {"sequence": 0, "last_tick_at": None}
    try:
        with open(REALTIME_STATE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"sequence": 0, "last_tick_at": None}


def _save_state(state):
    tmp_path = REALTIME_STATE_FILE + f".{os.getpid()}.tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(state, f)

        os.replace(tmp_path, REALTIME_STATE_FILE)
    except PermissionError:
        logger.warning("[realtime] State file locked, skipping this tick safely.")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except PermissionError:
                pass


def connection_status(interval_seconds: int = REFRESH_INTERVAL_SECONDS) -> tuple[str, datetime | None]:
    """
    Return (status, last_tick_datetime) based on how long since the last tick.

    Args:
        interval_seconds: Expected refresh interval in seconds.

    Returns:
        Tuple of status string and last tick datetime (or None).
    """
    state = _load_state()
    if not state.get("last_tick_at"):
        return "unknown", None

    last_tick_dt = datetime.fromisoformat(state["last_tick_at"])
    seconds_since = (_now_utc() - last_tick_dt).total_seconds()

    if seconds_since <= interval_seconds * RECONNECTING_MULTIPLIER:
        return "live", last_tick_dt
    elif seconds_since <= interval_seconds * OFFLINE_MULTIPLIER:
        return "reconnecting", last_tick_dt
    else:
        return "offline", last_tick_dt


# ---------------------------------------------------------------------------
# Vectorized vital sign perturbation
# ---------------------------------------------------------------------------

_VITAL_DEFAULTS: dict = {
    "heart_rate":       80,
    "respiratory_rate": 16,
    "spo2":             97,
    "systolic_bp":      115,
    "temperature":      36.8,
    "avpu_encoded":     0,
}

_VITAL_CLIPS: dict = {
    "heart_rate":       (35, 220),
    "respiratory_rate": (6,  45),
    "spo2":             (70, 100),
    "systolic_bp":      (55, 230),
    "temperature":      (34.0, 42.0),
}


def _perturb_vitals_vectorized(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Apply realistic random-walk perturbations to all patients simultaneously.

    Fills NaN values with physiological defaults before perturbation.
    Handles AVPU and oxygen_support state transitions stochastically.

    Args:
        df:  Live patient records DataFrame.
        rng: NumPy random generator instance.

    Returns:
        New DataFrame with updated vitals.
    """
    n = len(df)
    out = df.copy()

    # Fill any NaNs with physiological defaults
    for col, default in _VITAL_DEFAULTS.items():
        if col in out.columns:
            out[col] = out[col].fillna(default)

    # Validate AVPU
    if "avpu" in out.columns:
        invalid_avpu = ~out["avpu"].isin(AVPU_OPTIONS)
        if invalid_avpu.any():
            out.loc[invalid_avpu, "avpu"] = "Alert"

    # Heart rate
    out["heart_rate"] = np.clip(
        out["heart_rate"] + rng.normal(0, 2.5, n), 35, 220
    ).round().astype(int)

    # Respiratory rate
    out["respiratory_rate"] = np.clip(
        out["respiratory_rate"] + rng.normal(0, 1.0, n), 6, 45
    ).round().astype(int)

    # SpO2
    out["spo2"] = np.clip(
        out["spo2"] + rng.normal(0, 0.6, n), 70, 100
    ).round().astype(int)

    # Systolic BP
    out["systolic_bp"] = np.clip(
        out["systolic_bp"] + rng.normal(0, 2.5, n), 55, 230
    ).round().astype(int)

    # Temperature (°C)
    out["temperature"] = np.clip(
        out["temperature"] + rng.normal(0, 0.08, n), 34.0, 42.0
    ).round(1)

    # AVPU random state transitions (3% chance per tick)
    avpu_flip = rng.random(n) < 0.03
    if avpu_flip.any():
        new_avpu = rng.choice(
            AVPU_OPTIONS,
            size=int(avpu_flip.sum()),
            p=[0.70, 0.18, 0.09, 0.03],
        )
        out.loc[avpu_flip, "avpu"] = new_avpu

    return out


# ---------------------------------------------------------------------------
# RRT score calculation (18-point, using rrt_calculator exclusively)
# ---------------------------------------------------------------------------

def _compute_rrt_scores(df: pd.DataFrame) -> pd.Series:
    """
    Compute current 18-point RRT score for every patient row.

    Uses ai.rrt_calculator.calculate_rrt_score exclusively.

    Args:
        df: DataFrame with vital columns.

    Returns:
        Series of RRT scores.
    """
    scores = []
    for _, row in df.iterrows():
        avpu_val = str(row.get("avpu", "Alert"))
        score, _, _, _ = calculate_rrt_score(
            rr=float(row.get("respiratory_rate", 16)),
            spo2=float(row.get("spo2", 97)),
            hr=float(row.get("heart_rate", 80)),
            sbp=float(row.get("systolic_bp", 115)),
            temperature=float(row.get("temperature", 36.8)),
            avpu=avpu_val,
        )
        scores.append(score)
    return pd.Series(scores, index=df.index)


# ---------------------------------------------------------------------------
# BiLSTM-based future RRT prediction
# ---------------------------------------------------------------------------

def _predict_future_rrt_bilstm(
    df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Predict 4h and 8h RRT scores for all patients using the Bi-LSTM model.

    For patients with ≥8 history records: uses actual history sequence.
    For others: uses current vitals to build a synthetic sequence.

    Args:
        df:         Current live records.
        history_df: Full vital_history.csv contents (may be empty).

    Returns:
        Tuple of (predicted_rrt_4hr Series, predicted_rrt_8hr Series).
    """
    try:
        from ai.predict_bilstm import predict_from_current_vitals, predict_from_history_df

        scores_4h: list[int] = []
        scores_8h: list[int] = []

        for _, row in df.iterrows():
            pred = None

            # Try to use history if available
            if not history_df.empty:
                pid = str(row.get("patient_id", ""))
                pat_hist = history_df[
                    history_df["patient_id"].astype(str) == pid
                ].copy()

                if len(pat_hist) >= SEQUENCE_LENGTH:
                    # Map history columns to feature names
                    pat_hist = pat_hist.rename(columns={
                        "respiratory_rate": "RR",
                        "spo2":             "SpO2",
                        "heart_rate":       "HR",
                        "systolic_bp":      "SBP",
                        "temperature":      "Temperature",
                        "avpu_encoded":     "AVPU",
                    })
                    # Add AVPU encoded if missing
                    if "AVPU" not in pat_hist.columns and "avpu" in pat_hist.columns:
                        pat_hist["AVPU"] = pat_hist["avpu"].map(AVPU_ENCODING).fillna(0)
                    pred = predict_from_history_df(pat_hist)

            # Fallback to current vitals
            if pred is None:
                avpu_enc = AVPU_ENCODING.get(str(row.get("avpu", "Alert")), 0)
                vitals = {
                    "RR":          float(row.get("respiratory_rate", 16)),
                    "SpO2":        float(row.get("spo2", 97)),
                    "HR":          float(row.get("heart_rate", 80)),
                    "SBP":         float(row.get("systolic_bp", 115)),
                    "Temperature": float(row.get("temperature", 36.8)),
                    "AVPU":        avpu_enc,
                }
                pred = predict_from_current_vitals(vitals)

            scores_4h.append(int(pred["t4_rrt"]["score"]))
            scores_8h.append(int(pred["t8_rrt"]["score"]))

        return pd.Series(scores_4h, index=df.index), pd.Series(scores_8h, index=df.index)

    except Exception as exc:
        logger.warning(f"BiLSTM prediction failed: {exc}. Using fallback scores.")
        # Fallback: current score + small increment
        current = _compute_rrt_scores(df)
        fallback_4h = np.clip(current + 1, 0, 18).astype(int)
        fallback_8h = np.clip(current + 2, 0, 18).astype(int)
        return (
            pd.Series(fallback_4h, index=df.index),
            pd.Series(fallback_8h, index=df.index),
        )


# ---------------------------------------------------------------------------
# Main tick function
# ---------------------------------------------------------------------------

def run_synthetic_tick_if_due(
    df_patients: pd.DataFrame,
    interval_seconds: int = REFRESH_INTERVAL_SECONDS,
) -> pd.DataFrame:
    """
    Run a synthetic monitoring tick if the interval has elapsed.

    Actions per tick:
        1. Perturb all patient vitals (vectorized random walk).
        2. Recompute 18-point RRT scores.
        3. Run Bi-LSTM to predict 4h and 8h future RRT.
        4. Append current readings to vital_history.csv.
        5. Save updated records to live_future_records.csv.
        6. Update realtime_state.json with new sequence number and tick time.

    Args:
        df_patients:      Current patient records DataFrame.
        interval_seconds: How many seconds between ticks.

    Returns:
        Updated patient records DataFrame.
    """
    if df_patients.empty:
        return df_patients

    state = _load_state()
    last_tick = state.get("last_tick_at")

    due = (last_tick is None) or (
        (_now_utc() - datetime.fromisoformat(last_tick)).total_seconds() >= interval_seconds
    )

    if not due:
        return df_patients

    os.makedirs(DATA_DIR, exist_ok=True)

    rng = np.random.default_rng()
    updated = df_patients.copy()

    # Ensure required columns exist
    if "last_recorded_at" not in updated.columns:
        updated["last_recorded_at"] = None
    if "sequence" not in updated.columns:
        updated["sequence"] = 0
    if "avpu" not in updated.columns:
        updated["avpu"] = "Alert"
    if "temperature" not in updated.columns:
        # Accept either 'temperature' or 'temperature_c'
        if "temperature_c" in updated.columns:
            updated["temperature"] = updated["temperature_c"]
        else:
            updated["temperature"] = 36.8

    # Step 1: Perturb vitals
    updated = _perturb_vitals_vectorized(updated, rng)

    # Step 2: Compute current 18-point RRT scores
    updated["current_rrt_score"] = _compute_rrt_scores(updated)

    # Step 3: Add RRT category
    from ai.rrt_calculator import rrt_category_from_score
    updated["rrt_category"] = updated["current_rrt_score"].apply(
        lambda s: rrt_category_from_score(int(s))[1]
    )

    # Step 4: BiLSTM predictions
    history_df = load_all_history()
    pred_4h, pred_8h = _predict_future_rrt_bilstm(updated, history_df)
    updated["predicted_rrt_4hr"] = pred_4h.values
    updated["predicted_rrt_8hr"] = pred_8h.values

    # Step 5: Update timestamps and sequences
    recorded_at = _now_iso()
    start_seq   = int(state.get("sequence", 0))
    n           = len(updated)
    sequences   = np.arange(start_seq + 1, start_seq + 1 + n)

    updated["last_recorded_at"] = recorded_at
    updated["sequence"]         = sequences
    end_seq = int(sequences[-1]) if n else start_seq

    # Step 6: Save live records
    updated.to_csv(LIVE_RECORDS_FILE, index=False)

    # Step 7: Append to history
    avpu_encoded = updated["avpu"].map(AVPU_ENCODING).fillna(0).astype(int)
    history_chunk = pd.DataFrame({
        "patient_id":       updated["patient_id"].values,
        "recorded_at":      recorded_at,
        "sequence":         sequences,
        "heart_rate":       updated["heart_rate"].values,
        "respiratory_rate": updated["respiratory_rate"].values,
        "spo2":             updated["spo2"].values,
        "systolic_bp":      updated["systolic_bp"].values,
        "temperature":      updated["temperature"].values,
        "avpu":             updated["avpu"].values,
        "avpu_encoded":     avpu_encoded.values,
        "current_rrt_score": updated["current_rrt_score"].values,
    })

    write_header = not os.path.exists(VITAL_HISTORY_FILE)
    history_chunk.to_csv(VITAL_HISTORY_FILE, mode="a", header=write_header, index=False)

    # Step 8: Update state
    _save_state({"sequence": end_seq, "last_tick_at": recorded_at})

    logger.info(
        f"[realtime] Tick complete — {n} patients updated, seq {end_seq}, "
        f"critical: {(updated['current_rrt_score'] >= 12).sum()}"
    )
    return updated


# ---------------------------------------------------------------------------
# History loader
# ---------------------------------------------------------------------------

def load_patient_history(
    patient_id: str,
    hours_back: int = 8,
    chunksize: int = 20000,
) -> pd.DataFrame:
    """
    Load vital history for a specific patient from vital_history.csv.

    Args:
        patient_id: Patient ID string (e.g. "P7001").
        hours_back: How many hours back to retrieve.
        chunksize:  Chunk size for reading large files.

    Returns:
        DataFrame sorted by recorded_at, or empty DataFrame if none found.
    """
    empty_cols = [
        "recorded_at", "heart_rate", "respiratory_rate",
        "spo2", "systolic_bp", "temperature",
        "avpu", "avpu_encoded", "current_rrt_score",
    ]
    if not os.path.exists(VITAL_HISTORY_FILE):
        return pd.DataFrame(columns=empty_cols)

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours_back)
    pid    = str(patient_id)
    chunks: list[pd.DataFrame] = []

    try:
        for chunk in pd.read_csv(VITAL_HISTORY_FILE, chunksize=chunksize):
            sub = chunk[chunk["patient_id"].astype(str) == pid]
            if sub.empty:
                continue
            sub = sub.copy()
            sub["recorded_at"] = pd.to_datetime(sub["recorded_at"], utc=True, errors="coerce")
            sub = sub.dropna(subset=["recorded_at"])
            sub = sub[sub["recorded_at"] >= cutoff]
            if not sub.empty:
                chunks.append(sub)

    except Exception as exc:
        logger.warning(f"Failed reading vital history for {patient_id}: {exc}")
        return pd.DataFrame(columns=empty_cols)

    if not chunks:
        return pd.DataFrame(columns=empty_cols)

    hist = pd.concat(chunks, ignore_index=True)
    if "sequence" in hist.columns:
        hist = hist.drop_duplicates(subset="sequence", keep="last")
    hist = hist.sort_values("recorded_at").reset_index(drop=True)
    return hist


def load_all_history(chunksize: int = 50000) -> pd.DataFrame:
    """
    Load the entire vital_history.csv (for batch BiLSTM prediction).

    Args:
        chunksize: Read chunk size.

    Returns:
        Full history DataFrame or empty DataFrame.
    """
    if not os.path.exists(VITAL_HISTORY_FILE):
        return pd.DataFrame()

    try:
        chunks = []
        for chunk in pd.read_csv(VITAL_HISTORY_FILE, chunksize=chunksize):
            chunks.append(chunk)
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)
    except Exception as exc:
        logger.warning(f"Failed loading vital history: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Alert detection
# ---------------------------------------------------------------------------

def get_alerts(df_patients: pd.DataFrame) -> pd.DataFrame:
    """
    Return patients who should trigger RRT alerts.

    Trigger conditions:
        - current_rrt_score >= 12, OR
        - predicted_rrt_8hr >= 12

    Args:
        df_patients: Live patient records.

    Returns:
        Subset of df_patients requiring alerts.
    """
    if df_patients.empty:
        return pd.DataFrame()

    mask = (
        (df_patients["current_rrt_score"].fillna(0).astype(int) >= 12) |
        (df_patients.get("predicted_rrt_8hr", pd.Series(dtype=int)).fillna(0).astype(int) >= 12)
    )
    return df_patients[mask].copy()
