"""
config.py
---------
Central configuration for the RRT BiLSTM Surveillance System.
Single source of truth for all constants, paths, and thresholds.
"""

from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# Directory & file paths
# ---------------------------------------------------------------------------
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
DATA_DIR            = os.path.join(BASE_DIR, "data")
TRAINED_DIR         = os.path.join(DATA_DIR, "trained")

LIVE_RECORDS_FILE   = os.path.join(DATA_DIR, "live_future_records.csv")
VITAL_HISTORY_FILE  = os.path.join(DATA_DIR, "vital_history.csv")
REALTIME_STATE_FILE = os.path.join(DATA_DIR, "realtime_state.json")
USERS_FILE          = os.path.join(DATA_DIR, "users.csv")

BILSTM_MODEL_PATH   = os.path.join(TRAINED_DIR, "bilstm_model.keras")
SCALER_X_PATH       = os.path.join(TRAINED_DIR, "scaler_x.joblib")
SCALER_Y_PATH       = os.path.join(TRAINED_DIR, "scaler_y.joblib")

# ---------------------------------------------------------------------------
# RRT score thresholds
# ---------------------------------------------------------------------------
RRT_STABLE_MAX   = 5
RRT_WARNING_MAX  = 11
RRT_CRITICAL_MIN = 12
RRT_MAX_SCORE    = 18

# ---------------------------------------------------------------------------
# BiLSTM model parameters
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 8
N_FEATURES      = 6

FEATURE_COLS: list[str] = ["RR", "SpO2", "HR", "SBP", "Temperature", "AVPU"]
TARGET_COLS:  list[str] = [
    "RR_T4", "SpO2_T4", "HR_T4", "SBP_T4", "Temp_T4", "AVPU_T4",
    "RR_T8", "SpO2_T8", "HR_T8", "SBP_T8", "Temp_T8", "AVPU_T8",
]

# ---------------------------------------------------------------------------
# AVPU encoding/decoding
# ---------------------------------------------------------------------------
AVPU_OPTIONS: list[str] = ["Alert", "Voice", "Pain", "Unresponsive"]

AVPU_ENCODING: dict[str, int] = {
    "Alert":       0,
    "Voice":       1,
    "Pain":        2,
    "Unresponsive": 3,
}

AVPU_DECODING: dict[int, str] = {v: k for k, v in AVPU_ENCODING.items()}

# ---------------------------------------------------------------------------
# Physiological ranges (for clamping / validation)
# ---------------------------------------------------------------------------
VITAL_RANGES: dict[str, tuple[float, float]] = {
    "RR":          (6.0,  45.0),
    "SpO2":        (70.0, 100.0),
    "HR":          (35.0, 220.0),
    "SBP":         (55.0, 230.0),
    "Temperature": (34.0, 42.0),
    "AVPU":        (0.0,  3.0),
}

# ---------------------------------------------------------------------------
# Auth / roles
# ---------------------------------------------------------------------------
ROLES: list[str] = ["Nurse", "Physician", "RRT Team", "Admin"]

# ---------------------------------------------------------------------------
# Realtime engine
# ---------------------------------------------------------------------------
REFRESH_INTERVAL_SECONDS  = 60
RECONNECTING_MULTIPLIER   = 3    # >3× interval → reconnecting
OFFLINE_MULTIPLIER        = 6    # >6× interval → offline

# ---------------------------------------------------------------------------
# Patient ID generation
# ---------------------------------------------------------------------------
PATIENT_ID_PREFIX = "P"
PATIENT_ID_START  = 1001

# ---------------------------------------------------------------------------
# Overdue-reading thresholds (minutes) per ward
# ---------------------------------------------------------------------------
OVERDUE_MINUTES_BY_WARD: dict[str, int] = {
    "ICU":     30,
    "Special": 60,
    "General": 120,
}
