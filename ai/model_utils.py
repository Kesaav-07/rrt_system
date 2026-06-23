"""
ai/model_utils.py
-----------------
Utility helpers for BiLSTM model management.
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)


def model_exists() -> bool:
    """Return True if the trained model file exists on disk."""
    from config import BILSTM_MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH
    return (
        os.path.exists(BILSTM_MODEL_PATH)
        and os.path.exists(SCALER_X_PATH)
        and os.path.exists(SCALER_Y_PATH)
    )


def model_size_mb() -> float | None:
    """Return the size of the saved model in MB, or None if not found."""
    from config import BILSTM_MODEL_PATH
    if not os.path.exists(BILSTM_MODEL_PATH):
        return None
    size_bytes = os.path.getsize(BILSTM_MODEL_PATH)
    return round(size_bytes / (1024 * 1024), 2)


def delete_model_artifacts() -> None:
    """Delete all saved model artifacts (for retraining from scratch)."""
    from config import BILSTM_MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH
    for path in (BILSTM_MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH):
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted: {path}")
