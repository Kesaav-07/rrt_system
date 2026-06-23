"""
utils/validators.py
-------------------
Input validation helpers for vital signs and patient data.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

VITAL_BOUNDS: dict[str, tuple[float, float]] = {
    "RR":          (4.0,  70.0),
    "SpO2":        (50.0, 100.0),
    "HR":          (20.0, 300.0),
    "SBP":         (40.0, 300.0),
    "Temperature": (30.0, 43.5),
    "AVPU":        (0.0,  3.0),
}


def validate_vitals(vitals: dict) -> tuple[bool, list[str]]:
    """
    Validate a vitals dict against physiological bounds.

    Args:
        vitals: Dict with vital keys and numeric values.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: list[str] = []
    for key, (lo, hi) in VITAL_BOUNDS.items():
        if key not in vitals:
            continue
        try:
            val = float(vitals[key])
        except (TypeError, ValueError):
            errors.append(f"{key}: non-numeric value '{vitals[key]}'")
            continue
        if not (lo <= val <= hi):
            errors.append(f"{key}={val} is outside physiological range [{lo}, {hi}]")

    return len(errors) == 0, errors


def validate_patient_id(patient_id: str) -> bool:
    """Return True if patient_id is a non-empty string."""
    return bool(patient_id and str(patient_id).strip())
