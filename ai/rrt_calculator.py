"""
ai/rrt_calculator.py
--------------------
Official 18-point RRT (Rapid Response Team) Score Calculator.

SINGLE SOURCE OF TRUTH for all RRT scoring logic.
No other module may duplicate or re-implement these scoring functions.

Parameters (6 total, 0-3 points each, max = 18):
    1. Respiratory Rate (RR)
    2. SpO2
    3. Heart Rate (HR)
    4. Systolic Blood Pressure (SBP)
    5. Temperature
    6. AVPU

Risk Categories:
    Stable  : 0–5
    Warning : 6–11
    Critical: 12–18
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RRT_STABLE_MAX,
    RRT_WARNING_MAX,
    RRT_CRITICAL_MIN,
    RRT_MAX_SCORE,
    AVPU_ENCODING,
)


# ---------------------------------------------------------------------------
# Individual parameter scoring functions (0–3 each)
# ---------------------------------------------------------------------------

def score_rr(rr: float) -> int:
    """
    Score respiratory rate (breaths/min).

    Args:
        rr: Respiratory rate value.

    Returns:
        Integer score 0–3.
    """
    rr = float(rr)
    if rr < 8:
        return 3
    elif rr <= 11:
        return 2
    elif rr <= 14:
        return 1
    elif rr <= 20:
        return 0
    elif rr <= 24:
        return 1
    elif rr <= 29:
        return 2
    else:
        return 3


def score_spo2(spo2: float) -> int:
    """
    Score oxygen saturation (%).

    Args:
        spo2: SpO2 value.

    Returns:
        Integer score 0–3.
    """
    spo2 = float(spo2)
    if spo2 < 85:
        return 3
    elif spo2 < 90:
        return 2
    elif spo2 < 94:
        return 1
    else:
        return 0


def score_hr(hr: float) -> int:
    """
    Score heart rate (bpm).

    Args:
        hr: Heart rate value.

    Returns:
        Integer score 0–3.
    """
    hr = float(hr)
    if hr < 40:
        return 3
    elif hr < 50:
        return 2
    elif hr < 60:
        return 1
    elif hr <= 100:
        return 0
    elif hr <= 110:
        return 1
    elif hr <= 130:
        return 2
    else:
        return 3


def score_sbp(sbp: float) -> int:
    """
    Score systolic blood pressure (mmHg).

    Args:
        sbp: Systolic BP value.

    Returns:
        Integer score 0–3.
    """
    sbp = float(sbp)
    if sbp < 70:
        return 3
    elif sbp < 80:
        return 2
    elif sbp < 90:
        return 1
    elif sbp <= 140:
        return 0
    elif sbp <= 160:
        return 1
    elif sbp <= 180:
        return 2
    else:
        return 3


def score_temperature(temp: float) -> int:
    """
    Score body temperature (°C).

    Args:
        temp: Temperature value in Celsius.

    Returns:
        Integer score 0–3.
    """
    temp = float(temp)
    if temp < 35.0:
        return 3
    elif temp < 36.0:
        return 1
    elif temp <= 37.5:
        return 0
    elif temp <= 38.5:
        return 1
    elif temp <= 39.5:
        return 2
    else:
        return 3


def score_avpu(avpu: str | int | float) -> int:
    """
    Score AVPU (Alert / Voice / Pain / Unresponsive) consciousness level.

    Accepts both string labels and numeric encodings (0–3).

    Args:
        avpu: AVPU status as string or numeric encoding.

    Returns:
        Integer score 0–3.
    """
    if isinstance(avpu, (int, float)):
        v = int(round(float(avpu)))
    else:
        v = AVPU_ENCODING.get(str(avpu).strip().capitalize(), 0)

    if v == 0:      # Alert
        return 0
    elif v == 1:    # Voice
        return 1
    elif v == 2:    # Pain
        return 2
    else:           # Unresponsive
        return 3


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------

def calculate_rrt_score(
    rr: float,
    spo2: float,
    hr: float,
    sbp: float,
    temperature: float,
    avpu: str | int | float = "Alert",
) -> tuple[int, dict[str, int], str, str]:
    """
    Calculate the 18-point RRT score from six vital parameters.

    Args:
        rr:          Respiratory rate (breaths/min).
        spo2:        Oxygen saturation (%).
        hr:          Heart rate (bpm).
        sbp:         Systolic blood pressure (mmHg).
        temperature: Body temperature (°C).
        avpu:        AVPU level — string ("Alert", "Voice", "Pain",
                     "Unresponsive") or numeric encoding (0–3).

    Returns:
        Tuple of:
            total_score  (int)           : 0–18
            components   (dict[str,int]) : per-parameter sub-scores
            category     (str)           : "stable" | "warning" | "critical"
            label        (str)           : human-readable category label
    """
    components: dict[str, int] = {
        "RR":          score_rr(rr),
        "SpO2":        score_spo2(spo2),
        "HR":          score_hr(hr),
        "SBP":         score_sbp(sbp),
        "Temperature": score_temperature(temperature),
        "AVPU":        score_avpu(avpu),
    }

    total: int = min(sum(components.values()), RRT_MAX_SCORE)

    if total <= RRT_STABLE_MAX:
        category, label = "stable", "Stable"
    elif total <= RRT_WARNING_MAX:
        category, label = "warning", "Warning"
    else:
        category, label = "critical", "Critical"

    return total, components, category, label


def calculate_rrt_from_dict(vitals: dict) -> tuple[int, dict[str, int], str, str]:
    """
    Convenience wrapper: compute RRT score from a dict with keys
    RR, SpO2, HR, SBP, Temperature, and optionally AVPU.

    Args:
        vitals: Dict containing vital sign values.

    Returns:
        Same 4-tuple as calculate_rrt_score().
    """
    return calculate_rrt_score(
        rr=float(vitals["RR"]),
        spo2=float(vitals["SpO2"]),
        hr=float(vitals["HR"]),
        sbp=float(vitals["SBP"]),
        temperature=float(vitals["Temperature"]),
        avpu=vitals.get("AVPU", "Alert"),
    )


def rrt_category_from_score(score: int) -> tuple[str, str]:
    """
    Return (category, label) from a raw score.

    Args:
        score: RRT total score (0–18).

    Returns:
        Tuple (category, label).
    """
    if score <= RRT_STABLE_MAX:
        return "stable", "Stable"
    elif score <= RRT_WARNING_MAX:
        return "warning", "Warning"
    else:
        return "critical", "Critical"
