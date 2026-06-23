"""
ai/explainability.py
--------------------
Lightweight explainability utilities for the BiLSTM RRT predictions.

Provides permutation-based feature importance and per-parameter contribution
summaries without requiring SHAP (which conflicts with TF on some envs).
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

FEATURE_LABELS: dict[str, str] = {
    "RR":          "Respiratory Rate",
    "SpO2":        "Oxygen Saturation",
    "HR":          "Heart Rate",
    "SBP":         "Systolic BP",
    "Temperature": "Temperature",
    "AVPU":        "AVPU Level",
}


def explain_rrt_components(components: dict[str, int]) -> list[dict]:
    """
    Convert RRT sub-scores into a ranked, human-readable explanation list.

    Args:
        components: Dict mapping vital name → sub-score (0–3).

    Returns:
        List of dicts sorted by score descending, each with:
            name, label, score, max_score, pct, severity
    """
    results: list[dict] = []
    for key, score in components.items():
        pct = round(score / 3 * 100)
        if score == 0:
            severity = "normal"
        elif score == 1:
            severity = "mild"
        elif score == 2:
            severity = "moderate"
        else:
            severity = "severe"

        results.append({
            "name":      key,
            "label":     FEATURE_LABELS.get(key, key),
            "score":     score,
            "max_score": 3,
            "pct":       pct,
            "severity":  severity,
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)


def trend_direction(current_val: float, predicted_val: float, tolerance: float = 0.5) -> str:
    """
    Classify a vital sign trend direction.

    Args:
        current_val:   Current observed value.
        predicted_val: Predicted future value.
        tolerance:     Minimum absolute delta to classify as improving/worsening.

    Returns:
        "improving" | "worsening" | "stable"
    """
    delta = predicted_val - current_val
    if abs(delta) < tolerance:
        return "stable"
    return "worsening" if delta > 0 else "improving"


def generate_clinical_narrative(
    current_rrt: dict,
    t4_rrt: dict,
    t8_rrt: dict,
    current_vitals: dict,
    t4_vitals: dict,
) -> str:
    """
    Generate a brief clinical narrative from current and predicted RRT data.

    Args:
        current_rrt:    Dict with score, category, label.
        t4_rrt:         T+4h RRT dict.
        t8_rrt:         T+8h RRT dict.
        current_vitals: Current vitals dict.
        t4_vitals:      Predicted T+4h vitals dict.

    Returns:
        Multi-line narrative string for display.
    """
    curr_score = current_rrt["score"]
    t4_score   = t4_rrt["score"]
    t8_score   = t8_rrt["score"]
    curr_label = current_rrt["label"]
    t8_label   = t8_rrt["label"]

    delta_4 = t4_score - curr_score
    delta_8 = t8_score - curr_score

    trajectory = "stable"
    if delta_8 >= 3:
        trajectory = "deteriorating"
    elif delta_8 >= 1:
        trajectory = "mildly worsening"
    elif delta_8 <= -3:
        trajectory = "improving"
    elif delta_8 <= -1:
        trajectory = "mildly improving"

    lines: list[str] = [
        f"**Current status:** {curr_label} (RRT {curr_score}/18)",
        f"**8-hour forecast:** {t8_label} (RRT {t8_score}/18) — trajectory is **{trajectory}**.",
    ]

    # Highlight worst-scoring vital
    comp = current_rrt.get("components", {})
    if comp:
        worst = max(comp, key=lambda k: comp[k])
        worst_score = comp[worst]
        if worst_score >= 2:
            lines.append(
                f"**Primary concern:** {FEATURE_LABELS.get(worst, worst)} "
                f"contributes {worst_score}/3 to the current RRT score."
            )

    if delta_4 > 0:
        lines.append(f"Score is projected to rise by {delta_4} point(s) within 4 hours.")
    if t8_score >= 12:
        lines.append("⚠️ **RRT activation may be warranted within 8 hours.** Reassess urgently.")

    return "\n\n".join(lines)
