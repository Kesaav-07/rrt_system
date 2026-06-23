"""
utils/helpers.py
----------------
General-purpose helper functions.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def iso_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return utc_now().isoformat(timespec="seconds")


def safe_int(val, default: int = 0) -> int:
    """Safely convert val to int, returning default on failure."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def safe_float(val, default: float = 0.0) -> float:
    """Safely convert val to float, returning default on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def rrt_color(score: int) -> str:
    """Return a hex color string for an RRT score."""
    if score >= 12:
        return "#D7263D"
    elif score >= 6:
        return "#E8A33D"
    return "#2E8B57"


def rrt_emoji(category: str) -> str:
    """Return an emoji for an RRT category string."""
    return {"critical": "🔴", "warning": "🟠", "stable": "🟢"}.get(category, "⚪")
