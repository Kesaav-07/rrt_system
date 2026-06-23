"""
tests/test_realtime.py
----------------------
Unit tests for the realtime engine.
"""

import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime.realtime import get_alerts, connection_status


def _make_patients() -> pd.DataFrame:
    return pd.DataFrame([
        {"patient_id": "P1001", "current_rrt_score": 14, "predicted_rrt_8hr": 16, "ward": "ICU"},
        {"patient_id": "P1002", "current_rrt_score": 4,  "predicted_rrt_8hr": 5,  "ward": "General"},
        {"patient_id": "P1003", "current_rrt_score": 8,  "predicted_rrt_8hr": 13, "ward": "Special"},
    ])


class TestGetAlerts:
    def test_critical_patients_flagged(self):
        df = _make_patients()
        alerts = get_alerts(df)
        assert "P1001" in alerts["patient_id"].values

    def test_stable_not_alerted(self):
        df = _make_patients()
        alerts = get_alerts(df)
        assert "P1002" not in alerts["patient_id"].values

    def test_empty_df_returns_empty(self):
        alerts = get_alerts(pd.DataFrame())
        assert alerts.empty


class TestConnectionStatus:
    def test_returns_tuple(self):
        status, ts = connection_status()
        assert isinstance(status, str)
        assert status in ("live", "reconnecting", "offline", "unknown")
