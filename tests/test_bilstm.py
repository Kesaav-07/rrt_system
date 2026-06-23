"""
tests/test_bilstm.py
--------------------
Unit tests for the BiLSTM prediction module.
Tests run against the fallback path (no model required).
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.predict_bilstm import predict_from_current_vitals, predict_from_sequence
from config import SEQUENCE_LENGTH, N_FEATURES


STABLE_VITALS = {
    "RR": 16.0, "SpO2": 97.0, "HR": 75.0,
    "SBP": 120.0, "Temperature": 36.8, "AVPU": 0,
}


class TestPredictFromCurrentVitals:
    def test_returns_required_keys(self):
        result = predict_from_current_vitals(STABLE_VITALS)
        for key in ("current_vitals", "current_rrt", "t4_vitals", "t4_rrt",
                    "t8_vitals", "t8_rrt", "model_available"):
            assert key in result, f"Missing key: {key}"

    def test_rrt_score_in_range(self):
        result = predict_from_current_vitals(STABLE_VITALS)
        for horizon in ("current_rrt", "t4_rrt", "t8_rrt"):
            score = result[horizon]["score"]
            assert 0 <= score <= 18, f"{horizon} score {score} out of range"

    def test_rrt_category_valid(self):
        result = predict_from_current_vitals(STABLE_VITALS)
        valid = {"stable", "warning", "critical"}
        for horizon in ("current_rrt", "t4_rrt", "t8_rrt"):
            cat = result[horizon]["category"]
            assert cat in valid, f"Invalid category: {cat}"

    def test_avpu_string_input(self):
        vitals = {**STABLE_VITALS, "AVPU": "Alert"}
        result = predict_from_current_vitals(vitals)
        assert result["current_rrt"]["score"] == 0  # All normal → 0


class TestPredictFromSequence:
    def _make_sequence(self) -> list[list[float]]:
        return [[16.0, 97.0, 75.0, 120.0, 36.8, 0.0]] * SEQUENCE_LENGTH

    def test_valid_sequence(self):
        result = predict_from_sequence(self._make_sequence())
        assert "t4_rrt" in result and "t8_rrt" in result

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            predict_from_sequence([[16.0, 97.0, 75.0, 120.0, 36.8, 0.0]] * 5)

    def test_wrong_features_raises(self):
        with pytest.raises(ValueError):
            seq = [[16.0, 97.0, 75.0]] * SEQUENCE_LENGTH
            predict_from_sequence(seq)
