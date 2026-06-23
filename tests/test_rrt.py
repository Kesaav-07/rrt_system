"""
tests/test_rrt.py
-----------------
Unit tests for the 18-point RRT scoring engine.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.rrt_calculator import (
    score_rr,
    score_spo2,
    score_hr,
    score_sbp,
    score_temperature,
    score_avpu,
    calculate_rrt_score,
    rrt_category_from_score,
)


class TestScoreRR:
    def test_normal(self):       assert score_rr(16)  == 0
    def test_low_mild(self):     assert score_rr(12)  == 1
    def test_low_moderate(self): assert score_rr(10)  == 2
    def test_very_low(self):     assert score_rr(5)   == 3
    def test_high_mild(self):    assert score_rr(22)  == 1
    def test_high_moderate(self):assert score_rr(27)  == 2
    def test_very_high(self):    assert score_rr(35)  == 3


class TestScoreSpO2:
    def test_normal(self):       assert score_spo2(97)  == 0
    def test_mild_low(self):     assert score_spo2(92)  == 1
    def test_moderate_low(self): assert score_spo2(87)  == 2
    def test_critical(self):     assert score_spo2(80)  == 3


class TestScoreHR:
    def test_normal(self):       assert score_hr(80)   == 0
    def test_mild_high(self):    assert score_hr(105)  == 1
    def test_moderate_high(self):assert score_hr(120)  == 2
    def test_critical_high(self):assert score_hr(150)  == 3
    def test_critical_low(self): assert score_hr(30)   == 3


class TestScoreSBP:
    def test_normal(self):       assert score_sbp(120)  == 0
    def test_mild_high(self):    assert score_sbp(150)  == 1
    def test_moderate_high(self):assert score_sbp(170)  == 2
    def test_critical_high(self):assert score_sbp(200)  == 3
    def test_critical_low(self): assert score_sbp(60)   == 3


class TestScoreTemperature:
    def test_normal(self):       assert score_temperature(37.0) == 0
    def test_mild_low(self):     assert score_temperature(35.5) == 1
    def test_critical_low(self): assert score_temperature(34.0) == 3
    def test_mild_high(self):    assert score_temperature(38.0) == 1
    def test_moderate_high(self):assert score_temperature(39.0) == 2
    def test_critical_high(self):assert score_temperature(40.0) == 3


class TestScoreAVPU:
    def test_alert(self):        assert score_avpu("Alert")        == 0
    def test_voice(self):        assert score_avpu("Voice")        == 1
    def test_pain(self):         assert score_avpu("Pain")         == 2
    def test_unresponsive(self): assert score_avpu("Unresponsive") == 3
    def test_numeric_0(self):    assert score_avpu(0)              == 0
    def test_numeric_3(self):    assert score_avpu(3)              == 3


class TestCalculateRRTScore:
    def test_stable_patient(self):
        score, comps, cat, label = calculate_rrt_score(
            rr=16, spo2=97, hr=75, sbp=120, temperature=36.8, avpu="Alert"
        )
        assert score == 0
        assert cat == "stable"
        assert label == "Stable"
        assert all(v == 0 for v in comps.values())

    def test_critical_patient(self):
        score, comps, cat, label = calculate_rrt_score(
            rr=35, spo2=80, hr=150, sbp=60, temperature=40.0, avpu="Unresponsive"
        )
        assert score >= 12
        assert cat == "critical"

    def test_max_score_capped(self):
        score, _, _, _ = calculate_rrt_score(
            rr=5, spo2=80, hr=30, sbp=60, temperature=40.5, avpu="Unresponsive"
        )
        assert score <= 18


class TestRRTCategoryFromScore:
    def test_stable(self):   assert rrt_category_from_score(3)  == ("stable",   "Stable")
    def test_warning(self):  assert rrt_category_from_score(8)  == ("warning",  "Warning")
    def test_critical(self): assert rrt_category_from_score(14) == ("critical", "Critical")
    def test_boundary_5(self): assert rrt_category_from_score(5)[0]  == "stable"
    def test_boundary_6(self): assert rrt_category_from_score(6)[0]  == "warning"
    def test_boundary_11(self): assert rrt_category_from_score(11)[0] == "warning"
    def test_boundary_12(self): assert rrt_category_from_score(12)[0] == "critical"
