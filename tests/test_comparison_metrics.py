"""A short spike must remain visible even when its mean error is small."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_eclipse import compare, comparison_report


class Reference:
    days = np.arange(5, dtype=float)

    def get(self, key, arg=None):
        return np.full(5, 100.0)

    def unit(self, key, arg=None):
        return "SM3/SM3"


def test_spike_and_window_are_reported_per_well():
    s = {"TIME": np.arange(5), "WWPR:W1": np.array([np.nan, 100, 150, 100, 110])}
    refs = {"eclipse": Reference()}
    report = comparison_report(s, refs)
    assert report["metrics"]["mean"][0]["eclipse"] == pytest.approx(0.15)
    assert report["metrics"]["max"][0]["eclipse"] == pytest.approx(0.5)
    assert report["metrics"]["final"][0]["eclipse"] == pytest.approx(0.1)
    assert report["peaks"][0]["day"] == 2
    assert report["peaks"][0]["absolute_error"] == 50
    assert report["peaks"][0]["final_relative_error"] == pytest.approx(0.1)
    assert compare(s, refs, "max", 3, 4)[0]["eclipse"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="нет отчётных"):
        compare(s, refs, start_day=10)


def test_nonfinite_result_is_not_a_passing_comparison():
    s = {"TIME": np.arange(3), "FOPR": np.array([np.nan, 100, np.nan])}
    with pytest.raises(ValueError, match="нечисловые"):
        compare(s, {"eclipse": Reference()})


def test_zero_production_ratio_has_explicit_sample_coverage():
    s = {"TIME": np.arange(4), "FGOR": np.array([np.nan, np.nan, 100, 110])}
    report = comparison_report(s, {"eclipse": Reference()})
    peak = report["peaks"][0]
    assert peak["samples_compared"] == 2
    assert peak["samples_undefined"] == 1
    assert peak["last_compared_day"] == 3
