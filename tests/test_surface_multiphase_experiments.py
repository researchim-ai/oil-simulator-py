"""External measurements must not silently become fitted or extrapolated passes."""

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from validate_multiphase_experiments import predicted_increment, run, windows

ROOT = Path(__file__).resolve().parents[1]


def test_raw_experimental_holdout_exposes_annular_closure_failure():
    result = run(ROOT / "tests/data/surface/hannover_countercurrent")
    assert result["matched_wet_windows"] == 20
    assert result["wet_outside_dry_coverage"] == 11
    assert result["accepted_as_validated_for_annular_countercurrent"] is False
    assert result["fit_performed"] is False
    assert all(
        c["fraction_within_instrument_only_4pa_bound"] == 0.0
        for c in result["roughness_sensitivity"].values()
    )
    assert all(c["mean_absolute_error_pa"] > 10 for c in result["roughness_sensitivity"].values())


def test_steady_window_selection_does_not_depend_on_measured_pressure(monkeypatch):
    import validate_multiphase_experiments as module

    rows = [
        {"row": i + 3, "C": 0.05, "D": 150.0, "E": value, "F": 20.0, "G": 20.0, "H": 1.0}
        for i, value in enumerate(np.linspace(-100, 100, 60))
    ]
    monkeypatch.setattr(module, "read_xlsx", lambda path: [("synthetic_input_selection", rows)])
    a, rejected = windows("unused", wet=True)
    assert len(a) == 1 and not rejected
    for row in rows:
        row["E"] += 1e6
    b, rejected = windows("unused", wet=True)
    assert len(b) == 1 and not rejected
    assert (a[0]["first_row"], a[0]["last_row"]) == (b[0]["first_row"], b[0]["last_row"])


def test_steady_wall_closure_balance_matches_independent_scalar_colebrook():
    import math

    delta, alpha = predicted_increment(
        np.array([6.0]), np.array([0.05]), np.array([1.2]), roughness=0.0002
    )
    a = alpha[0]
    diameter = 0.085
    area = math.pi * diameter**2 / 4
    ul = -0.05 / (998.2 * area * a)
    ug = 6 / (1 - a)

    def factor(rho, velocity, mu):
        re = rho * abs(velocity) * diameter / mu
        assert re > 4000
        z = brentq(
            lambda z: z + 2 * math.log10(0.0002 / diameter / 3.7 + 2.51 * z / re),
            1,
            100,
            xtol=1e-13,
        )
        return 1 / z**2

    gas = factor(1.2, ug, 1.81e-5) * 1.2 * ug**2 / (2 * diameter)
    liquid = factor(998.2, ul, 0.001002) * 998.2 * ul**2 / (2 * diameter)
    assert -998.2 * 9.80665 + liquid == pytest.approx(-1.2 * 9.80665 - gas, abs=1e-6)
    dry = factor(1.2, 6, 1.81e-5) * 1.2 * 6**2 / (2 * diameter)
    assert delta[0] == pytest.approx(3 * (gas - dry), abs=1e-10)
