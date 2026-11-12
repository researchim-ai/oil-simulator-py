"""External measurements must not silently become fitted or extrapolated passes."""

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from validate_multiphase_experiments import (
    dry_series_sensitivity,
    predicted_increment,
    run,
    windows,
)

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


def test_dry_series_comparison_keeps_common_population_and_never_extrapolates():
    from validate_annular_film_experiment import score_dry_series

    def row(sheet, index, q, dp):
        return {
            "sheet": sheet,
            "first_row": index,
            "last_row": index + 59,
            "gas_volume_flow_m3_h": q,
            "pressure_loss_pa": dp,
            "temperature_k": 293.15,
            "absolute_pressure_pa": 1e5,
        }

    dry = [row("a", 1, 100, 10), row("a", 61, 200, 30), row("b", 1, 120, 24), row("b", 61, 180, 36)]
    wet = [row("wet", i, q, 50) for i, q in enumerate([110, 150, 190, 210])]
    diagnostic = dry_series_sensitivity(dry, wet)
    assert [r["all_dry_series_covered"] for r in diagnostic["cases"]] == [False, True, False, False]
    assert diagnostic["cases"][-1]["dry_series"] == {}
    assert diagnostic["cases"][1]["dry_series_spread_pa"] == 10
    predictions = [{**r, "predicted_wet_increment_pa": 35} for r in wet]
    score = score_dry_series(predictions, diagnostic)
    assert score["common_wet_windows"] == 1
    assert score["excluded_from_common_coverage"] == 3
    assert score["dry_series"]["a"]["bias_pa"] == 5
    assert score["dry_series"]["b"]["bias_pa"] == 15
    assert score["is_uncertainty_interval"] is False
    # A changed model cannot choose a more favorable reference population.
    predictions[0]["predicted_wet_increment_pa"] = 1e6
    assert score_dry_series(predictions, diagnostic) == score
    single = dry_series_sensitivity([r for r in dry if r["sheet"] == "a"], wet)
    single_score = score_dry_series(predictions, single)
    assert single_score["common_wet_windows"] == 3
    assert single_score["maximum_dry_series_spread_pa"] is None


def test_hannover_reference_run_difference_does_not_explain_annular_error():
    from validate_annular_film_experiment import validate

    report = validate(
        ROOT / "tests/data/surface/hannover_countercurrent", shear_velocity="interface"
    )
    assert report["within_declared_pressure_bounds"] is False
    assert "full_circumferential_wetting_not_confirmed" in report["qualification_blockers"]
    for result in report["roughness_sensitivity"].values():
        sensitivity = result["dry_series_sensitivity"]
        assert sensitivity["common_wet_windows"] == 9
        assert sensitivity["maximum_dry_series_spread_pa"] == pytest.approx(16.3883926492)
        for run_result in sensitivity["dry_series"].values():
            assert run_result["windows"] == 9
            assert run_result["mean_absolute_error_pa"] > 30
            assert run_result["within_instrument_only_4pa_count"] == 0


def test_perfect_pressure_match_does_not_certify_unobserved_wetting(monkeypatch):
    import torch
    import validate_annular_film_experiment as module

    original = module.countercurrent_film
    baseline = run(ROOT / "tests/data/surface/hannover_countercurrent")
    rows = [
        r
        for r in baseline["roughness_sensitivity"]["0.0"]["cases"]
        if r["liquid_mass_flow_kg_s"] / (np.pi * 0.085 * 0.001002) <= 400
    ]

    def perfect(jg, mass, rho, **kwargs):
        result = original(jg, mass, rho, **kwargs)
        observed = torch.tensor([r["measured_wet_increment_pa"] for r in rows])
        result["pressure_gradient_pa_m"] = -rho * 9.80665 - observed / 3
        return result

    monkeypatch.setattr(module, "countercurrent_film", perfect)
    monkeypatch.setattr(module, "darcy_factor", lambda re, roughness: torch.zeros_like(re))
    result = module.validate(ROOT / "tests/data/surface/hannover_countercurrent")
    assert result["within_declared_pressure_bounds"] is True
    assert result["accepted_as_validated_for_annular_countercurrent"] is False
    assert "pressure_increment_discrepancy" not in result["qualification_blockers"]
    assert "full_circumferential_wetting_not_confirmed" in result["qualification_blockers"]
