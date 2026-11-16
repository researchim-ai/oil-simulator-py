"""Real FIM/steady facility event boundaries and rejected corrupt accounting."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_fim_facility_forecast import audit
from run_fim_blackoil_surface import run_case


def config():
    c = json.loads((ROOT / "configs/fim_blackoil_facility_coupled.json").read_text())
    c["fluid"]["pvt_path"] = str(ROOT / c["fluid"]["pvt_path"])
    return c


@pytest.fixture(scope="module")
def forecast():
    c = config()
    original = copy.deepcopy(c)
    events = [
        {"time_days": 0.25, "controls": {"gas_valve_max_opening": 0.8}},
        {"time_days": 0.75, "controls": {}},
    ]
    result = run_case(c, 1e6, 1.0, 1.0, facility_schedule=events)
    assert c == original
    return c, events, result


def test_real_daily_forecast_hits_events_and_exports_actual_well_material(forecast):
    c, events, r = forecast
    checked = audit(c, r, expected_schedule=events, expected_days=1)
    assert checked["accepted"]
    assert checked["minimum_step_s"] >= 0.25 * 86400
    assert r["facility_control_events"] == [
        {**e, "applied_time_days": e["time_days"]} for e in events
    ]
    fractions = [
        (s["time_days"], s["facility_coupling"]["fraction"]) for s in r["accepted_substeps"]
    ]
    early = fractions[0][1]
    middle = [f for t, f in fractions if 0.25 < t <= 0.75]
    assert middle and max(middle) < early
    assert fractions[-1][1] > max(middle)
    assert max(map(abs, r["mass_balance_error_kg"])) < 0.01


@pytest.mark.parametrize(
    "mutation",
    [
        "product",
        "feed",
        "withdrawal",
        "event",
        "controls",
        "duration",
        "cap",
        "constraint",
        "bracket",
    ],
)
def test_independent_forecast_audit_rejects_corrupt_results(forecast, mutation):
    c, events, original = forecast
    r = copy.deepcopy(original)
    s = r["accepted_substeps"][0]
    plant = s["facility_coupling"]["facility"]
    if mutation == "product":
        plant["products"]["oil"]["oil_m3_s"] += 1
    elif mutation == "feed":
        plant["feed"]["gas_m3_s"] += 1
    elif mutation == "withdrawal":
        s["withdrawn_component_mass_kg"][0] += 1
    elif mutation == "event":
        r["facility_control_events"] = []
    elif mutation == "controls":
        s["facility_controls"] = {"gas_valve_max_opening": 0.1}
    elif mutation == "duration":
        r["elapsed_days"] = 0.5
    elif mutation == "cap":
        s["facility_coupling"]["applied_rate_caps_m3_day"][0] *= 2
    elif mutation == "bracket":
        s["facility_coupling"]["capacity_bracket"]["infeasible_fraction"] = 1.0
    elif mutation == "constraint":
        limit = c["simulation"]["coupled_facility"]["facility"]["limits"][0]
        plant["measurements"][limit["metric"]]["value"] = limit["maximum"] * 2
    with pytest.raises(ValueError):
        audit(c, r, expected_schedule=events, expected_days=1)


@pytest.mark.parametrize(
    "events",
    [
        [{"time_days": 1, "controls": {}}],
        [{"time_days": 0.5, "controls": {}}, {"time_days": 0.25, "controls": {}}],
        [{"time_days": 0, "controls": {"feed_scales": [0.1]}}],
        [{"time_days": float("nan"), "controls": {}}],
    ],
)
def test_invalid_operating_schedule_is_rejected_before_solving(events):
    with pytest.raises(ValueError):
        run_case(config(), 1e6, 1, facility_schedule=events)


def test_refinement_compares_fields_and_cannot_change_model(forecast, tmp_path):
    import gzip

    from compare_fim_facility_forecast import compare

    c, events, r = forecast
    bundle = {"config": c, "schedule": events, "requested_days": 1, "result": r}
    a, b = tmp_path / "a.gz", tmp_path / "b.gz"

    def save(path, content):
        with gzip.open(path, "wt") as handle:
            json.dump(content, handle)

    save(a, bundle)
    save(b, bundle)
    budgets = {
        "final_pressure_pa": 1000,
        "final_s_w": 0.001,
        "final_s_g": 0.001,
        "final_rs": 0.01,
        "withdrawal_relative": 0.01,
    }
    assert compare(a, b, budgets)["accepted"]
    modified = copy.deepcopy(bundle)
    modified["result"]["final_pressure_pa"][0] += 2000
    save(b, modified)
    assert not compare(a, b, budgets)["accepted"]
    modified["config"]["reservoir"]["porosity"] = 0.3
    save(b, modified)
    with pytest.raises(ValueError, match="same model"):
        compare(a, b, budgets)
    with pytest.raises(ValueError, match="Finite"):
        compare(a, a, {**budgets, "final_pressure_pa": float("inf")})


def test_explicit_null_control_defaults_are_auditable():
    c = config()
    c["simulation"]["coupled_facility"]["control"] = None
    r = run_case(c, 1e6, 1, 1)
    assert audit(c, r, expected_schedule=[], expected_days=1)["accepted"]


@pytest.mark.parametrize(
    "policy",
    [
        {"sensitivity_backend": "tree_operator"},
        {"well_block_backend": "tree_structured", "structured_memory_budget_bytes": 1048576},
    ],
)
def test_comparison_whitelists_tangent_policy_but_not_physics(forecast, tmp_path, policy):
    import gzip

    from compare_fim_facility_forecast import compare

    c, events, r = forecast
    base = {"config": c, "schedule": events, "requested_days": 1, "result": r}
    changed = copy.deepcopy(base)
    changed["config"]["simulation"]["physical_surface_network"].update(policy)
    a, b = tmp_path / "a.gz", tmp_path / "b.gz"

    def save(path, value):
        with gzip.open(path, "wt") as handle:
            json.dump(value, handle)

    save(a, base)
    save(b, changed)
    budgets = {
        "final_pressure_pa": 0.1,
        "final_s_w": 1e-8,
        "final_s_g": 1e-8,
        "final_rs": 1e-5,
        "withdrawal_relative": 1e-6,
    }
    with pytest.raises(ValueError, match="same model"):
        compare(a, b, budgets)
    assert compare(a, b, budgets, allow_numerical_policy=True)["accepted"]
    changed["config"]["reservoir"]["porosity"] = 0.3
    save(b, changed)
    with pytest.raises(ValueError, match="same model"):
        compare(a, b, budgets, allow_numerical_policy=True)
