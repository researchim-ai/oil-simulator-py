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


@pytest.mark.parametrize("values", [[1.0, float("nan"), 1.0], [], [True, 1.0, 1.0]])
def test_refinement_rejects_missing_or_nonfinite_fields_even_after_a_finite_first_cell(
    forecast, tmp_path, values
):
    import gzip
    from compare_fim_facility_forecast import compare

    c, events, r = forecast
    a = {"config": c, "schedule": events, "requested_days": 1, "result": r}
    b = copy.deepcopy(a)
    b["result"]["final_s_g"] = values
    paths = [tmp_path / "a.gz", tmp_path / "b.gz"]
    for path, bundle in zip(paths, [a, b], strict=True):
        with gzip.open(path, "wt") as handle:
            json.dump(bundle, handle)
    with pytest.raises(ValueError, match="finite final field"):
        compare(*paths, {
            "final_pressure_pa": 100.0, "final_s_w": 2e-5, "final_s_g": 2e-5,
            "final_rs": 0.01, "withdrawal_relative": 0.002,
        })


def test_explicit_null_control_defaults_are_auditable():
    c = config()
    c["simulation"]["coupled_facility"]["control"] = None
    r = run_case(c, 1e6, 1, 1)
    assert audit(c, r, expected_schedule=[], expected_days=1)["accepted"]


@pytest.mark.parametrize(
    "policy",
    [
        {"bhp_initialization": "rate_network"},
        {"sensitivity_backend": "tree_operator"},
        {"network_backend": "torch_device"},
        {"bhp_initialization": "rate_network"},
        {"hydraulic_backend": "torch_batch", "pipe_batch_size": 32},
        {"well_block_backend": "tree_structured", "structured_memory_budget_bytes": 1048576},
    ],
)
def test_comparison_whitelists_tangent_policy_but_not_physics(forecast, tmp_path, policy):
    import gzip

    from compare_fim_facility_forecast import compare

    c, events, r = forecast
    base = {"config": c, "schedule": events, "requested_days": 1, "result": r}
    changed = copy.deepcopy(base)
    target = changed["config"]["simulation"]["physical_surface_network"]
    if "hydraulic_backend" in policy:
        target = target["network"]
    target.update(policy)
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


def test_progress_callback_reports_completed_intervals_without_changing_result():
    c = config()
    rows = []
    original = run_case(c, 1e6, 0.5, 0.25)
    observed = run_case(c, 1e6, 0.5, 0.25, progress_callback=rows.append)
    assert [r["time_days"] for r in rows] == [0.25, 0.5]
    assert all(r["accepted_substeps"] > 0 for r in rows)
    for key in [
        "final_pressure_pa",
        "final_s_w",
        "final_s_g",
        "final_rs",
        "withdrawn_component_mass_kg",
    ]:
        assert original[key] == observed[key]


def test_long_report_comparison_rejects_physics_changes_and_bhp_error(forecast, tmp_path):
    import gzip

    from check_long_facility_trajectory import compare

    c, events, r = forecast
    base = {"config": c, "schedule": events, "requested_days": 1.0, "result": r}
    a, b = tmp_path / "a.gz", tmp_path / "b.gz"

    def save(path, bundle):
        with gzip.open(path, "wt") as handle:
            json.dump(bundle, handle)

    save(a, base)
    save(b, base)
    checked = compare(a, b)
    assert checked["accepted"] and checked["mode_mismatches"] == 0
    changed = copy.deepcopy(base)
    changed["result"]["accepted_substeps"][0]["surface"]["wells"]["P1"]["bhp_pa"] += 1000
    save(b, changed)
    assert not compare(a, b)["accepted"]
    changed = copy.deepcopy(base)
    changed["config"]["reservoir"]["porosity"] = 0.3
    save(b, changed)
    with pytest.raises(ValueError, match="Identical"):
        compare(a, b)


def test_depleted_reservoir_reuses_producing_warm_start_across_capacity_trials():
    # This model previously stopped at day 5: repeated 5% BHP reseeding
    # overshot into backflow at every retry, including very small dt.
    c = json.loads((ROOT / "benchmarks/reports/surface_gpu_year/depletion32_wells16_torch.json").read_text())
    c["fluid"]["pvt_path"] = str(ROOT / c["fluid"]["pvt_path"])
    r = run_case(c, 1e6, 10., 5.)
    assert audit(c, r, expected_schedule=[], expected_days=10.)["accepted"]
    assert r["elapsed_days"] == 10.
    assert max(r["final_pressure_pa"]) < 2.9e6
    for step in r["accepted_substeps"]:
        assert all(q >= 0 for well in step["surface"]["wells"].values()
                   for q in well["component_mass_rates_kg_s"])


def test_post_run_audit_failure_is_published_without_a_complete_summary(forecast, tmp_path, monkeypatch):
    import run_fim_facility_forecast as driver

    c, events, result = forecast
    config_path, schedule = tmp_path / "config.json", tmp_path / "schedule.json"
    config_path.write_text(json.dumps(c))
    schedule.write_text(json.dumps(events))
    output = tmp_path / "failed"
    monkeypatch.setattr(driver, "run_case", lambda *a, **kw: copy.deepcopy(result))

    def reject(*a, **kw):
        raise ValueError("deliberate post-run balance rejection")

    monkeypatch.setattr(driver, "audit", reject)
    monkeypatch.setattr(sys, "argv", [driver.__file__, "--config", str(config_path),
                                     "--schedule", str(schedule), "--days", "1",
                                     "--output", str(output)])
    with pytest.raises(ValueError, match="post-run balance"):
        driver.main()
    failure = json.loads((output / "failure.json").read_text())
    assert failure["stage"] == "accounting_audit"
    assert (output / "trajectory.json.gz").exists()
    assert not (output / "summary.json").exists()
    assert json.loads((output / "progress.json").read_text())["status"] == "incomplete"


def test_audit_rejects_ignored_requested_maximum_step(forecast):
    c,events,r=forecast
    c=copy.deepcopy(c);c['simulation']['max_time_step_days']=.01
    with pytest.raises(ValueError,match='maximum time step'):
        audit(c,r,expected_schedule=events,expected_days=1.)


def test_time_step_refinement_is_numerical_policy_but_not_physics():
    from compare_fim_facility_forecast import physical_config
    a={'config':config()};b=copy.deepcopy(a)
    b['config']['simulation'].update(max_time_step_days=.03125,initial_time_step_days=.00005)
    assert physical_config(a)!=physical_config(b)
    assert physical_config(a,allow_numerical_policy=True)==physical_config(b,allow_numerical_policy=True)
    b['config']['simulation']['max_time_step_days']=False
    with pytest.raises(ValueError,match='time-step policy'):
        physical_config(b,allow_numerical_policy=True)
