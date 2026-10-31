"""Safe capacity-limited curtailment and steady operating schedule restart."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_surface_facility import load_facility

from surface import (
    EquipmentOperatingError,
    FacilityCapacityControl,
    FacilityPeriod,
    HydraulicConvergenceError,
    run_facility_schedule,
)

ROOT = Path(__file__).resolve().parents[1]


def plant(power=20000):
    data = json.loads((ROOT / "configs/surface_gas_facility.json").read_text())
    for limit in data["limits"]:
        if limit["name"] == "total_utilities":
            limit["maximum"] = power
    return load_facility(data)


def test_capacity_keeps_feasible_side_and_reports_bracket_and_curtailment():
    f = plant()
    before = f.water_network.to_dict()
    control = FacilityCapacityControl()
    r = control.solve(f)
    assert r["status"] == "limited"
    assert 0.61 < r["fraction"] < 0.62
    assert r["facility"]["feasible"]
    assert 0 < r["upper_infeasible_fraction"] - r["fraction"] <= control.fraction_tolerance
    assert r["facility"]["measurements"]["facility_drive_power_total_w"]["value"] <= 20000
    assert r["limiting_constraints"][0]["name"] == "total_utilities"
    assert r["curtailed_standard_rates_m3_s"]["water_m3_s"] == pytest.approx(
        0.002 * (1 - r["fraction"])
    )
    assert f.water_network.to_dict() == before
    requested = control.solve(plant(50000))
    assert requested["status"] == "requested" and requested["evaluations"] == 1


def test_no_feasible_sample_is_not_a_fake_zero_production_result():
    r = FacilityCapacityControl().solve(plant(1))
    assert r["status"] == "no_feasible_sample"
    assert r["facility"] is None and r["fraction"] is None


def test_numerical_failure_is_never_a_capacity_violation(monkeypatch):
    f = plant()

    def fail(**kwargs):
        raise HydraulicConvergenceError("injected nonlinear failure")

    monkeypatch.setattr(f, "run", fail)
    with pytest.raises(HydraulicConvergenceError, match="injected"):
        FacilityCapacityControl().solve(f)


def periods():
    return [
        FacilityPeriod("full", 60),
        FacilityPeriod("half", 120, feed_scales=(0.5, 0.5), valve_openings={"water_choke": 0.9}),
        FacilityPeriod("stop", 30, shutdown=True),
        FacilityPeriod("restart", 60, feed_scales=(1, 1), valve_openings={"water_choke": 0.8}),
    ]


def test_schedule_closes_volumes_energy_and_restarts_identically():
    f = plant(50000)
    plan = periods()
    full = run_facility_schedule(f, plan)
    assert full["completed"] and full["elapsed_s"] == 270
    accepted = full["totals"]["accepted_standard_volumes_m3"]
    assert accepted == pytest.approx({"oil_m3": 0.18, "water_m3": 0.36, "gas_m3": 21.6})
    assert full["totals"]["product_standard_volumes_m3"] == pytest.approx(accepted, abs=1e-14)
    assert full["totals"]["curtailed_potential_standard_volumes_m3"]["water_m3"] == pytest.approx(
        0.03
    )
    assert full["checkpoint"]["ledger"][2]["facility"] is None
    first = run_facility_schedule(f, plan, stop_after=2)
    saved = json.loads(json.dumps(first["checkpoint"]))
    before = copy.deepcopy(saved)
    resumed = run_facility_schedule(f, plan, checkpoint=saved)
    assert saved == before
    assert resumed["totals"] == full["totals"]
    assert resumed["checkpoint"] == full["checkpoint"]


def test_capacity_control_is_applied_each_period_without_mutating_requested_rates():
    f = plant()
    plan = [FacilityPeriod("a", 60), FacilityPeriod("b", 60)]
    r = run_facility_schedule(f, plan, capacity_control=FacilityCapacityControl())
    rows = r["checkpoint"]["ledger"]
    assert all(row["dispatch"]["status"] == "limited" for row in rows)
    assert rows[0]["controls"]["feed_scales"] == [1, 1]
    assert rows[1]["controls"]["feed_scales"] == [1, 1]
    assert all(row["facility"]["feasible"] for row in rows)


def test_restart_rejects_changed_plant_plan_and_corruption():
    f = plant(50000)
    plan = periods()
    saved = run_facility_schedule(f, plan, stop_after=1)["checkpoint"]
    with pytest.raises(ValueError, match="model or operating plan"):
        run_facility_schedule(plant(49999), plan, checkpoint=saved)
    with pytest.raises(ValueError, match="model or operating plan"):
        run_facility_schedule(f, plan + [FacilityPeriod("extra", 1)], checkpoint=saved)
    bad = copy.deepcopy(saved)
    bad["elapsed_s"] += 1
    with pytest.raises(ValueError, match="checksum"):
        run_facility_schedule(f, plan, checkpoint=bad)


def test_property_model_failures_are_not_hidden_as_capacity_limits():
    data = json.loads((ROOT / "configs/surface_gas_facility.json").read_text())
    data["pvt"]["Bg"] = [1, 0.11, 0.025]
    with pytest.raises(EquipmentOperatingError, match="density mismatch"):
        FacilityCapacityControl().solve(load_facility(data))


def test_failed_period_leaves_the_supplied_checkpoint_unchanged():
    f = plant(50000)
    plan = [
        FacilityPeriod("good", 60),
        FacilityPeriod("bad", 60, valve_openings={"water_choke": 0}),
    ]
    saved = run_facility_schedule(f, plan, stop_after=1)["checkpoint"]
    before = copy.deepcopy(saved)
    with pytest.raises((ValueError, HydraulicConvergenceError)):
        run_facility_schedule(f, plan, checkpoint=saved)
    assert saved == before


def test_schedule_actuator_changes_survive_stop_and_restart():
    f = plant(50000)
    plan = [
        FacilityPeriod("less_head", 60, pump_speed_ratios={"water_booster": 0.9}),
        FacilityPeriod("export_pressure", 60, gas_header_pressure_pa=6500000),
    ]
    r = run_facility_schedule(f, plan)
    first, second = r["checkpoint"]["ledger"]
    assert second["controls"]["pump_speed_ratios"] == {"water_booster": 0.9}
    assert first["facility"]["network"]["pump_head_gains_m"]["water_booster"] < 16
    assert second["facility"]["gas_export"]["header"]["pressure_pa"] == pytest.approx(6500000)
    energy = r["totals"]["energies_j"]
    assert energy["facility_mechanical_energy_j"] == pytest.approx(
        energy["shaft_power_total_j"] + energy["gas_drive_power_j"]
    )


def test_capacity_rejects_disconnected_real_pump_power_feasible_regions():
    data = json.loads((ROOT / "configs/surface_facility.json").read_text())
    data["feeds"] = [{"water_m3_s": 0.0044}]
    pump = data["water_network"]["pumps"][0]
    for key in ("head_curve", "efficiency_curve", "npsh_curve"):
        pump.pop(key)
    pump["max_flow_m3_s"] = 0.0044
    data["limits"] = [
        {"name": "pump_power", "metric": "shaft_power_total_w", "unit": "W", "maximum": 300},
        {
            "name": "water_capacity",
            "metric": "water_handoff_mass_kg_s",
            "unit": "kg/s",
            "maximum": 4.3,
        },
    ]
    with pytest.raises(EquipmentOperatingError, match="Disconnected"):
        FacilityCapacityControl().solve(load_facility(data))


def test_exhausted_capacity_search_does_not_claim_optimality():
    with pytest.raises(HydraulicConvergenceError, match="evaluation limit"):
        FacilityCapacityControl(max_evaluations=17).solve(plant())
