"""Safe station staging, availability, common-speed tangents and hysteresis."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import Liquid, LiquidNetwork, Node, Pump, PumpStationControl, PumpStationOperatingError


def model(power=600.0):
    p = Pump(
        "p1",
        "supply",
        "delivery",
        20.0,
        1e6,
        0.0044,
        efficiency=0.75,
        max_shaft_power_w=power,
        npsh_required_m=3.0,
        inlet_diameter_m=0.08,
        check_valve=True,
    )
    return LiquidNetwork(
        Liquid(1000, 0.001, 2300),
        [Node("supply", pressure_pa=2e5), Node("delivery", source_mass_rate_kg_s=-4.0)],
        [],
        [p, replace(p, name="p2"), replace(p, name="p3")],
    )


def control(**kw):
    return PumpStationControl(
        ("p1", "p2", "p3"),
        "delivery",
        2e5 + 1000 * 9.80665 * 20,
        0.5,
        1.5,
        min_pump_flow_m3_s=0.0015,
        min_npsh_margin_m=0.5,
        pressure_tolerance_pa=0.01,
        **kw,
    )


def test_staging_standby_failover_and_deadband():
    net = model()
    c = control()
    r = c.solve(net)
    assert len(r.running_pumps) == 2
    assert r.running_pumps == ("p1", "p2")
    assert abs(r.regulation.pressure_error_pa) < 0.01
    assert all(q < 600 for q in r.network.pump_shaft_power_w.values())
    assert -r.network.boundary_mass_rates_kg_s["supply"] == pytest.approx(-4.0)
    assert abs(r.network.power_balance_error_w) < 1e-5
    assert r.evaluations >= c.scan_points * 7
    failover = c.solve(net, unavailable=["p1"], previous_running=r.running_pumps, initial=r.network)
    assert failover.running_pumps == ("p2", "p3")
    recovered = c.solve(net, previous_running=failover.running_pumps, initial=failover.network)
    assert recovered.running_pumps == failover.running_pumps
    assert recovered.selection_reason == "retained_by_power_deadband"
    assert all(p.open and p.speed_ratio == 1 for p in net.pumps)


def test_safe_state_and_common_speed_derivatives():
    net = model()
    r = control().solve(net)
    derivative = r.control.pressure_sensitivities(net, r.regulation, ["delivery"], ["delivery"])
    assert abs(derivative[0, 0]) < 1e-8
    speed = net.pump_speed_sensitivities(r.network, r.running_pumps, ["delivery"])
    assert (speed > 0).all()
    assert set(r.network.operating_controls["pump_speed_ratios"]) == {"p1", "p2", "p3"}


def test_infeasible_station_keeps_rejection_audit_and_no_unsafe_result():
    c = control(max_station_power_w=900.0)
    with pytest.raises(PumpStationOperatingError) as exc:
        c.solve(model())
    assert exc.value.evaluations > 0 and exc.value.candidates
    assert all(row["status"] != "feasible" for row in exc.value.candidates)
    with pytest.raises(PumpStationOperatingError):
        control().solve(model(), unavailable=["p1", "p2"])


def test_bad_configuration_and_numerical_failure_are_not_capacity(monkeypatch):
    c = control()
    net = model()
    with pytest.raises(ValueError, match="budget"):
        replace(c, max_configurations=1).solve(net)
    with pytest.raises(ValueError, match="Unknown"):
        c.solve(net, unavailable=["typo"])
    from surface import HydraulicConvergenceError

    def fail(*args, **kw):
        raise HydraulicConvergenceError("numerical fault")

    monkeypatch.setattr(LiquidNetwork, "solve", fail)
    with pytest.raises(HydraulicConvergenceError, match="numerical fault"):
        c.solve(net)


def test_facility_outage_shutdown_and_restart_are_conservative():
    import json

    from surface import FacilityPeriod, run_facility_schedule

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_surface_facility import load_facility

    root = Path(__file__).resolve().parents[1]
    plant = load_facility(json.loads((root / "configs/surface_station_facility.json").read_text()))
    raw = json.loads((root / "configs/surface_station_schedule.json").read_text())
    periods = [FacilityPeriod(**p) for p in raw["periods"]]
    full = run_facility_schedule(plant, periods)
    partial = run_facility_schedule(plant, periods, stop_after=2)
    saved = json.loads(json.dumps(partial["checkpoint"]))
    resumed = run_facility_schedule(plant, periods, checkpoint=saved)
    assert full["checkpoint"] == resumed["checkpoint"]
    assert full["totals"] == resumed["totals"]
    assert full["elapsed_s"] == resumed["elapsed_s"]
    ledger = full["periods"] if "periods" in full else full["checkpoint"]["ledger"]
    for row in ledger:
        if row["shutdown"]:
            assert row["facility"] is None
            assert row["energies_j"]["facility_mechanical_energy_j"] == 0
        else:
            result = row["facility"]
            assert max(map(abs, result["component_mass_residuals_kg_s"].values())) < 1e-12
            assert abs(result["network"]["power_balance_error_w"]) < 1e-5
            if row["name"] == "unit_1_unavailable":
                assert "water_booster_1" not in result["pump_station"]["running_pumps"]
    assert plant.station_control is not None
