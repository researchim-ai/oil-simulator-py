"""Replay-located trips, latched actions and conservative restart accounting."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    DynamicVessel,
    DynamicVesselNetwork,
    HydraulicConvergenceError,
    NetworkTrip,
    PengRobinsonFluid,
    PengRobinsonMixture,
    VesselConnection,
    VesselState,
    advance_protected_network,
    run_network_schedule,
)
from surface.operation import _digest
from surface.peng_robinson import R


def case():
    c = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/surface_thermal_mixture.json").read_text()
    )["fluid"]
    f = PengRobinsonMixture(
        tuple(PengRobinsonFluid(**x) for x in c["components"]), c["binary_interactions"]
    )
    v = DynamicVessel("tank", 1, 1, 1000)
    net = DynamicVesselNetwork({"tank": v})
    states = {"tank": v.initialize(f, 100, 300, [0.5, 0.5])}
    trip = NetworkTrip(
        "high-p", "tank", "pressure_pa", "above", 100.1, 1e-5, stop_heat_vessels=("tank",)
    )
    return f, v, net, states, trip


def test_pressure_event_replays_step_and_heat_stops_at_event_without_lost_time():
    f, _v, net, s, trip = case()
    r = advance_protected_network(
        net,
        s,
        f,
        10,
        [trip],
        heat={"tank": {"heat_w": 0.1, "utility_temperature_k": 400}},
        initial_dt_s=3,
    )
    assert len(r["events"]) == 1 and r["latched_trips"] == ["high-p"]
    event = r["events"][0]
    end = r["states"]["tank"]
    assert 0 < event["time_s"] < 10 and end["time_s"] == 10
    assert end["pressure_pa"] == pytest.approx(100.1, abs=trip.value_tolerance)
    heat = sum(step["heat_j"] for step in r["steps"])
    assert heat == pytest.approx(0.1 * event["time_s"], abs=1e-7)
    assert end["internal_energy_j"] - s["tank"].internal_energy_j == pytest.approx(heat, abs=1e-6)
    assert sum(step["dt_s"] for step in r["steps"]) == pytest.approx(10)
    cp = sum(c.ideal_cp_j_mol_k for c in f.components) / 2
    expected = (0.1 / 100) * sum(s["tank"].component_moles) * (cp - R) * 300 / 0.1
    assert event["time_s"] == pytest.approx(expected, rel=5e-5)


def test_time_location_stable_with_different_report_substeps():
    f, _v, net, s, trip = case()
    times = []
    for dt in (1.0, 3.0):
        r = advance_protected_network(
            net,
            s,
            f,
            6,
            [trip],
            heat={"tank": {"heat_w": 0.1, "utility_temperature_k": 400}},
            initial_dt_s=dt,
        )
        times.append(r["events"][0]["time_s"])
    assert times[0] == pytest.approx(times[1], abs=1e-5)


def test_connection_trip_closes_and_global_inventory_energy_stay_conservative():
    f, _, _, _, _ = case()
    v = {n: DynamicVessel(n, 1, 1, 1000) for n in ("a", "b")}
    s = {
        "a": v["a"].initialize(f, 200, 300, [0.3, 0.7]),
        "b": v["b"].initialize(f, 100, 300, [0.7, 0.3]),
    }
    net = DynamicVesselNetwork(v, (VesselConnection("line", "a", "b", 1e-5, 500, 0.01),))
    trip = NetworkTrip(
        "receiver-high", "b", "pressure_pa", "above", 100.2, 1e-5, close_connections=("line",)
    )
    r = advance_protected_network(net, s, f, 1, [trip], initial_dt_s=0.5)
    assert r["events"][0]["time_s"] < 0.5
    assert r["steps"][-1]["connections"][0]["molar_rate_mol_s"] == 0
    assert r["states"]["b"]["pressure_pa"] == pytest.approx(100.2, abs=1e-5)
    assert np.sum([x["component_moles"] for x in r["states"].values()], axis=0) == pytest.approx(
        np.sum([x.component_moles for x in s.values()], axis=0), abs=1e-9
    )
    assert sum(x["internal_energy_j"] for x in r["states"].values()) == pytest.approx(
        sum(x.internal_energy_j for x in s.values()), abs=1e-7
    )


def test_blocked_feed_is_separate_from_actual_inventory_and_persists_on_restart():
    f, _v, net, s, _ = case()
    total = sum(s["tank"].component_moles)
    trip = NetworkTrip(
        "inventory-high",
        "tank",
        "total_moles",
        "above",
        total + 0.0001,
        1e-9,
        stop_feed_vessels=("tank",),
    )
    feed = {
        "pressure_pa": 1000,
        "temperature_k": 300,
        "molar_rate_mol_s": 0.001,
        "mole_fractions": [0.5, 0.5],
    }
    periods = [
        {"duration_s": 1, "feeds": {"tank": {"inlet": feed}}},
        {"duration_s": 1, "feeds": {"tank": {"inlet": feed}}},
    ]
    full = run_network_schedule(
        f, net, s, periods, protections=[trip], stepper={"initial_dt_s": 0.5}
    )
    partial = run_network_schedule(
        f, net, s, periods, protections=[trip], stepper={"initial_dt_s": 0.5}, stop_after=1
    )
    saved = copy.deepcopy(partial)
    resumed = run_network_schedule(
        f,
        net,
        s,
        periods,
        protections=[trip],
        stepper={"initial_dt_s": 0.5},
        checkpoint=json.loads(json.dumps(partial)),
    )
    assert resumed == full and partial == saved
    assert full["latched_trips"] == ["inventory-high"]
    admitted = sum(full["totals"]["inlet_component_moles"])
    blocked = sum(full["blocked_feed_component_moles"])
    assert admitted == pytest.approx(0.0001, abs=1e-9)
    assert admitted + blocked == pytest.approx(0.002, abs=1e-10)
    assert sum(full["states"]["tank"]["component_moles"]) == pytest.approx(
        total + admitted, abs=1e-9
    )
    corrupt = copy.deepcopy(partial)
    corrupt["latched_trips"] = []
    corrupt["checksum"] = _digest({k: v for k, v in corrupt.items() if k != "checksum"})
    with pytest.raises(ValueError, match="latch"):
        run_network_schedule(
            f,
            net,
            s,
            periods,
            protections=[trip],
            stepper={"initial_dt_s": 0.5},
            checkpoint=corrupt,
        )


def test_initial_trip_applies_before_invalid_flow_and_unsafe_reset_retrips():
    f, v, net, _, _ = case()
    s = {"tank": v.initialize(f, 105, 300, [0.5, 0.5])}
    trip = NetworkTrip(
        "already-high", "tank", "pressure_pa", "above", 100.1, 1e-5, stop_feed_vessels=("tank",)
    )
    feed = {
        "pressure_pa": 50,
        "temperature_k": 300,
        "molar_rate_mol_s": 0.001,
        "mole_fractions": [0.5, 0.5],
    }
    periods = [
        {"duration_s": 0.1, "feeds": {"tank": {"low-source": feed}}},
        {
            "duration_s": 0.1,
            "feeds": {"tank": {"low-source": feed}},
            "reset_trips": ["already-high"],
        },
    ]
    r = run_network_schedule(f, net, s, periods, protections=[trip], stepper={"initial_dt_s": 0.1})
    assert [e["time_s"] for rec in r["records"] for e in rec["events"]] == [0, 0.1]
    assert all(e["initially_triggered"] for rec in r["records"] for e in rec["events"])
    assert r["states"]["tank"]["time_s"] == 0.2 and r["totals"]["inlet_component_moles"] == [0, 0]


def test_same_time_trips_deterministic_and_no_chatter():
    f, _v, net, s, trip = case()
    second = NetworkTrip(
        "another", "tank", "pressure_pa", "above", 100.1, 1e-5, stop_heat_vessels=("tank",)
    )
    r = advance_protected_network(
        net,
        s,
        f,
        6,
        [trip, second],
        heat={"tank": {"heat_w": 0.1, "utility_temperature_k": 400}},
        initial_dt_s=3,
    )
    assert [e["name"] for e in r["events"]] == ["another", "high-p"]
    assert r["events"][0]["time_s"] == r["events"][1]["time_s"]


def test_low_inventory_stops_withdrawal_before_depletion():
    f, _v, net, s, _ = case()
    initial = sum(s["tank"].component_moles)
    trip = NetworkTrip(
        "low-n",
        "tank",
        "total_moles",
        "below",
        initial - 0.0001,
        1e-9,
        stop_withdrawal_vessels=("tank",),
    )
    r = advance_protected_network(
        net, s, f, 1, [trip], withdrawals={"tank": {"bulk": 0.001}}, initial_dt_s=0.5
    )
    assert r["events"][0]["time_s"] == pytest.approx(0.1, abs=1e-6)
    assert sum(r["states"]["tank"]["component_moles"]) == pytest.approx(initial - 0.0001, abs=1e-9)
    assert sum(sum(step["outlet_component_moles"]) for step in r["steps"]) == pytest.approx(
        0.0001, abs=1e-9
    )


def test_liquid_volume_event_uses_real_two_phase_volume_not_density_guess():
    f, _, _, _, _ = case()
    v = DynamicVessel("tank", 0.1, 1e5, 5e6)
    net = DynamicVesselNetwork({"tank": v})
    s = {"tank": v.initialize(f, 3e6, 220, [0.5, 0.5])}
    before = v.inspect(s["tank"], f)["liquid_volume_fraction"]
    trial = net.advance(
        s, f, 1, heat={"tank": {"heat_w": 100, "utility_temperature_k": 300}}, initial_dt_s=1
    )
    after = v.inspect(VesselState(**trial["states"]["tank"]), f)["liquid_volume_fraction"]
    threshold = (before + after) / 2
    rule = NetworkTrip(
        "level",
        "tank",
        "liquid_volume_fraction",
        "above" if after > before else "below",
        threshold,
        1e-7,
        stop_heat_vessels=("tank",),
    )
    r = advance_protected_network(
        net,
        s,
        f,
        1,
        [rule],
        heat={"tank": {"heat_w": 100, "utility_temperature_k": 300}},
        initial_dt_s=1,
    )
    final = v.inspect(VesselState(**r["states"]["tank"]), f)["liquid_volume_fraction"]
    assert final == pytest.approx(threshold, abs=1e-7)
    assert 0 < r["events"][0]["time_s"] < 1


def test_bad_protection_and_hard_limits_remain_strict_and_transactional():
    f, _v, net, s, _trip = case()
    saved = {n: q.to_dict() for n, q in s.items()}
    with pytest.raises(ValueError, match="strictly inside"):
        bad = NetworkTrip(
            "hard", "tank", "pressure_pa", "above", 1000, 1e-5, stop_heat_vessels=("tank",)
        )
        advance_protected_network(net, s, f, 1, [bad])
    ineffective = NetworkTrip(
        "ineffective", "tank", "pressure_pa", "above", 100.1, 1e-5, stop_feed_vessels=("tank",)
    )
    narrow = DynamicVesselNetwork({"tank": DynamicVessel("tank", 1, 1, 101)})
    with pytest.raises(HydraulicConvergenceError):
        advance_protected_network(
            narrow,
            s,
            f,
            1,
            [ineffective],
            heat={"tank": {"heat_w": 100, "utility_temperature_k": 450}},
            initial_dt_s=0.2,
            minimum_dt_s=0.01,
        )
    assert {n: q.to_dict() for n, q in s.items()} == saved


def test_latched_heater_trip_allows_cooling_and_safe_manual_rearm():
    f, _, net, s, _ = case()
    trip = NetworkTrip(
        "heater-high", "tank", "pressure_pa", "above", 100.1, 1e-5, stop_heating_vessels=("tank",)
    )
    periods = [
        {"duration_s": 6, "heat": {"tank": {"heat_w": 0.1, "utility_temperature_k": 400}}},
        {"duration_s": 1, "heat": {"tank": {"heat_w": -0.1, "utility_temperature_k": 200}}},
        {
            "duration_s": 0.1,
            "reset_trips": ["heater-high"],
            "heat": {"tank": {"heat_w": 0.1, "utility_temperature_k": 400}},
        },
    ]
    r = run_network_schedule(f, net, s, periods, protections=[trip], stepper={"initial_dt_s": 3})
    assert sum(step["heat_j"] for step in r["records"][1]["steps"]) == pytest.approx(-0.1, abs=1e-9)
    assert sum(step["heat_j"] for step in r["records"][2]["steps"]) == pytest.approx(0.01, abs=1e-9)
    assert r["latched_trips"] == []


def test_observer_event_does_not_claim_full_interval_or_hide_partial_time():
    f, _, net, s, trip = case()
    r = net.advance(
        s,
        f,
        10,
        heat={"tank": {"heat_w": 0.1, "utility_temperature_k": 400}},
        initial_dt_s=3,
        stop_conditions=[trip],
    )
    assert 0 < r["duration_s"] < r["requested_duration_s"] == 10
    assert r["states"]["tank"]["time_s"] == pytest.approx(r["duration_s"])
    assert sum(step["dt_s"] for step in r["steps"]) == pytest.approx(r["duration_s"])


def test_event_root_failure_retries_without_losing_time_or_energy(monkeypatch):
    import surface.network_events as module

    f, _v, net, s, trip = case()
    original = module.brentq
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("forced scalar root failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "brentq", fail_once)
    r = advance_protected_network(
        net,
        s,
        f,
        6,
        [trip],
        heat={"tank": {"heat_w": 0.1, "utility_temperature_k": 400}},
        initial_dt_s=3,
    )
    assert r["failed_steps"] >= 1 and len(r["events"]) == 1
    assert r["states"]["tank"]["time_s"] == 6
    assert sum(step["dt_s"] for step in r["steps"]) == pytest.approx(6)
    assert r["states"]["tank"]["internal_energy_j"] - s["tank"].internal_energy_j == pytest.approx(
        0.1 * r["events"][0]["time_s"], abs=1e-6
    )
