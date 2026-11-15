"""Persistent controls, sampled latches, conservative coasting and replay."""

import copy
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from check_liquid_thermal_collocation import ConservativeReference, case
from check_liquid_transient import oscillator
from scipy.integrate import solve_ivp

from surface import (
    CollocationThermalState,
    CollocationThermalStation,
    ConservationError,
    HydraulicConvergenceError,
    LiquidHeatNetwork,
    LiquidTemperatureTrip,
    LiquidThermostat,
    PIDController,
    run_collocation_thermal_schedule,
)
from surface.operation import _digest


def passive(cap=10.0, interval=0.05):
    physical = oscillator(0.02)
    p = physical.initialize({"a": 600000.0, "b": 600000.0})
    heat = LiquidHeatNetwork(physical, 1000.0, {}, reference_temperature_k=300.0)
    t = heat.initialize(p)
    thermostat = LiquidThermostat(
        "heater", "a", 301.0, PIDController(500.0, 100.0, 0.0, -1000.0, 1000.0, tracking_time_s=0.1)
    )
    station = CollocationThermalStation(
        heat, [thermostat], control_interval_s=interval, max_heating_w=cap, max_cooling_w=20.0
    )
    return station, station.initialize(p, t)


def test_clock_and_pid_survive_partial_reports_without_duplicate_samples():
    station, initial = passive()
    whole = station.advance(initial, 0.4)
    state = initial
    samples = []
    for duration in (0.07, 0.11, 0.22):
        result = station.advance(state, duration)
        state = CollocationThermalState.from_dict(json.loads(json.dumps(result["state"])))
        samples.extend(result["samples"])
    assert [s["tick"] for s in samples] == [s["tick"] for s in whole["samples"]] == list(range(8))
    assert [s["time_s"] for s in samples] == pytest.approx([i * 0.05 for i in range(8)], abs=1e-15)
    assert state.next_tick == 8
    assert dict(state.heat.temperatures_k) == pytest.approx(
        whole["state"]["heat"]["temperatures_k"], abs=1e-10, rel=0
    )
    assert dict(state.held_duties_w) == pytest.approx(
        whole["state"]["held_duties_w"], abs=1e-7, rel=0
    )
    assert state.physical.time_s == pytest.approx(0.4, abs=1e-15)


def test_manual_duty_cap_has_exact_heating_and_antiwindup_tracking():
    station, initial = passive(cap=10.0)
    result = station.advance(
        initial,
        0.1,
        temperature_control_modes={"heater": "manual"},
        temperature_manual_duties_w={"heater": 100.0},
    )
    final = result["state"]
    assert final["heat"]["temperatures_k"]["a"] == pytest.approx(
        300.0 + 10.0 * 0.1 / (10.0 * 1000.0), abs=1e-11
    )
    assert all(s["controllers"]["heater"]["requested_duty_w"] == 100.0 for s in result["samples"])
    assert all(s["controllers"]["heater"]["applied_duty_w"] == 10.0 for s in result["samples"])
    automatic = station.advance(CollocationThermalState.from_dict(final), 0.05)
    # Reuse of PIDController provides bumpless transfer from the actual
    # constrained actuator, not the unconstrained 100 W request.
    assert automatic["samples"][0]["controllers"]["heater"]["requested_duty_w"] == 10.0
    assert max(abs(s["combined_energy_residual_j"]) for s in result["segments"]) < 1e-6


def test_held_output_changes_only_at_sampling_clock():
    station, initial = passive(cap=100.0)
    first = station.advance(
        initial,
        0.07,
        temperature_control_modes={"heater": "manual"},
        temperature_manual_duties_w={"heater": 40.0},
    )
    second = station.advance(
        CollocationThermalState.from_dict(first["state"]),
        0.08,
        temperature_control_modes={"heater": "off"},
    )
    # Off is a sampled thermostat command; 40 W remains applied from .07
    # to .1. The separate plant off/trip command acts at a period boundary.
    assert second["segments"][0]["thermal_controls"]["heat_duties_w"]["a"] == 40.0
    assert second["segments"][-1]["thermal_controls"]["heat_duties_w"]["a"] == 0.0
    assert second["state"]["heat"]["temperatures_k"]["a"] == pytest.approx(
        300.0 + 40.0 * 0.1 / 10000.0, abs=1e-10
    )


def trip_station():
    _, heat, p, t = case()
    rule = LiquidTemperatureTrip(
        "hot", "h", 300.02, delay_s=0.05, reset_margin_k=0.01, trip_station=True
    )
    station = CollocationThermalStation(heat, temperature_trips=[rule], control_interval_s=0.05)
    return station, station.initialize(p, t)


def test_delayed_temperature_latch_stops_torque_without_erasing_stored_energy():
    station, initial = trip_station()
    r = station.advance(initial, 0.15, motor_torques_nm={"a": 5.0, "b": 7.0})
    events = [e for e in r["events"] if e["type"] == "temperature_trip"]
    assert len(events) == 1 and events[0]["time_s"] == pytest.approx(0.1, abs=1e-14)
    assert r["state"]["trips"]["hot"]["tripped_at_s"] == pytest.approx(0.1)
    for segment in r["segments"]:
        start = segment["hydraulics"]["initial"]["time_s"]
        expected = {"a": 5.0, "b": 7.0} if start < 0.1 - 1e-14 else {"a": 0.0, "b": 0.0}
        assert segment["hydraulics"]["controls"]["motor_torques_nm"] == expected
    for a, b in zip(r["segments"][:-1], r["segments"][1:]):
        assert a["state"]["hydraulics"] == b["hydraulics"]["initial"]
        assert a["state"]["heat"] == b["initial_heat"]
    assert min(r["state"]["physical"]["speeds_rad_s"].values()) > 90.0
    assert not r["state"]["physical"]["tripped"]
    assert max(abs(s["combined_energy_residual_j"]) for s in r["segments"]) < 1e-6


def test_latched_temperature_trip_cannot_be_bypassed_by_operator_reset():
    station, initial = trip_station()
    first = station.advance(initial, 0.15, motor_torques_nm={"a": 5.0, "b": 7.0})
    state = CollocationThermalState.from_dict(first["state"])
    blocked = station.advance(
        state,
        0.03,
        reset_trip=True,
        reset_temperature_trips=["hot"],
        motor_torques_nm={"a": 5.0, "b": 7.0},
    )
    assert blocked["events"][0]["type"] == "reset_denied"
    assert all(
        s["hydraulics"]["controls"]["motor_torques_nm"] == {"a": 0.0, "b": 0.0}
        for s in blocked["segments"]
    )
    cooled = station.advance(state, 0.1, heat_duties_w={"h": -1e6})
    cold = CollocationThermalState.from_dict(cooled["state"])
    assert dict(cold.heat.temperatures_k)["h"] < 300.01
    reset = station.advance(
        cold, 0.02, reset_temperature_trips=["hot"], motor_torques_nm={"a": 5.0, "b": 7.0}
    )
    assert reset["events"][0]["type"] == "reset"
    assert reset["state"]["trips"]["hot"]["tripped_at_s"] is None
    assert reset["segments"][0]["hydraulics"]["controls"]["motor_torques_nm"] == {
        "a": 5.0,
        "b": 7.0,
    }


def test_violation_delay_survives_reporting_boundary():
    station, initial = trip_station()
    first = station.advance(initial, 0.07, motor_torques_nm={"a": 5.0, "b": 7.0})
    assert not first["events"]
    assert first["state"]["trips"]["hot"]["violating_since_s"] == 0.05
    next_state = CollocationThermalState.from_dict(json.loads(json.dumps(first["state"])))
    last = station.advance(next_state, 0.08, motor_torques_nm={"a": 5.0, "b": 7.0})
    assert len(last["events"]) == 1
    assert last["events"][0]["time_s"] == pytest.approx(0.1, abs=1e-14)


def test_period_changes_are_exact_and_checkpoint_replay_rejects_rechecksummed_forgery():
    station, initial = passive(cap=100.0)
    periods = [
        {
            "duration_s": 0.07,
            "temperature_control_modes": {"heater": "manual"},
            "temperature_manual_duties_w": {"heater": 40.0},
        },
        {"duration_s": 0.08, "temperature_control_modes": {"heater": "off"}},
    ]
    full = run_collocation_thermal_schedule(station, initial, periods)
    partial = run_collocation_thermal_schedule(station, initial, periods, stop_after=1)
    resumed = run_collocation_thermal_schedule(
        station, initial, periods, json.loads(json.dumps(partial))
    )
    assert full == resumed
    assert full["records"][1]["initial_state"]["physical"]["time_s"] == 0.07
    bad = copy.deepcopy(partial)
    bad["state"]["pids"]["heater"]["integral_output"] += 0.1
    bad["checksum"] = _digest({k: v for k, v in bad.items() if k != "checksum"})
    with pytest.raises(ConservationError, match="memory"):
        run_collocation_thermal_schedule(station, initial, periods, bad)
    bad = copy.deepcopy(partial)
    bad["records"][0]["state"]["heat"]["temperatures_k"]["a"] += 0.01
    bad["checksum"] = _digest({k: v for k, v in bad.items() if k != "checksum"})
    with pytest.raises(ConservationError, match="replay"):
        run_collocation_thermal_schedule(station, initial, periods, bad)


def test_json_model_and_state_roundtrip_and_failure_are_atomic():
    station, initial = passive()
    clone = CollocationThermalStation.from_dict(json.loads(json.dumps(station.to_dict())))
    restored = CollocationThermalState.from_dict(json.loads(json.dumps(initial.to_dict())))
    assert clone.advance(restored, 0.07) == station.advance(initial, 0.07)
    before = initial.to_dict()
    with pytest.raises(HydraulicConvergenceError, match="sample budget"):
        station.advance(initial, 0.2, max_samples=1)
    assert initial.to_dict() == before
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        station.advance(initial, 0.2, max_attempts=1)
    assert initial.to_dict() == before


def test_operator_trip_and_external_unavailability_persist_without_stock_reset():
    _, heat, p, t = case()
    station = CollocationThermalStation(heat)
    initial = station.initialize(p, t)
    trip = station.advance(initial, 0.02, mode="trip", motor_torques_nm={"a": 5.0, "b": 7.0})
    assert trip["events"][0]["type"] == "operator_trip"
    state = CollocationThermalState.from_dict(trip["state"])
    held = station.advance(state, 0.02, motor_torques_nm={"a": 5.0, "b": 7.0})
    assert held["state"]["operator_tripped"]
    assert all(
        s["hydraulics"]["controls"]["motor_torques_nm"] == {"a": 0.0, "b": 0.0}
        for s in held["segments"]
    )
    released = station.advance(
        CollocationThermalState.from_dict(held["state"]),
        0.02,
        reset_trip=True,
        unavailable=["a"],
        motor_torques_nm={"a": 5.0, "b": 7.0},
    )
    assert released["events"][0]["type"] == "operator_reset"
    assert all(
        s["hydraulics"]["controls"]["motor_torques_nm"] == {"a": 0.0, "b": 7.0}
        for s in released["segments"]
    )
    assert min(released["state"]["physical"]["speeds_rad_s"].values()) > 90.0
    persisted = station.advance(
        CollocationThermalState.from_dict(released["state"]),
        0.02,
        motor_torques_nm={"a": 5.0, "b": 7.0},
    )
    assert persisted["state"]["external_unavailable"] == ["a"]


def test_sampled_protection_does_not_claim_to_detect_between_sample_excursions():
    base, initial = passive()
    rule = LiquidTemperatureTrip("hot", "a", 300.05, trip_station=True)

    def cycle(interval):
        station = CollocationThermalStation(
            base.heat, temperature_trips=[rule], control_interval_s=interval
        )
        state = station.initialize(initial.physical, initial.heat)
        events = []
        for duration, power in ((0.01, 1e5), (0.01, -1e5), (0.04, 0.0)):
            result = station.advance(state, duration, heat_duties_w={"a": power})
            events.extend(result["events"])
            state = CollocationThermalState.from_dict(result["state"])
        return state, events

    slow, events = cycle(0.05)
    assert not events
    assert dict(slow.trips)["hot"].tripped_at_s is None
    fast, events = cycle(0.005)
    assert len(events) == 1
    assert dict(fast.trips)["hot"].tripped_at_s <= 0.01


def test_shutdown_trajectory_against_independent_piecewise_conservative_ode():
    station, initial = trip_station()
    result = station.advance(initial, 0.15, motor_torques_nm={"a": 5.0, "b": 7.0})
    reference = ConservativeReference(station.heat)
    y = reference.encode(initial.physical, initial.heat)
    previous = 0.0
    for segment in result["segments"]:
        stop = segment["state"]["hydraulics"]["time_s"]
        command = (5.0, 7.0) if previous < 0.1 - 1e-14 else (0.0, 0.0)
        rhs = ConservativeReference(station.heat, motor_torques_nm=command)
        expected = solve_ivp(
            rhs, (previous, stop), y, method="DOP853", rtol=1e-12, atol=1e-9, dense_output=True
        )
        assert expected.success
        for physical, thermal in zip(segment["hydraulics"]["steps"], segment["thermal_steps"]):
            time = physical["end_time_s"]
            from surface.torch_liquid import TorchLiquidEquations

            e = TorchLiquidEquations(station.physics)
            z = np.array(physical["scaled_state"])
            actual = np.r_[
                z[[0, 1, 2, 6, 7]] * np.r_[[e.ps] * 2, e.qs, [100, 110]],
                [thermal["temperatures_k"][n] for n in station.heat.names],
                [thermal["wall_temperatures_k"][n] for n in station.heat.wall_names],
                [thermal["masses_kg"][n] for n in reference.reservoirs],
            ]
            budget = np.r_[
                [1.0, 1.0, 1e-8, 1e-4, 1e-4],
                [1e-5] * (len(station.heat.names) + len(station.heat.wall_names)),
                [1e-7],
            ]
            assert np.all(abs(actual - rhs.observe(expected.sol(time))) < budget)
        y = expected.y[:, -1]
        previous = stop


def test_utility_cap_tracking_prevents_unbounded_integral_windup():
    station, initial = passive(cap=10.0)
    r = station.advance(initial, 2.0)
    requests = [s["controllers"]["heater"]["requested_duty_w"] for s in r["samples"]]
    assert requests[-1] < 30.0
    assert requests[-1] > 10.0
    assert r["state"]["held_duties_w"]["heater"] == 10.0
    energy = sum(
        s["thermal_controls"]["heat_duties_w"]["a"] * s["hydraulics"]["duration_s"]
        for s in r["segments"]
    )
    assert r["state"]["heat"]["temperatures_k"]["a"] == pytest.approx(
        300 + energy / 10000.0, abs=1e-10
    )


def test_shared_heating_and_cooling_caps_allocate_without_double_counting():
    base, initial = passive()
    units = [
        LiquidThermostat(n, location, target, PIDController(1.0, 1.0, 0.0, -1000.0, 1000.0))
        for n, location, target in (
            ("one", "a", 301.0),
            ("two", "b", 301.0),
            ("cool", "line::0", 299.0),
        )
    ]
    station = CollocationThermalStation(base.heat, units, max_heating_w=100.0, max_cooling_w=20.0)
    state = station.initialize(initial.physical, initial.heat)
    result = station.advance(
        state,
        0.1,
        temperature_control_modes={u.name: "manual" for u in units},
        temperature_manual_duties_w={"one": 100.0, "two": 100.0, "cool": -100.0},
    )
    for sample in result["samples"]:
        applied = {n: v["applied_duty_w"] for n, v in sample["controllers"].items()}
        assert applied == {"one": 50.0, "two": 50.0, "cool": -20.0}
    assert result["state"]["heat"]["temperatures_k"]["a"] == pytest.approx(300.0005, abs=1e-10)
    before = base.heat.inspect(initial.heat)
    from surface import LiquidHeatState

    after = base.heat.inspect(LiquidHeatState(**result["state"]["heat"]))
    assert after["fluid_energy_j"] - before["fluid_energy_j"] == pytest.approx(8.0, abs=1e-6)


def test_cli_checkpoint_replays_and_failure_keeps_existing_output(tmp_path):
    from run_collocation_thermal import run

    station, _ = passive()
    config = {
        "system": station.to_dict(),
        "initial": {"physical": {"pressures_pa": {"a": 600000.0, "b": 600000.0}}},
        "periods": [{"duration_s": 0.03}, {"duration_s": 0.04}],
    }
    full = run(config)
    partial = run(config, stop_after=1)
    assert run(config, json.loads(json.dumps(partial))) == full
    root = Path(__file__).resolve().parents[1]
    config["periods"][0]["duration_s"] = -0.1
    request = tmp_path / "bad.json"
    request.write_text(json.dumps(config))
    output = tmp_path / "result.json"
    output.write_text("preserve existing result")
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_collocation_thermal.py"),
            str(request),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert output.read_text() == "preserve existing result"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "change",
    [
        {"next_tick": -1},
        {"next_tick": True},
        {"next_tick": 10**400},
        {"clock_origin_s": 1.0},
        {"next_step_s": math.inf},
        {"held_duties_w": {"heater": 1000.0}},
    ],
)
def test_corrupt_controller_state_rejected(change):
    station, initial = passive()
    with pytest.raises((ValueError, TypeError)):
        station.advance(replace(initial, **change), 0.1)
