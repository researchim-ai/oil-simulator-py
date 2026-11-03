"""Physical relief flow, observed pump metrics, qualified trips and restart."""

import copy
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from check_controlled_liquid import reference_station
from check_liquid_protection import check_reference, relief_model, timer_model

from surface import (
    ConservationError,
    ControlledLiquidStation,
    EquipmentOperatingError,
    HydraulicConvergenceError,
    LiquidRelief,
    LiquidTransientNetwork,
    LiquidTrip,
    ProtectedLiquidState,
    ProtectedLiquidStation,
    inspect_liquid_pumps,
    run_protected_liquid_schedule,
)
from surface.operation import _digest


def low_flow_model(grace=0.02, delay=0.03):
    data = reference_station().to_dict()
    data["physics"]["bulk_modulus_pa"] = 5e7
    c = ControlledLiquidStation.from_dict(data)
    return ProtectedLiquidStation(
        c,
        [
            LiquidTrip(
                "minflow",
                "a",
                "pump_flow_m3_s",
                "below",
                0.0001,
                stop_pumps=("a",),
                startup_grace_s=grace,
                delay_s=delay,
                min_speed_ratio=0.6,
            )
        ],
        0.01,
    )


def pump_initial(model, speed=65.0):
    return model.initialize(
        {
            "physical_initial": {
                "pressures_pa": {"h": 550000, "t": 500000},
                "speeds_rad_s": {"a": speed},
            },
            "running_pumps": ["a"],
        }
    )


def test_independent_relief_blowdown_partial_lift_and_trip_time():
    result = check_reference()
    assert result["passed"], result
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "dp,expected", [(-100, 0), (0, 0), (10000, 0), (15000, 0.5), (20000, 1), (1e7, 1)]
)
def test_spring_lift_is_one_way_and_bounded(dp, expected):
    law = LiquidRelief(10000, 20000)
    assert law.opening(dp) == expected


def test_relief_mass_and_work_are_in_the_actual_network_ledger():
    model = relief_model()
    initial = model.initialize({"v": 600000})
    result = model.advance(initial, 0.1)
    after = result["properties"]
    before = model.inspect(initial)
    out = math.fsum(
        s["midpoint_branch_volume_rates_m3_s"]["relief"] * s["dt_s"] * 1000 for s in result["steps"]
    )
    assert out > 0
    assert before["mass_kg"]["v"] - after["mass_kg"]["v"] == pytest.approx(out, abs=1e-9)
    heat = math.fsum(s["friction_heat_j"] for s in result["steps"])
    assert before["acoustic_energy_j"] - after["acoustic_energy_j"] == pytest.approx(heat, abs=1e-7)
    assert all(s["relief_openings"]["relief"] == 1 for s in result["steps"])


@pytest.mark.parametrize("pressure", [490000, 500000, 510000])
def test_closed_or_reverse_relief_does_not_leak(pressure):
    model = relief_model()
    state = model.initialize({"v": pressure})
    result = model.advance(state, 0.02)
    assert result["state"]["pressures_pa"]["v"] == pressure
    assert all(s["midpoint_branch_volume_rates_m3_s"]["relief"] == 0 for s in result["steps"])


def test_full_open_valve_is_finite_capacity_not_a_pressure_clamp():
    model = relief_model()
    result = model.advance(model.initialize({"v": 1200000}), 0.001)
    assert result["state"]["pressures_pa"]["v"] > 1100000
    assert result["steps"][0]["relief_openings"]["relief"] == 1


@pytest.mark.parametrize("dt", [1.8935009897528264e-8, 1e-6, 1e-5])
def test_reopening_coasting_pump_has_a_nonsingular_initial_flow_guess(dt):
    data = json.loads((ROOT / "configs/surface_protected_liquid.json").read_text())
    physics = ProtectedLiquidStation.from_dict(data["system"]).controller.physics
    p = np.asarray([527138.41029926, 516695.19609969])
    q = np.asarray([0.00185696])
    w = np.asarray([74.75564578, 80.51245431, 90.27854159])
    boundary, valves = physics._controls()
    _, _, _, record = physics._step(
        p, q, w, dt, np.asarray([0.0, 0.0, 3.15127938]), boundary, valves
    )
    assert record["midpoint_branch_volume_rates_m3_s"]["p2"] > 0
    assert record["max_scaled_equation_residual"] < 1e-8
    assert abs(record["energy_residual_j"]) < 1e-7


def test_relief_opening_cannot_be_overridden_by_ordinary_controls():
    model = relief_model()
    with pytest.raises(ValueError, match="Pressure determines"):
        model.advance(model.initialize(), 0.01, valve_openings={"relief": 1})


def test_sensor_uses_dynamic_pressures_without_a_network_steady_solve(monkeypatch):
    physics = reference_station().physics
    state = physics.initialize({"h": 550000, "t": 500000}, speeds_rad_s={"a": 100, "b": 110})

    def forbidden(*a, **k):
        raise AssertionError("No steady network sensor")

    monkeypatch.setattr(physics.network, "solve", forbidden)
    sensor = inspect_liquid_pumps(physics, state)
    expected = math.sqrt((1000 * 9.80665 * 20 - 150000) / (1000 * 9.80665 * 1e6))
    assert sensor["a"]["flow_m3_s"] == pytest.approx(expected, abs=1e-13)
    assert sensor["a"]["npsh_margin_m"] is None


def test_sensor_rejects_endpoint_outside_pump_map():
    physics = reference_station().physics
    state = physics.initialize({"h": 410000, "t": 480000}, speeds_rad_s={"a": 100, "b": 110})
    with pytest.raises(EquipmentOperatingError, match="sensor flow outside"):
        inspect_liquid_pumps(physics, state)


def test_npsh_margin_is_available_head_minus_required_and_trips_a_valid_state():
    data = reference_station().to_dict()
    data["physics"]["bulk_modulus_pa"] = 5e7
    for p in data["physics"]["network"]["pumps"]:
        p.update(inlet_diameter_m=0.05, npsh_required_m=30.0)
    c = ControlledLiquidStation.from_dict(data)
    physical = c.physics.initialize({"h": 550000, "t": 500000}, speeds_rad_s={"a": 100, "b": 110})
    q = inspect_liquid_pumps(c.physics, physical)["a"]["flow_m3_s"]
    expected = (
        (400000 - 2000) / (1000 * 9.80665) + (q / (math.pi * 0.05**2 / 4)) ** 2 / (2 * 9.80665) - 30
    )
    assert inspect_liquid_pumps(c.physics, physical)["a"]["npsh_margin_m"] == pytest.approx(
        expected
    )
    p = ProtectedLiquidStation(
        c, [LiquidTrip("npsh", "a", "npsh_margin_m", "below", 20, stop_pumps=("a",))]
    )
    state = p.initialize({"physical": physical, "running_pumps": ["a", "b"]})
    result = p.advance(state, 0.005, mode="torque", motor_torques_nm={"a": 0, "b": 0})
    assert result["latched_trips"] == ["npsh"]
    assert result["events"][0]["time_s"] == 0
    assert all(s["motor_torques_nm"]["a"] == 0 for s in result["steps"])
    assert result["state"]["controlled"]["physical"]["speeds_rad_s"]["a"] > 99


def test_startup_grace_and_qualifying_delay_trigger_at_known_clock_time():
    model = low_flow_model()
    result = model.advance(pump_initial(model), 0.07, mode="torque", motor_torques_nm={"a": 0})
    events = [e for e in result["events"] if e["type"] == "trip"]
    assert len(events) == 1 and events[0]["time_s"] == pytest.approx(0.05)
    assert result["state"]["controlled"]["physical"]["speeds_rad_s"]["a"] > 60
    assert result["state"]["controlled"]["physical"]["unavailable"] == ["a"]


def test_unarmed_stopped_pump_does_not_raise_minflow_trip():
    model = low_flow_model(grace=0, delay=0)
    result = model.advance(pump_initial(model, 0), 0.04, mode="torque", motor_torques_nm={"a": 0})
    assert result["latched_trips"] == []
    assert all(not s["rules"]["minflow"]["armed"] for s in result["protection_samples"])


def test_delay_restarts_when_condition_clears():
    model = timer_model()
    first = model.advance(model.initialize(), 0.02, mode="torque")
    clear = model.advance(
        ProtectedLiquidState.from_dict(first["state"]),
        0.02,
        mode="torque",
        boundary_pressures_pa={"in": 300000},
    )
    assert clear["latched_trips"] == []
    last = model.advance(ProtectedLiquidState.from_dict(clear["state"]), 0.04, mode="torque")
    assert [e["time_s"] for e in last["events"] if e["type"] == "trip"] == pytest.approx([0.07])


def test_unsafe_reset_is_denied_and_hysteresis_is_respected():
    model = timer_model()
    first = model.advance(model.initialize(), 0.04, mode="torque")
    unsafe = model.advance(
        ProtectedLiquidState.from_dict(first["state"]),
        0.005,
        mode="torque",
        reset_trips=["high-inlet"],
    )
    assert unsafe["latched_trips"] == ["high-inlet"]
    assert unsafe["events"][0]["type"] == "reset_denied"
    band = model.advance(
        ProtectedLiquidState.from_dict(unsafe["state"]),
        0.005,
        mode="torque",
        reset_trips=["high-inlet"],
        boundary_pressures_pa={"in": 345000},
    )
    assert band["events"][0]["type"] == "reset_denied"
    safe = model.advance(
        ProtectedLiquidState.from_dict(band["state"]),
        0.005,
        mode="off",
        reset_trips=["high-inlet"],
        boundary_pressures_pa={"in": 330000},
    )
    assert safe["latched_trips"] == []
    assert not safe["state"]["controlled"]["physical"]["tripped"]


def test_operator_and_protection_latches_reset_independently():
    model = timer_model()
    protected = model.advance(model.initialize(), 0.04, mode="torque")
    operator = model.advance(ProtectedLiquidState.from_dict(protected["state"]), 0.001, mode="trip")
    guard_reset = model.advance(
        ProtectedLiquidState.from_dict(operator["state"]),
        0.002,
        mode="off",
        reset_trips=["high-inlet"],
        boundary_pressures_pa={"in": 300000},
    )
    assert guard_reset["latched_trips"] == [] and guard_reset["state"]["operator_tripped"]
    assert guard_reset["state"]["controlled"]["physical"]["tripped"]
    reset = model.advance(
        ProtectedLiquidState.from_dict(guard_reset["state"]),
        0.001,
        mode="off",
        reset_trip=True,
        boundary_pressures_pa={"in": 300000},
    )
    assert not reset["state"]["controlled"]["physical"]["tripped"]


def test_resetting_a_pump_trip_does_not_clear_external_unavailability():
    data = timer_model().to_dict()
    data["rules"][0].update(trip_station=False, stop_pumps=["a"], delay_s=0)
    model = ProtectedLiquidStation.from_dict(data)
    first = model.advance(model.initialize(), 0.01, mode="torque", unavailable=["a"])
    reset = model.advance(
        ProtectedLiquidState.from_dict(first["state"]),
        0.005,
        mode="off",
        reset_trips=["high-inlet"],
        boundary_pressures_pa={"in": 300000},
    )
    assert reset["latched_trips"] == []
    assert reset["state"]["external_unavailable"] == ["a"]
    assert reset["state"]["controlled"]["physical"]["unavailable"] == ["a"]


def test_report_partitions_preserve_guard_clock_delay_and_controller_ticks():
    model = timer_model()
    start = model.initialize()
    full = model.advance(start, 0.08, mode="torque")
    state = start
    events = []
    samples = []
    pids = []
    for dt in (0.013, 0.017, 0.05):
        r = model.advance(state, dt, mode="torque")
        state = ProtectedLiquidState.from_dict(r["state"])
        events.extend(r["events"])
        samples.extend(r["protection_samples"])
        pids.extend(r["control_samples"])
    assert [s["tick"] for s in samples] == [s["tick"] for s in full["protection_samples"]]
    assert [s["tick"] for s in pids] == [s["tick"] for s in full["control_samples"]]
    assert [e["time_s"] for e in events] == pytest.approx([0.03])
    assert dict(state.controlled.physical.pressures_pa) == pytest.approx(
        full["state"]["controlled"]["physical"]["pressures_pa"], abs=0.1
    )


def test_roundoff_tail_does_not_consume_an_extra_protection_tick():
    model = timer_model()
    first = model.advance(model.initialize(), math.nextafter(0.01, math.inf), mode="torque")
    assert [s["tick"] for s in first["protection_samples"]] == [0]
    second = model.advance(ProtectedLiquidState.from_dict(first["state"]), 0.001, mode="torque")
    assert [s["tick"] for s in second["protection_samples"]] == [1]


def test_control_budget_can_be_exhausted_when_remaining_interval_has_no_pid_tick():
    model = timer_model()
    result = model.advance(model.initialize(), 0.015, mode="torque", max_control_samples=1)
    assert len(result["control_samples"]) == 1
    with pytest.raises(HydraulicConvergenceError, match="sample budget"):
        model.advance(model.initialize(), 0.025, mode="torque", max_control_samples=1)


def test_protection_work_budget_and_failures_leave_initial_state_unchanged():
    model = timer_model()
    state = model.initialize()
    snapshot = state.to_dict()
    with pytest.raises(HydraulicConvergenceError, match="sample budget"):
        model.advance(state, 0.02, mode="torque", max_protection_samples=1)
    assert state.to_dict() == snapshot


def test_checkpoint_keeps_partial_delay_both_clocks_latches_and_action_sources():
    model = timer_model()
    initial = model.initialize()
    periods = [{"duration_s": 0.025, "mode": "torque"}, {"duration_s": 0.015, "mode": "torque"}]
    full = run_protected_liquid_schedule(model, initial, periods)
    part = run_protected_liquid_schedule(model, initial, periods, stop_after=1)
    assert part["state"]["trips"]["high-inlet"]["violating_since_s"] == 0
    resumed = run_protected_liquid_schedule(model, initial, periods, json.loads(json.dumps(part)))
    assert resumed == full
    for field, value in [("operator_tripped", True), ("next_tick", 99)]:
        bad = copy.deepcopy(part)
        bad["state"][field] = value
        bad["checksum"] = _digest({k: v for k, v in bad.items() if k != "checksum"})
        with pytest.raises(ConservationError, match="memory"):
            run_protected_liquid_schedule(model, initial, periods, bad)
    bad = copy.deepcopy(part)
    bad["records"][0]["protection_samples"][0]["rules"]["high-inlet"]["value"] += 1
    bad["checksum"] = _digest({k: v for k, v in bad.items() if k != "checksum"})
    with pytest.raises(ConservationError, match="replay"):
        run_protected_liquid_schedule(model, initial, periods, bad)


def test_model_json_roundtrip_and_corrupt_source_memory_rejection():
    model = timer_model()
    state = model.initialize()
    assert (
        ProtectedLiquidStation.from_dict(json.loads(json.dumps(model.to_dict()))).model_id
        == model.model_id
    )
    assert ProtectedLiquidState.from_dict(json.loads(json.dumps(state.to_dict()))) == state
    with pytest.raises(ValueError, match="provenance"):
        model.advance(replace(state, operator_tripped=True), 0.01, mode="torque")


@pytest.mark.parametrize(
    "changes",
    [
        {"equipment": "missing"},
        {"stop_pumps": ["unknown"], "trip_station": False},
        {"threshold": 1e9},
        {"name": " "},
        {"delay_s": -1},
    ],
)
def test_invalid_rules_are_rejected(changes):
    data = timer_model().to_dict()
    data["rules"][0].update(changes)
    with pytest.raises((ValueError, TypeError)):
        ProtectedLiquidStation.from_dict(data)


def test_npsh_rule_requires_configured_inlet_and_required_head():
    with pytest.raises(ValueError, match="NPSH margin protection"):
        ProtectedLiquidStation(
            reference_station(),
            [LiquidTrip("npsh", "a", "npsh_margin_m", "below", 2, stop_pumps=("a",))],
        )


def test_invalid_relief_mapping_and_manual_nominal_opening_are_rejected():
    data = relief_model().to_dict()
    data["relief_valves"] = {"unknown": {"cracking_pressure_pa": 1, "full_open_pressure_pa": 2}}
    with pytest.raises(ValueError):
        LiquidTransientNetwork.from_dict(data)
    data = relief_model().to_dict()
    data["network"]["valves"][0]["opening"] = 1
    with pytest.raises(ValueError, match="nominal opening"):
        LiquidTransientNetwork.from_dict(data)


def test_cli_error_does_not_replace_existing_output(tmp_path):
    config = tmp_path / "bad.json"
    out = tmp_path / "out.json"
    out.write_text("preserve")
    config.write_text(
        json.dumps(
            {"system": timer_model().to_dict(), "initial": {}, "periods": [{"duration_s": -1}]}
        )
    )
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_protected_liquid.py"),
            str(config),
            "--output",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 2 and out.read_text() == "preserve"
