"""Persistent control clock, dynamic feedback, actuator tracking and restart."""

import copy
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from check_controlled_liquid import check_reference, reference_station

from surface import (
    ConservationError,
    ControlledLiquidState,
    ControlledLiquidStation,
    HydraulicConvergenceError,
    PIDController,
    run_controlled_liquid_schedule,
)
from surface.operation import _digest


def example(**updates):
    data = json.loads((ROOT / "configs/surface_controlled_liquid.json").read_text())["system"]
    data.update(updates)
    return ControlledLiquidStation.from_dict(data)


def initial(model):
    return model.initialize(
        physical_initial={
            "pressures_pa": {"h": 550000, "t": 500000},
            "pipe_volume_rates_m3_s": {"line": 0.002},
            "speeds_rad_s": {"a": 100, "b": 110},
        }
    )


def steady_initial(model, names=("p1", "p2"), ratio=1.2):
    network = model.physics.network
    result = network.solve(
        pump_speed_ratios={n: ratio for n in names},
        pump_status={n: n in names for n in model.names},
    )
    torque = {
        n: result.pump_shaft_power_w[n] / (ratio * 100)
        + 0.005 * ratio * 100
        + model.physics.no_flow_power_w_at_nominal[n] * ratio**3 / (ratio * 100)
        for n in names
    }
    physical = model.physics.initialize(
        {n: result.pressures_pa[n] for n in model.physics.free},
        result.pipe_volume_rates_m3_s,
        {n: ratio * 100 if n in names else 0.0 for n in model.names},
    )
    return model.initialize(
        physical=physical,
        running_pumps=names,
        applied_torques_nm=torque,
        pressure_setpoint_pa=650000,
    )


def audit(model, start, result):
    assert math.fsum(s["dt_s"] for s in result["steps"]) == pytest.approx(
        result["duration_s"], abs=1e-10
    )
    assert max(abs(s["energy_residual_j"]) for s in result["steps"]) < 1e-4
    assert max(s["max_node_mass_residual_kg"] for s in result["steps"]) < 1e-8
    before = model.physics.inspect(start.physical)
    after = result["properties"]
    supplied = math.fsum(
        v for s in result["steps"] for v in s["boundary_mass_transfers_in_kg"].values()
    )
    assert sum(after["mass_kg"].values()) - sum(before["mass_kg"].values()) == pytest.approx(
        supplied, abs=1e-8
    )


def test_independent_sampled_cascade_and_vector_ode_refinement():
    result = check_reference()
    assert result["passed"], result
    json.dumps(result, allow_nan=False)


def test_pressure_feedback_comes_from_inventory_not_steady_dispatch_target():
    model = example()
    state = model.initialize()
    result = model.advance(state, 0.05)
    sample = result["control_samples"][0]
    assert sample["measured_pressure_pa"] == 500000
    assert sample["selected"] == ["p1", "p2"]
    assert result["state"]["physical"]["pressures_pa"]["h"] != 650000
    assert result["state"]["physical"]["speeds_rad_s"] == {"p1": 0, "p2": 0, "p3": 0}
    audit(model, state, result)


def test_report_partition_does_not_resample_or_reset_the_controller_clock():
    model = reference_station(0.001)
    state = initial(model)
    whole = model.advance(state, 0.08, pressure_setpoint_pa=570000)
    samples = []
    for duration in (0.0073, 0.0197, 0.0111, 0.0419):
        result = model.advance(state, duration, pressure_setpoint_pa=570000)
        samples.extend(result["control_samples"])
        state = ControlledLiquidState.from_dict(result["state"])
    assert [s["tick"] for s in samples] == [0, 1, 2, 3]
    assert [s["time_s"] for s in samples] == pytest.approx([0, 0.02, 0.04, 0.06], abs=1e-13)
    assert state.next_tick == whole["state"]["next_tick"]
    assert dict(state.physical.pressures_pa) == pytest.approx(
        whole["state"]["physical"]["pressures_pa"], abs=0.5
    )
    assert dict(state.physical.speeds_rad_s) == pytest.approx(
        whole["state"]["physical"]["speeds_rad_s"], abs=1e-5
    )


def test_exact_tick_at_report_end_is_processed_once_in_next_report():
    model = reference_station()
    first = model.advance(initial(model), 0.02, pressure_setpoint_pa=570000)
    assert [s["tick"] for s in first["control_samples"]] == [0]
    second = model.advance(
        ControlledLiquidState.from_dict(first["state"]), 0.001, pressure_setpoint_pa=570000
    )
    assert [s["tick"] for s in second["control_samples"]] == [1]
    assert second["control_samples"][0]["feedback_duration_s"] == pytest.approx(0.02)


def test_sub_ulp_report_tail_does_not_consume_next_tick_with_old_commands():
    model = reference_station()
    first = model.advance(
        initial(model),
        math.nextafter(0.02, math.inf),
        mode="torque",
        motor_torques_nm={"a": 5, "b": 7},
    )
    assert [s["tick"] for s in first["control_samples"]] == [0]
    second = model.advance(
        ControlledLiquidState.from_dict(first["state"]),
        0.001,
        mode="torque",
        motor_torques_nm={"a": 9, "b": 7},
    )
    assert [s["tick"] for s in second["control_samples"]] == [1]
    assert second["control_samples"][0]["motor_commands_nm"]["a"] == 9


def test_auto_enforces_unit_count_limits_even_during_manual_stage_dwell():
    data = example(minimum_stage_dwell_s=100).to_dict()
    data["dispatch_policy"]["max_running"] = 2
    model = ControlledLiquidStation.from_dict(data)
    state = steady_initial(model, names=("p1", "p2", "p3"), ratio=1.2)
    result = model.advance(state, 0.01)
    assert len(result["state"]["selected"]) == 2
    assert result["dispatches"][0]["time_s"] == 0


def test_actual_applied_torque_is_accumulated_across_partial_reports():
    model = reference_station()
    state = initial(model)
    first = model.advance(state, 0.013, mode="torque", motor_torques_nm={"a": 5, "b": 7})
    middle = ControlledLiquidState.from_dict(first["state"])
    assert dict(middle.torque_integrals_nm_s) == pytest.approx({"a": 0.065, "b": 0.091})
    next_part = model.advance(middle, 0.008, mode="torque", motor_torques_nm={"a": 5, "b": 7})
    sample = next_part["control_samples"][0]
    assert sample["time_s"] == pytest.approx(0.02)
    assert sample["applied_torques_nm"] == pytest.approx({"a": 5, "b": 7})
    assert sample["feedback_duration_s"] == pytest.approx(0.02)


def test_power_limited_antiwindup_tracks_zero_applied_torque():
    data = reference_station().to_dict()
    data["physics"]["max_station_electrical_draw_w"] = 0
    model = ControlledLiquidStation.from_dict(data)
    state = initial(model)
    result = model.advance(state, 0.08, pressure_setpoint_pa=700000)
    assert all(
        abs(v) < 1e-12 for s in result["control_samples"] for v in s["applied_torques_nm"].values()
    )
    assert any(v > 0 for s in result["control_samples"] for v in s["motor_commands_nm"].values())
    assert all(s["peak_electrical_draw_w"] == 0 for s in result["steps"])
    audit(model, state, result)


def test_physical_retry_does_not_repeat_pid_integrations(monkeypatch):
    model = reference_station()
    original = model.physics._step
    called = []

    def retry(*args, **kwargs):
        called.append(1)
        if len(called) == 1:
            raise HydraulicConvergenceError("injected physical retry")
        return original(*args, **kwargs)

    monkeypatch.setattr(model.physics, "_step", retry)
    result = model.advance(initial(model), 0.05, pressure_setpoint_pa=570000)
    assert [s["tick"] for s in result["control_samples"]] == [0, 1, 2]
    assert result["retry_reasons"]["injected physical retry"] == 1


def test_pid_step_always_uses_fixed_digital_sample_duration(monkeypatch):
    model = reference_station()
    original = PIDController.step
    durations = []

    def observe(self, *args, **kwargs):
        durations.append(args[3])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PIDController, "step", observe)
    state = initial(model)
    for dt in (0.013, 0.011, 0.021):
        result = model.advance(state, dt, pressure_setpoint_pa=570000)
        state = ControlledLiquidState.from_dict(result["state"])
    assert durations == [0.02] * 9  # Three clock ticks, three PIDs each.


def test_failure_cuts_motor_between_ticks_without_advancing_pid_memory():
    model = reference_station()
    first = model.advance(initial(model), 0.007, mode="torque", motor_torques_nm={"a": 5, "b": 7})
    state = ControlledLiquidState.from_dict(first["state"])
    failed = model.advance(
        state, 0.003, mode="torque", motor_torques_nm={"a": 5, "b": 7}, unavailable=["a"]
    )
    assert failed["control_samples"] == []
    assert failed["state"]["speed_pids"] == first["state"]["speed_pids"]
    assert all(
        s["motor_torques_nm"]["a"] == 0 and s["motor_torques_nm"]["b"] == 7 for s in failed["steps"]
    )
    assert failed["state"]["physical"]["speeds_rad_s"]["a"] > 99


def test_reserve_selection_at_next_tick_overrides_dwell_but_return_does_not():
    model = example(minimum_stage_dwell_s=100)
    first = model.advance(steady_initial(model), 0.007)
    failed = model.advance(
        ControlledLiquidState.from_dict(first["state"]), 0.06, unavailable=["p1"]
    )
    assert failed["dispatches"][0]["time_s"] == pytest.approx(0.05)
    assert failed["state"]["selected"] == ["p2", "p3"]
    assert failed["state"]["physical"]["speeds_rad_s"]["p1"] > 100
    assert failed["steps"][0]["midpoint_branch_volume_rates_m3_s"]["p1"] > 0
    inherited = model.advance(ControlledLiquidState.from_dict(failed["state"]), 0.01)
    assert inherited["state"]["physical"]["unavailable"] == ["p1"]
    restored = model.advance(
        ControlledLiquidState.from_dict(inherited["state"]), 0.04, unavailable=[]
    )
    assert restored["state"]["selected"] == ["p2", "p3"] and restored["dispatches"] == []


def test_sealed_outlet_uses_dynamic_feedback_without_fake_steady_pressure(monkeypatch):
    model = example(minimum_stage_dwell_s=0)
    state = steady_initial(model)

    def forbidden(*args, **kwargs):
        raise AssertionError("Sealed topology must not ask for a steady target")

    monkeypatch.setattr(model.physics.network, "solve", forbidden)
    result = model.advance(state, 0.02, valve_openings={"export": 0})
    assert result["state"]["dispatch_status"] == "dynamic_feedback_only"
    assert (
        result["control_samples"][0]["measured_pressure_pa"]
        == dict(state.physical.pressures_pa)["h"]
    )
    assert all(
        abs(s["midpoint_branch_volume_rates_m3_s"]["export"]) < 1e-10 for s in result["steps"]
    )
    audit(model, state, result)


def test_planning_numerical_failure_is_not_hidden_as_infeasible_target(monkeypatch):
    model = example()

    def fail(*args, **kwargs):
        raise HydraulicConvergenceError("broken steady candidate")

    monkeypatch.setattr(model._planner, "_dispatch", fail)
    with pytest.raises(HydraulicConvergenceError, match="broken steady"):
        model.advance(model.initialize(), 0.01)


def test_unattainable_target_is_explicit_and_does_not_expand_hardware():
    model = example()
    result = model.advance(model.initialize(), 0.1, pressure_setpoint_pa=900000)
    assert result["state"]["dispatch_status"] == "target_infeasible"
    assert result["pressure_error_pa"] > 300000
    assert max(result["state"]["physical"]["speeds_rad_s"].values()) <= 130


def test_trip_and_reset_between_ticks_preserve_fixed_clock_and_zero_held_output():
    model = reference_station()
    first = model.advance(initial(model), 0.007, mode="torque", motor_torques_nm={"a": 5, "b": 7})
    trip = model.advance(ControlledLiquidState.from_dict(first["state"]), 0.003, mode="trip")
    assert trip["state"]["physical"]["tripped"] and trip["control_samples"] == []
    latched = model.advance(
        ControlledLiquidState.from_dict(trip["state"]), 0.003, pressure_setpoint_pa=570000
    )
    assert latched["effective_mode"] == "off"
    reset = model.advance(
        ControlledLiquidState.from_dict(latched["state"]),
        0.003,
        mode="torque",
        reset_trip=True,
        motor_torques_nm={"a": 5, "b": 7},
    )
    assert not reset["state"]["physical"]["tripped"]
    assert reset["state"]["next_tick"] == 1
    assert all(s["electrical_energy_j"] == 0 for s in reset["steps"])


def test_unit_order_does_not_reset_switch_time():
    model = example(minimum_stage_dwell_s=0)
    state = steady_initial(model, names=("p2", "p1"))
    result = model.advance(state, 0.01)
    assert set(result["state"]["selected"]) == {"p1", "p2"}
    assert result["state"]["last_switch_time_s"] == 0


def test_pending_explicit_redispatch_survives_a_report_with_no_sample():
    model = example()
    first = model.advance(model.initialize(), 0.01)
    pending = model.advance(ControlledLiquidState.from_dict(first["state"]), 0.01, redispatch=True)
    assert pending["state"]["pending_redispatch"]
    assert pending["control_samples"] == []
    # Dwell may defer it further: it must remain pending until eligible.
    next_report = model.advance(ControlledLiquidState.from_dict(pending["state"]), 0.04)
    assert next_report["state"]["pending_redispatch"] and not next_report["dispatches"]


def test_checkpoint_in_middle_of_sample_preserves_clock_tracking_and_pid_history():
    model = reference_station()
    state = initial(model)
    periods = [{"duration_s": dt, "pressure_setpoint_pa": 570000} for dt in (0.013, 0.017, 0.035)]
    full = run_controlled_liquid_schedule(model, state, periods)
    part = run_controlled_liquid_schedule(model, state, periods, stop_after=1)
    assert part["state"]["feedback_duration_s"] == pytest.approx(0.013)
    resumed = run_controlled_liquid_schedule(model, state, periods, json.loads(json.dumps(part)))
    assert resumed == full
    for field, value in (("next_tick", 2), ("feedback_duration_s", 0.01)):
        corrupt = copy.deepcopy(part)
        corrupt["state"][field] = value
        corrupt["checksum"] = _digest({k: v for k, v in corrupt.items() if k != "checksum"})
        with pytest.raises(ConservationError, match="memory"):
            run_controlled_liquid_schedule(model, state, periods, corrupt)
    corrupt = copy.deepcopy(part)
    corrupt["records"][0]["control_samples"][0]["motor_commands_nm"]["a"] += 1
    corrupt["checksum"] = _digest({k: v for k, v in corrupt.items() if k != "checksum"})
    with pytest.raises(ConservationError, match="replay"):
        run_controlled_liquid_schedule(model, state, periods, corrupt)


def test_json_model_and_state_roundtrip():
    model = reference_station()
    assert (
        ControlledLiquidStation.from_dict(json.loads(json.dumps(model.to_dict()))).model_id
        == model.model_id
    )
    state = initial(model)
    assert ControlledLiquidState.from_dict(json.loads(json.dumps(state.to_dict()))) == state
    with pytest.raises(ValueError, match="clock"):
        model.advance(replace(state, next_tick=3), 0.001, pressure_setpoint_pa=570000)


@pytest.mark.parametrize(
    "updates",
    [
        {"control_interval_s": 0},
        {"pressure_node": "in"},
        {"speed_controllers": {}},
        {"minimum_stage_dwell_s": -1},
    ],
)
def test_invalid_controller_configuration_is_rejected(updates):
    data = reference_station().to_dict()
    data.update(updates)
    with pytest.raises((ValueError, TypeError)):
        ControlledLiquidStation.from_dict(data)


@pytest.mark.parametrize(
    "args",
    [
        {"mode": "unknown"},
        {"mode": "trip", "reset_trip": True},
        {"max_control_samples": True},
        {"unavailable": ["unknown"]},
        {"speed_ratio": 2},
    ],
)
def test_invalid_control_commands_are_rejected(args):
    model = reference_station()
    with pytest.raises((ValueError, TypeError)):
        model.advance(initial(model), 0.01, pressure_setpoint_pa=570000, **args)


def test_control_and_physical_work_budgets_fail_without_modifying_initial_state():
    model = reference_station()
    state = initial(model)
    snapshot = state.to_dict()
    with pytest.raises(HydraulicConvergenceError, match="sample budget"):
        model.advance(state, 0.05, pressure_setpoint_pa=570000, max_control_samples=1)
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        model.advance(state, 0.05, pressure_setpoint_pa=570000, max_substeps=1)
    assert state.to_dict() == snapshot


def test_cli_failure_preserves_previous_output(tmp_path):
    model = reference_station()
    config = tmp_path / "bad.json"
    config.write_text(
        json.dumps({"system": model.to_dict(), "initial": {}, "periods": [{"duration_s": -1}]})
    )
    output = tmp_path / "out.json"
    output.write_text("keep")
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_controlled_liquid.py"),
            str(config),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 2 and output.read_text() == "keep"
