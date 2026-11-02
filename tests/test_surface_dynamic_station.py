"""Coupled shafts, station limits, dispatch, stopping and restart integrity."""

import copy
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from check_dynamic_station import check_reference, independent_hydraulics, reference_model

from surface import (
    ConservationError,
    DynamicPumpNetwork,
    DynamicPumpStation,
    HydraulicConvergenceError,
    PumpStationState,
    run_dynamic_station_schedule,
)
from surface.operation import _digest


def station(**updates):
    data = json.loads((ROOT / "configs/surface_dynamic_station.json").read_text())["system"]
    data.update(updates)
    return DynamicPumpStation.from_dict(data)


def audit(result):
    for step in result["steps"]:
        assert abs(step["energy_residual_j"]) < 1e-4
        assert abs(math.fsum(step["boundary_mass_transfers_kg"].values())) < 1e-8
        for u in step["units"].values():
            assert abs(u["shaft_energy_residual_j"]) < 1e-5
            assert min(u[k] for k in ("motor_heat_j", "drag_heat_j", "pump_heat_j")) >= -1e-8
            assert u["final_speed_rad_s"] >= 0
    assert math.fsum(s["dt_s"] for s in result["steps"]) == pytest.approx(
        result["duration_s"], abs=1e-11
    )
    assert math.fsum(s["dt_s"] for s in result["control_samples"]) == pytest.approx(
        result["duration_s"], abs=1e-11
    )


def test_independent_heterogeneous_vector_ode_and_second_order_refinement():
    result = check_reference()
    assert result["passed"], result
    json.dumps(result, allow_nan=False)


def test_cross_shaft_coupling_matches_independent_common_header():
    model = reference_model()
    # Faster b can close a's check valve without stopping a's shaft.
    omega = np.asarray((60.0, 110.0))
    header, q, _ = independent_hydraulics(omega)
    result = model.advance(
        model.initialize(dict(zip(model.names, omega))),
        0.2,
        mode="torque",
        motor_torques_nm={"a": 0, "b": 0},
    )
    audit(result)
    assert q[0] == 0
    assert result["steps"][0]["midpoint_pump_flows_m3_s"]["a"] == 0
    assert result["state"]["speeds_rad_s"]["a"] > 0
    changed, _, _ = independent_hydraulics(np.asarray((60.0, 90.0)))
    assert abs(header - changed) > 10000


def test_single_unit_reduces_to_existing_coupled_pump_model():
    model = reference_model(0.05)
    net = type(model.network)(
        model.network.fluid, model.network.nodes, model.network.pipes, [model.network.pumps[0]]
    )
    drive = model.drives["a"]
    multi = DynamicPumpStation(
        net,
        {"a": drive},
        {"a": 100},
        pressure_node="header",
        maximum_dt_s=0.05,
        max_speed_change_rad_s=10,
    )
    single = DynamicPumpNetwork(
        net, drive, 100, measurement_node="header", maximum_dt_s=0.05, max_speed_change_rad_s=10
    )
    a = multi.advance(multi.initialize({"a": 90}), 1, mode="torque", motor_torques_nm={"a": 6})
    b = single.advance(single.initialize(90), 1, mode="manual", motor_torque_nm=6)
    assert a["state"]["speeds_rad_s"]["a"] == pytest.approx(
        b["state"]["angular_velocity_rad_s"], abs=1e-7
    )
    audit(a)


def test_independent_stopping_events_share_clock_and_do_not_invent_export():
    model = reference_model(0.2)
    # Below opening pressure, no hydraulic load, disable drag and P0.
    model.drives = {n: replace(d, viscous_drag_nm_s=0) for n, d in model.drives.items()}
    model.no_flow_power_w_at_nominal = {n: 0 for n in model.names}
    result = model.advance(
        model.initialize({"a": 10, "b": 20}), 5, mode="torque", motor_torques_nm={"a": -4, "b": -8}
    )
    audit(result)
    assert result["state"]["speeds_rad_s"] == {"a": 0, "b": 0}
    assert sum(s["electrical_energy_j"] for s in result["steps"]) == pytest.approx(
        -0.7 * (0.5 * 10**2 + 0.5 * 1.4 * 20**2), abs=1e-7
    )
    assert any(
        s["units"]["a"]["final_speed_rad_s"] == 0 and s["units"]["b"]["final_speed_rad_s"] > 0
        for s in result["steps"]
    )
    resting = result["steps"][-1]
    assert resting["electrical_energy_j"] == 0
    assert resting["units"]["a"]["rest_constraint_impulse_nm_s"] > 0


def test_weak_acceleration_has_no_artificial_zero_speed_deadband():
    model = reference_model()
    result = model.advance(model.initialize(), 0.1, mode="torque", motor_torques_nm={"a": 2e-8})
    assert 0 < result["state"]["speeds_rad_s"]["a"] < 1e-8


@pytest.mark.parametrize("capacity", [0.0, 100.0, 500.0])
def test_group_gross_draw_limit_and_conservative_peak_envelope(capacity):
    model = reference_model(0.1)
    model.max_station_electrical_draw_w = capacity
    result = model.advance(
        model.initialize({"a": 90, "b": 100}),
        0.6,
        mode="torque",
        motor_torques_nm={"a": 20, "b": 20},
    )
    audit(result)
    assert all(s["peak_electrical_draw_w"] <= capacity + 1e-7 for s in result["steps"])
    if capacity == 0:
        assert all(s["electrical_energy_j"] == 0 for s in result["steps"])
    else:
        assert any(s["draw_allocation_factor"] < 1 for s in result["steps"])


def test_import_is_not_offset_by_other_shaft_regeneration():
    model = reference_model()
    model.max_station_electrical_draw_w = 100
    model.max_station_electrical_export_w = 50
    result = model.advance(
        model.initialize({"a": 90, "b": 100}),
        0.4,
        mode="torque",
        motor_torques_nm={"a": 20, "b": -10},
    )
    audit(result)
    assert all(
        s["peak_electrical_draw_w"] <= 100 + 1e-8 and s["peak_electrical_export_w"] <= 50 + 1e-8
        for s in result["steps"]
    )


@pytest.mark.parametrize("regeneration", [0.0, 0.7])
def test_zero_export_budget_keeps_dissipative_brake_but_blocks_regenerative_drive(regeneration):
    model = reference_model()
    model.drives = {
        n: replace(d, regeneration_efficiency=regeneration) for n, d in model.drives.items()
    }
    model.max_station_electrical_export_w = 0
    result = model.advance(
        model.initialize({"a": 20, "b": 20}),
        0.4,
        mode="torque",
        motor_torques_nm={"a": -8, "b": -8},
    )
    audit(result)
    assert all(s["peak_electrical_export_w"] == 0 for s in result["steps"])
    assert result["steps"][0]["units"]["a"]["motor_torque_nm"] == (-8 if regeneration == 0 else 0)


def test_local_drive_power_and_speed_governor():
    model = reference_model()
    model.drives["a"] = replace(model.drives["a"], max_speed_rad_s=90.1, max_electrical_draw_w=1000)
    result = model.advance(
        model.initialize({"a": 90, "b": 100}), 1, mode="torque", motor_torques_nm={"a": 20, "b": 20}
    )
    audit(result)
    assert result["state"]["speeds_rad_s"]["a"] <= 90.1
    assert all(
        s["units"]["a"]["motor_torque_nm"]
        * max(s["units"]["a"]["initial_speed_rad_s"], s["units"]["a"]["final_speed_rad_s"])
        / 0.9
        <= 1000 + 1e-8
        for s in result["steps"]
    )


def test_dispatch_prices_full_electrical_cost_not_only_pump_shaft_work():
    model = station()
    # p1 is hydraulically identical but has an inefficient motor.
    model.drives["p1"] = replace(model.drives["p1"], motor_efficiency=0.5)
    selected, _, plan = model._dispatch(400000, (), (), {})
    assert selected == ("p2", "p3")
    assert plan["steady_electrical_power_w"] <= 1000
    selected, _, retained = model._dispatch(400000, (), ("p3", "p2"), {})
    assert set(selected) == {"p2", "p3"}
    assert retained["status"] == "retained_by_electrical_deadband"


def test_failure_overrides_stage_dwell_and_failed_pump_coasts_physically():
    model = station(minimum_stage_dwell_s=100)
    first = model.advance(model.initialize({"p1": 104, "p2": 104}), 0.2)
    fail = model.advance(PumpStationState.from_dict(first["state"]), 0.4, unavailable=["p1"])
    audit(fail)
    assert fail["state"]["target_running"] == ["p2", "p3"]
    assert fail["dispatches"][0]["time_s"] == 0.2
    assert fail["state"]["speeds_rad_s"]["p1"] > 0
    assert fail["steps"][0]["midpoint_pump_flows_m3_s"]["p1"] > 0
    assert all(s["units"]["p1"]["motor_torque_nm"] == 0 for s in fail["steps"])
    inherited = model.advance(PumpStationState.from_dict(fail["state"]), 0.2)
    assert inherited["state"]["unavailable"] == ["p1"]
    restored = model.advance(PumpStationState.from_dict(inherited["state"]), 0.2, unavailable=[])
    assert restored["state"]["unavailable"] == []
    assert (
        restored["dispatches"] == []
    )  # Dwell blocks elective switching, not availability updates.


def test_unattainable_target_is_reported_without_enlarging_hardware_maps():
    model = station()
    result = model.advance(model.initialize(), 0.4, pressure_setpoint_pa=900000)
    assert result["state"]["dispatch_status"] == "target_infeasible"
    assert result["pressure_error_pa"] > 500000
    assert all(v <= 110 for v in result["state"]["speeds_rad_s"].values())
    audit(result)


def test_closed_loop_start_and_reserve_recover_pressure_under_station_power_cap():
    model = station()
    start = model.advance(model.initialize(), 20)
    failure = model.advance(PumpStationState.from_dict(start["state"]), 20, unavailable=["p1"])
    for result, tolerance in ((start, 2000), (failure, 1000)):
        audit(result)
        assert abs(result["pressure_error_pa"]) < tolerance
        assert len(result["control_samples"]) == 100
        assert all(s["peak_electrical_draw_w"] <= 1000 + 1e-7 for s in result["steps"])
    assert start["state"]["target_running"] == ["p1", "p2"]
    assert failure["state"]["target_running"] == ["p2", "p3"]
    assert failure["state"]["speeds_rad_s"]["p1"] > 0  # Real coasting, not deletion.


def test_infeasible_fallback_respects_max_running_after_manual_operation():
    model = station()
    model.dispatch_policy = replace(model.dispatch_policy, max_running=2)
    chosen, _, plan = model._dispatch(900000, (), ("p1", "p2", "p3"), {})
    assert len(chosen) == 2 and plan["status"] == "target_infeasible"


def test_trip_latches_and_requires_explicit_reset():
    model = station()
    result = model.advance(model.initialize({"p1": 90}), 0.2, mode="trip")
    continued = model.advance(PumpStationState.from_dict(result["state"]), 0.2)
    assert continued["effective_mode"] == "off" and continued["state"]["tripped"]
    reset = model.advance(PumpStationState.from_dict(continued["state"]), 0.2, reset_trip=True)
    assert not reset["state"]["tripped"] and reset["effective_mode"] == "auto"


def test_reordering_same_units_does_not_reset_physical_stage_dwell_clock():
    model = station(minimum_stage_dwell_s=0)
    manual = model.advance(
        model.initialize(), 0.2, mode="manual", running_pumps=["p3", "p2"], speed_ratio=0
    )
    automatic = model.advance(PumpStationState.from_dict(manual["state"]), 0.2)
    assert set(automatic["state"]["target_running"]) == {"p2", "p3"}
    assert automatic["state"]["last_switch_time_s"] == manual["state"]["last_switch_time_s"]


def test_physical_retries_do_not_resample_pid(monkeypatch):
    model = station()
    original = model._step
    attempts = []

    def once(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise HydraulicConvergenceError("injected retry")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_step", once)
    result = model.advance(
        model.initialize(), 0.4, mode="manual", running_pumps=["p1"], speed_ratio=0.9
    )
    assert len(result["control_samples"]) == 2
    assert result["retry_reasons"]["injected retry"] == 1
    assert [s["time_s"] for s in result["control_samples"]] == [0.0, 0.2]
    audit(result)


def test_dispatch_does_not_disguise_hydraulic_failure_as_unattainable_target(monkeypatch):
    model = station()

    def fail(*args, **kwargs):
        raise HydraulicConvergenceError("broken network solve")

    monkeypatch.setattr(model.network, "solve", fail)
    with pytest.raises(HydraulicConvergenceError, match="broken network"):
        model._dispatch(400000, (), (), {})


def test_configuration_and_state_json_roundtrip():
    model = station()
    clone = DynamicPumpStation.from_dict(json.loads(json.dumps(model.to_dict())))
    assert clone.model_id == model.model_id
    initial = model.initialize({"p1": 70}, unavailable=["p3"])
    recovered = PumpStationState.from_dict(json.loads(json.dumps(initial.to_dict())))
    assert recovered == initial


def test_checkpoint_replays_full_physics_and_controller_memory():
    model = reference_model()
    initial = model.initialize({"a": 90, "b": 96})
    periods = [
        {"duration_s": 0.4, "mode": "torque", "motor_torques_nm": {"a": 6, "b": 8}},
        {"duration_s": 0.2, "mode": "trip"},
        {"duration_s": 0.2, "mode": "off"},
    ]
    full = run_dynamic_station_schedule(model, initial, periods)
    partial = run_dynamic_station_schedule(model, initial, periods, stop_after=1)
    resumed = run_dynamic_station_schedule(model, initial, periods, json.loads(json.dumps(partial)))
    assert resumed == full
    corrupt = copy.deepcopy(partial)
    corrupt["records"][0]["steps"][0]["units"]["a"]["motor_work_j"] += 1
    corrupt["checksum"] = _digest({k: v for k, v in corrupt.items() if k != "checksum"})
    with pytest.raises(ConservationError, match="replay"):
        run_dynamic_station_schedule(model, initial, periods, corrupt)
    wrong = copy.deepcopy(partial)
    wrong["state"]["unavailable"] = ["a"]
    wrong["checksum"] = _digest({k: v for k, v in wrong.items() if k != "checksum"})
    with pytest.raises(ConservationError, match="memory"):
        run_dynamic_station_schedule(model, initial, periods, wrong)


@pytest.mark.parametrize(
    "updates",
    [
        {"minimum_dt_s": 1, "maximum_dt_s": 0.1},
        {"max_station_electrical_draw_w": -1},
        {"speed_controllers": {"p1": None, "p2": None, "p3": None}},
    ],
)
def test_invalid_station_configuration_is_rejected(updates):
    with pytest.raises((ValueError, TypeError)):
        station(**updates)
