"""Acoustic inventories, coupled shafts, valve closure and source-bound replay."""

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from check_liquid_transient import check_reference, coupled_model, oscillator, oscillator_reference

from surface import (
    ConservationError,
    EquipmentOperatingError,
    HydraulicConvergenceError,
    LiquidTransientNetwork,
    LiquidTransientState,
    run_liquid_transient_schedule,
)
from surface.operation import _digest


def clone(model, **updates):
    data = json.loads(json.dumps(model.to_dict()))
    data.update(updates)
    return LiquidTransientNetwork.from_dict(data)


def coupled_initial(model):
    return model.initialize({"h": 550000, "t": 500000}, {"line": 0.002}, {"a": 100, "b": 110})


def audit(model, initial, result):
    for step in result["steps"]:
        assert abs(step["energy_residual_j"]) < 1e-4
        assert step["max_node_mass_residual_kg"] < 1e-8
        assert abs(step["mass_residual_kg"]) < 1e-8
        assert (
            max(abs(x) for x in step["unit_shaft_energy_residuals_j"].values()) < 1e-4
            if model.names
            else True
        )
        assert (
            min(
                step[k]
                for k in ("motor_heat_j", "shaft_drag_heat_j", "pump_heat_j", "friction_heat_j")
            )
            >= -1e-7
        )
    assert math.fsum(s["dt_s"] for s in result["steps"]) == pytest.approx(
        result["duration_s"], abs=1e-11
    )
    before = model.inspect(initial)
    after = result["properties"]
    transferred = math.fsum(
        v for s in result["steps"] for v in s["boundary_mass_transfers_in_kg"].values()
    )
    assert math.fsum(after["mass_kg"].values()) - math.fsum(
        before["mass_kg"].values()
    ) == pytest.approx(transferred, abs=1e-8)
    stored_change = math.fsum(
        after[k] - before[k]
        for k in ("acoustic_energy_j", "pipe_kinetic_energy_j", "shaft_kinetic_energy_j")
    )
    boundary_grid_heat = math.fsum(
        s["electrical_energy_j"]
        + s["boundary_work_in_j"]
        - s["motor_heat_j"]
        - s["shaft_drag_heat_j"]
        - s["pump_heat_j"]
        - s["friction_heat_j"]
        for s in result["steps"]
    )
    assert stored_change == pytest.approx(boundary_grid_heat, abs=1e-4)


def test_independent_temporal_spatial_and_coupled_ode_references():
    result = check_reference()
    assert result["passed"], result
    json.dumps(result, allow_nan=False)


def test_closed_line_inventory_and_damped_oscillation_do_not_need_pressure_anchor():
    model = oscillator()
    initial = model.initialize({"a": 601000, "b": 599000})
    result = model.advance(initial, 0.02)
    audit(model, initial, result)
    assert result["state"]["pressures_pa"]["a"] < 600000
    assert result["state"]["pipe_volume_rates_m3_s"]["line"] < 0
    assert (
        result["properties"]["acoustic_energy_j"] + result["properties"]["pipe_kinetic_energy_j"]
        < model.inspect(initial)["acoustic_energy_j"]
    )


def test_reversal_symmetry_of_signed_inertial_friction():
    model = oscillator()
    a = model.advance(model.initialize({"a": 601000, "b": 599000}), 0.005)
    b = model.advance(model.initialize({"a": 599000, "b": 601000}), 0.005)
    assert a["state"]["pressures_pa"]["a"] == pytest.approx(
        b["state"]["pressures_pa"]["b"], abs=1e-5
    )
    assert a["state"]["pipe_volume_rates_m3_s"]["line"] == pytest.approx(
        -b["state"]["pipe_volume_rates_m3_s"]["line"], abs=1e-10
    )


def test_quiescent_closed_network_preserves_pressure_and_zero_flow():
    model = oscillator()
    initial = model.initialize()
    result = model.advance(initial, 0.01)
    assert result["state"]["pressures_pa"] == dict(initial.pressures_pa)
    assert result["state"]["pipe_volume_rates_m3_s"] == dict(initial.pipe_volume_rates_m3_s)
    audit(model, initial, result)


def test_station_equilibrium_matches_existing_steady_network():
    data = json.loads((ROOT / "configs/surface_liquid_transient.json").read_text())
    model = LiquidTransientNetwork.from_dict(data["system"])
    initial = model.initialize(**data["initial"])
    result = model.advance(
        initial, 0.02, **{k: v for k, v in data["periods"][0].items() if k != "duration_s"}
    )
    assert result["state"]["speeds_rad_s"] == pytest.approx(dict(initial.speeds_rad_s), abs=1e-6)
    assert result["state"]["pressures_pa"] == pytest.approx(dict(initial.pressures_pa), abs=1e-3)
    audit(model, initial, result)


def test_closing_massless_valve_retains_fluid_inertia_and_storage():
    model = coupled_model()
    initial = coupled_initial(model)
    result = model.advance(
        initial, 0.02, motor_torques_nm={"a": 5, "b": 7}, valve_openings={"export": 0}
    )
    audit(model, initial, result)
    assert (
        max(abs(s["midpoint_branch_volume_rates_m3_s"]["export"]) for s in result["steps"]) < 1e-10
    )
    assert result["steps"][0]["midpoint_branch_volume_rates_m3_s"]["line"] > 0
    assert result["state"]["pressures_pa"]["t"] > 500000
    assert result["properties"]["pipe_kinetic_energy_j"] > 0


def test_trip_preserves_coasting_pump_flow_and_latches_until_reset():
    model = coupled_model()
    initial = coupled_initial(model)
    trip = model.advance(initial, 0.002, mode="trip")
    audit(model, initial, trip)
    assert trip["state"]["speeds_rad_s"]["a"] > 99
    assert trip["steps"][0]["midpoint_branch_volume_rates_m3_s"]["a"] > 0
    latched = model.advance(
        LiquidTransientState(**trip["state"]), 0.002, motor_torques_nm={"a": 20}
    )
    assert latched["effective_mode"] == "off"
    assert all(s["electrical_energy_j"] == 0 for s in latched["steps"])
    reset = model.advance(
        LiquidTransientState(**latched["state"]), 0.002, reset_trip=True, motor_torques_nm={"a": 5}
    )
    assert not reset["state"]["tripped"] and reset["effective_mode"] == "torque"


def test_unavailability_persists_without_erasing_shaft_energy():
    model = coupled_model()
    initial = coupled_initial(model)
    failed = model.advance(initial, 0.002, motor_torques_nm={"a": 5, "b": 7}, unavailable=["a"])
    assert failed["state"]["speeds_rad_s"]["a"] > 99
    assert failed["steps"][0]["motor_torques_nm"]["a"] == 0
    inherited = model.advance(
        LiquidTransientState(**failed["state"]), 0.002, motor_torques_nm={"a": 5}
    )
    assert inherited["state"]["unavailable"] == ["a"]
    restored = model.advance(
        LiquidTransientState(**inherited["state"]), 0.002, unavailable=[], motor_torques_nm={"a": 5}
    )
    assert restored["steps"][0]["motor_torques_nm"]["a"] > 0


@pytest.mark.parametrize("capacity", [0.0, 100.0, 1000.0])
def test_group_power_budget_survives_pressure_and_flow_dynamics(capacity):
    model = clone(coupled_model(), max_station_electrical_draw_w=capacity)
    initial = coupled_initial(model)
    result = model.advance(initial, 0.02, motor_torques_nm={"a": 20, "b": 20})
    audit(model, initial, result)
    assert all(s["peak_electrical_draw_w"] <= capacity + 1e-7 for s in result["steps"])


def test_two_stop_events_and_rest_do_not_export_fictitious_work():
    model = coupled_model(0.002)
    data = json.loads(json.dumps(model.to_dict()))
    for d in data["drives"].values():
        d["viscous_drag_nm_s"] = 0
    data["no_flow_power_w_at_nominal"] = {"a": 0, "b": 0}
    model = LiquidTransientNetwork.from_dict(data)
    initial = model.initialize({"h": 500000, "t": 480000}, speeds_rad_s={"a": 2, "b": 4})
    result = model.advance(initial, 0.4, motor_torques_nm={"a": -10, "b": -10})
    audit(model, initial, result)
    assert result["state"]["speeds_rad_s"] == {"a": 0, "b": 0}
    assert math.fsum(s["electrical_energy_j"] for s in result["steps"]) == pytest.approx(
        -0.7 * (0.5 * 0.5 * 2**2 + 0.5 * 0.7 * 4**2), abs=1e-6
    )
    assert result["steps"][-1]["electrical_energy_j"] == 0


def test_zero_power_at_rest_does_not_create_rotor_motion():
    model = clone(coupled_model(), max_station_electrical_draw_w=0)
    initial = model.initialize({"h": 500000, "t": 480000})
    result = model.advance(initial, 0.004, motor_torques_nm={"a": 20, "b": 20})
    assert result["state"]["speeds_rad_s"] == {"a": 0, "b": 0}


def test_weak_positive_torque_is_not_a_stopping_event():
    model = coupled_model()
    initial = model.initialize({"h": 500000, "t": 480000})
    result = model.advance(initial, 0.002, motor_torques_nm={"a": 2e-8})
    assert 0 < result["state"]["speeds_rad_s"]["a"] < 1e-8
    assert result["state"]["speeds_rad_s"]["b"] == 0


def test_dissipative_braking_remains_available_with_zero_export_capacity():
    data = coupled_model().to_dict()
    for drive in data["drives"].values():
        drive["regeneration_efficiency"] = 0
    data["max_station_electrical_export_w"] = 0
    model = LiquidTransientNetwork.from_dict(data)
    initial = model.initialize({"h": 500000, "t": 480000}, speeds_rad_s={"a": 2, "b": 4})
    result = model.advance(initial, 0.004, motor_torques_nm={"a": -10, "b": -10})
    audit(model, initial, result)
    assert result["state"]["speeds_rad_s"]["a"] < 2
    assert all(s["electrical_energy_j"] == 0 and s["motor_heat_j"] > 0 for s in result["steps"])


def test_final_pump_flow_cannot_escape_declared_map_despite_bounded_trial_efficiency():
    model = coupled_model()
    initial = model.initialize({"h": 410000, "t": 480000}, speeds_rad_s={"a": 100, "b": 110})
    with pytest.raises(EquipmentOperatingError, match="flow outside"):
        model.advance(initial, 0.001, motor_torques_nm={"a": 5, "b": 7})


def test_local_pressure_guard_retries_without_losing_report_time():
    model = clone(oscillator(), max_pressure_change_pa=5)
    initial = model.initialize({"a": 601000, "b": 599000})
    result = model.advance(initial, 0.005)
    assert result["failed_steps"] > 0
    assert "Pressure change guard" in result["retry_reasons"]
    assert result["state"]["time_s"] == 0.005
    audit(model, initial, result)


def test_embedded_temporal_accuracy_refines_wave_phase_and_preserves_ledgers():
    expected, _ = oscillator_reference(0.02)
    errors, counts = [], []
    for factor in (1.0, 1 / 64):
        model = clone(
            oscillator(),
            temporal_error_control=True,
            relative_tolerance=1e-5 * factor,
            pressure_tolerance_pa=factor,
            flow_tolerance_m3_s=1e-9 * factor,
            speed_tolerance_rad_s=1e-5 * factor,
        )
        initial = model.initialize({"a": 601000, "b": 599000})
        result = model.advance(initial, 0.02)
        audit(model, initial, result)
        errors.append(
            max(
                abs(result["state"]["pressures_pa"][n] - expected[i])
                for i, n in enumerate(("a", "b"))
            )
        )
        counts.append(len(result["steps"]))
        assert all(
            s["temporal_error_ratio"] is not None and s["temporal_error_ratio"] <= 1
            for s in result["steps"]
        )
    assert errors[1] < errors[0] / 2 and counts[1] > counts[0]


def test_model_state_and_configuration_json_roundtrip():
    model = coupled_model()
    recovered = LiquidTransientNetwork.from_dict(json.loads(json.dumps(model.to_dict())))
    assert recovered.model_id == model.model_id
    initial = coupled_initial(model)
    assert LiquidTransientState(**json.loads(json.dumps(initial.to_dict()))) == initial


def test_source_bound_restart_replays_physics_and_rejects_recomputed_checksum_forgery():
    model = oscillator()
    initial = model.initialize({"a": 601000, "b": 599000})
    periods = [{"duration_s": 0.002}, {"duration_s": 0.003}]
    full = run_liquid_transient_schedule(model, initial, periods)
    partial = run_liquid_transient_schedule(model, initial, periods, stop_after=1)
    assert (
        run_liquid_transient_schedule(model, initial, periods, json.loads(json.dumps(partial)))
        == full
    )
    corrupt = copy.deepcopy(partial)
    corrupt["records"][0]["steps"][0]["acoustic_energy_change_j"] += 1
    corrupt["checksum"] = _digest({k: v for k, v in corrupt.items() if k != "checksum"})
    with pytest.raises(ConservationError, match="replay"):
        run_liquid_transient_schedule(model, initial, periods, corrupt)
    corrupt = copy.deepcopy(partial)
    corrupt["state"]["pressures_pa"]["a"] += 1
    corrupt["checksum"] = _digest({k: v for k, v in corrupt.items() if k != "checksum"})
    with pytest.raises(ConservationError, match="memory"):
        run_liquid_transient_schedule(model, initial, periods, corrupt)
    with pytest.raises(ValueError, match="fingerprint"):
        run_liquid_transient_schedule(model, initial, periods + [{"duration_s": 0.001}], partial)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_compression_fraction": 0.1},
        {"maximum_dt_s": 1e-9},
        {"storage_volumes_m3": {"a": 0.01}},
        {"bulk_modulus_pa": -1},
    ],
)
def test_invalid_model_domains_are_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        clone(oscillator(), **changes)


def test_invalid_pressure_controls_and_substep_budget_are_rejected():
    model = oscillator()
    with pytest.raises(EquipmentOperatingError):
        model.initialize({"a": 1000})
    with pytest.raises(ValueError):
        model.advance(model.initialize(), 0.01, boundary_pressures_pa={"a": 600000})
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        model.advance(model.initialize(), 0.01, max_substeps=1)


def test_cli_failure_does_not_replace_existing_output(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("preserve me")
    config = tmp_path / "invalid.json"
    config.write_text(
        json.dumps(
            {"system": oscillator().to_dict(), "initial": {}, "periods": [{"duration_s": -0.01}]}
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_liquid_transient.py"),
            str(config),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert output.read_text() == "preserve me"
