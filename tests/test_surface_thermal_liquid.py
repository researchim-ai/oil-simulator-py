"""Thermal recycle controls, independent latch ownership and source-bound replay."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from surface import ThermalLiquidState, ThermalLiquidStation
from surface.operation import _digest
from surface.thermal_liquid_schedule import run_thermal_liquid_schedule


def setup(**heat_updates):
    d = json.loads((ROOT / "configs/surface_thermal_liquid.json").read_text())
    d["system"]["heat"].update(heat_updates)
    m = ThermalLiquidStation.from_dict(d["system"])
    return m, m.initialize(**d["initial"]), d


def test_thermal_station_model_state_and_json_roundtrip():
    m, s, _ = setup()
    assert (
        ThermalLiquidStation.from_dict(json.loads(json.dumps(m.to_dict()))).model_id == m.model_id
    )
    assert ThermalLiquidState.from_dict(json.loads(json.dumps(s.to_dict()))) == s


@pytest.mark.parametrize("motor,shaft", [(0.0, 0.0), (1.0, 1.0), (0.3, 0.4)])
def test_closed_circuit_grid_energy_is_converted_exactly_once(motor, shaft):
    m, s, d = setup(motor_heat_to_fluid_fraction=motor, shaft_heat_to_fluid_fraction=shaft)
    cmd = {k: v for k, v in d["periods"][0].items() if k != "duration_s"}
    r = m.advance(s, 0.1, **cmd)
    before = m.heat.inspect(s.heat)
    after = r["thermal_properties"]
    expected = sum(
        x["pump_heat_j"]
        + x["friction_heat_j"]
        + motor * x["motor_heat_j"]
        + shaft * x["shaft_drag_heat_j"]
        for x in r["steps"]
    )
    delta = (
        after["fluid_energy_j"]
        + after["wall_energy_j"]
        - before["fluid_energy_j"]
        - before["wall_energy_j"]
    )
    assert delta == pytest.approx(expected, abs=2e-7)
    assert max(abs(x["combined_energy_residual_j"]) for x in r["thermal_audits"]) < 1e-5
    assert after["temperatures_k"]["in"] > 300
    if motor == shaft == 1:
        assert delta == pytest.approx(
            sum(
                x["electrical_energy_j"]
                + x["boundary_work_in_j"]
                - x["shaft_kinetic_energy_change_j"]
                - x["pipe_kinetic_energy_change_j"]
                - x["acoustic_energy_change_j"]
                for x in r["steps"]
            ),
            abs=2e-7,
        )


def test_partial_reports_keep_one_thermal_pid_clock_and_actual_shared_capacity():
    m, s, d = setup()
    cmd = {
        k: v
        for k, v in d["periods"][0].items()
        if k not in ("duration_s", "temperature_control_modes")
    }
    cmd.update(
        temperature_control_modes={"cool-header": "manual", "cool-suction": "manual"},
        temperature_manual_duties_w={"cool-header": -2000, "cool-suction": -2000},
    )
    a = m.advance(s, 0.013, **cmd)
    b = m.advance(ThermalLiquidState.from_dict(a["state"]), 0.087, **cmd)
    assert [
        x["time_s"] for x in a["temperature_samples"] + b["temperature_samples"]
    ] == pytest.approx([0, 0.05])
    for sample in a["temperature_samples"] + b["temperature_samples"]:
        assert sum(v["applied_duty_w"] for v in sample["controllers"].values()) == -1000
        assert all(v["requested_duty_w"] == -2000 for v in sample["controllers"].values())
    assert b["state"]["heat"]["temperatures_k"]["h"] < 300
    # The suction receives valve dissipation, so its temperature can rise
    # even while the TOTAL circuit loses energy under the shared cooler cap.
    before = m.heat.inspect(s.heat)
    after = b["thermal_properties"]
    assert (
        after["fluid_energy_j"] + after["wall_energy_j"]
        < before["fluid_energy_j"] + before["wall_energy_j"]
    )


def test_temperature_trip_cannot_be_cleared_by_operator_reset_or_external_commands():
    m, s, d = setup()
    # Explicitly initialize a hot physical inventory; no artificial temperature
    # mutation after the run has started.
    initial = copy.deepcopy(d["initial"])
    initial["temperatures_k"] = {"h": 300.3}
    s = m.initialize(**initial)
    cmd = {k: v for k, v in d["periods"][0].items() if k != "duration_s"}
    r = m.advance(s, 0.15, unavailable=["p3"], **cmd)
    assert r["latched_temperature_trips"] == ["hot-liquid"]
    assert r["state"]["operator_tripped"] is False
    assert r["state"]["recycling"]["protected"]["operator_tripped"] is True
    assert r["state"]["external_unavailable"] == ["p3"]
    next_state = ThermalLiquidState.from_dict(r["state"])
    denied = m.advance(
        next_state, 0.05, reset_trip=True, reset_temperature_trips=["hot-liquid"], **cmd
    )
    assert denied["temperature_events"][0]["type"] == "reset_denied"
    assert denied["latched_temperature_trips"] == ["hot-liquid"]
    assert denied["state"]["external_unavailable"] == ["p3"]
    assert all(
        abs(v) < 1e-10 for step in denied["steps"] for v in step["motor_torques_nm"].values()
    )


def test_cooling_is_real_energy_removal_and_safe_reset_preserves_operator_trip():
    m, s, d = setup()
    initial = copy.deepcopy(d["initial"])
    initial["temperatures_k"] = {"h": 300.2}
    s = m.initialize(**initial)
    # Advance beyond the configured delay, then actively cool the retained
    # inventories while the drives remain tripped.
    r = m.advance(
        s,
        0.15,
        mode="trip",
        valve_openings={"export": 0},
        temperature_control_modes={"cool-header": "manual", "cool-suction": "manual"},
        temperature_manual_duties_w={"cool-header": -2000, "cool-suction": -2000},
    )
    a = ThermalLiquidState.from_dict(r["state"])
    r = m.advance(
        a,
        4,
        mode="off",
        valve_openings={"export": 0},
        temperature_control_modes={"cool-header": "manual", "cool-suction": "manual"},
        temperature_manual_duties_w={"cool-header": -2000, "cool-suction": -2000},
    )
    assert sum(x["external_heat_duty_j"] for x in r["thermal_steps"]) == pytest.approx(
        -4000, abs=1e-6
    )
    assert r["state"]["heat"]["temperatures_k"]["h"] < 300.13
    r = m.advance(
        ThermalLiquidState.from_dict(r["state"]),
        0.05,
        mode="off",
        valve_openings={"export": 0},
        reset_temperature_trips=["hot-liquid"],
    )
    assert not r["latched_temperature_trips"]
    assert r["state"]["operator_tripped"] is True
    assert r["state"]["recycling"]["protected"]["operator_tripped"] is True


def test_temperature_restart_replay_covers_thermal_memory_and_partial_clock():
    m, s, d = setup()
    cmd = {k: v for k, v in d["periods"][0].items() if k != "duration_s"}
    periods = [{"duration_s": 0.013, **cmd}, {"duration_s": 0.057, **cmd}]
    full = run_thermal_liquid_schedule(m, s, periods)
    prefix = run_thermal_liquid_schedule(m, s, periods, stop_after=1)
    assert run_thermal_liquid_schedule(m, s, periods, checkpoint=prefix) == full
    changed = copy.deepcopy(prefix)
    changed["state"]["heat"]["temperatures_k"]["h"] += 0.01
    changed["checksum"] = _digest({k: v for k, v in changed.items() if k != "checksum"})
    with pytest.raises(RuntimeError, match="memory"):
        run_thermal_liquid_schedule(m, s, periods, checkpoint=changed)


def test_thermal_clock_and_action_provenance_cannot_be_forged():
    m, s, _ = setup()
    with pytest.raises(ValueError, match="clock"):
        m.advance(replace(s, next_tick=5), 0.01)
    with pytest.raises(ValueError, match="provenance"):
        m.advance(replace(s, external_unavailable=("p1",)), 0.01)


@pytest.mark.parametrize(
    "command",
    [
        {"heat_duties_w": {"h": 100}},
        {"temperature_control_modes": {"missing": "auto"}},
        {"temperature_control_modes": {"cool-header": "manual"}},
        {"temperature_manual_duties_w": {"cool-header": -3000}},
        {"reset_temperature_trips": ["missing"]},
        {"temperature_setpoints_k": {"cool-header": 100}},
        {"max_thermal_samples": True},
        {"temperature_control_modes": {"cool-header": "teleport"}},
    ],
)
def test_conflicting_unsafe_or_unknown_thermal_controls(command):
    m, s, _ = setup()
    with pytest.raises(ValueError):
        m.advance(s, 0.01, **command)


def test_actual_reverse_export_transports_heat_and_mass_from_correct_boundary():
    m, _, _d = setup()
    initial = {
        "recycling_initial": {
            "protected_initial": {
                "controller_initial": {
                    "physical_initial": {"pressures_pa": {"h": 500000, "t": 500000}}
                }
            }
        }
    }
    s = m.initialize(**initial)
    r = m.advance(
        s,
        0.1,
        mode="off",
        boundary_pressures_pa={"out": 600000},
        boundary_temperatures_k={"out": 310},
        temperature_control_modes={"cool-header": "off", "cool-suction": "off"},
    )
    assert any(x["midpoint_branch_volume_rates_m3_s"]["export"] < 0 for x in r["steps"])
    assert sum(x["boundary_mass_transfers_in_kg"]["out"] for x in r["steps"]) > 0
    assert sum(x["external_advective_heat_in_j"] for x in r["thermal_steps"]) > 0
    assert r["state"]["heat"]["temperatures_k"]["t"] > 300
    assert max(abs(x["combined_energy_residual_j"]) for x in r["thermal_audits"]) < 1e-5


def test_finite_suction_cannot_be_overridden_with_infinite_temperature_feed():
    m, s, d = setup()
    cmd = {k: v for k, v in d["periods"][0].items() if k != "duration_s"}
    with pytest.raises(ValueError, match="boundary"):
        m.advance(s, 0.01, boundary_temperatures_k={"in": 300}, **cmd)


def test_forward_check_bounds_do_not_disguise_true_unsupported_turbine_regime():
    m, _s, _ = setup()
    p = m.recycling.physics
    physical = p.initialize({"h": 300000, "t": 300000}, speeds_rad_s={"p1": 100})
    with pytest.raises(ValueError):
        p.advance(
            physical, 0.001, motor_torques_nm={"p1": 0}, valve_openings={"export": 0, "recycle": 0}
        )


def test_regenerative_braking_returns_heat_and_grid_export_without_double_counting():
    m, s, _ = setup(motor_heat_to_fluid_fraction=1, shaft_heat_to_fluid_fraction=1)
    r = m.advance(
        s,
        0.15,
        mode="torque",
        motor_torques_nm={"p1": -10, "p2": -10},
        valve_openings={"export": 0},
        temperature_control_modes={"cool-header": "off", "cool-suction": "off"},
    )
    assert sum(x["electrical_energy_j"] for x in r["steps"]) < 0
    assert sum(x["motor_heat_j"] for x in r["steps"]) > 0
    assert sum(x["shaft_kinetic_energy_change_j"] for x in r["steps"]) < 0
    assert abs(sum(x["equipment_heat_out_j"] for x in r["thermal_steps"])) < 1e-6
    assert max(abs(x["combined_energy_residual_j"]) for x in r["thermal_audits"]) < 1e-5


def test_actual_export_depletes_finite_suction_without_implicit_makeup():
    m, _, _ = setup()
    physics = m.recycling.physics
    steady = physics.network.solve(
        pump_speed_ratios={"p1": 1, "p2": 1, "p3": 1},
        pump_status={"p1": True, "p2": True, "p3": False},
        valve_openings={"recycle": 0},
    )
    torque = {n: (steady.pump_shaft_power_w[n] + 10) / 100 + 0.005 * 100 for n in ("p1", "p2")}
    s = m.initialize(
        recycling_initial={
            "protected_initial": {
                "controller_initial": {
                    "physical_initial": {
                        "pressures_pa": {n: steady.pressures_pa[n] for n in ("h", "t")},
                        "pipe_volume_rates_m3_s": steady.pipe_volume_rates_m3_s,
                        "speeds_rad_s": {"p1": 100, "p2": 100},
                    },
                    "running_pumps": ["p1", "p2"],
                    "applied_torques_nm": torque,
                }
            }
        }
    )
    commands = {
        "mode": "torque",
        "motor_torques_nm": torque,
        "temperature_control_modes": {"cool-header": "off", "cool-suction": "off"},
    }
    r = m.advance(s, 1, **commands)
    assert r["state"]["heat"]["masses_kg"]["in"] == pytest.approx(
        5 - steady.boundary_mass_rates_kg_s["in"], abs=1e-7
    )
    next_state = ThermalLiquidState.from_dict(r["state"])
    before = next_state.to_dict()
    with pytest.raises(ValueError, match="depleted"):
        m.advance(next_state, 1, **commands)
    assert next_state.to_dict() == before
