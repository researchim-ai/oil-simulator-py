"""Independent shaft motion/work limits, finite stop, maps and restart."""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    EquipmentOperatingError,
    Liquid,
    Pump,
    RotatingDrive,
    RotorState,
    pump_shaft_load,
)


def drive(**options):
    return RotatingDrive(2, 1000, 100, 100, **options)


def test_constant_torque_acceleration_and_independent_work():
    p = drive(motor_efficiency=0.8)
    r = p.advance(p.initialize(10), 3, 12, 4)
    assert r["state"]["angular_velocity_rad_s"] == pytest.approx(22)
    assert r["angle_rad"] == pytest.approx(48)
    assert r["motor_work_j"] == pytest.approx(576)
    assert r["load_work_j"] == pytest.approx(192)
    assert r["kinetic_energy_change_j"] == pytest.approx(384)
    assert r["electrical_energy_j"] == pytest.approx(720)
    assert r["motor_heat_j"] == pytest.approx(144)
    assert abs(r["energy_residual_j"]) < 1e-10


@pytest.mark.parametrize("drag", [0, 1e-12, 0.01, 0.5, 1, 10, 1e4])
def test_drag_solution_and_heat_against_independent_quadrature(drag):
    p = drive(viscous_drag_nm_s=drag)
    r = p.advance(p.initialize(7), 0.7, 9, 2)

    def speed(t):
        if drag == 0:
            return 7 + 7 * t / 2
        return 7 * math.exp(-drag * t / 2) + 7 * -math.expm1(-drag * t / 2) / drag

    expected_angle = quad(speed, 0, 0.7, epsabs=1e-11)[0]
    expected_heat = quad(lambda t: drag * speed(t) ** 2, 0, 0.7, epsabs=1e-9)[0]
    assert r["state"]["angular_velocity_rad_s"] == pytest.approx(speed(0.7), rel=1e-12)
    assert r["angle_rad"] == pytest.approx(expected_angle, rel=1e-11)
    assert r["drag_heat_j"] == pytest.approx(expected_heat, rel=1e-10, abs=1e-11)
    assert abs(r["energy_residual_j"]) < 1e-8


def test_exact_exponential_coast_and_finite_stopping_event():
    p = drive(viscous_drag_nm_s=2)
    r = p.advance(p.initialize(10), 1, 0)
    assert r["state"]["angular_velocity_rad_s"] == pytest.approx(10 / math.e)
    assert r["drag_heat_j"] == pytest.approx(100 * (1 - math.exp(-2)))
    r = p.advance(p.initialize(10), 20, 0, 4)
    stop = math.log(6)
    assert r["moving_duration_s"] == pytest.approx(stop)
    assert r["state"]["angular_velocity_rad_s"] == 0
    assert r["rest_constraint_impulse_nm_s"] == pytest.approx(4 * (20 - stop))
    assert r["load_work_j"] + r["drag_heat_j"] == pytest.approx(100)
    held = p.advance(p.initialize(), 5, 1, 2)
    assert held["state"]["angular_velocity_rad_s"] == 0
    assert held["angle_rad"] == held["electrical_energy_j"] == 0


def test_regenerative_braking_does_not_return_more_than_stored_energy():
    p = drive(regeneration_efficiency=0.7)
    r = p.advance(p.initialize(10), 10, -4)
    assert r["moving_duration_s"] == pytest.approx(5)
    assert r["kinetic_energy_change_j"] == pytest.approx(-100)
    assert r["electrical_energy_j"] == pytest.approx(-70)
    assert r["motor_heat_j"] == pytest.approx(30)
    assert r["energy_residual_j"] == pytest.approx(0, abs=1e-12)


def test_power_limits_check_trajectory_peak_not_interval_average():
    p = drive(max_electrical_draw_w=10)
    with pytest.raises(EquipmentOperatingError, match="draw"):
        p.advance(p.initialize(), 1, 10)
    p = drive(max_electrical_export_w=10)
    with pytest.raises(EquipmentOperatingError, match="Regeneration"):
        p.advance(p.initialize(10), 100, -4)
    with pytest.raises(EquipmentOperatingError, match="speed"):
        replace(drive(), max_speed_rad_s=5).advance(
            replace(drive(), max_speed_rad_s=5).initialize(), 10, 10
        )


def test_partition_and_serialized_state_preserve_motion_and_work():
    p = drive(viscous_drag_nm_s=0.7, motor_efficiency=0.9)
    state = p.initialize(20)
    full = p.advance(state, 4, 3, 8)
    first = p.advance(state, 1.25, 3, 8)
    resumed = RotorState(**json.loads(json.dumps(first["state"])))
    last = p.advance(resumed, 2.75, 3, 8)
    assert last["state"]["angular_velocity_rad_s"] == pytest.approx(
        full["state"]["angular_velocity_rad_s"], abs=1e-12
    )
    for key in ("angle_rad", "electrical_energy_j", "drag_heat_j", "load_work_j"):
        assert first[key] + last[key] == pytest.approx(full[key], rel=1e-12, abs=1e-10)
    with pytest.raises(ValueError, match="mismatch"):
        replace(p, inertia_kg_m2=3).advance(resumed, 1, 3)


def test_existing_pump_map_affinity_and_load_power_identity():
    p = Pump("pump", "in", "out", 20, 1e6, 0.004, efficiency=0.8)
    liquid = Liquid(1000, 0.001, 2000)
    base = pump_shaft_load(p, liquid, 100, 100, 0.002, 2e5)
    half = pump_shaft_load(p, liquid, 50, 100, 0.001, 2e5)
    assert base["load_torque_nm"] * 100 == pytest.approx(base["shaft_power_w"])
    assert half["shaft_power_w"] == pytest.approx(base["shaft_power_w"] / 8)
    assert half["load_torque_nm"] == pytest.approx(base["load_torque_nm"] / 4)
    assert pump_shaft_load(p, liquid, 0, 100, 0, 2e5)["shaft_power_w"] == 0
    with pytest.raises(EquipmentOperatingError):
        pump_shaft_load(p, liquid, 0, 100, 0.001, 2e5)
    with pytest.raises(EquipmentOperatingError):
        pump_shaft_load(p, liquid, 25, 100, 0.002, 2e5)


@pytest.mark.parametrize(
    "options",
    [
        {"inertia_kg_m2": 0},
        {"motor_efficiency": 1.1},
        {"viscous_drag_nm_s": -1},
        {"max_electrical_draw_w": float("nan")},
    ],
)
def test_invalid_parameters(options):
    with pytest.raises(ValueError):
        RotatingDrive(
            **{
                "inertia_kg_m2": 2,
                "max_speed_rad_s": 100,
                "max_motor_torque_nm": 10,
                "max_braking_torque_nm": 10,
                **options,
            }
        )


def test_schedule_restart_and_forged_work_are_audited():
    from surface import ConservationError, run_drive_schedule
    from surface.operation import _digest

    p = drive(viscous_drag_nm_s=0.3)
    initial = p.initialize(10)
    periods = [
        {"duration_s": 1, "motor_torque_nm": 8},
        {"duration_s": 3, "motor_torque_nm": -4, "load_torque_nm": 1},
    ]
    full = run_drive_schedule(p, initial, periods)
    partial = run_drive_schedule(p, initial, periods, stop_after=1)
    assert run_drive_schedule(p, initial, periods, json.loads(json.dumps(partial))) == full
    forged = json.loads(json.dumps(partial))
    forged["records"][0]["motor_work_j"] += 1
    forged["checksum"] = _digest({k: v for k, v in forged.items() if k != "checksum"})
    with pytest.raises(ConservationError, match="audit"):
        run_drive_schedule(p, initial, periods, forged)
    with pytest.raises(ValueError, match="fingerprint"):
        run_drive_schedule(p, initial, [*periods, periods[0]], partial)
    with pytest.raises(ValueError):
        run_drive_schedule(p, initial, periods, stop_after=True)


def test_unrepresentable_trajectory_is_a_numerical_failure_not_capacity():
    from surface import ConservationError

    p = RotatingDrive(2, 1e308, 1e308, 100)
    with pytest.raises(ConservationError, match="representable"):
        p.advance(p.initialize(), 10, 1e308)


def test_dissipative_braking_can_have_zero_electrical_export():
    p = drive(regeneration_efficiency=0, max_electrical_export_w=0)
    r = p.advance(p.initialize(10), 10, -4)
    assert r["electrical_energy_j"] == 0
    assert r["motor_heat_j"] == pytest.approx(100)
    assert r["state"]["angular_velocity_rad_s"] == 0
