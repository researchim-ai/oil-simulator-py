"""Conservative plenum/recycle, finite anti-surge hardware and shaft dynamics."""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    CompressorEnvelopeError,
    CompressorMap,
    CompressorTrainState,
    ConnectionActuator,
    DynamicCompressorTrain,
    IdealGas,
    MappedGasCompressor,
    PIDController,
    RotatingDrive,
)


def model(dt=0.01, recycle_area=0.003):
    mapping = CompressorMap(
        (60, 100, 140),
        (0, 0.5, 1),
        ((0.12, 0.3, 0.48), (0.25, 0.5, 0.75), (0.4, 0.8, 1.2)),
        ((1.2, 1.12, 1.02), (1.5, 1.3, 1.05), (1.9, 1.6, 1.1)),
        ((0.65, 0.78, 0.65), (0.68, 0.8, 0.67), (0.65, 0.78, 0.65)),
    )
    return DynamicCompressorTrain(
        MappedGasCompressor(mapping, IdealGas(500, 1.3)),
        RotatingDrive(
            5,
            140,
            400,
            200,
            viscous_drag_nm_s=0.2,
            motor_efficiency=0.9,
            regeneration_efficiency=0.7,
            max_electrical_draw_w=50000,
        ),
        ConnectionActuator(3, 0.5, 1, time_constant_s=0.05),
        PIDController(0.6, 1, 0, 0, 1, tracking_time_s=0.2),
        2,
        1e5,
        300,
        1e5,
        0.002,
        recycle_area,
        speed_controller=PIDController(8, 10, 0, -200, 400, tracking_time_s=0.2),
        maximum_dt_s=dt,
    )


def audit(p, initial, result):
    json.dumps(result, allow_nan=False)
    s = result["state"]
    rows = result["steps"]
    assert s["time_s"] == pytest.approx(initial.time_s + result["duration_s"], abs=1e-12)
    assert math.fsum(x["dt_s"] for x in rows) == pytest.approx(result["duration_s"], abs=1e-12)
    assert s["mass_kg"] - initial.mass_kg == pytest.approx(
        math.fsum(x["net_suction_mass_kg"] - x["export_mass_kg"] for x in rows), abs=1e-8
    )
    kinetic = (
        0.5
        * p.drive.inertia_kg_m2
        * (s["angular_velocity_rad_s"] ** 2 - initial.angular_velocity_rad_s**2)
    )
    incoming = math.fsum(
        x["electrical_energy_j"] + x["suction_enthalpy_j"] - x["export_enthalpy_j"] for x in rows
    )
    rejected = math.fsum(
        x["motor_heat_j"]
        + x["drag_heat_j"]
        + x["mechanical_heat_j"]
        + x["wall_heat_j"]
        + x["conditioning_heat_j"]
        for x in rows
    )
    assert incoming - rejected == pytest.approx(
        s["internal_energy_j"] - initial.internal_energy_j + kinetic, abs=5e-5
    )
    assert min(x["entropy_production_j_k"] for x in rows) >= -1e-7


def test_closed_plenum_retains_mass_and_energy_while_unloaded_shaft_coasts():
    p = model()
    s = p.initialize(
        pressure_pa=2e5, temperature_k=350, angular_velocity_rad_s=80, recycle_position=0
    )
    r = p.advance(s, 0.2, mode="off", export_opening=0, recycle_actuator_mode="stuck")
    audit(p, s, r)
    assert r["state"]["mass_kg"] == pytest.approx(s.mass_kg, abs=1e-12)
    assert r["state"]["internal_energy_j"] == pytest.approx(s.internal_energy_j, abs=1e-8)
    assert r["state"]["angular_velocity_rad_s"] == pytest.approx(
        80 * math.exp(-0.2 * 0.2 / 5), rel=1e-12
    )


def test_adiabatic_choked_blowdown_refines_against_independent_mass_energy_solution():
    errors = []
    for dt in (0.04, 0.02, 0.01):
        p = model(dt)
        p.control_interval_s = 0.2
        s = p.initialize(pressure_pa=4e5, recycle_position=0)
        r = p.advance(s, 0.2, mode="off", export_opening=1, recycle_actuator_mode="stuck")
        g = 1.3
        rg = 500.0
        factor = math.sqrt(g * (2 / (g + 1)) ** ((g + 1) / (g - 1)))
        m0 = 4e5 * 2 / (rg * 300)
        flow0 = 0.002 * 4e5 / math.sqrt(rg * 300) * factor
        a = (g + 1) / 2
        fraction = (1 + (a - 1) * flow0 / m0 * 0.2) ** (-1 / (a - 1))
        assert 1e5 / (4e5 * fraction**g) < (2 / (g + 1)) ** (g / (g - 1))
        errors.append(abs(r["state"]["mass_kg"] - m0 * fraction))
        audit(p, s, r)
    assert errors[-1] < 0.001
    assert errors[0] / errors[1] > 1.8 and errors[1] / errors[2] > 1.8


def test_recycle_returns_mass_and_conditioning_heat_instead_of_creating_gas():
    p = model()
    s = p.initialize(
        pressure_pa=1.3e5, temperature_k=330, angular_velocity_rad_s=85, recycle_position=0.4
    )
    r = p.advance(s, 5, mode="run", speed_setpoint_rad_s=85, export_opening=0)
    audit(p, s, r)
    assert sum(x["recycle_mass_kg"] for x in r["steps"]) > 0.5
    assert sum(x["conditioning_heat_j"] for x in r["steps"]) > 1000
    assert all(x["export_mass_kg"] == 0 for x in r["steps"])
    assert r["steps"][-1]["operating_point"]["surge_margin"] > 0.1


def test_anti_surge_opens_recycle_after_export_turndown():
    p = model()
    s = p.initialize(
        pressure_pa=1.3e5, temperature_k=330, angular_velocity_rad_s=85, recycle_position=0
    )
    r = p.advance(s, 5, mode="run", speed_setpoint_rad_s=85, export_opening=0.8)
    state = CompressorTrainState.from_dict(r["state"])
    turned = p.advance(state, 5, mode="run", speed_setpoint_rad_s=85, export_opening=0.05)
    audit(p, state, turned)
    assert turned["state"]["recycle_position"] > 0.2
    assert min(x["operating_point"]["surge_margin"] for x in turned["steps"]) >= 0
    assert any(x["fast_open_override"] for x in turned["control_samples"])


def test_stuck_closed_valve_does_not_fake_a_safe_surge_solution():
    p = model()
    s = p.initialize(
        pressure_pa=1.25e5, temperature_k=320, angular_velocity_rad_s=85, recycle_position=0
    )
    with pytest.raises(CompressorEnvelopeError) as error:
        p.advance(
            s,
            5,
            mode="run",
            speed_setpoint_rad_s=85,
            export_opening=0,
            anti_surge_mode="manual",
            manual_recycle_opening=0,
            recycle_actuator_mode="stuck",
        )
    assert error.value.boundary == "surge"
    assert error.value.last_accepted_state is not None
    safe = CompressorTrainState.from_dict(error.value.last_accepted_state)
    shut = p.advance(safe, 0.2, mode="trip", export_opening=0)
    audit(p, safe, shut)
    assert shut["state"]["tripped"]


def test_trip_latch_and_actuator_retries_do_not_resample_controllers():
    p = model()
    p.recycle_actuator = replace(p.recycle_actuator, max_position_change=0.005)
    s = p.initialize(recycle_position=0, angular_velocity_rad_s=80)
    r = p.advance(s, 0.1, mode="trip", export_opening=0)
    audit(p, s, r)
    assert len(r["control_samples"]) == 2 and r["failed_steps"] > 0
    assert r["state"]["recycle_position"] > 0
    latched = p.advance(
        CompressorTrainState.from_dict(r["state"]), 0.1, mode="run", speed_setpoint_rad_s=85
    )
    assert latched["effective_mode"] == "off"
    assert all(x["gas_work_j"] == 0 for x in latched["steps"])


def test_recycle_duty_limit_is_enforced():
    p = model()
    p.max_recycle_cooling_w = 1
    s = p.initialize(pressure_pa=1.3e5, temperature_k=350, recycle_position=1)
    with pytest.raises(CompressorEnvelopeError) as error:
        p.advance(s, 0.1, mode="off")
    assert error.value.boundary == "recycle_duty"


def test_motor_power_limit_and_unmapped_startup_are_explicit():
    p = model()
    p.drive = replace(p.drive, max_electrical_draw_w=10000)
    s = p.initialize(pressure_pa=1.2e5, temperature_k=320, angular_velocity_rad_s=85)
    r = p.advance(s, 0.2, mode="run", speed_control_mode="manual", motor_torque_nm=400)
    audit(p, s, r)
    assert all(x["peak_electrical_draw_w"] <= 10000 * (1 + 1e-12) for x in r["steps"])
    with pytest.raises(CompressorEnvelopeError):
        p.advance(p.initialize(), 0.1, mode="run", speed_setpoint_rad_s=85)
    zero = model()
    zero.drive = replace(zero.drive, max_electrical_draw_w=0)
    r = zero.advance(zero.initialize(), 0.1, mode="standby", motor_torque_nm=100)
    assert r["state"]["angular_velocity_rad_s"] == 0


def test_undersized_recycle_has_a_real_capacity_failure():
    p = model(recycle_area=1e-7)
    s = p.initialize(pressure_pa=1.25e5, temperature_k=320, angular_velocity_rad_s=85)
    with pytest.raises(CompressorEnvelopeError) as error:
        p.advance(s, 3, mode="run", speed_setpoint_rad_s=85, export_opening=0)
    assert error.value.boundary == "surge"


def test_schedule_restart_and_controller_energy_tampering():
    import copy

    from surface import ConservationError, run_compressor_schedule
    from surface.operation import _digest

    p = model()
    initial = p.initialize(pressure_pa=1.2e5, temperature_k=310, angular_velocity_rad_s=85)
    periods = [
        {"duration_s": 0.1, "mode": "run", "speed_setpoint_rad_s": 85, "export_opening": 0.5},
        {"duration_s": 0.1, "mode": "trip", "export_opening": 0.5},
        {"duration_s": 0.1, "mode": "standby", "reset_trip": True, "motor_torque_nm": -100},
    ]
    full = run_compressor_schedule(p, initial, periods)
    partial = run_compressor_schedule(p, initial, periods, stop_after=1)
    assert run_compressor_schedule(p, initial, periods, json.loads(json.dumps(partial))) == full
    for key in ("conditioning_heat_j", "net_suction_mass_kg"):
        bad = copy.deepcopy(partial)
        bad["records"][0]["steps"][0][key] += 1e-4 if key == "net_suction_mass_kg" else 1
        bad["checksum"] = _digest({k: v for k, v in bad.items() if k != "checksum"})
        with pytest.raises(ConservationError, match="audit"):
            run_compressor_schedule(p, initial, periods, bad)
    bad = copy.deepcopy(partial)
    bad["records"][0]["control_samples"][0]["anti_surge"]["state"]["integral_output"] += 1
    bad["checksum"] = _digest({k: v for k, v in bad.items() if k != "checksum"})
    with pytest.raises(ConservationError, match="PID replay"):
        run_compressor_schedule(p, initial, periods, bad)
    with pytest.raises(ValueError, match="fingerprint"):
        run_compressor_schedule(p, initial, [*periods, periods[0]], partial)


def test_corrected_speed_ceiling_overrides_larger_hardware_speed_limit():
    p = model()
    p.drive = replace(p.drive, max_speed_rad_s=200, max_electrical_draw_w=1e6)
    p.suction_temperature_k = 432.0
    p.plenum_volume_m3 = 100.0
    s = p.initialize(pressure_pa=1.1e5, temperature_k=432.0, angular_velocity_rad_s=167.8)
    r = p.advance(s, 0.1, mode="run", speed_control_mode="manual", motor_torque_nm=400)
    assert r["state"]["angular_velocity_rad_s"] <= 168.0
    assert r["state"]["angular_velocity_rad_s"] > 167.9
    assert any(x["motor_limited"] for x in r["steps"])
    audit(p, s, r)
