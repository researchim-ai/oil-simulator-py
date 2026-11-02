"""Coupled shaft/network versus independent hydraulics and ODE limits."""

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    DynamicPumpNetwork,
    EquipmentOperatingError,
    Liquid,
    LiquidNetwork,
    Node,
    PIDController,
    Pipe,
    Pump,
    PumpDriveState,
    RotatingDrive,
)


def model(dt=0.1, controller=None, **drive_options):
    network = LiquidNetwork(
        Liquid(1000, 0.1, 2000),
        [Node("in", pressure_pa=2e5), Node("delivery"), Node("out", pressure_pa=2.5e5)],
        [Pipe("line", "delivery", "out", 100, 0.1)],
        [Pump("pump", "in", "delivery", 20, 1e6, 0.004, efficiency=0.8, check_valve=True)],
    )
    drive = RotatingDrive(
        **{
            "inertia_kg_m2": 1,
            "max_speed_rad_s": 150,
            "max_motor_torque_nm": 20,
            "max_braking_torque_nm": 10,
            "viscous_drag_nm_s": 0.02,
            **drive_options,
        }
    )
    return DynamicPumpNetwork(
        network,
        drive,
        100,
        controller,
        measurement_node="delivery",
        control_interval_s=0.2,
        maximum_dt_s=dt,
        max_speed_change_rad_s=10,
    )


def audit(result):
    for s in result["steps"]:
        assert abs(s["shaft_energy_residual_j"]) < 1e-5
        assert abs(s["energy_residual_j"]) < 1e-4
        assert (
            min(s[k] for k in ("motor_heat_j", "pump_heat_j", "drag_heat_j", "passive_heat_j"))
            >= -1e-8
        )
        assert max(abs(x) for x in s["endpoint"]["node_mass_residuals_kg_s"].values()) < 1e-8
    assert math.fsum(s["dt_s"] for s in result["steps"]) == pytest.approx(
        math.fsum(s["dt_s"] for s in result["control_samples"]), abs=1e-12
    )


def independent_load(speed):
    # Independent quadratic pump + Poiseuille resistance, no simulator calls.
    rise = 1000 * 9.80665 * 20 * (speed / 100) ** 2 - 50000
    if rise <= 0:
        return 0
    resistance = 128 * 0.1 * 100 / (math.pi * 0.1**4)
    quadratic = 1000 * 9.80665 * 1e6
    flow = 2 * rise / (resistance + math.sqrt(resistance**2 + 4 * quadratic * rise))
    head_pressure = 50000 + resistance * flow
    return head_pressure * flow / (0.8 * speed)


def test_midpoint_refines_against_independent_nonlinear_ode():
    reference = solve_ivp(
        lambda t, y: [6 - 0.02 * y[0] - independent_load(y[0])],
        (0, 4),
        [60.0],
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    ).y[0, -1]
    errors = []
    for dt in (0.2, 0.1, 0.05):
        p = model(dt)
        r = p.advance(p.initialize(60), 4, mode="manual", motor_torque_nm=6)
        audit(r)
        errors.append(abs(r["state"]["angular_velocity_rad_s"] - reference))
    assert errors[-1] < 2e-5
    assert errors[0] / errors[1] > 3.5 and errors[1] / errors[2] > 3.5


def test_startup_load_not_frozen_and_coast_reduces_flow():
    p = model()
    r = p.advance(p.initialize(), 6, mode="manual", motor_torque_nm=12)
    audit(r)
    assert r["state"]["angular_velocity_rad_s"] > 50
    assert max(s["pump_load_torque_nm"] for s in r["steps"]) > 0
    assert min(s["pump_load_torque_nm"] for s in r["steps"]) == 0
    coast = p.advance(PumpDriveState.from_dict(r["state"]), 5, mode="off")
    audit(coast)
    assert coast["state"]["angular_velocity_rad_s"] < r["state"]["angular_velocity_rad_s"]
    assert (
        coast["endpoint"]["pump_volume_rates_m3_s"]["pump"]
        < r["endpoint"]["pump_volume_rates_m3_s"]["pump"]
    )
    assert all(s["electrical_energy_j"] == 0 for s in coast["steps"])


def test_braking_hits_rest_and_holds_without_exporting_fictitious_energy():
    p = model(regeneration_efficiency=0.7, viscous_drag_nm_s=0)
    r = p.advance(p.initialize(20), 5, mode="manual", motor_torque_nm=-10)
    audit(r)
    assert r["state"]["angular_velocity_rad_s"] == 0
    assert r["endpoint"]["pump_volume_rates_m3_s"]["pump"] == 0
    assert sum(s["electrical_energy_j"] for s in r["steps"]) == pytest.approx(-140, abs=1e-7)


def test_drive_power_and_speed_limits_preserve_work_balance():
    p = model(max_electrical_draw_w=150, max_speed_rad_s=70)
    r = p.advance(p.initialize(40), 30, mode="manual", motor_torque_nm=20)
    audit(r)
    assert r["state"]["angular_velocity_rad_s"] <= 70
    assert any(s["torque_limited"] for s in r["steps"])
    for s in r["steps"]:
        assert s["motor_torque_nm"] * max(
            s["initial_speed_rad_s"], s["final_speed_rad_s"]
        ) <= 150 * (1 + 1e-10)
    p = model(max_speed_rad_s=55)
    r = p.advance(p.initialize(54), 2, mode="manual", motor_torque_nm=20)
    assert r["state"]["angular_velocity_rad_s"] == pytest.approx(55)
    assert any(s["speed_limited"] for s in r["steps"])
    audit(r)


def test_controller_sampling_is_independent_of_plant_retry():
    pid = PIDController(0, 0, 0, -10, 20)
    p = model(0.2, pid)
    p.max_speed_change_rad_s = 0.05
    r = p.advance(p.initialize(), 0.4, mode="manual", motor_torque_nm=20)
    assert len(r["control_samples"]) == 2
    assert r["failed_steps"] > 0
    assert r["state"]["time_s"] == 0.4
    audit(r)


def test_original_pump_operating_limits_are_not_disabled_on_accepted_solution():
    p = model()
    pump = replace(p.pump, max_shaft_power_w=1.0)
    net = LiquidNetwork(p.network.fluid, p.network.nodes, p.network.pipes, [pump])
    p = DynamicPumpNetwork(net, p.drive, 100, measurement_node="delivery")
    with pytest.raises(EquipmentOperatingError):
        p.initialize(100)


def test_pi_pressure_feedback_reaches_target_and_manual_transfer_is_bumpless():
    pid = PIDController(0.003, 0.0003, 0, -10, 20, tracking_time_s=0.3)
    p = model(0.1, pid)
    r = p.advance(p.initialize(), 60, setpoint=260000)
    audit(r)
    assert abs(r["endpoint"]["pressures_pa"]["delivery"] - 260000) < 100
    manual = p.advance(PumpDriveState.from_dict(r["state"]), 0.4, mode="manual", motor_torque_nm=3)
    auto = p.advance(PumpDriveState.from_dict(manual["state"]), 0.4, setpoint=262000)
    assert auto["control_samples"][0]["output"] == manual["state"]["applied_torque_nm"]


def test_emergency_trip_latches_until_explicit_reset():
    pid = PIDController(0.003, 0.0003, 0, -10, 20)
    p = model(0.1, pid)
    r = p.advance(p.initialize(80), 0.4, mode="trip")
    assert r["state"]["tripped"]
    r = p.advance(PumpDriveState.from_dict(r["state"]), 0.4, mode="auto", setpoint=260000)
    assert r["effective_mode"] == "off"
    assert all(s["motor_torque_nm"] == 0 for s in r["steps"])
    r = p.advance(
        PumpDriveState.from_dict(r["state"]), 0.4, mode="auto", setpoint=260000, reset_trip=True
    )
    assert not r["state"]["tripped"]
    assert r["effective_mode"] == "auto"


def test_period_restart_checks_controller_work_and_actuator_memory():
    import copy
    import json

    from surface import ConservationError, run_pump_drive_schedule
    from surface.operation import _digest

    pid = PIDController(0.003, 0.0003, 0, -10, 20)
    p = model(0.1, pid)
    initial = p.initialize(55)
    periods = [
        {"duration_s": 0.4, "mode": "manual", "motor_torque_nm": 4},
        {"duration_s": 0.4, "mode": "auto", "setpoint": 260000},
        {"duration_s": 0.4, "mode": "trip"},
    ]
    full = run_pump_drive_schedule(p, initial, periods)
    partial = run_pump_drive_schedule(p, initial, periods, stop_after=1)
    resumed = run_pump_drive_schedule(p, initial, periods, json.loads(json.dumps(partial)))
    assert resumed == full
    for key in ("electrical_energy_j", "pump_heat_j"):
        bad = copy.deepcopy(partial)
        bad["records"][0]["steps"][0][key] += 1
        bad["checksum"] = _digest({k: v for k, v in bad.items() if k != "checksum"})
        with pytest.raises(ConservationError, match="audit"):
            run_pump_drive_schedule(p, initial, periods, bad)
    bad = copy.deepcopy(partial)
    bad["records"][0]["state"]["applied_torque_nm"] = 5
    bad["state"]["applied_torque_nm"] = 5
    bad["checksum"] = _digest({k: v for k, v in bad.items() if k != "checksum"})
    with pytest.raises(ConservationError, match="memory"):
        run_pump_drive_schedule(p, initial, periods, bad)
    bad = copy.deepcopy(partial)
    bad["records"][0]["control_samples"][-1]["state"]["integral_output"] += 1
    bad["records"][0]["state"]["controller_state"]["integral_output"] += 1
    bad["state"]["controller_state"]["integral_output"] += 1
    bad["checksum"] = _digest({k: v for k, v in bad.items() if k != "checksum"})
    with pytest.raises(ConservationError, match="controller replay"):
        run_pump_drive_schedule(p, initial, periods, bad)
    with pytest.raises(ValueError, match="fingerprint"):
        run_pump_drive_schedule(model(0.05, pid), model(0.05, pid).initialize(55), periods, partial)


def test_zero_drag_startup_and_ulp_control_clock_finish_exact_interval():
    p = model(0.1, viscous_drag_nm_s=0)
    r = p.advance(p.initialize(15.123), 1.3, mode="manual", motor_torque_nm=6)
    assert r["state"]["angular_velocity_rad_s"] == pytest.approx(15.123 + 6 * 1.3, abs=1e-9)
    assert r["state"]["time_s"] == 1.3
    assert len(r["control_samples"]) == 7
    audit(r)


def test_declared_no_flow_loss_loads_rotating_blocked_pump_and_is_audited():
    base = model()
    p = DynamicPumpNetwork(
        base.network, base.drive, 100, measurement_node="delivery", no_flow_power_w_at_nominal=100.0
    )
    a, ta = base._hydraulics(20, {})
    b, tb = p._hydraulics(20, {})
    assert b.pump_volume_rates_m3_s["pump"] == a.pump_volume_rates_m3_s["pump"] == 0
    assert tb - ta == pytest.approx(0.8 / 20)
    r = p.advance(p.initialize(20), 0.4, mode="off")
    audit(r)
    assert sum(s["pump_heat_j"] for s in r["steps"]) > 0


def test_zero_electrical_capacity_cannot_accelerate_the_shaft():
    p = model(max_electrical_draw_w=0)
    r = p.advance(p.initialize(), 0.4, mode="manual", motor_torque_nm=20)
    assert r["state"]["angular_velocity_rad_s"] == 0
    assert sum(s["electrical_energy_j"] for s in r["steps"]) == 0
    audit(r)


def test_tabulated_head_curve_is_used_inside_coupled_load_solution():
    from surface import EquipmentCurve

    base = model()
    pump = replace(base.pump, head_curve=EquipmentCurve((0.0, 0.004), (20.0, 0.0)))
    network = LiquidNetwork(base.network.fluid, base.network.nodes, base.network.pipes, [pump])
    p = DynamicPumpNetwork(network, base.drive, 100, measurement_node="delivery", maximum_dt_s=0.1)
    r = p.advance(p.initialize(80), 0.4, mode="manual", motor_torque_nm=4)
    resistance = 128 * 0.1 * 100 / (math.pi * 0.1**4)
    for step in r["steps"]:
        ratio = 0.5 * (step["initial_speed_rad_s"] + step["final_speed_rad_s"]) / 100
        expected = max(
            0.0,
            (1000 * 9.80665 * 20 * ratio**2 - 50000) / (resistance + 1000 * 9.80665 * 5000 * ratio),
        )
        assert step["midpoint_flow_m3_s"] == pytest.approx(expected, abs=1e-10)
    audit(r)


def test_external_boundary_disturbance_changes_flow_without_hidden_state_mutation():
    p = model()
    initial = p.initialize(80)
    normal = p.advance(initial, 0.2, mode="manual", motor_torque_nm=4)
    higher = p.advance(
        initial,
        0.2,
        mode="manual",
        motor_torque_nm=4,
        network_options={"boundary_pressures_pa": {"out": 270000.0}},
    )
    assert (
        higher["endpoint"]["pump_volume_rates_m3_s"]["pump"]
        < normal["endpoint"]["pump_volume_rates_m3_s"]["pump"]
    )
    assert p.network.nodes[-1].pressure_pa == 250000.0
    audit(higher)


def test_optional_null_boundary_overrides_match_no_override():
    pid = PIDController(0.003, 0.0003, 0, -10, 20)
    p = model(0.1, pid)
    state = p.initialize(60)
    normal = p.advance(state, 0.2, setpoint=260000)
    null = p.advance(
        state,
        0.2,
        setpoint=260000,
        network_options={"boundary_pressures_pa": None, "valve_openings": None},
    )
    assert null == normal
