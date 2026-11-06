"""Signed speed/flow feedback and low-speed trips remain capability checked."""

import json
from pathlib import Path

import pytest

from surface.actuator import ConnectionActuator
from surface.pr_control_channels import PRMotorControl, PRSurfaceTrip
from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork
from surface.pr_gravity_transport import PRGravityTransportNetwork
from surface.pr_surface_control import PRControlledSurface, PRSurfacePI

ROOT = Path(__file__).resolve().parents[1]


def setup_signed():
    cfg = json.loads((ROOT / "configs/surface_pr_signed_pump_reversal.json").read_text())
    raw = cfg["model"]
    raw["drives"][0]["initial_speed_rad_s"] = -0.1
    physics = raw["drives"][0]["drive"]
    physics["max_electrical_draw_w"] = physics["max_electrical_export_w"] = 1000.0
    model = PREquilibriumTransportNetwork.from_dict(raw)
    plant = PRGravityTransportNetwork(model, {}, [])
    motor = PRMotorControl(
        "motor",
        "booster",
        ConnectionActuator(100.0, 100.0, 0.5, max_position_change=1.0),
        -2.0,
        2.0,
    )
    return cfg, plant, motor


def test_negative_speed_feedback_and_envelope_are_independently_audited():
    from check_pr_controlled_surface import audit
    from run_pr_controlled_surface import run

    cfg, plant, motor = setup_signed()
    control = PRControlledSurface(
        plant,
        [],
        [PRSurfacePI("motor", -0.2, 0.01, 0.01, measurement="speed", action=-1)],
        motors=[motor],
        initial_step_s=1e-6,
    )
    request = {
        "model": control.to_dict(),
        "initial": {
            "plant": {"transport": cfg["initial"]["nodes"], "columns": {}, "wall_temperatures": {}},
            "positions": {"motor": 0.5},
            "integrals": {},
        },
        "schedule": [{"duration_s": 1e-6}],
    }
    result = json.loads(json.dumps(run(request)))
    first = result["intervals"][0]["steps"][0]
    assert first["measurements"]["motor"] == -0.1
    assert first["motor_envelope"]["motor"]["maximum_speed_rad_s"] >= 0.1
    assert audit(request, result)["accepted"]
    first["motor_envelope"]["motor"]["maximum_speed_rad_s"] = 0.0
    with pytest.raises(ValueError):
        audit(request, result)


def test_negative_speed_trip_has_signed_hysteresis():
    trip = PRSurfaceTrip("reverse_speed", "motor", "speed", "booster", -0.2, -0.1, direction="low")
    assert trip.unsafe(-0.3) and not trip.unsafe(0.3)
    assert trip.can_reset(0.0) and not trip.can_reset(-0.15)
    with pytest.raises(ValueError):
        PRSurfaceTrip("bad", "motor", "pressure", "body", -1.0, 0.0, direction="low")


def test_negative_setpoint_is_rejected_for_unsigned_machine():
    cfg = json.loads((ROOT / "configs/surface_pr_driven_pump.json").read_text())
    plant = PRGravityTransportNetwork(PREquilibriumTransportNetwork.from_dict(cfg["model"]), {}, [])
    motor = PRMotorControl("motor", "booster", ConnectionActuator(1.0, 1.0, 0.0), 0.0, 1.0)
    with pytest.raises(ValueError, match="setpoint"):
        PRControlledSurface(
            plant, [], [PRSurfacePI("motor", -1.0, 1.0, 1.0, measurement="speed")], motors=[motor]
        )


def test_signed_drive_checks_endpoint_speed_not_only_midpoint_map():
    from surface.pr_machine_drive import RotorBundle

    cfg, plant, _ = setup_signed()
    raw = plant.transport.to_dict()
    raw["pumps"][0]["signed_map"]["maximum_speed_rad_s"] = 100.0
    raw["drives"][0]["initial_speed_rad_s"] = 99.0
    raw["drives"][0]["motor_torque_nm"] = 20.0
    raw["drives"][0]["drive"]["max_electrical_draw_w"] = None
    raw["drives"][0]["drive"]["max_electrical_export_w"] = None
    raw["drives"][0]["drive"]["viscous_drag_nm_s"] = 0.0
    model = PREquilibriumTransportNetwork.from_dict(raw)
    state = model.initialize(cfg["initial"]["nodes"])
    bundle = RotorBundle(model, state, 0.01, model.default_openings, {}, 0)
    y = bundle.initial[0].clone()
    y[0] = 1.01
    _, _, valid, _ = bundle.evaluate(
        y, {"booster": {"shaft_power_w": y[0] * 0, "load_torque_nm": y[0] * 0}}, {}, []
    )
    assert not bool(valid)
