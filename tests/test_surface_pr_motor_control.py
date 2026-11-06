"""Common-plant torque feedback, latched trips and conservative shutdown."""

import json
from pathlib import Path

import pytest

from surface.actuator import ConnectionActuator
from surface.pr_control_channels import PRMotorControl, PRSurfaceTrip
from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork
from surface.pr_gravity_transport import PRGravityTransportNetwork
from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState, PRSurfacePI

ROOT = Path(__file__).resolve().parents[1]


def setup(trips=(), feedback=True):
    config = json.loads((ROOT / "configs/surface_pr_driven_pump.json").read_text())
    transport = PREquilibriumTransportNetwork.from_dict(config["model"])
    plant = PRGravityTransportNetwork(transport, {}, [])
    controls = (
        [PRSurfacePI("motor", 101.0, 0.01, 0.01, measurement="speed", action=-1)]
        if feedback
        else []
    )
    model = PRControlledSurface(
        plant,
        [],
        controls,
        motors=[
            PRMotorControl(
                "motor",
                "booster",
                ConnectionActuator(100.0, 100.0, 0.0, max_position_change=1.0),
                0.0,
                4.0,
            )
        ],
        trips=trips,
        initial_step_s=1e-5,
        maximum_step_s=0.001,
    )
    physical = plant.initialize(transport.initialize(config["initial"]["nodes"]), {}, {})
    return model, model.initialize(physical, {"motor": 0.5}), config["initial"]["nodes"]


def test_motor_coordinate_is_a_bounded_physical_torque():
    motor = PRMotorControl("m", "shaft", ConnectionActuator(1.0, 1.0, 0.0), -2.0, 4.0)
    assert motor.torque(0.0) == -2.0
    assert motor.torque(1 / 3) == pytest.approx(0.0)
    assert motor.torque(1.0) == 4.0
    with pytest.raises(ValueError):
        motor.torque(1.1)


def test_debounce_latch_and_hysteresis_have_independent_sampled_semantics():
    trip = PRSurfaceTrip("speed", "motor", "speed", "booster", 110.0, 90.0, delay_s=0.03)
    assert trip.step(False, 0.0, 120.0, 0.01) == (False, 0.01, False)
    assert trip.step(False, 0.01, 120.0, 0.02) == (True, 0.03, False)
    assert trip.step(True, 0.03, 80.0, 0.02) == (True, 0.03, True)
    assert trip.step(False, 0.01, 100.0, 0.01) == (False, 0.0, False)
    assert trip.can_reset(80.0) and not trip.can_reset(100.0)


def test_speed_pi_changes_real_rotor_and_preserves_whole_plant_energy():
    m, s, _ = setup()
    n, r = m._step(s, 1e-5, {}, {}, {})
    assert r["measurements"]["motor"] == 100.0
    assert r["effective_motor_torque_nm"]["motor"] == pytest.approx(4 * dict(n.positions)["motor"])
    assert dict(n.plant.transport.rotors)["booster"] != 100.0
    assert max(map(abs, r["plant"]["conservation_residuals"])) < 1e-7
    restored = type(m).from_dict(json.loads(json.dumps(m.to_dict())))
    restored.inspect(type(n).from_dict(json.loads(json.dumps(n.to_dict()))))
    with pytest.raises(ValueError, match="second torque"):
        m.advance(s, 1e-5, drive_torque_nm={"booster": 2.0})


def test_immediate_trip_overrides_manual_target_and_tracks_actual_output():
    trip = PRSurfaceTrip(
        "overspeed", "motor", "speed", "booster", 90.0, 80.0, isolate_switches=("booster",)
    )
    m, s, _ = setup((trip,))
    n, r = m._step(s, 1e-5, {"motor": 1.0}, {}, {})
    assert r["actuators"]["motor"]["mode"] == "fail_safe"
    assert not r["plant"]["openings"]["booster"]
    assert {name: latched for name, latched, _ in n.trips}["overspeed"]
    c = m.controllers[0]
    proportional = c.proportional_gain * c.action * (r["measurements"]["motor"] - c.setpoint)
    assert dict(n.integrals)["motor"] == pytest.approx(
        dict(n.positions)["motor"] - c.opening_bias - proportional
    )
    before = n.to_dict()
    with pytest.raises(ValueError, match="hysteresis"):
        m.advance(n, 1e-5, reset_trips=["overspeed"])
    assert n.to_dict() == before
    _, stuck = m._step(s, 1e-5, {"motor": 1.0}, {"motor": "stuck"}, {})
    assert stuck["protections"]["overspeed"]["actuator_blocked"]
    assert stuck["actuators"]["motor"]["final_position"] == 0.5


def test_trip_delay_is_an_interval_boundary_and_survives_restart():
    trip = PRSurfaceTrip(
        "delayed",
        "motor",
        "speed",
        "booster",
        90.0,
        80.0,
        delay_s=1e-5,
        isolate_switches=("booster",),
    )
    m, s, _ = setup((trip,))
    report = m.advance(s, 3e-5)
    active = [step for step in report["steps"] if step["protections"]["delayed"]["active"]]
    assert active
    assert active[0]["state"]["plant"]["transport"]["time_s"] - active[0]["dt_s"] == pytest.approx(
        1e-5, abs=1e-13
    )
    last = PRControlledSurfaceState.from_dict(report["state"])
    restored = type(m).from_dict(json.loads(json.dumps(m.to_dict())))
    again = restored.advance(
        PRControlledSurfaceState.from_dict(json.loads(json.dumps(last.to_dict()))), 1e-5
    )
    assert again["state"]["trips"]["delayed"]["latched"]
    assert all(not x["plant"]["openings"]["booster"] for x in again["steps"])


def test_reset_requires_hysteresis_and_cannot_clear_another_active_protection():
    cold = PRSurfaceTrip(
        "cold",
        "motor",
        "temperature",
        "feed",
        350.000001,
        350.000002,
        direction="low",
        isolate_switches=("booster",),
    )
    speed = PRSurfaceTrip(
        "speed", "motor", "speed", "booster", 90.0, 80.0, isolate_switches=("booster",)
    )
    m, s, _ = setup((cold, speed))
    heated = m.advance(s, 0.0001, heat_w={"feed": 1000.0}, utility_temperature_k={"feed": 400.0})
    state = PRControlledSurfaceState.from_dict(heated["state"])
    assert dict(state.plant.transport.nodes)["feed"].temperature_k > cold.reset_threshold
    continued = m.advance(state, 1e-5, reset_trips=["cold"])
    assert not continued["state"]["trips"]["cold"]["latched"]
    assert continued["state"]["trips"]["speed"]["latched"]
    assert all(step["actuators"]["motor"]["mode"] == "fail_safe" for step in continued["steps"])


def test_independent_motor_and_protection_auditor_rejects_damaged_latch():
    from check_pr_controlled_surface import audit
    from run_pr_controlled_surface import run

    trip = PRSurfaceTrip(
        "delayed",
        "motor",
        "speed",
        "booster",
        90.0,
        80.0,
        delay_s=1e-5,
        isolate_switches=("booster",),
    )
    m, _, initial = setup((trip,))
    cfg = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": {
                    "plant": {"transport": initial, "columns": {}, "wall_temperatures": {}},
                    "positions": {"motor": 0.5},
                    "integrals": {},
                },
                "schedule": [{"duration_s": 3e-5}],
            }
        )
    )
    r = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, r)["accepted"]
    damaged = json.loads(json.dumps(r))
    damaged["intervals"][0]["steps"][-1]["state"]["trips"]["delayed"]["latched"] = False
    with pytest.raises(ValueError):
        audit(cfg, damaged)


def test_flow_feedback_uses_solved_pump_flow_and_bootstraps_without_invented_reading():
    from check_pr_controlled_surface import audit
    from run_pr_controlled_surface import run

    m, s, initial = setup()
    pi = PRSurfacePI("motor", 5e-5, 1e2, 1e2, measurement="pump_flow", sensor="booster", action=-1)
    controlled = PRControlledSurface(m.plant, [], [pi], motors=m.motors, initial_step_s=1e-5)
    start = controlled.initialize(s.plant, {"motor": 0.5})
    assert controlled.measure(start, pi) is None
    cfg = json.loads(
        json.dumps(
            {
                "model": controlled.to_dict(),
                "initial": {
                    "plant": {"transport": initial, "columns": {}, "wall_temperatures": {}},
                    "positions": {"motor": 0.5},
                    "integrals": {},
                },
                "schedule": [{"duration_s": 1e-5}],
            }
        )
    )
    result = json.loads(json.dumps(run(cfg)))
    report = result["intervals"][0]
    assert report["steps"][0]["measurements"]["motor"] is None
    assert any(step["measurements"]["motor"] is not None for step in report["steps"][1:])
    last = report["steps"][-1]
    assert (
        report["state"]["machine_feedback"]["booster"]
        == last["plant"]["steps"][-1]["transport"]["pump_connections"]["booster"]["flow_m3_s"]
    )
    assert audit(cfg, result)["accepted"]
    restart = {**cfg, "initial": {"state": result["intervals"][-1]["state"]}}
    assert audit(restart, json.loads(json.dumps(run(restart))))["accepted"]
    damaged = json.loads(json.dumps(result))
    damaged["intervals"][0]["steps"][-1]["state"]["machine_feedback"]["booster"] *= 1.1
    with pytest.raises(ValueError):
        audit(cfg, damaged)


def test_antisurge_recycle_uses_live_map_and_bumpless_off_on_feedback():
    from check_pr_controlled_surface import audit
    from run_pr_controlled_surface import run

    cfg = json.loads((ROOT / "configs/surface_pr_controlled_antisurge.json").read_text())
    r = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, r)["accepted"]
    assert r["intervals"][0]["steps"][0]["measurements"]["recycle"] is not None
    assert all(s["measurements"]["recycle"] is None for s in r["intervals"][1]["steps"])
    assert r["intervals"][1]["state"]["inactive_feedback"] == ["recycle"]
    assert "inactive_feedback" not in r["intervals"][2]["state"]
    before = r["intervals"][1]["state"]["positions"]["recycle"]
    resumed = r["intervals"][2]["steps"][0]
    assert abs(resumed["actuators"]["recycle"]["command"] - before) < 1e-7
