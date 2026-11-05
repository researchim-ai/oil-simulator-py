"""Valve resistance, actual actuator tracking and conservative coupled control."""

import json
import math

import pytest
from test_surface_pr_gravity_transport import setup

from surface.actuator import ConnectionActuator
from surface.pr_surface_control import (
    PRControlledSurface,
    PRControlledSurfaceState,
    PRControlValve,
    PRSurfacePI,
)


def controlled(feedback=False):
    plant, s, _ = setup()
    valve = PRControlValve("v", "port", 1.0, ConnectionActuator(1.0, 1.0, 0.0, time_constant_s=0.1))
    pi = [PRSurfacePI("v", 1e6, 1e-6, 1e-8)] if feedback else []
    m = PRControlledSurface(
        plant, [valve], pi, initial_step_s=1e-7, maximum_step_s=1e-6, minimum_step_s=1e-10
    )
    return m, m.initialize(s, {"v": 0.5})


def test_valve_coefficient_is_a_positive_passive_effective_area_law():
    a = ConnectionActuator(1.0, 1.0, 0.0)
    linear = PRControlValve("v", "p", 2.0, a)
    assert linear.coefficient(1.0) == 2.0
    assert linear.coefficient(0.5) == 8.0
    assert linear.coefficient(0.0) is None
    equal = PRControlValve(
        "v", "p", 2.0, a, characteristic="modified_equal_percentage", rangeability=50.0
    )
    fraction = (50.0**0.5 - 1) / 49
    assert equal.coefficient(0.5) == pytest.approx(2 / fraction**2)
    with pytest.raises(ValueError):
        linear.coefficient(-1.0)


def test_pi_tracking_has_independent_exact_solution_and_bumpless_manual():
    c = PRSurfacePI("v", 1e6, 1e-6, 1e-8, tracking_time_s=2.0)
    measurement = 1.1e6
    old = 0.2
    position = 0.1
    dt = 0.3
    error = measurement - c.setpoint
    equilibrium = (
        2 * c.integral_gain_per_s * error + position - c.opening_bias - c.proportional_gain * error
    )
    expected = equilibrium + (old - equilibrium) * math.exp(-dt / 2)
    target, new, _ = c.update(measurement, old, position, dt)
    assert new == pytest.approx(expected)
    assert target == pytest.approx(c.opening_bias + c.proportional_gain * error + new)
    manual, integral, _ = c.update(measurement, old, position, dt, manual=0.7)
    assert manual == 0.7
    automatic, _, _ = c.update(measurement, integral, 0.7, 0.0)
    assert automatic == pytest.approx(0.7)
    for _ in range(100):
        _, integral, _ = c.update(1e9, integral, 0.0, 1.0)
    assert math.isfinite(integral) and abs(integral) <= c.maximum_integral_abs


def test_controlled_step_matches_actuator_and_preserves_plant_energy():
    m, s = controlled()
    new, r = m._step(s, 1e-7, {"v": 1.0}, {}, {})
    reference = m.valves[0].actuator.advance(0.5, 1.0, 1e-7)["final_position"]
    assert dict(new.positions)["v"] == pytest.approx(reference, abs=1e-15)
    assert r["effective_valve_K"]["v"] == pytest.approx(1 / reference**2)
    assert max(map(abs, r["plant"]["conservation_residuals"])) < 1e-5
    assert r["valve_envelope"]["v"]["maximum_pressure_drop_fraction"] <= 0.05
    restored = PRControlledSurface.from_dict(json.loads(json.dumps(m.to_dict())))
    restored.inspect(PRControlledSurfaceState.from_dict(json.loads(json.dumps(new.to_dict()))))


def test_pressure_feedback_and_fail_safe_are_stateful():
    m, s = controlled(feedback=True)
    new, r = m._step(s, 1e-7, {}, {}, {})
    assert "v" in r["measurements"]
    assert math.isfinite(dict(new.integrals)["v"])
    # A fully closed failed valve has exactly zero material/energy transfer.
    closed = m.initialize(s.plant, {"v": 0.0})
    stopped, r = m._step(closed, 1e-7, {}, {"v": "fail_safe"}, {})
    assert dict(stopped.positions)["v"] == 0.0
    for step in r["plant"]["steps"]:
        assert step["columns"]["ports"]["port"]["flux"][:-2] == [0.0, 0.0]
        assert step["columns"]["ports"]["port"]["flux"][-1] == 0.0


def test_controlled_interval_exact_time_restart_and_atomic_invalid_command():
    m, s = controlled()
    original = s.to_dict()
    r = m.advance(s, 1e-7, targets={"v": 0.5})
    new = PRControlledSurfaceState.from_dict(r["state"])
    assert new.time_s == 1e-7
    restored = PRControlledSurface.from_dict(json.loads(json.dumps(m.to_dict())))
    a = m.advance(new, 1e-7, targets={"v": 0.5})
    b = restored.advance(
        PRControlledSurfaceState.from_dict(json.loads(json.dumps(new.to_dict()))),
        1e-7,
        targets={"v": 0.5},
    )
    assert a == b
    with pytest.raises(ValueError):
        m.advance(s, 1e-7, targets={"unknown": 1.0})
    assert s.to_dict() == original


def test_level_feedback_uses_actual_phase_inventory_and_interface():
    plant, s, _ = setup(temperature=196.2)
    valve = PRControlValve("v", "port", 1.0, ConnectionActuator(1.0, 1.0, 0.0))
    pi = PRSurfacePI("v", 0.01, 1.0, measurement="level", component="Ethane")
    m = PRControlledSurface(plant, [valve], [pi])
    state = m.initialize(s, {"v": 0.5})
    assert m.measure(state, pi) == pytest.approx(s.columns[0][1].interface_heights_m[0], abs=1e-12)


def test_lower_opening_reduces_actual_flow_with_conserved_energy():
    plant, s, _ = setup()
    valve = PRControlValve("v", "port", 1e8, ConnectionActuator(1.0, 1.0, 0.0))
    m = PRControlledSurface(plant, [valve])
    flows = []
    for opening in [1.0, 0.5]:
        state = m.initialize(s, {"v": opening})
        _new, r = m._step(state, 1e-7, {}, {"v": "stuck"}, {})
        flows.append(r["plant"]["steps"][-1]["columns"]["ports"]["port"]["flux"][0])
        assert max(map(abs, r["plant"]["conservation_residuals"])) < 1e-5
    assert 0 < flows[1] < flows[0]


def test_valve_domain_rejection_is_atomic():
    from surface.liquid import PropertyModelError

    plant, s, _ = setup()
    valve = PRControlValve(
        "v", "port", 1e8, ConnectionActuator(1.0, 1.0, 0.0), maximum_pressure_drop_fraction=1e-12
    )
    m = PRControlledSurface(plant, [valve])
    state = m.initialize(s, {"v": 1.0})
    original = state.to_dict()
    with pytest.raises(PropertyModelError, match="low-pressure-drop"):
        m._step(state, 1e-7, {}, {"v": "stuck"}, {})
    assert state.to_dict() == original


def test_independent_control_auditor_does_not_replay_runtime_and_rejects_tampering(monkeypatch):
    import copy
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))

    def load(name):
        spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    runner = load("run_pr_controlled_surface")
    checker = load("check_pr_controlled_surface")
    config = json.loads((root / "configs/surface_pr_controlled_surface.json").read_text())
    config["schedule"] = config["schedule"][:1]
    report = json.loads(json.dumps(runner.run(config)))

    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime dynamics called by independent controller auditor")

    monkeypatch.setattr(PRControlledSurface, "_step", forbidden)
    monkeypatch.setattr(PRControlledSurface, "measure", forbidden)
    monkeypatch.setattr(PRSurfacePI, "update", forbidden)
    monkeypatch.setattr(ConnectionActuator, "advance", forbidden)
    assert checker.audit(config, report)["accepted"]
    broken = copy.deepcopy(report)
    broken["intervals"][0]["steps"][0]["state"]["positions"]["v"] += 0.01
    with pytest.raises(ValueError):
        checker.audit(config, broken)
