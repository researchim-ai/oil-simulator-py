"""Actual implicit PI/actuator/PR-plant coupling, including ideal tau=0."""

import pytest
import torch
from test_surface_pr_motor_control import setup as motor_setup
from test_surface_pr_surface_control import controlled

from surface.pr_surface_collocation import PRSurfaceCollocation, _ControlStages


@pytest.mark.parametrize("feedback", [False, True])
def test_valve_and_gravitational_plant_share_stages(feedback):
    model, state = controlled(feedback=feedback)
    solver = PRSurfaceCollocation(model, phase_fallback=False)
    result = solver.advance(state, 1e-7, **({} if feedback else {"targets": {"v": 1.0}}))
    assert result["fallback_intervals"] == 0
    assert max(map(abs, result["conservation_residuals"])) < 1e-6
    step = result["steps"][0]
    assert step["method"] == "sdirk2-common"
    if not feedback:
        expected = model.valves[0].actuator.advance(0.5, 1.0, 1e-7)["final_position"]
        assert dict(result["state"]["positions"])["v"] == pytest.approx(expected, abs=1e-12)


def test_ideal_motor_speed_pi_is_implicit_without_artificial_time_constant():
    model, state, _ = motor_setup()
    assert model.motors[0].actuator.time_constant_s == 0
    solver = PRSurfaceCollocation(model, phase_fallback=False)
    result = solver.advance(state, 1e-5)
    assert result["fallback_intervals"] == 0
    assert max(map(abs, result["conservation_residuals"])) < 1e-7
    step = result["steps"][0]["plant"]
    stages = step["transport"]["stage_couplings"]
    for stage in stages:
        c = stage["controls"]
        assert abs(c["velocities"][0]) <= 100
        target = min(1.0, max(0.0, 0.5 - 0.01 * (c["measurements"][0] - 101.0) + c["integrals"][0]))
        assert c["targets"][0] == pytest.approx(target, abs=1e-12)
    assert dict(result["state"]["plant"]["transport"]["rotors"])["booster"] > 100


def test_implicit_valve_feedback_jacobian_and_sparse_support():
    model, state = controlled(feedback=True)
    solver = PRSurfaceCollocation(model, phase_fallback=False)
    controls = _ControlStages(model, state, {}, {}, {})
    prepared, _ = solver.plant._problem(
        state.plant, 2, {"port": True}, {}, {}, {}, _controls=controls
    )
    y, _, _, evaluate, rates, *_ = prepared

    def fn(z):
        row, point = evaluate(z)
        return row - 1e-7 * rates(point)

    jac = torch.func.jacfwd(fn)(y)
    direction = torch.linspace(-1e-3, 1e-3, len(y), dtype=y.dtype)
    eps = 1e-5
    fd = (fn(y + eps * direction) - fn(y - eps * direction)) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=1e-4, atol=2e-6)
    mask = torch.zeros_like(jac, dtype=torch.bool)
    mask[evaluate.pattern] = True
    assert not bool((jac[~mask].abs() > 1e-10).any())


def test_second_order_actuator_against_exact_exponential_on_isolated_plant():
    import math

    model, old = controlled()
    solver = PRSurfaceCollocation(model, phase_fallback=False)
    errors = []
    duration = 0.01
    expected = 0.55 - 0.05 * math.exp(-duration / 0.1)
    for count in (1, 2, 4):
        state = old
        for _ in range(count):
            state, _ = solver._step(
                state, duration / count, {"v": 0.55}, {}, {"openings": {"port": False}}
            )
        errors.append(abs(dict(state.positions)["v"] - expected))
    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.8


def test_runner_explicit_common_integrator_and_unsupported_trips_are_visible():
    from run_pr_controlled_surface import run

    from surface.pr_control_channels import PRSurfaceTrip

    model, state = controlled(feedback=True)
    request = {
        "model": model.to_dict(),
        "initial": {"state": state.to_dict()},
        "schedule": [{"duration_s": 1e-7}],
    }
    result = run(request, integrator="sdirk2-common")
    assert result["numerical_policy"]["surface_integrator"] == "sdirk2-common"
    assert result["intervals"][0]["fallback_intervals"] == 0
    tripped, _, _ = motor_setup(
        trips=(PRSurfaceTrip("trip", "motor", "speed", "booster", 110, 90),)
    )
    with pytest.raises(ValueError, match="protection"):
        PRSurfaceCollocation(tripped)


def test_independent_stage_audit_rejects_forged_control_or_endpoint():
    import copy
    import json

    from audit_pr_surface_collocation import audit
    from run_pr_controlled_surface import run

    model, state = controlled(feedback=True)
    request = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"state": state.to_dict()},
                "schedule": [{"duration_s": 1e-7}],
            }
        )
    )
    result = json.loads(json.dumps(run(request, integrator="sdirk2-common")))
    assert audit(request, result)["accepted"]
    broken = copy.deepcopy(result)
    broken["intervals"][0]["steps"][0]["plant"]["transport"]["stage_couplings"][0]["controls"][
        "velocities"
    ][0] += 0.1
    with pytest.raises(ValueError):
        audit(request, broken)
    broken = copy.deepcopy(result)
    broken["intervals"][0]["steps"][0]["state"]["positions"]["v"] += 0.1
    with pytest.raises(ValueError):
        audit(request, broken)


def test_closed_seat_uses_explicit_auditable_legacy_fallback():
    model, state = controlled()
    state = model.initialize(state.plant, {"v": 0.0})
    result = PRSurfaceCollocation(model).advance(state, 1e-7, targets={"v": 0.0})
    assert result["fallback_intervals"] == 1
    assert all(r["method"] == "controlled_backward_euler_fallback" for r in result["steps"])
    assert max(map(abs, result["conservation_residuals"])) < 1e-6
