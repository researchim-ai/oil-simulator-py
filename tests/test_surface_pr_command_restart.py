"""Known command discontinuities reset hints, not physics or error budgets."""

import pytest
from test_surface_pr_surface_control import controlled

from surface.liquid import HydraulicConvergenceError, PropertyModelError
from surface.pr_surface_collocation import PRSurfaceCollocation


def advance(solver, state, *, target=0.6, **extra):
    report = solver.advance(state, 1e-7, targets={"v": target}, openings={"port": False}, **extra)
    return type(state).from_dict(report["state"]), report


def test_unchanged_command_carries_hint_and_changed_command_restarts():
    model, state = controlled(feedback=False)
    solver = PRSurfaceCollocation(model, phase_fallback=False)
    first, one = advance(solver, state)
    assert not one["command_step_restart"]
    second, two = advance(solver, first)
    assert not two["command_step_restart"]
    assert two["initial_step_hint_s"] == first.next_step_s
    third, three = advance(solver, second, target=0.7)
    assert three["command_step_restart"]
    assert three["initial_step_hint_s"] == min(second.next_step_s, model.initial_step_s)
    assert third.time_s == second.time_s + 1e-7
    # Rewinding is a fresh numerical context, not a consecutive schedule event.
    _, rewound = advance(solver, state, target=0.8)
    assert not rewound["command_step_restart"]


def test_empty_commands_and_explicit_normal_mode_do_not_reset():
    model, state = controlled(feedback=True)
    solver = PRSurfaceCollocation(model, phase_fallback=False)
    one = solver.advance(state, 1e-7, openings={"port": False})
    state = type(state).from_dict(one["state"])
    two = solver.advance(
        state, 1e-7, openings={"port": False}, actuator_modes={"v": "normal"}, heat_w={}
    )
    assert not two["command_step_restart"]
    assert two["initial_step_hint_s"] == state.next_step_s


@pytest.mark.parametrize("first,second", [(1, 1.0), (0.0, -0.0)])
def test_equivalent_integer_and_float_targets_do_not_reset(first, second):
    model, state = controlled(feedback=False)
    solver = PRSurfaceCollocation(model, phase_fallback=False)
    state, _ = advance(solver, state, target=first)
    _, two = advance(solver, state, target=second)
    assert not two["command_step_restart"]


def test_failed_interval_does_not_publish_event_context(monkeypatch):
    model, state = controlled(feedback=False)
    solver = PRSurfaceCollocation(model, phase_fallback=False)
    state, _ = advance(solver, state)
    previous = solver._last_advance_commands, solver._last_advance_time
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise PropertyModelError("Invalid trial")

        patch.setattr(solver, "_step", fail)
        with pytest.raises(HydraulicConvergenceError):
            advance(solver, state, target=0.7)
    assert (solver._last_advance_commands, solver._last_advance_time) == previous
    _, result = advance(solver, state, target=0.7)
    assert result["command_step_restart"]


def test_restart_switch_keeps_existing_error_and_balance_contract():
    model, old = controlled(feedback=True)
    results = []
    for enabled in (False, True):
        solver = PRSurfaceCollocation(model)
        solver._restart_on_command_change = enabled
        state = old
        for opening in (True, False, True):
            row = solver.advance(state, 1e-6, openings={"port": opening})
            assert all(0 <= s["error_ratio"] <= 1 for s in row["steps"])
            assert max(map(abs, row["conservation_residuals"])) < 1e-6
            state = type(state).from_dict(row["state"])
        results.append(state)
    assert model._error(*results) < 1
