"""Cross-solve chord predictions must satisfy the original nonlinear contract."""

import pytest
import torch

from surface.torch_nonlinear import DenseNewtonHistory, TorchNewton


def solve(history, function, start=1.0, key="liquid", coefficient=0.1, bound=10.0):
    solver = TorchNewton("cpu", maximum_iterations=30)
    x = torch.tensor([start], dtype=torch.float64)
    with torch.no_grad():
        result = solver.solve(
            function,
            x,
            x.new_full((1,), -bound),
            x.new_full((1,), bound),
            tolerance=3e-12,
            linearization_history=history,
            linearization_key=key,
            time_coefficient=coefficient,
        )
    assert bool(result[2])
    assert float(result[1].abs().max()) <= 3e-12
    return solver, result[0]


def test_nearby_time_equations_share_factors_and_retain_strict_residual():
    history = DenseNewtonHistory()
    solve(history, lambda x: x - 1.01)
    solver, root = solve(history, lambda x: x - 1.02, start=1.01)
    assert root.item() == pytest.approx(1.02, abs=3e-12)
    assert solver.cross_solve_reuses == 1 and solver.jacobian_builds == 0


@pytest.mark.parametrize("change", ["phase", "step", "bounds", "state"])
def test_changed_contract_forces_fresh_tangent(change):
    history = DenseNewtonHistory()
    solve(history, lambda x: x - 1.01)
    kwargs = {
        "phase": {"key": "gas-liquid"},
        "step": {"coefficient": 0.2},
        "bounds": {"bound": 9.0},
        "state": {"start": 2.0},
    }[change]
    solver, _ = solve(history, lambda x: 2 * (x - 1.02), **kwargs)
    assert solver.cross_solve_reuses == 0 and solver.jacobian_builds == 1


def test_wrong_direction_refreshes_before_backtracking_or_acceptance():
    history = DenseNewtonHistory()
    solve(history, lambda x: x - 1.01)
    solver, root = solve(history, lambda x: -2 * (x - 1.02), start=1.01)
    assert root.item() == pytest.approx(1.02, abs=3e-12)
    assert solver.cross_solve_refreshes == 1 and solver.jacobian_builds == 1


def test_small_inactive_stock_ceiling_drift_does_not_discard_a_valid_tangent():
    history = DenseNewtonHistory()
    solver, root = solve(history, lambda x: x - 1.01)
    x = root.clone()
    with torch.no_grad():
        _, row, ok, _ = solver.solve(
            lambda z: z - 1.02,
            x,
            x.new_full((1,), -10.0),
            x.new_full((1,), 10.000001),
            linearization_history=history,
            linearization_key="liquid",
            time_coefficient=0.1,
        )
    assert bool(ok) and float(row.abs().max()) <= 2e-11
    assert solver.cross_solve_reuses == 1 and solver.jacobian_builds == 0


def test_periodic_refresh_bounds_the_lifetime_of_a_successful_tangent():
    history = DenseNewtonHistory()
    builds = []
    for i in range(6):
        target = 1.0 + (i + 1) * 0.001
        solver, _ = solve(history, lambda x, target=target: x - target, start=target - 0.001)
        builds.append(solver.jacobian_builds)
    assert builds == [1, 0, 0, 0, 1, 0]


def test_failed_solve_does_not_publish_a_history_entry():
    history = DenseNewtonHistory()
    solve(history, lambda x: x - 1.01)
    solver = TorchNewton("cpu", maximum_iterations=2)
    x = torch.ones(1, dtype=torch.float64)
    with torch.no_grad():
        _, _, ok, _ = solver.solve(
            lambda z: z * 0 + 1,
            x,
            -10 * x,
            10 * x,
            linearization_history=history,
            linearization_key="liquid",
            time_coefficient=0.1,
        )
    assert not bool(ok) and history.entry is None


def test_pr_thermal_trajectory_reuses_tangents_without_changing_energy_or_temperature():
    from check_pr_collocation import thermal_fixture

    from surface.pr_collocation import PRCollocation

    finals = []
    for enabled in (False, True):
        model, state = thermal_fixture()
        integrator = PRCollocation(model, phase_fallback=False, scheme="radau3")
        integrator._reuse_step_jacobian = enabled
        for _ in range(10):
            state, record = integrator._step(
                state, 0.01, model.default_openings, [0.0, 0.0], [320.0, 360.0], {}
            )
            assert max(map(abs, record["conservation_residuals"])) < 1e-8
        finals.append([s.temperature_k for _, s in state.nodes])
        if enabled:
            assert integrator._stage_linearization.hits > 0
            assert integrator._stage_linearization.builds < 10
    assert finals[1] == pytest.approx(finals[0], abs=1e-8)
