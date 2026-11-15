"""Exact trial-vector reuse must preserve globalization and failure status."""

import pytest
import torch

from surface.torch_nonlinear import TorchNewton


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("case", ["roundoff", "bound", "nonlinear"])
def test_identical_trial_reuse_preserves_result_and_counts_saved_calls(sparse, case):
    start = torch.tensor([1e16 if case == "roundoff" else 0.1], dtype=torch.float64)
    if case == "roundoff":
        function = lambda x: x - 1e16 + 0.25
        lower, upper = start - 100, start + 100
    elif case == "bound":
        function = lambda x: x - 10
        lower, upper = start.new_zeros(1), start.new_ones(1)
    else:
        function = lambda x: x.square() - 2
        lower, upper = start.new_zeros(1), start.new_full((1,), 20)
    results = []
    for enabled in (False, True):
        calls = [0]

        def residual(x):
            calls[0] += 1
            return function(x)

        solver = TorchNewton("cpu", maximum_iterations=30, dense_limit=0 if sparse else 128)
        solver.host_convergence = True
        solver._reuse_identical_trials = enabled
        result = solver.solve(
            residual,
            start,
            lower,
            upper,
            ([0], [0]) if sparse else None,
            tolerance=3e-12,
        )
        results.append((solver, result, calls[0]))
    old, new = results
    for i in (0, 1, 2):
        assert torch.equal(old[1][i], new[1][i])
    assert old[0].stop_reason == new[0].stop_reason
    assert old[0].iterations == new[0].iterations
    assert old[0].jacobian_builds == new[0].jacobian_builds
    assert old[1][3] - new[1][3] == new[0].identical_trial_skips
    assert old[2] - new[2] == new[0].identical_trial_skips
    if case != "nonlinear":
        assert not bool(new[1][2])
        assert new[0].identical_trial_skips > 0
    else:
        assert bool(new[1][2])
        assert float(new[1][1].abs().max()) <= 3e-12


def test_new_solve_does_not_reuse_values_from_previous_residual():
    solver = TorchNewton("cpu")
    x = torch.zeros(1, dtype=torch.float64)
    for target in (1.0, 3.0):
        result, residual, ok, _ = solver.solve(
            lambda x: x - target, x, x - 10, x + 10, tolerance=3e-12
        )
        assert bool(ok)
        assert float(result[0]) == pytest.approx(target, abs=3e-12)
        assert float(residual.abs().max()) <= 3e-12
        assert solver.identical_trial_skips == 0
