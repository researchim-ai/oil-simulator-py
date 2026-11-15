"""Stop roundoff plateaus only inside a caller's strict residual budget."""

import pytest
import torch

from surface.torch_nonlinear import TorchNewton

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("floor", [1e-11, 2e-8])
def test_cancellation_plateau_never_accepts_an_out_of_budget_residual(device, floor):
    y = torch.tensor([0.0, 0.9 * floor], dtype=torch.float64, device=device)

    def residual(z):
        # The first value is below one ULP of the large enthalpy-like offset;
        # AD still correctly sees derivative one. The second coordinate makes
        # small L2 improvements without reducing the dominating infinity norm.
        return torch.stack(((z[0] + 1e10) - 1e10 + floor, z[1]))

    outputs = []
    for acceptance in (None, 1.1e-11):
        solver = TorchNewton(device, maximum_iterations=40)
        solver.host_convergence = True
        result = solver.solve(
            residual,
            y,
            y.new_full((2,), -1e6),
            y.new_full((2,), 1e6),
            tolerance=3e-12,
            maximum_update=y.new_tensor([float("inf"), 0.001 * floor]),
            stagnation_tolerance=acceptance,
        )
        outputs.append((solver, result))
    if floor < 1.1e-11:
        assert bool(outputs[1][1][2])
        assert outputs[1][0].stop_reason == "stagnated_within_tolerance"
        assert outputs[1][1][3] < outputs[0][1][3] / 4
        assert float(outputs[1][1][1].abs().max()) == pytest.approx(floor)
    else:
        assert not bool(outputs[1][1][2])
        assert outputs[1][0].stop_reason != "stagnated_within_tolerance"
        assert outputs[1][1][3] == outputs[0][1][3]
        torch.testing.assert_close(outputs[1][1][0], outputs[0][1][0], rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_attainable_target_still_converges_to_requested_tolerance(device):
    y = torch.tensor([2.0, 3.0], device=device, dtype=torch.float64)
    solver = TorchNewton(device, maximum_iterations=20)
    solver.host_convergence = True
    x, r, ok, _ = solver.solve(
        lambda z: torch.stack((z[0].square() - 2.0, z[1].square() - 3.0)),
        y,
        y.new_zeros(2),
        y.new_full((2,), 10.0),
        tolerance=3e-12,
        stagnation_tolerance=1e-11,
    )
    assert bool(ok) and float(r.abs().max()) <= 3e-12
    assert solver.stop_reason == "target_tolerance"
    torch.testing.assert_close(x.square(), y.new_tensor([2.0, 3.0]), atol=3e-12, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("certified_row", [True, False])
def test_roundoff_scale_only_applies_to_its_certified_row(device, certified_row):
    floor = 1.2e-11
    y = torch.tensor([0.0, 0.5 * floor], device=device, dtype=torch.float64)

    def residual(z):
        return torch.stack(((z[0] + 1e10) - 1e10 + floor, z[1]))

    outputs = []
    for supplied in (False, True):
        solver = TorchNewton(device, maximum_iterations=40)
        solver.host_convergence = True
        bounds = y.new_tensor([2e-11, 0.0] if certified_row else [0.0, 2e-11])
        result = solver.solve(
            residual,
            y,
            y.new_full((2,), -1e6),
            y.new_full((2,), 1e6),
            tolerance=3e-12,
            maximum_update=y.new_tensor([float("inf"), 0.001 * floor]),
            stagnation_tolerance=1e-11,
            roundoff_tolerance=(lambda bounds=bounds: bounds) if supplied else None,
        )
        outputs.append((solver, result))
    if certified_row:
        assert outputs[1][0].stop_reason == "stagnated_within_roundoff"
        assert outputs[1][1][3] < outputs[0][1][3] / 4
    else:
        assert outputs[1][0].stop_reason != "stagnated_within_roundoff"
        assert outputs[1][1][3] == outputs[0][1][3]
    assert bool(outputs[1][1][2]) is certified_row
    assert float(outputs[1][1][1].abs().max()) == pytest.approx(floor)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("acceptance", [None, 1e-11])
def test_small_but_out_of_budget_stalled_residual_is_not_success(device, acceptance):
    start = torch.tensor([0.0], dtype=torch.float64, device=device)
    solver = TorchNewton(device, maximum_iterations=4)
    solver.host_convergence = True
    # At the bound there is no feasible root. The defect is smaller than the
    # old generic 1e-8 cutoff, but larger than either explicit caller budget.
    _, residual, ok, _ = solver.solve(
        lambda x: x + 2e-10,
        start,
        start.clone(),
        start.new_ones(1),
        tolerance=3e-12,
        stagnation_tolerance=acceptance,
    )
    assert float(residual.abs().max()) == pytest.approx(2e-10)
    assert not bool(ok)


@pytest.mark.parametrize("device", DEVICES)
def test_iteration_budget_does_not_override_requested_convergence(device):
    start = torch.tensor([0.0], dtype=torch.float64, device=device)
    solver = TorchNewton(device, maximum_iterations=1)
    solver.host_convergence = True
    _, residual, ok, _ = solver.solve(
        lambda x: x - 1e-9,
        start,
        start.new_full((1,), -1.0),
        start.new_ones(1),
        maximum_update=start.new_full((1,), 1e-12),
        tolerance=3e-12,
    )
    assert solver.stop_reason == "iteration_budget"
    assert float(residual.abs().max()) > 3e-12
    assert not bool(ok)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_target_tolerance_is_rejected_before_residual_evaluation(tolerance):
    start = torch.zeros(1, dtype=torch.float64)

    def forbidden(_):
        raise AssertionError("Invalid numerical policy must fail before solving")

    with pytest.raises(ValueError, match="finite and positive"):
        TorchNewton("cpu").solve(forbidden, start, start - 1, start + 1, tolerance=tolerance)


def test_shorter_plateau_confirmation_keeps_the_same_best_qualified_iterate():
    start = torch.tensor([0.0, 0.9e-11], dtype=torch.float64)
    results = []
    for patience in (4, 2):
        solver = TorchNewton("cpu", maximum_iterations=40)
        solver._stagnation_patience = patience
        result = solver.solve(
            lambda x: torch.stack(((x[0] + 1e10) - 1e10 + 1e-11, x[1])),
            start,
            start - 1e6,
            start + 1e6,
            maximum_update=start.new_tensor([float("inf"), 1e-14]),
            tolerance=3e-12,
            stagnation_tolerance=1.1e-11,
        )
        assert bool(result[2])
        assert solver.stop_reason == "stagnated_within_tolerance"
        results.append(result)
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    assert results[1][3] < results[0][3]


def test_productive_progress_inside_stagnation_budget_reaches_target():
    solver = TorchNewton("cpu", maximum_iterations=20)
    start = torch.tensor([9e-12], dtype=torch.float64)
    _, residual, ok, _ = solver.solve(
        lambda x: x,
        start,
        start.new_full((1,), -1.0),
        start.new_ones(1),
        maximum_update=start.new_full((1,), 1e-12),
        tolerance=3e-12,
        stagnation_tolerance=1e-11,
    )
    assert bool(ok) and float(residual.abs().max()) <= 3e-12
    assert solver.stop_reason == "target_tolerance"
