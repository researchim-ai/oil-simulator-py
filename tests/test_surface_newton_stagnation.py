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
    assert float(outputs[1][1][1].abs().max()) == pytest.approx(floor)
