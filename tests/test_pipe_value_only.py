"""Value-only quadrature retains momentum, implicit AD and physical failures."""

import pytest
import torch
from test_surface_blackoil_properties import model

from surface.blackoil_pipe import BlackOilPipe, ChokedFlowError
from surface.blackoil_pipe_batch import PipeBatch
from surface.liquid import HydraulicConvergenceError


class FullGraphPipe(BlackOilPipe):
    def _integral(self, *args, **kwargs):
        with torch.enable_grad():
            return super()._integral(*args, **kwargs)


class FullGraphBatch(PipeBatch):
    def integral(self, *args, **kwargs):
        with torch.enable_grad():
            return super().integral(*args, **kwargs)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_pressure_sensitivities_and_balances_match_full_graph_reference(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    m = model().to(device)
    specs = [('phase', 300, .08, 30, 1e-5), ('gas', 100, .15, 5, 2e-5),
             ('water', 80, .03, 0, 1e-5)]
    pipes = [BlackOilPipe(*s) for s in specs]
    boundaries = [2.72e6, 1.1e6, 2.4e6]
    rates = [[.85, 2., .06], [.85, 1., .144], [0., .001, 0.]]
    a = PipeBatch(pipes, m).solve(boundaries, rates)
    b = FullGraphBatch(pipes, m).solve(boundaries, rates)
    assert a == b
    for spec, boundary, rate in zip(specs, boundaries, rates, strict=True):
        a = BlackOilPipe(*spec).required_inlet(boundary, rate, m)
        b = FullGraphPipe(*spec).required_inlet(boundary, rate, m)
        assert a == b


def test_integral_values_and_hessians_match_at_moving_and_coincident_boundaries():
    m = model()
    pipe = BlackOilPipe('p', 300, .08, 30, 1e-5)
    for lower, upper in ((2.72e6, 3.2e6), (3e6, 3.2e6), (2.8e6, 3e6)):
        results = []
        for cls in (PipeBatch, FullGraphBatch):
            x = torch.tensor([lower, upper, .85, 2., .06, 0.], dtype=torch.float64, requires_grad=True)
            value = cls([pipe], m).checked(x[:1], x[1:2], x[2:5][None, :], x[5:6])[0]
            gradient = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
            hessian = torch.stack([torch.autograd.grad(g, x, retain_graph=True)[0] for g in gradient])
            results.append((value, gradient, hessian))
        for a, b in zip(*results, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        kernel = PipeBatch([pipe], m)
        with torch.no_grad():
            value_only = kernel.checked(x[:1].detach(), x[1:2].detach(),
                                        x[2:5][None, :].detach(), x[5:6].detach())[0]
        assert not value_only.requires_grad
        torch.testing.assert_close(value_only, results[0][0], rtol=0, atol=0)


def test_value_only_bracketing_does_not_hide_physical_choking():
    for cls in (PipeBatch, FullGraphBatch):
        with pytest.raises((ChokedFlowError, HydraulicConvergenceError)):
            cls([BlackOilPipe('choked', 100, .01)], model()).solve([2e6], [[0., 0., 1.]])
