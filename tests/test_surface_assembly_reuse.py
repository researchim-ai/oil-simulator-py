"""Reuse table interval searches without regrouping pressure derivatives."""
import pytest
import torch
from test_surface_blackoil_properties import model


@pytest.mark.parametrize('pressure,gas', [(0.6e6, .144), (3e6, .024), (3e6, .144), (1e6, .024)])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_scalar_flash_values_jacobian_hessian_are_exact(pressure, gas, device):
    m = model().to(device)
    x = torch.tensor([pressure, .85, 2., gas], dtype=torch.float64, device=device)
    def evaluate(z):
        state = m.tensor_state(z[0], z[1:])
        return torch.cat([v.reshape(-1) for v in state.values()])
    def evaluate_all():
        return (evaluate(x), torch.autograd.functional.jacobian(evaluate, x),
                torch.autograd.functional.hessian(lambda z: evaluate(z).square().sum(), x))
    current = evaluate_all()
    # Original per-column interpolation, same physical model and operations.
    m._column_interpolator = lambda p, axis: lambda values: m.interpolate(p, axis, values)
    original = evaluate_all()
    for a, b in zip(current, original):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_scalar_flash_uses_one_search_per_pressure_axis(monkeypatch):
    m = model()
    original = torch.searchsorted
    calls = []
    def counted(axis, pressure, *args, **kwargs):
        calls.append(axis)
        return original(axis, pressure, *args, **kwargs)
    monkeypatch.setattr(torch, 'searchsorted', counted)
    m.tensor_state(torch.tensor([2.4e6, 3.1e6], dtype=torch.float64),
                   torch.tensor([.85, 2., .024], dtype=torch.float64))
    # Main PVT pressure, Rs selection, and two undersaturated pressure tables.
    assert len(calls) == 4
    assert sum(a is m.pressures for a in calls) == 1
