"""Two selected ragged PVTO tables, scalar arithmetic and higher derivatives."""

from types import SimpleNamespace

import pytest
import torch

from surface.blackoil_properties import BlackOilMixture
from surface.blackoil_pvto_batch import PackedPVTO


def snapshot(count=5):
    tables = {}
    for i in range(count):
        axis = torch.linspace(1e5, 5e6, 3 + i % 5, dtype=torch.float64)
        tables[float(i)] = (axis, 1.0 + i * 0.003 + axis * 1e-9, 4.0 - axis * 1e-8 + i * 0.01)
    return SimpleNamespace(oil_tables=tables, rho=torch.ones(3, dtype=torch.float64))


def test_ragged_axes_and_exact_knots_match_scalar_and_second_derivatives():
    model = snapshot()
    packed = PackedPVTO(model)
    index = torch.tensor([0, 3, 1, 4])
    p = torch.tensor(
        [[1e5, 2.55e6, 5e6], [1.2e6, 2e6, 4e6], [1.1e6, 3e6, 4e6], [2e6, 3e6, 5e6]],
        dtype=torch.float64,
        requires_grad=True,
    )
    for column in [1, 2]:
        expected = torch.stack(
            [
                BlackOilMixture.interpolate(
                    p[i],
                    list(model.oil_tables.values())[int(j)][0],
                    list(model.oil_tables.values())[int(j)][column],
                )
                for i, j in enumerate(index)
            ]
        )
        actual = packed.interpolate(p, index, column)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        ga = torch.autograd.grad(actual.square().sum(), p, create_graph=True)[0]
        ge = torch.autograd.grad(expected.square().sum(), p, create_graph=True)[0]
        torch.testing.assert_close(ga, ge, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(ga.sum(), p)[0], torch.autograd.grad(ge.sum(), p)[0], rtol=0, atol=0
        )


def test_quadrature_lookup_storage_does_not_acquire_a_table_dimension(monkeypatch):
    model = snapshot(64)
    packed = PackedPVTO(model)
    p = torch.full((128, 40, 32), 2.4e6, dtype=torch.float64)
    index = torch.arange(128) % 64
    original = torch.stack

    def forbidden_stack(*args, **kwargs):
        raise AssertionError("No stack over all PVTO tables per quadrature point")

    monkeypatch.setattr(torch, "stack", forbidden_stack)
    out = packed.interpolate(p, index, 1)
    assert out.shape == p.shape and torch.isfinite(out).all()
    monkeypatch.setattr(torch, "stack", original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_selected_tables_stay_on_cuda():
    model = snapshot()
    model.rho = model.rho.cuda()
    model.oil_tables = {k: tuple(x.cuda() for x in v) for k, v in model.oil_tables.items()}
    packed = PackedPVTO(model)
    p = torch.full((16, 32), 2.4e6, device="cuda", dtype=torch.float64)
    result = packed.interpolate(p, torch.arange(16, device="cuda") % 5, 1)
    assert result.is_cuda and torch.isfinite(result).all()
