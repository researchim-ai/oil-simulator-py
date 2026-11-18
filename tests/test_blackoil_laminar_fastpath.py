"""Laminar exactness and unchanged mixed/transition derivatives."""

import pytest
import torch

from surface.blackoil_pipe import _darcy


def test_laminar_values_and_two_derivative_orders_are_analytic(monkeypatch):
    def unused(*args, **kwargs):
        raise AssertionError("Laminar packet must not evaluate turbulent logarithms")

    monkeypatch.setattr(torch, "log10", unused)
    re = torch.tensor([1.0, 20.0, 800.0, 1999.999], dtype=torch.float64, requires_grad=True)
    rough = torch.full_like(re, 1e-4, requires_grad=True)
    value = _darcy(re, rough)
    first, rough_first = torch.autograd.grad(value.sum(), (re, rough), create_graph=True)
    second = torch.autograd.grad(first.sum(), re)[0]
    torch.testing.assert_close(value, 64 / re, rtol=0, atol=0)
    torch.testing.assert_close(first, -64 / re.square(), rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(second, 128 / re.pow(3), rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(rough_first, torch.zeros_like(rough_first), rtol=0, atol=0)


@pytest.mark.parametrize("reynolds", [100.0, 1999.999, 2000.0, 2000.001, 3000.0, 4000.0, 1e5])
def test_scalar_and_mixed_packet_keep_same_value_gradient_and_hessian(reynolds):
    # A mixed packet forces the full transition/Colebrook path. The other
    # row is independent, so its value cannot change this row's derivatives.
    a = torch.tensor([reynolds, 1e-4], dtype=torch.float64, requires_grad=True)

    def scalar(x):
        return _darcy(x[:1], x[1:])[0]

    def mixed(x):
        re = torch.stack((x[0], x.new_tensor(1e5)))
        rough = torch.stack((x[1], x.new_tensor(1e-3)))
        return _darcy(re, rough)[0]

    torch.testing.assert_close(scalar(a), mixed(a), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(scalar, a),
        torch.autograd.functional.jacobian(mixed, a),
        rtol=1e-12,
        atol=1e-14,
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(scalar, a),
        torch.autograd.functional.hessian(mixed, a),
        rtol=1e-12,
        atol=1e-14,
    )


def test_saturated_batch_omits_unused_pvto_but_matches_mixed_packet_hessian(monkeypatch):
    from test_surface_blackoil_properties import model

    from surface.blackoil_pipe_batch import tensor_state
    from surface.blackoil_pvto_batch import PackedPVTO

    m = model()
    # The first row is deeply saturated; the second has low GOR and forces
    # the usual selected-PVTO path for the combined packet.
    x = torch.tensor([1.2e6, 0.85, 2.0, 0.144], dtype=torch.float64)

    def scalar(z):
        return tensor_state(m, z[:1, None], z[None, 1:])["specific_volume"].sum()

    def mixed(z):
        pressure = torch.stack((z[0], z.new_tensor(1.5e6)))[:, None]
        masses = torch.stack((z[1:], z.new_tensor([0.85, 0.2, 0.001])))
        return tensor_state(m, pressure, masses)["specific_volume"][0].sum()

    torch.testing.assert_close(scalar(x), mixed(x), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(scalar, x),
        torch.autograd.functional.jacobian(mixed, x),
        rtol=1e-13,
        atol=1e-13,
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(scalar, x),
        torch.autograd.functional.hessian(mixed, x),
        rtol=1e-12,
        atol=1e-12,
    )

    def unused(*args, **kwargs):
        raise AssertionError("All-saturated CPU packet must not select PVTO tables")

    monkeypatch.setattr(PackedPVTO, "prepare_lookup", unused)
    assert torch.isfinite(scalar(x))
