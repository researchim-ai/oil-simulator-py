"""Compact MGS/Givens retains linear accuracy while removing scalar dispatch."""

import pytest
import torch
from test_surface_equilibrium_thermodynamics import fluid

from surface.torch_krylov import gmres_device as compact
from surface.torch_linear import gmres_device as reference

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["diagonal", "nonsymmetric", "scaled", "zero"])
def test_compact_matches_reference_and_direct_linear_solution(device, case):
    gen = torch.Generator(device=device).manual_seed(739)
    n = 64
    a = torch.randn((n, n), generator=gen, device=device, dtype=torch.float64) * 0.01
    a = a + torch.eye(n, device=device, dtype=torch.float64) * 2
    if case == "diagonal":
        a = torch.diag(torch.linspace(0.5, 2.0, n, device=device, dtype=torch.float64))
    if case == "scaled":
        a = a * torch.logspace(-6, 6, n, device=device, dtype=torch.float64)[:, None]
    truth = torch.randn(n, generator=gen, device=device, dtype=torch.float64)
    b = a @ truth if case != "zero" else torch.zeros_like(truth)
    x = compact(lambda v: a @ v, b, a.diagonal())
    old = reference(lambda v: a @ v, b, a.diagonal())
    direct = torch.linalg.solve(a, b)
    torch.testing.assert_close(x, old, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(x, direct, rtol=2e-9, atol=2e-10)
    residual = (a @ x - b) / a.abs().sum(-1)
    assert float(residual.abs().max()) < 1e-10


@pytest.mark.parametrize("device", DEVICES)
def test_compact_accepts_physical_block_preconditioning(device):
    a = torch.diag(
        torch.tensor([1e-5, 2.0, 3e4, 4.0, 5.0, 6.0], device=device, dtype=torch.float64)
    )
    a[1, 2] = 0.01
    a[4, 5] = 0.2
    b = torch.ones(6, device=device, dtype=torch.float64)
    precondition = lambda v: v / a.diagonal()
    actual = compact(lambda v: a @ v, b, a.diagonal(), preconditioner=precondition)
    expected = reference(lambda v: a @ v, b, a.diagonal(), preconditioner=precondition)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(a @ actual, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_eos_hot_path_never_constructs_host_tensor_constants(device, monkeypatch):
    f = fluid(device)
    p, t, z = map(f.tensor, (1e6, 320.0, [0.2, 0.8]))
    expected = f.tensor_homogeneous(p, t, z)

    def forbidden(*args, **kwargs):
        raise AssertionError("Host tensor allocation in EOS hot path")

    monkeypatch.setattr(f, "tensor", forbidden)
    actual = f.tensor_homogeneous(p, t, z)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    jac = torch.func.jacfwd(
        lambda pt: f.tensor_homogeneous(pt[0], pt[1], z)["molar_enthalpy_j_mol"]
    )(torch.stack((p, t)))
    assert bool(torch.isfinite(jac).all())
