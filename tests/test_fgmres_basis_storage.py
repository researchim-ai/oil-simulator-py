"""Lazy storage must preserve flexible Arnoldi, restart and failure behavior."""
import pytest
import torch

from solver.fgmres import fgmres


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("restart", [4, 19, 50])
def test_growth_and_restart_with_flexible_preconditioner(device, restart, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    gen = torch.Generator().manual_seed(731)
    a = torch.randn(61, 61, generator=gen, dtype=torch.float64) * .03
    a += torch.diag(torch.linspace(1., 25., 61, dtype=torch.float64))
    a = a.to(device)
    b = torch.linspace(-1., 2., 61, device=device, dtype=torch.float64)
    calls = 0

    def precondition(v):
        nonlocal calls
        calls += 1
        return v * (1. + .02 * (calls % 3))

    stats = {}
    original = torch.empty

    def poisoned(*args, **kwargs):
        return original(*args, **kwargs).fill_(float("nan"))

    monkeypatch.setattr(torch, "empty", poisoned)
    x, info = fgmres(lambda v: a @ v, b, precondition, restart=restart,
                     max_iter=100, tol=1e-11, stats=stats)
    assert info == 0
    assert torch.linalg.vector_norm(a @ x - b) / torch.linalg.vector_norm(b) < 1.1e-11
    torch.testing.assert_close(x, torch.linalg.solve(a, b), atol=1e-10, rtol=1e-10)
    assert calls == stats["iters"]
    if restart > 8:
        assert stats["basis_growths"] > 0
    else:
        assert stats["restarts"] > 0
    assert stats["basis_capacity"] <= restart


def test_early_exit_allocates_only_small_basis(monkeypatch):
    # Poison new buffers: neither unused rows nor old restart contents may leak.
    original = torch.empty

    def poisoned(*args, **kwargs):
        return original(*args, **kwargs).fill_(float("nan"))

    monkeypatch.setattr(torch, "empty", poisoned)
    b = torch.arange(1., 101., dtype=torch.float64)
    stats = {}
    x, info = fgmres(lambda v: 2. * v, b, restart=50, tol=1e-12, stats=stats)
    assert info == 0 and stats["iters"] == 1
    torch.testing.assert_close(x, b / 2.)
    assert stats["basis_capacity"] == 8 and stats["basis_growths"] == 0
    assert stats["basis_storage_bytes"] == 17 * b.numel() * b.element_size()


def test_zero_rhs_and_exhausted_restart():
    stats = {}
    x, info = fgmres(lambda v: v, torch.zeros(10), stats=stats)
    assert info == 0 and stats["basis_storage_bytes"] == 0
    assert not x.any()
    diagonal = torch.linspace(1., 100., 80, dtype=torch.float64)
    b = torch.ones_like(diagonal)
    x, info = fgmres(lambda v: diagonal * v, b, restart=19,
                     max_iter=1, tol=1e-15, stats=stats)
    assert info == 1 and stats["iters"] == 19
    assert stats["basis_growths"] == 2 and stats["basis_capacity"] == 19
    assert torch.isfinite(x).all()
    assert torch.linalg.vector_norm(diagonal * x - b) > 1e-15
