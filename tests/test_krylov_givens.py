"""Krylov scalar fusion must preserve rotations and the true solved system."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from solver import krylov_cuda
from solver.fgmres import fgmres


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('j', [0, 3, 22])
@pytest.mark.parametrize('strided', [False, True])
def test_rotations_match_eager_bitwise(device, dtype, j, strided, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    gen = torch.Generator(device=device).manual_seed(814)
    H = torch.randn(24, 23, dtype=dtype, device=device, generator=gen)
    if strided:
        H = H.T.contiguous().T
    angles = torch.arange(23, device=device, dtype=dtype)*.17
    cs, sn = angles.cos(), angles.sin()
    g = torch.randn(24, dtype=dtype, device=device, generator=gen)
    reference = [x.clone() for x in (H, cs, sn, g)]
    before = [x.clone() for x in (H, cs, sn, g)]
    with monkeypatch.context() as patch:
        patch.setattr(krylov_cuda, 'triton', None)
        krylov_cuda.rotate_givens(*reference, j)
    krylov_cuda.rotate_givens(H, cs, sn, g, j)
    for actual, expected in zip((H, cs, sn, g), reference):
        assert torch.equal(actual, expected)
    # Repeated calls on the same inputs must also be deterministic.
    krylov_cuda.rotate_givens(*before, j)
    assert all(torch.equal(a, b) for a, b in zip(before, (H, cs, sn, g)))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_restarted_flexible_solver_has_small_true_residual(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    gen = torch.Generator().manual_seed(12)
    A = torch.randn(31, 31, dtype=torch.float64, generator=gen)*.15
    A += torch.diag(torch.linspace(3., 8., 31, dtype=torch.float64))
    b = torch.randn(31, dtype=torch.float64, generator=gen)
    expected = torch.linalg.solve(A, b)
    A, b = A.to(device), b.to(device)
    counter = 0

    def precondition(r):
        nonlocal counter
        counter += 1
        return r/A.diag() * (1.+.01*(counter % 3))

    stats = {}

    x, info = fgmres(lambda v: A@v, b, precondition, restart=4,
                     max_iter=20, tol=1e-11, stats=stats)
    assert info == 0 and stats['iters'] > 4
    assert float(torch.linalg.vector_norm(A@x-b)/torch.linalg.vector_norm(b)) < 1e-11
    torch.testing.assert_close(x.cpu(), expected, rtol=1e-10, atol=1e-11)
