"""MGS fusion preserves IEEE arithmetic and the full Krylov trajectory."""
import importlib
import pytest
import torch
from solver.krylov_cuda import orthogonalize_subtract


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_update_exact(dtype, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    generator = torch.Generator(device=device).manual_seed(47)
    w = torch.randn(4099, generator=generator, device=device, dtype=dtype)
    v = torch.randn(4099, generator=generator, device=device, dtype=dtype)
    c = torch.tensor(.718281828, device=device, dtype=dtype)
    expected = w - c*v
    actual = orthogonalize_subtract(w, v, c)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.data_ptr() not in (w.data_ptr(), v.data_ptr())
    # Cancellation, large dynamic range, infinities, and subnormal operands.
    values = torch.tensor([0., -0., 1e-30, 1e30, float('inf'), float('nan')], device=device, dtype=dtype)
    torch.testing.assert_close(orthogonalize_subtract(values, values, c), values-c*values,
                               rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_full_krylov_matches_eager(monkeypatch):
    module = importlib.import_module('solver.fgmres')
    torch.manual_seed(11)
    a = torch.randn(48, 48, dtype=torch.float64, device='cuda')
    a = a @ a.T + torch.eye(48, dtype=a.dtype, device=a.device)*20
    b = torch.randn(48, dtype=a.dtype, device=a.device)
    actual_stats = {}; expected_stats = {}
    actual, info = module.fgmres(lambda x: a@x, b, restart=8, max_iter=15, tol=1e-10, stats=actual_stats)
    monkeypatch.setattr(module, 'orthogonalize_subtract', lambda w,v,c: w-c*v)
    expected, expected_info = module.fgmres(lambda x: a@x, b, restart=8, max_iter=15, tol=1e-10, stats=expected_stats)
    assert info == expected_info == 0
    assert actual_stats == expected_stats
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert float(torch.linalg.vector_norm(a@actual-b)/torch.linalg.vector_norm(b)) < 1e-10
