import torch
from typing import Tuple


def chebyshev_smooth(csr_A: torch.Tensor, rhs: torch.Tensor, x0: torch.Tensor,
                     iters: int = 2, omega: float = 0.7) -> Tuple[torch.Tensor, float]:
    """Упрощённый Chebyshev smoother на GPU.

    csr_A – torch.sparse_csr_tensor (float64 рекомендуется)
    rhs, x0 – dense torch.Tensor
    Возвращает (x, res_norm).
    """
    assert rhs.device == csr_A.device == x0.device, "csr_A, rhs, x0 must be on the same device"
    x = x0.clone()

    # Приближаем λ_max через несколько итераций мощности для устойчивости
    v = torch.randn_like(rhs)
    v = v / (v.norm() + 1e-30)
    power_iters = 5
    lam_max = None
    for _ in range(power_iters):
        Av = torch.sparse.mm(csr_A, v.unsqueeze(1)).squeeze(1)
        # Rayleigh quotient (SPD): (v^T A v) / (v^T v)
        num = torch.dot(v, Av)
        den = torch.dot(v, v).clamp_min(1e-30)
        lam_est = (num / den).abs()
        lam_max = lam_est if lam_max is None else torch.maximum(lam_max, lam_est)
        v = Av / (Av.norm() + 1e-30)
    lam_max = lam_max + 1e-12
    # Консервативная нижняя граница для устойчивости
    lam_min = lam_max / 80.0

    d = (lam_max + lam_min) / 2.0
    c = (lam_max - lam_min) / 2.0
    alpha = d / c
    beta = omega / c

    p_prev = torch.zeros_like(rhs)

    for _ in range(iters):
        r = rhs - torch.sparse.mm(csr_A, x.unsqueeze(1)).squeeze(1)
        p = beta * r + alpha * p_prev
        x = x + p
        p_prev = p

    res = torch.norm(rhs - torch.sparse.mm(csr_A, x.unsqueeze(1)).squeeze(1)).item()
    return x, res 