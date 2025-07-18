import torch
from typing import Tuple


def chebyshev_smooth(csr_A: torch.Tensor, rhs: torch.Tensor, x0: torch.Tensor,
                     iters: int = 2, omega: float = 0.7) -> Tuple[torch.Tensor, float]:
    """Упрощённый Chebyshev smoother на GPU.

    csr_A – torch.sparse_csr_tensor (float32)
    rhs, x0 – dense torch.Tensor
    Возвращает (x, res_norm).
    """
    x = x0.clone()

    # Приближаем λ_max через одну итерацию мощности
    v = torch.randn_like(rhs)
    v = v / v.norm()
    Av = torch.sparse.mm(csr_A, v.unsqueeze(1)).squeeze(1)
    lam_max = torch.dot(v, Av).abs() + 1e-6
    lam_min = 0.1 * lam_max  # грубая оценка

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