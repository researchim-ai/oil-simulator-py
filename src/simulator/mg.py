import torch
from typing import Tuple


def restrict(fine: torch.Tensor, shape_f: Tuple[int, int, int]):
    """Усреднение 2×2×2 блоков. Работает и для 2-D, когда nz=1."""
    nx, ny, nz = shape_f
    cf = fine.view(nx, ny, nz)
    # если nz==1, упрощаем
    if nz == 1:
        cf = cf.view(nx, ny)
        coarse = 0.25 * (
            cf[0::2, 0::2] + cf[1::2, 0::2] + cf[0::2, 1::2] + cf[1::2, 1::2]
        )
        return coarse.flatten(), coarse.shape
    # 3-D (редко в тестах)
    coarse = 0.125 * (
        cf[0::2, 0::2, 0::2] + cf[1::2, 0::2, 0::2] + cf[0::2, 1::2, 0::2] + cf[1::2, 1::2, 0::2] +
        cf[0::2, 0::2, 1::2] + cf[1::2, 0::2, 1::2] + cf[0::2, 1::2, 1::2] + cf[1::2, 1::2, 1::2]
    )
    return coarse.flatten(), coarse.shape


def prolong(coarse_vec: torch.Tensor, shape_c: Tuple[int, int, int]):
    nx_c, ny_c, nz_c = shape_c
    if nz_c == 1:
        c = coarse_vec.view(nx_c, ny_c)
        fine = torch.zeros((nx_c*2, ny_c*2), device=c.device, dtype=c.dtype)
        fine[0::2, 0::2] = c
        fine[1::2, 0::2] = c
        fine[0::2, 1::2] = c
        fine[1::2, 1::2] = c
        return fine.flatten()
    # 3-D: простой injection
    c = coarse_vec.view(nx_c, ny_c, nz_c)
    fine = torch.zeros((nx_c*2, ny_c*2, nz_c*2), device=c.device, dtype=c.dtype)
    fine[0::2, 0::2, 0::2] = c
    fine[1::2, 0::2, 0::2] = c
    fine[0::2, 1::2, 0::2] = c
    fine[1::2, 1::2, 0::2] = c
    fine[0::2, 0::2, 1::2] = c
    fine[1::2, 0::2, 1::2] = c
    fine[0::2, 1::2, 1::2] = c
    fine[1::2, 1::2, 1::2] = c
    return fine.flatten()


def jacobi(A, x, b, w=0.8, iters=3):
    """Простой сглаживатель Якоби для плотных матриц.
    Важно: ожидает dense A; на sparse тензорах использовать нельзя.
    """
    D = torch.diag(A)
    R = A - torch.diag(D)
    for _ in range(iters):
        x = (1-w)*x + w*(b - torch.mv(R, x)) / D
    return x


def v_cycle(A, b, x, shape, levels=2):
    """V-цикл MG для учебных целей. Работает с плотными A.
    На разрежённых матрицах не поддерживается.
    """
    if levels == 1 or min(shape) <= 4:
        # грубая сетка — решаем напрямую
        return torch.linalg.solve(A, b)
    # Пред- и пост-сглаживание
    x = jacobi(A, x, b)
    r = b - torch.mv(A, x)
    r_fine = r.clone()
    r_coarse, shape_c = restrict(r_fine, shape)
    # Грубая матрица (галеркин): Pᵀ A P ; здесь используем простое приближение — диагональ
    size_c = r_coarse.numel()
    A_c = torch.diag(torch.ones(size_c, device=A.device, dtype=A.dtype))
    e_c0 = torch.zeros_like(r_coarse)
    e_c = v_cycle(A_c, r_coarse, e_c0, (*shape_c, 1) if len(shape_c)==2 else shape_c, levels-1)
    e_f = prolong(e_c, shape_c)
    x = x + e_f
    x = jacobi(A, x, b)
    return x


def mg_precond(A: torch.Tensor, b: torch.Tensor, shape: Tuple[int,int,int]):
    x0 = torch.zeros_like(b)
    return v_cycle(A, b, x0, shape, levels=2) 