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


def jacobi(A, x, b, w=0.67, iters=5):
    """Простой сглаживатель Якоби для плотных матриц.
    Важно: ожидает dense A; на sparse тензорах использовать нельзя.
    """
    D = torch.diag(A)
    R = A - torch.diag(D)
    for _ in range(iters):
        x = (1-w)*x + w*(b - torch.mv(R, x)) / (D + 1e-30)
    return x


def build_RP(shape_f: Tuple[int, int, int], device, dtype, A: torch.Tensor=None, smooth_P: bool=True, omega: float=0.67):
    """Строит согласованные R и P для 2×2(×2) коарсинга.
    R — усреднение блока; P = s · R^T, где s=4 (2D) или s=8 (3D).
    Возвращает (R, P, shape_c).
    """
    nx, ny, nz = shape_f
    assert nx % 2 == 0 and ny % 2 == 0 and nz % 2 == 0, "Размеры для коарсинга должны быть чётными"

    if nz == 1:
        nx_c, ny_c, nz_c = nx // 2, ny // 2, 1
        s = 4.0
    else:
        nx_c, ny_c, nz_c = nx // 2, ny // 2, nz // 2
        s = 8.0

    Nf = nx * ny * nz
    Nc = nx_c * ny_c * nz_c
    R = torch.zeros((Nc, Nf), device=device, dtype=dtype)

    def fidx(i, j, k):
        # Линейный индекс на fine (k — самая быстрая координата)
        return (i * ny + j) * nz + k

    def cidx(I, J, K):
        # Линейный индекс на coarse
        return (I * ny_c + J) * nz_c + K

    if nz == 1:
        w = 1.0 / 4.0
        for I in range(nx_c):
            for J in range(ny_c):
                row = cidx(I, J, 0)
                for di in (0, 1):
                    for dj in (0, 1):
                        col = fidx(2 * I + di, 2 * J + dj, 0)
                        R[row, col] = w
    else:
        w = 1.0 / 8.0
        for I in range(nx_c):
            for J in range(ny_c):
                for K in range(nz_c):
                    row = cidx(I, J, K)
                    for di in (0, 1):
                        for dj in (0, 1):
                            for dk in (0, 1):
                                col = fidx(2 * I + di, 2 * J + dj, 2 * K + dk)
                                R[row, col] = w

    P = (R.transpose(0, 1) * s)
    # Smoothed Aggregation: P ← (I − ω D^{-1}A) P
    if smooth_P and A is not None:
        D = torch.diag(A)
        Dinv = 1.0 / (D + 1e-30)
        AP = A @ P
        P = P - omega * (Dinv.unsqueeze(1) * AP)
    return R, P, (nx_c, ny_c, nz_c)


def restrict_vec(vec_f: torch.Tensor, R: torch.Tensor):
    return R @ vec_f.view(-1)


def prolong_vec(vec_c: torch.Tensor, P: torch.Tensor):
    return P @ vec_c.view(-1)


def v_cycle(A, b, x, shape, levels=2, cache=None):
    """V-цикл MG для учебных целей (dense). Использует галёркинский coarse A_c = R A P.
    """
    if cache is None:
        cache = {}

    if levels == 1 or min(shape) <= 4:
        return torch.linalg.solve(A, b)

    # Пред-сглаживание
    x = jacobi(A, x, b)
    r_f = b - A @ x

    # Получаем/строим R, P и A_c для данного shape/device/dtype
    key = (shape, str(A.device), str(A.dtype), id(A))
    if key not in cache:
        R, P, shape_c = build_RP(shape, A.device, A.dtype, A=A, smooth_P=True, omega=0.67)
        A_c = R @ A @ P
        cache[key] = (R, P, A_c, shape_c)
    else:
        R, P, A_c, shape_c = cache[key]

    # Рестрикция невязки и рекурсивный solve на coarse
    r_c = restrict_vec(r_f, R)
    e_c0 = torch.zeros_like(r_c)
    e_c = v_cycle(A_c, r_c, e_c0, shape_c, levels - 1, cache)

    # Пролонгация и коррекция
    e_f = prolong_vec(e_c, P)
    x = x + e_f

    # Пост-сглаживание
    x = jacobi(A, x, b)
    return x


def mg_precond(A: torch.Tensor, b: torch.Tensor, shape: Tuple[int,int,int]):
    x0 = torch.zeros_like(b)
    return v_cycle(A, b, x0, shape, levels=2) 