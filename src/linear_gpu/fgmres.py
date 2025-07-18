import torch
from typing import Callable, Tuple


def _matvec(A, x: torch.Tensor) -> torch.Tensor:
    """Унифицированное умножение матрицы (dense/sparse) или callable на вектор x."""
    if callable(A):
        return A(x)
    # Приводим dtype при необходимости
    if x.dtype != A.dtype:
        x = x.to(A.dtype)
    # Поддерживаем CSR-матрицы PyTorch
    if hasattr(A, "is_sparse_csr") and A.is_sparse_csr:  # для совместимости с старыми версиями
        return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    return (A @ x).to(x.dtype)


def _deflate(vec: torch.Tensor, basis: torch.Tensor):
    """Удаляет проекцию vec на подпространство, заданное колонками basis (ортонорм)."""
    if basis is None or basis.numel() == 0:
        return vec
    coeffs = torch.matmul(basis.T, vec)
    return vec - basis @ coeffs


def fgmres(A,
           b: torch.Tensor,
           M: Callable[[torch.Tensor], torch.Tensor] = None,
           tol: float = 1e-8,
           restart: int = 50,
           max_iter: int = 400,
           deflation_basis: torch.Tensor = None,
           min_iters: int = 3) -> Tuple[torch.Tensor, int, int]:
    """Flexible GMRES (FGMRES).

    Практически повторяет нашу реализацию GMRES, но модулирует предобуславливатель *на каждой*
    итерации ‒ то есть M может изменять внутреннее состояние между вызовами.

    Возвращает (x, info, iters) где info=0 означает успешную сходимость.
    """
    device, dtype = b.device, b.dtype
    n = b.numel()
    x = torch.zeros_like(b)

    precond = (lambda r: r) if M is None else M

    b_norm = torch.norm(b)
    # Если правая часть практически нулевая, по умолчанию возвращаем ноль.
    # Однако, если требуется минимум итераций, позволяем алгоритму сделать их,
    # чтобы избежать вырожденного δ≈0, мешающего line-search.
    if b_norm < 1e-30 and min_iters == 0:
        return x, 0, 0

    r = b - _matvec(A, x)
    # дефляция начального резидуала
    r = _deflate(r, deflation_basis)
    r = precond(r)
    beta = torch.norm(r)
    # Ранний выход разрешаем только после min_iters итераций
    if beta / b_norm < tol and min_iters <= 0:
        return x, 0, 0

    V = [r / beta]
    H = torch.zeros(restart + 1, restart, device=device, dtype=dtype)
    cs = torch.zeros(restart, device=device, dtype=dtype)
    sn = torch.zeros(restart, device=device, dtype=dtype)
    g = torch.zeros(restart + 1, device=device, dtype=dtype)
    g[0] = beta

    # ------------------------------------------------------------------
    # Helper: robust solve for small (k×k) upper-Hessenberg H
    # ------------------------------------------------------------------
    def _robust_solve(H_sub: torch.Tensor, g_sub: torch.Tensor):
        """Пытается решить H_sub y = g_sub; при сингулярности использует lstsq."""
        try:
            # Проверяем обусловленность; cond() дорогая, но k<=restart<=50
            if torch.isnan(H_sub).any() or torch.isinf(H_sub).any():
                raise RuntimeError("H contains NaN/Inf")
            cond = torch.linalg.cond(H_sub)
            if torch.isfinite(cond) and cond < 1e12:
                return torch.linalg.solve(H_sub, g_sub)
        except Exception:
            pass
        # fallback: least-squares (псевдообратная) + крошечная Tikhonov
        try:
            eye = torch.eye(H_sub.shape[1], device=H_sub.device, dtype=H_sub.dtype)
            H_reg = H_sub + 1e-12 * eye  # стабилизация
            return torch.linalg.lstsq(H_reg, g_sub).solution
        except Exception:
            # В крайнем случае возвращаем ноль
            return torch.zeros_like(g_sub)

    total_iters = 0

    for outer in range(max_iter // restart + 1):
        for j in range(restart):
            w = _matvec(A, V[j])
            w = _deflate(w, deflation_basis)
            w = precond(w)  # flexible step: актуальный предобуславливатель

            # Gram-Schmidt
            for i in range(j + 1):
                H[i, j] = torch.dot(V[i], w)
                w = w - H[i, j] * V[i]
            H[j + 1, j] = torch.norm(w)
            if H[j + 1, j] < 1e-14:
                # счастливый случай – точный Крылов
                break
            V.append(w / H[j + 1, j])

            # Givens rotations
            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp
            denom = torch.sqrt(H[j, j] ** 2 + H[j + 1, j] ** 2)
            cs[j] = H[j, j] / denom
            sn[j] = H[j + 1, j] / denom
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]

            residual = torch.abs(g[j + 1])
            # Учёт минимального числа итераций
            if (residual / b_norm < tol) and ((total_iters + j + 1) >= min_iters):
                # сформировать y и x
                y = _robust_solve(H[:j + 1, :j + 1], g[:j + 1])
                update = sum(y[i] * V[i] for i in range(j + 1))
                x = x + update
                total_iters += j + 1 + outer * restart
                return x, 0, total_iters
        # restart
        y = _robust_solve(H[:j + 1, :j + 1], g[:j + 1])
        update = sum(y[i] * V[i] for i in range(j + 1))
        x = x + update
        r = b - _matvec(A, x)
        r = _deflate(r, deflation_basis)
        r = precond(r)
        beta = torch.norm(r)
        if (beta / b_norm < tol) and (total_iters >= min_iters):
            total_iters += (j + 1) + outer * restart
            return x, 0, total_iters
        # prepare for next cycle
        V = [r / beta]
        H.zero_(); cs.zero_(); sn.zero_(); g.zero_(); g[0] = beta
        total_iters += restart
    return x, 1, total_iters 