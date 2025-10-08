import torch
from typing import Callable, Tuple


def _matvec(A, x: torch.Tensor) -> torch.Tensor:
    """Унифицированное умножение матрицы (dense/sparse) или callable на вектор x."""
    if callable(A):
        return A(x)
    if x.dtype != A.dtype:
        x = x.to(A.dtype)
    if hasattr(A, "is_sparse_csr") and A.is_sparse_csr:
        return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    return (A @ x).to(x.dtype)


def _deflate(vec: torch.Tensor, basis: torch.Tensor | None):
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
           min_iters: int = 0) -> Tuple[torch.Tensor, int, int]:
    """Flexible GMRES (right-preconditioned, Saad'93).
    
    ПРАВИЛЬНАЯ схема:
    - z_j = M^{-1} v_j  (предобуславливаем базис V)
    - w = A z_j         (матвектор с предобусловленным)
    - x += Z y          (обновление через Z, не V!)
    
    Возвращает (x, info, iters), где info=0 ⇔ достигли tol.
    """
    device, dtype = b.device, b.dtype
    n = b.numel()
    x = torch.zeros_like(b)

    matvec = lambda v: _matvec(A, v)
    precond = (lambda r: r) if M is None else M

    b_norm = torch.norm(b)
    if b_norm < 1e-30 and min_iters == 0:
        return x, 0, 0

    def _robust_solve(H_sub: torch.Tensor, g_sub: torch.Tensor):
        try:
            if torch.isnan(H_sub).any() or torch.isinf(H_sub).any():
                raise RuntimeError("H contains NaN/Inf")
            cond = torch.linalg.cond(H_sub)
            if torch.isfinite(cond) and cond < 1e12:
                return torch.linalg.solve(H_sub, g_sub)
        except Exception:
            pass
        try:
            eye = torch.eye(H_sub.shape[1], device=H_sub.device, dtype=H_sub.dtype)
            H_reg = H_sub + 1e-12 * eye
            return torch.linalg.lstsq(H_reg, g_sub).solution
        except Exception:
            return torch.zeros_like(g_sub)

    total_iters = 0
    e1 = torch.zeros(restart + 1, device=device, dtype=dtype); e1[0] = 1.0

    k_outer = 0
    while total_iters < max_iter:
        # Истинный резидуал (right-precond): r = b - A x
        r = b - matvec(x)
        r = _deflate(r, deflation_basis)
        beta = torch.norm(r)
        if beta / b_norm < tol and total_iters >= min_iters:
            return x, 0, total_iters
        if beta < 1e-30:
            return x, 0, total_iters

        # Базы Крылова:
        V = [r / beta]          # v_j (ортонормальный базис в пространстве невязок)
        Z = []                  # z_j = M^{-1} v_j (предобусловленные векторы)
        H = torch.zeros((restart + 1, restart), device=device, dtype=dtype)
        cs = torch.zeros(restart, device=device, dtype=dtype)
        sn = torch.zeros(restart, device=device, dtype=dtype)
        g = beta * e1.clone()

        j_last = -1
        for j in range(restart):
            if total_iters >= max_iter:
                break

            vj = V[j]
            zj = precond(vj)        # ★ предобуславливаем v_j
            Z.append(zj)

            w = matvec(zj)          # ★ матвектор с z_j
            w = _deflate(w, deflation_basis)

            # Arnoldi
            for i in range(j + 1):
                H[i, j] = torch.dot(V[i], w)
                w = w - H[i, j] * V[i]
            H[j + 1, j] = torch.norm(w)

            if H[j + 1, j] < 1e-14:
                j_last = j
                break
            V.append(w / H[j + 1, j])

            # Givens rotations
            for i in range(j):
                tmp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = tmp
            denom = torch.sqrt(H[j, j] ** 2 + H[j + 1, j] ** 2)
            cs[j] = H[j, j] / denom
            sn[j] = H[j + 1, j] / denom
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]

            resid = torch.abs(g[j + 1])
            total_iters += 1
            j_last = j
            
            # Стандартная проверка сходимости по g[j+1]
            if (resid / b_norm < tol) and (total_iters >= min_iters):
                y = _robust_solve(H[:j + 1, :j + 1], g[:j + 1])
                # ★ обновление через Z, не V!
                x = x + sum(Z[i] * y[i] for i in range(j + 1))
                return x, 0, total_iters

        # Restart (или happy breakdown)
        y = _robust_solve(H[:j_last + 1, :j_last + 1], g[:j_last + 1]) if j_last >= 0 else torch.zeros(0, device=device, dtype=dtype)
        if j_last >= 0 and len(Z) >= j_last + 1:
            x = x + sum(Z[i] * y[i] for i in range(j_last + 1))
        else:
            return x, 1, total_iters

        # проверка после рестарта
        r = b - matvec(x)
        if torch.norm(r) / b_norm < tol and total_iters >= min_iters:
            return x, 0, total_iters

        k_outer += 1

    return x, 1, total_iters 