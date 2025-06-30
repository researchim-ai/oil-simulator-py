import torch
from typing import Callable, Tuple


def _matvec(A, x: torch.Tensor) -> torch.Tensor:
    """Возвращает A @ x для dense/sparse или callable."""
    if callable(A):
        return A(x)
    if x.dtype != A.dtype:
        x = x.to(A.dtype)
    if A.is_sparse_csr:
        return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    return (A @ x).to(x.dtype)


def gmres(A, b: torch.Tensor, M: Callable[[torch.Tensor], torch.Tensor] = None,
         tol: float = 1e-8, restart: int = 50, max_iter: int = 400) -> Tuple[torch.Tensor, int]:
    """Простой GMRES с перезапуском (flexible=false).

    Parameters
    ----------
    A : матрица (dense, sparse_csr) или callable v -> A v
    b : RHS
    M : предобуславливатель, функция r -> M^{-1} r
    tol : относительная норма невязки
    restart : размер подпространства Крылова
    max_iter : макс. итераций (Arnoldi шагов)
    Returns
    -------
    x, info  (info=0 если сошлось, 1 иначе)
    """
    device = b.device
    dtype = b.dtype
    n = b.numel()
    x = torch.zeros_like(b)
    if M is None:
        precond = lambda r: r
    else:
        precond = M

    r = precond(b - _matvec(A, x))
    beta = torch.norm(r)
    if beta < tol:
        return x, 0

    # Givens параметры
    cs = torch.zeros(restart, device=device, dtype=dtype)
    sn = torch.zeros(restart, device=device, dtype=dtype)

    V = [r / beta]
    H = torch.zeros(restart + 1, restart, device=device, dtype=dtype)

    g = torch.zeros(restart + 1, device=device, dtype=dtype)

    outer = 0
    while outer < max_iter:
        g.zero_()
        g[0] = beta
        for j in range(restart):
            w = precond(_matvec(A, V[j]))
            # ортогонализация
            for i in range(j + 1):
                H[i, j] = torch.dot(V[i], w)
                w = w - H[i, j] * V[i]
            H[j + 1, j] = torch.norm(w)
            if H[j + 1, j] != 0:
                V.append(w / H[j + 1, j])
            # применяем предыдущие вращения
            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp
            # новая ротация
            denom = torch.sqrt(H[j, j] ** 2 + H[j + 1, j] ** 2)
            if denom == 0:
                cs[j] = 1.0
                sn[j] = 0.0
            else:
                cs[j] = H[j, j] / denom
                sn[j] = H[j + 1, j] / denom
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            # обновляем g
            temp = cs[j] * g[j] + sn[j] * g[j + 1]
            g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1]
            g[j] = temp
            residual = torch.abs(g[j + 1])
            if residual < tol:
                # вычисляем решение
                y = torch.linalg.solve(H[:j + 1, :j + 1], g[:j + 1])
                update = sum(y[i] * V[i] for i in range(j + 1))
                x = x + update
                return x, 0
        # перезапуск
        y = torch.linalg.solve(H[:restart, :restart], g[:restart])
        update = sum(y[i] * V[i] for i in range(restart))
        x = x + update
        # новый резид
        r = precond(b - _matvec(A, x))
        beta = torch.norm(r)
        if beta < tol:
            return x, 0
        # подготовка к следующему циклу
        V = [r / beta]
        H.zero_()
        cs.zero_()
        sn.zero_()
        outer += restart
    return x, 1  # не сошлось 