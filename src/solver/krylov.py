import torch
from typing import Callable

def gmres(matvec: Callable[[torch.Tensor], torch.Tensor],
          b: torch.Tensor,
          M: Callable[[torch.Tensor], torch.Tensor] = None,
          restart: int = 50,
          tol: float = 1e-6,
          maxiter: int = 400):
    """Базовый GMRES(m) с левым предобуславливанием.
    matvec: функция A(x)
    M: предобуславливатель (возвращает M⁻¹ v). Если None – без предп.
    Возвращает (x, info)  info=0 – успех.
    """
    device = b.device
    n = b.numel()
    x = torch.zeros_like(b)
    r = b.clone()
    if M is not None:
        r = M(r)
    beta = torch.norm(r)
    if beta < tol:
        return x, 0

    Q = []  # ортогональный базис Крылова
    H = torch.zeros((restart+1, restart), device=device)

    for outer in range(0, maxiter, restart):
        # начальный вектор
        v = r / beta
        Q = [v]
        g = torch.zeros(restart+1, device=device)
        g[0] = beta

        for j in range(restart):
            # матричное умножение
            w = matvec(Q[j])
            if M is not None:
                w = M(w)

            # модифицированная ортогонализация Грамма-Шмидта
            for i in range(j+1):
                H[i, j] = torch.dot(Q[i], w)
                w = w - H[i, j] * Q[i]
            H[j+1, j] = torch.norm(w)
            if H[j+1, j] == 0:
                k = j
                break
            Q.append(w / H[j+1, j])

            # решаем least-squares посредством QR-фактора (исп Torch lstsq)
            y = torch.linalg.lstsq(H[:j+2, :j+1], g[:j+2]).solution
            resid = torch.abs(g[j+1] - (H[j+1, :j+1] @ y))
            if resid < tol:
                k = j
                break
        else:
            k = restart-1

        # Обновление решения
        y = torch.linalg.lstsq(H[:k+2, :k+1], g[:k+2]).solution
        dx = sum(y[i] * Q[i] for i in range(k+1))
        x = x + dx

        # Проверка остатка
        r = b - matvec(x)
        if M is not None:
            r = M(r)
        if torch.norm(r) < tol:
            return x, 0
        beta = torch.norm(r)
    return x, 1 