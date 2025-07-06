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
    """🚀 ПРОМЫШЛЕННЫЙ GMRES для симулятора мирового уровня.

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

    # 🎯 ПРОМЫШЛЕННАЯ ДИАГНОСТИКА
    b_norm = torch.norm(b)
    print(f"  GMRES: ||b||={b_norm:.3e}, tol={tol:.3e}, restart={restart}, max_iter={max_iter}")
    
    if b_norm < 1e-15:
        print("  GMRES: Нулевая RHS, возвращаем ноль")
        return x, 0

    r = precond(b - _matvec(A, x))
    beta = torch.norm(r)
    print(f"  GMRES: Начальная невязка ||r||={beta:.3e}")
    
    if beta < tol * b_norm:
        print("  GMRES: Уже сошлось на старте")
        return x, 0

    # Givens параметры
    cs = torch.zeros(restart, device=device, dtype=dtype)
    sn = torch.zeros(restart, device=device, dtype=dtype)

    V = [r / beta]
    H = torch.zeros(restart + 1, restart, device=device, dtype=dtype)

    g = torch.zeros(restart + 1, device=device, dtype=dtype)

    outer = 0
    best_x = x.clone()
    best_residual = beta
    stagnation_count = 0
    
    while outer < max_iter:
        g.zero_()
        g[0] = beta
        
        # 🎯 ARNOLDI ПРОЦЕСС с улучшенной диагностикой
        for j in range(restart):
            w = precond(_matvec(A, V[j]))
            
            # 🎯 МОНИТОРИНГ качества предобуславливателя
            if j == 0:
                precond_effect = torch.norm(w) / torch.norm(V[j])
                print(f"  GMRES: Эффективность предобуславливателя: {precond_effect:.3e}")
            
            # ортогонализация Gram-Schmidt
            for i in range(j + 1):
                H[i, j] = torch.dot(V[i], w)
                w = w - H[i, j] * V[i]
                
            H[j + 1, j] = torch.norm(w)
            
            # 🎯 ПРОВЕРКА на breakdown
            if H[j + 1, j] < 1e-15:
                print(f"  GMRES: Breakdown на j={j}, ||w||={H[j + 1, j]:.3e}")
                # Добавляем случайный вектор для продолжения
                w = torch.randn_like(w) * 1e-12
                H[j + 1, j] = torch.norm(w)
            
            V.append(w / H[j + 1, j])
            
            # применяем предыдущие вращения
            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp
                
            # новая ротация
            denom = torch.sqrt(H[j, j] ** 2 + H[j + 1, j] ** 2)
            if denom < 1e-15:
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
            relative_residual = residual / b_norm
            
            # 🎯 ДИНАМИЧЕСКОЕ логирование прогресса
            if j % 10 == 0 or j < 5:
                print(f"  GMRES: j={j}, ||r||={residual:.3e}, rel={relative_residual:.3e}")
            
            # 🎯 ПРОВЕРКА сходимости
            if relative_residual < tol:
                print(f"  GMRES: Сошлось на j={j}!")
                # вычисляем решение
                try:
                    y = torch.linalg.solve(H[:j + 1, :j + 1], g[:j + 1])
                    max_i = min(j + 1, len(V))
                    update = sum(y[i] * V[i] for i in range(max_i))
                    x = x + update
                    
                    # 🎯 ФИНАЛЬНАЯ проверка
                    final_residual = torch.norm(b - _matvec(A, x))
                    print(f"  GMRES: Финальная невязка: {final_residual:.3e}")
                    return x, 0
                except Exception as e:
                    print(f"  GMRES: Ошибка в решении системы: {e}")
                    break
            
            # 🎯 СОХРАНЕНИЕ лучшего решения
            if residual < best_residual:
                best_residual = residual
                try:
                    y = torch.linalg.solve(H[:j + 1, :j + 1], g[:j + 1])
                    max_i = min(j + 1, len(V))
                    update = sum(y[i] * V[i] for i in range(max_i))
                    best_x = x + update
                    stagnation_count = 0
                except:
                    pass
            else:
                stagnation_count += 1
                
        # 🎯 ПЕРЕЗАПУСК с улучшенной стратегией
        print(f"  GMRES: Перезапуск после {restart} итераций, ||r||={residual:.3e}")
        
        try:
            y = torch.linalg.solve(H[:restart, :restart], g[:restart])
            max_i = min(restart, len(V))
            update = sum(y[i] * V[i] for i in range(max_i))
            x = x + update
        except Exception as e:
            print(f"  GMRES: Ошибка в перезапуске: {e}, используем лучшее решение")
            x = best_x.clone()
            
        # новый резидуал
        r = precond(b - _matvec(A, x))
        beta = torch.norm(r)
        relative_residual = beta / b_norm
        
        print(f"  GMRES: После перезапуска: ||r||={beta:.3e}, rel={relative_residual:.3e}")
        
        if relative_residual < tol:
            print("  GMRES: Сошлось после перезапуска!")
            return x, 0
            
        # 🎯 АДАПТИВНАЯ стратегия против стагнации
        if stagnation_count > 20:
            print("  GMRES: Стагнация обнаружена, используем лучшее решение")
            return best_x, 1
            
        # подготовка к следующему циклу
        V = [r / beta]
        H.zero_()
        cs.zero_()
        sn.zero_()
        outer += restart
        
    # 🎯 ВОЗВРАТ лучшего найденного решения
    print(f"  GMRES: Не сошлось за {max_iter} итераций")
    print(f"  GMRES: Лучшая невязка: {best_residual:.3e}")
    return best_x, 1 