# tools/diag_jfnk_mg.py
import os
import numpy as np

# ---------- Утилиты для проекции и масштабов ----------

def default_project(x_hat, l_hat=None, u_hat=None):
    """Проекция на боксы в hat-пространстве (если есть).
    Если у тебя уже есть своя P(x_hat), передай её вместо этой."""
    if l_hat is None and u_hat is None:
        return x_hat
    if l_hat is None: l_hat = -np.inf*np.ones_like(x_hat)
    if u_hat is None: u_hat =  np.inf*np.ones_like(x_hat)
    return np.minimum(np.maximum(x_hat, l_hat), u_hat)

def project_direction(x_hat, v_hat, l_hat=None, u_hat=None):
    """Проекция направления: компоненты, «упертые» в активные границы, зануляем,
    чтобы FD не прыгал через активные constraints."""
    if l_hat is None and u_hat is None:
        return v_hat
    v = v_hat.copy()
    if l_hat is None: l_hat = -np.inf*np.ones_like(x_hat)
    if u_hat is None: u_hat =  np.inf*np.ones_like(x_hat)
    active_low  = np.isclose(x_hat, l_hat)
    active_high = np.isclose(x_hat, u_hat)
    v[active_low & (v<0)] = 0.0
    v[active_high & (v>0)] = 0.0
    return v

def hat_fd_step(x_hat, v_hat, eps0=1e-7):
    """Компонентный шаг в hat-пространстве: h_i = eps0 * max(1, |x_i|)."""
    mag = np.maximum(1.0, np.abs(x_hat))
    h = eps0 * mag
    # нормируем направление, чтобы типовой ||v||2 ~ 1
    vn = np.linalg.norm(v_hat) + 1e-30
    return h * (v_hat / vn)

# ---------- FD-Jv, согласованный с проекцией и масштабами ----------

def Jv_fd_consistent(F_hat, x_hat, v_hat, project=default_project, l_hat=None, u_hat=None, eps0=1e-7):
    """
    F_hat: callable(x_hat)->F(x_hat), вся нелинейка строго в hat-пространстве
    x_hat: текущий вектор состояния
    v_hat: направление (в hat)
    project: проекция P(x) на допустимую область (в hat)
    l_hat/u_hat: границы в hat (если есть)
    eps0: базовый безразмерный шаг (подбирается сканом)

    Алгоритм:
      1) Проецируем направление (с учетом активных ограничений)
      2) Компонентный шаг h в hat-пространстве
      3) Возмущаем состояние: x+ = P(x + h)
      4) Jv ≈ (F(x+) - F(x)) / (эффективный скаляр eps), где eps ~ средний масштаб h вдоль v
    """
    v_proj = project_direction(x_hat, v_hat, l_hat, u_hat)
    if np.allclose(v_proj, 0.0):
        return np.zeros_like(v_hat)

    h = hat_fd_step(x_hat, v_proj, eps0=eps0)
    x_plus = project(x_hat + h, l_hat, u_hat)

    Fx = F_hat(x_hat)
    Fx_plus = F_hat(x_plus)

    # Эффективный "скалярный" шаг: средняя величина проекции h на v (чтобы не делить поэлементно)
    denom = np.dot(np.abs(v_proj), np.abs(h)) / (np.linalg.norm(v_proj) + 1e-30)
    denom = max(denom, 1e-30)
    Jv = (Fx_plus - Fx) / denom
    return Jv

# ---------- Диагностики: η-скан и проверка "линейной модели" ----------

def eta_scan(F_hat, x_hat, project=default_project, l_hat=None, u_hat=None,
             nvec=3, seed=42, etas=None):
    """
    Сканируем φ(η) = ||F(P(x+η v)) - F(x)|| / η по лог-сетке η.
    Ищем плато линейности. Возвращаем рекомендации по eps0.
    """
    if etas is None:
        etas = np.geomspace(1e-12, 1e-2, 13)

    rng = np.random.default_rng(seed)
    Fx = F_hat(x_hat)
    base = np.linalg.norm(Fx)
    out = []
    for k in range(nvec):
        v = rng.normal(size=x_hat.size)
        v = v / (np.linalg.norm(v) + 1e-30)
        row = []
        for eta in etas:
            x_eta = project(x_hat + eta * project_direction(x_hat, v, l_hat, u_hat), l_hat, u_hat)
            phi = np.linalg.norm(F_hat(x_eta) - Fx) / eta
            row.append(phi)
        out.append(row)

    out = np.array(out)  # shape = (nvec, len(etas))
    # примитивная эвристика для "плато": берем среднее по векторам
    mean_curve = out.mean(axis=0)
    # ищем диапазон, где кривая относительно плоская
    diffs = np.abs(np.gradient(np.log10(mean_curve + 1e-300), np.log10(etas)))
    plateau_idx = np.where(diffs < 0.1)[0]
    rec_eps = None
    if plateau_idx.size > 0:
        # берем середину плато
        rec_eps = float(etas[int(np.median(plateau_idx))])

    print("\n[JFNK-DIAG] ETA-SCAN")
    print("  etas         :", " ".join([f"{e:.0e}" for e in etas]))
    for i, row in enumerate(out):
        print(f"  vec#{i+1}  φ(η):", " ".join([f"{v: .2e}" for v in row]))
    print("  mean φ(η)    :", " ".join([f"{v: .2e}" for v in mean_curve]))
    if rec_eps is not None:
        print(f"  ⇒ Рекомендуемый eps0 (hat) ~ {rec_eps:.1e} (середина плато)")
    else:
        print("  ⇒ Плато не найдено — FD-шаги несогласованы (проверь проекцию/клиппинг).")
    return rec_eps, etas, out

def newton_model_check(F_hat, x_hat, delta_hat, project=default_project, l_hat=None, u_hat=None, eps0=1e-7):
    """
    Сравнивает предсказанное линейной моделью уменьшение ||F|| с фактическим.
    Возвращает список (alpha, ||F(x+αδ)||, pred_norm) и cos угла между F'(x)δ и разностью F(x+αδ)-F(x).
    """
    Fx = F_hat(x_hat)
    Jv = Jv_fd_consistent(F_hat, x_hat, delta_hat, project, l_hat, u_hat, eps0)
    norms = []
    alphas = [1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5]
    for a in alphas:
        x_new = project(x_hat + a * project_direction(x_hat, delta_hat, l_hat, u_hat), l_hat, u_hat)
        F_new = F_hat(x_new)
        pred = Fx + a * Jv
        norms.append((a, np.linalg.norm(F_new), np.linalg.norm(pred)))
    # косинус между F_new - Fx и Jv при α=1e-2 (как в твоих логах)
    a_ref = 1e-2
    x_ref = project(x_hat + a_ref * project_direction(x_hat, delta_hat, l_hat, u_hat), l_hat, u_hat)
    dF = F_hat(x_ref) - Fx
    num = np.dot(dF, Jv)
    den = (np.linalg.norm(dF) + 1e-30) * (np.linalg.norm(Jv) + 1e-30)
    cosang = num / den
    print("\n[JFNK-DIAG] NEWTON MODEL CHECK")
    for a, nf, npred in norms:
        print(f"  α={a: .0e}:  ||F(x+αδ)||={nf: .3e}  ||Fx + α Jv||={npred: .3e}")
    print(f"  cos( dF , Jv ) @ α=1e-2 : {cosang: .3f}  (близко к 1 — хорошо)")
    return norms, cosang

# ---------- «Внешний» хелпер для быстрого вызова из твоего bench ----------

def run_jfnk_diagnostics(F_hat, x_hat, delta_hat=None, project=default_project, l_hat=None, u_hat=None):
    """
    Быстрая проверка: скан ε, сверка модели Ньютона, рекомендации по eps0.
    delta_hat: можешь передать последнее направление из GMRES; если None — возьмём случайное.
    """
    rec_eps, etas, scan = eta_scan(F_hat, x_hat, project, l_hat, u_hat, nvec=3)
    if delta_hat is None:
        v = np.random.default_rng(0).normal(size=x_hat.size); v /= (np.linalg.norm(v)+1e-30)
        delta_hat = v
    newton_model_check(F_hat, x_hat, delta_hat, project, l_hat, u_hat, eps0=rec_eps or 1e-7)

# tools/diag_jfnk_mg.py  (добавить в конец файла)

def _apply_op(A, x):
    # <<< ADAPT HERE >>> если A — не numpy-матрица, а линейный оператор с .dot(x) или __call__
    if hasattr(A, 'dot'): 
        return A.dot(x)
    if callable(A): 
        return A(x)
    raise TypeError("Don't know how to apply operator A")

def _apply_R(R, x):
    # <<< ADAPT HERE >>> аналогично для рестриктора
    if hasattr(R, 'dot'):
        return R.dot(x)
    if callable(R):
        return R(x)
    raise TypeError("Don't know how to apply restriction R")

def _apply_P(P, x):
    # <<< ADAPT HERE >>> аналогично для пролонгации
    if hasattr(P, 'dot'):
        return P.dot(x)
    if callable(P):
        return P(x)
    raise TypeError("Don't know how to apply prolongation P")

def mg_level_norms(mg, r0):
    """Спускаем r по уровням: r_ℓ = R r_{ℓ-1}, печатаем ||r_ℓ||2 и размер."""
    r = r0.copy()
    print("\n[MG-DIAG] Restriction chain (residual norms by level):")
    print(f"  L0: shape={r.shape[0]:6d}  ||r||2={np.linalg.norm(r): .3e}")
    for ell in range(1, len(mg.levels)):
        R = mg.levels[ell].R  # <<< ADAPT HERE >>> возьми правильный R для перехода ℓ-1 -> ℓ
        r = _apply_R(R, r)
        print(f"  L{ell}: shape={r.shape[0]:6d}  ||r||2={np.linalg.norm(r): .3e}")
    return

def mg_check_rap(mg, ntests=3, seed=0):
    """
    Проверяет A_c P ≈ R A_f P для каждого перехода ℓ->ℓ+1.
    Печатает относительный дефект ||(A_c P - R A_f P) z|| / ||z||.
    """
    rng = np.random.default_rng(seed)
    print("\n[MG-DIAG] RAP consistency checks:")
    for ell in range(len(mg.levels) - 1):
        Af = mg.levels[ell].A
        Ac = mg.levels[ell+1].A
        R  = mg.levels[ell+1].R  # рестриктор с fine->coarse
        P  = mg.levels[ell].P    # пролонгатор coarse->fine (обратная связь)
        errs = []
        for _ in range(ntests):
            zc = rng.normal(size=mg.levels[ell+1].A.shape[0])
            Pfzc  = _apply_P(P, zc)          # fine-sized vector
            left  = _apply_op(Ac, zc)        # A_c z
            left2 = _apply_P(P, left)        # P(A_c z) -> fine
            right = _apply_op(Af, Pfzc)      # A_f (P z)
            right2= _apply_R(R, right)       # R A_f P z -> coarse
            # сравним на coarse (лучше): A_c z  vs  R A_f P z
            num = np.linalg.norm(left - right2)
            den = np.linalg.norm(left) + np.linalg.norm(right2) + 1e-30
            errs.append(num / (den/2))
        print(f"  ℓ={ell}→{ell+1}: mean rel-defect ≈ {np.mean(errs): .2e} (↓ 1e-12…1e-6 хорошо)")
    return

def mg_cap_levels(mg, min_dim=8):
    """
    Возвращает число допустимых уровней, пока все размеры ≥ min_dim и чётные.
    Можно использовать, чтобы отключить «мертвый» L3 на 60×60×30.
    """
    max_levels = 1
    for ell, lev in enumerate(mg.levels):
        # <<< ADAPT HERE >>> получи (nx,ny,nz) уровня. Если нет, возьми размер вектора и оцени кубический корень.
        shape = getattr(lev, "shape", None)
        if shape is None:
            n = lev.A.shape[0]
            # грубо как куб:
            k = int(round(n ** (1/3)))
            shape = (k,k,k)
        if any((d < min_dim) or (d % 2 == 1) for d in shape if d is not None):
            break
        max_levels = ell+1
    print(f"\n[MG-DIAG] level cap suggestion: use first {max_levels} level(s)")
    return max_levels

def run_mg_diagnostics(mg, r0):
    mg_level_norms(mg, r0)
    mg_check_rap(mg)
    mg_cap_levels(mg)
