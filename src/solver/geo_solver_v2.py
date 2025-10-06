"""Geo-AMG v2 – каркас с эквилибрированием D^-1 A D^-1 и GeoLevel-иерархией.

! ВАЖНО: это минимально рабочая версия для первых тестов на 32^3 и 60×60×30.
  Сглаживатель – damped Jacobi, один V-cycle на итерацию.
  K-cycle/Chebyshev/L1-GS будут добавлены позже.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
import os
import numpy as np

from solver.geo_level import GeoLevel, build_level_csr, build_level_from_csr


__all__ = ["GeoSolverV2"]

def _csr_diag_pos(crow: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    """Безопасно возвращает индексы диагональных элементов CSR.

    Если для какой-либо строки диагональ отсутствует, выбрасывает RuntimeError.
    (В большинстве наших операторов диагональ должна быть всегда.)
    """
    n_rows = int(crow.numel() - 1)
    row_idx = torch.repeat_interleave(torch.arange(n_rows, device=col.device), crow[1:] - crow[:-1])
    pos_all = torch.nonzero(col == row_idx, as_tuple=False).squeeze(1)
    diag_idx = torch.full((n_rows,), -1, dtype=torch.int64, device=col.device)
    if pos_all.numel() > 0:
        rows = row_idx[pos_all]
        diag_idx[rows] = pos_all
    if (diag_idx < 0).any():
        missing = int((diag_idx < 0).sum().item())
        raise RuntimeError(f"CSR has {missing} rows without diagonal")
    return diag_idx

def _row_equilibrate_csr(A_csr: torch.Tensor, eps: float = 1e-20, *, lvl: GeoLevel | None = None):
    """Left scaling: W A, где W = diag(1/||row||₁). Возвращает (A_scaled, w_rows).
    Защита от повторного применения на уровне: если lvl._row_scaled, пропускаем.
    """
    if lvl is not None and getattr(lvl, "_row_scaled", False):
        # уже эквилибрировано ранее – возвращаем текущие веса (единицы по умолчанию)
        w_prev = getattr(lvl, "W_rows", None)
        if w_prev is None:
            w_prev = torch.ones(A_csr.size(0), device=A_csr.device, dtype=A_csr.dtype)
        return A_csr, w_prev
    crow = A_csr.crow_indices()
    col  = A_csr.col_indices()
    vals = A_csr.values()

    row_counts = crow[1:] - crow[:-1]
    row_idx = torch.repeat_interleave(torch.arange(crow.numel()-1, device=A_csr.device), row_counts)
    row_abs_sum = torch.zeros(crow.numel()-1, device=A_csr.device, dtype=vals.dtype)
    row_abs_sum.index_add_(0, row_idx, vals.abs())

    w = 1.0 / torch.clamp(row_abs_sum, min=eps)
    vals.mul_(w[row_idx])   # только слева (по строкам)
    if lvl is not None:
        lvl._row_scaled = True
    return A_csr, w


def build_block_maps(shape_f, device, coarsen=(2, 2, 2)):
    """Маппинг fine→coarse для общего (cz, cy, cx): (2,2,2) или (1,2,2) и др.
    Возвращает parent_idx, shape_c, n_c, child_cnt.
    """
    nz, ny, nx = shape_f
    cz, cy, cx = coarsen
    nz_c = (nz + cz - 1) // cz
    ny_c = (ny + cy - 1) // cy
    nx_c = (nx + cx - 1) // cx

    z = torch.arange(nz, device=device)
    y = torch.arange(ny, device=device)
    x = torch.arange(nx, device=device)
    Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")

    Zc = torch.div(Z, cz, rounding_mode="floor")
    Yc = torch.div(Y, cy, rounding_mode="floor")
    Xc = torch.div(X, cx, rounding_mode="floor")
    parent = (Zc * ny_c + Yc) * nx_c + Xc

    parent_idx = parent.reshape(-1).to(torch.int64)
    n_c = int(parent_idx.max().item()) + 1
    child_cnt = torch.bincount(parent_idx, minlength=n_c).to(torch.int64)

    return parent_idx, (nz_c, ny_c, nx_c), n_c, child_cnt


def build_P_csr(shape_f, device, coarsen=(2, 2, 2)):
    """Piecewise-constant prolongation с произвольным coarsen."""
    parent_idx, shape_c, n_c, child_cnt = build_block_maps(shape_f, device, coarsen=coarsen)
    n_f = parent_idx.numel()

    crow = torch.arange(n_f + 1, device=device, dtype=torch.int64)
    col  = parent_idx
    val  = torch.ones(n_f, device=device, dtype=torch.float64)

    P = torch.sparse_csr_tensor(crow, col, val, size=(n_f, n_c),
                                device=device, dtype=torch.float64)
    return P, parent_idx, n_c, child_cnt, shape_c


def build_R_csr(P, child_cnt, *, weights: torch.Tensor | None = None,
                mode: str = "uniform", gamma: float = 1.0):
    """Restriction R = W · P^T:
    - mode="uniform": W = diag(1/child_cnt)
    - mode="weighted": W = diag( w_child / sum_coarse(w_child) ), где w_child берём из `weights` (L1-сумма строки или |Aii|)
    """
    P_coo = P.to_sparse_coo()
    fine = P_coo.indices()[0]
    coarse = P_coo.indices()[1]
    vals = P_coo.values()

    if mode == "weighted" and weights is not None:
        w_child = weights[fine].to(vals.dtype).clamp_min(1e-30)
        if gamma != 1.0:
            w_child = w_child.pow(gamma)
        denom = torch.zeros(P.shape[1], device=P.device, dtype=vals.dtype)
        denom.index_add_(0, coarse, w_child)
        denom = torch.clamp(denom, min=1e-30)
        w = (w_child / denom[coarse]) * vals
    else:
        cnt = child_cnt[coarse].to(vals.dtype).clamp_min(1.0)
        w = (1.0 / cnt) * vals

    Rt_indices = torch.stack([coarse, fine], dim=0)
    Rt_coo = torch.sparse_coo_tensor(Rt_indices, w,
                                     size=(P.shape[1], P.shape[0]),
                                     device=P.device, dtype=P.dtype)
    return Rt_coo.to_sparse_csr()



def rap_pc_const_gpu(Af_csr: torch.Tensor,
                     parent_idx: torch.Tensor,
                     n_c: int,
                     child_cnt,
                     *,
                     weights_row: torch.Tensor | None = None,
                     mode: str = "uniform") -> torch.Tensor:
    """
    Ac = R Af P для piecewise-constant P и усредняющего R = diag(1/child_cnt) · Pᵀ.
    GPU: агрегируем строки/столбцы Af по parent_idx.
    """
    device = Af_csr.device
    crow = Af_csr.crow_indices()
    col  = Af_csr.col_indices()
    val  = Af_csr.values()

    # индексы fine-строк для каждого nnz
    row_counts = crow[1:] - crow[:-1]
    row_f = torch.repeat_interleave(torch.arange(crow.numel()-1, device=device, dtype=col.dtype),
                                    row_counts)

    # coarse индексы
    I = parent_idx[row_f.long()]   # coarse row
    J = parent_idx[col.long()]     # coarse col

    # вес из R по строке coarse I
    if mode == "weighted" and weights_row is not None:
        w_child = weights_row[row_f.long()].to(val.dtype).clamp_min(1e-30)
        denom = torch.zeros(n_c, device=device, dtype=val.dtype)
        denom.index_add_(0, I, w_child)
        w = (w_child / denom[I]) * val
    else:
        cntI = child_cnt[I].to(val.dtype).clamp_min(1.0)
        w = val / cntI

    idx = torch.stack([I, J], dim=0)  # 2 x nnz
    Ac = torch.sparse_coo_tensor(idx, w, (n_c, n_c),
                                 device=device, dtype=val.dtype).coalesce().to_sparse_csr()
    # санитация NaN/Inf
    v = Ac.values()
    if not torch.isfinite(v).all():
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        Ac = torch.sparse_csr_tensor(Ac.crow_indices(), Ac.col_indices(), v,
                                     size=Ac.size(), device=Ac.device, dtype=Ac.dtype)
    return Ac


def _stats(name, x):
    import torch, numpy as np
    if torch.is_tensor(x):
        cpu = x.detach().cpu()
        arr = cpu.numpy()
    else:
        arr = np.asarray(x)
    fin = np.isfinite(arr)
    print(f"[{name}] shape={arr.shape} "
          f"min={arr.min():.3e} max={arr.max():.3e} "
          f"norm2={np.linalg.norm(arr):.3e} "
          f"nan={np.isnan(arr).sum()} inf={np.isinf(arr).sum()} "
          f"finite%={(fin.sum()/fin.size*100):.2f}%")

# ==== DEBUG HELPERS =========================================================
def _vstats(tag, v):
    return (f"{tag}: ||·||2={v.norm():.3e}  ||·||inf={v.abs().max():.3e}  "
            f"min={v.min():.3e}  max={v.max():.3e}")

def _alpha_stats(r_pre, corr_vec, A_apply):
    A_corr = A_apply(corr_vec)
    num = torch.dot(r_pre, corr_vec)
    den = torch.dot(corr_vec, A_corr).clamp_min(1e-30)
    alpha = (num / den).clamp_max(1.0)
    return alpha, num, den, A_corr
# ============================================================================


class GeoSolverV2:
    """Экспериментальная реализация геометрического AMG с эквилибрированием."""

    def __init__(self, reservoir, *, omega: float = 0.8,
                 max_coarse_ratio: int = 500, device: str | None = None,
                 cycle_type: str = "W", cycles_per_call: int = 2,
                 pre_smooth: int = 3, post_smooth: int = 3,
                 omega_fine: float = 0.8,
                 smoother_fine: str = "rbgs",  # rbgs|linez|chebyshev
                 cheby_tail: int = 0,
                 delta_clip_factor: float | None = 1000.0,
                 clip_kappa: float = 5.0,
                 max_levels: int | None = None,
                 debug: bool | None = None,
                 default_tol: float = 1e-6,
                 default_max_iter: int = 10,
                 rap_check_debug: bool = True,
                 rap_max_check_n: int = 300000,
                 restrict_mode: str = "weighted",
                 semicoarsen: bool = True,
                 kcycle: bool = True,
                 cheby_kappa: float = 80.0,
                 smooth_prolong: bool = True,
                 prolong_omega: float = 0.67,
                 prolong_sweeps: int = 1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.omega = float(omega)
        self.cycle_type = cycle_type.upper()
        self.cycles_per_call = max(1, cycles_per_call)
        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth
        self.omega_fine = omega_fine
        self.cheby_tail = cheby_tail
        self.smoother_fine = smoother_fine.lower()
        self.delta_clip_factor = delta_clip_factor
        self.clip_kappa = clip_kappa
        self.default_tol = default_tol
        self.default_max_iter = default_max_iter
        self.rap_check_debug = bool(rap_check_debug)
        try:
            self.rap_max_check_n = int(os.environ.get("GEO_RAP_DEBUG_MAX_N", str(int(rap_max_check_n))))
        except Exception:
            self.rap_max_check_n = int(rap_max_check_n)
        self.restrict_mode = restrict_mode
        self.cheby_kappa = float(cheby_kappa)
        self.semicoarsen = bool(semicoarsen)
        self.kcycle = bool(kcycle)
        # Параметры усиления рестрикции/адаптивного цикла/проекции константы
        try:
            self.restrict_gamma = float(os.environ.get("GEO_R_GAMMA", "1.5"))
        except Exception:
            self.restrict_gamma = 1.5
        self.project_const = True
        self._rho_hist: dict[int, float] = {}
        # Smoothed prolongation (on-the-fly SA)
        self.smooth_prolong = bool(smooth_prolong)
        self.prolong_omega = float(prolong_omega)
        self.prolong_sweeps = int(max(1, prolong_sweeps))

        # Если активен левый row-scale (по умолчанию да) — Chebyshev может быть нестабилен.
        # Переведём сглаживатель на RBGS по умолчанию.
        if self.smoother_fine == "chebyshev":
            if os.environ.get("OIL_DEBUG", "0") == "1":
                print("[geo2] Chebyshev отключён (левый row-scale ломает SPD). Переключаемся на RBGS.")
            self.smoother_fine = "rbgs"


        # Режим подробного лога (env OIL_DEBUG=1 или явный debug=True)
        if debug is None:
            debug = bool(int(os.environ.get("OIL_DEBUG", "0")))
        self.debug = debug

        # 1. Перенос проницаемостей (permute к (nz, ny, nx))
        kx = reservoir.permeability_x.to(self.device, dtype=torch.float64).permute(2, 1, 0).contiguous()
        ky = reservoir.permeability_y.to(self.device, dtype=torch.float64).permute(2, 1, 0).contiguous()
        kz = reservoir.permeability_z.to(self.device, dtype=torch.float64).permute(2, 1, 0).contiguous()

        self.nz, self.ny, self.nx = kx.shape
        self.hx, self.hy, self.hz = map(float, reservoir.grid_size)

        # 2. Базовый уровень: сбор CSR
        base_lvl = GeoLevel(kx, ky, kz, self.hx, self.hy, self.hz, device=self.device)

        # --- [NEW] Left row equilibration ---------------------------------

        base_lvl.A_csr, w_rows0 = _row_equilibrate_csr(base_lvl.A_csr, lvl=base_lvl)
        base_lvl.diag = base_lvl.diag * w_rows0
        # ВАЖНО: если A ← W·A, то RHS должен быть b_hat ← W·b_hat.
        # Нормализуем и клампим W_rows для численной устойчивости.
        w = w_rows0.clone()
        med_w = torch.median(w)
        if torch.isfinite(med_w) and med_w > 0:
            w = w / med_w
        w = torch.clamp(w, 1e-6, 1e6)
        self.W_rows = w
        if self.debug:
            print(f"[Geo2] W_rows L0: min={self.W_rows.min().item():.3e} med={torch.median(self.W_rows).item():.3e} max={self.W_rows.max().item():.3e}")

        def _pin_rowcol(A_csr, idx):
            crow = A_csr.crow_indices(); col = A_csr.col_indices(); val = A_csr.values()
            s, e = int(crow[idx].item()), int(crow[idx+1].item())
            # зануляем строку и столбец
            val[s:e] = 0.0
            mask = (col == idx)
            val[mask] = 0.0
            # ставим 1 на диагональ: если нет диага в строке – используем первую позицию строки и правим col
            rel = torch.nonzero(col[s:e] == idx, as_tuple=False)
            if rel.numel():
                val[s + int(rel[0])] = 1.0
            else:
                if e == s:
                    raise RuntimeError("Empty CSR row — cannot pin")
                val[s] = 1.0
                col[s] = int(idx)
        diag_orig = base_lvl.diag

        # -------- Дополнительная диагностика строкового преобладания --------
        if self.debug:
            crow = base_lvl.A_csr.crow_indices()
            vals = base_lvl.A_csr.values().abs()
            row_counts = crow[1:] - crow[:-1]
            row_idx = torch.repeat_interleave(torch.arange(crow.numel()-1, device=self.device), row_counts)
            row_abs_sum = torch.zeros_like(diag_orig)
            row_abs_sum.index_add_(0, row_idx, vals)
            off_diag_sum = row_abs_sum - diag_orig.abs()
            ratio = off_diag_sum / diag_orig.abs().clamp_min(1e-30)
            print(f"[CHECK] row ratio off/diag: min={ratio.min().item():.3e}, median={ratio.median().item():.3e}, max={ratio.max().item():.3e}")
            worst = torch.topk(ratio, 5).values
            print(f"[CHECK] worst 5 ratios: {worst.tolist()}")

        # --- DEBUG: исходная диагональ и норма матрицы ----
        # исходная диагональ и норма матрицы – только один раз
        if self.debug and not hasattr(self, "_dbg_logged"):
            a_vals = base_lvl.A_csr.values()
            print(f"[DBG] A_phys diag min={diag_orig.min().item():.3e}, max={diag_orig.max().item():.3e}")
            print(f"[DBG] ||A_phys||_1 = {torch.norm(a_vals,1).item():.3e}")
            self._dbg_logged = True
        # Первое эквилибрирование: S1 = diag_orig^{-1/4}
        # Симметричное эквилибрирование в 2 шага:
        # S1 = diag(A0)^(-1/4)  → A1 = S1 A0 S1
        # S2 = diag(A1)^(-1/2) → A2 = S2 A1 S2  (diag(A2)≈1)
        scale_full = 1.0 / torch.sqrt(diag_orig.clamp_min(1e-20))  # diag^{-1/2}
        scale_sqrt = torch.sqrt(scale_full)                         # diag^{-1/4} (S1)
        # применяем S1
        crow = base_lvl.A_csr.crow_indices()
        col  = base_lvl.A_csr.col_indices()
        vals = base_lvl.A_csr.values()
        row_counts = crow[1:] - crow[:-1]
        row_idx = torch.repeat_interleave(
            torch.arange(crow.numel()-1, device=self.device), row_counts
        )
        vals.mul_(scale_sqrt[row_idx] * scale_sqrt[col])

        diag_idx_tmp = _csr_diag_pos(crow, col)
        assert diag_idx_tmp.numel() == (crow.numel()-1), "CSR diag not found for some rows"

        diag_tmp = vals[diag_idx_tmp].abs().clone()
        second_scale = 1.0 / torch.sqrt(diag_tmp.clamp_min(1e-20))  # S2
        # применяем S2
        vals.mul_(second_scale[row_idx] * second_scale[col])

        diag_final = vals[diag_idx_tmp].abs().clone()
        base_lvl.diag = diag_final  # теперь diag ≈ 1
        # сохраним «физическую» диагональ до симм‑масштаба для последующих весов/anchor
        base_lvl.diag_phys = diag_tmp.clone()

        # === PIN null mode AFTER equilibration ===
        # Anchor выбираем по «физической» диагонали ДО симм‑масштаба
        self.anchor_fine = int(base_lvl.diag_phys.argmax().item())
        _pin_rowcol(base_lvl.A_csr, self.anchor_fine)
        base_lvl.anchor = self.anchor_fine
        base_lvl.diag[self.anchor_fine] = 1.0
        # inv_l1 пересчитается ниже, но на всякий случай обнулим позже ещё раз

# --- пересчёт inv_l1 для базового уровня в эквилибрированном масштабе ---
        crow = base_lvl.A_csr.crow_indices()
        vals_hat = base_lvl.A_csr.values().abs()  # уже отмасштабированные значения
        row_counts = crow[1:] - crow[:-1]
        row_idx = torch.repeat_interleave(torch.arange(crow.numel()-1, device=self.device), row_counts)
        row_abs_sum_hat = torch.zeros_like(diag_final)
        row_abs_sum_hat.index_add_(0, row_idx, vals_hat)

        # Относительный порог изоляции на L0: 1e-6 * median(Σ|A_ij|), с минимумом 1e-30
        med0 = row_abs_sum_hat.median()
        iso_thr0 = torch.clamp(1e-6 * med0, min=torch.tensor(1e-30, device=med0.device))
        iso_mask0 = row_abs_sum_hat < iso_thr0
        safe_sum0 = row_abs_sum_hat.clone(); safe_sum0[iso_mask0] = 1.0
        base_lvl.inv_l1 = (1.0 / safe_sum0).clamp_max(1.0)
        base_lvl.inv_l1[iso_mask0] = 0.0
        base_lvl.inv_l1[self.anchor_fine] = 0.0
        # веса для рестрикции на L0 после эквилибрирования
        base_lvl.row_abs_sum = row_abs_sum_hat

        # --- Гибридная диагональ релаксации (после эквилибрирования L0) ---
        # Если |A_ii| доминирует: inv = 1/|A_ii|, иначе 1/Σ|A_ij|
        off_sum0 = (row_abs_sum_hat - base_lvl.diag).clamp_min(0.0)
        use_diag0 = base_lvl.diag >= 0.2 * off_sum0
        invD0 = torch.empty_like(base_lvl.diag)
        invD0[use_diag0]  = 1.0 / base_lvl.diag[use_diag0].clamp_min(1e-30)
        invD0[~use_diag0] = 1.0 / row_abs_sum_hat[~use_diag0].clamp_min(1e-30)
        invD0[iso_mask0]  = 0.0
        invD0[self.anchor_fine] = 0.0
        base_lvl.inv_relax = invD0.clamp_max(4.0)

        if os.environ.get("OIL_DEBUG", "0") == "1":
            print(
                f"[ISO L0] isolated rows (<{iso_thr0.item():.3e}): {iso_mask0.sum().item()}/{len(iso_mask0)}"
            )
            print(
                f"[L0] row_abs_sum_hat: min={row_abs_sum_hat.min().item():.3e}, "
                f"median={row_abs_sum_hat.median().item():.3e}, max={row_abs_sum_hat.max().item():.3e}"
            )
            print(
                f"[L0] inv_l1 stats: min={base_lvl.inv_l1.min().item():.3e}, "
                f"median={base_lvl.inv_l1.median().item():.3e}, max={base_lvl.inv_l1.max().item():.3e}"
            )

        if self.debug:
            print(
                f"[GeoSolverV2] inv_l1 stats after scaling: min={base_lvl.inv_l1.min().item():.3e}, "
                f"median={base_lvl.inv_l1.median().item():.3e}, max={base_lvl.inv_l1.max().item():.3e}"
            )

        # ---------------- Диагностика эффективности AMG (после построения всех уровней) ----------------

        # ------------------------------------------------------------------
        # Итоговый симметричный масштаб S_total = S2 · S1  (элемент-wise).
        # Он удовлетворяет A_hat = S_total · A_phys · S_total, diag(A_hat) ≈ 1.
        # Для согласованного решения системы A δ = rhs нужно масштабировать
        # RHS и восстанавливать δ именно через S_total, а НЕ только через
        # первый шаг scale_full.  Ошибка здесь приводила к гигантским δ и
        # нестабильности Geo-AMG.
        # ------------------------------------------------------------------
        S_total = scale_sqrt * second_scale  # diag-vector
        self.S_total = S_total.to(self.device, torch.float64)
        self.Dinv = self.S_total.clone()        # масштаб в hat-пространство
        self.D    = (1.0 / self.S_total).clone()  # обратно в физическое
        # Сохраним базовые масштабы на уровне L0 для межуровневых конверсий
        base_lvl.Dinv = self.Dinv
        base_lvl.D    = self.D


        # --- диагностика после одного шага ---
        if self.debug and (torch.isnan(vals).any() or torch.isinf(vals).any()):
            nan_cnt = torch.isnan(vals).sum().item()
            inf_cnt = torch.isinf(vals).sum().item()
            print(f"[GeoSolverV2] ВНИМАНИЕ: после эквилибрирования NaN={nan_cnt}, Inf={inf_cnt}")
        if self.debug:
            print(f"[GeoSolverV2] scale_full min={scale_full.min().item():.3e}, max={scale_full.max().item():.3e}")
        if self.debug:
            print(f"[GeoSolverV2] scale_sqrt min={scale_sqrt.min().item():.3e}, max={scale_sqrt.max().item():.3e}")
            print(f"[GeoSolverV2] Â values: min={vals.min().item():.3e}, max={vals.max().item():.3e}")
        if self.debug:
            print(f"[DBG] ||A_hat||_1 = {torch.norm(vals,1).item():.3e}")
        if self.debug:
            print(f"[GeoSolverV2] diagÂ (sqrt(diag_orig)) min={diag_final.min().item():.3e}, max={diag_final.max().item():.3e}")

        # Итоговый масштаб RHS: именно S_total, а не scale_full
        if self.debug:
            print(f"[GeoSolverV2] Dinv min={self.Dinv.min().item():.3e}, max={self.Dinv.max().item():.3e}")

        def _equilibrate_level(lvl: GeoLevel):  # noqa: D401
            """Нормирует матрицу уровня: A ← S A S, diag(A)=1, возвращает S."""
            # --- Первый симметричный шаг: S1 = diag^{-1/4} -----------------
            diag = lvl.diag
            S_full = 1.0 / torch.sqrt(diag.clamp_min(1e-20))
            S1 = torch.sqrt(S_full)  # diag^{-1/4}

            crow = lvl.A_csr.crow_indices()
            col = lvl.A_csr.col_indices()
            vals = lvl.A_csr.values()
            row_counts = crow[1:] - crow[:-1]
            row_idx = torch.repeat_interleave(torch.arange(crow.numel()-1, device=self.device), row_counts)
            vals.mul_(S1[row_idx] * S1[col])  # A1 = S1 A S1

            # --- Второй симметричный шаг: S2 = diag(A1)^{-1/2} -------------
            # diag_idx = crow[1:] - 1
            diag_idx = _csr_diag_pos(crow, col)


            diag_A1 = vals[diag_idx].abs().clone()
            S2 = 1.0 / torch.sqrt(diag_A1.clamp_min(1e-20))  # diag^{-1/2}
            vals.mul_(S2[row_idx] * S2[col])                 # A2 = S2 A1 S2

            diag_final = vals[diag_idx].abs().clone()
            lvl.diag = diag_final

            # --- DEBUG: статистики после полного эквилибрирования ---------
            if os.environ.get("OIL_DEBUG", "0") == "1":
                print(
                    f"[EQ L{len(self.levels)}] diag stats: min={diag_final.min().item():.3e}, "
                    f"median={diag_final.median().item():.3e}, max={diag_final.max().item():.3e}"
                )

            return S1 * S2  # полный масштаб уровня

        # ---------------- Hierarchy of coarser grids -----------------
        # Нормируем базовый уровень (уже сделано выше вручную)
        self.levels: List[GeoLevel] = [base_lvl]

        max_lvls = max_levels or 99
        MIN_N = 8
        for lev in range(1, max_lvls):
            fine_lvl = self.levels[-1]
            nz_f, ny_f, nx_f = fine_lvl.kx.shape

            # 0) пересчёт row_abs_sum для ТЕКУЩЕГО fine уровня (после всех скейлов)
            crow_f = fine_lvl.A_csr.crow_indices(); val_f = fine_lvl.A_csr.values().abs()
            rc_f = crow_f[1:] - crow_f[:-1]
            ridx_f = torch.repeat_interleave(torch.arange(rc_f.numel(), device=self.device), rc_f)
            row_abs_f = torch.zeros(rc_f.numel(), device=self.device, dtype=torch.float64)
            row_abs_f.index_add_(0, ridx_f, val_f)
            fine_lvl.row_abs_sum = row_abs_f

            # Анизотропно-осознанный выбор оси для semi-coarsening
            stride_x, stride_y, stride_z = 1, nx_f, nx_f * ny_f
            crow_f = fine_lvl.A_csr.crow_indices()
            col_f  = fine_lvl.A_csr.col_indices()
            val_fa = fine_lvl.A_csr.values().abs()
            rc_f   = crow_f[1:] - crow_f[:-1]
            ridx_f = torch.repeat_interleave(torch.arange(rc_f.numel(), device=self.device), rc_f)
            diff   = (col_f - ridx_f).abs()
            sum_x  = val_fa[(diff == stride_x)].sum()
            sum_y  = val_fa[(diff == stride_y)].sum()
            sum_z  = val_fa[(diff == stride_z)].sum()
            tot    = row_abs_f.sum().clamp_min(1e-30)
            rx, ry, rz = float((sum_x/tot).item()), float((sum_y/tot).item()), float((sum_z/tot).item())
            axis = max((rx,'x'), (ry,'y'), (rz,'z'))[1]
            thr  = 0.55
            use_semi = self.semicoarsen and ((axis=='z' and rz>thr and nz_f>=3) or
                                             (axis=='y' and ry>thr and ny_f>=3) or
                                             (axis=='x' and rx>thr and nx_f>=3))
            if use_semi:
                coarsen = (1, 2, 2) if axis == 'z' else ((2, 1, 2) if axis == 'y' else (2, 2, 1))
            else:
                coarsen = (2, 2, 2)
            # line-GS пока только по z
            fine_lvl.use_line = use_semi and (axis == 'z')
            if os.environ.get('OIL_DEBUG','0') == '1':
                print(f"[ANISO] rx={rx:.2f} ry={ry:.2f} rz={rz:.2f} → coarsen={coarsen}")

            # 1) строим маппинг (с учётом выбранного coarsen)
            P, parent_idx, n_c, child_cnt, shape_c = build_P_csr((nz_f, ny_f, nx_f), self.device, coarsen=coarsen)

            # --- стоп-условия ---
            # плохо коарснимся: размер почти не уменьшился
            if n_c >= fine_lvl.n_cells or n_c <= MIN_N:
                break
            # форма уже 1×1×1 или одинаковая с прошлой
            if shape_c == (1, 1, 1):
                break
            # нет детей у какого-то coarse-узла (не должно быть, но пусть)
            if (child_cnt == 0).any():
                break

            # если прошли проверки — строим R (weighted/uniform) и сохраняем операторы
            # Энергетические веса: Σ_j |A_ij| (если есть), иначе phys‑diag на L0
            weights_row = getattr(fine_lvl, 'row_abs_sum', None)
            if weights_row is None:
                weights_row = getattr(fine_lvl, 'diag_phys', None)
            # gamma берём из self.restrict_gamma (можно задавать через env GEO_R_GAMMA)
            R_w = build_R_csr(P, child_cnt, weights=weights_row, mode='weighted', gamma=getattr(self, 'restrict_gamma', 1.5))
            R_u = build_R_csr(P, child_cnt, weights=None,       mode='uniform')
            fine_lvl.P = P
            fine_lvl.R_weighted = R_w
            fine_lvl.R_uniform  = R_u
            fine_lvl.R = R_w         

            # 2) RAP
            rap_mode = 'weighted' if (weights_row is not None) else 'uniform'
            if rap_mode == 'weighted' and (weights_row is None):
                if os.environ.get('OIL_DEBUG','0') == '1':
                    print("[RAP] weights_row=None → fallback to uniform")
                rap_mode = 'uniform'
            A_csr = rap_pc_const_gpu(fine_lvl.A_csr, parent_idx, n_c, child_cnt,
                                     weights_row=weights_row, mode=rap_mode)
            # Мягкая починка диагонали на coarse после RAP
            A_csr = self._ensure_csr_diagonal_(A_csr)

            # ---- CHECK RAP (без densify на больших N) ---------------------------------
            if self.debug and self.rap_check_debug:
                try:
                    n_fine = fine_lvl.A_csr.size(0)
                    if n_fine <= self.rap_max_check_n:
                        # Без денсификации: выборочная RAP‑проверка по нескольким столбцам
                        self._rap_check_sample(fine_lvl, A_csr, k=5)
                    else:
                        if self.debug:
                            print(f"[RAPCHK L{len(self.levels)}] skip sample check (n={n_fine} > {self.rap_max_check_n})")
                except Exception as _e:
                    print(f"[RAPCHK] skip due to error: {_e}")


            if A_csr._nnz() == 0:
                break
            # 3) diag_c и inv_l1
            crow = A_csr.crow_indices()
            col  = A_csr.col_indices()
            vals = A_csr.values()

            # diag_idx = crow[1:] - 1
            diag_idx = _csr_diag_pos(crow, col)


            diag_c = vals[diag_idx].abs().clone()

            # L1 surrogate
            row_counts = crow[1:] - crow[:-1]
            row_idx = torch.repeat_interleave(torch.arange(n_c, device=self.device), row_counts)
            row_abs_sum = torch.zeros(n_c, device=self.device, dtype=torch.float64)
            row_abs_sum.index_add_(0, row_idx, vals.abs())
            iso_mask = row_abs_sum < 1e-8
            safe = row_abs_sum.clone(); safe[iso_mask] = 1.0
            inv_l1 = (1.0 / safe)
            inv_l1[iso_mask] = 0.0

            # 4) создаём GeoLevel из CSR (используем фактический coarsen)
            cz, cy, cx = coarsen
            nz_c, ny_c, nx_c = shape_c
            hx_c, hy_c, hz_c = fine_lvl.hx * cx, fine_lvl.hy * cy, fine_lvl.hz * cz

            lvl = build_level_from_csr(
                A_csr, diag_c, inv_l1,
                shape_c,
                hx_c, hy_c, hz_c,
                device=self.device
            )

            # [NEW] сначала левое эквилибрирование
            lvl.A_csr, w_rows_c = _row_equilibrate_csr(lvl.A_csr, lvl=lvl)
            lvl.diag = lvl.diag * w_rows_c
            # нормализуем и клампим W_rows на уровне
            med_c = torch.median(w_rows_c)
            if torch.isfinite(med_c) and med_c > 0:
                w_rows_c = w_rows_c / med_c
            lvl.W_rows = torch.clamp(w_rows_c, 1e-6, 1e6)
            # потом твой симметричный шаг
            S_c = _equilibrate_level(lvl)
            # Сохраним масштабы уровня после симм-эквилибрирования
            lvl.Dinv = S_c
            lvl.D    = 1.0 / S_c

            if self.debug:
                crow = lvl.A_csr.crow_indices(); val = lvl.A_csr.values()
                diag = val[crow[1:]-1]
                print(f"[LVL {len(self.levels)}] n={lvl.n_cells} nnz={lvl.A_csr._nnz()} "
                    f"diag[min,med,max]=({diag.min():.3e},{diag.median():.3e},{diag.max():.3e}) "
                    f"A[min,max]=({val.min():.3e},{val.max():.3e})")


            # пересчёт inv_l1 после эквилибрирования (относительный порог)
            crow = lvl.A_csr.crow_indices(); col = lvl.A_csr.col_indices(); vals = lvl.A_csr.values().abs()
            rc = crow[1:] - crow[:-1]
            ridx = torch.repeat_interleave(torch.arange(lvl.n_cells, device=self.device), rc)
            row_abs = torch.zeros(lvl.n_cells, device=self.device, dtype=torch.float64)
            row_abs.index_add_(0, ridx, vals)
            lvl.row_abs_sum = row_abs
            # Относительный порог изоляции: 1e-6 * median(Σ|A_ij|), min=1e-30
            med_c = row_abs.median()
            thr_c = torch.clamp(1e-6 * med_c, min=torch.tensor(1e-30, device=row_abs.device))
            iso = row_abs < thr_c
            safe = row_abs.clone(); safe[iso] = 1.0
            lvl.inv_l1 = 1.0 / safe
            lvl.inv_l1[iso] = 0.0

            # --- inv_relax для уровня lvl (после эквилибрирования) ---
            off_sum = (row_abs - lvl.diag).clamp_min(0.0)
            use_diag = lvl.diag >= 0.2 * off_sum
            invD = torch.empty_like(lvl.diag)
            invD[use_diag]  = 1.0 / lvl.diag[use_diag].clamp_min(1e-30)
            invD[~use_diag] = 1.0 / row_abs[~use_diag].clamp_min(1e-30)
            invD[iso] = 0.0
            lvl.inv_relax = invD.clamp_max(4.0)

            # 5) pin null mode (тот же индекс в coarse, соответствующий anchor-ячейке родителя)
            #    Берём coarse-ячею, в которую попал fine anchor:
            fine_anchor = int(getattr(fine_lvl, "anchor", self.anchor_fine))
            # страхуемся на случай несоответствия размеров (должно быть редко)
            if fine_anchor >= parent_idx.numel():
                fine_anchor = parent_idx.numel() - 1

            anchor_c = int(parent_idx[fine_anchor].item())
            lvl.anchor = anchor_c

            def _pin_rowcol_local(A_csr, idx):
                crow = A_csr.crow_indices(); col = A_csr.col_indices(); val = A_csr.values()
                s, e = int(crow[idx].item()), int(crow[idx+1].item())
                val[s:e] = 0.0
                mask = (col == idx)
                val[mask] = 0.0
                rel = torch.nonzero(col[s:e] == idx, as_tuple=False)
                if rel.numel():
                    val[s + int(rel[0])] = 1.0
                else:
                    if e == s:
                        raise RuntimeError("Empty CSR row — cannot pin")
                    val[s] = 1.0
                    col[s] = int(idx)

            _pin_rowcol_local(A_csr, anchor_c)
            # diag_c[anchor_c] = 1.0
            # inv_l1[anchor_c] = 0.0
            # lvl.A_csr = A_csr
            # lvl.diag = diag_c
            # lvl.inv_l1 = inv_l1
            lvl.diag[anchor_c] = 1.0
            lvl.inv_l1[anchor_c] = 0.0
            if hasattr(lvl, 'inv_relax'):
                lvl.inv_relax[anchor_c] = 0.0
            lvl.anchor = anchor_c  # можно сохранить

            if self.debug:
                print(f"[GeoSolverV2] built level {len(self.levels)} (Galerkin): n={lvl.n_cells}")
            self.levels.append(lvl)

        if self.debug:
            vtest = torch.randn(self.levels[0].n_cells, dtype=torch.float64, device=self.device)
            Av = self._apply_A(0, vtest)
            print(f"[SPD?] <v,Av>={torch.dot(vtest, Av).item():.3e}")


    # ------------------------------------------------------------------
    def _apply_A(self, lvl_idx: int, x: torch.Tensor) -> torch.Tensor:
        lvl = self.levels[lvl_idx]
        Ax = torch.sparse.mm(lvl.A_csr, x.view(-1, 1)).squeeze(1)
        return Ax

    def _vec2vol(self, lvl_idx: int, v: torch.Tensor) -> torch.Tensor:
        nz, ny, nx = self.levels[lvl_idx].kx.shape
        return v.view(nz, ny, nx)

    def _vol2vec(self, vol: torch.Tensor) -> torch.Tensor:
        return vol.reshape(-1)

    def _to_hat(self, v):  return self.Dinv * v
    def _to_phys(self, v): return self.D * v

    def _left_scale_rhs(self, v: torch.Tensor) -> torch.Tensor:
        """Левое строковое эквилибрирование RHS: b̂ ← W_rows · b̂.
        Должно использоваться во всех входах в AMG, так как Â = S · (W_rows · A_phys) · S.
        """
        return self.W_rows * v

    def _apply_anchor(self, v: torch.Tensor):
        # держим нулевой мод на самом тонком уровне
        v[self.anchor_fine] = 0.0
        return v

    def _rap_check_sample(self, fine_lvl, Ac: torch.Tensor, k: int = 5, seed: int | None = None):
        """Выборочная RAP‑проверка без денсификации: сравниваем несколько столбцов.
        Печатаем относительную L1‑ошибку для k случайных единичных столбцов.
        """
        R, P = fine_lvl.R, fine_lvl.P
        Af = fine_lvl.A_csr
        n_c = int(Ac.size(0))
        if n_c == 0:
            return
        if seed is not None:
            torch.manual_seed(int(seed))
        idx = torch.randint(0, n_c, (min(k, n_c),), device=Ac.device)
        errs = []
        for j in idx:
            ej = torch.zeros(n_c, dtype=Ac.dtype, device=Ac.device)
            ej[j] = 1.0
            # PyTorch не поддерживает sparse@sparse, поэтому считаем так: ref = R*(Af*(P*e_j))
            u = torch.sparse.mm(P, ej[:, None])           # (n_f,1) dense
            v = torch.sparse.mm(Af, u)                    # (n_f,1) dense
            ref = torch.sparse.mm(R, v).squeeze(1)        # (n_c,)
            got = torch.sparse.mm(Ac, ej[:, None]).squeeze(1)
            rel = (ref - got).abs().sum() / (ref.abs().sum() + 1e-30)
            errs.append(float(rel.item()))
        if self.debug and errs:
            med = sorted(errs)[len(errs)//2] if len(errs) % 2 == 1 else (sorted(errs)[len(errs)//2-1] + sorted(errs)[len(errs)//2]) / 2.0
            mx = max(errs)
            print(f"[RAPCHK] sample rel_err median={med:.3e}, max={mx:.3e}")

    def _restrict_vec(self, lvl_idx: int, r_f: torch.Tensor) -> torch.Tensor:
        """r_c = R * r_f"""
        R = self.levels[lvl_idx].R
        return torch.sparse.mm(R, r_f.view(-1, 1)).squeeze(1)

    def _prolong_vec(self, lvl_idx: int, e_c: torch.Tensor) -> torch.Tensor:
        """e_f = P * e_c"""
        P = self.levels[lvl_idx].P
        return torch.sparse.mm(P, e_c.view(-1, 1)).squeeze(1)

    def _prolong_vec_smoothed(self, lvl_idx: int, e_c_hat: torch.Tensor) -> torch.Tensor:
        """
        Smoothed Aggregation пролонгация на лету:
        1) e_f0_hat = D_f^{-1} · P · (D_c · e_c_hat)
        2) e_f_hat  = e_f0_hat − ω · invD_f · (A_f · e_f0_hat) [× sweeps]
        invD_f берём из inv_relax (L1-Jacobi surrogate).
        """
        lvl_f = self.levels[lvl_idx]
        lvl_c = self.levels[lvl_idx + 1]
        # Базовая P-инъекция с корректными межуровневыми масштабами
        e_c_phys = lvl_c.D * e_c_hat
        e_f_phys0 = self._prolong_vec(lvl_idx, e_c_phys)
        e_f_hat = lvl_f.Dinv * e_f_phys0
        invD = getattr(lvl_f, 'inv_relax', lvl_f.inv_l1)
        for _ in range(self.prolong_sweeps):
            Ae = self._apply_A(lvl_idx, e_f_hat)
            e_f_hat = e_f_hat - self.prolong_omega * (invD * Ae)
        # Нейтрализуем якорь для нулевого модуса
        if hasattr(lvl_f, 'anchor'):
            try:
                e_f_hat[lvl_f.anchor] = 0.0
            except Exception:
                pass
        return e_f_hat

    @staticmethod
    def _ensure_csr_diagonal_(A: torch.Tensor) -> torch.Tensor:
        """Гарантирует наличие диагонали в каждой строке CSR. Мягко чинит при отсутствии.
        Меняет только col/val, структуру indptr не трогает.
        """
        crow = A.crow_indices(); col = A.col_indices(); val = A.values()
        n = int(crow.numel() - 1)
        row_idx = torch.repeat_interleave(torch.arange(n, device=col.device), crow[1:] - crow[:-1])
        pos_all = torch.nonzero(col == row_idx, as_tuple=False).squeeze(1)
        has_diag = torch.zeros(n, device=col.device, dtype=torch.bool)
        if pos_all.numel() > 0:
            has_diag[row_idx[pos_all]] = True
        miss = torch.nonzero(~has_diag, as_tuple=False).squeeze(1)
        for i in miss.tolist():
            s = int(crow[i].item()); e = int(crow[i+1].item())
            if e == s:
                continue  # пустая строка — оставим как есть
            row_abs = val[s:e].abs()
            jrel = int(torch.argmin(row_abs).item())
            j = s + jrel
            col[j] = i
            val[j] = torch.clamp(row_abs.sum(), min=torch.tensor(1e-30, device=val.device, dtype=val.dtype))
        return torch.sparse_csr_tensor(crow, col, val, size=A.size(), device=A.device, dtype=A.dtype)

    def _check_RAP(self, fine_lvl, Ac, parent_idx, child_cnt):
        # восстановим Ac_ref = R A_f P и сравним
        # Лёгкая RAP‑проверка без денсификации
        self._rap_check_sample(fine_lvl, Ac, k=5)


    def _jacobi(self, lvl_idx: int, x: torch.Tensor, b: torch.Tensor, iters: int = 2) -> torch.Tensor:
        lvl = self.levels[lvl_idx]
        # -------- L1-Jacobi диагональный суррогат --------
        # --- L1-Jacobi: denom = Σ_j |A_ij| -------------------------------
        inv_diag = getattr(lvl, 'inv_relax', lvl.inv_l1)
        omega = self.omega_fine if lvl_idx == 0 else self.omega
        if self.delta_clip_factor is not None:
            init_clip = self.delta_clip_factor * (b.abs().max().item() + 1e-12)
        else:
            init_clip = None
        for k in range(iters):
            r = b - self._apply_A(lvl_idx, x)
            x = x + omega * inv_diag * r
            if self.debug and lvl_idx == 0 and k < 3:
                print(
                    f"[DBG-JAC] k={k} |r|₂={r.norm().item():.3e} |r|∞={r.abs().max().item():.3e} "
                    f"|invD|∞={inv_diag.max().item():.3e} |invD*r|∞={(inv_diag*r).abs().max().item():.3e} ω={omega:.2f}"
                )

            # --- динамический clip только на самом тонком уровне ---
            # пропускаем клип первые 2 итерации на L0, чтобы не душить апдейты
            if init_clip is not None and lvl_idx == 0 and k >= 2:
                dyn_clip = self.clip_kappa * (x.abs().max().item() + 1e-12)
                clip_val = max(init_clip, dyn_clip)
                x = torch.clamp(x, -clip_val, clip_val)

            # --- диагностика NaN/Inf ---
            if not torch.isfinite(x).all():
                n_bad = (~torch.isfinite(x)).sum().item()
                print(f"[GeoSolverV2] NaN/Inf обнаружены на уровне {lvl_idx} после Jacobi: count={n_bad}")
                finite_mask = torch.isfinite(x)
                print(f"  finite x: min={x[finite_mask].min().item():.3e}, max={x[finite_mask].max().item():.3e}")
                break
        return x

    # ------------------ Chebyshev tail -----------------------------
    def _chebyshev(self, lvl_idx: int, x: torch.Tensor, b: torch.Tensor,
                    iters: int = 3) -> torch.Tensor:
        lvl = self.levels[lvl_idx]
        # Границы спектра по Гершгорину (на эквилиброванных уровнях даёт хорошую верхнюю оценку)
        row_abs = getattr(lvl, 'row_abs_sum', None)
        if row_abs is None:
            crow = lvl.A_csr.crow_indices(); col = lvl.A_csr.col_indices(); val = lvl.A_csr.values().abs()
            rc = crow[1:] - crow[:-1]
            ridx = torch.repeat_interleave(torch.arange(rc.numel(), device=val.device), rc)
            row_abs = torch.zeros(rc.numel(), device=val.device, dtype=val.dtype)
            row_abs.index_add_(0, ridx, val)
        lam_max = float(row_abs.max().item() * 1.10)
        lam_min = lam_max / max(10.0, getattr(self, 'cheby_kappa', 80.0))
        theta = (lam_max + lam_min) / 2.0
        delta = (lam_max - lam_min) / 2.0

        r = b - self._apply_A(lvl_idx, x)
        p = r / theta
        x = x + p
        alpha_prev = 1.0 / theta
        for _ in range(iters - 1):
            r = b - self._apply_A(lvl_idx, x)
            beta = (delta * alpha_prev / 2.0) ** 2
            alpha = 1.0 / (theta - beta / alpha_prev)
            p = alpha * r + beta * p
            x = x + p
            alpha_prev = alpha
        return x

    def _rb_gs(self, lvl_idx: int, x: torch.Tensor, b: torch.Tensor, iters: int = 2) -> torch.Tensor:
        """Red-Black Gauss–Seidel smoother (GPU-friendly)."""
        lvl = self.levels[lvl_idx]
        inv_diag = getattr(lvl, 'inv_relax', lvl.inv_l1)
        omega = self.omega_fine if lvl_idx == 0 else self.omega
        if lvl_idx == 0:
            omega = self.omega_fine = float(torch.clamp(torch.tensor(omega), 0.10, 0.95))

        red_mask, black_mask = lvl.is_red, lvl.is_black

        init_clip = None
        if self.delta_clip_factor is not None:
            init_clip = self.delta_clip_factor * (b.abs().max() + 1e-12)

        clip_hits = 0

        for k in range(iters):
            # r_before
            r_before = b - self._apply_A(lvl_idx, x)

            # red sweep
            x[red_mask] += omega * inv_diag[red_mask] * r_before[red_mask]

            # recompute residual once
            r_mid = b - self._apply_A(lvl_idx, x)

            # black sweep
            x[black_mask] += omega * inv_diag[black_mask] * r_mid[black_mask]

            # r_after
            r_after = b - self._apply_A(lvl_idx, x)

            mu = r_after.norm() / (r_before.norm() + 1e-30)

            # --- адаптация шага только на самом тонком уровне ------------
            if lvl_idx == 0:
                # жёсткие пределы, чтобы ω не «умирал» и не зашкаливал
                MIN_OMEGA, MAX_OMEGA = 0.10, 0.95

                # если сглаживание почти не работает (μ ≳ 0.9) — УВЕЛИЧИВАЕМ ω
                if mu > 0.9:
                    self.omega_fine = min(self.omega_fine * 1.25 + 1e-3, MAX_OMEGA)
                # если и так хорошо сглаживает (μ ≲ 0.5) — слегка ПРИЖИМАЕМ ω (устойчивость)
                elif mu < 0.5:
                    self.omega_fine = max(self.omega_fine * 0.9, MIN_OMEGA)

                # используем обновлённый шаг прямо в текущей итерации
                omega = self.omega_fine

        
            if self.debug and lvl_idx == 0:
                # ограничить синхронизации: .item() только тут
                print(f"[SMOOTH L{lvl_idx}] k={k} μ={mu.item():.3e} "
                    f"||r||₂_before={r_before.norm().item():.3e} after={r_after.norm().item():.3e}")

            # dynamic clip only on finest — пропустим первые 2 итерации
            if init_clip is not None and lvl_idx == 0 and k >= 2:
                dyn_clip = self.clip_kappa * (x.abs().max() + 1e-12)
                clip_val = torch.maximum(init_clip, dyn_clip)
                # посчитаем до clamp:
                clipped_mask = (x.abs() >= clip_val)
                clip_hits += int(clipped_mask.sum().item())
                torch.clamp_(x, -clip_val, clip_val)

            if self.debug and lvl_idx == 0 and k < 3:
                inc_mag = (omega * inv_diag * r_mid).abs().max().item()
                print(f"[DBG-RBGS] k={k} |ω|={omega} |r|₂={r_mid.norm().item():.3e} |r|∞={r_mid.abs().max().item():.3e} "
                    f"|invD|∞={inv_diag.max().item():.3e} |ω*invD*r|∞={inc_mag:.3e} "
                    f"clip_init={init_clip.item() if init_clip is not None else 0:.3e}")

        if self.debug and lvl_idx == 0:
            print(f"[SMOOTH L{lvl_idx}] total_clip_hits={clip_hits}")
        
        if lvl_idx == 0:
            self._apply_anchor(x)


        return x


    def _line_gs_z(self, lvl_idx: int, x: torch.Tensor, b: torch.Tensor, iters: int = 1) -> torch.Tensor:
        """Вертикальный (по z) RB Line–Gauss–Seidel с точным решением 1-D три-диаг. систем.

        Для каждой колонки (i,j) решаем A_z δ = r_col (Thomas) и обновляем x.
        Пока CPU-вариант: столбцы небольшие, поэтому копируем данные на CPU.
        """
        lvl = self.levels[lvl_idx]
        nz, ny, nx = lvl.kx.shape
        stride_z = nx * ny
        total = lvl.n_cells

        omega = 1.0  # для точного решения шаг = 1

        for _ in range(iters):
            r = (b - self._apply_A(lvl_idx, x)).cpu().numpy()
            a_dn = lvl.a_dn.cpu().numpy()
            a_up = lvl.a_up.cpu().numpy()
            diag = lvl.diag.cpu().numpy()

            x_cpu = x.cpu().numpy()

            # две покраски по (i+j) % 2
            for color in (0, 1):
                for j in range(ny):
                    for i in range(nx):
                        if ((i + j) & 1) != color:
                            continue
                        base = j * nx + i  # k=0 index in linear array
                        if base >= total:
                            continue
                        # извлекаем колонку
                        idxs = base + np.arange(nz) * stride_z
                        a_d = diag[idxs]
                        a_u = a_up[idxs[:-1]]  # len nz-1, value connects k to k+1
                        a_l = a_dn[idxs[1:]]    # len nz-1
                        rhs_col = r[idxs]

                        # Thomas algorithm
                        c_prime = np.zeros_like(a_u)
                        d_prime = np.zeros_like(rhs_col)
                        c_prime[0] = a_u[0] / a_d[0]
                        d_prime[0] = rhs_col[0] / a_d[0]
                        for k in range(1, nz-1):
                            denom = a_d[k] - a_l[k-1] * c_prime[k-1]
                            denom = np.sign(denom) * max(abs(denom), 1e-30)
                            c_prime[k] = a_u[k] / denom
                            d_prime[k] = (rhs_col[k] - a_l[k-1] * d_prime[k-1]) / denom
                        denom_last = a_d[-1] - a_l[-1] * c_prime[-1]
                        denom_last = np.sign(denom_last) * max(abs(denom_last), 1e-30)
                        d_prime_last = (rhs_col[-1] - a_l[-1] * d_prime[-1]) / denom_last

                        # back substitution
                        sol = np.zeros_like(rhs_col)
                        sol[-1] = d_prime_last
                        for k in range(nz-2, -1, -1):
                            sol[k] = d_prime[k] - c_prime[k] * sol[k+1]

                        x_cpu[idxs] += omega * sol

            x = torch.from_numpy(x_cpu).to(x.device)
        return x

    # Helpers -----------------------------------------------------------------
    def _restrict3d(self, vol3d: torch.Tensor) -> torch.Tensor:
        z, y, x = vol3d.shape
        z_c, y_c, x_c = (z + 1) // 2, (y + 1) // 2, (x + 1) // 2

        # собираем сумму детей
        acc = vol3d[0::2, 0::2, 0::2].clone()
        cnt = torch.ones_like(acc)

        if x > 1:
            acc += vol3d[0::2, 0::2, 1::2]; cnt += 1
        if y > 1:
            acc += vol3d[0::2, 1::2, 0::2]; cnt += 1
        if y > 1 and x > 1:
            acc += vol3d[0::2, 1::2, 1::2]; cnt += 1
        if z > 1:
            acc += vol3d[1::2, 0::2, 0::2]; cnt += 1
        if z > 1 and x > 1:
            acc += vol3d[1::2, 0::2, 1::2]; cnt += 1
        if z > 1 and y > 1:
            acc += vol3d[1::2, 1::2, 0::2]; cnt += 1
        if z > 1 and y > 1 and x > 1:
            acc += vol3d[1::2, 1::2, 1::2]; cnt += 1

        return acc / cnt


    def _prolong3d(self, coarse3d: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        """Trilinear prolongation 2×2×2 <- coarse.
        """
        if os.environ.get("OIL_DEBUG", "0") == "1":
            n_c = coarse3d.norm().item()
        fine = F.interpolate(coarse3d[None, None, ...], size=target_shape, mode="trilinear", align_corners=False)[0, 0]

        fine_norm = fine.norm().item()
        if os.environ.get("OIL_DEBUG", "0") == "1":
            print(f"[PR] prolong norm ratio={fine_norm/(n_c+1e-30):.3e} (fine/coarse)")
        return fine

    def _v_cycle(self, lvl_idx: int, x_vec: torch.Tensor, b_vec: torch.Tensor) -> torch.Tensor:
        """Recursive V-/W-cycle in 3-D form, input/output flattened."""
        lvl = self.levels[lvl_idx]
        debug_any = os.environ.get("OIL_DEBUG", "0") == "1"
        debug_top = debug_any and lvl_idx == 0

        # согласуем RHS с левым скейлированием текущего уровня (кроме L0, там уже сделано на входе solve)
        if lvl_idx > 0:
            b_vec = lvl.W_rows * b_vec

        # initial residual
        r_in = b_vec - self._apply_A(lvl_idx, x_vec)

        if self.debug:
            print(f"[VC L{lvl_idx}] START  {_vstats('r_in', r_in)}   ||x||2={x_vec.norm():.3e}")

        if debug_any:
            print(f"[VC] L{lvl_idx} start ‖x‖={x_vec.norm():.3e} ‖r‖₂={r_in.norm():.3e}")

        if debug_top:
            _stats(f"V{lvl_idx}.b_vec", b_vec)

        # Coarsest grid
        if lvl_idx == len(self.levels) - 1 or lvl.n_cells <= 128:
            anc = lvl.anchor
            A = lvl.A_csr.to_dense()
            A[anc, :] = 0; A[:, anc] = 0; A[anc, anc] = 1.0
            # RHS уже согласован ранее
            b_fix = b_vec.clone()
            b_fix[anc] = 0.0
            x_sol = torch.linalg.solve(A, b_fix.view(-1, 1)).squeeze(1)
            return x_sol
        elif lvl.n_cells <= 5000:
            # Усилим coarse‑решатель: более строгий критерий 1e-3
            x_sol = torch.zeros_like(b_vec)
            invD = getattr(lvl, 'inv_relax', lvl.inv_l1)
            r = b_vec - self._apply_A(lvl_idx, x_sol)
            r0 = r.norm() + 1e-30
            z = invD * r
            p = z.clone()
            rz_old = torch.dot(r, z)
            for _ in range(100):  # верхняя граница
                Ap = self._apply_A(lvl_idx, p)
                denom = torch.dot(p, Ap).clamp_min(1e-30)
                alpha = rz_old / denom
                x_sol = x_sol + alpha * p
                r = r - alpha * Ap
                if r.norm() <= 1e-3 * r0:
                    break
                z = invD * r
                rz_new = torch.dot(r, z)
                beta = rz_new / rz_old
                p = z + beta * p
                rz_old = rz_new
            return x_sol


        nz, ny, nx = lvl.kx.shape

        # ---------- pre-smooth ----------
        if getattr(lvl, 'use_line', False):
            x_vec = self._line_gs_z(lvl_idx, x_vec, b_vec, iters=max(1, self.pre_smooth // 2))
        else:
            if self.smoother_fine == "chebyshev":
                x_vec = self._chebyshev(lvl_idx, x_vec, b_vec, iters=max(3, self.pre_smooth))
            elif self.smoother_fine == "jacobi":
                x_vec = self._jacobi(lvl_idx, x_vec, b_vec, iters=self.pre_smooth)
            else:  # rbgs
                x_vec = self._rb_gs(lvl_idx, x_vec, b_vec, iters=self.pre_smooth)

        r_pre = b_vec - self._apply_A(lvl_idx, x_vec)
        # убрать компоненту константы (near-nullspace) из RHS уровня
        if getattr(self, 'project_const', True):
            r_pre = r_pre - r_pre.mean()
        mu_pre = r_pre.norm() / (r_in.norm() + 1e-30)
        if debug_top:
            _stats(f"V{lvl_idx}.r_after_pre", r_pre)
        if debug_any:
            print(f"[VC] L{lvl_idx} after pre-smooth ‖x‖={x_vec.norm():.3e} ‖r‖₂={r_pre.norm():.3e}  μ_pre={mu_pre.item():.3e}")

        # ---------- restrict (Galerkin) с межуровневыми масштабами ----------
        # r_c_hat = S_c · R · (D_f · r_f_hat)
        lvl_f = self.levels[lvl_idx]
        lvl_c = self.levels[lvl_idx + 1]
        # если пред-сглаживание слабо — можно использовать uniform рестрикцию
        R_use = getattr(lvl_f, 'R_weighted', None)
        if (mu_pre.item() > 0.85) and hasattr(lvl_f, 'R_uniform'):
            R_use = lvl_f.R_uniform
        if R_use is None:
            R_use = lvl_f.R
        r_c_vec = torch.sparse.mm(R_use, (lvl_f.D * r_pre).view(-1, 1)).squeeze(1)
        r_c_vec = lvl_c.Dinv * r_c_vec
        # обнуляем RHS в якорной ячейке coarse-уровня
        anc_c = lvl_c.anchor
        r_c_vec[anc_c] = 0.0
        x_c_vec = torch.zeros_like(r_c_vec)

        # ---------- recurse ---------- (авто W/V на верхнем уровне по эффективности сглаживания)
        x_c_vec = self._v_cycle(lvl_idx + 1, x_c_vec, r_c_vec)
        if lvl_idx == 0:
            extra = 1 if mu_pre > 0.8 else (0 if mu_pre < 0.5 else 0)
            for _ in range(int(extra)):
                x_c_vec = self._v_cycle(lvl_idx + 1, x_c_vec, r_c_vec)

        # ---------- prolong с межуровневыми масштабами ----------
        # По умолчанию применяем сглаженную пролонгацию (on-the-fly SA)
        if self.smooth_prolong and (lvl_idx < len(self.levels) - 1):
            corr_vec = self._prolong_vec_smoothed(lvl_idx, x_c_vec)
        else:
            corr_phys_c = lvl_c.D * x_c_vec
            corr_phys_f = self._prolong_vec(lvl_idx, corr_phys_c)
            corr_vec = lvl_f.Dinv * corr_phys_f

        # Энергетический шаг: α = (r, e) / (e, A e) — устойчивее при mismatch масштаба
        A_corr = self._apply_A(lvl_idx, corr_vec)
        num = torch.dot(r_pre, corr_vec)
        den = torch.dot(corr_vec, A_corr).clamp_min(1e-30)
        alpha_unclamped = num / den
        alpha = torch.clamp(alpha_unclamped, 0.0, 2.5)

        # Диагностика геометрии шага: α*, cosθ, cap
        if debug_any or debug_top:
            rt = num
            tt = den
            rr = torch.dot(r_pre, r_pre).clamp_min(1e-30)
            alpha_star = alpha_unclamped
            cos_theta = (rt / (rr.sqrt() * tt.sqrt().clamp_min(1e-30))).item()
            hit_cap = bool((alpha - alpha_unclamped).abs() > 1e-12)
            try:
                print(f"[CCorr L{lvl_idx}] alpha*={alpha_star.item():.3e} alpha={alpha.item():.3e} cosθ={cos_theta:.3f} hit_cap={hit_cap}")
            except Exception:
                pass
        x_vec = x_vec + alpha * corr_vec



        # Или оставить оптимальный шаг как у тебя:
        # A_corr = self._apply_A(lvl_idx, corr_vec)
        # num = torch.dot(r_pre, corr_vec)
        # den = torch.dot(corr_vec, A_corr).clamp_min(1e-30)
        # alpha = num / den
        # x_vec = x_vec + alpha * corr_vec




        if lvl_idx == 0:
            self._apply_anchor(x_vec)

        r_corr = b_vec - self._apply_A(lvl_idx, x_vec)
        rho_corr = r_corr.norm() / (r_pre.norm() + 1e-30)
        if debug_top:
            _stats(f"V{lvl_idx}.r_after_corr", r_corr)
        if debug_any:
            print(f"[VC] L{lvl_idx} after coarse corr ‖x‖={x_vec.norm():.3e} ‖r‖₂={r_corr.norm():.3e}  ρ_corr={rho_corr.item():.3e}")

        # Mini-K-cycle: всегда выполняем вторую коррекцию (по невязке)
        if self.kcycle and (lvl_idx < len(self.levels) - 1):
            r_mid = b_vec - self._apply_A(lvl_idx, x_vec)
            if getattr(self, 'project_const', True):
                r_mid = r_mid - r_mid.mean()
            # второй coarse RHS с межуровневыми масштабами
            r_c2 = torch.sparse.mm(lvl_f.R, (lvl_f.D * r_mid).view(-1, 1)).squeeze(1)
            r_c2 = lvl_c.Dinv * r_c2
            r_c2[lvl_c.anchor] = 0.0
            x_c2 = torch.zeros_like(r_c2)
            x_c2 = self._v_cycle(lvl_idx + 1, x_c2, r_c2)
            if self.smooth_prolong:
                corr2 = self._prolong_vec_smoothed(lvl_idx, x_c2)
            else:
                corr2 = lvl_f.Dinv * self._prolong_vec(lvl_idx, (lvl_c.D * x_c2))
            A_corr2 = self._apply_A(lvl_idx, corr2)
            num2 = torch.dot(r_mid, corr2)
            den2 = torch.dot(corr2, A_corr2).clamp_min(1e-30)
            alpha2_unclamped = num2 / den2
            alpha2 = torch.clamp(alpha2_unclamped, 0.0, 2.5)
            # Диагностика второго шага
            if debug_any or debug_top:
                rt2 = num2
                tt2 = den2
                rr2 = torch.dot(r_mid, r_mid).clamp_min(1e-30)
                alpha2_star = alpha2_unclamped
                cos_theta2 = (rt2 / (rr2.sqrt() * tt2.sqrt().clamp_min(1e-30))).item()
                hit_cap2 = bool((alpha2 - alpha2_unclamped).abs() > 1e-12)
                try:
                    print(f"[CCorr2 L{lvl_idx}] alpha*={alpha2_star.item():.3e} alpha={alpha2.item():.3e} cosθ={cos_theta2:.3f} hit_cap={hit_cap2}")
                except Exception:
                    pass
            x_vec = x_vec + alpha2 * corr2

        # Адаптивный дополнительный coarse-проход, если коррекция дважды подряд слабая
        prev_rho = self._rho_hist.get(lvl_idx, None)
        self._rho_hist[lvl_idx] = float(rho_corr)
        if (prev_rho is not None) and (prev_rho > 0.90) and (rho_corr > 0.90) and (lvl_idx < len(self.levels) - 1):
            r_mid2 = b_vec - self._apply_A(lvl_idx, x_vec)
            if getattr(self, 'project_const', True):
                r_mid2 = r_mid2 - r_mid2.mean()
            r_c3 = torch.sparse.mm(lvl_f.R, (lvl_f.D * r_mid2).view(-1, 1)).squeeze(1)
            r_c3 = lvl_c.Dinv * r_c3
            r_c3[lvl_c.anchor] = 0.0
            x_c3 = self._v_cycle(lvl_idx + 1, torch.zeros_like(r_c3), r_c3)
            if self.smooth_prolong:
                corr3 = self._prolong_vec_smoothed(lvl_idx, x_c3)
            else:
                corr3 = lvl_f.Dinv * self._prolong_vec(lvl_idx, (lvl_c.D * x_c3))
            A_corr3 = self._apply_A(lvl_idx, corr3)
            num3 = torch.dot(r_mid2, A_corr3)
            den3 = torch.dot(A_corr3, A_corr3).clamp_min(1e-30)
            alpha3 = torch.clamp(num3 / den3, 0.0, 2.0)
            x_vec = x_vec + alpha3 * corr3

        # ---------- post-smooth ----------
        if getattr(lvl, 'use_line', False):
            x_vec = self._line_gs_z(lvl_idx, x_vec, b_vec, iters=max(1, self.post_smooth // 2))
        else:
            if self.smoother_fine == "chebyshev":
                x_vec = self._chebyshev(lvl_idx, x_vec, b_vec, iters=max(3, self.post_smooth))
            elif self.smoother_fine == "jacobi":
                x_vec = self._jacobi(lvl_idx, x_vec, b_vec, iters=self.post_smooth)
            else:
                x_vec = self._rb_gs(lvl_idx, x_vec, b_vec, iters=self.post_smooth)

        # Дополнительный лёгкий проход на L1/L2 для снятия ВЧ после пролонгации
        if lvl_idx <= 2:
            if getattr(lvl, 'use_line', False):
                x_vec = self._line_gs_z(lvl_idx, x_vec, b_vec, iters=1)
            else:
                if self.smoother_fine == "chebyshev":
                    x_vec = self._chebyshev(lvl_idx, x_vec, b_vec, iters=1)
                elif self.smoother_fine == "jacobi":
                    x_vec = self._jacobi(lvl_idx, x_vec, b_vec, iters=1)
                else:
                    x_vec = self._rb_gs(lvl_idx, x_vec, b_vec, iters=1)

        # Chebyshev tail
        if lvl_idx == 0 and self.cheby_tail > 0 and self.smoother_fine != "chebyshev":
            x_vec = self._chebyshev(lvl_idx, x_vec, b_vec, iters=self.cheby_tail)

        if lvl_idx == 0:
            self._apply_anchor(x_vec)

        r_out = b_vec - self._apply_A(lvl_idx, x_vec)
        rho_v = r_out.norm() / (r_in.norm() + 1e-30)

        if debug_any:
            print(f"[VC] L{lvl_idx} end   ‖r_out‖₂={r_out.norm():.3e}  ρ_v={rho_v.item():.3e}")

        return x_vec


    # ------------------------------------------------------------------
    def solve(self, rhs: torch.Tensor, *, tol: float | None = None, max_iter: int | None = None): # noqa: D401

        tol = self.default_tol if tol is None else tol
        max_iter = self.default_max_iter if max_iter is None else max_iter

        if getattr(self, "debug", False):
            _stats("S0.rhs_in", rhs)

        """Решает A δ = rhs (rhs в физических единицах) и возвращает δ в физических единицах."""
        # Переносим rhs на устройство решателя и корректный dtype
        rhs = rhs.to(device=self.device, dtype=torch.float64)
        # Согласуем RHS с Â = S · (W_rows · A_phys) · S: b̂ = S · (W_rows · b_phys)
        rhs_hat = self._to_hat(self._left_scale_rhs(rhs))
        rhs_hat[self.anchor_fine] = 0.0

        if self.debug:
            A0v = self.levels[0].A_csr.values()
            print(f"[SCALE] ||rhs_hat||inf={rhs_hat.abs().max():.3e}  "
                f"||A_hat||inf={A0v.abs().max():.3e}  ratio={rhs_hat.abs().max()/A0v.abs().max():.3e}")


        # ---- DEBUG -------------------------------------------------------
        if self.debug or os.environ.get("OIL_DEBUG", "0") == "1":
            rhs_l2  = rhs_hat.norm().item()
            rhs_inf = rhs_hat.abs().max().item()
            print(f"[GeoSolverV2] DEBUG RHS_hat: ||·||₂={rhs_l2:.3e}, ||·||_inf={rhs_inf:.3e}")
        # --- DEBUG: сравнение масштаба RHS и Â ---
        if self.debug:
            try:
                rhs_hat_inf = torch.norm(rhs_hat, p=float('inf')).item()
                a_vals = self.levels[0].A_csr.values()
                a_hat_inf = torch.norm(a_vals, p=float('inf')).item()
                print(f"[DBG] RHS_hat_inf={rhs_hat_inf:.3e}, A_hat_inf={a_hat_inf:.3e}, ratio={rhs_hat_inf/a_hat_inf if a_hat_inf>0 else float('inf'):.3e}")
            except Exception as _e:
                print(f"[DBG] Не удалось вычислить масштаб RHS/Â: {_e}")
        x = torch.zeros_like(rhs_hat)
        for it in range(max_iter):
            mult = 2 if self.cycle_type == "W" else 1
            for _ in range(self.cycles_per_call * mult):
                x = self._v_cycle(0, x, rhs_hat)

            # self._apply_anchor(x)   # <-- вставка
            res = rhs_hat - self._apply_A(0, x)

            if self.debug:
                print(f"[SOLV] it={it}  ||res||2={res.norm():.3e}  rel={res.norm()/(rhs_hat.norm()+1e-12):.3e}")



            # ----- CHECK FINITE -------------------------------------------------
            if (not torch.isfinite(x).all()) or (not torch.isfinite(res).all()):
                n_x_finite  = torch.isfinite(x).sum().item()
                n_res_finite = torch.isfinite(res).sum().item()
                print(
                    f"[GeoSolverV2] ❌ NaN/Inf detected after V-cycle: "
                    f"finite(x)={n_x_finite}/{x.numel()}, finite(res)={n_res_finite}/{res.numel()}, "
                    f"|x|_max={x.abs().max().item():.3e}, |res|_max={res.abs().max().item():.3e}"
                )
                # Возвращаем None, чтобы вызывающая сторона могла переключиться
                return None
            if self.debug:
                # расширенный лог одной строки
                Ax = rhs_hat - res
                print(
                    f"[DBG] it={it} ‖x‖={x.norm():.3e} ‖Ax‖={Ax.norm():.3e} "
                    f"res_norm={res.norm():.3e} δmin={x.min():.3e} δmax={x.max():.3e} "
                    f"rmin={res.min():.3e} rmax={res.max():.3e}"
                )
            res_norm = res.norm() / (rhs_hat.norm() + 1e-12)
            print(f"[GeoSolverV2] iter {it}: res_norm={res_norm.item():.3e}")
            # если за 2 итерации res_norm почти не падает – пересобрать только RHS масштаб:
            # Удалён «перезапуск RHS» — он менял масштаб задачи на лету
            
            if not torch.isfinite(res_norm):
                print("[GeoSolverV2] res_norm стал NaN/Inf – прерываем solve")
                break
            if res_norm < tol:
                break
        # решение в hat → сразу обратный diag-scale (левый row-scale не меняет неизвестные)
        delta_phys = self._to_phys(x)
        return delta_phys.to(rhs.device)
    
    def solve_hat(self, b_hat: torch.Tensor, *, tol=None, max_iter=None):
        # Вход уже в hat: b_hat = S_total · (W_rows · b_phys). Дополнительный W_rows не применяем.
        x = torch.zeros_like(b_hat)
        r0 = b_hat
        rhs0 = torch.linalg.norm(r0)

        max_iter = max_iter or 2
        tol = tol or 1e-1            # для предобуславливателя достаточно 0.1

        prev_rel = float('inf')
        for it in range(max_iter):
            x = self._v_cycle(0, x, b_hat)        # один V-cycle
            r = b_hat - self.levels[0].matvec_hat(x)
            rel = (r.norm() / (rhs0 + 1e-30)).item()
            # early exit
            if rel < tol: 
                break
            # стагнация (снижение <5%) — выходим
            if prev_rel - rel < 0.05 * prev_rel:
                break
            prev_rel = rel
        return x


    # (Опционально) обёртка, если где-то ещё вызывается "старый" solve с физическими единицами.
    def solve_phys(self,
                   rhs_phys: torch.Tensor,
                   *,
                   tol: float | None = None,
                   max_iter: int | None = None) -> torch.Tensor:
        """
        Старый интерфейс: rhs в физических единицах -> решение в физических единицах.
        Использует solve_hat внутри.

        Оставь только если реально нужно.
        """
        rhs_phys = rhs_phys.to(device=self.device, dtype=torch.float64)
        # Согласуем RHS с эквилибрированной матрицей: b̂ = S · (W_rows · b_phys)
        rhs_hat = self._to_hat(self._left_scale_rhs(rhs_phys))
        rhs_hat[self.anchor_fine] = 0.0

        delta_hat = self.solve_hat(rhs_hat, tol=tol, max_iter=max_iter)
        if delta_hat is None:
            return None

        # обратно в физические (левый row-scale не применяется к неизвестным)
        delta_phys = self._to_phys(delta_hat)
        return delta_phys.to(dtype=rhs_phys.dtype, device=rhs_phys.device)
   

    def apply_prec(self, b_hat: torch.Tensor, cycles: int = 1) -> torch.Tensor:
        if b_hat.abs().max() < 1e-20:
            return torch.zeros_like(b_hat)
        rhs = b_hat.clone()
        rhs[self.anchor_fine] = 0.0
        x = torch.zeros_like(rhs, dtype=torch.float64, device=self.device)
        for _ in range(cycles):
            x = self._v_cycle(0, x, rhs)
        return x
    
    def apply_prec_phys(self, rhs_phys: torch.Tensor, cycles: int = 1) -> torch.Tensor:
        # b̂ = S · (W_rows · b_phys)
        rhs_hat = self._to_hat(self._left_scale_rhs(rhs_phys.to(self.device, torch.float64)))
        if rhs_hat.abs().max() < 1e-20:
            return torch.zeros_like(rhs_phys).to(self.device, torch.float64)
        # Предобуславливатель в hat: лишний W_rows не нужен
        rhs_hat = rhs_hat
        rhs_hat[self.anchor_fine] = 0.0
        x_hat = torch.zeros_like(rhs_hat)
        for _ in range(cycles):
            x_hat = self._v_cycle(0, x_hat, rhs_hat)
        return self._to_phys(x_hat)
    
    def apply_prec_hat(self, rhs_p_hat: torch.Tensor, cycles: int = 1) -> torch.Tensor:
        """
        CPR-режим: принимает и возвращает вектор РОВНО в hat-пространстве (pressure-блок).
        Никакого phys↔hat внутри. Anchor обнуляется.
        """
        if rhs_p_hat.abs().max() < 1e-20:
            return torch.zeros_like(rhs_p_hat)
        rhs = rhs_p_hat.to(device=self.device, dtype=torch.float64).clone()
        rhs[self.anchor_fine] = 0.0
        x = torch.zeros_like(rhs)
        for _ in range(max(1, cycles)):
            x = self._v_cycle(0, x, rhs)
        # Возвращаем приблизительное решение в тех же hat-координатах (без деления на W)
        x[self.anchor_fine] = 0.0
        return x  # HAT!