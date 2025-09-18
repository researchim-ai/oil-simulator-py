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

def _row_equilibrate_csr(A_csr: torch.Tensor, eps: float = 1e-20):
    """Left scaling: W A, где W = diag(1/||row||₁). Возвращает (A_scaled, w_rows)."""
    crow = A_csr.crow_indices()
    col  = A_csr.col_indices()
    vals = A_csr.values()

    row_counts = crow[1:] - crow[:-1]
    row_idx = torch.repeat_interleave(torch.arange(crow.numel()-1, device=A_csr.device), row_counts)
    row_abs_sum = torch.zeros(crow.numel()-1, device=A_csr.device, dtype=vals.dtype)
    row_abs_sum.index_add_(0, row_idx, vals.abs())

    w = 1.0 / torch.clamp(row_abs_sum, min=eps)
    vals.mul_(w[row_idx])   # только слева (по строкам)
    return A_csr, w


def build_block_maps(shape_f, device):
    """Маппинг fine→coarse для (почти) 2x2x2 коарсенинга.
    Возвращает:
        parent_idx (LongTensor, n_f)
        shape_c    (nz_c, ny_c, nx_c)
        n_c        (int)
        child_cnt  (LongTensor, n_c)  # сколько fine-узлов у каждого coarse-узла
    """
    nz, ny, nx = shape_f
    nz_c, ny_c, nx_c = (nz + 1) // 2, (ny + 1) // 2, (nx + 1) // 2

    z = torch.arange(nz, device=device)
    y = torch.arange(ny, device=device)
    x = torch.arange(nx, device=device)
    Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")

    Zc = Z // 2
    Yc = Y // 2
    Xc = X // 2
    parent = (Zc * ny_c + Yc) * nx_c + Xc

    parent_idx = parent.reshape(-1).to(torch.int64)
    n_c = int(parent_idx.max().item()) + 1
    child_cnt = torch.bincount(parent_idx, minlength=n_c).to(torch.int64)

    return parent_idx, (nz_c, ny_c, nx_c), n_c, child_cnt


def build_P_csr(shape_f, device):
    """Piecewise-constant prolongation."""
    parent_idx, shape_c, n_c, child_cnt = build_block_maps(shape_f, device)
    n_f = parent_idx.numel()

    crow = torch.arange(n_f + 1, device=device, dtype=torch.int64)
    col  = parent_idx
    val  = torch.ones(n_f, device=device, dtype=torch.float64)

    P = torch.sparse_csr_tensor(crow, col, val, size=(n_f, n_c),
                                device=device, dtype=torch.float64)
    return P, parent_idx, n_c, child_cnt, shape_c


def build_R_csr(P, child_cnt):
    """Restriction: среднее по детям coarse-узла, R = D^{-1} P^T, где D=diag(child_cnt)."""
    P_coo = P.to_sparse_coo()
    fine = P_coo.indices()[0]   # строки в P
    coarse = P_coo.indices()[1] # столбцы в P

    w = (1.0 / child_cnt[coarse].to(P.dtype)) * P_coo.values()
    Rt_indices = torch.stack([coarse, fine], dim=0)  # транспонируем
    Rt_coo = torch.sparse_coo_tensor(Rt_indices, w,
                                     size=(P.shape[1], P.shape[0]),
                                     device=P.device, dtype=P.dtype)
    return Rt_coo.to_sparse_csr()



def rap_pc_const_gpu(Af_csr: torch.Tensor,
                     parent_idx: torch.Tensor,
                     n_c: int,
                     child_cnt) -> torch.Tensor:
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

    # вес из R (среднее по детям соответствующей coarse-строки)
    w = val / child_cnt[I].to(val.dtype)

    idx = torch.stack([I, J], dim=0)  # 2 x nnz
    Ac = torch.sparse_coo_tensor(idx, w, (n_c, n_c),
                                 device=device, dtype=val.dtype).coalesce().to_sparse_csr()
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
                 default_max_iter: int = 10):
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

        base_lvl.A_csr, w_rows0 = _row_equilibrate_csr(base_lvl.A_csr)
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
            crow = A_csr.crow_indices()
            col  = A_csr.col_indices()
            val  = A_csr.values()

            # row
            s, e = crow[idx].item(), crow[idx+1].item()
            val[s:e] = 0.0
            # col
            mask = (col == idx)
            val[mask] = 0.0
            # diag (последний элемент в строке)
            val[e-1] = 1.0
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

        row_starts, row_ends = crow[:-1], crow[1:]
        diag_idx_tmp = torch.empty_like(row_starts, dtype=torch.int64)
        for i in range(row_starts.numel()):
            s, e = int(row_starts[i]), int(row_ends[i])
            pos = torch.nonzero(col[s:e] == i, as_tuple=False)
            assert pos.numel() == 1, "diag not found or multiple diags"
            diag_idx_tmp[i] = s + int(pos.item())

        diag_tmp = vals[diag_idx_tmp].abs().clone()
        second_scale = 1.0 / torch.sqrt(diag_tmp.clamp_min(1e-20))  # S2
        # применяем S2
        vals.mul_(second_scale[row_idx] * second_scale[col])

        diag_final = vals[diag_idx_tmp].abs().clone()
        base_lvl.diag = diag_final  # теперь diag ≈ 1

        # === PIN null mode AFTER equilibration ===
        self.anchor_fine = int(base_lvl.diag.argmax().item())   # берем самую «жесткую» ячейку
        _pin_rowcol(base_lvl.A_csr, self.anchor_fine)
        base_lvl.anchor = self.anchor_fine
        # inv_l1 пересчитается ниже, но на всякий случай обнулим позже ещё раз

# --- пересчёт inv_l1 для базового уровня в эквилибрированном масштабе ---
        crow = base_lvl.A_csr.crow_indices()
        vals_hat = base_lvl.A_csr.values().abs()  # уже отмасштабированные значения
        row_counts = crow[1:] - crow[:-1]
        row_idx = torch.repeat_interleave(torch.arange(crow.numel()-1, device=self.device), row_counts)
        row_abs_sum_hat = torch.zeros_like(diag_final)
        row_abs_sum_hat.index_add_(0, row_idx, vals_hat)

        iso_mask0 = row_abs_sum_hat < 1e-8
        safe_sum0 = row_abs_sum_hat.clone(); safe_sum0[iso_mask0] = 1.0
        base_lvl.inv_l1 = (1.0 / safe_sum0).clamp_max(1.0)
        base_lvl.inv_l1[iso_mask0] = 0.0
        base_lvl.inv_l1[self.anchor_fine] = 0.0

        if os.environ.get("OIL_DEBUG", "0") == "1":
            print(
                f"[ISO L0] isolated rows (<1e-8): {iso_mask0.sum().item()}/{len(iso_mask0)}"
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


        # --- диагностика после одного шага ---
        if self.debug and (torch.isnan(vals).any() or torch.isinf(vals).any()):
            nan_cnt = torch.isnan(vals).sum().item()
            inf_cnt = torch.isinf(vals).sum().item()
            print(f"[GeoSolverV2] ВНИМАНИЕ: после эквилибрирования NaN={nan_cnt}, Inf={inf_cnt}")
        if self.debug:
            print(f"[GeoSolverV2] scale_full min={scale_full.min().item():.3e}, max={scale_full.max().item():.3e}")
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
            row_starts = crow[:-1]; row_ends = crow[1:]
            diag_idx = torch.empty_like(row_starts, dtype=torch.int64)
            for i in range(row_starts.numel()):
                s, e = row_starts[i].item(), row_ends[i].item()
                row_cols = col[s:e]
                # ищем индекс элемента, где col == i
                pos = torch.nonzero(row_cols == i, as_tuple=False)
                assert pos.numel() == 1, "diag not found or multiple diags"
                diag_idx[i] = s + pos.item()


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

            # 1) строим маппинг (без дорогостоящего RAP, если дальше не пойдём)
            P, parent_idx, n_c, child_cnt, shape_c = build_P_csr((nz_f, ny_f, nx_f), self.device)

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

            # если прошли проверки — строим R и сохраняем операторы
            R = build_R_csr(P, child_cnt)
            fine_lvl.P = P
            fine_lvl.R = R         

            # 2) RAP
            A_csr = rap_pc_const_gpu(fine_lvl.A_csr, parent_idx, n_c, child_cnt)

            # ---- CHECK RAP -------------------------------------------------------------
            if self.debug:
                R = R  # уже построен
                P = P
                Af = fine_lvl.A_csr
                Ac_ref = torch.sparse.mm(torch.sparse.mm(R, Af), P).to_dense()
                Ac_dense = A_csr.to_dense()
                diff = (Ac_ref - Ac_dense).abs()
                rel_err = diff.sum() / (Ac_ref.abs().sum() + 1e-30)
                print(f"[RAPCHK L{len(self.levels)}] rel_err={rel_err.item():.3e}, "
                    f"||diff||1={diff.sum().item():.3e}")


            if A_csr._nnz() == 0:
                break
            # 3) diag_c и inv_l1
            crow = A_csr.crow_indices()
            col  = A_csr.col_indices()
            vals = A_csr.values()

            # diag_idx = crow[1:] - 1
            row_starts = crow[:-1]; row_ends = crow[1:]
            diag_idx = torch.empty_like(row_starts, dtype=torch.int64)
            for i in range(row_starts.numel()):
                s, e = row_starts[i].item(), row_ends[i].item()
                row_cols = col[s:e]
                # ищем индекс элемента, где col == i
                pos = torch.nonzero(row_cols == i, as_tuple=False)
                assert pos.numel() == 1, "diag not found or multiple diags"
                diag_idx[i] = s + pos.item()


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

            # 4) создаём GeoLevel из CSR (используем build_level_csr если он есть)
            nz_c, ny_c, nx_c = nz_f // 2, ny_f // 2, nx_f // 2
            hx_c, hy_c, hz_c = fine_lvl.hx * 2, fine_lvl.hy * 2, fine_lvl.hz * 2

            lvl = build_level_from_csr(
                A_csr, diag_c, inv_l1,
                shape_c,
                hx_c, hy_c, hz_c,
                device=self.device
            )

            # [NEW] сначала левое эквилибрирование
            lvl.A_csr, w_rows_c = _row_equilibrate_csr(lvl.A_csr)
            lvl.diag = lvl.diag * w_rows_c
            # нормализуем и клампим W_rows на уровне
            med_c = torch.median(w_rows_c)
            if torch.isfinite(med_c) and med_c > 0:
                w_rows_c = w_rows_c / med_c
            lvl.W_rows = torch.clamp(w_rows_c, 1e-6, 1e6)
            # потом твой симметричный шаг
            _ = _equilibrate_level(lvl)

            if self.debug:
                crow = lvl.A_csr.crow_indices(); val = lvl.A_csr.values()
                diag = val[crow[1:]-1]
                print(f"[LVL {len(self.levels)}] n={lvl.n_cells} nnz={lvl.A_csr._nnz()} "
                    f"diag[min,med,max]=({diag.min():.3e},{diag.median():.3e},{diag.max():.3e}) "
                    f"A[min,max]=({val.min():.3e},{val.max():.3e})")


            # пересчёт inv_l1 после эквилибрирования
            crow = lvl.A_csr.crow_indices(); col = lvl.A_csr.col_indices(); vals = lvl.A_csr.values().abs()
            rc = crow[1:] - crow[:-1]
            ridx = torch.repeat_interleave(torch.arange(lvl.n_cells, device=self.device), rc)
            row_abs = torch.zeros(lvl.n_cells, device=self.device, dtype=torch.float64)
            row_abs.index_add_(0, ridx, vals)
            iso = row_abs < 1e-8
            safe = row_abs.clone(); safe[iso] = 1.0
            lvl.inv_l1 = (1.0 / safe)
            lvl.inv_l1[iso] = 0.0

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
                s, e = crow[idx].item(), crow[idx+1].item()
                val[s:e] = 0.0
                mask = (col == idx)
                val[mask] = 0.0
                val[e-1] = 1.0

            _pin_rowcol_local(A_csr, anchor_c)
            # diag_c[anchor_c] = 1.0
            # inv_l1[anchor_c] = 0.0
            # lvl.A_csr = A_csr
            # lvl.diag = diag_c
            # lvl.inv_l1 = inv_l1
            lvl.diag[anchor_c] = 1.0
            lvl.inv_l1[anchor_c] = 0.0
            lvl.anchor = anchor_c  # можно сохранить

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

    def _apply_anchor(self, v: torch.Tensor):
        # держим нулевой мод на самом тонком уровне
        v[self.anchor_fine] = 0.0
        return v

    def _restrict_vec(self, lvl_idx: int, r_f: torch.Tensor) -> torch.Tensor:
        """r_c = R * r_f"""
        R = self.levels[lvl_idx].R
        return torch.sparse.mm(R, r_f.view(-1, 1)).squeeze(1)

    def _prolong_vec(self, lvl_idx: int, e_c: torch.Tensor) -> torch.Tensor:
        """e_f = P * e_c"""
        P = self.levels[lvl_idx].P
        return torch.sparse.mm(P, e_c.view(-1, 1)).squeeze(1)

    def _check_RAP(self, fine_lvl, Ac, parent_idx, child_cnt):
        # восстановим Ac_ref = R A_f P и сравним
        R, P = fine_lvl.R, fine_lvl.P
        Af = fine_lvl.A_csr
        Ac_ref = torch.sparse.mm(torch.sparse.mm(R, Af), P)
        diff = (Ac_ref.to_dense() - Ac.to_dense()).abs()
        rel = diff.sum() / (Ac_ref.abs().sum() + 1e-30)
        print(f"[RAPCHK] L{len(self.levels)} rel_err={rel.item():.3e} "
            f"||diff||1={diff.sum().item():.3e}")


    def _jacobi(self, lvl_idx: int, x: torch.Tensor, b: torch.Tensor, iters: int = 2) -> torch.Tensor:
        lvl = self.levels[lvl_idx]
        # -------- L1-Jacobi диагональный суррогат --------
        # --- L1-Jacobi: denom = Σ_j |A_ij| -------------------------------
        inv_diag = lvl.inv_l1
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
            if init_clip is not None and lvl_idx == 0:
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
        diag = lvl.diag
        # степенной метод 3 итерации для λ_max
        v = torch.rand_like(x)
        v = v / (v.norm() + 1e-12)
        for _ in range(3):
            v = self._apply_A(lvl_idx, v)
            v = v / (v.norm() + 1e-12)
        Av = self._apply_A(lvl_idx, v)
        lam_max = torch.dot(v, Av).item() * 1.05
        lam_min = lam_max / 30.0
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
        inv_diag = lvl.inv_l1
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

            # dynamic clip only on finest
            if init_clip is not None and lvl_idx == 0:
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
                            c_prime[k] = a_u[k] / denom
                            d_prime[k] = (rhs_col[k] - a_l[k-1] * d_prime[k-1]) / denom
                        denom_last = a_d[-1] - a_l[-1] * c_prime[-1]
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

        # initial residual
        r_in = b_vec - self._apply_A(lvl_idx, x_vec)

        if self.debug:
            print(f"[VC L{lvl_idx}] START  {_vstats('r_in', r_in)}   ||x||2={x_vec.norm():.3e}")

        if debug_any:
            print(f"[VC] L{lvl_idx} start ‖x‖={x_vec.norm():.3e} ‖r‖₂={r_in.norm():.3e}")

        if debug_top:
            _stats(f"V{lvl_idx}.b_vec", b_vec)

        # Coarsest grid
        if lvl_idx == len(self.levels) - 1 or lvl.n_cells <= 256:
            anc = lvl.anchor
            A = lvl.A_csr.to_dense()
            A[anc, :] = 0; A[:, anc] = 0; A[anc, anc] = 1.0
            b_fix = b_vec.clone(); b_fix[anc] = 0.0
            x_sol = torch.linalg.solve(A, b_fix.view(-1, 1)).squeeze(1)
            return x_sol


        nz, ny, nx = lvl.kx.shape

        # ---------- pre-smooth ----------
        if self.smoother_fine == "rbgs":
            x_vec = self._rb_gs(lvl_idx, x_vec, b_vec, iters=self.pre_smooth)

        r_pre = b_vec - self._apply_A(lvl_idx, x_vec)
        mu_pre = r_pre.norm() / (r_in.norm() + 1e-30)
        if debug_top:
            _stats(f"V{lvl_idx}.r_after_pre", r_pre)
        if debug_any:
            print(f"[VC] L{lvl_idx} after pre-smooth ‖x‖={x_vec.norm():.3e} ‖r‖₂={r_pre.norm():.3e}  μ_pre={mu_pre.item():.3e}")

        # ---------- restrict (Galerkin) ----------
        r_c_vec = self._restrict_vec(lvl_idx, r_pre)
        # обнуляем RHS в якорной ячейке coarse-уровня
        anc_c = self.levels[lvl_idx + 1].anchor
        r_c_vec[anc_c] = 0.0
        x_c_vec = torch.zeros_like(r_c_vec)

        # ---------- recurse ----------
        x_c_vec = self._v_cycle(lvl_idx + 1, x_c_vec, r_c_vec)

        # ---------- prolong ----------
        corr_vec = self._prolong_vec(lvl_idx, x_c_vec)

        # пробный шаг
        x_try = x_vec + corr_vec
        r_try = b_vec - self._apply_A(lvl_idx, x_try)
        rho_tmp = r_try.norm() / (r_pre.norm() + 1e-30)

        if rho_tmp > 1.0:
            # line-search по энергии: alpha = (r_pre·corr)/(corr·A corr)
            A_corr = self._apply_A(lvl_idx, corr_vec)
            num = torch.dot(r_pre, corr_vec)
            den = torch.dot(corr_vec, A_corr).clamp_min(1e-30)
            alpha = torch.clamp(num / den, 0.0, 1.0)

            x_try2 = x_vec + alpha * corr_vec
            r_try2 = b_vec - self._apply_A(lvl_idx, x_try2)
            if r_try2.norm() < r_pre.norm():
                x_vec = x_try2
            # иначе вообще не добавляем коррекцию
        else:
            x_vec = x_try



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

        # Если грубая коррекция ухудшила остаток — сделаем ещё один заход на coarse
        if rho_corr > 1.0 and lvl_idx == 0:
            # второй заход на тот же coarse уровень (минимальный W-cycle)
            r_c_vec2 = self._restrict_vec(lvl_idx, r_corr)
            x_c_vec2 = torch.zeros_like(r_c_vec2)
            x_c_vec2 = self._v_cycle(lvl_idx + 1, x_c_vec2, r_c_vec2)
            corr_vec2 = self._prolong_vec(lvl_idx, x_c_vec2)
            x_vec = x_vec + corr_vec2
            r_corr = b_vec - self._apply_A(lvl_idx, x_vec)
            rho_corr = r_corr.norm() / (r_pre.norm() + 1e-30)

        # ---------- post-smooth ----------
        if self.smoother_fine == "rbgs":
            x_vec = self._rb_gs(lvl_idx, x_vec, b_vec, iters=self.post_smooth)

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
        # Полная согласованность: Â = S·(W·A_phys)·S, значит b̂ = W·(S·b_phys)
        rhs_hat = self.W_rows * self._to_hat(rhs)
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
            if it == 1 and res_norm > 0.9:
                # подравнять RHS по L∞ к матрице, дешевый «перезапуск»
                rhs_hat = rhs_hat / rhs_hat.abs().max().clamp_min(1e-30) * \
                        self.levels[0].A_csr.values().abs().max()
                x.zero_()
                continue
            
            if not torch.isfinite(res_norm):
                print("[GeoSolverV2] res_norm стал NaN/Inf – прерываем solve")
                break
            if res_norm < tol:
                break
        # решение в hat → снимаем левый row-scale, затем обратный diag-scale
        delta_hat  = x / self.W_rows
        delta_phys = self._to_phys(delta_hat)
        return delta_phys.to(rhs.device)
    
    def solve_hat(self, b_hat: torch.Tensor, *, tol=None, max_iter=None):
        # CPR preconditioner: 1–2 V-cycles is enough
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
        rhs_hat = self._to_hat(rhs_phys)
        rhs_hat[self.anchor_fine] = 0.0

        delta_hat = self.solve_hat(rhs_hat, tol=tol, max_iter=max_iter)
        if delta_hat is None:
            return None

        # обратно в физические
        delta_hat = delta_hat / self.W_rows
        delta_phys = self._to_phys(delta_hat)
        return delta_phys.to(dtype=rhs_phys.dtype, device=rhs_phys.device)
   

    def apply_prec(self, b_hat: torch.Tensor, cycles: int = 1) -> torch.Tensor:
        rhs = b_hat.clone()
        rhs[self.anchor_fine] = 0.0
        x = torch.zeros_like(rhs, dtype=torch.float64, device=self.device)
        for _ in range(cycles):
            x = self._v_cycle(0, x, rhs)
        return x
    
    def apply_prec_phys(self, rhs_phys: torch.Tensor, cycles: int = 1) -> torch.Tensor:
        rhs_hat = self._to_hat(rhs_phys.to(self.device, torch.float64))
        rhs_hat = rhs_hat * self.W_rows        # если вы действительно оставляете row-scale
        rhs_hat[self.anchor_fine] = 0.0
        x_hat = torch.zeros_like(rhs_hat)
        for _ in range(cycles):
            x_hat = self._v_cycle(0, x_hat, rhs_hat)
        x_hat = x_hat / self.W_rows
        return self._to_phys(x_hat)
    
    def apply_prec_hat(self, rhs_p_hat: torch.Tensor, cycles: int = 1) -> torch.Tensor:
        """
        CPR-режим: принимает и возвращает вектор РОВНО в hat-пространстве (pressure-блок).
        Никакого phys↔hat внутри. Anchor обнуляется.
        """
        rhs = rhs_p_hat.to(device=self.device, dtype=torch.float64).clone()
        # согласуем с левой строковой нормировкой: b̂_geo = W_rows · b̂_global
        rhs = rhs * self.W_rows
        rhs[self.anchor_fine] = 0.0
        x = torch.zeros_like(rhs)
        for _ in range(max(1, cycles)):
            x = self._v_cycle(0, x, rhs)
        # возвращаем обратно «без W», чтобы M ≈ Â^{-1} в исходных hat-координатах
        x = x / self.W_rows
        x[self.anchor_fine] = 0.0
        return x  # HAT!