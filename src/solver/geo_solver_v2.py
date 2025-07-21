"""Geo-AMG v2 – каркас с эквилибрированием D^-1 A D^-1 и GeoLevel-иерархией.

! ВАЖНО: это минимально рабочая версия для первых тестов на 32^3 и 60×60×30.
  Сглаживатель – damped Jacobi, один V-cycle на итерацию.
  K-cycle/Chebyshev/L1-GS будут добавлены позже.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F
import os
import numpy as np

from solver.geo_level import GeoLevel, build_level_csr

__all__ = ["GeoSolverV2"]

class GeoSolverV2:
    """Экспериментальная реализация геометрического AMG с эквилибрированием."""

    def __init__(self, reservoir, *, omega: float = 0.8,
                 max_coarse_ratio: int = 500, device: str | None = None,
                 cycle_type: str = "W", cycles_per_call: int = 3,
                 pre_smooth: int = 3, post_smooth: int = 3,
                 omega_fine: float = 0.35,
                 smoother_fine: str = "rbgs",  # rbgs|linez|chebyshev
                 cheby_tail: int = 3,
                 delta_clip_factor: float | None = 1000.0,
                 clip_kappa: float = 5.0,
                 max_levels: int | None = None,
                 debug: bool | None = None):
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

# --- Больше не применяем глобальный matrix_scale: работаем в физическом масштабе ---
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
        scale_full = 1.0 / torch.sqrt(diag_orig.clamp_min(1e-20))  # diag^{-1/2}
        scale_sqrt = torch.sqrt(scale_full)                         # diag^{-1/4} = S1

        # --- масштабируем values ---
        crow = base_lvl.A_csr.crow_indices()
        col = base_lvl.A_csr.col_indices()
        vals = base_lvl.A_csr.values()

        # Для каждой строки i повторяем scale[i] row_nnz раз
        row_counts = crow[1:] - crow[:-1]
        row_idx = torch.repeat_interleave(torch.arange(crow.numel()-1, device=self.device), row_counts)
        vals.mul_(scale_sqrt[row_idx] * scale_sqrt[col])

        # ---- Второй симметричный шаг: диагональ к 1 ----------------
        diag_idx_tmp = crow[1:] - 1
        diag_tmp = vals[diag_idx_tmp].abs().clone()
        # Второе симметричное эквилибрирование: S2 = diag(A1)^{-1/2}
        second_scale = 1.0 / torch.sqrt(diag_tmp.clamp_min(1e-20))

        # масштабируем ещё раз: A ← S2 A S2
        vals.mul_(second_scale[row_idx] * second_scale[col])

        diag_final = vals[diag_idx_tmp].abs().clone()
        base_lvl.diag = diag_final  # теперь diag ≈ 1

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
        # Старый ранний блок диагностики перемещён в конец __init__.
        # Деактивируем его, чтобы не выполнялся до инициализации self.levels.
        if False and os.environ.get("OIL_DEBUG", "0") == "1":
            try:
                if len(self.levels) > 0:
                    base_lvl = self.levels[0]
                    n_cells = base_lvl.n_cells
                    with torch.no_grad():
                        e = torch.randn(n_cells, device=self.device, dtype=torch.float64)
                        e = e / (e.norm() + 1e-30)

                        x0 = torch.zeros_like(e)
                        x_smooth = self._rb_gs(0, x0.clone(), -e, iters=1)
                        smooth_factor = (x_smooth.norm() / e.norm()).item()

                        x_v = self._v_cycle(0, x0.clone(), -e)
                        vcycle_factor = (x_v.norm() / e.norm()).item()

                        print(f"[DIAG] RBGS factor={smooth_factor:.3e}, V-cycle factor={vcycle_factor:.3e}")
            except Exception as _e:
                print(f"[DIAG] Ошибка диагностики AMG: {_e}")

        # ------------------------------------------------------------------
        # Итоговый симметричный масштаб S_total = S2 · S1  (элемент-wise).
        # Он удовлетворяет A_hat = S_total · A_phys · S_total, diag(A_hat) ≈ 1.
        # Для согласованного решения системы A δ = rhs нужно масштабировать
        # RHS и восстанавливать δ именно через S_total, а НЕ только через
        # первый шаг scale_full.  Ошибка здесь приводила к гигантским δ и
        # нестабильности Geo-AMG.
        # ------------------------------------------------------------------
        S_total = scale_sqrt * second_scale  # diag-vector

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
        self.Dinv = S_total  # = S2 · S1
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
            diag_idx = crow[1:] - 1
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


        # Для построения грубых уровней используем исходные проницаемости
        kx_c, ky_c, kz_c = kx, ky, kz
        hx, hy, hz = self.hx, self.hy, self.hz

        def pool(t):
            return F.avg_pool3d(t[None, None, ...], kernel_size=2, stride=2, padding=0)[0, 0]

        lvl_count = 1  # уже есть базовый уровень
        while kx_c.numel() > 1 and min(kx_c.shape) >= 2:
            if max_levels is not None and lvl_count >= max_levels:
                break
            kx_c = pool(kx_c); ky_c = pool(ky_c); kz_c = pool(kz_c)
            hx *= 2.0; hy *= 2.0; hz *= 2.0
            lvl = GeoLevel(kx_c, ky_c, kz_c, hx, hy, hz, device=self.device)
            # эквилибрируем каждый новый уровень
            _equilibrate_level(lvl)
            # Пересчитываем inv_l1 для отмасштабированного уровня
            crow_c = lvl.A_csr.crow_indices()
            vals_c = lvl.A_csr.values().abs()
            row_counts_c = crow_c[1:] - crow_c[:-1]
            row_idx_c = torch.repeat_interleave(torch.arange(crow_c.numel()-1, device=self.device), row_counts_c)
            row_abs_sum_c = torch.zeros_like(lvl.diag)
            row_abs_sum_c.index_add_(0, row_idx_c, vals_c)

            iso_mask_c = row_abs_sum_c < 1e-8
            safe_sum_c = row_abs_sum_c.clone(); safe_sum_c[iso_mask_c] = 1.0
            lvl.inv_l1 = (1.0 / safe_sum_c).clamp_max(1.0)
            lvl.inv_l1[iso_mask_c] = 0.0

            if os.environ.get("OIL_DEBUG", "0") == "1":
                print(
                    f"[ISO L{lvl_count}] isolated rows (<1e-8): {iso_mask_c.sum().item()}/{len(iso_mask_c)}"
                )
                print(
                    f"[L{lvl_count}] row_abs_sum_hat: min={row_abs_sum_c.min().item():.3e}, "
                    f"median={row_abs_sum_c.median().item():.3e}, max={row_abs_sum_c.max().item():.3e}"
                )
                print(
                    f"[L{lvl_count}] inv_l1 stats: min={lvl.inv_l1.min().item():.3e}, "
                    f"median={lvl.inv_l1.median().item():.3e}, max={lvl.inv_l1.max().item():.3e}"
                )
            print(f"[GeoSolverV2] built level {len(self.levels)}: n={lvl.n_cells}")
            self.levels.append(lvl)
            lvl_count += 1

# ----------------- Диагностика эффективности после построения иерархии -----------------
        if os.environ.get("OIL_DEBUG", "0") == "1":
            try:
                base_lvl = self.levels[0]
                n_cells_diag = base_lvl.n_cells
                with torch.no_grad():
                    e = torch.randn(n_cells_diag, device=self.device, dtype=torch.float64)
                    e = e / (e.norm() + 1e-30)

                    x0 = torch.zeros_like(e)
                    x_smooth = self._rb_gs(0, x0.clone(), -e, iters=1)
                    smooth_factor = (x_smooth.norm() / e.norm()).item()

                    x_v = self._v_cycle(0, x0.clone(), -e)
                    vcycle_factor = (x_v.norm() / e.norm()).item()

                print(f"[DIAG] RBGS factor={smooth_factor:.3e}, V-cycle factor={vcycle_factor:.3e}")
            except Exception as _e:
                print(f"[DIAG] Ошибка диагностики AMG (post-build): {_e}")

    # ------------------------------------------------------------------
    def _apply_A(self, lvl_idx: int, x: torch.Tensor) -> torch.Tensor:
        lvl = self.levels[lvl_idx]
        Ax = torch.sparse.mm(lvl.A_csr, x.view(-1, 1)).squeeze(1)
        return Ax

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
                if init_clip is not None:
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
        # L1 surrogate diag: 1 / Σ |A_ij|
        inv_diag = lvl.inv_l1
        omega = self.omega_fine if lvl_idx == 0 else self.omega

        red_mask = lvl.is_red
        black_mask = lvl.is_black

        if self.delta_clip_factor is not None:
            init_clip = self.delta_clip_factor * (b.abs().max().item() + 1e-12)
        else:
            init_clip = None

        for k in range(iters):
            # red sweep (in-place)
            r = b - self._apply_A(lvl_idx, x)
            x[red_mask] += omega * inv_diag[red_mask] * r[red_mask]

            # black sweep (in-place, с учётом уже обновлённого красного цвета)
            r = b - self._apply_A(lvl_idx, x)
            x[black_mask] += omega * inv_diag[black_mask] * r[black_mask]

            # динамический clip только на самом тонком уровне
            if init_clip is not None and lvl_idx == 0:
                dyn_clip = self.clip_kappa * (x.abs().max().item() + 1e-12)
                clip_val = max(init_clip, dyn_clip)
                torch.clamp_(x, -clip_val, clip_val)

            if self.debug and lvl_idx == 0 and k < 3:
                inc_mag = (omega * inv_diag * r).abs().max().item()
                print(
                    f"[DBG-RBGS] k={k} |r|₂={r.norm().item():.3e} |r|∞={r.abs().max().item():.3e} |invD|∞={inv_diag.max().item():.3e} "
                    f"|ω*invD*r|∞={inc_mag:.3e} clip={init_clip if init_clip else 0:.3e}"
                )
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
        """Half-weight restriction 2×2×2 -> coarse.
        """
        if os.environ.get("OIL_DEBUG", "0") == "1":
            n_before = vol3d.norm().item()
        coarse = F.avg_pool3d(vol3d[None, None, ...], kernel_size=2, stride=2, padding=0)[0, 0]
        if os.environ.get("OIL_DEBUG", "0") == "1":
            print(f"[RS] restrict norm ratio={(coarse.norm()/(n_before+1e-30)):.3e}")
        return coarse

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
        """Recursive V-/W-cycle in 3-D form, input / output вектор (flattened)."""
        lvl = self.levels[lvl_idx]
        debug_any = os.environ.get("OIL_DEBUG", "0") == "1"
        debug_top = debug_any and lvl_idx == 0
        if debug_any:
            r0 = b_vec - self._apply_A(lvl_idx, x_vec)
            print(f"[VC] L{lvl_idx} start ‖x‖={x_vec.norm():.3e} ‖r‖₂={r0.norm():.3e}")

        # Coarsest grid: если ≤8 неизвестных – решаем точно, иначе 30 Jacobi
        if lvl_idx == len(self.levels) - 1 or lvl.n_cells <= 8:
            if lvl.n_cells <= 8:
                # Точное решение маленькой системы, но сначала эквилибрируем
                A_dense = lvl.A_csr.to_dense()
                b_dense = b_vec.clone()

                # Row-scale: denom = Σ|A_ij|, min 1e-12
                row_sum = A_dense.abs().sum(dim=1).clamp_min(1e-12)
                D = torch.diag(1.0 / row_sum)
                A_equil = D @ A_dense
                b_equil = D @ b_dense

                # DEBUG: подробная проверка матрицы coarse уровня
                if os.environ.get("OIL_DEBUG", "0") == "1":
                    print(f"[COARSE L{lvl_idx}] row_sum: {row_sum.tolist()}")
                    print(f"[COARSE L{lvl_idx}] diag: {torch.diag(A_equil).tolist()}")
                    try:
                        svals = torch.linalg.svdvals(A_equil)
                        print(f"[COARSE L{lvl_idx}] σ_min={svals.min().item():.3e}, σ_max={svals.max().item():.3e}, cond={svals.max()/svals.min() if svals.min()>0 else float('inf'):.3e}")
                    except Exception as _svd_e:
                        print(f"[COARSE L{lvl_idx}] SVD error: {_svd_e}")
                    print(f"[COARSE L{lvl_idx}] RHS: {b_equil.tolist()}")

                    # --- Регуляризация: добавляем объёмный член C*I ---------
                    reg_coeff = 1e-3  # эквивалент φβ/Δt на грубой сетке
                    A_equil_reg = A_equil + torch.eye(A_equil.size(0), dtype=A_equil.dtype, device=A_equil.device) * reg_coeff

                    try:
                        sol_scaled = torch.linalg.solve(A_equil_reg, b_equil.view(-1,1)).squeeze(1)
                        return sol_scaled  # уже в исходном масштабе b_equil
                    except Exception:
                        # fallback Jacobi если лин. система вырождена
                        x_vec = self._jacobi(lvl_idx, x_vec, b_vec, iters=100)
            else:
                x_vec = self._jacobi(lvl_idx, x_vec, b_vec, iters=30)

        nz, ny, nx = lvl.kx.shape
        x3d = x_vec.reshape(nz, ny, nx)
        b3d = b_vec.reshape_as(x3d)

        # Pre-smoothing
        if self.smoother_fine == "rbgs":
            x_vec = self._rb_gs(lvl_idx, x_vec, b_vec, iters=self.pre_smooth)
        if debug_any:
            r1 = b_vec - self._apply_A(lvl_idx, x_vec)
            print(f"[VC] L{lvl_idx} after pre-smooth ‖x‖={x_vec.norm():.3e} ‖r‖₂={r1.norm():.3e}")
        x3d = x_vec.reshape_as(x3d)

        # residual (3-D)
        r_vec = b_vec - self._apply_A(lvl_idx, x_vec)
        r3d = r_vec.reshape_as(x3d)

        # restrict residual to coarse grid
        r_c3d = self._restrict3d(r3d)
        x_c3d = torch.zeros_like(r_c3d)

        # recurse (V-cycle)
        x_c_vec = self._v_cycle(lvl_idx + 1, x_c3d.reshape(-1), r_c3d.reshape(-1))
        x_c3d = x_c_vec.reshape_as(r_c3d)

        # prolongate error to fine grid
        correction = self._prolong3d(x_c3d, (nz, ny, nx))

        if debug_any:
            print(f"[VC] L{lvl_idx} ‖r_c‖₂={r_c3d.norm():.3e} → ‖e_c‖₂={x_c3d.norm():.3e}, gain={x_c3d.norm()/(r_c3d.norm()+1e-30):.3e}")
            print(f"[VC] L{lvl_idx} prolong ‖P e_c‖₂={correction.norm():.3e}, ratio={correction.norm()/(x_c3d.norm()+1e-30):.3e}")
        # apply coarse-grid correction
        x_res = x_vec - correction.view(-1)
        if debug_any:
            r2 = b_vec - self._apply_A(lvl_idx, x_res)
            print(f"[VC] L{lvl_idx} after coarse corr ‖x‖={x_res.norm():.3e} ‖r‖₂={r2.norm():.3e}")
        x3d = x3d + correction

        # post-smooth
        if self.smoother_fine == "rbgs":
            x_vec = self._rb_gs(lvl_idx, x_res, b_vec, iters=self.post_smooth)
        if debug_any:
            r3 = b_vec - self._apply_A(lvl_idx, x_vec)
            print(f"[VC] L{lvl_idx} after post-smooth ‖x‖={x_vec.norm():.3e} ‖r‖₂={r3.norm():.3e}")
        x3d = x_vec.reshape_as(x3d)

        # Chebyshev tail (если явно включён и не совпадает со smoother)
        if lvl_idx == 0 and self.cheby_tail > 0 and self.smoother_fine != "chebyshev":
            x_vec = self._chebyshev(lvl_idx, x_vec, b_vec, iters=self.cheby_tail)

        return x_vec

    # ------------------------------------------------------------------
    def solve(self, rhs: torch.Tensor, tol: float = 1e-6, max_iter: int = 10):  # noqa: D401
        """Решает A δ = rhs и возвращает физический δ (масштаб снят)."""
        # Переносим rhs на устройство решателя и корректный dtype
        if rhs.device.type != self.device:
            rhs = rhs.to(self.device)
        if rhs.dtype != torch.float64:
            rhs = rhs.to(dtype=torch.float64)

        # ------------- RHS: работаем напрямую в hat-пространстве ------------
        rhs_hat = rhs.clone()  # без умножения на S_total (Dinv)

        # ------------- Row-scale RHS (корректировка масштаба) -------------
        row_norm = rhs_hat.norm() / math.sqrt(rhs_hat.numel()) + 1e-30
        rhs_hat  = rhs_hat / row_norm

        # ---- DEBUG -------------------------------------------------------
        if self.debug or os.environ.get("OIL_DEBUG", "0") == "1":
            rhs_l2  = rhs_hat.norm().item()
            rhs_inf = rhs_hat.abs().max().item()
            print(f"[GeoSolverV2] DEBUG RHS_hat: ||·||₂={rhs_l2:.3e}, ||·||_inf={rhs_inf:.3e}, row_norm={row_norm:.3e}")
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
            for _ in range(self.cycles_per_call):
                x = self._v_cycle(0, x, rhs_hat)
                if self.cycle_type == "W":
                    x = self._v_cycle(0, x, rhs_hat)
            res = rhs_hat - self._apply_A(0, x)
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
            if not torch.isfinite(res_norm):
                print("[GeoSolverV2] res_norm стал NaN/Inf – прерываем solve")
                break
            if res_norm < tol:
                break
        # Возвращаем прежний row-scale (эквилибрирование не применялось)
        delta_phys = x * row_norm
        if torch.isfinite(delta_phys).all():
            print(f"[GeoSolverV2] DEBUG solve: ||rhs_hat||={rhs_hat.norm().item():.3e}, ||x||={x.norm().item():.3e}, ||delta_phys||={delta_phys.norm().item():.3e}")
        else:
            print("[GeoSolverV2] DEBUG solve: NaN/Inf in delta_phys")
        return delta_phys.cpu().numpy() 