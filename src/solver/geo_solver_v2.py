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
                 omega_fine: float = 0.60,
                 smoother_fine: str = "linez",  # linez|rbgs|chebyshev
                 cheby_tail: int = 3,
                 delta_clip_factor: float = 5.0,
                 clip_kappa: float = 5.0,
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
        scale_full = 1.0 / torch.sqrt(diag_orig.clamp_min(1e-20))  # = diag^{-1/2} после scale
        scale_sqrt = torch.sqrt(scale_full)                         # = diag^{-1/4}

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
        second_scale = 1.0 / torch.sqrt(diag_tmp.clamp_min(1e-20))

        # масштабируем ещё раз: A ← S2 A S2
        vals.mul_(second_scale[row_idx] * second_scale[col])

        diag_final = vals[diag_idx_tmp].abs().clone()
        base_lvl.diag = diag_final  # теперь diag≈1

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

        # Итоговый масштаб RHS: S_total = scale_full (выравнивает RHS с Â)
        self.Dinv = scale_full
        if self.debug:
            print(f"[GeoSolverV2] Dinv min={self.Dinv.min().item():.3e}, max={self.Dinv.max().item():.3e}")

        def _equilibrate_level(lvl: GeoLevel):  # noqa: D401
            """Нормирует матрицу уровня: A ← S A S, diag(A)=1, возвращает S."""
            diag = lvl.diag
            S_full = 1.0 / torch.sqrt(diag.clamp_min(1e-20))
            S = torch.sqrt(S_full)  # симметричное: diag^{-1/4}

            crow = lvl.A_csr.crow_indices()
            col = lvl.A_csr.col_indices()
            vals = lvl.A_csr.values()
            row_counts = crow[1:] - crow[:-1]
            row_idx = torch.repeat_interleave(torch.arange(crow.numel()-1, device=self.device), row_counts)
            vals.mul_(S[row_idx] * S[col])  # S A S
            diag_idx = crow[1:] - 1
            diag_new = vals[diag_idx].abs().clone()
            lvl.diag = diag_new
            return S

        # ---------------- Hierarchy of coarser grids -----------------
        # Нормируем базовый уровень (уже сделано выше вручную)
        self.levels: List[GeoLevel] = [base_lvl]


        target_coarse = max(base_lvl.n_cells // max_coarse_ratio, 2000)

        # Для построения грубых уровней используем исходные проницаемости
        kx_c, ky_c, kz_c = kx, ky, kz
        hx, hy, hz = self.hx, self.hy, self.hz

        def pool(t):
            return F.avg_pool3d(t[None, None, ...], kernel_size=2, stride=2, padding=0)[0, 0]

        while kx_c.numel() > target_coarse and min(kx_c.shape) >= 4:
            kx_c = pool(kx_c); ky_c = pool(ky_c); kz_c = pool(kz_c)
            hx *= 2.0; hy *= 2.0; hz *= 2.0
            lvl = GeoLevel(kx_c, ky_c, kz_c, hx, hy, hz, device=self.device)
            # эквилибрируем каждый новый уровень
            _equilibrate_level(lvl)
            print(f"[GeoSolverV2] built level {len(self.levels)}: n={lvl.n_cells}")
            self.levels.append(lvl)

    # ------------------------------------------------------------------
    def _apply_A(self, lvl_idx: int, x: torch.Tensor) -> torch.Tensor:
        lvl = self.levels[lvl_idx]
        Ax = torch.sparse.mm(lvl.A_csr, x.view(-1, 1)).squeeze(1)
        return Ax

    def _jacobi(self, lvl_idx: int, x: torch.Tensor, b: torch.Tensor, iters: int = 2) -> torch.Tensor:
        lvl = self.levels[lvl_idx]
        # -------- L1-Jacobi диагональный суррогат --------
        # --- L1-Jacobi: denom = Σ_j |A_ij| -------------------------------
        crow = lvl.A_csr.crow_indices()
        row_counts = crow[1:] - crow[:-1]
        row_idx = torch.repeat_interleave(torch.arange(crow.numel()-1, device=self.device), row_counts)
        abs_vals = lvl.A_csr.values().abs()
        row_abs_sum = torch.zeros_like(lvl.diag)
        row_abs_sum.index_add_(0, row_idx, abs_vals)

        denom = row_abs_sum.clamp_min(1e-12)
        inv_diag = 1.0 / denom
        omega = self.omega_fine if lvl_idx == 0 else self.omega
        init_clip = self.delta_clip_factor * (b.abs().max().item() + 1e-12)
        for _ in range(iters):
            r = b - self._apply_A(lvl_idx, x)
            x = x + omega * inv_diag * r

            # --- динамический clip только на самом тонком уровне ---
            if lvl_idx == 0 and self.delta_clip_factor is not None:
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
        crow = lvl.A_csr.crow_indices()
        row_counts = crow[1:] - crow[:-1]
        row_idx = torch.repeat_interleave(torch.arange(crow.numel()-1, device=self.device), row_counts)
        abs_vals = lvl.A_csr.values().abs()
        row_abs_sum = torch.zeros_like(lvl.diag)
        row_abs_sum.index_add_(0, row_idx, abs_vals)
        inv_diag = 1.0 / row_abs_sum.clamp_min(1e-12)
        omega = self.omega_fine if lvl_idx == 0 else self.omega

        red_mask = lvl.is_red
        black_mask = lvl.is_black

        init_clip = self.delta_clip_factor * (b.abs().max().item() + 1e-12)

        for _ in range(iters):
            # red sweep (in-place)
            r = b - self._apply_A(lvl_idx, x)
            x[red_mask] += omega * inv_diag[red_mask] * r[red_mask]

            # black sweep (in-place, с учётом уже обновлённого красного цвета)
            r = b - self._apply_A(lvl_idx, x)
            x[black_mask] += omega * inv_diag[black_mask] * r[black_mask]

            # динамический clip только на самом тонком уровне
            if lvl_idx == 0 and self.delta_clip_factor is not None:
                dyn_clip = self.clip_kappa * (x.abs().max().item() + 1e-12)
                clip_val = max(init_clip, dyn_clip)
                torch.clamp_(x, -clip_val, clip_val)
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
        """Volume-weighted restriction 2×2×2 → coarse grid."""
        return F.avg_pool3d(vol3d[None, None, ...], kernel_size=2, stride=2, padding=0)[0, 0]

    def _prolong3d(self, coarse3d: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        """Linear prolongation to fine grid size."""
        return F.interpolate(coarse3d[None, None, ...], size=target_shape, mode="trilinear", align_corners=False)[0, 0]

    def _v_cycle(self, lvl_idx: int, x_vec: torch.Tensor, b_vec: torch.Tensor) -> torch.Tensor:
        """Recursive V-/W-cycle in 3-D form, input / output вектор (flattened)."""
        lvl = self.levels[lvl_idx]
        # Coarsest grid: 30 Jacobi
        if lvl_idx == len(self.levels) - 1 or lvl.n_cells < 64:
            return self._jacobi(lvl_idx, x_vec, b_vec, iters=30)

        nz, ny, nx = lvl.kx.shape
        x3d = x_vec.reshape(nz, ny, nx)
        b3d = b_vec.reshape_as(x3d)

        # pre-smooth
        if lvl_idx == 0 and self.smoother_fine == "linez":
            x_vec = self._line_gs_z(lvl_idx, x_vec, b_vec, iters=self.pre_smooth)
        elif lvl_idx == 0 and self.smoother_fine == "rbgs":
            x_vec = self._rb_gs(lvl_idx, x_vec, b_vec, iters=self.pre_smooth)
        elif lvl_idx == 0 and self.smoother_fine == "chebyshev":
            x_vec = self._chebyshev(lvl_idx, x_vec, b_vec, iters=self.pre_smooth)
        else:
            x_vec = self._jacobi(lvl_idx, x_vec, b_vec, iters=self.pre_smooth)
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

        # prolongate error
        e3d = self._prolong3d(x_c3d, x3d.shape)
        x3d = x3d + e3d

        # post-smooth
        if lvl_idx == 0 and self.smoother_fine == "linez":
            x_vec = self._line_gs_z(lvl_idx, x3d.reshape(-1), b_vec, iters=self.post_smooth)
        elif lvl_idx == 0 and self.smoother_fine == "rbgs":
            x_vec = self._rb_gs(lvl_idx, x3d.reshape(-1), b_vec, iters=self.post_smooth)
        elif lvl_idx == 0 and self.smoother_fine == "chebyshev":
            x_vec = self._chebyshev(lvl_idx, x3d.reshape(-1), b_vec, iters=self.post_smooth)
        else:
            x_vec = self._jacobi(lvl_idx, x3d.reshape(-1), b_vec, iters=self.post_smooth)
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

        # эквилибрируем RHS: rhs̃ = S⁻¹ rhs
        rhs_hat = rhs * self.Dinv
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
        # Для согласованности с CPR возвращаем физическое решение, т.е.
        # снимаем эквилибрирующий масштаб: δ_phys = x / S,  где S = self.Dinv.
        delta_phys = x * self.Dinv
        if torch.isfinite(delta_phys).all():
            print(f"[GeoSolverV2] DEBUG solve: ||rhs_hat||={rhs_hat.norm().item():.3e}, ||x||={x.norm().item():.3e}, ||delta_phys||={delta_phys.norm().item():.3e}")
        else:
            print("[GeoSolverV2] DEBUG solve: NaN/Inf in delta_phys")
        return delta_phys.cpu().numpy() 