# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch


# ------------------------------------------------------------
#                   УТИЛИТЫ
# ------------------------------------------------------------

def safe_norm2(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    v = torch.linalg.norm(x.double())
    if not torch.isfinite(v):
        return float("inf")
    return float(v.item())


def clamp_inplace(x: torch.Tensor, lo: float, hi: float):
    torch.clamp_(x, lo, hi)


# ------------------------------------------------------------
#                   ДАННЫЕ УРОВНЯ МГС
# ------------------------------------------------------------

@dataclass
class LevelData:
    n: int
    inv_l1: torch.Tensor            # 1 / диагональ или 1 / L1-норма строки
    is_red: torch.Tensor
    is_black: torch.Tensor
    R: Optional[torch.Tensor] = None
    P: Optional[torch.Tensor] = None
    anchor_mask: Optional[torch.Tensor] = None


# ------------------------------------------------------------
#                   GEO SOLVER V2 (V-cycle)
# ------------------------------------------------------------

class GeoSolverV2:
    """
    Устойчивый мультигрид (V-cycle) для pressure-блока в hat-пространстве.

    Все матричные операции прокидываются снаружи через коллбеки:
      * apply_A(lvl_idx, x)      → A_lvl * x
      * restrict(lvl_idx, r)     → ограничение
      * prolong(lvl_idx, e_c)    → продление

    Вызов внешне:
        x_phys = solver.apply_prec_phys(rhs_phys, cycles=2)

    где to_hat / to_phys превращают физ. вектор в безразмерный и обратно,
    а W_rows – левая эквилибрация строк (1/||row|| или diag).
    """

    def __init__(
        self,
        levels: List[LevelData],
        W_rows: torch.Tensor,
        to_hat: Callable[[torch.Tensor], torch.Tensor],
        to_phys: Callable[[torch.Tensor], torch.Tensor],
        apply_A: Callable[[int, torch.Tensor], torch.Tensor],
        restrict_op: Callable[[int, torch.Tensor], torch.Tensor],
        prolong_op: Callable[[int, torch.Tensor], torch.Tensor],
        device: torch.device,
        debug: bool = False,
        # smooth / cycle params
        geo_pre: int = 3,
        geo_post: int = 3,
        geo_max_iter: int = 15,
        geo_tol: float = 1e-6,
        rbgs_iters_fine: int = 2,
        rbgs_iters_coarse: int = 2,
        # omega control
        omega0_fine: float = 0.3,
        omega_bounds: Tuple[float, float] = (0.05, 0.9),
        # clipping
        clip_kappa: float = 2.0,
        delta_clip_factor: float = 1.0,
    ):
        self.levels = levels
        self.L = len(levels)
        self.device = device
        self.debug = debug

        self.W_rows = W_rows.to(device=device, dtype=torch.float64)
        self._to_hat = to_hat
        self._to_phys = to_phys

        # callbacks
        self._apply_A_cb = apply_A
        self._restrict_cb = restrict_op
        self._prolong_cb = prolong_op

        # multigrid params
        self.geo_pre = geo_pre
        self.geo_post = geo_post
        self.geo_max_iter = geo_max_iter
        self.geo_tol = geo_tol
        self.rbgs_iters_fine = rbgs_iters_fine
        self.rbgs_iters_coarse = rbgs_iters_coarse

        # omega
        self._omega_fine0 = omega0_fine
        self.omega_fine_min, self.omega_fine_max = omega_bounds
        self.omega_fine = omega0_fine
        self.omega = 0.5  # на грубых уровнях фикс.

        # clipping
        self.clip_kappa = clip_kappa
        self.delta_clip_factor = delta_clip_factor

        self.anchor_fine = (
            levels[0].anchor_mask if levels[0].anchor_mask is not None
            else torch.zeros(levels[0].n, dtype=torch.bool, device=device)
        )

    # ------------------- ПУБЛИЧНЫЙ ИНТЕРФЕЙС -------------------

    def apply_prec_phys(self, rhs_phys: torch.Tensor, cycles: int = 2) -> torch.Tensor:
        """
        Полный устойчивый вызов предобуславливателя:
          1) phys → hat
          2) левая эквилибрация W_rows
          3) нормировка по ||·||∞
          4) несколько V-cycle
          5) обратные масштабы, hat → phys
        """
        self._reset_omegas()

        rhs_hat = self._to_hat(rhs_phys.to(self.device, torch.float64))
        rhs_hat = rhs_hat * self.W_rows
        rhs_hat[self.anchor_fine] = 0.0

        rhs_inf = rhs_hat.abs().max().item()
        if not np.isfinite(rhs_inf) or rhs_inf == 0.0:
            return torch.zeros_like(rhs_phys, dtype=torch.float64, device=self.device)

        scale_rhs = 1.0 / rhs_inf
        rhs_scaled = rhs_hat * scale_rhs

        x_scaled = torch.zeros_like(rhs_scaled)

        for _ in range(max(1, cycles)):
            x_scaled = self._v_cycle(0, x_scaled, rhs_scaled)

        # обратные масштабы
        x_hat = x_scaled * (1.0 / scale_rhs)
        x_hat = x_hat / self.W_rows

        delta_phys = self._to_phys(x_hat)
        return delta_phys

    # ------------------- V-CYCLE -------------------

    def _reset_omegas(self):
        self.omega_fine = self._omega_fine0

    def _v_cycle(self, lvl_idx: int, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        lvl = self.levels[lvl_idx]

        # предсглаживание
        x = self._rb_gs(lvl_idx, x, b, iters=self.geo_pre)

        # остаток
        r = b - self._apply_A(lvl_idx, x)

        # выход по tol на fine
        if lvl_idx == 0 and safe_norm2(r) < self.geo_tol:
            return x

        # рекурсия
        if lvl_idx < self.L - 1:
            r_c = self._restrict(lvl_idx, r)
            e_c = torch.zeros_like(r_c)
            e_c = self._v_cycle(lvl_idx + 1, e_c, r_c)
            e_f = self._prolong(lvl_idx, e_c)
            x = x + e_f

        # постсглаживание
        x = self._rb_gs(lvl_idx, x, b, iters=self.geo_post)

        if lvl_idx == 0:
            self._apply_anchor(x)
        return x

    def _rb_gs(self, lvl_idx: int, x: torch.Tensor, b: torch.Tensor, iters: int = 2) -> torch.Tensor:
        lvl = self.levels[lvl_idx]
        inv_diag = lvl.inv_l1
        omega = self.omega_fine if lvl_idx == 0 else self.omega
        red_mask, black_mask = lvl.is_red, lvl.is_black

        init_clip = None
        if self.delta_clip_factor is not None:
            init_clip = self.delta_clip_factor * (b.abs().max() + 1e-12)

        for k in range(iters):
            r_before = b - self._apply_A(lvl_idx, x)

            # red
            x[red_mask] += omega * inv_diag[red_mask] * r_before[red_mask]

            # recompute
            r_mid = b - self._apply_A(lvl_idx, x)

            # black
            x[black_mask] += omega * inv_diag[black_mask] * r_mid[black_mask]

            r_after = b - self._apply_A(lvl_idx, x)

            if lvl_idx == 0:
                mu = (r_after.norm() / (r_before.norm() + 1e-30)).item()
                if mu > 0.90:
                    self.omega_fine *= 0.5
                elif mu > 0.75:
                    self.omega_fine *= 0.8
                elif mu < 0.20:
                    self.omega_fine *= 1.3
                elif mu < 0.35:
                    self.omega_fine *= 1.15

                if not np.isfinite(self.omega_fine):
                    self.omega_fine = self._omega_fine0

                self.omega_fine = float(
                    min(max(self.omega_fine, self.omega_fine_min), self.omega_fine_max)
                )
                omega = self.omega_fine

                if self.debug:
                    inc_mag = (omega * inv_diag * r_mid).abs().max().item()
                    print(
                        f"[DBG-RBGS] k={k} |ω|={omega:.3e} |r|₂={r_mid.norm().item():.3e} "
                        f"|r|∞={r_mid.abs().max().item():.3e} |invD|∞={inv_diag.max().item():.3e} "
                        f"|ω*invD*r|∞={inc_mag:.3e}"
                    )

            if init_clip is not None and lvl_idx == 0:
                dyn_clip = self.clip_kappa * (x.abs().max() + 1e-12)
                clip_val = torch.maximum(init_clip, dyn_clip)
                clamp_inplace(x, -clip_val, clip_val)

            if not torch.isfinite(x).all():
                x = torch.nan_to_num(x, 0.0, 0.0, 0.0)

        return x

    # ------------------- ОПЕРАТОРЫ/ЯКОРЬ -------------------

    def _apply_anchor(self, x: torch.Tensor):
        if self.anchor_fine is not None and self.anchor_fine.any():
            x[self.anchor_fine] = 0.0

    def _apply_A(self, lvl_idx: int, x: torch.Tensor) -> torch.Tensor:
        return self._apply_A_cb(lvl_idx, x)

    def _restrict(self, lvl_idx: int, r: torch.Tensor) -> torch.Tensor:
        return self._restrict_cb(lvl_idx, r)

    def _prolong(self, lvl_idx: int, e_c: torch.Tensor) -> torch.Tensor:
        return self._prolong_cb(lvl_idx, e_c)
