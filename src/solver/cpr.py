# -*- coding: utf-8 -*-
"""
CPR(2-stage) предобуславливатель, совместимый с твоим текущим jfnk.py.
Внутри он пытается построить GeoSolverV2 по твоей (показанной) сигнатуре.
Если нужные объекты в simulator.linops отсутствуют — Stage-1 делает Jacobi=I.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, Callable, List
import torch

from .geo_solver_v2 import GeoSolverV2, LevelData  # твоя реализация

__all__ = ["CPRPreconditioner"]


class CPRPreconditioner:
    """
    Двухступенчатый CPR:

      Stage 1 (pressure):  dp = M_p^{-1} * rhs_p  — через GeoSolverV2 (если есть уровни и операторы).
      Stage 2 (saturation): ds ≈ rhs_s - Aps * dp  (Jacobi=I по умолчанию) + кламп.

    Ожидается порядок неизвестных: [p(0..N-1), Sw(0..N-1)].
    Если у тебя 3 фазы — замени split/merge на свои.
    """

    def __init__(
        self,
        simulator,
        backend: str = "geo2",
        smoother: str = "rbgs",
        scaler=None,
        geo_params: Optional[Dict[str, Any]] = None,
        geo_tol: float = 1e-6,
        geo_max_iter: int = 10,
        gmres_tol: float = 1e-3,           # оставляем, чтобы не ломать сигнатуру
        gmres_max_iter: int = 60,          # оставляем, чтобы не ломать сигнатуру
        **kwargs,
    ):
        self.sim = simulator
        self.scaler = scaler
        self.backend = backend
        self.smoother = smoother
        self.geo_params = geo_params or {}
        self.geo_tol = geo_tol
        self.geo_max_iter = geo_max_iter

        # число ячеек
        self.n_cells = (
            scaler.n_cells if scaler is not None
            else simulator.reservoir.dimensions[0]
               * simulator.reservoir.dimensions[1]
               * simulator.reservoir.dimensions[2]
        )

        sp = simulator.sim_params
        self.debug: bool = bool(sp.get("cpr_debug", False))
        self.cycles_stage1: int = int(sp.get("geo_cycles", self.geo_params.get("cycles_per_call", 1)))
        self.max_sat_correction: float = float(sp.get("max_sat_corr", 5e-3))

        # опциональные пользовательские колбэки
        self.Aps_times: Optional[Callable[[torch.Tensor], torch.Tensor]] = kwargs.get("Aps_times", None)
        self.solve_sat_cb: Optional[Callable[[torch.Tensor], torch.Tensor]] = kwargs.get("solve_sat", None)

        # split/merge под 2 переменные на ячейку
        self.split_fn = self._split_2vars
        self.merge_fn = self._merge_2vars

        # Попробуем собрать GeoSolverV2. Если не сможем — Stage-1 будет Jacobi=I.
        self.geo_solver: Optional[GeoSolverV2] = None
        self._build_geo_solver_safe()

        if self.debug:
            print(f"[CPR] geo_solver={'ON' if self.geo_solver is not None else 'OFF (Jacobi-I fallback)'}; "
                  f"cycles_stage1={self.cycles_stage1}")

    # ------------------------------------------------------------------ #
    #                              PUBLIC                                #
    # ------------------------------------------------------------------ #
    def apply(self, rhs_phys: torch.Tensor) -> torch.Tensor:
        """delta_phys ≈ M^{-1} rhs_phys (всё в физических единицах)."""
        rhs_p, rhs_s = self.split_fn(rhs_phys)

        # Stage-1: давление
        dp = self._solve_pressure(rhs_p)

        # Stage-2: насыщенности
        if self.Aps_times is not None:
            rhs_s_eff = rhs_s - self.Aps_times(dp)
        else:
            rhs_s_eff = rhs_s

        # нормировка rhs_s для устойчивости последнего шага
        inf_s = rhs_s_eff.abs().max().item()
        scale_s = 1.0 / (inf_s if inf_s > 0.0 and torch.isfinite(torch.tensor(inf_s)) else 1.0)
        rhs_s_scaled = rhs_s_eff * scale_s

        if self.solve_sat_cb is not None:
            ds_scaled = self.solve_sat_cb(rhs_s_scaled)
        else:
            ds_scaled = rhs_s_scaled  # Jacobi ~ I

        ds = ds_scaled * (1.0 / scale_s)

        if self.max_sat_correction > 0.0:
            torch.clamp_(ds, -self.max_sat_correction, self.max_sat_correction)

        if self.debug:
            print(f"[CPR] ||rhs_p||={rhs_p.norm():.3e}, ||rhs_s||={rhs_s.norm():.3e}, "
                  f"||dp||={dp.norm():.3e}, ||ds||={ds.norm():.3e}, max|ds|={ds.abs().max().item():.3e}")

        return self.merge_fn(dp, ds)

    # ------------------------------------------------------------------ #
    #                         PRESSURE STAGE                              #
    # ------------------------------------------------------------------ #
    def _solve_pressure(self, rhs_p: torch.Tensor) -> torch.Tensor:
        if self.geo_solver is None:
            # Jacobi ~ I
            return rhs_p.clone()

        # GeoSolverV2.apply_prec_phys(rhs, cycles)
        try:
            return self.geo_solver.apply_prec_phys(rhs_p, cycles=self.cycles_stage1)
        except TypeError:
            # вдруг сигнатура без cycles
            return self.geo_solver.apply_prec_phys(rhs_p)

    # ------------------------------------------------------------------ #
    #                          GEO SOLVER BUILD                           #
    # ------------------------------------------------------------------ #
    def _build_geo_solver_safe(self):
        """
        Пытаемся вытащить из simulator.linops всё, что нужно для GeoSolverV2.
        Если чего-то не хватает — выходим в fallback (Jacobi=I).
        """
        linops = getattr(self.sim, "linops", None)
        if linops is None:
            return

        # ---- обязательные вещи ----
        levels: Optional[List[LevelData]] = getattr(linops, "geo_levels", None)
        apply_A = getattr(linops, "apply_A_level", None)
        restrict_op = getattr(linops, "restrict_level", None)
        prolong_op = getattr(linops, "prolong_level", None)

        if levels is None or apply_A is None or restrict_op is None or prolong_op is None:
            # ничего не строим
            if self.debug:
                print("[CPR] linops.{geo_levels,apply_A_level,restrict_level,prolong_level} не найдены → Jacobi=I")
            return

        # ---- опциональные вещи ----
        W_rows = getattr(linops, "W_rows", None)
        if W_rows is None:
            # единичная левая эквилибрация
            n = levels[0].n
            W_rows = torch.ones(n, dtype=torch.float64, device=getattr(self.sim, "device", torch.device("cpu")))

        # скейлеры давления (можешь поменять на свои)
        to_hat = getattr(linops, "to_hat_p", None)
        to_phys = getattr(linops, "to_phys_p", None)
        if to_hat is None or to_phys is None:
            # делаем identity
            def to_hat(x: torch.Tensor) -> torch.Tensor:  # type: ignore
                return x
            def to_phys(x: torch.Tensor) -> torch.Tensor:  # type: ignore
                return x

        device = getattr(self.sim, "device", torch.device("cpu"))

        # Параметры для GeoSolverV2
        geo_pre = int(self.geo_params.get("pre_smooth", self.sim.sim_params.get("geo_pre", 2)))
        geo_post = int(self.geo_params.get("post_smooth", self.sim.sim_params.get("geo_post", 2)))
        rbgs_fine = int(self.geo_params.get("rbgs_iters_fine", 2))
        rbgs_coarse = int(self.geo_params.get("rbgs_iters_coarse", 2))
        omega0_f = float(self.geo_params.get("omega0_fine", 0.3))
        omega_bounds = tuple(self.geo_params.get("omega_bounds", (0.05, 0.9)))
        clip_kappa = float(self.geo_params.get("clip_kappa", 2.0))
        delta_clip_factor = float(self.geo_params.get("delta_clip_factor", 1.0))
        geo_debug = bool(self.sim.sim_params.get("geo_debug", False))

        try:
            self.geo_solver = GeoSolverV2(
                levels=levels,
                W_rows=W_rows,
                to_hat=to_hat,
                to_phys=to_phys,
                apply_A=apply_A,
                restrict_op=restrict_op,
                prolong_op=prolong_op,
                device=device,
                debug=geo_debug,
                geo_pre=geo_pre,
                geo_post=geo_post,
                geo_max_iter=self.geo_max_iter,
                geo_tol=self.geo_tol,
                rbgs_iters_fine=rbgs_fine,
                rbgs_iters_coarse=rbgs_coarse,
                omega0_fine=omega0_f,
                omega_bounds=omega_bounds,
                clip_kappa=clip_kappa,
                delta_clip_factor=delta_clip_factor,
            )
        except TypeError as e:
            # если вдруг сигнатура всё-таки другая — не падаем, просто идём в Jacobi=I
            if self.debug:
                print(f"[CPR] Не удалось сконструировать GeoSolverV2: {e} → Jacobi=I")
            self.geo_solver = None

    # ------------------------------------------------------------------ #
    #                     SPLIT/MERGE (2 variables/cell)                  #
    # ------------------------------------------------------------------ #
    def _split_2vars(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.n_cells
        p = v[:n]
        s = v[n:2 * n] if v.numel() >= 2 * n else torch.zeros_like(p)
        return p, s

    def _merge_2vars(self, dp: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
        if ds.numel() == 0:
            return dp
        return torch.cat([dp, ds], dim=0)
