#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, argparse, random, json
import torch
import numpy as np

# Отключаем тестовые патчи trans_patch для «боевых» прогонов бенчмарка
os.environ.setdefault("OIL_SIM_SKIP_PATCHES", "1")

# путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator
from solver.cpr import CPRPreconditioner  # если вдруг требуется где-то внутри

# bench/bench.py
from tools.diag_jfnk_mg import (
    run_jfnk_diagnostics, run_mg_diagnostics, mg_cap_levels,
    Jv_fd_consistent
)


# --------------------------------------------------
# Генерация случайных скважин
# --------------------------------------------------
def make_wells(nx, ny, nz, *, n_wells=50,
               inj_fraction=0.3,
               inj_rate=8e4,
               prod_rate=-6e4):
    wells = []
    n_inj = int(n_wells * inj_fraction)
    n_prod = n_wells - n_inj

    def rnd(dim):  # избегаем границ
        return random.randint(1, max(dim - 2, 1))

    for i in range(n_inj):
        wells.append(dict(
            name=f"INJ{i+1}", type="injector",
            i=rnd(nx), j=rnd(ny), k=rnd(nz),
            radius={"radius": 0.1, "well_index": 1e-3},
            control_type="bhp", control_value=30.0  # МПа
        ))
    for i in range(n_prod):
        wells.append(dict(
            name=f"PROD{i+1}", type="producer",
            i=rnd(nx), j=rnd(ny), k=rnd(nz),
            radius={"radius": 0.1, "well_index": 1e-3},
            control_type="bhp", control_value=10.0  # МПа
        ))
    return wells


def bench_case(nx, ny, nz, *,
               mode="fi",
               steps=1,
               newton_max=1,
               jfnk_max=5,
               geo_cycles=1,
               geo_pre=2,
               geo_post=2,
               geo_levels=6,
               dt_days=0.02,
               debug=True,
               config_path: str | None = None):
    n_cells = nx * ny * nz
    print(f"\n===== Case {nx}x{ny}x{nz}  (N={n_cells/1e6:.2f} M) mode={mode} =====")

    # Если задан путь к конфигу — используем его как единый источник правды
    cfg = None
    if config_path:
        with open(config_path, 'r') as _f:
            cfg = json.load(_f)
        # bench не включает принудительный отладочный лог, если есть конфиг
    elif debug:
        os.environ["OIL_DEBUG"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Reservoir / Fluid / Wells ----------------------------------------
    if cfg is not None:
        res = Reservoir(cfg['reservoir'], device=device)
        fluid = Fluid(cfg['fluid'], res, device=device)
        wells = WellManager(cfg.get('wells', []), res)
    else:
        res_cfg = {
            "dimensions": [nx, ny, nz],
            "grid_size": [20.0, 20.0, 5.0],
            "permeability": 100.0,
            "k_vertical_fraction": 0.1,
            "porosity": 0.2,
            "c_rock": 1e-5,
            "pressure_ref": 20.0e6,
        }
        res = Reservoir(res_cfg, device=device)

        fluid_cfg = {
            "pressure": 20.0,
            "s_w": 0.2,
            "mu_oil": 1.0,
            "mu_water": 0.5,
            "mu_gas": 0.05,
            "rho_oil": 850.0,
            "rho_water": 1000.0,
            "rho_gas": 150.0,
        }
        fluid = Fluid(fluid_cfg, res, device=device)

        wells = WellManager(
            make_wells(nx, ny, nz, n_wells=50, inj_fraction=0.3,
                       inj_rate=8e4, prod_rate=-6e4),
            res
        )

    if cfg is not None:
        sim_params = cfg.get('simulation', {}).copy()
        # Если в конфиге не задано время, подставим из CLI для совместимости бенча
        sim_params.setdefault("time_step_days", dt_days)
        sim_params.setdefault("total_time_days", steps * 1.0)
        sim_params.setdefault("solver_type", "fully_implicit" if mode == "fi" else "impes")
        sim_params.setdefault("use_cuda", device.type == "cuda")
    else:
        if mode == "fi":
            sim_params = {
                "solver_type": "fully_implicit",
                "jacobian": "jfnk",
                "backend": "geo2",
                "time_step_days": dt_days,
                "adaptive_dt": True,
                "total_time_days": steps * 1.0,
                "verbose": True,
                "use_cuda": device.type == "cuda",
                # лимиты итераций и толерансы
                "newton_max_iter": int(newton_max),
                "jfnk_max_iter": int(jfnk_max),
                "newton_tolerance": 1e-8,
                "newton_rtol": 1e-8,
                # Geo-AMG
                "geo_cycles": geo_cycles,
                "geo_pre": geo_pre,
                "geo_post": geo_post,
                "geo_levels": geo_levels,
                # безопасность
                "line_search_min_alpha": 0.0002,
                "ptc": "always",
                "advanced_threshold": 1_000_000_000,
                "p_step_max": 1e9,
                "delta_y_max": 5.0,
            }
        else:
            sim_params = {
                "solver_type": "impes",
                "total_time_days": steps * 1.0,
                "time_step_days": 1.0,
                "verbose": True,
                "use_cuda": device.type == "cuda",
            }

    # Если конфиг задан — НЕ переопределяем его параметрами CLI, кроме логов при явном запросе
    if cfg is None:
        sim_params["cpr_backend"]   = args.cpr_backend
        sim_params["geo_tol"]       = args.geo_tol
        sim_params["geo_max_iter"]  = args.geo_max_iter
        sim_params["gmres_tol"]     = args.gmres_tol
        sim_params["gmres_max_iter"]= args.gmres_max_iter
        sim_params["geo_cycles"]    = args.geo_cycles
        sim_params["geo_pre"]       = args.geo_pre
        sim_params["geo_post"]      = args.geo_post
        sim_params["geo_levels"]    = args.geo_levels
    if args.log_json_dir:
        sim_params["log_json_dir"] = args.log_json_dir

    # --- Хуки/обёртки для диагностики и экспериментов -----------------------
    # 1) FD-консистентная замена Jv (можно включать/выключать без правок Solver)
    def _override_Jv_factory(F_hat, x_hat, project, l_hat, u_hat):
        # eps0 можно подобрать автоматом через eta-scan; пока стартуем с 1e-7
        eps0 = float(os.getenv("FD_EPS0", "1e-7"))
        return lambda v_hat: Jv_fd_consistent(
            F_hat, x_hat, v_hat, project=project, l_hat=l_hat, u_hat=u_hat, eps0=eps0
        )

    # 2) Диагностика JFNK (eta-scan + сверка линейной модели)
    def _hook_before_gmres(sim, dt_sec, newton_iter, x_hat, F_hat, project, l_hat, u_hat):
        if int(os.getenv("JFNK_DIAG", "1")):
            run_jfnk_diagnostics(F_hat, x_hat, project=project, l_hat=l_hat, u_hat=u_hat)

    # 3) Диагностика multigrid (цепочка рестрикции, RAP, совет по числу уровней)
    def _hook_after_mg_build(sim, mg_obj, rhs_vec):
        if int(os.getenv("MG_DIAG", "1")):
            run_mg_diagnostics(mg_obj, rhs_vec)
        if int(os.getenv("MG_CAP", "0")):
            suggested = mg_cap_levels(mg_obj, min_dim=8)
            # если у твоего MG есть метод ограничения уровней — используем:
            if hasattr(mg_obj, "set_max_levels"):
                mg_obj.set_max_levels(suggested)


    sim = Simulator(res, fluid, wells, sim_params, device=device)

    sim_params["hook_before_gmres"]  = _hook_before_gmres
    sim_params["hook_after_mg_build"] = _hook_after_mg_build
    sim_params["override_Jv_factory"] = _override_Jv_factory  # по умолчанию Solver может игнорировать


    dt_sec = sim_params["time_step_days"] * 86400.0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for step in range(steps):
        ok = sim.run_step(dt_sec)
        if not ok:
            print(f"⚠️  Шаг {step+1} не сошёлся – прерываем кейс")
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"{step+1} шаг(ов) выполнено, elapsed={time.time()-t0:.2f}s, GPU={device.type=='cuda'}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Быстрый бенч для GeoSolverV2 с подробными логами."
    )
    ap.add_argument("--nx", type=int, default=60)
    ap.add_argument("--ny", type=int, default=60)
    ap.add_argument("--nz", type=int, default=30)
    ap.add_argument("--mode", choices=["fi", "impes"], default="fi")
    ap.add_argument("--steps", type=int, default=1)

    ap.add_argument("--newton", type=int, default=1, help="max_newton_iter")
    ap.add_argument("--jfnk", type=int, default=5, help="jfnk_max_iter")

    ap.add_argument("--geo_cycles", type=int, default=1)
    ap.add_argument("--geo_pre", type=int, default=2)
    ap.add_argument("--geo_post", type=int, default=2)
    ap.add_argument("--geo_levels", type=int, default=6)

    ap.add_argument("--dt", type=float, default=0.02, help="начальный шаг (сутки)")
    ap.add_argument("--debug", action="store_true", default=True)
    ap.add_argument("--config", type=str, default="", help="путь к JSON-конфигу (если задан, CLI-параметры не переопределяют конфиг)")
    ap.add_argument('--log-json-dir', type=str, default='', help='каталог для JSON-логов (per-итерация/шаг)')
    ap.add_argument('--cpr-backend', default='geo2')
    ap.add_argument('--geo-tol', type=float, default=1e-6)
    ap.add_argument('--geo-max-iter', type=int, default=10)
    ap.add_argument('--gmres-tol', type=float, default=1e-4)
    ap.add_argument('--gmres-max-iter', type=int, default=300)



    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bench_case(args.nx, args.ny, args.nz,
               mode=args.mode,
               steps=args.steps,
               newton_max=args.newton,
               jfnk_max=args.jfnk,
               geo_cycles=args.geo_cycles,
               geo_pre=args.geo_pre,
               geo_post=args.geo_post,
               geo_levels=args.geo_levels,
               dt_days=args.dt,
               debug=args.debug,
               config_path=(args.config or None))
