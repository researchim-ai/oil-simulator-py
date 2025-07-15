#!/usr/bin/env python3
"""
fi_debug_small.py – компактный отладочный лаунчер полностью-неявной схемы
для микромодели 3×3×1 (9 ячеек). Скрипт показывает ВСЕ ключевые метрики:
  • номер шага Ньютона, ||F|| и ||F||_scaled
  • параметры GMRES (tol, restart, max_iter)
  • итоговую информацию GMRES (info, итерации, ||delta||)
  • итог: сходился/не сходился, изменение давления/насыщенности

Запуск:
    python fi_debug_small.py [--dt 3600] [--max-it 30]

По умолчанию выполняется один шаг FI. Для расширенного анализа можно
включить --steps N и смотреть динамику между шагами.
"""
import argparse, sys, os, json, torch
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator

parser = argparse.ArgumentParser(description="Debug small fully-implicit run (3×3×1)")
parser.add_argument('--dt', type=float, default=3600.0, help='Шаг по времени, сек')
parser.add_argument('--max-it', type=int, default=30, help='Максимум итераций Ньютона')
parser.add_argument('--steps', type=int, default=1, help='Сколько временных шагов выполнить')
args = parser.parse_args()

device = torch.device('cpu')

# ── Параметры модели ───────────────────────────────────────────────────────────
res_cfg = {
    "dimensions": [3, 3, 1],
    "grid_size": [1.0, 1.0, 1.0],
    "porosity": 0.25,
    "permeability": 100.0,   # mD
    "k_vertical_fraction": 1.0,
}

fluid_cfg = {
    "mu_water": 1.0,   # cP
    "mu_oil": 10.0,    # cP
    "rho_water_ref": 1000.0,
    "rho_oil_ref": 800.0,
    "initial_pressure": 1e5,  # 0.1 МПа
    "initial_sw": 0.2,
    "relative_permeability": {"nw": 2, "no": 2},
    "pc_scale": 0.0,
}

well_cfgs = [
    {"name": "INJ", "type": "injector", "i": 0, "j": 0, "k": 0,
     "radius": 0.1, "control_type": "rate", "control_value": 100.0},
    {"name": "PROD", "type": "producer", "i": 2, "j": 2, "k": 0,
     "radius": 0.1, "control_type": "bhp", "control_value": 0.1},
]

sim_params = {
    "solver_type": "fully_implicit",
    "jacobian": "jfnk",
    "newton_tolerance": 1e-6,
    "newton_max_iter": args.max_it,
    "verbose": True,          # важен детальный вывод
    "gmres_min_tol": 1e-8,
}

reservoir = Reservoir(res_cfg, device)
fluid = Fluid(fluid_cfg, reservoir, device)
well_manager = WellManager(well_cfgs, reservoir)

sim = Simulator(reservoir, fluid, well_manager, sim_params, device)

print("\n================ RUN =================")
for step in range(args.steps):
    print(f"\n--- Шаг {step+1}/{args.steps} dt={args.dt/3600:.2f} ч ---")
    prev_p = fluid.pressure.clone()
    prev_sw = fluid.s_w.clone()
    ok = sim.run_step(dt=args.dt)
    print(f"Результат: {'успех' if ok else 'НЕ сошлось'}")
    # Статистика
    if hasattr(sim, '_fisolver'):
        its = getattr(sim._fisolver, 'last_newton_iters', None)
        gm = getattr(sim._fisolver, 'last_gmres_iters', None)
        print(f"Итерации: Newton={its}, GMRES(total)={gm}")
    p_change = fluid.pressure - prev_p
    sw_change = fluid.s_w - prev_sw
    print(f"ΔP (min/mean/max): {p_change.min():.3e}/{p_change.mean():.3e}/{p_change.max():.3e} Pa")
    print(f"ΔSw (min/mean/max): {sw_change.min():.3e}/{sw_change.mean():.3e}/{sw_change.max():.3e}")
    p_std = torch.std(p_change).item()
    if p_std < 1e-2:
        print("⚠️  Давление практически не изменилось — возможная причина: trust-region или демпфирование")
    print(f"σ(Δдавление) = {p_std:.3e} Па")

print("\n================ DONE ================") 