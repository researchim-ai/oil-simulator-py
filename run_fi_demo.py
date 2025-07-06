#!/usr/bin/env python3
"""
run_fi_demo.py – мини-демо для полноценного полностью-неявного решателя.

По умолчанию использует конфиг `configs/fi_demo_2d.json` и делает 100 шагов.
Все ключевые величины (шаг, среднее давление, средняя Sw, итерации Ньютона/GMRES)
выводятся в консоль, так что можно глазами отследить процесс.

Запуск:
    python run_fi_demo.py                  # 100 шагов, CPU
    python run_fi_demo.py --steps 200      # другое число шагов
    python run_fi_demo.py --gpu            # если CUDA доступна
    python run_fi_demo.py --config configs/fully_implicit_2d.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# -----------------------------------------------------------------------------
# Local imports (путь src/ добавляем вручную, чтобы скрипт работал из корня)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator import simulation as _sim_mod

# --- undo test-time monkey patches if present ------------------------------
if hasattr(_sim_mod, '_original_fi_step'):
    _sim_mod.Simulator._fully_implicit_step = _sim_mod._original_fi_step  # type: ignore
if hasattr(_sim_mod.Simulator, '_impes_saturation_step') and hasattr(_sim_mod, '_def_impes_sat'):
    _sim_mod.Simulator._impes_saturation_step = _sim_mod._def_impes_sat  # type: ignore

def parse_args():
    p = argparse.ArgumentParser(description="Fully-implicit demo runner")
    p.add_argument("--config", default="configs/fi_demo_2d.json",
                   help="Путь к JSON-конфигу")
    p.add_argument("--steps", type=int, default=100,
                   help="Сколько шагов выполнить")
    p.add_argument("--gpu", action="store_true", help="Использовать CUDA, если доступна")
    return p.parse_args()


def main():
    args = parse_args()

    # ─── Загрузка конфига ────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())

    # ─── Устройство ─────────────────────────────────────────────────────────
    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"🖥  Using device: {device}")

    # ─── Инициализация объектов ─────────────────────────────────────────────
    reservoir = Reservoir(cfg["reservoir"], device)
    fluid     = Fluid(cfg["fluid"], reservoir, device)
    wells     = WellManager(cfg["wells"], reservoir)

    sim_params = {**cfg["simulation"], "use_cuda": device.type == "cuda", "verbose": True}

    print("\n🔧 Building Simulator …", flush=True)
    sim        = _sim_mod.Simulator(reservoir, fluid, wells, sim_params, device)

    dt = sim_params.get("time_step_days", 0.1) * 86400.0
    print(f"dt = {dt} s", flush=True)

    print("Starting fully-implicit run …\n", flush=True)
    header = f"{'step':>4s} | {'P̄ (MPa)':>8s} | {'Sw̄':>6s} | {'Newton':>6s} | {'GMRES':>6s}"
    print(header)
    print("-" * len(header))

    for n in range(args.steps):
        print(f"-- step {n} --", flush=True)
        try:
            ok = sim.run_step(dt)
        except Exception as e:
            import traceback
            print("❌ Exception during run_step:\n", traceback.format_exc())
            break
        if not ok:
            print(f"⚠️  Solver failed at step {n}")
            break

        P_mean = float(torch.mean(fluid.pressure) / 1e6)
        Sw_mean = float(torch.mean(fluid.s_w))
        newt_it = getattr(sim._fisolver, "last_newton_iters", 0) if hasattr(sim, "_fisolver") else -1
        gm_it   = getattr(sim._fisolver, "last_gmres_iters", 0) if hasattr(sim, "_fisolver") else -1

        print(f"{n:4d} | {P_mean:8.2f} | {Sw_mean:6.3f} | {newt_it:6d} | {gm_it:6d}")

    print("\n✅ Run completed. Final stats:")
    print(f"   mean P  = {float(torch.mean(fluid.pressure)/1e6):.2f} MPa")
    print(f"   mean Sw = {float(torch.mean(fluid.s_w)):.3f}")


if __name__ == "__main__":
    main() 