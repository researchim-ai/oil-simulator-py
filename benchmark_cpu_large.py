#!/usr/bin/env python3
"""Quick CPU benchmark for the fully-implicit JFNK solver.

Usage
-----
python benchmark_cpu_large.py --nx 100 --ny 100 --nz 100 --steps 1

It constructs a homogeneous reservoir with the given dimensions, no wells,
then runs *steps* fully-implicit time steps (default dt = 1 day) on CPU.
After completion it prints total wall-time as well as Newton and GMRES
iteration statistics gathered from the solver.

The goal is to sanity-check CPU scalability on ~1–10 million cells.
"""
from __future__ import annotations

import argparse
import time
import sys

# Ensure local src/ is on PYTHONPATH
import os
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import torch

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator


def _make_wells(pattern: str, nx: int, ny: int, nz: int):
    """Return list of well dicts according to *pattern*.

    Supported patterns:
    - none: empty list
    - corner: one injector (NW) + one producer (SE)
    - five: classic 5-spot – 1 injector center, 4 producers at corners
    """
    if pattern == "none":
        return []

    wells = []
    mid_k = nz // 2

    if pattern == "corner":
        wells = [
            {
                "name": "INJ-1",
                "type": "injector",
                "coordinates": [0, 0, mid_k],
                "radius": 0.1,
                "control": {"type": "rate", "value": 1000.0},  # m3/day
            },
            {
                "name": "PROD-1",
                "type": "producer",
                "coordinates": [nx - 1, ny - 1, mid_k],
                "radius": 0.1,
                "control": {"type": "bhp", "value": 10.0},  # MPa
            },
        ]
    elif pattern == "five":
        wells.append(
            {
                "name": "INJ-C",
                "type": "injector",
                "coordinates": [nx // 2, ny // 2, mid_k],
                "radius": 0.1,
                "control": {"type": "rate", "value": 2000.0},
            }
        )
        for name, (i, j) in [
            ("P-NW", (0, 0)),
            ("P-NE", (nx - 1, 0)),
            ("P-SW", (0, ny - 1)),
            ("P-SE", (nx - 1, ny - 1)),
        ]:
            wells.append(
                {
                    "name": name,
                    "type": "producer",
                    "coordinates": [i, j, mid_k],
                    "radius": 0.1,
                    "control": {"type": "bhp", "value": 10.0},
                }
            )
    else:
        raise ValueError(f"Unknown well pattern '{pattern}'")

    return wells


def build_simulator(nx: int, ny: int, nz: int, *, dt_days: float = 1.0, well_pattern: str = "corner"):
    """Constructs a fully implicit Simulator on CPU with simple well pattern."""
    reservoir_cfg = {
        "dimensions": [nx, ny, nz],
        "grid_size": [10.0, 10.0, 10.0],  # uniform 10 m cells
        "porosity": 0.2,
        "permeability": 100.0,
        "k_vertical_fraction": 1.0,
    }
    fluid_cfg = {
        "pressure": 20.0,  # MPa
        "s_w": 0.2,
        "mu_oil": 2.0,
        "mu_water": 1.0,
        "rho_oil": 850.0,
        "rho_water": 1000.0,
        "c_oil": 1e-5,
        "c_water": 1e-5,
        "c_rock": 1e-5,
        "relative_permeability": {
            "sw_cr": 0.1,
            "so_r": 0.2,
            "nw": 2.0,
            "no": 2.0,
        },
        "capillary_pressure": {"pc_scale": 0.0, "pc_exponent": 1.5},
    }
    sim_cfg = {
        "solver_type": "fully_implicit",
        "jacobian": "jfnk",
        "time_step_days": dt_days,
        "newton_max_iter": 15,
        "newton_tolerance": 1e-4,
        "use_cuda": False,
        "verbose": False,
    }

    device = torch.device("cpu")
    reservoir = Reservoir(reservoir_cfg, device=device)
    fluid = Fluid(reservoir=reservoir, config=fluid_cfg, device=device)
    wells_cfg = _make_wells(well_pattern, nx, ny, nz)
    wm = WellManager(wells_cfg, reservoir)

    sim = Simulator(reservoir, fluid, wm, sim_cfg, device=device)
    return sim, dt_days * 86400.0


def main():
    parser = argparse.ArgumentParser(description="CPU JFNK benchmark")
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--steps", type=int, default=1, help="number of time steps")
    parser.add_argument("--dt", type=float, default=1.0, help="time-step in days")
    parser.add_argument(
        "--wells",
        choices=["none", "corner", "five"],
        default="corner",
        help="well pattern to insert into the grid",
    )
    args = parser.parse_args()

    n_cells = args.nx * args.ny * args.nz
    print(f"▶ Building {args.nx}×{args.ny}×{args.nz} grid  (N={n_cells:,}) on CPU …")

    sim, dt_sec = build_simulator(
        args.nx, args.ny, args.nz, dt_days=args.dt, well_pattern=args.wells
    )

    t0 = time.time()
    completed = 0
    for step in range(args.steps):
        ok = sim.run_step(dt_sec)
        if not ok:
            print("✖ Solver failed to converge on step", step + 1)
            break
        completed += 1
    wall = time.time() - t0

    if completed > 0:
        fisolver = sim._fisolver  # type: ignore[attr-defined]
        newton = getattr(fisolver, "last_newton_iters", None)
        gmres = getattr(fisolver, "last_gmres_iters", None)
        print(
            f"✓ Completed {completed} steps in {wall:.2f} s  "
            f"({wall / completed:.2f} s/step).  "
            f"Newton iters: {newton}, GMRES vectors: {gmres}."
        )
    else:
        print("No steps completed – see solver log above.")


if __name__ == "__main__":
    main() 