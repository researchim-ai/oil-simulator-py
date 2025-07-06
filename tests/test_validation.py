import json
import torch
import math
import time
from pathlib import Path

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator

CONFIGS = [
    Path("configs/fully_implicit_2d.json"),
    Path("configs/medium_2d.json"),
]

NEWTON_LIMIT = 12
GMRES_LIMIT = 300  # conservative upper bound per step

def _run_once(cfg_path: Path):
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # Force fully implicit / JFNK mode and deterministic CPU run
    sim_cfg = cfg["simulation"]
    sim_cfg["solver_type"] = "fully_implicit"
    sim_cfg["jacobian"] = "jfnk"
    sim_cfg["use_cuda"] = False
    sim_cfg["verbose"] = False

    # Construct objects
    device = torch.device("cpu")
    reservoir = Reservoir(cfg["reservoir"], device=device)
    fluid = Fluid(reservoir=reservoir, config=cfg["fluid"], device=device)
    wells_cfg = cfg.get("wells", [])
    wm = WellManager(wells_cfg, reservoir)

    sim = Simulator(reservoir, fluid, wm, sim_cfg, device=device)

    dt_days = sim_cfg.get("time_step_days", 1.0)
    dt_sec = dt_days * 86400.0

    ok = sim.run_step(dt_sec)
    assert ok, f"Simulation did not converge for {cfg_path.name}"

    # Fetch solver diagnostics
    solver = sim._fisolver  # created inside run_step
    n_iter = getattr(solver, "last_newton_iters", math.inf)
    g_iter = getattr(solver, "last_gmres_iters", math.inf)

    assert n_iter <= NEWTON_LIMIT, (
        f"Too many Newton iterations ({n_iter}) for {cfg_path.name}"
    )
    assert g_iter <= GMRES_LIMIT, (
        f"Too many GMRES iterations ({g_iter}) for {cfg_path.name}"
    )


def test_validation_configs():
    for cfg in CONFIGS:
        assert cfg.exists(), f"Missing config {cfg}"
        _run_once(cfg) 