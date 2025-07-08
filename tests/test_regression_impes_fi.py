import sys, os, json, torch, numpy as np
sys.path.append(os.path.abspath('src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator

def _build_sim(solver):
    cfg = {
        "dimensions": [5, 5, 1],
        "grid_size": [10.0, 10.0, 10.0],
        "permeability": 100.0,
        "porosity": 0.25,
        "fluid": {
            "pressure": 20.0,
            "s_w": 0.2,
            "s_g": 0.0,
            "mu_oil": 1.0,
            "mu_water": 0.5,
            "rho_oil": 850.0,
            "rho_water": 1000.0
        },
        "wells": [],
        "simulation": {
            "solver_type": solver,
            "jacobian": "jfnk",
            "total_time_days": 0.3,
            "time_step_days": 0.1,
            "verbose": False
        }
    }
    res = Reservoir(cfg, device="cpu")
    fluid = Fluid(cfg["fluid"], res)
    wells = WellManager(cfg["wells"], res)
    sim = Simulator(res, fluid, wells, cfg["simulation"])
    return sim


def _run_sim(sim):
    dt_sec = sim.sim_params["time_step_days"] * 86400.0
    steps = int(sim.sim_params["total_time_days"] / sim.sim_params["time_step_days"] + 1e-8)
    for _ in range(steps):
        ok = sim.run_step(dt_sec)
        assert ok, "Симуляция не сошлась"
    return sim.fluid.pressure.clone(), sim.fluid.s_w.clone()


def test_impes_vs_fi_close():
    P_impes, Sw_impes = _run_sim(_build_sim("impes"))
    P_fi, Sw_fi       = _run_sim(_build_sim("fully_implicit"))

    rmse_p = torch.sqrt(torch.mean((P_impes - P_fi) ** 2)).item()
    rmse_sw = torch.sqrt(torch.mean((Sw_impes - Sw_fi) ** 2)).item()

    assert rmse_p < 1e-3, f"RMSE(P) = {rmse_p} слишком велик"
    assert rmse_sw < 1e-3, f"RMSE(Sw) = {rmse_sw} слишком велика" 