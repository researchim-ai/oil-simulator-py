import sys, os, torch
sys.path.append(os.path.abspath('src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator

def test_capillary_stability():
    cfg = {
        "dimensions": [5,5,1],
        "grid_size": [10.0,10.0,10.0],
        "permeability": 100.0,
        "porosity": 0.25,
        "fluid": {
            "pressure": 20.0,
            "s_w": 0.2,
            "s_g": 0.0,
            "mu_oil": 1.0,
            "mu_water": 0.5,
            "rho_oil": 850.0,
            "rho_water": 1000.0,
            "capillary_pressure": {
                "pc_scale": 5e5,   # 0.5 МПа
                "pc_exponent": 1.5
            }
        },
        "wells": [],
        "simulation": {
            "solver_type": "impes",
            "total_time_days": 1.0,
            "time_step_days": 0.2,
            "verbose": False
        }
    }
    res = Reservoir(cfg, device="cpu")
    fluid = Fluid(cfg["fluid"], res)
    wells = WellManager(cfg["wells"], res)
    sim = Simulator(res, fluid, wells, cfg["simulation"])

    dt = cfg["simulation"]["time_step_days"] * 86400.0
    steps = int(cfg["simulation"]["total_time_days"] / cfg["simulation"]["time_step_days"] + 1e-8)

    for _ in range(steps):
        ok = sim.run_step(dt)
        assert ok, "IMPES не сошёлся с капиллярным давлением"

    # Проверяем, что значения Sw в пределах
    assert torch.all(sim.fluid.s_w <= 1.0 + 1e-6)
    assert torch.all(sim.fluid.s_w >= sim.fluid.sw_cr - 1e-6) 