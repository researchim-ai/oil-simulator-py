import sys, os
sys.path.append(os.path.abspath("src"))
import torch
import pytest

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.simulation import Simulator


def make_simulator(solver_type: str):
    # Мини-конфиг для 1×1×1 ячейки
    reservoir_cfg = {
        "dimensions": [1, 1, 1],
        "grid_size": [10.0, 10.0, 10.0],
        "permeability": 100.0,  # мД
        "porosity": 0.2,
    }
    fluid_cfg = {
        "pressure": 10.0,      # МПа
        "s_w": 0.2,
        "s_g": 0.1,
        "mu_oil": 1.0,         # cP
        "mu_water": 0.5,
        "mu_gas": 0.05,
        "rho_oil": 850.0,
        "rho_water": 1000.0,
        "rho_gas": 150.0,
        "relative_permeability": {"nw": 2, "no": 2, "ng": 2, "sw_cr": 0.1, "so_r": 0.1},
    }
    sim_params = {
        "solver": solver_type,
        "jacobian": "jfnk" if solver_type == "fully_implicit" else "manual",
    }
    reservoir = Reservoir(reservoir_cfg)
    fluid = Fluid(fluid_cfg, reservoir)
    sim = Simulator(reservoir, fluid, well_manager=None, sim_params=sim_params)
    return sim


@pytest.mark.parametrize("solver_type", ["impes", "fully_implicit"])
def test_three_phase_step(solver_type):
    sim = make_simulator(solver_type)
    ok = sim.run_step(dt=86400.0)  # один день
    assert ok, f"{solver_type} step failed"
    # Проверяем, что насыщенности корректные
    sw = sim.fluid.s_w.item()
    sg = getattr(sim.fluid, "s_g", torch.tensor(0.0)).item()
    assert 0.0 <= sw <= 1.0
    assert 0.0 <= sg <= 1.0
    assert sw + sg <= 1.0 + 1e-6 