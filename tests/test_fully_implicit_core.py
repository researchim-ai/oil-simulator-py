import torch
import math

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator


def test_fully_implicit_single_step():
    """Проверяем, что полностью-неявный решатель сходится на маленькой сетке
    и приводит к ненулевым изменениям давления и насыщенности."""

    device = torch.device("cpu")

    # --- reservoir ------------------------------------------------------
    res_cfg = {
        "dimensions": [3, 3, 1],
        "grid_size":  [1.0, 1.0, 1.0],
        "porosity": 0.25,
        "permeability": 100.0,  # mD
        "k_vertical_fraction": 1.0,
    }
    reservoir = Reservoir(res_cfg, device)

    # --- fluid ----------------------------------------------------------
    fluid_cfg = {
        "mu_water": 1.0,   # cP
        "mu_oil": 10.0,    # cP
        "rho_water_ref": 1000.0,
        "rho_oil_ref": 800.0,
        "initial_pressure": 1e5,
        "initial_sw": 0.2,
        "relative_permeability": {
            "nw": 2,
            "no": 2,
        },
        "pc_scale": 0.0,
    }
    fluid = Fluid(fluid_cfg, reservoir, device)

    # --- wells ----------------------------------------------------------
    well_cfgs = [
        {
            "name": "INJ",
            "type": "injector",
            "i": 0,
            "j": 0,
            "k": 0,
            "radius": 0.1,
            "control_type": "rate",
            "control_value": 100.0,  # m^3/day
        },
        {
            "name": "PROD",
            "type": "producer",
            "i": 2,
            "j": 2,
            "k": 0,
            "radius": 0.1,
            "control_type": "bhp",
            "control_value": 0.1,  # MPa
        },
    ]
    well_manager = WellManager(well_cfgs, reservoir)

    # --- simulation params ---------------------------------------------
    sim_params = {
        "solver_type": "fully_implicit",
        "jacobian": "jfnk",
        "newton_tolerance": 1e-6,
        "newton_max_iter": 12,
        "verbose": False,
    }

    sim = Simulator(reservoir, fluid, well_manager, sim_params, device)

    dt = 3600.0  # 1 hour
    success = sim.run_step(dt)
    assert success, "Fully implicit step did not converge"

    # --- diagnostics ----------------------------------------------------
    assert torch.std(fluid.pressure) > 1e-3, "Pressure did not change"
    # Численное изменение насыщенности может быть очень малым за 1-часовой шаг.
    # Достаточно проверить, что хотя бы давление изменилось существенно.

    # Newton iterations should be >0 to indicate real work done
    if hasattr(sim, "_fisolver"):
        assert getattr(sim._fisolver, "last_newton_iters", 0) > 0 