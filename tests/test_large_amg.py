import sys, os, pytest
sys.path.append(os.path.abspath('src'))

try:
    import petsc4py # noqa: F401
    HAS_PETSC = True
except ImportError:
    HAS_PETSC = False

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator

@pytest.mark.skipif(not HAS_PETSC, reason="petsc4py не установлен")
def test_large_grid_with_amg():
    cfg = {
        "dimensions": [30, 30, 5],
        "grid_size": [20.0, 20.0, 10.0],
        "permeability": 150.0,
        "k_vertical_fraction": 0.2,
        "porosity": 0.22,
        "fluid": {
            "pressure": 25.0,
            "s_w": 0.2,
            "s_g": 0.0,
            "mu_oil": 1.2,
            "mu_water": 0.6,
            "rho_oil": 850.0,
            "rho_water": 1000.0
        },
        "wells": [],
        "simulation": {
            "solver_type": "fully_implicit",
            "jacobian": "jfnk",
            "backend": "hypre",
            "total_time_days": 0.5,
            "time_step_days": 0.5,
            "verbose": False
        }
    }
    res = Reservoir(cfg, device="cpu")
    fluid = Fluid(cfg["fluid"], res)
    wells = WellManager(cfg["wells"], res)
    sim = Simulator(res, fluid, wells, cfg["simulation"])

    dt = cfg["simulation"]["time_step_days"] * 86400.0
    ok = sim.run_step(dt)
    assert ok, "FI+AMG шаг не сошёлся" 