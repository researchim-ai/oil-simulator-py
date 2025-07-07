import sys, os, torch, pytest, time
sys.path.append(os.path.abspath("src"))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.simulation import Simulator

GRID = [20, 20, 5]
DT_SEC = 43200.0  # 0.5 суток
NSTEPS = 5


def make_simulator(solver_type: str):
    reservoir_cfg = {
        "dimensions": GRID,
        "grid_size": [10.0, 10.0, 5.0],
        "permeability": 100.0,
        "porosity": 0.2,
        "k_vertical_fraction": 0.2,
    }
    fluid_cfg = {
        "pressure": 15.0,
        "s_w": 0.25,
        "s_g": 0.05,
        "mu_oil": 1.0,
        "mu_water": 0.5,
        "mu_gas": 0.04,
        "rho_oil": 850,
        "rho_water": 1000,
        "rho_gas": 120,
        "relative_permeability": {"nw": 2, "no": 2, "ng": 2, "sw_cr": 0.15, "so_r": 0.15},
    }
    sim_params = {
        "solver": solver_type,
        "jacobian": "jfnk" if solver_type == "fully_implicit" else "manual",
        "newton_max_iter": 10,
        "cg_max_iter": 300,
    }
    res = Reservoir(reservoir_cfg)
    fluid = Fluid(fluid_cfg, res)
    sim = Simulator(res, fluid, well_manager=None, sim_params=sim_params)
    return sim


@pytest.mark.parametrize("solver_type", ["impes", "fully_implicit"])
def test_three_phase_mid(solver_type):
    sim = make_simulator(solver_type)
    t0 = time.time()
    for step in range(NSTEPS):
        ok = sim.run_step(DT_SEC)
        assert ok, f"{solver_type} failed at step {step}"
    print(f"{solver_type} {NSTEPS} steps finished in {time.time()-t0:.1f}s")
    # Проверяем диапазон насыщенностей
    sw_max = sim.fluid.s_w.max().item()
    sg_max = sim.fluid.s_g.max().item()
    assert sw_max <= 1.0 and sg_max <= 1.0 and (sw_max+sg_max)<=1.01 