import sys, os, torch, pytest, time
sys.path.append(os.path.abspath("src"))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.simulation import Simulator

GRID = [60, 60, 10]
DT_SEC = 86400.0  # 1 сутки
NSTEPS = 3

def make_sim(solver):
    res_cfg = {
        "dimensions": GRID,
        "grid_size": [20.0, 20.0, 10.0],
        "porosity": 0.22,
        "permeability": 150.0,
        "k_vertical_fraction": 0.2,
    }
    fluid_cfg = {
        "pressure": 20.0,
        "s_w": 0.25,
        "s_g": 0.05,
        "mu_oil": 1.2,
        "mu_water": 0.6,
        "mu_gas": 0.04,
        "rho_oil": 850,
        "rho_water": 1000,
        "rho_gas": 90,
        "relative_permeability": {"nw": 2, "no": 2, "ng": 2, "sw_cr": 0.15, "so_r": 0.15},
    }
    sim_params = {
        "solver": solver,
        "jacobian": "jfnk" if solver == "fully_implicit" else "manual",
        "newton_max_iter": 12,
        "cg_max_iter": 400,
        "verbose": True,
    }
    res = Reservoir(res_cfg)
    fluid = Fluid(fluid_cfg, res)
    return Simulator(res, fluid, well_manager=None, sim_params=sim_params)


def total_masses(sim):
    vol = sim.reservoir.cell_volume
    phi = sim.reservoir.porosity
    rho_w = sim.fluid.rho_w
    rho_o = sim.fluid.rho_o
    rho_g = sim.fluid.rho_g
    sw = sim.fluid.s_w
    sg = sim.fluid.s_g
    mw = torch.sum(phi * sw * rho_w) * vol
    mo = torch.sum(phi * (1 - sw - sg) * rho_o) * vol
    mg = torch.sum(phi * sg * rho_g) * vol
    return mw.item(), mo.item(), mg.item()


@pytest.mark.parametrize("solver", ["impes", "fully_implicit"])
def test_large_case(solver):
    sim = make_sim(solver)
    mw0, mo0, mg0 = total_masses(sim)
    print(f"\n{solver}: initial masses (t) W={mw0:.2e}, O={mo0:.2e}, G={mg0:.2e}")
    start = time.time()
    for step in range(NSTEPS):
        ok = sim.run_step(DT_SEC)
        assert ok, f"{solver} failed at step {step}"
        mw, mo, mg = total_masses(sim)
        print(f"  Step {step+1}: Sw[{sim.fluid.s_w.min():.3f},{sim.fluid.s_w.max():.3f}], "
              f"Sg[{sim.fluid.s_g.min():.3f},{sim.fluid.s_g.max():.3f}], "
              f"masses W={mw:.2e},O={mo:.2e},G={mg:.2e}")
        # масс-баланс колеблется <5% из-за сжимаемости
        assert abs(mw - mw0)/mw0 < 0.05
    print(f"{solver} finished {NSTEPS} steps in {time.time()-start:.1f}s") 