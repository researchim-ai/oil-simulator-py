import json, torch, pytest, math
from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.simulation import Simulator

CONFIG = "configs/fully_implicit_3d.json"
DT_SEC = 86400.0  # 1 day

def test_mass_balance_blackoil():
    with open(CONFIG) as f:
        cfg = json.load(f)
    res = Reservoir.from_config(cfg)
    fluid = Fluid.from_config(cfg)

    sim = Simulator(res, fluid, well_manager=None, sim_params={"solver_type": "fully_implicit", "jacobian": "jfnk", "newton_max_iter": 20, "verbose": False})

    mass_before = sim._compute_total_mass().item()
    ok = sim.run_step(DT_SEC)
    assert ok, "Simulation step failed to converge"
    mass_after = sim._compute_total_mass().item()

    rel_err = abs(mass_after - mass_before) / (mass_before + 1e-12)
    assert rel_err < 1e-3, f"Mass balance error too large: {rel_err:.2e}" 