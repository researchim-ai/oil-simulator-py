import json
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator


def test_pcog_smoke_runs():
    cfg = {
        "description": "pcog smoke",
        "simulation": {"solver_type": "impes", "total_time_days": 1, "time_step_days": 1},
        "reservoir": {"dimensions": [5,5,1], "grid_size": [10,10,10], "porosity": 0.2, "permeability": 100.0, "k_vertical_fraction": 1.0},
        "fluid": {
            "pressure": 20.0,
            "s_w": 0.2,
            "s_g": 0.05,
            "relative_permeability": {"model": "stone2"},
            "capillary_pressure_og": {"pc_scale": 1e5, "pc_exponent": 1.5}
        },
        "wells": [
            {"name": "INJ-G", "type": "injector", "i": 1, "j": 2, "k": 0, "radius": 0.1, "control_type": "rate", "control_value": 10.0, "injected_phase": "gas"},
            {"name": "PROD", "type": "producer", "i": 3, "j": 2, "k": 0, "radius": 0.1, "control_type": "bhp", "control_value": 18.0}
        ]
    }

    device = torch.device('cpu')
    res = Reservoir(cfg['reservoir'], device)
    wells = WellManager(cfg['wells'], res)
    fl = Fluid(cfg['fluid'], res, device)
    sim = Simulator(res, fl, wells, cfg['simulation'], device)

    dt = cfg['simulation']['time_step_days'] * 86400
    ok = sim.run_step(dt)
    assert ok
    # Bounds remain valid
    assert torch.all(fl.s_w >= fl.sw_cr) and torch.all(fl.s_g >= fl.sg_cr)
    assert torch.all(fl.s_w + fl.s_g <= 1.0 - fl.so_r + 1e-6)


