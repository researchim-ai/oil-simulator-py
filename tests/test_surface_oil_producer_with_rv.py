import json
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator


def _pvt(tmp_path):
    P = [10, 20, 40, 60]
    data = {
        "pressure_MPa": P,
        "Bo": [1.2, 1.1, 1.05, 1.03],
        "Bw": [1.02, 1.01, 1.00, 0.99],
        "Bg": [0.01, 0.005, 0.003, 0.002],
        "mu_o_cP": [2.0, 1.5, 1.3, 1.2],
        "mu_w_cP": [0.5, 0.52, 0.53, 0.54],
        "mu_g_cP": [0.02, 0.025, 0.03, 0.035],
        "Rs_m3m3": [80, 120, 150, 160],
        "Rv_m3m3": [0.0, 0.01, 0.015, 0.02],
        "units": {"pressure": "MPa"}
    }
    path = os.path.join(tmp_path, "pvt_rv.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def test_surface_oil_producer_with_rv_smoke(tmp_path):
    pvt_path = _pvt(tmp_path)
    cfg = {
        "simulation": {"solver_type": "impes", "total_time_days": 1, "time_step_days": 1},
        "reservoir": {"dimensions": [5,5,1], "grid_size": [10,10,10], "porosity": 0.2, "permeability": 100.0, "k_vertical_fraction": 1.0},
        "fluid": {"pressure": 20.0, "s_w": 0.3, "s_g": 0.05, "pvt_path": pvt_path},
        "wells": [
            {"name": "PROD", "type": "producer", "i": 2, "j": 2, "k": 0, "radius": 0.1,
             "control_type": "rate", "control_value": 100.0, "rate_type": "surface", "surface_phase": "oil"}
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
    assert torch.all(fl.s_w >= fl.sw_cr)
    assert torch.all(fl.s_g >= fl.sg_cr)
    assert torch.all(fl.s_w + fl.s_g <= 1.0 - fl.so_r + 1e-6)

