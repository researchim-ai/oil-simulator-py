import json
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from simulator.pvt import PVTTable
from simulator.reservoir import Reservoir
from simulator.fluid import Fluid


def _write_pvt_with_rv(tmp_path):
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


def test_pvt_rv_loading_and_eval(tmp_path):
    pvt_path = _write_pvt_with_rv(tmp_path)
    pvt = PVTTable(pvt_path)
    out = pvt.eval(torch.tensor([15.0, 55.0]).numpy())
    assert "Rv" in out
    assert (out["Rv"] >= 0).all()


def test_fluid_rho_g_with_rv(tmp_path):
    pvt_path = _write_pvt_with_rv(tmp_path)
    cfg = {
        "dimensions": [2,2,1],
        "grid_size": [10,10,10],
        "porosity": 0.2,
        "permeability": 100.0,
        "k_vertical_fraction": 1.0
    }
    res = Reservoir(cfg, torch.device('cpu'))
    fl = Fluid({"pressure": 40.0, "s_w": 0.2, "s_g": 0.1, "pvt_path": pvt_path}, res, torch.device('cpu'))
    rho_g = fl.rho_g
    assert torch.isfinite(rho_g).all()
    assert (rho_g > 0).all()

