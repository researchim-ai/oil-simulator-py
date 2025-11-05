import torch
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from simulator.reservoir import Reservoir
from simulator.fluid import Fluid


def test_stone2_reduces_to_two_phase_when_sg_zero():
    res_cfg = {
        "dimensions": [4, 4, 1],
        "grid_size": [10.0, 10.0, 10.0],
        "porosity": 0.2,
        "permeability": 100.0,
        "k_vertical_fraction": 1.0
    }
    relperm = {
        "model": "stone2",
        "nw": 2.0, "no": 2.0, "ng": 2.0,
        "ko_end_w": 1.0, "ko_end_g": 1.0
    }
    fluid_cfg = {
        "pressure": 20.0,
        "s_w": 0.3,
        "s_g": 0.0,
        "relative_permeability": relperm
    }
    device = torch.device('cpu')
    res = Reservoir(res_cfg, device)
    fl = Fluid(fluid_cfg, res, device)

    s_w = torch.full(res.dimensions, 0.6)
    fl.s_w = s_w
    fl.s_g = torch.zeros_like(s_w)

    kro_stone = fl.calc_oil_kr(s_w, fl.s_g)
    # Corey two-phase oil-water
    s_norm = (s_w - fl.sw_cr) / (1 - fl.sw_cr - fl.so_r)
    kro_corey = torch.clamp(1 - s_norm, 0.0, 1.0) ** fl.no

    assert torch.allclose(kro_stone, kro_corey, atol=1e-6)


def test_stone2_bounds():
    res_cfg = {"dimensions": [2,2,1], "grid_size": [10,10,10], "porosity": 0.2, "permeability": 100.0, "k_vertical_fraction": 1.0}
    fluid_cfg = {"pressure": 20.0, "s_w": 0.2, "s_g": 0.1, "relative_permeability": {"model": "stone2"}}
    device = torch.device('cpu')
    res = Reservoir(res_cfg, device)
    fl = Fluid(fluid_cfg, res, device)
    # At max water+gas (leaving minimal oil), kro should be ~0
    fl.s_w = torch.full(res.dimensions, 1.0 - fl.so_r - 1e-6)
    fl.s_g = torch.zeros_like(fl.s_w)
    kro = fl.calc_oil_kr(fl.s_w, fl.s_g)
    assert (kro <= 1e-3).all()


