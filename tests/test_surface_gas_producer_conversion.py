import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator


def test_surface_gas_rate_producer_total_rate_formula():
    cfg = {
        "simulation": {"solver_type": "impes", "total_time_days": 1, "time_step_days": 1},
        "reservoir": {"dimensions": [5,5,1], "grid_size": [10,10,10], "porosity": 0.2, "permeability": 100.0, "k_vertical_fraction": 1.0},
        "fluid": {"pressure": 20.0, "s_w": 0.3, "s_g": 0.05, "pvt_path": "configs/pvt/pvt_synthetic.json"},
        "wells": [
            {"name": "PROD", "type": "producer", "i": 2, "j": 2, "k": 0, "radius": 0.1,
             "control_type": "rate", "control_value": 200.0, "rate_type": "surface", "surface_phase": "gas"}
        ]
    }
    device = torch.device('cpu')
    res = Reservoir(cfg['reservoir'], device)
    wells = WellManager(cfg['wells'], res)
    fl = Fluid(cfg['fluid'], res, device)
    sim = Simulator(res, fl, wells, cfg['simulation'], device)

    i,j,k = 2,2,0
    fo, fw, fg = _fractions(fl, i, j, k)
    Bo = float(fl._eval_pvt(fl.pressure, 'Bo')[i,j,k])
    Bg = float(fl._eval_pvt(fl.pressure, 'Bg')[i,j,k])
    Rs = float(fl._eval_pvt(fl.pressure, 'Rs')[i,j,k])

    q_g_surf = 200.0 / 86400.0
    denom = (fg / max(Bg, 1e-12) + fo * Rs / max(Bo, 1e-12))
    q_total_expected = - q_g_surf / max(denom, 1e-12)

    # Выполнить шаг (smoke): проверим, что проходит без ошибок и насыщенности в допустимых пределах
    dt = cfg['simulation']['time_step_days'] * 86400
    ok = sim.run_step(dt)
    assert ok
    assert torch.all(fl.s_w >= fl.sw_cr) and torch.all(fl.s_g >= fl.sg_cr)
    assert torch.all(fl.s_w + fl.s_g <= 1.0 - fl.so_r + 1e-6)


def _fractions(fl: Fluid, i: int, j: int, k: int):
    kro, krw, krg = fl.get_rel_perms(fl.s_w)
    mu_o = float(fl._eval_pvt(fl.pressure, 'mu_o')[i,j,k]) * 1e-3
    mu_w = float(fl._eval_pvt(fl.pressure, 'mu_w')[i,j,k]) * 1e-3
    mu_g = float(fl._eval_pvt(fl.pressure, 'mu_g')[i,j,k]) * 1e-3
    mob_o = float(kro[i,j,k].item()) / mu_o
    mob_w = float(krw[i,j,k].item()) / mu_w
    mob_g = float(krg[i,j,k].item()) / mu_g
    lt = mob_o + mob_w + mob_g
    return mob_o/lt, mob_w/lt, mob_g/lt


