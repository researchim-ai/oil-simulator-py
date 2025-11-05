import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator


def test_wopr_limit_throttles_surface_oil_producer():
    cfg = {
        "simulation": {"solver_type": "impes", "total_time_days": 1, "time_step_days": 1},
        "reservoir": {"dimensions": [5,5,1], "grid_size": [10,10,10], "porosity": 0.2, "permeability": 100.0, "k_vertical_fraction": 1.0},
        "fluid": {"pressure": 20.0, "s_w": 0.3, "s_g": 0.02, "pvt_path": "configs/pvt/pvt_synthetic.json"},
        "wells": [
            {"name": "PROD", "type": "producer", "i": 2, "j": 2, "k": 0, "radius": 0.1,
             "control_type": "rate", "control_value": 50000.0, "rate_type": "surface", "surface_phase": "oil",
             "limits": {"wopr": 100.0}}
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
    w = wells.get_wells()[0]
    # Проверим, что surface oil rate не превышает лимит с запасом на численность
    assert w.last_surface_rates.get('oil', 0.0) <= 100.0 * 1.10


def test_bhp_min_autoswitch_reduces_outflow():
    cfg_base = {
        "simulation": {"solver_type": "impes", "total_time_days": 1, "time_step_days": 1},
        "reservoir": {"dimensions": [5,5,1], "grid_size": [10,10,10], "porosity": 0.2, "permeability": 100.0, "k_vertical_fraction": 1.0},
        "fluid": {"pressure": 10.0, "s_w": 0.2, "s_g": 0.0, "pvt_path": "configs/pvt/pvt_synthetic.json"}
    }
    # Без bhp_min
    cfg1 = dict(cfg_base)
    cfg1["wells"] = [{"name": "PROD1", "type": "producer", "i": 2, "j": 2, "k": 0, "radius": 0.1,
                       "control_type": "rate", "control_value": 20000.0}]
    # С bhp_min высоким, чтобы сработал троттлинг
    cfg2 = dict(cfg_base)
    cfg2["wells"] = [{"name": "PROD2", "type": "producer", "i": 2, "j": 2, "k": 0, "radius": 0.1,
                       "control_type": "rate", "control_value": 20000.0, "bhp_min": 9.5}]

    device = torch.device('cpu')
    # Прогон 1
    res1 = Reservoir(cfg1['reservoir'], device)
    wells1 = WellManager(cfg1['wells'], res1)
    fl1 = Fluid(cfg1['fluid'], res1, device)
    sim1 = Simulator(res1, fl1, wells1, cfg1['simulation'], device)
    sim1.run_step(cfg1['simulation']['time_step_days'] * 86400)
    q1 = abs(wells1.get_wells()[0].last_q_total or 0.0)

    # Прогон 2 (с bhp_min)
    res2 = Reservoir(cfg2['reservoir'], device)
    wells2 = WellManager(cfg2['wells'], res2)
    fl2 = Fluid(cfg2['fluid'], res2, device)
    sim2 = Simulator(res2, fl2, wells2, cfg2['simulation'], device)
    sim2.run_step(cfg2['simulation']['time_step_days'] * 86400)
    q2 = abs(wells2.get_wells()[0].last_q_total or 0.0)

    # Ожидаем, что отбор при bhp_min будет не больше, чем без ограничения
    assert q2 <= q1 + 1e-9

