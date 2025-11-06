import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator


def test_component_balance_with_rv_surface_producer_and_gas_injector():
    cfg = {
        "simulation": {"solver_type": "impes", "total_time_days": 1, "time_step_days": 1},
        "reservoir": {"dimensions": [5,5,1], "grid_size": [10,10,10], "porosity": 0.25, "permeability": 200.0, "k_vertical_fraction": 1.0},
        "fluid": {"pressure": 20.0, "s_w": 0.25, "s_g": 0.05, "pvt_path": "configs/pvt/pvt_synthetic.json"},
        "wells": [
            {"name": "INJ-G", "type": "injector", "i": 1, "j": 1, "k": 0, "radius": 0.1,
             "control_type": "rate", "control_value": 500.0, "rate_type": "surface", "injected_phase": "gas"},
            {"name": "PROD", "type": "producer", "i": 3, "j": 3, "k": 0, "radius": 0.1,
             "control_type": "rate", "control_value": 300.0, "rate_type": "surface", "surface_phase": "liquid"}
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

    cb = sim.component_balance
    # Компонентный баланс: in - out ~= accum (с допуском)
    oil_residual = (cb['oil']['in'] - cb['oil']['out']) - cb['oil']['accum']
    gas_residual = (cb['gas']['in'] - cb['gas']['out']) - cb['gas']['accum']
    assert abs(oil_residual) < 1e-2 * (1.0 + abs(cb['oil']['accum']))
    assert abs(gas_residual) < 1e-2 * (1.0 + abs(cb['gas']['accum']))

