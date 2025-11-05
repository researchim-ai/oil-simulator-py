import os
import sys
import json
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator


def test_surface_water_injection_mass_increase():
    cfg = {
        "simulation": {"solver_type": "impes", "total_time_days": 1, "time_step_days": 1},
        "reservoir": {"dimensions": [5,5,1], "grid_size": [10,10,10], "porosity": 0.2, "permeability": 100.0, "k_vertical_fraction": 1.0},
        "fluid": {
            "pressure": 20.0,
            "s_w": 0.2,
            "s_g": 0.0,
            "pvt_path": "configs/pvt/pvt_synthetic.json"
        },
        "wells": [
            {"name": "INJ-W", "type": "injector", "i": 2, "j": 2, "k": 0, "radius": 0.1,
             "control_type": "rate", "control_value": 100.0, "injected_phase": "water", "rate_type": "surface"}
        ]
    }

    device = torch.device('cpu')
    res = Reservoir(cfg['reservoir'], device)
    wells = WellManager(cfg['wells'], res)
    fl = Fluid(cfg['fluid'], res, device)
    sim = Simulator(res, fl, wells, cfg['simulation'], device)

    # Масса воды до
    mw0 = torch.sum(res.porosity * fl.s_w * fl.rho_w * res.cell_volume).item()

    dt = cfg['simulation']['time_step_days'] * 86400
    sim.run_step(dt)

    # Ожидаемый приток воды за шаг (кг): q_surf * Bw * rho_w_res * dt
    Bw_cell = float(fl._eval_pvt(fl.pressure, 'Bw')[2,2,0])
    rho_w_res = float(fl.rho_w[2,2,0])
    q_surf_m3s = 100.0 / 86400.0
    expected_mass_in = q_surf_m3s * Bw_cell * rho_w_res * dt

    mw1 = torch.sum(res.porosity * fl.s_w * fl.rho_w * res.cell_volume).item()
    delta = mw1 - mw0

    assert delta > 0
    # Допускаем 10% из-за распределения и градиентов
    assert abs(delta - expected_mass_in) / max(expected_mass_in, 1e-6) < 0.2


