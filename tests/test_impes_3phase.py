import os
import sys
import json
import torch
import numpy as np

# Добавляем src в путь для импорта компонентов
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator


def _compute_total_masses(reservoir: Reservoir, fluid: Fluid) -> dict:
    phi = reservoir.porosity
    cell_vol = reservoir.cell_volume
    rho_w = fluid.rho_w
    rho_o = fluid.rho_o
    rho_g = fluid.rho_g
    s_w = fluid.s_w
    s_o = fluid.s_o
    s_g = fluid.s_g

    mw = torch.sum(phi * s_w * rho_w * cell_vol).item()
    mo = torch.sum(phi * s_o * rho_o * cell_vol).item()
    mg = torch.sum(phi * s_g * rho_g * cell_vol).item()
    return {"water": mw, "oil": mo, "gas": mg, "total": mw + mo + mg}


def _run_3phase_impes(config_path: str, total_time_days: int = 5, time_step_days: float = 1.0):
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Параметры симуляции
    config['simulation']['solver_type'] = 'impes'
    config['simulation']['total_time_days'] = total_time_days
    config['simulation']['time_step_days'] = time_step_days

    device = torch.device('cpu')
    reservoir = Reservoir(config=config['reservoir'], device=device)
    well_manager = WellManager(config['wells'], reservoir)
    fluid = Fluid(config=config['fluid'], reservoir=reservoir, device=device)
    sim = Simulator(reservoir, fluid, well_manager, config['simulation'], device=device)

    time_step_sec = time_step_days * 86400
    num_steps = int(total_time_days / time_step_days)

    masses_initial = _compute_total_masses(reservoir, fluid)

    for _ in range(num_steps):
        ok = sim.run_step(dt=time_step_sec)
        assert ok, "IMPES шаг не сошёлся на 3-фазном кейсе"

    masses_final = _compute_total_masses(reservoir, fluid)
    return sim, reservoir, fluid, masses_initial, masses_final


def test_impes_3phase_smoke():
    cfg = "configs/gas_injection_3phase.json"
    sim, reservoir, fluid, m0, m1 = _run_3phase_impes(cfg, total_time_days=5, time_step_days=1.0)

    # Проверка NaN
    assert not torch.isnan(fluid.pressure).any(), "NaN в поле давления"
    assert not torch.isnan(fluid.s_w).any(), "NaN в поле Sw"
    assert not torch.isnan(fluid.s_g).any(), "NaN в поле Sg"

    # Границы насыщенностей
    assert torch.all(fluid.s_w >= fluid.sw_cr) and torch.all(fluid.s_w <= 1.0 - fluid.so_r)
    assert torch.all(fluid.s_g >= fluid.sg_cr) and torch.all(fluid.s_g <= 1.0 - fluid.so_r - fluid.s_w + 1e-8)
    s_sum = fluid.s_w + fluid.s_g + fluid.s_o
    assert torch.allclose(s_sum, torch.ones_like(s_sum), atol=1e-6), "Сумма насыщенностей отличается от 1"


def test_impes_3phase_mass_balance():
    cfg = "configs/gas_injection_3phase.json"
    sim, reservoir, fluid, m0, m1 = _run_3phase_impes(cfg, total_time_days=10, time_step_days=1.0)

    # Считаем чистый приток массы (in - out)
    net_in_water = sim.mass_balance['water']['in'] - sim.mass_balance['water']['out']
    net_in_gas = sim.mass_balance['gas']['in'] - sim.mass_balance['gas']['out']
    net_in_oil = sim.mass_balance['oil']['in'] - sim.mass_balance['oil']['out']
    net_in_total = net_in_water + net_in_gas + net_in_oil

    delta_mass = m1['total'] - m0['total']

    # Проверяем баланс массы с разумной толерантностью (численная диссипация/округления)
    atol = max(1e-6, 1e-3 * max(abs(m0['total']), 1.0))
    assert abs(delta_mass - net_in_total) <= 5 * atol, (
        f"Нарушение баланса массы: dM={delta_mass:.6e}, net_in={net_in_total:.6e}"
    )


