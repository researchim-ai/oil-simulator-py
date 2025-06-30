import torch
import numpy as np
import json
import os
import sys

# Добавляем src в путь для импорта компонентов симулятора
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator

def run_fi_simulation_limited_steps(config_path: str, max_steps: int = 50):
    """Запускает полностью неявную схему, ограничивая число шагов max_steps."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Переопределяем общее время, чтобы число шагов было ровно max_steps
    dt_days = config['simulation']['time_step_days']
    config['simulation']['total_time_days'] = dt_days * max_steps

    dev = torch.device('cpu')

    reservoir = Reservoir(config=config['reservoir'], device=dev)
    wells = WellManager(config['wells'], reservoir)
    fluid = Fluid(reservoir=reservoir, config=config['fluid'], device=dev)

    sim = Simulator(reservoir, fluid, wells, config['simulation'])

    dt_sec = dt_days * 86400

    for step in range(max_steps):
        converged = sim.run_step(dt=dt_sec)
        assert converged, f"Солвер не сошёлся на шаге {step+1} из {max_steps}"

    return fluid.pressure.cpu().numpy(), fluid.s_w.cpu().numpy()


def test_fully_implicit_smoke_50steps():
    """Укороченный дымовой тест: 50 шагов вместо 200 для экономии времени CI."""
    config_path = "configs/fully_implicit_2d.json"

    pressure, saturation = run_fi_simulation_limited_steps(config_path, max_steps=50)

    # Проверяем отсутствие NaN и корректные пределы насыщенности
    assert not np.isnan(pressure).any(), "В давлении обнаружены NaN"
    assert not np.isnan(saturation).any(), "В насыщенности обнаружены NaN"
    assert np.all((saturation >= 0) & (saturation <= 1)), "Насыщенность вышла за пределы [0,1]" 