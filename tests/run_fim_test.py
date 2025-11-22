import json
import torch
import numpy as np
import sys
import os

# Добавляем корневую директорию в путь, чтобы Python видел пакет src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator.reservoir import Reservoir
from src.simulator.fluid import Fluid
from src.simulator.well import WellManager
from src.simulator.fully_implicit_solver import FullyImplicitSolver

def test_fim():
    print("=== Запуск теста Fully Implicit Solver (Small Grid) ===")
    
    # Загружаем конфиг
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'test_fim_small.json')
    with open(config_path, "r") as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Инициализация
    reservoir = Reservoir(config['reservoir'], device)
    fluid = Fluid(config['fluid'], reservoir, device)
    well_manager = WellManager(config['wells'], reservoir)
    
    solver = FullyImplicitSolver(reservoir, fluid, well_manager, config['simulation'])
    
    print(f"Сетка: {solver.nx}x{solver.ny}x{solver.nz} ({solver.n_cells} ячеек)")
    print("Начинаем симуляцию...")
    
    # Запускаем несколько шагов
    n_steps = 5
    dt = config['simulation']['time_step_days'] * 86400
    
    for step in range(n_steps):
        print(f"\n--- Шаг {step+1} (dt={dt/86400:.2f} days) ---")
        success = solver.step(dt)
        
        if success:
            p_mean = fluid.pressure.mean().item() / 1e6
            sw_mean = fluid.s_w.mean().item()
            print(f"Шаг успешен. P_mean={p_mean:.2f} MPa, Sw_mean={sw_mean:.4f}")
        else:
            print("Шаг НЕ сошелся!")
            break

if __name__ == "__main__":
    test_fim()
