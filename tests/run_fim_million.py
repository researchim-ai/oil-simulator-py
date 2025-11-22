import json
import torch
import numpy as np
import sys
import os
import time

# Добавляем корневую директорию в путь
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator.reservoir import Reservoir
from src.simulator.fluid import Fluid
from src.simulator.well import WellManager
from src.simulator.manual_assembly_solver import ManualFIMSolver

def test_fim_million():
    print("=== Запуск теста Fully Implicit Solver (Million Grid) ===")
    
    # Загружаем конфиг
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'test_fim_million.json')
    with open(config_path, "r") as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Инициализация
    print("Инициализация модели...")
    t0 = time.time()
    reservoir = Reservoir(config['reservoir'], device)
    fluid = Fluid(config['fluid'], reservoir, device)
    well_manager = WellManager(config['wells'], reservoir)
    
    # Используем ManualFIMSolver (Sparse)
    solver = ManualFIMSolver(reservoir, fluid, well_manager, config['simulation'])
    
    print(f"Инициализация завершена за {time.time()-t0:.2f} сек")
    print(f"Сетка: {solver.nx}x{solver.ny}x{solver.nz} ({solver.n_cells} ячеек)")
    print("Начинаем симуляцию...")
    
    # Запускаем несколько шагов
    n_steps = 3
    dt = config['simulation']['time_step_days'] * 86400
    
    for step in range(n_steps):
        print(f"\n--- Шаг {step+1} (dt={dt/86400:.2f} days) ---")
        t_start = time.time()
        
        try:
            success = solver.step(dt)
        except Exception as e:
            print(f"Ошибка при шаге: {e}")
            import traceback
            traceback.print_exc()
            break
            
        t_end = time.time()
        print(f"Время шага: {t_end - t_start:.2f} сек")
        
        if success:
            p_mean = fluid.pressure.mean().item() / 1e6
            sw_mean = fluid.s_w.mean().item()
            print(f"Шаг успешен. P_mean={p_mean:.6f} MPa, Sw_mean={sw_mean:.7f}")
        else:
            print("Шаг НЕ сошелся!")
            break

if __name__ == "__main__":
    test_fim_million()

