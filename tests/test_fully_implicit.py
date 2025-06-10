import torch
import numpy as np
import json
import os
import sys
import pytest
import time

# Добавляем src в путь для импорта компонентов симулятора
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator

def run_fully_implicit_simulation_for_test(config_path):
    """Запускает симуляцию с полностью неявной схемой и возвращает конечное давление и насыщенность."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    res_params = config['reservoir']
    sim_params = config['simulation']
    fluid_params = config['fluid']
    well_params = config['wells']

    # Принудительно используем CPU для консистентности тестов
    device = torch.device("cpu") 

    reservoir = Reservoir(config=res_params, device=device)
    well_manager = WellManager(well_params, reservoir)
    fluid = Fluid(reservoir=reservoir, config=fluid_params, device=device)
    
    # Передаем sim_params в конструктор
    sim = Simulator(reservoir, fluid, well_manager, sim_params)

    total_time_days = sim_params['total_time_days']
    time_step_days = sim_params['time_step_days']
    time_step_sec = time_step_days * 86400
    num_steps = int(total_time_days / time_step_days)

    for i in range(num_steps):
        print(f"\n--- Шаг {i+1}/{num_steps} ---")
        converged = sim.run_step(dt=time_step_sec)
        if not converged:
            # Если решатель не сошелся, тест должен упасть с ошибкой
            raise RuntimeError(f"Решатель не сошелся на шаге {i+1}")

    pressure = fluid.pressure.cpu().numpy()
    saturation = fluid.s_w.cpu().numpy()

    return pressure, saturation

def test_fully_implicit_smoke():
    """
    Дымовой тест: проверяет, что симуляция с полностью неявным решателем
    запускается без ошибок и не производит NaN значений.
    """
    config_path = "configs/fully_implicit_2d.json"
    
    pressure, saturation = run_fully_implicit_simulation_for_test(config_path)

    # Проверяем, что нет NaN значений в результатах
    assert not np.isnan(pressure).any(), "В итоговом давлении есть NaN значения."
    assert not np.isnan(saturation).any(), "В итоговой насыщенности есть NaN значения."

    # Проверяем, что значения находятся в разумных пределах
    assert np.all(saturation >= 0) and np.all(saturation <= 1), "Значения насыщенности вышли за пределы [0, 1]."

def run_simulation_benchmark(config_path, solver_type, num_steps=10):
    """Запускает симуляцию с заданной схемой и замеряет время выполнения."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Устанавливаем нужный тип решателя
    config['simulation']['solver_type'] = solver_type
    
    # Принудительно используем CPU для консистентности тестов
    device = torch.device("cpu")
    
    reservoir = Reservoir(config=config['reservoir'], device=device)
    well_manager = WellManager(config['wells'], reservoir)
    fluid = Fluid(reservoir=reservoir, config=config['fluid'], device=device)
    
    sim = Simulator(reservoir, fluid, well_manager, config['simulation'])
    
    time_step_days = config['simulation']['time_step_days']
    time_step_sec = time_step_days * 86400
    
    # Замеряем время выполнения
    start_time = time.time()
    steps_completed = 0
    
    try:
        for i in range(num_steps):
            print(f"\n--- Шаг {i+1}/{num_steps} [{solver_type}] ---")
            converged = sim.run_step(dt=time_step_sec)
            if not converged:
                break
            steps_completed += 1
    except Exception as e:
        print(f"Ошибка при выполнении симуляции: {e}")
    
    elapsed_time = time.time() - start_time
    
    return {
        'solver_type': solver_type,
        'steps_completed': steps_completed,
        'total_time': elapsed_time,
        'time_per_step': elapsed_time / max(steps_completed, 1)
    }

def test_performance_comparison():
    """
    Сравнивает производительность полностью неявной и IMPES схем.
    """
    print("\n\n=== ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ ===")
    
    # Копируем конфигурацию для IMPES
    with open("configs/fully_implicit_2d.json", 'r') as f:
        config = json.load(f)
    
    # Используем одинаковый шаг по времени для справедливого сравнения
    config['simulation']['time_step_days'] = 1
    impes_config = config.copy()
    impes_config['simulation']['solver_type'] = 'impes'
    
    # Сохраняем временную конфигурацию для IMPES
    with open("configs/temp_impes_benchmark.json", 'w') as f:
        json.dump(impes_config, f, indent=4)
    
    # Запускаем бенчмарки
    num_steps = 10
    
    impes_results = run_simulation_benchmark("configs/temp_impes_benchmark.json", 'impes', num_steps)
    fully_implicit_results = run_simulation_benchmark("configs/fully_implicit_2d.json", 'fully_implicit', num_steps)
    
    # Удаляем временный файл
    os.remove("configs/temp_impes_benchmark.json")
    
    # Выводим результаты
    print("\n=== РЕЗУЛЬТАТЫ ТЕСТА ПРОИЗВОДИТЕЛЬНОСТИ ===")
    print(f"IMPES схема: {impes_results['steps_completed']}/{num_steps} шагов, {impes_results['total_time']:.2f} сек. ({impes_results['time_per_step']:.2f} сек/шаг)")
    print(f"Полностью неявная схема: {fully_implicit_results['steps_completed']}/{num_steps} шагов, {fully_implicit_results['total_time']:.2f} сек. ({fully_implicit_results['time_per_step']:.2f} сек/шаг)")
    
    # Сравниваем скорость
    if impes_results['time_per_step'] > 0 and fully_implicit_results['time_per_step'] > 0:
        speedup = impes_results['time_per_step'] / fully_implicit_results['time_per_step']
        print(f"Ускорение: {speedup:.2f}x" if speedup > 1 else f"Замедление: {1/speedup:.2f}x")
    
    # Проверяем, что оба решателя завершили хотя бы несколько шагов
    assert impes_results['steps_completed'] > 0, "IMPES схема не смогла выполнить ни одного шага"
    assert fully_implicit_results['steps_completed'] > 0, "Полностью неявная схема не смогла выполнить ни одного шага" 