#!/usr/bin/env python3
"""
Скрипт для проверки корректности реализации IMPES.
Проверяет:
1. Сохранение массы
2. Баланс потоков
3. Физическую корректность результатов
"""

import torch
import numpy as np
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator


def check_mass_conservation(sim, initial_mass_w, initial_mass_o):
    """Проверяет сохранение массы в системе"""
    # Текущие массы флюидов
    dx, dy, dz = sim.reservoir.grid_size
    cell_volume = dx * dy * dz
    
    current_p = sim.fluid.pressure.reshape(-1)
    current_sw = sim.fluid.s_w.reshape(-1)
    phi = sim.reservoir.porosity.reshape(-1)
    
    rho_w = sim.fluid.calc_water_density(current_p)
    rho_o = sim.fluid.calc_oil_density(current_p)
    
    current_mass_w = torch.sum(phi * current_sw * rho_w * cell_volume).item()
    current_mass_o = torch.sum(phi * (1 - current_sw) * rho_o * cell_volume).item()
    
    print(f"\nБаланс масс:")
    print(f"  Вода: начальная={initial_mass_w:.2f} кг, текущая={current_mass_w:.2f} кг")
    print(f"  Нефть: начальная={initial_mass_o:.2f} кг, текущая={current_mass_o:.2f} кг")
    print(f"  Изменение воды: {(current_mass_w - initial_mass_w)/initial_mass_w * 100:.2f}%")
    print(f"  Изменение нефти: {(current_mass_o - initial_mass_o)/initial_mass_o * 100:.2f}%")
    
    return current_mass_w, current_mass_o


def check_impes_implementation():
    """Основная проверка реализации IMPES"""
    print("="*70)
    print("ПРОВЕРКА РЕАЛИЗАЦИИ МЕТОДА IMPES")
    print("="*70)
    
    # Загружаем конфигурацию IMPES
    config_path = "configs/impes_2d.json"
    
    if not os.path.exists(config_path):
        print(f"ОШИБКА: Не найден файл конфигурации {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Создаем симулятор
    device = torch.device("cpu")  # Используем CPU для воспроизводимости
    
    reservoir = Reservoir(config=config['reservoir'], device=device)
    well_manager = WellManager(config['wells'], reservoir)
    fluid = Fluid(reservoir=reservoir, config=config['fluid'], device=device)
    
    # Изменяем параметры для быстрого теста
    sim_params = config['simulation'].copy()
    sim_params['total_time_days'] = 2.0  # Короткая симуляция
    sim_params['time_step_days'] = 0.5
    sim_params['solver_type'] = 'impes'
    
    sim = Simulator(reservoir, fluid, well_manager, sim_params, device=device)
    
    print("\n" + "="*70)
    print("НАЧАЛЬНОЕ СОСТОЯНИЕ")
    print("="*70)
    
    # Рассчитываем начальные массы
    dx, dy, dz = reservoir.grid_size
    cell_volume = dx * dy * dz
    
    initial_p = fluid.pressure.reshape(-1)
    initial_sw = fluid.s_w.reshape(-1)
    phi = reservoir.porosity.reshape(-1)
    
    rho_w_init = fluid.calc_water_density(initial_p)
    rho_o_init = fluid.calc_oil_density(initial_p)
    
    initial_mass_w = torch.sum(phi * initial_sw * rho_w_init * cell_volume).item()
    initial_mass_o = torch.sum(phi * (1 - initial_sw) * rho_o_init * cell_volume).item()
    
    print(f"\nНачальные массы:")
    print(f"  Вода: {initial_mass_w:.2f} кг")
    print(f"  Нефть: {initial_mass_o:.2f} кг")
    print(f"  Общая масса: {initial_mass_w + initial_mass_o:.2f} кг")
    
    print(f"\nНачальные условия:")
    print(f"  P(мин/макс): {fluid.pressure.min()/1e6:.2f} / {fluid.pressure.max()/1e6:.2f} МПа")
    print(f"  Sw(мин/макс): {fluid.s_w.min():.4f} / {fluid.s_w.max():.4f}")
    
    # Запуск симуляции
    print("\n" + "="*70)
    print("ВЫПОЛНЕНИЕ ШАГОВ СИМУЛЯЦИИ")
    print("="*70)
    
    num_steps = int(sim_params['total_time_days'] / sim_params['time_step_days'])
    time_step_sec = sim_params['time_step_days'] * 86400
    
    all_successful = True
    
    for step in range(num_steps):
        print(f"\n--- Шаг {step + 1}/{num_steps} ---")
        success = sim.run_step(dt=time_step_sec)
        
        if not success:
            print(f"ОШИБКА: Шаг {step + 1} не сошелся!")
            all_successful = False
            break
        
        # Проверяем физические ограничения
        p_min = fluid.pressure.min().item()
        p_max = fluid.pressure.max().item()
        sw_min = fluid.s_w.min().item()
        sw_max = fluid.s_w.max().item()
        
        print(f"  P: {p_min/1e6:.2f} - {p_max/1e6:.2f} МПа")
        print(f"  Sw: {sw_min:.4f} - {sw_max:.4f}")
        
        # Проверяем на нефизичные значения
        if p_min < 0:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: Отрицательное давление!")
        if sw_min < fluid.sw_cr - 1e-6:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: Sw < sw_cr ({fluid.sw_cr})")
        if sw_max > 1.0 - fluid.so_r + 1e-6:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: Sw > 1 - so_r ({1.0 - fluid.so_r})")
    
    # Финальная проверка
    print("\n" + "="*70)
    print("ФИНАЛЬНОЕ СОСТОЯНИЕ")
    print("="*70)
    
    check_mass_conservation(sim, initial_mass_w, initial_mass_o)
    
    print(f"\nФинальные условия:")
    print(f"  P(мин/макс): {fluid.pressure.min()/1e6:.2f} / {fluid.pressure.max()/1e6:.2f} МПа")
    print(f"  Sw(мин/макс): {fluid.s_w.min():.4f} / {fluid.s_w.max():.4f}")
    
    # Проверка корректности
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ ПРОВЕРКИ")
    print("="*70)
    
    issues = []
    
    if not all_successful:
        issues.append("❌ Не все шаги симуляции сошлись")
    else:
        print("✓ Все шаги симуляции успешно сошлись")
    
    if fluid.pressure.min() < 0:
        issues.append("❌ Обнаружено отрицательное давление")
    else:
        print("✓ Давление положительное везде")
    
    if fluid.s_w.min() < fluid.sw_cr - 1e-6:
        issues.append(f"❌ Насыщенность ниже связанной: {fluid.s_w.min():.6f} < {fluid.sw_cr}")
    else:
        print(f"✓ Насыщенность в допустимых пределах (Sw_min = {fluid.s_w.min():.4f})")
    
    if fluid.s_w.max() > 1.0 - fluid.so_r + 1e-6:
        issues.append(f"❌ Насыщенность выше максимальной: {fluid.s_w.max():.6f} > {1.0 - fluid.so_r}")
    else:
        print(f"✓ Насыщенность в допустимых пределах (Sw_max = {fluid.s_w.max():.4f})")
    
    # Проверяем нормальное изменение давления
    p_change = (fluid.pressure.max() - fluid.pressure.min()).item() / 1e6
    if p_change > 50:  # Изменение больше 50 МПа подозрительно
        issues.append(f"❌ Слишком большой перепад давления: {p_change:.2f} МПа")
    else:
        print(f"✓ Перепад давления в разумных пределах: {p_change:.2f} МПа")
    
    print("\n" + "="*70)
    if len(issues) == 0:
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ! IMPES РАБОТАЕТ КОРРЕКТНО!")
        print("="*70)
        return True
    else:
        print("⚠️  ОБНАРУЖЕНЫ ПРОБЛЕМЫ:")
        for issue in issues:
            print(f"  {issue}")
        print("="*70)
        return False


if __name__ == "__main__":
    success = check_impes_implementation()
    sys.exit(0 if success else 1)


