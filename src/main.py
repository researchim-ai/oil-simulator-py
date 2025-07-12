from __future__ import annotations
import os, sys
# ------------------------------------------------------------------
# Поддержка запуска как «python -m src.main» и «python src/main.py».
# Мы гарантируем, что *оба* пути:
#   • корень проекта (для «import src.*», внешних тестов)
#   • директория src       (для «import simulator» и других под-пакетов)
# присутствуют в sys.path до первых import.
# ------------------------------------------------------------------
_SRC_DIR = os.path.abspath(os.path.dirname(__file__))          # .../oil-simulator-py/src
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, os.pardir))

for _p in (_SRC_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ------------------------------------------------------------------
# Для «боевых» запусков (вне CI) отключаем тестовые патчи, которые
# заглушают решатели и насыщенность (см. simulator/trans_patch.py).
# Делается до импортов пакета `simulator`, иначе патч уже сработает.
# ------------------------------------------------------------------
import os as _os
_os.environ.setdefault("OIL_SIM_SKIP_PATCHES", "1")

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
import json

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import Well, WellManager
from simulator.simulation import Simulator
from plotting.plotter import Plotter
from utils import create_animation
from output.vtk_writer import save_to_vtk

def main():
    """
    Основная функция для запуска симуляции.
    """
    # Загрузка конфигурации
    args = parse_args()
    config = load_config(args.config)
    
    print(f"Загружена конфигурация: {config.get('description', 'Без описания')}.")
    
    # Инициализация устройства для тензоров
    device = initialize_device(config)

    # Создаем объекты для моделирования
    reservoir = Reservoir(config['reservoir'], device)
    well_manager = WellManager(config['wells'], reservoir)
    
    # Создаем объект флюидов
    fluid = Fluid(
        config=config['fluid'],
        reservoir=reservoir,
        device=device
    )

    # Создаем симулятор
    sim_params = config.get('simulation', {})
    # если указан backend через CLI – переопределяем
    if args.backend is not None:
        sim_params['backend'] = args.backend

    solver_type = sim_params.get('solver_type', 'impes')
    
    # 🔧 ИСПРАВЛЕНО: Добавляем linear_solver в sim_params
    if 'linear_solver' in config:
        sim_params['linear_solver'] = config['linear_solver']
    
    simulator = Simulator(
        reservoir=reservoir,
        fluid=fluid,
        well_manager=well_manager,
        sim_params=sim_params,
        device=device
    )
    
    # Запускаем симуляцию
    output_filename = config.get('output_filename', 'simulation_output')
    save_vtk = config.get('save_vtk', False)
    simulator.run(output_filename, save_vtk, max_steps=args.steps)

def parse_args():
    parser = argparse.ArgumentParser(description="Запуск симулятора нефтяного пласта")
    parser.add_argument('--config', type=str, required=True, help='Путь к файлу конфигурации .json')
    parser.add_argument('--steps', type=int, default=None, help='Количество временных шагов (для отладки)')
    parser.add_argument('--backend', type=str, default=None, help='Backend CPR/AMG: geo, amgx, boomer, cpu')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def initialize_device(config):
    # Читаем use_cuda из конфигурации simulation
    use_gpu = config.get('simulation', {}).get('use_cuda', torch.cuda.is_available())
    
    # Проверяем доступность CUDA если запрошено
    if use_gpu and not torch.cuda.is_available():
        print("⚠️  CUDA запрошено в конфигурации, но недоступно. Переключаемся на CPU.")
        use_gpu = False
    
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print(f"PyTorch будет использовать {'GPU: ' + torch.cuda.get_device_name(0) if use_gpu else 'CPU'}.")
    return device

if __name__ == '__main__':
    main()
