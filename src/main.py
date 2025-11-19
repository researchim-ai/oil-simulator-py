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
    device = initialize_device()

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
    solver_type = sim_params.get('solver_type', 'impes')
    
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
    save_vtk_intermediate = config.get('save_vtk_intermediate', False)
    simulator.run(output_filename, save_vtk=save_vtk, save_vtk_intermediate=save_vtk_intermediate)

def parse_args():
    parser = argparse.ArgumentParser(description="Запуск симулятора нефтяного пласта")
    parser.add_argument('--config', type=str, required=True, help='Путь к файлу конфигурации .json')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def initialize_device():
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print(f"PyTorch будет использовать {'GPU: ' + torch.cuda.get_device_name(0) if use_gpu else 'CPU'}.")
    return device

if __name__ == '__main__':
    main()
