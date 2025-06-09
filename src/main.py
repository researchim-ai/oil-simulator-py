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

def main():
    """
    Основная функция для запуска симулятора нефтяного пласта.
    """
    parser = argparse.ArgumentParser(description="Запуск симулятора нефтяного пласта")
    parser.add_argument('--config', type=str, required=True, help='Путь к файлу конфигурации .json')
    args = parser.parse_args()

    # --- 1. Загрузка конфигурации ---
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print(f"Загружена конфигурация: {config['description']}")
    
    # Извлечение параметров из конфига
    res_params = config['reservoir']
    sim_params = config['simulation']
    fluid_params = config['fluid']
    well_params = config['wells']
    output_filename = config['output_filename']
    
    # --- 2. Настройка окружения ---
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print(f"PyTorch будет использовать {'GPU: ' + torch.cuda.get_device_name(0) if use_gpu else 'CPU'}.")

    # --- 3. Создание компонентов модели ---
    print("\nСоздание модели пласта...")
    reservoir = Reservoir(
        config=res_params,
        device=device
    )

    print("Создание менеджера скважин...")
    well_manager = WellManager()
    for well_info in well_params:
        well_manager.add_well(
            Well(
                name=well_info['name'],
                well_type=well_info['type'],
                coordinates=tuple(well_info['coordinates']),
                reservoir_dimensions=tuple(res_params['dimensions']),
                rate=well_info['rate']
            )
        )
    
    print("\nИнициализация флюидов и начальных условий...")
    fluid = Fluid(
        reservoir=reservoir,
        config=fluid_params,
        device=device
    )

    # --- 4. Запуск симуляции ---
    sim = Simulator(reservoir, fluid, well_manager)
    
    total_time_days = sim_params['total_time_days']
    time_step_days = sim_params['time_step_days']
    time_step_sec = time_step_days * 86400
    num_steps = int(total_time_days / time_step_days)
    save_interval = sim_params.get("save_interval", num_steps) # Сохранять каждые N шагов

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Создаем папку для промежуточных результатов, если нужно
    intermediate_results_dir = os.path.join(results_dir, "intermediate")
    if save_interval < num_steps:
        os.makedirs(intermediate_results_dir, exist_ok=True)
    
    print(f"\nЗапуск симуляции на {num_steps} шагов по {time_step_days} дней...")
    print(f"Результаты будут сохраняться каждые {save_interval} шагов.")

    plotter = Plotter(reservoir)

    for i in tqdm(range(num_steps), desc="Симуляция"):
        sim.run_step(dt=time_step_sec)
        
        # Сохранение промежуточных результатов
        if (i + 1) % save_interval == 0 and save_interval < num_steps:
            p_current = fluid.pressure.cpu().numpy()
            sw_current = fluid.s_w.cpu().numpy()
            
            time_info = f"День {int((i + 1) * time_step_days)}"
            filename = f"{output_filename}_step_{i+1}.png"
            filepath = os.path.join(intermediate_results_dir, filename)
            
            plotter.save_plots(p_current, sw_current, filepath, time_info=time_info)
    
    print("\nСимуляция завершена.")

    # --- 5. Сохранение и визуализация финальных результатов ---
    p_final = fluid.pressure.cpu().numpy()
    sw_final = fluid.s_w.cpu().numpy()
    
    print(f"Итоговое давление: Мин={p_final.min()/1e6:.2f} МПа, Макс={p_final.max()/1e6:.2f} МПа")
    print(f"Итоговая водонасыщенность: Мин={sw_final.min():.4f}, Макс={sw_final.max():.4f}")

    # Сохранение числовых данных
    results_txt_path = os.path.join(results_dir, f"{output_filename}.txt")
    with open(results_txt_path, 'w') as f:
        f.write("Final Pressure (MPa):\n")
        f.write(np.array2string(p_final/1e6, threshold=np.inf, formatter={'float_kind':lambda x: "%.2f" % x}))
        f.write("\n\nFinal Water Saturation:\n")
        f.write(np.array2string(sw_final, threshold=np.inf, formatter={'float_kind':lambda x: "%.4f" % x}))
    print(f"Числовые результаты сохранены в файл {results_txt_path}")

    # Сохранение финальных графиков
    final_plot_path = os.path.join(results_dir, f"{output_filename}_final.png")
    plotter.save_plots(p_final, sw_final, final_plot_path, time_info=f"День {total_time_days} (Final)")
    print(f"Финальные графики сохранены в файл {final_plot_path}")

    # --- 6. Создание анимации (если нужно) ---
    if save_interval < num_steps:
        animation_fps = sim_params.get("animation_fps", 5)
        if animation_fps > 0:
            gif_path = os.path.join(results_dir, f"{output_filename}.gif")
            print(f"\nСоздание анимации с {animation_fps} FPS...")
            create_animation(intermediate_results_dir, gif_path, fps=animation_fps)


if __name__ == '__main__':
    main()
