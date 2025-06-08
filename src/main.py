import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import Well, WellManager
from simulator.simulation import Simulator

def save_results_to_txt(pressure, saturation, filename):
    """
    Сохраняет 2D-срезы данных в текстовый файл.
    """
    p_slice = pressure[:, :, int(pressure.shape[2] / 2)] / 1e6 # МПа
    s_slice = saturation[:, :, int(saturation.shape[2] / 2)]

    with open(filename, 'w') as f:
        f.write("# Pressure (MPa) slice\n")
        np.savetxt(f, p_slice, fmt='%.4f', delimiter='\t')
        f.write("\n# Water Saturation slice\n")
        np.savetxt(f, s_slice, fmt='%.4f', delimiter='\t')
    print(f"Числовые результаты сохранены в файл {filename}")

def plot_results(pressure, saturation, filename):
    """
    Сохраняет 2D-карты давления и насыщенности в файлы.
    """
    # Мы будем визуализировать центральный слой по оси Z
    p_slice = pressure[:, :, int(pressure.shape[2] / 2)]
    s_slice = saturation[:, :, int(saturation.shape[2] / 2)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(p_slice / 1e6, cmap='jet', origin='lower')
    ax1.set_title('Давление (МПа)')
    ax1.set_xlabel('Ячейка X')
    ax1.set_ylabel('Ячейка Y')
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(s_slice, cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax2.set_title('Насыщенность водой')
    ax2.set_xlabel('Ячейка X')
    ax2.set_ylabel('Ячейка Y')
    fig.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Графики сохранены в файл {filename}")

def main():
    """
    Главная функция для запуска симулятора.
    """
    print("Запуск симулятора нефтяного месторождения...")

    # Проверка доступности GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"PyTorch будет использовать GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("PyTorch будет использовать CPU.")

    # --- Параметры модели ---
    
    # УМЕНЬШЕННЫЕ ПАРАМЕТРЫ ДЛЯ БЫСТРОГО ТЕСТА
    grid_dimensions = (20, 20, 1)    # Уменьшенный 2D грид для скорости
    grid_size = (25.0, 25.0, 10.0)   # Размеры ячейки в метрах (dx, dy, dz)
    porosity = 0.2
    permeability = 100 # мД

    # Создаем экземпляр пласта
    reservoir = Reservoir(
        dimensions=grid_dimensions,
        grid_size=grid_size,
        porosity=porosity,
        permeability=permeability,
        device=device
    )

    print("Создан менеджер скважин.")
    well_manager = WellManager()
    
    # Координаты скважин, адаптированные под новый грид
    prod_coords = (5, 5, 0)
    inj_coords = (15, 15, 0)

    # Добывающая скважина
    well_manager.add_well(
        Well(
            name="PROD-1",
            well_type="producer",
            coordinates=prod_coords,
            reservoir_dimensions=grid_dimensions,
            rate=15.0  # м^3/сутки
        )
    )
    # Нагнетательная скважина
    well_manager.add_well(
        Well(
            name="INJ-1",
            well_type="injector",
            coordinates=inj_coords,
            reservoir_dimensions=grid_dimensions,
            rate=15.0  # м^3/сутки
        )
    )

    # Инициализация флюидной системы
    fluid_system = Fluid(
        reservoir=reservoir,
        mu_oil=1.0, mu_water=0.5,
        rho_oil=850.0, rho_water=1000.0,
        initial_pressure=200e5,
        initial_sw=0.1,
        cf=4e-10, # 4e-5 1/бар -> 4e-10 1/Па
        sw_cr=0.1,
        so_r=0.2,
        nw=2.0,
        no=2.0,
        krw_end=0.8,
        kro_end=0.9
    )

    # Инициализация симулятора
    sim = Simulator(
        reservoir=reservoir,
        fluid_system=fluid_system,
        well_manager=well_manager
    )

    # Параметры симуляции
    total_time_days = 30  # Уменьшенное время симуляции
    time_step_days = 1    # Увеличенный шаг, должно быть стабильно на малом гриде
    
    # Конвертация времени в секунды для расчетов
    total_time_sec = total_time_days * 86400
    time_step_sec = time_step_days * 86400
    num_steps = int(total_time_days / time_step_days)

    # Создаем директорию для результатов, если ее нет
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nЗапуск симуляции на {num_steps} шагов по {time_step_days} дней...")

    # Запускаем цикл симуляции
    for i in tqdm(range(num_steps), desc="Симуляция"):
        sim.run_step(dt=time_step_sec)

    print("\nСимуляция завершена.")
    final_pressure = sim.fluid.pressure.cpu().numpy()
    final_sw = sim.fluid.s_w.cpu().numpy()
    print(f"Итоговое давление: Мин={final_pressure.min()/1e6:.2f} МПа, Макс={final_pressure.max()/1e6:.2f} МПа")
    print(f"Итоговая водонасыщенность: Мин={final_sw.min():.4f}, Макс={final_sw.max():.4f}")

    # Сохранение и визуализация результатов
    txt_filename = os.path.join(results_dir, "final_results.txt")
    png_filename = os.path.join(results_dir, "final_results.png")
    
    save_results_to_txt(final_pressure, final_sw, txt_filename)
    plot_results(final_pressure, final_sw, png_filename)


if __name__ == "__main__":
    main()
