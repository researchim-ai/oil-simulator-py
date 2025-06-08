import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import Well, WellManager
from simulator.simulation import Simulator

def plot_results(pressure, saturation, filename_prefix):
    """
    Сохраняет 2D-карты давления и насыщенности в файлы.
    """
    # Мы будем визуализировать центральный слой по оси Z
    p_slice = pressure[:, :, int(pressure.shape[2] / 2)]
    s_slice = saturation[:, :, int(saturation.shape[2] / 2)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(p_slice.T / 1e6, cmap='jet', origin='lower')
    ax1.set_title('Давление (МПа)')
    ax1.set_xlabel('Ячейка X')
    ax1.set_ylabel('Ячейка Y')
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(s_slice.T, cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax2.set_title('Насыщенность водой')
    ax2.set_xlabel('Ячейка X')
    ax2.set_ylabel('Ячейка Y')
    fig.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}.png")
    print(f"\nРезультаты сохранены в файл {filename_prefix}.png")

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

    # Параметры для нашей первой модели пласта
    dims = (100, 100, 10)  # 100x100 ячеек в длину/ширину, 10 в высоту
    grid_size = (20.0, 20.0, 5.0) # Размеры ячейки в метрах
    porosity = 0.2  # Пористость 20%
    permeability = 100.0 # Проницаемость 100 мД

    # Создаем экземпляр пласта
    reservoir = Reservoir(
        dimensions=dims,
        grid_size=grid_size,
        porosity=porosity,
        permeability=permeability,
        device=device
    )

    # Параметры флюидов и начальные условия
    fluid_properties = {
        'oil': {'viscosity': 1.0, 'density': 850.0}, # вязкость в сП, плотность в кг/м^3
        'water': {'viscosity': 0.5, 'density': 1000.0},
        'compressibility': 4e-5 # 1/бар
    }
    initial_pressure = 200e5  # 200 бар в Паскалях
    initial_sw = 0.1 # начальная водонасыщенность 10%

    # Создаем экземпляр флюидной системы
    fluid_system = Fluid(
        reservoir=reservoir,
        initial_pressure=initial_pressure,
        initial_sw=initial_sw,
        fluid_properties=fluid_properties
    )

    # Создание и настройка скважин
    well_manager = WellManager()

    # Добывающая скважина в углу (25, 25)
    producer = Well(
        name="PROD-1",
        location=(25, 25, 5), # i, j, k
        well_type="producer",
        rate=100.0 # 100 м^3/сутки
    )
    well_manager.add_well(producer)

    # Нагнетательная скважина в противоположном углу (75, 75)
    injector = Well(
        name="INJ-1",
        location=(75, 75, 5), # i, j, k
        well_type="injector",
        rate=-100.0 # -100 м^3/сутки (отрицательный для нагнетания)
    )
    well_manager.add_well(injector)

    # Инициализация симулятора
    sim = Simulator(
        reservoir=reservoir,
        fluid_system=fluid_system,
        well_manager=well_manager
    )

    # Параметры симуляции
    time_step = 30 * 24 * 3600  # 30 дней в секундах
    num_steps = 10

    print(f"\nЗапуск симуляции на {num_steps} шагов по {time_step / (24*3600):.0f} дней...")

    # Запускаем цикл симуляции
    for i in tqdm(range(num_steps), desc="Симуляция"):
        sim.run_step(dt=time_step)

    print("\nСимуляция завершена.")
    final_pressure = sim.fluid.pressure.cpu().numpy()
    final_sw = sim.fluid.s_w.cpu().numpy()
    print(f"Итоговое давление: Мин={final_pressure.min()/1e6:.2f} МПа, Макс={final_pressure.max()/1e6:.2f} МПа")
    print(f"Итоговая водонасыщенность: Мин={final_sw.min():.4f}, Макс={final_sw.max():.4f}")

    # Визуализация результатов
    plot_results(final_pressure, final_sw, "final_results")


if __name__ == "__main__":
    main()
