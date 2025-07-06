import numpy as np
from pyevtk.hl import gridToVTK
import os

def save_to_vtk(reservoir, fluid, filename="simulation_results"):
    """
    Сохраняет данные симуляции в файл формата VTK для 3D-визуализации.

    :param reservoir: Экземпляр класса Reservoir.
    :param fluid: Экземпляр класса Fluid.
    :param filename: Базовое имя выходного файла (без расширения).
    """
    print(f"\nСохранение результатов в формате VTK...")
    
    nx, ny, nz = reservoir.dimensions
    dx, dy, dz = reservoir.grid_size.cpu().numpy()

    # Создаем координаты узлов сетки.
    # VTK ожидает координаты узлов, поэтому массив должен быть на 1 больше, чем число ячеек.
    x = np.arange(0, (nx + 1) * dx, dx)
    y = np.arange(0, (ny + 1) * dy, dy)
    z = np.arange(0, (nz + 1) * dz, dz)

    # Получаем данные с устройства и конвертируем в NumPy
    pressure_mpa = fluid.pressure.cpu().numpy() / 1e6 # Конвертируем в МПа для удобства
    saturation_w = fluid.s_w.cpu().numpy()
    saturation_o = fluid.s_o.cpu().numpy()
    
    # pyevtk требует, чтобы данные были в формате Fortran order (по колонкам)
    # и соответствовали размеру (nx, ny, nz). Наши тензоры уже в правильном порядке.
    pressure_reshaped = pressure_mpa.reshape((nx, ny, nz), order='F')
    saturation_w_reshaped = saturation_w.reshape((nx, ny, nz), order='F')
    saturation_o_reshaped = saturation_o.reshape((nx, ny, nz), order='F')
    # Используем доступные тензоры проницаемости
    perm_h_reshaped = reservoir.permeability_x.cpu().numpy().reshape((nx, ny, nz), order='F')
    perm_v_reshaped = reservoir.permeability_z.cpu().numpy().reshape((nx, ny, nz), order='F')
    
    # Создаем словарь с данными для ячеек
    cell_data = {
        "Pressure_MPa": pressure_reshaped,
        "Water_Saturation": saturation_w_reshaped,
        "Oil_Saturation": saturation_o_reshaped,
        "Permeability_H": perm_h_reshaped,
        "Permeability_V": perm_v_reshaped
    }
    
    # Если пользователь передал путь с каталогами, считаем его полным.
    if os.path.dirname(filename):
        filepath = filename
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    else:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)

    # Используем gridToVTK для сохранения данных
    gridToVTK(filepath, x, y, z, cellData=cell_data)
    
    print(f"Результаты успешно сохранены в файл {filepath}.vts") 