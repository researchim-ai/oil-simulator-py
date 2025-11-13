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
    print("\nСохранение результатов в формате VTK...")
    
    nx, ny, nz = reservoir.dimensions
    grid_size = reservoir.grid_size.detach().cpu().numpy() if hasattr(reservoir.grid_size, "detach") else np.asarray(reservoir.grid_size)
    dx, dy, dz = grid_size

    x = np.arange(0, (nx + 1) * dx, dx)
    y = np.arange(0, (ny + 1) * dy, dy)
    z = np.arange(0, (nz + 1) * dz, dz)

    pressure_mpa = (fluid.pressure.detach().cpu().numpy() / 1e6).reshape((nx, ny, nz), order='F')
    sw = fluid.s_w.detach().cpu().numpy().reshape((nx, ny, nz), order='F')
    so = fluid.s_o.detach().cpu().numpy().reshape((nx, ny, nz), order='F')

    kx = reservoir.permeability_x.detach().cpu().numpy().reshape((nx, ny, nz), order='F')
    ky = reservoir.permeability_y.detach().cpu().numpy().reshape((nx, ny, nz), order='F')
    kz = reservoir.permeability_z.detach().cpu().numpy().reshape((nx, ny, nz), order='F')
    kh = 0.5 * (kx + ky)

    cell_data = {
        "Pressure_MPa": pressure_mpa,
        "Water_Saturation": sw,
        "Oil_Saturation": so,
        "Perm_Kx_m2": kx,
        "Perm_Ky_m2": ky,
        "Perm_Kz_m2": kz,
        "Perm_Kh_m2": kh,
    }

    if hasattr(fluid, "s_g") and fluid.s_g is not None:
        sg = fluid.s_g.detach().cpu().numpy().reshape((nx, ny, nz), order='F')
        cell_data["Gas_Saturation"] = sg

    if getattr(fluid, "pvt", None) is not None:
        Bo = fluid._eval_pvt(fluid.pressure, 'Bo').detach().cpu().numpy().reshape((nx, ny, nz), order='F')
        Bw = fluid._eval_pvt(fluid.pressure, 'Bw').detach().cpu().numpy().reshape((nx, ny, nz), order='F')
        Bg = fluid._eval_pvt(fluid.pressure, 'Bg').detach().cpu().numpy().reshape((nx, ny, nz), order='F')
        cell_data.update({
            "Bo": Bo,
            "Bw": Bw,
            "Bg": Bg,
        })

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    gridToVTK(filepath, x, y, z, cellData=cell_data)
    print(f"Результаты успешно сохранены в файл {filepath}.vtr") 