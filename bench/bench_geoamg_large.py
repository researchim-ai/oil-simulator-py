import json, time, sys, os
import torch, numpy as np

# Добавляем путь проекта
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
# менеджер скважин и генератор случайных скважин
from simulator.well import WellManager
import random

# --------------------------------------------------
# Вспомогательная функция: генерация N скважин
# --------------------------------------------------
def make_wells(nx, ny, nz, *, n_wells=50,
               inj_fraction=0.3,
               inj_rate=80000.0,  # м³/сут, положительный
               prod_rate=-60000.0  # м³/сут, отрицательный
               ):
    """Создаёт список конфигураций скважин.

    - inj_fraction доля инжекторов (остальное – добывающие)
    - Для простоты все скважины контролируются по дебиту (rate).
      Инжекция: +500 м³/сут, добыча: −500 м³/сут.
    """
    wells = []
    n_inj = int(n_wells * inj_fraction)
    n_prod = n_wells - n_inj

    # Чтобы скважины не попадали точно на границу, берём диапазон [1, dim-2]
    def random_coord(dim):
        return random.randint(1, max(dim - 2, 1))

    for idx in range(n_inj):
        i = random_coord(nx)
        j = random_coord(ny)
        k = random_coord(nz)
        wells.append({
            "name": f"INJ{idx+1}",
            "type": "injector",
            "i": i,
            "j": j,
            "k": k,
            "radius": {"radius": 0.1, "well_index": 1e-3},  # явно задаём огромный WI
            "control_type": "bhp",
            "control_value": 30.0  # МПа
        })

    for idx in range(n_prod):
        i = random_coord(nx)
        j = random_coord(ny)
        k = random_coord(nz)
        wells.append({
            "name": f"PROD{idx+1}",
            "type": "producer",
            "i": i,
            "j": j,
            "k": k,
            "radius": {"radius": 0.1, "well_index": 1e-3},
            "control_type": "bhp",
            "control_value": 10.0  # МПа
        })

    return wells

from simulator.simulation import Simulator
from solver.cpr import CPRPreconditioner


def make_reservoir(nx, ny, nz):
    """Генерирует однородный Reservoir с заданными размерами."""
    # одинаковая проницаемость и пористость
    grid_size = (20.0, 20.0, 5.0)
    perm = 100.0  # мД
    poro = 0.2
    rock_compr = 1e-5

    res = Reservoir(dimensions=(nx, ny, nz), grid_size=grid_size,
                     permeability=perm, porosity=poro, rock_compressibility=rock_compr)
    return res


def make_fluid(res):
    fluid_cfg = {
        "pressure": 20.0,   # МПа
        "s_w": 0.2,
        "mu_oil": 1.0,
        "mu_water": 0.5,
        "mu_gas": 0.05,
        "rho_oil": 850.0,
        "rho_water": 1000.0,
        "rho_gas": 150.0,
    }
    fluid = Fluid(fluid_cfg, res, device=device)
    return fluid


def bench_case(nx, ny, nz, mode="fi", steps=100):
    n_cells = nx * ny * nz
    print(f"\n===== Case {nx}x{ny}x{nz}  (N={n_cells/1e6:.2f} M) mode={mode} =====")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_cfg = {
        "dimensions": [nx, ny, nz],
        "grid_size": [20.0, 20.0, 5.0],
        "permeability": 100.0,
        "k_vertical_fraction": 0.1,
        "porosity": 0.2,
        "c_rock": 1e-5,
    }
    res = Reservoir(res_cfg, device=device)
    fluid_cfg = {
        "pressure": 20.0,   # МПа
        "s_w": 0.2,
        "mu_oil": 1.0,
        "mu_water": 0.5,
        "mu_gas": 0.05,
        "rho_oil": 850.0,
        "rho_water": 1000.0,
        "rho_gas": 150.0,
    }
    fluid = Fluid(fluid_cfg, res, device=device)
    # Генерируем ~50 скважин для динамики
    well_cfgs = make_wells(nx, ny, nz, n_wells=50,
                           inj_fraction=0.3,
                           inj_rate=80000.0,
                           prod_rate=-60000.0)
    wells = WellManager(well_cfgs, res)

    if mode == "fi":
        sim_params = {
            "solver_type": "fully_implicit",
            "jacobian": "jfnk",
            "backend": "geo",  # GeoAMG
            "total_time_days": steps*1.0,
            "time_step_days": 1.0,
            "verbose": True,
            "use_cuda": device.type == "cuda",
        }
    else:  # IMPES
        sim_params = {
            "solver_type": "impes",
            "total_time_days": steps*1.0,
            "time_step_days": 1.0,
            "verbose": True,
            "use_cuda": device.type == "cuda",
        }

    sim = Simulator(res, fluid, wells, sim_params, device=device)

    # --- Выполним несколько шагов, чтобы увидеть динамику --
    dt_sec = sim_params["time_step_days"] * 86400.0
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    for step in range(steps):
        ok = sim.run_step(dt_sec)
        if not ok:
            print(f"⚠️  Шаг {step+1} не сошёлся – прерываем кейс")
            break
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"{steps} шагов выполнено, elapsed={time.time()-t0:.2f}s, GPU={device.type=='cuda'}")


if __name__ == "__main__":
    # 🔬 Быстрый прогон на маленькой модели
    bench_case(60, 60, 30, mode="fi", steps=100)
    bench_case(60, 60, 30, mode="impes", steps=100)

    # 💪 Большая модель – только FI, чтобы было интереснее
    bench_case(100, 100, 100, mode="fi", steps=100) 