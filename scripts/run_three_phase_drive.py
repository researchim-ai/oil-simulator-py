import os, json, sys, time
sys.path.append(os.path.abspath("src"))

os.environ["OIL_SIM_SKIP_PATCHES"] = "1"  # отключаем заглушки

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator

CONFIG_PATH = os.path.join("configs", "three_phase_drive.json")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sim(cfg):
    res = Reservoir(cfg, device="cpu")
    fluid = Fluid(cfg["fluid"], res)
    wells = WellManager(cfg["wells"], res)
    sim_params = cfg["simulation"]
    return Simulator(res, fluid, wells, sim_params)


def main():
    cfg = load_config(CONFIG_PATH)
    sim = build_sim(cfg)

    # Используем встроенный драйвер симулятора, который сохраняет PNG
    # и формирует GIF-анимацию, а при need – VTK.
    print("Запускаем полную симуляцию с записью графиков и GIF...")
    sim.run(output_filename="three_phase_drive", save_vtk=True)


if __name__ == "__main__":
    main() 