#!/usr/bin/env python3
"""
Пример запуска трёхфазного сценария с экзолюцией газа.
"""
import json, sys, os
from pathlib import Path

# Добавляем корень src/ в PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / 'src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator

CONFIG = REPO_ROOT / 'configs' / 'gas_exsolution_3d.json'

def main():
    with open(CONFIG) as f:
        cfg = json.load(f)

    reservoir = Reservoir(cfg['grid'])
    fluid     = Fluid(cfg['fluid'], reservoir)
    wells     = WellManager(cfg['wells'], reservoir)
    sim       = Simulator(reservoir, fluid, wells, cfg['simulation'])

    sim.run(output_filename='gas_exsolution', save_vtk=True)

if __name__ == '__main__':
    main() 