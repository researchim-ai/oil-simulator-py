import sys
import os
import json
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator

def test_manual_fim_integration():
    print("=== Testing Manual FIM Integration in Simulator ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    config_path = 'configs/test_manual_fim.json'
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 1. Initialize Objects
    reservoir = Reservoir(config['reservoir'], device=device)
    
    # Ensure fluid uses reservoir
    fluid = Fluid(config['fluid'], reservoir, device=device)
    
    well_manager = WellManager(config['wells'], reservoir)
    
    # 2. Initialize Simulator
    sim_params = config['simulation']
    simulator = Simulator(reservoir, fluid, well_manager, sim_params, device=device)
    
    # Verify solver type
    if simulator.solver_type != 'fully_implicit':
        print(f"Error: Solver type is {simulator.solver_type}, expected 'fully_implicit'")
        return
    
    if not hasattr(simulator, 'fim_solver'):
        print("Error: fim_solver not initialized!")
        return
        
    print("Simulator initialized successfully with ManualFIMSolver.")
    
    # 3. Run Simulation
    print("\n--- Starting Simulation ---")
    simulator.run(output_filename="test_manual_fim", save_vtk=False)
    print("\n--- Simulation Completed ---")

if __name__ == "__main__":
    test_manual_fim_integration()

