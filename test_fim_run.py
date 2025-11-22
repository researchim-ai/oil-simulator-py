import sys
import os
import json
import torch
import time
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator

def run_fim_test():
    print("=== Running Fully Implicit Simulation Test ===")
    
    # Load Config
    config_path = 'configs/fully_implicit_2d.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # PATCH: Force Rate control for Producer until BHP is implemented
    print("NOTE: Patching wells to use low RATE control for stability testing")
    for well in config['wells']:
        if well['type'] == 'producer':
            well['control_type'] = 'rate'
            well['control_value'] = 1.0 # m3/day (Reduced from 50.0)
        elif well['type'] == 'injector':
            well['control_value'] = 1.0 # m3/day (Reduced from 25.0)
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize Components
    reservoir = Reservoir(config['reservoir'], device=device)
    fluid = Fluid(config['fluid'], reservoir, device=device)
    well_manager = WellManager(config['wells'], reservoir)
    
    # Initialize Simulator
    sim_params = config['simulation']
    simulator = Simulator(reservoir, fluid, well_manager, sim_params, device=device)
    
    # Run Loop
    t = 0.0
    dt = sim_params['time_step_days'] * 86400.0
    t_end = sim_params['total_time_days'] * 86400.0
    
    step = 0
    start_time = time.time()
    
    pressures = []
    times = []
    
    try:
        while t < t_end:
            print(f"\n--- Time Step {step+1}, Time: {t/86400:.2f} days ---")
            success = simulator.run_step(dt)
            
            if not success:
                print("Simulation failed to converge.")
                break
                
            t += dt
            step += 1
            
            # Record Average Pressure
            avg_p = fluid.pressure.mean().item() / 1e6 # MPa
            pressures.append(avg_p)
            times.append(t/86400.0)
            
            print(f"Step completed. Avg Pressure: {avg_p:.2f} MPa")
            
            if step >= 10: # Run 10 steps for demonstration
                print("Test limit reached (10 steps).")
                break
                
    except KeyboardInterrupt:
        print("Simulation interrupted.")
        
    elapsed = time.time() - start_time
    print(f"\nSimulation finished in {elapsed:.2f}s")
    print(f"Speed: {step/elapsed:.2f} steps/sec")

if __name__ == "__main__":
    run_fim_test()

