import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator

def run_simulation(sim_type, steps, dt, config_base):
    print(f"\n--- Running {sim_type.upper()} Simulation ---")
    
    # Deep copy config to avoid side effects
    config = copy.deepcopy(config_base)
    config['simulation']['solver_type'] = sim_type
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Components
    reservoir = Reservoir(config['reservoir'], device=device)
    fluid = Fluid(config['fluid'], reservoir, device=device)
    well_manager = WellManager(config['wells'], reservoir)
    
    # Initialize Simulator
    sim_params = config['simulation']
    simulator = Simulator(reservoir, fluid, well_manager, sim_params, device=device)
    
    # Run Loop
    t = 0.0
    dt_sec = dt * 86400.0
    
    for step in range(steps):
        print(f"Step {step+1}/{steps}", end='\r')
        success = simulator.run_step(dt_sec)
        if not success:
            print(f"\n{sim_type} failed at step {step+1}")
            return None, None
            
    print(f"\n{sim_type} completed.")
    return fluid.pressure.detach().cpu().numpy(), fluid.s_w.detach().cpu().numpy()

def main():
    # Common Configuration
    common_config = {
        "simulation": {
            "total_time_days": 10,
            "time_step_days": 0.5, # Will be overridden
            "newton_max_iter": 20,
            "newton_tolerance": 1e-2,
            "dt_increase_factor": 1.0, # Fixed step for comparison
            "dt_reduction_factor": 1.0,
            "max_time_step_attempts": 1,
            "use_cuda": True
        },
        "reservoir": {
            "dimensions": [30, 30, 1],
            "grid_size": [10.0, 10.0, 5.0],
            "porosity": 0.2,
            "permeability": 100,
            "k_vertical_fraction": 0.1
        },
        "fluid": {
            "pressure": 20.0,
            "s_w": 0.2,
            "mu_oil": 1.0,
            "mu_water": 0.5,
            "rho_oil": 850.0,
            "rho_water": 1000.0,
            "c_oil": 1e-5,
            "c_water": 1e-5,
            "c_rock": 0.0,
            "s_wr": 0.2,
            "s_or": 0.2,
            "pc_scale": 0.0
        },
        "wells": [
            {
                "name": "Inj", "type": "injector", "i": 5, "j": 15, "k": 0,
                "radius": 0.1, "control_type": "rate", "control_value": 1.0
            },
            {
                "name": "Prod", "type": "producer", "i": 25, "j": 15, "k": 0,
                "radius": 0.1, "control_type": "rate", "control_value": 1.0
            }
        ]
    }

    # Run Parameters
    n_days = 4.0 # Reduced to ensure FIM finishes
    dt_fim = 0.5 # FIM can take large steps
    dt_impes = 0.05 # IMPES needs small steps for stability
    
    steps_fim = int(n_days / dt_fim)
    steps_impes = int(n_days / dt_impes)
    
    # 1. Run IMPES
    p_impes, sw_impes = run_simulation('impes', steps_impes, dt_impes, common_config)
    
    # 2. Run FIM
    p_fim, sw_fim = run_simulation('fully_implicit', steps_fim, dt_fim, common_config)
    
    if p_impes is None or p_fim is None:
        print("Simulation failed.")
        return

    # 3. Compare
    p_diff = np.abs(p_impes - p_fim) / 1e6 # MPa
    sw_diff = np.abs(sw_impes - sw_fim)
    
    print("\n=== Comparison Results ===")
    print(f"Max Pressure Diff: {p_diff.max():.4f} MPa")
    print(f"Mean Pressure Diff: {p_diff.mean():.4f} MPa")
    print(f"Max Saturation Diff: {sw_diff.max():.4f}")
    print(f"Mean Saturation Diff: {sw_diff.mean():.4f}")
    
    # Visual check
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.title(f"IMPES Pressure (dt={dt_impes})")
    plt.imshow(p_impes[:,:,0]/1e6)
    plt.colorbar()
    
    plt.subplot(2, 3, 2)
    plt.title(f"FIM Pressure (dt={dt_fim})")
    plt.imshow(p_fim[:,:,0]/1e6)
    plt.colorbar()
    
    plt.subplot(2, 3, 3)
    plt.title("Diff (MPa)")
    plt.imshow(p_diff[:,:,0])
    plt.colorbar()
    
    plt.subplot(2, 3, 4)
    plt.title(f"IMPES Sw")
    plt.imshow(sw_impes[:,:,0], vmin=0.2, vmax=0.8)
    plt.colorbar()
    
    plt.subplot(2, 3, 5)
    plt.title(f"FIM Sw")
    plt.imshow(sw_fim[:,:,0], vmin=0.2, vmax=0.8)
    plt.colorbar()
    
    plt.subplot(2, 3, 6)
    plt.title("Diff Sw")
    plt.imshow(sw_diff[:,:,0])
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('fim_vs_impes_verification.png')
    print("Comparison plot saved to fim_vs_impes_verification.png")

if __name__ == "__main__":
    main()

