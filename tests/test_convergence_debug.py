import sys
import os
import json
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.manual_assembly_solver import ManualFIMSolver

def test_convergence():
    print("=== Debugging Manual FIM Convergence ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Config similar to test_manual_fim.json but smaller/simpler
    config = {
        "reservoir": {
            "nx": 3, "ny": 3, "nz": 1,
            "grid_size": [10.0, 10.0, 10.0],
            "porosity": 0.2,
            "permeability": 100.0,
            "pressure_initial": 20.0,
            "sw_initial": 0.2
        },
        "fluid": {
            "mu_oil": 1.0, "mu_water": 0.5, "mu_gas": 0.02,
            "rho_oil": 850.0, "rho_water": 1000.0, "rho_gas": 1.0,
            "c_oil": 1e-5, "c_water": 1e-5, "c_gas": 1e-3,
            "relative_permeability": {
                "model": "corey", "nw": 2.0, "no": 2.0
            }
        },
        "wells": [
            {
                "name": "INJ", "type": "injector",
                "i": 0, "j": 0, "k": 0,
                "radius": 0.1,
                "control_type": "rate", "control_value": 10.0, # Small rate
                "injected_phase": "water"
            }
        ],
        "simulation": {
            "newton_max_iter": 10,
            "newton_tol_mass": 1e-3
        }
    }
    
    reservoir = Reservoir(config['reservoir'], device=device)
    fluid = Fluid(config['fluid'], reservoir, device=device)
    well_manager = WellManager(config['wells'], reservoir)
    
    solver = ManualFIMSolver(reservoir, fluid, well_manager, config['simulation'])
    
    dt = 1.0 * 86400.0
    
    print(f"\n--- Testing Step with dt={dt/86400} days ---")
    
    # Initial State
    p = fluid.pressure.clone().double()
    sw = fluid.s_w.clone().double()
    m_o_old, m_w_old = solver.calc_masses(p, sw)
    
    # Iteration 1
    print("\nIteration 1:")
    R, J_vals = solver.assemble_system(p, sw, m_o_old, m_w_old, dt)
    norm_R = R.norm().item()
    print(f"  Residual Norm: {norm_R:.4e}")
    
    # Solve
    indptr = solver.builder.indptr.cpu().numpy()
    indices = solver.builder.indices.cpu().numpy()
    vals_np = J_vals.cpu().numpy()
    n_blocks = solver.n_cells
    bsr = sp.bsr_matrix((vals_np, indices, indptr), shape=(2*n_blocks, 2*n_blocks))
    rhs = -R.cpu().numpy()
    
    dx_np = spsolve(bsr, rhs)
    dx = torch.from_numpy(dx_np).to(device)
    
    dp = dx[0::2].reshape(reservoir.nx, reservoir.ny, reservoir.nz)
    dsw = dx[1::2].reshape(reservoir.nx, reservoir.ny, reservoir.nz)
    
    print(f"  Max dP: {dp.abs().max().item():.4e}")
    print(f"  Max dSw: {dsw.abs().max().item():.4e}")
    
    # Update
    p_new = p + dp
    sw_new = sw + dsw
    
    # Check Residual at p_new
    print("\nChecking Residual at p_new (Newton prediction):")
    R_new, _ = solver.assemble_system(p_new, sw_new, m_o_old, m_w_old, dt)
    norm_R_new = R_new.norm().item()
    print(f"  New Residual Norm: {norm_R_new:.4e}")
    
    if norm_R_new < norm_R:
        print("  SUCCESS: Residual decreased.")
    else:
        print("  FAILURE: Residual increased or stagnated.")
        
        # Debug linear system validity
        print("\n  Checking Linear System J*dx + R ?= 0")
        J_dx = bsr @ dx_np
        resid = J_dx + R.cpu().numpy()
        print(f"  Linear System Residual: {np.linalg.norm(resid):.4e}")

if __name__ == "__main__":
    test_convergence()

