import sys
import os
import json
import torch
import numpy as np
import scipy.sparse as sp

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.simulator.reservoir import Reservoir
from src.simulator.fluid import Fluid
from src.simulator.well import WellManager
from src.simulator.fully_implicit_solver import FullyImplicitSolver
from src.simulator.manual_assembly_solver import ManualFIMSolver

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def compare_solvers():
    print("=== FIM Solver Comparison (Autograd vs Manual) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Config
    config_path = 'configs/test_fim_compare.json'
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    config = load_config(config_path)
    
    # --- Initialize Common Objects ---
    reservoir = Reservoir(config['reservoir'])
    # Ensure reservoir tensors are double for comparison
    reservoir.porosity = reservoir.porosity.double()
    reservoir.permeability_x = reservoir.permeability_x.double()
    reservoir.permeability_y = reservoir.permeability_y.double()
    reservoir.permeability_z = reservoir.permeability_z.double()
    
    fluid = Fluid(config['fluid'], reservoir)
    # Init fluid state
    p_init = config['reservoir']['pressure_initial'] * 1e6 # MPa -> Pa
    sw_init = config['reservoir']['sw_initial']
    
    fluid.pressure = torch.full((reservoir.nx, reservoir.ny, reservoir.nz), p_init, device=device, dtype=torch.float64)
    fluid.s_w = torch.full((reservoir.nx, reservoir.ny, reservoir.nz), sw_init, device=device, dtype=torch.float64)
    fluid.s_o = 1.0 - fluid.s_w
    fluid.s_g = torch.zeros_like(fluid.s_w)
    
    well_manager = WellManager(config['wells'], reservoir)
    
    dt = 1.0 * 86400.0 # 1 day in seconds
    
    # --- Autograd Solver ---
    print("\n--- Running Autograd Solver ---")
    auto_solver = FullyImplicitSolver(reservoir, fluid, well_manager, {})
    
    # Prepare state for autograd
    p_curr = fluid.pressure.clone().detach().requires_grad_(True)
    sw_curr = fluid.s_w.clone().detach().requires_grad_(True)
    
    # Pre-calculate masses (same as in step method)
    with torch.no_grad():
        mass_o_old, mass_w_old = auto_solver.calc_masses(p_curr, sw_curr)
    
    # 1. Compute Residuals
    res_o, res_w = auto_solver.compute_residuals_autograd(p_curr, sw_curr, mass_o_old, mass_w_old, dt)
    R_auto = torch.cat([res_o.view(-1), res_w.view(-1)])
    
    # 2. Compute Jacobian
    print("Computing Autograd Jacobian...")
    def residual_func(p, sw):
        ro, rw = auto_solver.compute_residuals_autograd(p, sw, mass_o_old, mass_w_old, dt)
        return torch.cat([ro.view(-1), rw.view(-1)])
    
    J_tuple = torch.autograd.functional.jacobian(residual_func, (p_curr, sw_curr))
    # J_tuple is (dR/dP, dR/dSw). Each is (2N, N, N, N). Reshape to (2N, N).
    n_cells = auto_solver.n_cells
    J_dP = J_tuple[0].reshape(2*n_cells, n_cells)
    J_dSw = J_tuple[1].reshape(2*n_cells, n_cells)
    J_auto = torch.cat([J_dP, J_dSw], dim=1) # (2N, 2N)
    
    print(f"Autograd Residual Norm: {R_auto.norm().item():.6e}")
    print(f"Autograd Jacobian Norm: {J_auto.norm().item():.6e}")
    
    # --- Manual Solver ---
    print("\n--- Running Manual Solver ---")
    manual_solver = ManualFIMSolver(reservoir, fluid, well_manager, {})
    
    # 1. Assemble System
    # Note: Manual solver uses accumulated masses internally, so we pass p, sw and old masses
    # Need to ensure mass calculation is identical. 
    # ManualFIMSolver.calc_masses is almost identical to FullyImplicitSolver.calc_masses
    
    # We use the same p_curr, sw_curr values (detached)
    p_manual = p_curr.detach()
    sw_manual = sw_curr.detach()
    
    # Recalculate old masses using ManualSolver to be consistent with its internal logic
    m_o_old_man, m_w_old_man = manual_solver.calc_masses(p_manual, sw_manual)
    
    # Assemble
    R_manual_raw, J_manual_vals = manual_solver.assemble_system(p_manual, sw_manual, m_o_old_man, m_w_old_man, dt)
    
    # 2. Convert to Block Format for comparison
    # Manual R is [O1, W1, O2, W2...]
    # Autograd R is [O1..ON, W1..WN]
    
    R_manual_O = R_manual_raw[0::2]
    R_manual_W = R_manual_raw[1::2]
    R_manual = torch.cat([R_manual_O, R_manual_W])
    
    print(f"Manual Residual Norm: {R_manual.norm().item():.6e}")
    
    # 3. Convert Jacobian to Dense Block Format
    # J_manual_vals is (NNZ, 2, 2). Blocks are [[dRo/dP, dRo/dSw], [dRw/dP, dRw/dSw]]
    indptr = manual_solver.builder.indptr.cpu().numpy()
    indices = manual_solver.builder.indices.cpu().numpy()
    data = J_manual_vals.cpu().numpy()
    
    bsr = sp.bsr_matrix((data, indices, indptr), shape=(2*n_cells, 2*n_cells))
    J_manual_dense_np = bsr.todense()
    J_manual_dense_interleaved = torch.from_numpy(J_manual_dense_np).to(device)
    
    # Permute rows and cols to block format
    # Rows: even (Oil), odd (Water)
    rows_perm = torch.cat([torch.arange(0, 2*n_cells, 2), torch.arange(1, 2*n_cells, 2)])
    # Cols: even (P), odd (Sw)
    cols_perm = torch.cat([torch.arange(0, 2*n_cells, 2), torch.arange(1, 2*n_cells, 2)])
    
    J_manual_ordered = J_manual_dense_interleaved[rows_perm][:, cols_perm]
    
    print(f"Manual Jacobian Norm: {J_manual_ordered.norm().item():.6e}")
    
    # --- Comparison ---
    print("\n--- Comparison ---")
    diff_R = (R_auto - R_manual).norm().item()
    rel_diff_R = diff_R / (R_auto.norm().item() + 1e-10)
    print(f"Residual Difference L2: {diff_R:.6e} (Rel: {rel_diff_R:.6e})")
    
    diff_J = (J_auto - J_manual_ordered).norm().item()
    rel_diff_J = diff_J / (J_auto.norm().item() + 1e-10)
    print(f"Jacobian Difference L2: {diff_J:.6e} (Rel: {rel_diff_J:.6e})")
    
    # Always extract blocks for detailed check if requested or for FD check
    J_OO = J_auto[:n_cells, :n_cells]
    J_OW = J_auto[:n_cells, n_cells:]
    J_WO = J_auto[n_cells:, :n_cells]
    J_WW = J_auto[n_cells:, n_cells:]
    
    M_OO = J_manual_ordered[:n_cells, :n_cells]
    M_OW = J_manual_ordered[:n_cells, n_cells:]
    M_WO = J_manual_ordered[n_cells:, :n_cells]
    M_WW = J_manual_ordered[n_cells:, n_cells:]

    if rel_diff_R > 1e-5:
        print("WARNING: Residuals differ significantly!")
        # ...
        
    if rel_diff_J > 1e-5:
        print("WARNING: Jacobians differ significantly!")
        # Check blocks
        
        dOO = (J_OO - M_OO).norm().item() / (J_OO.norm().item() + 1e-10)
        dOW = (J_OW - M_OW).norm().item() / (J_OW.norm().item() + 1e-10)
        dWO = (J_WO - M_WO).norm().item() / (J_WO.norm().item() + 1e-10)
        dWW = (J_WW - M_WW).norm().item() / (J_WW.norm().item() + 1e-10)
        
        print(f"  Block dRo/dP  Rel Diff: {dOO:.6e}")
        print(f"  Block dRo/dSw Rel Diff: {dOW:.6e}")
        print(f"  Block dRw/dP  Rel Diff: {dWO:.6e}")
        print(f"  Block dRw/dSw Rel Diff: {dWW:.6e}")
        
        print(f"  DEBUG Norms:")
        print(f"    Auto J_OO: {J_OO.norm().item():.6e}")
        print(f"    Man  J_OO: {M_OO.norm().item():.6e}")
        print(f"    Auto J_WO: {J_WO.norm().item():.6e}")
        print(f"    Man  J_WO: {M_WO.norm().item():.6e}")

    # --- Finite Difference Check for J_OO ---
    print("\n--- Finite Difference Check (Cell 0) ---")
    # Perturb P[0]
    epsilon = 1e-3
    p_pert = p_curr.detach().clone()
    p_pert[0,0,0] += epsilon
    
    # Autograd Residual at perturbed P
    ro_pert, rw_pert = auto_solver.compute_residuals_autograd(p_pert, sw_curr, mass_o_old, mass_w_old, dt)
    R_auto_pert = torch.cat([ro_pert.view(-1), rw_pert.view(-1)])
    
    # Finite Diff Column 0
    # dR/dP_0 approx (R(p+eps) - R(p)) / eps
    # We focus on Row 0 (Oil Eq for Cell 0) -> diagonal element
    dRo_dP_0_FD = (ro_pert[0,0,0] - res_o[0,0,0]) / epsilon
    print(f"FD dRo[0]/dP[0]: {dRo_dP_0_FD.item():.6e}")
    
    # Autograd Value
    print(f"Auto J_OO[0,0]: {J_OO[0,0].item():.6e}")
    
    # Manual Value
    # Need to map cell (0,0,0) to flat index. It is 0.
    # J_manual_ordered is (Oil0, Water0, Oil1...) ?
    # No, J_manual_ordered is Block Ordered:
    # Rows: 0..N (Oil), N..2N (Water)
    # Cols: 0..N (P), N..2N (Sw)
    print(f"Man  J_OO[0,0]: {M_OO[0,0].item():.6e}")

if __name__ == "__main__":
    compare_solvers()

