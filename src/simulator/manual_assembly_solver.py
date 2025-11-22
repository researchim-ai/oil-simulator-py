import torch
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, bicgstab, LinearOperator, spilu
from .reservoir import Reservoir
from .fluid import Fluid
from .well import WellManager
from .sparse_utils import SparseMatrixBuilder

try:
    # Try relative import first (package mode)
    from ..solver.classical_amg import ClassicalAMG
    HAS_AMG = True
except (ImportError, ValueError):
    try:
        # Try absolute import (script mode with src in path)
        from solver.classical_amg import ClassicalAMG
        HAS_AMG = True
    except ImportError:
        print("Warning: ClassicalAMG not found, CPR disabled.")
        HAS_AMG = False

class CPRPreconditioner(LinearOperator):
    """
    Constrained Pressure Residual (CPR) Preconditioner.
    Использует AMG для давления и ILU для полной системы.
    """
    def __init__(self, J_bsr, n_cells, device='cuda'):
        self.shape = J_bsr.shape
        self.dtype = J_bsr.dtype
        self.n_cells = n_cells
        self.device = device
        self.J_bsr = J_bsr
        
        # 1. Extract Pressure Matrix A_p
        indptr = J_bsr.indptr
        indices = J_bsr.indices
        data_blocks = J_bsr.data # (NNZ, 2, 2)
        
        # A_p_data = J_00 + J_10 (Row sum of derivatives wrt P)
        data_p = data_blocks[:, 0, 0] + data_blocks[:, 1, 0]
        
        # Scale data_p to reasonable range (e.g. max abs val = 1.0) to help AMG thresholds
        max_val = np.max(np.abs(data_p))
        if max_val > 0:
            data_p = data_p / max_val
            self.p_scale = max_val
        else:
            self.p_scale = 1.0
        
        self.A_p = sp.csr_matrix((data_p, indices, indptr), shape=(n_cells, n_cells))
        
        # Init AMG
        try:
            # ClassicalAMG auto-detects device (cuda if available)
            # Use theta=0.25 (standard). 
            # mixed_precision=True to save memory.
            # coarsening_method='pmis' (default) is usually faster/more aggressive than RS.
            # nullspace_dim=0 disables expensive energy-minimization (saves time/memory)
            self.amg = ClassicalAMG(
                self.A_p, 
                max_levels=10, 
                theta=0.25,
                mixed_precision=True,
                nullspace_dim=0 
            )
            # Sync device with AMG choice
            self.amg_device = self.amg.device
        except Exception as e:
            print(f"CPR AMG init failed: {e}, falling back to simple ILU")
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
            self.amg = None

        # 2. ILU for Smoothing
        csc = J_bsr.tocsc()
        try:
            self.ilu = spilu(csc, drop_tol=1e-4, fill_factor=2.0)
        except Exception as e:
             print(f"CPR ILU init failed: {e}")
             self.ilu = None

    def _matvec(self, r):
        # 1. AMG Step (Pressure Correction)
        x_amg_full = np.zeros_like(r)
        
        if self.amg:
            # RHS for pressure: sum of residuals
            r_p = r[0::2] + r[1::2]
            
            # Scale RHS to match A_p scaling
            r_p_scaled = r_p / self.p_scale
            
            # Ensure we use the device where AMG resides
            b_torch = torch.from_numpy(r_p_scaled).to(self.amg.device)
            
            with torch.no_grad():
                x_p_torch = self.amg.solve(b_torch, tol=1e-2, max_iter=5)
            
            x_p = x_p_torch.cpu().numpy()
            x_amg_full[0::2] = x_p
            
        # 2. Residual update
        # r' = r - A * x_amg
        # Note: J_bsr matvec is fast
        if self.amg:
            r_prime = r - self.J_bsr @ x_amg_full
        else:
            r_prime = r
            
        # 3. ILU Smoothing
        if self.ilu:
            x_ilu = self.ilu.solve(r_prime)
        else:
            x_ilu = r_prime # No smoothing fallback (bad)
            
        return x_amg_full + x_ilu

class ManualFIMSolver:
    """
    Fully Implicit Solver с ручной сборкой разреженного Якобиана (CSR/BSR).
    Оптимизирован по памяти: подходит для сеток 1M+ ячеек.
    """
    def __init__(self, reservoir: Reservoir, fluid: Fluid, well_manager: WellManager, params: dict):
        self.reservoir = reservoir
        self.fluid = fluid
        self.well_manager = well_manager
        self.device = reservoir.device
        
        # Геометрия
        self.nx, self.ny, self.nz = reservoir.dimensions
        self.n_cells = self.nx * self.ny * self.nz
        
        # Параметры решателя
        self.max_newton = params.get('newton_max_iter', 15)
        # Tolerance for mass balance in kg/s
        # Default to 1e-6 kg/s (~0.1 kg/day) if not specified
        self.tol_mass = params.get('newton_tol_mass', params.get('newton_tolerance', 1e-6))
        
        # Построитель структуры матрицы (делаем 1 раз)
        print("Building sparse matrix structure...")
        self.builder = SparseMatrixBuilder(self.nx, self.ny, self.nz, block_size=2, device=self.device)
        print(f"Structure built. NNZ blocks: {self.builder.n_nonzero}")
        
        # Предварительный расчет геометрической части трансмиссибильности
        self._compute_geom_transmissibility()
        
        # Поровый объем
        dx, dy, dz = reservoir.grid_size
        self.dx, self.dy, self.dz = dx, dy, dz
        if hasattr(reservoir, 'cell_volume'):
            vol = reservoir.cell_volume
        else:
            vol = dx * dy * dz
        self.pore_volume = vol * reservoir.porosity
        self.pore_volume = self.pore_volume.view(-1).double()

    def _compute_geom_transmissibility(self):
        kx = self.reservoir.permeability_x
        # Note: Permeability is already in SI (m^2) from Reservoir
        
        dx, dy, dz = self.reservoir.grid_size
        
        # Geometric factors (Area / Length) [m]
        Gx = 2 * (dy * dz) / dx
        Gy = 2 * (dx * dz) / dy
        Gz = 2 * (dx * dy) / dz
        
        # Transmissibility of cells [m^3]
        Tx_cell = Gx * kx
        Ty_cell = Gy * self.reservoir.permeability_y
        Tz_cell = Gz * self.reservoir.permeability_z
        
        # Harmonic mean for connections (nx-1, ny, nz)
        self.Tx = 2 * Tx_cell[:-1,:,:] * Tx_cell[1:,:,:] / (Tx_cell[:-1,:,:] + Tx_cell[1:,:,:] + 1e-32)
        self.Ty = 2 * Ty_cell[:,:-1,:] * Ty_cell[:,1:,:] / (Ty_cell[:,:-1,:] + Ty_cell[:,1:,:] + 1e-32)
        self.Tz = 2 * Tz_cell[:,:,:-1] * Tz_cell[:,:,1:] / (Tz_cell[:,:,:-1] + Tz_cell[:,:,1:] + 1e-32)
        
        self.Tx_flat = self.Tx.view(-1).double()

    def step(self, dt):
        """Выполняет шаг по времени"""
        p_old = self.fluid.pressure.clone().double()
        sw_old = self.fluid.s_w.clone().double()
        
        p_curr = p_old.clone()
        sw_curr = sw_old.clone()
        
        mass_o_old, mass_w_old = self.calc_masses(p_old, sw_old)
        
        for iter_k in range(self.max_newton):
            # Сборка
            R, J_values = self.assemble_system(p_curr, sw_curr, mass_o_old, mass_w_old, dt)
            
            # Проверка сходимости
            max_res = torch.max(torch.abs(R))
            print(f"  Iter {iter_k}: Max Res = {max_res:.2e}")
            
            if max_res < self.tol_mass:
                print("  Converged!")
                self.fluid.pressure.copy_(p_curr.float())
                self.fluid.s_w.copy_(sw_curr.float())
                self.fluid.s_o.copy_(1.0 - sw_curr.float())
                return True
            
            # Решение линейной системы
            dx = self.solve_cpu(J_values, R)
            
            if dx is None:
                return False
            
            # Обновление
            dp = dx[0::2].reshape(self.nx, self.ny, self.nz)
            dsw = dx[1::2].reshape(self.nx, self.ny, self.nz)
            
            # Damping
            dsw = torch.clamp(dsw, -0.2, 0.2)
            dp = torch.clamp(dp, -50e5, 50e5)
            
            p_curr = p_curr + dp.to(self.device)
            sw_curr = sw_curr + dsw.to(self.device)
            sw_curr = torch.clamp(sw_curr, 0.0, 1.0)

        return False

    def calc_masses(self, p, sw):
        # Use Fluid properties for consistency
        rho_w_ref = self.fluid.rho_water_ref
        rho_o_ref = self.fluid.rho_oil_ref
        c_w = self.fluid.water_compressibility
        c_o = self.fluid.oil_compressibility
        
        rho_w = rho_w_ref * (1.0 + c_w * (p - 1e5))
        rho_o = rho_o_ref * (1.0 + c_o * (p - 1e5))
        
        # Use pre-calculated pore volume (flat) reshaped to 3D
        pv = self.pore_volume.view(self.nx, self.ny, self.nz)
        
        mass_w = pv * sw * rho_w
        mass_o = pv * (1-sw) * rho_o
        return mass_o, mass_w

    def assemble_system(self, p, sw, m_o_old, m_w_old, dt):
        """
        Сборка системы вручную (векторизованно).
        """
        R = torch.zeros(self.n_cells * 2, device=self.device, dtype=torch.float64)
        J_vals = torch.zeros((self.builder.n_nonzero, 2, 2), device=self.device, dtype=torch.float64)
        
        # Extract Fluid Properties
        rho_w_ref = self.fluid.rho_water_ref
        rho_o_ref = self.fluid.rho_oil_ref
        c_w = self.fluid.water_compressibility
        c_o = self.fluid.oil_compressibility
        mu_w = self.fluid.mu_water
        mu_o = self.fluid.mu_oil
        
        # --- 1. Accumulation Term ---
        m_o, m_w = self.calc_masses(p, sw)
        acc_o = (m_o - m_o_old) / dt
        acc_w = (m_w - m_w_old) / dt
        
        R[0::2] += acc_o.view(-1)
        R[1::2] += acc_w.view(-1)
        
        # Jacobian (Diagonal)
        p_flat = p.view(-1)
        sw_flat = sw.view(-1)
        
        rho_w = rho_w_ref * (1.0 + c_w * (p_flat - 1e5))
        rho_o = rho_o_ref * (1.0 + c_o * (p_flat - 1e5))
        drho_w_dp = rho_w_ref * c_w
        drho_o_dp = rho_o_ref * c_o
        
        dMw_dSw = self.pore_volume * rho_w / dt
        dMw_dP  = self.pore_volume * sw_flat * drho_w_dp / dt
        dMo_dSw = self.pore_volume * (-rho_o) / dt
        dMo_dP  = self.pore_volume * (1-sw_flat) * drho_o_dp / dt
        
        diag_idx = self.builder.diag_map
        J_vals[diag_idx, 0, 0] += dMo_dP
        J_vals[diag_idx, 0, 1] += dMo_dSw
        J_vals[diag_idx, 1, 0] += dMw_dP
        J_vals[diag_idx, 1, 1] += dMw_dSw
        
        # --- Helper Function for Flux Logic ---
        def process_direction(p_slice_L, p_slice_R, sw_slice_L, sw_slice_R, T_slice, 
                            map_x_pos_L, map_x_neg_R, idx_L, idx_R):
            dpot = p_slice_L - p_slice_R
            up = (dpot >= 0).double()
            down = 1.0 - up
            
            # Closure to use captured fluid props
            def get_props_correct(pres, sat):
                rho_w_ = rho_w_ref * (1.0 + c_w * (pres - 1e5))
                rho_o_ = rho_o_ref * (1.0 + c_o * (pres - 1e5))
                
                # Relative Permeability (Simplified Corey)
                sw_irr = 0.2; sor = 0.2
                swn = torch.clamp((sat - sw_irr)/(1-sw_irr-sor), 0.0, 1.0)
                dswn_ds = 1.0 / (1-sw_irr-sor)
                # Zero derivative outside range handled by clamp gradient? 
                # Manual gradient:
                # If sat < sw_irr, swn=0. If sat > 1-sor, swn=1.
                # Derivative is 0 in those regions.
                # We can implement this explicitly or rely on mask.
                
                krw_ = swn ** 2
                kro_ = (1-swn) ** 2
                
                # Derivatives
                dkrw_ds = 2 * swn * dswn_ds
                dkro_ds = -2 * (1-swn) * dswn_ds
                
                # Mask derivatives where swn is clamped (0 or 1)
                mask_active = (sat > sw_irr) & (sat < (1-sor))
                dkrw_ds = dkrw_ds * mask_active.double()
                dkro_ds = dkro_ds * mask_active.double()

                lam_w_ = krw_ * rho_w_ / mu_w
                lam_o_ = kro_ * rho_o_ / mu_o
                
                dlam_w_ds_ = dkrw_ds * rho_w_ / mu_w
                dlam_w_dp_ = krw_ * drho_w_dp / mu_w
                
                dlam_o_ds_ = dkro_ds * rho_o_ / mu_o
                dlam_o_dp_ = kro_ * drho_o_dp / mu_o
                
                return lam_o_, lam_w_, dlam_o_ds_, dlam_o_dp_, dlam_w_ds_, dlam_w_dp_

            lam_o_L, lam_w_L, dlo_ds_L, dlo_dp_L, dlw_ds_L, dlw_dp_L = get_props_correct(p_slice_L, sw_slice_L)
            lam_o_R, lam_w_R, dlo_ds_R, dlo_dp_R, dlw_ds_R, dlw_dp_R = get_props_correct(p_slice_R, sw_slice_R)

            lam_o_face = lam_o_L * up + lam_o_R * down
            lam_w_face = lam_w_L * up + lam_w_R * down
            
            flux_o = T_slice * lam_o_face * dpot
            flux_w = T_slice * lam_w_face * dpot
            
            R[2*idx_L] += flux_o.view(-1)
            R[2*idx_L+1] += flux_w.view(-1)
            R[2*idx_R] -= flux_o.view(-1)
            R[2*idx_R+1] -= flux_w.view(-1)
            
            # Derivatives
            # 1. dFlux/dX_L
            df_o_dp_L = T_slice * (dlo_dp_L * up * dpot + lam_o_face)
            df_w_dp_L = T_slice * (dlw_dp_L * up * dpot + lam_w_face)
            df_o_ds_L = T_slice * (dlo_ds_L * up * dpot)
            df_w_ds_L = T_slice * (dlw_ds_L * up * dpot)
            
            # To Diagonal L
            map_diag_L = self.builder.diag_map[idx_L]
            J_vals[map_diag_L, 0, 0] += df_o_dp_L.view(-1)
            J_vals[map_diag_L, 0, 1] += df_o_ds_L.view(-1)
            J_vals[map_diag_L, 1, 0] += df_w_dp_L.view(-1)
            J_vals[map_diag_L, 1, 1] += df_w_ds_L.view(-1)
            
            # To Off-Diagonal R (link R->L i.e. x_neg_map[R])
            J_vals[map_diag_L, 1, 0] += df_w_dp_L.view(-1)
            J_vals[map_diag_L, 1, 1] += df_w_ds_L.view(-1)
            
            # To Off-Diagonal R (link R->L i.e. x_neg_map[R])
            J_vals[map_x_neg_R, 0, 0] -= df_o_dp_L.view(-1)
            J_vals[map_x_neg_R, 0, 1] -= df_o_ds_L.view(-1)
            J_vals[map_x_neg_R, 1, 0] -= df_w_dp_L.view(-1)
            J_vals[map_x_neg_R, 1, 1] -= df_w_ds_L.view(-1)
            
            # 2. dFlux/dX_R
            df_o_dp_R = T_slice * (dlo_dp_R * down * dpot - lam_o_face)
            df_w_dp_R = T_slice * (dlw_dp_R * down * dpot - lam_w_face)
            df_o_ds_R = T_slice * (dlo_ds_R * down * dpot)
            df_w_ds_R = T_slice * (dlw_ds_R * down * dpot)
            
            # To Diagonal R
            map_diag_R = self.builder.diag_map[idx_R]
            J_vals[map_diag_R, 0, 0] -= df_o_dp_R.view(-1)
            J_vals[map_diag_R, 0, 1] -= df_o_ds_R.view(-1)
            J_vals[map_diag_R, 1, 0] -= df_w_dp_R.view(-1)
            J_vals[map_diag_R, 1, 1] -= df_w_ds_R.view(-1)
            
            # To Off-Diagonal L (link L->R i.e. x_pos_map[L])
            J_vals[map_x_pos_L, 0, 0] += df_o_dp_R.view(-1)
            J_vals[map_x_pos_L, 0, 1] += df_o_ds_R.view(-1)
            J_vals[map_x_pos_L, 1, 0] += df_w_dp_R.view(-1)
            J_vals[map_x_pos_L, 1, 1] += df_w_ds_R.view(-1)

        # Grid Indices
        grid_idx = torch.arange(self.n_cells, device=self.device).reshape(self.nx, self.ny, self.nz)
        
        # --- X Direction ---
        idx_L_x = grid_idx[:-1,:,:].reshape(-1)
        idx_R_x = grid_idx[1:,:,:].reshape(-1)
        process_direction(
            p[:-1,:,:], p[1:,:,:], sw[:-1,:,:], sw[1:,:,:], self.Tx,
            self.builder.x_pos_map[idx_L_x], self.builder.x_neg_map[idx_R_x],
            idx_L_x, idx_R_x
        )
        
        # --- Y Direction ---
        idx_L_y = grid_idx[:,:-1,:].reshape(-1)
        idx_R_y = grid_idx[:,1:,:].reshape(-1)
        process_direction(
            p[:,:-1,:], p[:,1:,:], sw[:,:-1,:], sw[:,1:,:], self.Ty,
            self.builder.y_pos_map[idx_L_y], self.builder.y_neg_map[idx_R_y],
            idx_L_y, idx_R_y
        )
        
        # --- Z Direction ---
        idx_L_z = grid_idx[:,:,:-1].reshape(-1)
        idx_R_z = grid_idx[:,:,1:].reshape(-1)
        process_direction(
            p[:,:,:-1], p[:,:,1:], sw[:,:,:-1], sw[:,:,1:], self.Tz,
            self.builder.z_pos_map[idx_L_z], self.builder.z_neg_map[idx_R_z],
            idx_L_z, idx_R_z
        )
        
        # --- Wells ---
        self.apply_wells(R, J_vals, p, sw)
        
        return R, J_vals

    def apply_wells(self, R, J_vals, p, sw):
        """
        Применяет скважины (источники/стоки) и добавляет их производные в Якобиан.
        """
        if not self.well_manager.wells:
            return

        # Extract Fluid Properties
        rho_w_ref = self.fluid.rho_water_ref
        rho_o_ref = self.fluid.rho_oil_ref
        c_w = self.fluid.water_compressibility
        c_o = self.fluid.oil_compressibility
        mu_w = self.fluid.mu_water
        mu_o = self.fluid.mu_oil
        
        drho_w_dp = rho_w_ref * c_w
        drho_o_dp = rho_o_ref * c_o
        
        # Получаем скалярные значения или тензоры
        p_flat = p.view(-1)
        sw_flat = sw.view(-1)

        for well in self.well_manager.wells:
            idx = well.cell_index_flat
            map_diag = self.builder.diag_map[idx]
            
            # Rate in m3/s
            q_surf_sec = well.control_value / 86400.0
            
            if well.type == 'injector':
                # Инжекция (всегда вода для простоты теста)
                if well.injected_phase == 'gas':
                    continue
                    
                mass_rate = q_surf_sec * rho_w_ref
                R[2*idx + 1] -= mass_rate
                # Производная по давлению/насыщенности для rate control injector = 0
                
            elif well.type == 'producer':
                # Добыча (Total Liquid Rate)
                pres = p_flat[idx]
                sat = sw_flat[idx]
                
                # --- PVT & Relperm Logic (Same as assemble) ---
                rho_w = rho_w_ref * (1.0 + c_w * (pres - 1e5))
                rho_o = rho_o_ref * (1.0 + c_o * (pres - 1e5))
                
                sw_irr=0.2; sor=0.2
                swn = torch.clamp((sat - sw_irr)/(1-sw_irr-sor), 0.0, 1.0)
                dswn_ds = 1.0 / (1-sw_irr-sor)
                
                mask_active = (sat > sw_irr) & (sat < (1-sor))
                if not mask_active: dswn_ds = 0.0 # Scalar check
                
                krw = swn ** 2
                dkrw_ds = 2 * swn * dswn_ds
                
                kro = (1-swn) ** 2
                dkro_ds = -2 * (1-swn) * dswn_ds
                
                lam_w = krw * rho_w / mu_w
                lam_o = kro * rho_o / mu_o
                
                dlam_w_ds = dkrw_ds * rho_w / mu_w
                dlam_w_dp = krw * drho_w_dp / mu_w
                
                dlam_o_ds = dkro_ds * rho_o / mu_o
                dlam_o_dp = kro * drho_o_dp / mu_o
                
                lam_tot = lam_w + lam_o + 1e-24
                dlam_tot_ds = dlam_w_ds + dlam_o_ds
                dlam_tot_dp = dlam_w_dp + dlam_o_dp
                
                fw = lam_w / lam_tot
                fo = lam_o / lam_tot
                
                # Derivatives of fractional flow
                dfw_ds = (dlam_w_ds * lam_o - lam_w * dlam_o_ds) / (lam_tot**2)
                dfw_dp = (dlam_w_dp * lam_o - lam_w * dlam_o_dp) / (lam_tot**2)
                
                dfo_ds = -dfw_ds
                dfo_dp = -dfw_dp
                
                # Rates
                q_w_mass = q_surf_sec * fw * rho_w_ref
                q_o_mass = q_surf_sec * fo * rho_o_ref
                
                # Add to Residual
                R[2*idx + 0] += q_o_mass
                R[2*idx + 1] += q_w_mass
                
                # Add to Jacobian (Diagonal)
                dq_w_dsw = q_surf_sec * rho_w_ref * dfw_ds
                dq_w_dp  = q_surf_sec * rho_w_ref * dfw_dp
                dq_o_dsw = q_surf_sec * rho_o_ref * dfo_ds
                dq_o_dp  = q_surf_sec * rho_o_ref * dfo_dp
                
                J_vals[map_diag, 0, 0] += dq_o_dp
                J_vals[map_diag, 0, 1] += dq_o_dsw
                J_vals[map_diag, 1, 0] += dq_w_dp
                J_vals[map_diag, 1, 1] += dq_w_dsw

    def solve_cpu(self, J_vals, R):
        """
        Решает систему на CPU через SciPy.
        Для больших сеток использует итеративный решатель BiCGStab c CPR-AMG предобуславливателем.
        """
        indptr = self.builder.indptr.cpu().numpy()
        indices = self.builder.indices.cpu().numpy()
        n_blocks = self.n_cells
        
        vals_np = J_vals.cpu().numpy() # [NNZ, 2, 2]
        bsr = sp.bsr_matrix((vals_np, indices, indptr), shape=(2*n_blocks, 2*n_blocks))
        
        rhs = -R.cpu().numpy()
        
        try:
            if self.n_cells < 2000: # Small grid -> Direct Solver
                x = spsolve(bsr, rhs)
            else:
                # Iterative Solver Strategy
                
                # 1. Try CPR Preconditioner (AMG for Pressure + ILU)
                M = None
                if HAS_AMG: # Enable CPR if available
                    # print("    Building CPR Preconditioner...")
                    try:
                        cpr = CPRPreconditioner(bsr, self.n_cells, device=self.device)
                        M = cpr
                    except Exception as e:
                        print(f"    CPR init failed ({e}), falling back to ILU")
                
                # 2. Fallback to Standard ILU
                if M is None:
                    csc = bsr.tocsc()
                    ilu = spilu(csc, drop_tol=1e-4, fill_factor=2.0)
                    M_x = lambda x: ilu.solve(x)
                    M = LinearOperator((2*n_blocks, 2*n_blocks), M_x)
                
                # print(f"    Solving linear system (BiCGStab)...")
                
                # Args for bicgstab
                kwargs = {'maxiter': 50, 'M': M}
                try:
                    scipy_ver = int(scipy.__version__.split('.')[1])
                    if scipy_ver >= 12: 
                         kwargs['rtol'] = 1e-5
                    else:
                         kwargs['tol'] = 1e-5
                except:
                     kwargs['tol'] = 1e-5
                
                x, info = bicgstab(bsr, rhs, **kwargs)
                
                if info > 0:
                    print(f"    Linear solver did not converge in {info} iterations")
                elif info < 0:
                    print("    Linear solver failed")
                    return None
                    
            return torch.from_numpy(x)
        except TypeError as te:
            print(f"Solver type error: {te}")
            return None
        except Exception as e:
            print(f"Solver failed: {e}")
            return None
