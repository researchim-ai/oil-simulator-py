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
        
        # Предварительный расчет геометрической части трансмиссибильности и разницы глубин
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
        
        # Кэшируем параметры релперм для быстрого доступа
        self.sw_irr = getattr(self.fluid, 'sw_cr', 0.2)
        self.so_r = getattr(self.fluid, 'so_r', 0.2)
        self.nw = getattr(self.fluid, 'nw', 2.0)
        self.no = getattr(self.fluid, 'no', 2.0)

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
        
        # Разница глубин (Z centers)
        # Z обычно растет вниз. Поток L->R. dZ = Z_L - Z_R.
        if hasattr(self.reservoir, 'z_centers'):
             # [nx], [ny], [nz] -> broadcast to [nx, ny, nz]
             xc = self.reservoir.x_centers.view(self.nx, 1, 1).expand(self.nx, self.ny, self.nz)
             yc = self.reservoir.y_centers.view(1, self.ny, 1).expand(self.nx, self.ny, self.nz)
             zc = self.reservoir.z_centers.view(1, 1, self.nz).expand(self.nx, self.ny, self.nz)
             
             self.dZ_x = zc[:-1,:,:] - zc[1:,:,:]
             self.dZ_y = zc[:,:-1,:] - zc[:,1:,:]
             self.dZ_z = zc[:,:,:-1] - zc[:,:,1:]
        else:
             self.dZ_x = torch.zeros_like(self.Tx)
             self.dZ_y = torch.zeros_like(self.Ty)
             self.dZ_z = torch.zeros_like(self.Tz)

    def step(self, dt, current_time=None):
        """Выполняет шаг по времени
        dt: шаг времени в СЕКУНДАХ
        current_time: текущее абсолютное время симуляции в СЕКУНДАХ (для ramping)
        """
        dt_sec = dt
        
        p_old = self.fluid.pressure.clone().double()
        sw_old = self.fluid.s_w.clone().double()
        
        p_curr = p_old.clone()
        sw_curr = sw_old.clone()
        
        mass_o_old, mass_w_old = self.calc_masses(p_old, sw_old)
        
        initial_res_norm = None
        
        for iter_k in range(self.max_newton):
            # Сборка (передаем dt и current_time)
            R, J_values = self.assemble_system(p_curr, sw_curr, mass_o_old, mass_w_old, dt_sec, current_time=current_time)
            
            # Проверка сходимости
            # Нормализуем невязку на dt для корректного сравнения? 
            # Обычно tolerance задается для mass rate error [kg/s].
            # R в [kg/s].
            max_res = torch.max(torch.abs(R))
            print(f"  Iter {iter_k}: Max Res = {max_res:.2e} kg/s")
            
            if iter_k == 0:
                initial_res_norm = max_res
            
            if max_res < self.tol_mass:
                print("  Converged!")
                self.fluid.pressure.copy_(p_curr.float())
                self.fluid.s_w.copy_(sw_curr.float())
                self.fluid.s_o.copy_(1.0 - sw_curr.float())
                return True
            
            # Адаптивный допуск для линейного решателя
            rel_newton_err = 1.0
            if initial_res_norm is not None and initial_res_norm > 1e-10:
                rel_newton_err = (max_res / initial_res_norm).item()
            
            # Решение линейной системы
            dx = self.solve_cpu(J_values, R, current_newton_rel_error=rel_newton_err)
            
            if dx is None:
                return False
            
            # Обновление с Line Search и Safe Scaling
            dp_full = dx[0::2].reshape(self.nx, self.ny, self.nz).to(self.device)
            dsw_full = dx[1::2].reshape(self.nx, self.ny, self.nz).to(self.device)
            
            # Safe Scaling (вместо жесткого clamp, который искажает направление)
            # Ограничиваем максимальное изменение за шаг:
            # Saturation: 0.2
            # Pressure: 20 bar (20e5 Pa)
            max_dsw = torch.max(torch.abs(dsw_full))
            max_dp = torch.max(torch.abs(dp_full))
            
            scale_factor = 1.0
            if max_dsw > 0.2:
                scale_factor = min(scale_factor, 0.2 / max_dsw.item())
            if max_dp > 50e5: # 50 bar limit
                scale_factor = min(scale_factor, 50e5 / max_dp.item())
            
            if scale_factor < 1.0:
                # print(f"    Scaling Update by {scale_factor:.4f} (Max dSw={max_dsw:.2f}, Max dP={max_dp/1e5:.1f} bar)")
                dp_full *= scale_factor
                dsw_full *= scale_factor
            
            # Backtracking Line Search
            alpha = 1.0
            min_alpha = 0.125
            accepted = False
            
            # Текущая норма невязки (для сравнения)
            current_res_norm = max_res
            
            while alpha >= min_alpha:
                p_trial = p_curr + alpha * dp_full
                sw_trial = sw_curr + alpha * dsw_full
                sw_trial = torch.clamp(sw_trial, 0.0, 1.0)
                
                # Проверяем невязку в новой точке
                # Важно: используем те же массы "old", но новые p/sw
                R_trial, _ = self.assemble_system(p_trial, sw_trial, mass_o_old, mass_w_old, dt_sec, current_time=current_time)
                res_trial = torch.max(torch.abs(R_trial))
                
                # Условие принятия (Armijo-like, simplified): невязка уменьшилась
                # Или хотя бы не выросла катастрофически (allow small increase initially?)
                # Strict decrease is safest for FIM.
                if res_trial < current_res_norm:
                    p_curr = p_trial
                    sw_curr = sw_trial
                    accepted = True
                    if alpha < 1.0:
                        print(f"    Line search: reduced step alpha={alpha}, Res {current_res_norm:.2e} -> {res_trial:.2e}")
                    break
                else:
                    # print(f"    Line search reject alpha={alpha} (Res {res_trial:.2e} >= {current_res_norm:.2e})")
                    alpha *= 0.5
            
            if not accepted:
                print(f"    Line search failed to reduce residual. Force taking small step alpha={min_alpha}")
                p_curr = p_curr + min_alpha * dp_full
                sw_curr = sw_curr + min_alpha * dsw_full
                sw_curr = torch.clamp(sw_curr, 0.0, 1.0)
                # Это скорее всего приведет к расходимости на след итерации и уменьшению dt, что и нужно.

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

    def assemble_system(self, p, sw, m_o_old, m_w_old, dt, current_time=None):
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
        def process_direction(p_slice_L, p_slice_R, sw_slice_L, sw_slice_R, T_slice, dZ_slice,
                            map_x_pos_L, map_x_neg_R, idx_L, idx_R):
            
            # Gravity Constant
            g_const = 9.80665
            
            # Calculate Densities for Gravity Term
            # rho = rho_ref * (1 + c * (P - 1e5))
            rho_w_L = rho_w_ref * (1.0 + c_w * (p_slice_L - 1e5))
            rho_o_L = rho_o_ref * (1.0 + c_o * (p_slice_L - 1e5))
            rho_w_R = rho_w_ref * (1.0 + c_w * (p_slice_R - 1e5))
            rho_o_R = rho_o_ref * (1.0 + c_o * (p_slice_R - 1e5))
            
            # Average density for interface
            rho_w_avg = 0.5 * (rho_w_L + rho_w_R)
            rho_o_avg = 0.5 * (rho_o_L + rho_o_R)
            
            # Potential Difference including Gravity: dPhi = dP - rho_avg * g * dZ
            # dZ_slice = Z_L - Z_R
            grav_term_w = rho_w_avg * g_const * dZ_slice
            grav_term_o = rho_o_avg * g_const * dZ_slice
            
            dpot_w = (p_slice_L - p_slice_R) - grav_term_w
            dpot_o = (p_slice_L - p_slice_R) - grav_term_o
            
            # Upstream weighting based on POTENTIAL difference (Counter-current flow possible)
            up_w = (dpot_w >= 0).double()
            down_w = 1.0 - up_w
            
            up_o = (dpot_o >= 0).double()
            down_o = 1.0 - up_o
            
            # Closure to use captured fluid props
            def get_props_correct(pres, sat):
                rho_w_ = rho_w_ref * (1.0 + c_w * (pres - 1e5))
                rho_o_ = rho_o_ref * (1.0 + c_o * (pres - 1e5))
                
                # Relative Permeability
                sw_irr = self.sw_irr
                sor = self.so_r
                nw = self.nw
                no = self.no
                
                # Normalized saturation
                denom = 1.0 - sw_irr - sor
                swn = torch.clamp((sat - sw_irr)/(denom + 1e-12), 0.0, 1.0)
                dswn_ds = 1.0 / (denom + 1e-12)
                
                krw_ = swn ** nw
                kro_ = (1-swn) ** no
                
                # Derivatives
                dkrw_ds = nw * (swn ** (nw - 1)) * dswn_ds
                dkro_ds = -no * ((1-swn) ** (no - 1)) * dswn_ds
                
                # Mask derivatives where swn is clamped (0 or 1)
                # Important for stability
                mask_active = (sat > sw_irr) & (sat < (1.0-sor))
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

            lam_o_face = lam_o_L * up_o + lam_o_R * down_o
            lam_w_face = lam_w_L * up_w + lam_w_R * down_w
            
            flux_o = T_slice * lam_o_face * dpot_o
            flux_w = T_slice * lam_w_face * dpot_w
            
            R[2*idx_L] += flux_o.view(-1)
            R[2*idx_L+1] += flux_w.view(-1)
            R[2*idx_R] -= flux_o.view(-1)
            R[2*idx_R+1] -= flux_w.view(-1)
            
            # Derivatives of Potential Difference
            # d(dpot)/dP_L = 1 - 0.5 * g * dZ * drho/dP_L
            # d(dpot)/dP_R = -1 - 0.5 * g * dZ * drho/dP_R
            
            gdZ = g_const * dZ_slice
            
            ddpot_w_dp_L = 1.0 - 0.5 * gdZ * drho_w_dp
            ddpot_w_dp_R = -1.0 - 0.5 * gdZ * drho_w_dp
            
            ddpot_o_dp_L = 1.0 - 0.5 * gdZ * drho_o_dp
            ddpot_o_dp_R = -1.0 - 0.5 * gdZ * drho_o_dp

            # 1. dFlux/dX_L
            # dF/dP_L = T * (dlam_face/dP_L * dpot + lam_face * ddpot/dP_L)
            # dlam_face/dP_L = dlam_L/dP_L * up
            
            df_o_dp_L = T_slice * (dlo_dp_L * up_o * dpot_o + lam_o_face * ddpot_o_dp_L)
            df_w_dp_L = T_slice * (dlw_dp_L * up_w * dpot_w + lam_w_face * ddpot_w_dp_L)
            
            df_o_ds_L = T_slice * (dlo_ds_L * up_o * dpot_o)
            df_w_ds_L = T_slice * (dlw_ds_L * up_w * dpot_w)
            
            # To Diagonal L
            map_diag_L = self.builder.diag_map[idx_L]
            J_vals[map_diag_L, 0, 0] += df_o_dp_L.view(-1)
            J_vals[map_diag_L, 0, 1] += df_o_ds_L.view(-1)
            J_vals[map_diag_L, 1, 0] += df_w_dp_L.view(-1)
            J_vals[map_diag_L, 1, 1] += df_w_ds_L.view(-1)
            
            # To Off-Diagonal R (link R->L i.e. x_neg_map[R])
            J_vals[map_x_neg_R, 0, 0] -= df_o_dp_L.view(-1)
            J_vals[map_x_neg_R, 0, 1] -= df_o_ds_L.view(-1)
            J_vals[map_x_neg_R, 1, 0] -= df_w_dp_L.view(-1)
            J_vals[map_x_neg_R, 1, 1] -= df_w_ds_L.view(-1)
            
            # 2. dFlux/dX_R
            # dF/dP_R = T * (dlam_face/dP_R * dpot + lam_face * ddpot/dP_R)
            # dlam_face/dP_R = dlam_R/dP_R * down
            
            df_o_dp_R = T_slice * (dlo_dp_R * down_o * dpot_o + lam_o_face * ddpot_o_dp_R)
            df_w_dp_R = T_slice * (dlw_dp_R * down_w * dpot_w + lam_w_face * ddpot_w_dp_R)
            
            df_o_ds_R = T_slice * (dlo_ds_R * down_o * dpot_o)
            df_w_ds_R = T_slice * (dlw_ds_R * down_w * dpot_w)
            
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
            p[:-1,:,:], p[1:,:,:], sw[:-1,:,:], sw[1:,:,:], self.Tx, self.dZ_x,
            self.builder.x_pos_map[idx_L_x], self.builder.x_neg_map[idx_R_x],
            idx_L_x, idx_R_x
        )
        
        # --- Y Direction ---
        idx_L_y = grid_idx[:,:-1,:].reshape(-1)
        idx_R_y = grid_idx[:,1:,:].reshape(-1)
        process_direction(
            p[:,:-1,:], p[:,1:,:], sw[:,:-1,:], sw[:,1:,:], self.Ty, self.dZ_y,
            self.builder.y_pos_map[idx_L_y], self.builder.y_neg_map[idx_R_y],
            idx_L_y, idx_R_y
        )
        
        # --- Z Direction ---
        idx_L_z = grid_idx[:,:,:-1].reshape(-1)
        idx_R_z = grid_idx[:,:,1:].reshape(-1)
        process_direction(
            p[:,:,:-1], p[:,:,1:], sw[:,:,:-1], sw[:,:,1:], self.Tz, self.dZ_z,
            self.builder.z_pos_map[idx_L_z], self.builder.z_neg_map[idx_R_z],
            idx_L_z, idx_R_z
        )
        
        # --- Wells ---
        self.apply_wells(R, J_vals, p, sw, current_time=current_time)
        
        return R, J_vals

    def apply_wells(self, R, J_vals, p, sw, current_time=None):
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
        
        sim_time_days = (current_time / 86400.0) if current_time is not None else float('inf')

        for well in self.well_manager.wells:
            idx = well.cell_index_flat
            map_diag = self.builder.diag_map[idx]
            
            ctrl_type, ctrl_val = well.get_current_control(sim_time_days)
            
            # Rate in m3/s
            # q_surf_sec = well.control_value / 86400.0 
            # Используем ctrl_val
            
            if well.type == 'injector':
                # Инжекция (всегда вода для простоты теста)
                if well.injected_phase == 'gas':
                    continue
                
                if ctrl_type == 'rate':
                    q_surf_sec = ctrl_val / 86400.0
                    mass_rate = q_surf_sec * rho_w_ref
                    R[2*idx + 1] -= mass_rate
                    # Производная по давлению/насыщенности для rate control injector = 0
                elif ctrl_type == 'bhp':
                    # P_bh control injector
                    p_bh = ctrl_val * 1e6
                    # Injector logic: q = PI * mob * (P - P_bh) ?
                    # Обычно для инжектора mob берется по насыщенности в ячейке или апстрим?
                    # Для простоты - Total Mobility ячейки
                    # q < 0 (injection). If P < P_bh, term is negative.
                    # R += q => R += PI * mob * (P - P_bh)
                    
                    pres = p_flat[idx]
                    sat = sw_flat[idx]
                    # ... Need calculate mobilities (copy paste logic or extract?)
                    # For now, ignore BHP injectors or implement simplified
                    pass 
                    
            elif well.type == 'producer':
                # Добыча (Total Liquid Rate)
                pres = p_flat[idx]
                sat = sw_flat[idx]
                
                # --- PVT & Relperm Logic (Same as assemble) ---
                rho_w = rho_w_ref * (1.0 + c_w * (pres - 1e5))
                rho_o = rho_o_ref * (1.0 + c_o * (pres - 1e5))
                
                sw_irr = self.sw_irr
                sor = self.so_r
                nw = self.nw
                no = self.no
                
                denom = 1.0 - sw_irr - sor
                swn = torch.clamp((sat - sw_irr)/(denom + 1e-12), 0.0, 1.0)
                dswn_ds = 1.0 / (denom + 1e-12)
                
                krw = swn ** nw
                dkrw_ds = nw * (swn ** (nw - 1)) * dswn_ds
                
                kro = (1-swn) ** no
                dkro_ds = -no * ((1-swn) ** (no - 1)) * dswn_ds
                
                mask_active = (sat > sw_irr) & (sat < (1-sor))
                dkrw_ds = dkrw_ds * mask_active.double()
                dkro_ds = dkro_ds * mask_active.double()
                
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
                if ctrl_type == 'rate':
                    q_surf_sec = ctrl_val / 86400.0
                    
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
                
                elif ctrl_type == 'bhp':
                    # P_bh control
                    # q_mass = PI * mob_phase * (P - P_bh) ?
                    # Usually q_vol = PI * mob_total * (P - P_bh)
                    # And q_phase_vol = q_vol * f_phase
                    # q_phase_mass = q_vol * f_phase * rho_phase
                    
                    p_bh = ctrl_val * 1e6
                    
                    # q_tot_vol = PI * lam_tot * (pres - p_bh)
                    # Derivatives needed w.r.t pres and sat
                    
                    # q_o_mass = PI * lam_o * (pres - p_bh) ? Or via total?
                    # Standard: q_phase = PI * lam_phase * (P - P_bh)
                    
                    q_o_mass = well.well_index * lam_o * (pres - p_bh)
                    q_w_mass = well.well_index * lam_w * (pres - p_bh)
                    
                    R[2*idx + 0] += q_o_mass
                    R[2*idx + 1] += q_w_mass
                    
                    # Derivs: d(lam * (P - Pbh))/dX = dlam/dX * (P-Pbh) + lam * dP/dX
                    
                    dqo_dp = well.well_index * (dlam_o_dp * (pres - p_bh) + lam_o)
                    dqo_ds = well.well_index * (dlam_o_ds * (pres - p_bh))
                    
                    dqw_dp = well.well_index * (dlam_w_dp * (pres - p_bh) + lam_w)
                    dqw_ds = well.well_index * (dlam_w_ds * (pres - p_bh))
                    
                    J_vals[map_diag, 0, 0] += dqo_dp
                    J_vals[map_diag, 0, 1] += dqo_ds
                    J_vals[map_diag, 1, 0] += dqw_dp
                    J_vals[map_diag, 1, 1] += dqw_ds

    def solve_cpu(self, J_vals, R, current_newton_rel_error=1.0):
        """
        Решает систему линейных уравнений.
        Использует FGMRES на GPU если доступен, иначе падает на CPU SciPy.
        """
        indptr = self.builder.indptr
        indices = self.builder.indices
        n_blocks = self.n_cells
        
        # Подготовка GPU структур
        # J_vals: [NNZ, 2, 2]
        # R: [2*N]
        
        # Пробуем использовать GPU FGMRES
        try:
            from solver.fgmres import fgmres
            from .cpr_gpu import CPRPreconditionerGPU # Import local helper
            
            # 1. Оператор A @ x
            # J_vals имеет размер [NNZ, 2, 2]. Нужно эффективно умножать.
            # Для BSR умножения на PyTorch нет встроенного, но мы можем 
            # либо использовать sparse_csr (если блоки развернуть), 
            # либо написать свое ядро.
            # Проще всего: развернуть BSR в CSR (NNZ*4) и использовать torch.sparse.
            
            # Разворачиваем блоки 2x2 в CSR
            # Это делается один раз за решение, или на лету?
            # Лучше один раз собрать CSR матрицу для текущего шага Ньютона
            
            # Данные для CSR (crow, col, val)
            # indptr: [N+1] -> crow: [2N+1]
            # indices: [NNZ] -> col: [4*NNZ]
            
            # Эффективная конвертация BSR 2x2 -> CSR на PyTorch
            # J_vals: [NNZ, 2, 2] -> [NNZ, 4] flat values
            # Но порядок в CSR другой.
            # BSR: row i -> cols [j1, j2...] -> blocks [2x2]
            # CSR: row 2i -> cols [2j1, 2j1+1, 2j2, 2j2+1] -> vals [00, 01, 00, 01]
            #      row 2i+1 -> cols [2j1, 2j1+1, 2j2, 2j2+1] -> vals [10, 11, 10, 11]
            
            # Для простоты сейчас (MVP) соберем CSR через Python/Numpy конверсию,
            # НО оставим данные на GPU.
            # TODO: Написать pure torch kernel для BSR matvec
            
            # --- Вариант 1: Torch Sparse CSR (beta) ---
            # Требует int64 индексов
            
            # Подготовка индексов (делаем 1 раз при инициализации builder, но сейчас тут)
            if not hasattr(self, 'csr_crow') or self.csr_crow is None:
                # Конвертируем структуру BSR -> CSR 2x2
                # Это нетривиально векторизовать без кастомного кернела.
                # План Б: Использовать matvec через "размазывание"
                pass
                
            # --- Вариант 2: Matvec через scatter/gather (медленно) ---
            
            # --- Вариант 3: Конвертация через CPU Scipy (текущее решение для надежности) ---
            # Но мы хотим GPU.
            
            # --- Вариант 4: Прямое умножение J_vals * x ---
            # x: [N, 2]
            # y = zeros
            # Для каждого блока k (строка i, столбец j):
            # y[i] += block[k] @ x[j]
            # Это можно сделать через index_add_
            
            # Реализация Matvec на PyTorch (GPU)
            rows = torch.repeat_interleave(torch.arange(self.n_cells, device=self.device), 
                                         (indptr[1:] - indptr[:-1]).long())
            cols = indices.long()
            
            # --- Row Scaling (Jacobi Preconditioning) ---
            # Нормализуем уравнения, чтобы диагональные элементы были ~1.
            # Это критически важно для FGMRES, так как диагональные элементы давления (~1e-6)
            # и насыщенности (~1e3) различаются на 9 порядков.
            
            # Вычисляем масштабные коэффициенты из диагональных блоков
            diag_vals = J_vals[self.builder.diag_map] # [N, 2, 2]
            d00 = diag_vals[:, 0, 0]
            d11 = diag_vals[:, 1, 1]
            
            # Избегаем деления на ноль
            scale_0 = 1.0 / (torch.abs(d00) + 1e-12)
            scale_1 = 1.0 / (torch.abs(d11) + 1e-12)
            
            # Масштабируем правую часть R
            # R: [2*N].
            R_view = R.view(-1, 2)
            R_scaled = torch.empty_like(R_view)
            R_scaled[:, 0] = R_view[:, 0] * scale_0
            R_scaled[:, 1] = R_view[:, 1] * scale_1
            R_scaled_flat = R_scaled.view(-1)
            
            # Масштабируем матрицу J для matvec (Left Scaling)
            # J_vals[k] умножается на scale[rows[k]]
            s0_expanded = scale_0[rows].unsqueeze(1) # [NNZ, 1]
            s1_expanded = scale_1[rows].unsqueeze(1) # [NNZ, 1]
            
            # Копия матрицы (in-place scale J_vals опасно, если он где-то переиспользуется, но здесь ок)
            # J_vals создается заново в assemble_system, так что можно править in-place для скорости
            J_vals[:, 0, :] *= s0_expanded
            J_vals[:, 1, :] *= s1_expanded
            
            def matvec_gpu_scaled(x_vec):
                # x_vec: [2*N] -> [N, 2]
                x_blk = x_vec.view(-1, 2)
                
                # Gather x values: x_j = x_blk[cols] -> [NNZ, 2]
                x_j = x_blk[cols]
                
                # Batch matmul: y_val = J_vals @ x_j.unsqueeze(-1) 
                # [NNZ, 2, 2] @ [NNZ, 2, 1] -> [NNZ, 2, 1]
                y_vals = torch.bmm(J_vals, x_j.unsqueeze(-1)).squeeze(-1) # [NNZ, 2]
                
                # Scatter add to rows
                y = torch.zeros_like(x_blk)
                y.index_add_(0, rows, y_vals)
                
                return y.view(-1)

            # 2. Предобуславливатель (CPR)
            if HAS_AMG:
                # Создаем CPR c уже масштабированной матрицей
                # Это делает A_p и inv_diag хорошо обусловленными (~1)
                cpr = CPRPreconditionerGPU(J_vals, indptr, indices, self.n_cells, self.device)
                M_op = cpr.matvec
            else:
                M_op = None
                
            # 3. Запуск FGMRES
            # Адаптивный толеранс (Eisenstat-Walker):
            # Строже (0.01), чтобы направление было точным
            adaptive_tol = max(1e-6, min(0.01, 0.01 * current_newton_rel_error))
            
            # print(f"    Solving with GPU FGMRES (tol={adaptive_tol:.1e})...")
            
            # Решаем масштабированную систему: (S*J) * x = (S*(-R))
            x, info = fgmres(matvec_gpu_scaled, -R_scaled_flat, M_mv=M_op, max_iter=20, restart=50, tol=adaptive_tol, verbose=True)
            
            if info != 0:
                # Не сошелся, но fgmres возвращает лучшее приближение
                # print("    FGMRES max iters reached")
                pass
                
            return x
            
        except ImportError:
            print("FGMRES not found, falling back to CPU")
        except Exception as e:
            print(f"GPU Solver failed ({e}), falling back to CPU")
            # Fallback code below...

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
