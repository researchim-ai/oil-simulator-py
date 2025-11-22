import torch
import torch.nn.functional as F
import numpy as np
from .fluid import Fluid
from .reservoir import Reservoir
from .well import WellManager

class FullyImplicitSolver:
    """
    Полностью неявный (Fully Implicit) решатель на основе PyTorch Autograd.
    
    Автоматически вычисляет Якобиан системы уравнений сохранения массы,
    используя граф вычислений PyTorch. Это гарантирует согласованность 
    физики и производных.
    """
    def __init__(self, reservoir: Reservoir, fluid: Fluid, well_manager: WellManager, params: dict):
        self.reservoir = reservoir
        self.fluid = fluid
        self.well_manager = well_manager
        self.device = reservoir.device
        
        # Параметры
        self.max_newton_iter = params.get('newton_max_iter', 15)
        self.tol_mass = params.get('newton_tol_mass', 1.0) # кг/день (абсолютная ошибка)
        self.tol_norm = params.get('newton_tol_norm', 1e-4) # Относительная норма невязки
        
        self.nx, self.ny, self.nz = reservoir.dimensions
        self.n_cells = self.nx * self.ny * self.nz
        
        # Поровый объем
        dx, dy, dz = reservoir.grid_size
        self.dx, self.dy, self.dz = dx, dy, dz
        
        if hasattr(reservoir, 'cell_volume'):
            vol = reservoir.cell_volume
        else:
            vol = dx * dy * dz
            
        self.pore_volume = vol * reservoir.porosity
        
        # --- Расчет трансмиссибильностей (Geometric Part) ---
        # T = (K * A) / L
        # Для гармонического среднего на грани i+1/2:
        # T_{i+1/2} = 2 * (T_i * T_{i+1}) / (T_i + T_{i+1})
        # где T_i = k_i * A / (dx/2) = 2 * k_i * dy * dz / dx
        
        # X transmissibility (между i и i+1)
        kx = reservoir.permeability_x # (nx, ny, nz) SI (m^2)
        # k_si = 1e-15 # Removed: input is already SI
        
        geom_x = 2 * (dy * dz) / dx
        Tx_cell = geom_x * kx
        
        # Гармоническое среднее между соседями
        t1 = Tx_cell[:-1, :, :]
        t2 = Tx_cell[1:, :, :]
        self.Tx = 2 * t1 * t2 / (t1 + t2 + 1e-20)
        
        # Y transmissibility
        geom_y = 2 * (dx * dz) / dy
        Ty_cell = geom_y * reservoir.permeability_y
        t1 = Ty_cell[:, :-1, :]
        t2 = Ty_cell[:, 1:, :]
        self.Ty = 2 * t1 * t2 / (t1 + t2 + 1e-20)
        
        # Z transmissibility
        geom_z = 2 * (dx * dy) / dz
        Tz_cell = geom_z * reservoir.permeability_z
        t1 = Tz_cell[:, :, :-1]
        t2 = Tz_cell[:, :, 1:]
        self.Tz = 2 * t1 * t2 / (t1 + t2 + 1e-20)


    def step(self, dt: float):
        """
        Шаг по времени с использованием Ньютона-Рафсона и Autograd.
        """
        # 1. Подготовка переменных для Autograd
        # Работаем в float64 для стабильности FIM (это стандарт)
        p_old = self.fluid.pressure.detach().clone().double()
        sw_old = self.fluid.s_w.detach().clone().double()
        
        # Приводим свойства пласта к double тоже
        self.Tx = self.Tx.double()
        self.Ty = self.Ty.double()
        self.Tz = self.Tz.double()
        self.pore_volume = self.pore_volume.double()
        
        # Начальное приближение (из предыдущего шага)
        p_curr = p_old.clone().requires_grad_(True)
        sw_curr = sw_old.clone().requires_grad_(True)
        
        # Расчет масс на предыдущем шаге (для Accumulation term)
        with torch.no_grad():
            mass_o_old, mass_w_old = self.calc_masses(p_old, sw_old)
        
        converged = False
        for iter_idx in range(self.max_newton_iter):
            # Обнуляем градиенты
            if p_curr.grad is not None: p_curr.grad.zero_()
            if sw_curr.grad is not None: sw_curr.grad.zero_()
            
            # --- 2. Прямой проход (Forward Pass): Расчет невязок ---
            res_o, res_w = self.compute_residuals_autograd(p_curr, sw_curr, mass_o_old, mass_w_old, dt)
            
            # Собираем полную невязку
            R = torch.cat([res_o.view(-1), res_w.view(-1)])
            
            # Проверка сходимости
            res_norm = torch.norm(R).item()
            max_res = R.abs().max().item()
            
            print(f"    Iter {iter_idx}: Max Res = {max_res:.2e}, L2 Norm = {res_norm:.2e}")
            
            if max_res < self.tol_mass:
                converged = True
                break
                
            # --- 3. Обратный проход (Backward Pass): Якобиан ---
            # Для прототипа на малых сетках используем плотный якобиан
            inputs = (p_curr, sw_curr)
            
            def residual_func(p, sw):
                ro, rw = self.compute_residuals_autograd(p, sw, mass_o_old, mass_w_old, dt)
                return torch.cat([ro.view(-1), rw.view(-1)])
            
            if self.n_cells > 5000:
                 print("    Grid too large for Autograd Dense Jacobian. Need Sparse implementation.")
                 return False
            
            J = torch.autograd.functional.jacobian(residual_func, inputs)
            # J[0]: dR/dP (2N, N), J[1]: dR/dSw (2N, N)
            J_mat = torch.cat([J[0].reshape(2*self.n_cells, -1), J[1].reshape(2*self.n_cells, -1)], dim=1)
            
            # --- 4. Решение линейной системы ---
            # Явно приводим типы для надежности
            J_mat = J_mat.double()
            R = R.double()
            
            try:
                # Используем dense solver (LU)
                dx = torch.linalg.solve(J_mat, -R)
            except RuntimeError as e:
                print(f"    Linear solver failed: {e}")
                return False
                
            # --- 5. Обновление переменных ---
            dp = dx[:self.n_cells].reshape(self.nx, self.ny, self.nz)
            dsw = dx[self.n_cells:].reshape(self.nx, self.ny, self.nz)
            
            # Демпфирование
            dsw = torch.clamp(dsw, -0.2, 0.2)
            
            with torch.no_grad():
                p_curr += dp
                sw_curr += dsw
                sw_curr.clamp_(0.0, 1.0)
                
            # Re-enable grad
            p_curr.requires_grad_(True)
            sw_curr.requires_grad_(True)

        if converged:
            with torch.no_grad():
                # Возвращаем в float32 для остальной части симулятора, если надо
                self.fluid.pressure.copy_(p_curr.float())
                self.fluid.s_w.copy_(sw_curr.float())
                self.fluid.s_o.copy_(1.0 - sw_curr.float() - self.fluid.s_g)
            return True
        else:
            return False

    def calc_masses(self, p, sw):
        """Считает массы компонентов в каждой ячейке (кг)"""
        phi = self.reservoir.porosity.double()
        vol = self.dx * self.dy * self.dz 
        
        # Use Fluid properties
        rho_o_ref = self.fluid.rho_oil_ref
        rho_w_ref = self.fluid.rho_water_ref
        c_o = self.fluid.oil_compressibility
        c_w = self.fluid.water_compressibility
        
        rho_o = rho_o_ref * (1.0 + c_o * (p - 1e5))
        rho_w = rho_w_ref * (1.0 + c_w * (p - 1e5))
        
        mass_o = vol * phi * (1 - sw) * rho_o
        mass_w = vol * phi * sw * rho_w
        
        return mass_o, mass_w

    def compute_residuals_autograd(self, p, sw, mass_o_old, mass_w_old, dt):
        """
        Полный расчет физики с поддержкой градиентов.
        Возвращает тензоры невязок (nx, ny, nz).
        """
        # 1. Accumulation term
        mass_o_new, mass_w_new = self.calc_masses(p, sw)
        acc_o = (mass_o_new - mass_o_old) / dt
        acc_w = (mass_w_new - mass_w_old) / dt
        
        # 2. Flux term
        # Need consistent properties with Manual Solver
        rho_o_ref = self.fluid.rho_oil_ref
        rho_w_ref = self.fluid.rho_water_ref
        c_o = self.fluid.oil_compressibility
        c_w = self.fluid.water_compressibility
        
        rho_o = rho_o_ref * (1.0 + c_o * (p - 1e5))
        rho_w = rho_w_ref * (1.0 + c_w * (p - 1e5))

        # --- X direction ---
        p_i = p[:-1, :, :]
        p_ip1 = p[1:, :, :]
        dpot_x = p_i - p_ip1
        up_x = (dpot_x > 0).double() # .double() for casting
        
        # Мобильности
        # Упрощенные Corey кривые для double
        sw_irr = 0.2
        so_r = 0.2
        sw_norm = (sw - sw_irr) / (1 - sw_irr - so_r)
        sw_norm = torch.clamp(sw_norm, 0.0, 1.0)
        krw = sw_norm ** 2.0
        kro = (1 - sw_norm) ** 2.0
        
        mu_o = self.fluid.mu_oil
        mu_w = self.fluid.mu_water
        
        # Mass Mobility: kr * rho / mu
        mob_o = kro * rho_o / mu_o
        mob_w = krw * rho_w / mu_w
        
        # Мобильности на гранях X
        mob_o_face_x = mob_o[:-1,:,:] * up_x + mob_o[1:,:,:] * (1 - up_x)
        mob_w_face_x = mob_w[:-1,:,:] * up_x + mob_w[1:,:,:] * (1 - up_x)
        
        flux_o_x = self.Tx * mob_o_face_x * dpot_x
        flux_w_x = self.Tx * mob_w_face_x * dpot_x
        
        div_o = torch.zeros_like(p)
        div_w = torch.zeros_like(p)
        
        div_o[:-1,:,:] += flux_o_x
        div_o[1:,:,:]  -= flux_o_x
        div_w[:-1,:,:] += flux_w_x
        div_w[1:,:,:]  -= flux_w_x
        
        # --- Y direction ---
        p_j = p[:, :-1, :]
        p_jp1 = p[:, 1:, :]
        dpot_y = p_j - p_jp1
        up_y = (dpot_y > 0).double()
        
        mob_o_face_y = mob_o[:,:-1,:] * up_y + mob_o[:,1:,:] * (1 - up_y)
        mob_w_face_y = mob_w[:,:-1,:] * up_y + mob_w[:,1:,:] * (1 - up_y)
        
        flux_o_y = self.Ty * mob_o_face_y * dpot_y
        flux_w_y = self.Ty * mob_w_face_y * dpot_y
        
        div_o[:,:-1,:] += flux_o_y
        div_o[:,1:,:]  -= flux_o_y
        div_w[:,:-1,:] += flux_w_y
        div_w[:,1:,:]  -= flux_w_y

        # --- Z direction ---
        p_k = p[:, :, :-1]
        p_kp1 = p[:, :, 1:]
        dpot_z = p_k - p_kp1
        up_z = (dpot_z > 0).double()
        
        mob_o_face_z = mob_o[:,:,:-1] * up_z + mob_o[:,:,1:] * (1 - up_z)
        mob_w_face_z = mob_w[:,:,:-1] * up_z + mob_w[:,:,1:] * (1 - up_z)
        
        flux_o_z = self.Tz * mob_o_face_z * dpot_z
        flux_w_z = self.Tz * mob_w_face_z * dpot_z
        
        div_o[:,:,:-1] += flux_o_z
        div_o[:,:,1:]  -= flux_o_z
        div_w[:,:,:-1] += flux_w_z
        div_w[:,:,1:]  -= flux_w_z
        
        # 3. Wells
        wells_o = torch.zeros_like(p)
        wells_w = torch.zeros_like(p)
        
        for well in self.well_manager.wells:
            ii, jj, kk = well.i, well.j, well.k
            if ii >= self.nx or jj >= self.ny or kk >= self.nz: continue
            
            if well.control_type == 'rate':
                q_surf = well.control_value / 86400.0
                if well.type == 'injector':
                     if well.injected_phase == 'water':
                         rho_sc = self.fluid.rho_water_ref
                         mass_rate = q_surf * rho_sc
                         wells_w[ii, jj, kk] -= mass_rate # Inject = minus in residual
                         
                elif well.type == 'producer':
                    # Use same mobilities as calculated above (cell-based)
                    mob_o_cell = mob_o[ii, jj, kk]
                    mob_w_cell = mob_w[ii, jj, kk]
                    mob_total = mob_o_cell + mob_w_cell + 1e-20
                    
                    frac_o = mob_o_cell / mob_total
                    frac_w = mob_w_cell / mob_total
                    
                    q_surf_o = q_surf * frac_o
                    q_surf_w = q_surf * frac_w
                    
                    rho_o_sc = self.fluid.rho_oil_ref
                    rho_w_sc = self.fluid.rho_water_ref
                    
                    wells_o[ii, jj, kk] += q_surf_o * rho_o_sc
                    wells_w[ii, jj, kk] += q_surf_w * rho_w_sc

        
        res_o = acc_o + div_o + wells_o
        res_w = acc_w + div_w + wells_w
        
        return res_o, res_w
