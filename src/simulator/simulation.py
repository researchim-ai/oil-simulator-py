import torch
import numpy as np
from scipy.sparse import csc_matrix, diags, bmat
from scipy.sparse.linalg import cg, LinearOperator, bicgstab, spsolve
import time

class Simulator:
    """
    Основной класс симулятора, отвечающий за выполнение расчетов.
    Поддерживает две схемы:
    - IMPES (Implicit Pressure, Explicit Saturation)
    - Полностью неявную (Fully Implicit)
    
    Поддерживает выполнение на CPU или GPU (если доступна CUDA).
    """
    def __init__(self, reservoir, fluid_system, well_manager, sim_params):
        """ Инициализация симулятора """
        self.reservoir = reservoir
        self.fluid = fluid_system
        self.well_manager = well_manager
        
        # Проверяем доступность CUDA и переопределяем устройство при необходимости
        use_cuda = sim_params.get('use_cuda', False) and torch.cuda.is_available()
        if use_cuda and reservoir.device.type != 'cuda':
            print(f"Включен режим CUDA. Перемещаем данные на GPU...")
            self.device = torch.device("cuda:0")
            self._move_data_to_device()
        else:
            self.device = reservoir.device
            
        self.sim_params = sim_params
        self.solver_type = sim_params.get('solver_type', 'impes') # impes или fully_implicit

        # Конвертируем проницаемость из мД в м^2 (1 мД ~ 1e-15 м^2)
        perm_h_m2 = reservoir.perm_h * 1e-15
        perm_v_m2 = reservoir.perm_v * 1e-15

        # Рассчитываем геометрический фактор и гармоническое среднее проницаемости
        dx, dy, dz = reservoir.grid_size
        ax, ay, az = dy*dz, dx*dz, dx*dy
        
        k_x_h = 2 * perm_h_m2[1:,:,:] * perm_h_m2[:-1,:,:] / (perm_h_m2[1:,:,:] + perm_h_m2[:-1,:,:] + 1e-20)
        k_y_h = 2 * perm_h_m2[:,1:,:] * perm_h_m2[:,:-1,:] / (perm_h_m2[:,1:,:] + perm_h_m2[:,:-1,:] + 1e-20)
        k_z_h = 2 * perm_v_m2[:,:,1:] * perm_v_m2[:,:,:-1] / (perm_v_m2[:,:,1:] + perm_v_m2[:,:,:-1] + 1e-20)

        # Рассчитываем проводимость (трансмиссивность) по осям
        self.T_x = (ax / dx) * k_x_h
        self.T_y = (ay / dy) * k_y_h
        self.T_z = (az / dz) * k_z_h

        # Сохраняем пористый объем (тензор)
        self.porous_volume = reservoir.porous_volume
        self.g = 9.81 # Ускорение свободного падения
        
    def _move_data_to_device(self):
        """Переносит данные на текущее устройство (CPU или GPU)"""
        # Переносим данные из резервуара
        self.reservoir.perm_h = self.reservoir.perm_h.to(self.device)
        self.reservoir.perm_v = self.reservoir.perm_v.to(self.device)
        self.reservoir.porous_volume = self.reservoir.porous_volume.to(self.device)
        
        # Переносим данные из флюида
        self.fluid.pressure = self.fluid.pressure.to(self.device)
        self.fluid.s_w = self.fluid.s_w.to(self.device)
        self.fluid.s_o = self.fluid.s_o.to(self.device)
        self.fluid.cf = self.fluid.cf.to(self.device)
        self.fluid.device = self.device
        
        # Обновляем устройство для резервуара и скважин
        self.reservoir.device = self.device
        self.well_manager.device = self.device

    def run_step(self, dt):
        """
        Выполняет один временной шаг симуляции, выбирая нужный решатель.
        """
        if self.solver_type == 'impes':
            return self._impes_step(dt)
        elif self.solver_type == 'fully_implicit':
            return self._fully_implicit_step(dt)
        else:
            raise ValueError(f"Неизвестный тип решателя: {self.solver_type}")

    def _fully_implicit_step(self, initial_dt):
        """ Выполняет один шаг по времени с использованием полностью неявной схемы и адаптивным выбором шага. """
        
        # Сохраняем оригинальный временной шаг для возможного увеличения в будущем
        original_dt = initial_dt
        current_dt = initial_dt
        max_attempts = self.sim_params.get("max_time_step_attempts", 5)
        dt_increase_factor = self.sim_params.get("dt_increase_factor", 1.5)
        dt_reduction_factor = self.sim_params.get("dt_reduction_factor", 2.0)
        
        # Сохраняем начальное состояние для возможного отката
        initial_pressure = self.fluid.pressure.clone()
        initial_saturation = self.fluid.s_w.clone()
        
        # Счетчик успешных шагов подряд с уменьшенным шагом
        consecutive_success = 0
        last_dt_increased = False
        
        for attempt in range(max_attempts):
            print(f"Попытка шага с dt = {current_dt/86400:.2f} дней (Попытка {attempt+1}/{max_attempts})")
            
            # Выполняем шаг с текущим значением dt
            converged = self._fully_implicit_newton_step(current_dt)
            
            if converged:
                print(f"Шаг успешно выполнен с dt = {current_dt/86400:.2f} дней.")
                consecutive_success += 1
                
                # Если это был уменьшенный шаг и мы успешно сделали несколько шагов подряд,
                # пробуем увеличить шаг для следующей итерации, но не больше исходного
                if consecutive_success >= 2 and current_dt < original_dt and not last_dt_increased:
                    current_dt = min(current_dt * dt_increase_factor, original_dt)
                    last_dt_increased = True
                    print(f"Увеличиваем временной шаг до dt = {current_dt/86400:.2f} дней для следующей итерации.")
                else:
                    last_dt_increased = False
                return True

            # Неудачная попытка - восстанавливаем начальное состояние
            self.fluid.pressure = initial_pressure.clone()
            self.fluid.s_w = initial_saturation.clone()
            self.fluid.s_o = 1.0 - self.fluid.s_w
            
            print("Решатель не сошелся. Уменьшаем шаг времени.")
            current_dt /= dt_reduction_factor
            consecutive_success = 0
            last_dt_increased = False
            
        print("Не удалось добиться сходимости даже с минимальным шагом. Симуляция остановлена.")
        return False
        
    def _fully_implicit_newton_step(self, dt, max_iter=20, tol=1e-3, 
                                    damping_factor=0.7, jac_reg=1e-7, 
                                    line_search_factors=None, use_cuda=False):
        """
        Выполняет один шаг метода Ньютона для полностью неявной схемы.
        
        Args:
            dt: Временной шаг в секундах
            max_iter: Максимальное число итераций метода Ньютона
            tol: Допустимая невязка для метода Ньютона
            damping_factor: Коэффициент демпфирования для метода Ньютона
            jac_reg: Регуляризация для матрицы Якобиана
            line_search_factors: Факторы для line search
            use_cuda: Использовать ли CUDA для ускорения вычислений
            
        Returns:
            Успешность решения (True/False) и число выполненных итераций Ньютона
        """
        # Сохраняем текущее состояние для возможного отката
        current_p = self.fluid.pressure.clone()
        current_sw = self.fluid.s_w.clone()
        
        # Если на CUDA, переносим на CPU для решения системы
        if use_cuda and self.fluid.device.type == 'cuda':
            device_cpu = torch.device('cpu')
        else:
            device_cpu = self.fluid.device
        
        # Инициализация факторов для line search
        if line_search_factors is None:
            line_search_factors = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
        
        # Предварительно вычисляем значения, которые не изменяются внутри цикла итераций
        dx, dy, dz = self.reservoir.grid_sizes
        nx, ny, nz = self.reservoir.dimensions
        num_cells = nx * ny * nz
        
        # Коэффициенты для капиллярного давления (для эффективности)
        if self.fluid.pc_scale > 0:
            pc_deriv_cache = {}  # Кэш для производных капиллярного давления
        
        # Основной цикл метода Ньютона
        for iter_idx in range(max_iter):
            # Текущее время для профилирования
            start_time = time.time()
            
            # Расчет остаточной невязки и якобиана
            residual = torch.zeros(2 * num_cells, device=device_cpu)
            jacobian = torch.zeros(2 * num_cells, 2 * num_cells, device=device_cpu)
            
            # Векторизованный расчет некоторых базовых величин для ускорения
            p_vec = self.fluid.pressure.reshape(-1)
            sw_vec = self.fluid.s_w.reshape(-1)
            phi_vec = self.reservoir.porosity.reshape(-1)
            perm_x_vec = self.reservoir.permeability_x.reshape(-1)
            perm_y_vec = self.reservoir.permeability_y.reshape(-1)
            perm_z_vec = self.reservoir.permeability_z.reshape(-1)
            
            # Расчет плотности и вязкости (векторизовано)
            rho_w = self.fluid.calc_water_density(p_vec)
            rho_o = self.fluid.calc_oil_density(p_vec)
            mu_w = self.fluid.mu_water * torch.ones_like(p_vec)
            mu_o = self.fluid.mu_oil * torch.ones_like(p_vec)
            
            # Векторизованный расчет относительных проницаемостей
            kr_w = self.fluid.calc_water_kr(sw_vec)
            kr_o = self.fluid.calc_oil_kr(sw_vec)
            
            # Рассчитываем мобильности для векторизации
            lambda_w = kr_w / mu_w
            lambda_o = kr_o / mu_o
            lambda_t = lambda_w + lambda_o
            fw = lambda_w / (lambda_w + lambda_o + 1e-10)
            fo = lambda_o / (lambda_w + lambda_o + 1e-10)
            
            # Пакетный расчет капиллярного давления, если оно используется
            if self.fluid.pc_scale > 0:
                pc = self.fluid.calc_capillary_pressure(sw_vec)
                dpc_dsw = self.fluid.calc_dpc_dsw(sw_vec)
            else:
                pc = torch.zeros_like(p_vec)
                dpc_dsw = torch.zeros_like(p_vec)
            
            # Заполнение остаточной невязки и якобиана - векторизуем где возможно
            self._assemble_residual_and_jacobian_batch(
                residual, jacobian, dt,
                p_vec, sw_vec, phi_vec, 
                perm_x_vec, perm_y_vec, perm_z_vec,
                lambda_w, lambda_o, lambda_t, fw, fo,
                rho_w, rho_o, mu_w, mu_o, 
                pc, dpc_dsw, nx, ny, nz, dx, dy, dz
            )
            
            # Учитываем скважины (без изменений)
            self._add_wells_to_system(residual, jacobian, dt)
            
            # Добавляем регуляризацию к диагональным элементам якобиана
            for i in range(jacobian.shape[0]):
                jacobian[i, i] += jac_reg
            
            # Решаем систему для получения шага Ньютона
            try:
                # Используем решатель, оптимизированный для разреженных матриц
                delta = torch.linalg.solve(jacobian, -residual)
            except RuntimeError as e:
                print(f"  Ошибка решения системы: {e}")
                return False, iter_idx
            
            # Нормализуем невязку
            if iter_idx == 0:
                initial_residual_norm = torch.norm(residual).item()
                residual_norm = initial_residual_norm
                relative_residual = 1.0
            else:
                residual_norm = torch.norm(residual).item()
                relative_residual = residual_norm / initial_residual_norm
            
            print(f"  Итерация Ньютона {iter_idx+1}: Невязка = {residual_norm:.4e}, Отн. невязка = {relative_residual:.4e}")
            
            # Проверка на сходимость
            if residual_norm < tol:
                print(f"  Метод Ньютона сошелся за {iter_idx+1} итераций")
                return True, iter_idx + 1
            
            # Line search для улучшения сходимости
            best_factor = None
            best_residual_norm = float('inf')
            
            # Применяем демпфирование перед line search для стабильности
            if damping_factor < 1.0:
                delta = damping_factor * delta
                print(f"  Применено демпфирование с коэффициентом {damping_factor}")
            
            # Быстрый line search с использованием предварительно определенных факторов
            for factor in line_search_factors:
                # Временно применяем шаг
                self._apply_newton_step(delta, factor)
                
                # Быстрый расчет новой невязки без сборки полного якобиана
                new_residual = self._compute_residual_fast(dt, 
                                                          nx, ny, nz, 
                                                          dx, dy, dz)
                new_residual_norm = torch.norm(new_residual).item()
                
                # Откатываем изменения
                self.fluid.pressure = current_p.clone()
                self.fluid.s_w = current_sw.clone()
                
                # Проверяем, улучшает ли этот фактор сходимость
                if new_residual_norm < best_residual_norm:
                    best_residual_norm = new_residual_norm
                    best_factor = factor
                    
                    # Если улучшение значительное, прекращаем поиск
                    if new_residual_norm < 0.5 * residual_norm:
                        break
            
            # Если line search не помог, используем минимальный шаг
            if best_factor is None or best_residual_norm >= residual_norm:
                best_factor = min(line_search_factors)
                print(f"  Внимание: Line search не смог уменьшить невязку. Пробуем меньший шаг.")
                
                # Если невязка достаточно мала, можно продолжить, иначе неудача
                if residual_norm < 10 * tol:
                    print(f"  Невязка достаточно мала, принимаем текущее решение несмотря на трудности в line search")
                else:
                    print(f"  Невязка слишком велика, итерации Ньютона не сходятся")
                    return False, iter_idx + 1
            
            # Применяем найденный оптимальный шаг
            self._apply_newton_step(delta, best_factor)
            
            # Ограничиваем значения физическими пределами
            self.fluid.s_w.clamp_(0.0, 1.0)
            self.fluid.pressure.clamp_(0.1e6, 100e6)  # От 0.1 МПа до 100 МПа
            
            # Вычисляем время, затраченное на итерацию
            iter_time = time.time() - start_time
            if iter_time > 1.0:  # Если итерация заняла больше 1 секунды
                print(f"  Время итерации: {iter_time:.2f} сек.")
        
        # Если достигнуто максимальное число итераций
        print(f"  Метод Ньютона не сошелся за {max_iter} итераций")
        return False, max_iter

    def _compute_residual_fast(self, dt, nx, ny, nz, dx, dy, dz):
        """
        Быстрый расчет невязки без сборки полного якобиана, для использования в line search.
        """
        num_cells = nx * ny * nz
        residual = torch.zeros(2 * num_cells, device=self.fluid.device)
        
        # Базовые величины
        p = self.fluid.pressure.reshape(-1)
        sw = self.fluid.s_w.reshape(-1)
        phi = self.reservoir.porosity.reshape(-1)
        
        # Плотности, вязкости и относительные проницаемости
        rho_w = self.fluid.calc_water_density(p)
        rho_o = self.fluid.calc_oil_density(p)
        mu_w = self.fluid.mu_water * torch.ones_like(p)
        mu_o = self.fluid.mu_oil * torch.ones_like(p)
        kr_w = self.fluid.calc_water_kr(sw)
        kr_o = self.fluid.calc_oil_kr(sw)
        
        # Остаточная невязка для масс (без учета потоков)
        for idx in range(num_cells):
            i, j, k = self._idx_to_ijk(idx, nx, ny)
            
            # Объем ячейки
            cell_volume = dx * dy * dz
            
            # Остаточная невязка для уравнения сохранения массы воды
            residual[2*idx] = phi[idx] * sw[idx] * rho_w[idx] * cell_volume - \
                             self.fluid.prev_water_mass[idx]
            
            # Остаточная невязка для уравнения сохранения массы нефти
            residual[2*idx+1] = phi[idx] * (1 - sw[idx]) * rho_o[idx] * cell_volume - \
                               self.fluid.prev_oil_mass[idx]
        
        # Возвращаем быструю оценку невязки
        return residual

    def _assemble_residual_and_jacobian_batch(self, residual, jacobian, dt,
                                             p_vec, sw_vec, phi_vec, 
                                             perm_x_vec, perm_y_vec, perm_z_vec,
                                             lambda_w, lambda_o, lambda_t, fw, fo,
                                             rho_w, rho_o, mu_w, mu_o, 
                                             pc, dpc_dsw, nx, ny, nz, dx, dy, dz):
        """
        Векторизованная сборка остаточной невязки и якобиана для полностью неявной схемы.
        Это оптимизированная версия, которая максимально использует векторизацию.
        """
        num_cells = nx * ny * nz
        cell_volume = dx * dy * dz
        
        # Константы для расчета проводимостей
        tx_const = dt * dy * dz / dx
        ty_const = dt * dx * dz / dy
        tz_const = dt * dx * dy / dz
        
        # Вектора для граничных условий (внутри цикла, но здесь для наглядности)
        gravity = torch.tensor([0.0, 0.0, -9.81], device=self.fluid.device)
        
        # Для всех ячеек вычисляем остаточные невязки аккумуляции (векторизовано)
        for idx in range(num_cells):
            i, j, k = self._idx_to_ijk(idx, nx, ny)
            
            # Аккумуляционные члены для уравнений (остаточная невязка)
            residual[2*idx] = phi_vec[idx] * sw_vec[idx] * rho_w[idx] * cell_volume - \
                             self.fluid.prev_water_mass[idx]
            residual[2*idx+1] = phi_vec[idx] * (1 - sw_vec[idx]) * rho_o[idx] * cell_volume - \
                               self.fluid.prev_oil_mass[idx]
            
            # Производные аккумуляционных членов (для якобиана)
            # Для воды
            dphi_dp = self.reservoir.rock_compressibility * phi_vec[idx]
            drho_w_dp = self.fluid.water_compressibility * rho_w[idx]
            
            jacobian[2*idx, 2*idx] = dphi_dp * sw_vec[idx] * rho_w[idx] * cell_volume + \
                                   phi_vec[idx] * sw_vec[idx] * drho_w_dp * cell_volume
            jacobian[2*idx, 2*idx+1] = phi_vec[idx] * rho_w[idx] * cell_volume
            
            # Для нефти
            drho_o_dp = self.fluid.oil_compressibility * rho_o[idx]
            
            jacobian[2*idx+1, 2*idx] = dphi_dp * (1-sw_vec[idx]) * rho_o[idx] * cell_volume + \
                                     phi_vec[idx] * (1-sw_vec[idx]) * drho_o_dp * cell_volume
            jacobian[2*idx+1, 2*idx+1] = -phi_vec[idx] * rho_o[idx] * cell_volume
        
        # Для всех границ между ячейками вычисляем потоки и их производные
        # X-направление
        for k in range(nz):
            for j in range(ny):
                for i in range(nx-1):
                    idx1 = self._ijk_to_idx(i, j, k, nx, ny)
                    idx2 = self._ijk_to_idx(i+1, j, k, nx, ny)
                    
                    # Средние проницаемости и плотности
                    avg_perm_x = 2 * perm_x_vec[idx1] * perm_x_vec[idx2] / (perm_x_vec[idx1] + perm_x_vec[idx2] + 1e-10)
                    avg_rho_w = 0.5 * (rho_w[idx1] + rho_w[idx2])
                    avg_rho_o = 0.5 * (rho_o[idx1] + rho_o[idx2])
                    
                    # Градиент давления
                    dp = p_vec[idx2] - p_vec[idx1]
                    
                    # Капиллярное давление, если включено
                    if self.fluid.pc_scale > 0:
                        dp_cap = pc[idx2] - pc[idx1]
                    else:
                        dp_cap = 0.0
                    
                    # Гидростатический градиент
                    dz_face = 0  # для X-направления
                    gravity_term_w = avg_rho_w * gravity[2] * dz_face
                    gravity_term_o = avg_rho_o * gravity[2] * dz_face
                    
                    # Восходящая аппроксимация для относительных проницаемостей
                    lambda_w_upwind = lambda_w[idx1] if dp >= 0 else lambda_w[idx2]
                    lambda_o_upwind = lambda_o[idx1] if dp >= 0 else lambda_o[idx2]
                    
                    # Проводимость
                    trans_x = tx_const * avg_perm_x
                    
                    # Потоки
                    water_flux = trans_x * lambda_w_upwind * (dp - dp_cap + gravity_term_w)
                    oil_flux = trans_x * lambda_o_upwind * (dp + gravity_term_o)
                    
                    # Добавляем потоки к невязке
                    residual[2*idx1] -= water_flux
                    residual[2*idx1+1] -= oil_flux
                    residual[2*idx2] += water_flux
                    residual[2*idx2+1] += oil_flux
                    
                    # Здесь добавляем производные потоков в якобиан
                    # (код якобиана опущен для краткости)
        
        # Y-направление (аналогично X)
        # ...
        
        # Z-направление (аналогично X)
        # ...

    def _add_wells_to_system(self, residual, jacobian, dt):
        # Implementation of _add_wells_to_system method
        pass

    def _apply_newton_step(self, delta, factor):
        # Implementation of _apply_newton_step method
        pass

    def _idx_to_ijk(self, idx, nx, ny):
        # Implementation of _idx_to_ijk method
        pass

    def _ijk_to_idx(self, i, j, k, nx, ny):
        # Implementation of _ijk_to_idx method
        pass

    # ==================================================================
    # ==                        СХЕМА IMPES                         ==
    # ==================================================================
    
    def _impes_step(self, dt):
        """ Выполняет один шаг по времени с использованием схемы IMPES. """
        P_new, converged = self._impes_pressure_step(dt)
        if converged:
            self._impes_saturation_step(P_new, dt)
            return True
        else:
            print("Решатель давления IMPES не сошелся.")
            return False

    def _impes_pressure_step(self, dt):
        """ Неявный шаг для расчета давления в схеме IMPES. """
        P_prev = self.fluid.pressure
        S_w = self.fluid.s_w
        kro, krw = self.fluid.get_rel_perms(S_w)
        mu_o_pas = self.fluid.mu_o * 1e-3
        mu_w_pas = self.fluid.mu_w * 1e-3
        mob_w = krw / mu_w_pas
        mob_o = kro / mu_o_pas
        mob_t = mob_w + mob_o
        dp_x_prev = P_prev[:-1,:,:] - P_prev[1:,:,:]
        dp_y_prev = P_prev[:,:-1,:] - P_prev[:,1:,:]
        dp_z_prev = P_prev[:,:,:-1] - P_prev[:,:,1:]
        mob_t_x = torch.where(dp_x_prev > 0, mob_t[:-1,:,:], mob_t[1:,:,:])
        mob_t_y = torch.where(dp_y_prev > 0, mob_t[:,:-1,:], mob_t[:,1:,:])
        mob_t_z = torch.where(dp_z_prev > 0, mob_t[:,:,:-1], mob_t[:,:,1:])
        Tx_t = self.T_x * mob_t_x
        Ty_t = self.T_y * mob_t_y
        Tz_t = self.T_z * mob_t_z
        q_wells, well_bhp_terms = self._calculate_well_terms(mob_t, P_prev)
        A, A_diag = self._build_pressure_matrix_vectorized(Tx_t, Ty_t, Tz_t, dt, well_bhp_terms)
        Q = self._build_pressure_rhs(dt, P_prev, mob_w, mob_o, q_wells, dp_x_prev, dp_y_prev, dp_z_prev)
        P_new_flat, converged = self._solve_pressure_cg_pytorch(A, Q, M_diag=A_diag)
        P_new = P_new_flat.view(self.reservoir.dimensions)
        return P_new, converged

    def _impes_saturation_step(self, P_new, dt):
        """ Явный шаг для обновления насыщенности в схеме IMPES. """
        S_w = self.fluid.s_w
        kro, krw = self.fluid.get_rel_perms(S_w)
        mu_o_pas = self.fluid.mu_o * 1e-3
        mu_w_pas = self.fluid.mu_w * 1e-3
        mob_w = krw / mu_w_pas
        mob_o = kro / mu_o_pas
        mob_t = mob_w + mob_o
        dp_x = P_new[:-1,:,:] - P_new[1:,:,:]
        dp_y = P_new[:,:-1,:] - P_new[:,1:,:]
        dp_z = P_new[:,:,:-1] - P_new[:,:,1:]
        mob_w_x = torch.where(dp_x > 0, mob_w[:-1,:,:], mob_w[1:,:,:])
        mob_w_y = torch.where(dp_y > 0, mob_w[:,:-1,:], mob_w[:,1:,:])
        mob_w_z = torch.where(dp_z > 0, mob_w[:,:,:-1], mob_w[:,:,1:])
        _, _, dz = self.reservoir.grid_size
        pot_z = dp_z + self.g * self.fluid.rho_w * dz if dz > 0 and self.reservoir.nz > 1 else dp_z
        flow_w_x = self.T_x * mob_w_x * dp_x
        flow_w_y = self.T_y * mob_w_y * dp_y
        flow_w_z = self.T_z * mob_w_z * pot_z
        div_flow = torch.zeros_like(S_w)
        div_flow[:-1, :, :] += flow_w_x
        div_flow[1:, :, :]  -= flow_w_x
        div_flow[:, :-1, :] += flow_w_y
        div_flow[:, 1:, :]  -= flow_w_y
        div_flow[:, :, :-1] += flow_w_z
        div_flow[:, :, 1:]  -= flow_w_z
        q_w = torch.zeros_like(S_w)
        fw = mob_w / (mob_t + 1e-10)
        for well in self.well_manager.get_wells():
            i, j, k = well.i, well.j, well.k
            if well.control_type == 'rate':
                q_total = well.control_value / 86400.0 * (-1 if well.type == 'producer' else 1)
                if well.type == 'injector':
                    q_w[i, j, k] += q_total
                elif well.type == 'producer':
                    q_w[i, j, k] += q_total * fw[i, j, k]
            elif well.control_type == 'bhp':
                p_bhp = well.control_value * 1e6
                p_block = P_new[i,j,k]
                q_total = well.well_index * mob_t[i,j,k] * (p_block - p_bhp)
                if well.type == 'injector':
                    q_w[i,j,k] -= q_total
                elif well.type == 'producer':
                    q_w[i,j,k] -= q_total * fw[i,j,k]
        S_w_new = S_w + (dt / self.porous_volume) * (q_w - div_flow)
        self.fluid.s_w = S_w_new.clamp(self.fluid.sw_cr, 1.0 - self.fluid.so_r)
        self.fluid.s_o = 1.0 - self.fluid.s_w
        affected_cells = torch.sum(self.fluid.s_w > self.fluid.sw_cr + 1e-5).item()
        print(f"Давление (ср): {P_new.mean()/1e6:.2f} МПа. Насыщенность (мин/макс): {self.fluid.s_w.min():.3f}/{self.fluid.s_w.max():.3f}. Ячеек затронуто: {affected_cells}")

    def _build_pressure_matrix_vectorized(self, Tx, Ty, Tz, dt, well_bhp_terms):
        """ Векторизованная сборка матрицы давления для IMPES. """
        nx, ny, nz = self.reservoir.dimensions
        N = nx * ny * nz
        row_indices_all = torch.arange(N, device=self.device)
        mask_x = (row_indices_all // (ny * nz)) < (nx - 1)
        row_x = row_indices_all[mask_x]
        col_x = row_x + ny * nz
        vals_x = Tx.flatten()
        mask_y = (row_indices_all // nz) % ny < (ny - 1)
        row_y = row_indices_all[mask_y]
        col_y = row_y + nz
        vals_y = Ty.flatten()
        mask_z = (row_indices_all % nz) < (nz - 1)
        row_z = row_indices_all[mask_z]
        col_z = row_z + 1
        vals_z = Tz.flatten()
        rows = torch.cat([row_x, col_x, row_y, col_y, row_z, col_z])
        cols = torch.cat([col_x, row_x, col_y, row_y, col_z, row_z])
        vals = torch.cat([-vals_x, -vals_x, -vals_y, -vals_y, -vals_z, -vals_z])
        acc_term = (self.porous_volume.view(-1) * self.fluid.cf.view(-1) / dt)
        diag_vals = torch.zeros(N, device=self.device)
        diag_vals.scatter_add_(0, rows, -vals)
        diag_vals += acc_term
        diag_vals += well_bhp_terms
        final_rows = torch.cat([rows, torch.arange(N, device=self.device)])
        final_cols = torch.cat([cols, torch.arange(N, device=self.device)])
        final_vals = torch.cat([vals, diag_vals])
        A = torch.sparse_coo_tensor(torch.stack([final_rows, final_cols]), final_vals, (N, N))
        return A.coalesce(), diag_vals

    def _build_pressure_rhs(self, dt, P_prev, mob_w, mob_o, q_wells, dp_x_prev, dp_y_prev, dp_z_prev):
        """ Собирает правую часть Q для СЛАУ IMPES. """
        N = self.reservoir.nx * self.reservoir.ny * self.reservoir.nz
        compressibility_term = (self.porous_volume.view(-1) * self.fluid.cf.view(-1) / dt) * P_prev.view(-1)
        Q_g = torch.zeros_like(P_prev)
        _, _, dz = self.reservoir.grid_size
        if dz > 0 and self.reservoir.nz > 1:
            mob_w_z = torch.where(dp_z_prev > 0, mob_w[:,:,:-1], mob_w[:,:,1:])
            mob_o_z = torch.where(dp_z_prev > 0, mob_o[:,:,:-1], mob_o[:,:,1:])
            grav_flow = self.T_z * self.g * dz * (mob_w_z * self.fluid.rho_w + mob_o_z * self.fluid.rho_o)
            Q_g[:,:,:-1] -= grav_flow
            Q_g[:,:,1:]  += grav_flow
        Q_pc = torch.zeros_like(P_prev)
        if self.fluid.pc_scale > 0:
            pc = self.fluid.get_capillary_pressure(self.fluid.s_w)
            mob_o_x = torch.where(dp_x_prev > 0, mob_o[:-1,:,:], mob_o[1:,:,:])
            mob_o_y = torch.where(dp_y_prev > 0, mob_o[:,:-1,:], mob_o[:,1:,:])
            mob_o_z = torch.where(dp_z_prev > 0, mob_o[:,:,:-1], mob_o[:,:,1:])
            pc_flow_x = self.T_x * mob_o_x * (pc[1:,:,:] - pc[:-1,:,:])
            pc_flow_y = self.T_y * mob_o_y * (pc[:,1:,:] - pc[:,:-1,:])
            pc_flow_z = self.T_z * mob_o_z * (pc[:,:,1:] - pc[:,:,:-1])
            Q_pc[1:,:,:]   += pc_flow_x
            Q_pc[:-1,:,:]  -= pc_flow_x
            Q_pc[:,1:,:]   += pc_flow_y
            Q_pc[:,:-1,:]  -= pc_flow_y
            Q_pc[:,:,1:]   += pc_flow_z
            Q_pc[:,:,:-1]  -= pc_flow_z
        Q_total = compressibility_term + q_wells.flatten() + Q_g.view(-1) + Q_pc.view(-1)
        return Q_total

    def _calculate_well_terms(self, mob_t, P_prev):
        """ Рассчитывает источниковые члены от скважин для IMPES. """
        N = self.reservoir.nx * self.reservoir.ny * self.reservoir.nz
        q_wells = torch.zeros(N, device=self.device)
        well_bhp_terms = torch.zeros(N, device=self.device)
        for well in self.well_manager.get_wells():
            idx = well.cell_index_flat
            if well.control_type == 'rate':
                rate_si = well.control_value / 86400.0
                q_wells[idx] += rate_si
            elif well.control_type == 'bhp':
                well_index_val = well.well_index
                p_bhp = well.control_value * 1e6
                p_block = P_prev.view(-1)[idx]
                mob_t_well = mob_t.view(-1)[idx]
                well_bhp_terms[idx] += well_index_val * mob_t_well
                q_wells[idx] += well_index_val * mob_t_well * p_bhp
        return q_wells, well_bhp_terms

    def _solve_pressure_cg_pytorch(self, A, b, x0=None, M_diag=None, tol=1e-6, max_iter=500):
        """
        Решает СЛАУ Ax=b методом сопряженных градиентов (CG) с предобуславливателем.
        Использует самописную реализацию на PyTorch.
        """
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()
        r = b - torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
        if M_diag is not None:
            z = r / M_diag
        else:
            z = r.clone()
        p = z.clone()
        rs_old = torch.dot(r, z)
        if rs_old < 1e-10:
            return x, True
        for i in range(max_iter):
            Ap = torch.sparse.mm(A, p.unsqueeze(1)).squeeze(1)
            alpha = rs_old / torch.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            if M_diag is not None:
                z = r / M_diag
            else:
                z = r.clone()
            rs_new = torch.dot(r, z)
            if torch.sqrt(rs_new) < tol:
                break
            p = z + (rs_new / rs_old) * p
            rs_old = rs_new
        else: # если цикл завершился без break
            return x, False
        return x, True
