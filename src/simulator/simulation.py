import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, diags, bmat
from scipy.sparse.linalg import cg, LinearOperator, bicgstab, spsolve
import time
import os
import datetime
import copy

class Simulator:
    """
    Основной класс симулятора, отвечающий за выполнение расчетов.
    Поддерживает две схемы:
    - IMPES (Implicit Pressure, Explicit Saturation)
    - Полностью неявную (Fully Implicit)
    
    Поддерживает выполнение на CPU или GPU (если доступна CUDA).
    """
    def __init__(self, reservoir, fluid, well_manager, sim_params, device=None):
        """
        Инициализирует симулятор.
        
        Args:
            reservoir: Объект пласта
            fluid: Объект флюидов
            well_manager: Объект менеджера скважин
            sim_params: Параметры симуляции
            device: Устройство для вычислений (CPU/GPU)
        """
        self.reservoir = reservoir
        self.fluid = fluid
        self.well_manager = well_manager
        self.sim_params = sim_params
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Получаем тип решателя
        self.solver_type = sim_params.get('solver_type', 'impes')
        
        print(f"Инициализация симулятора...")
        print(f"  Тип решателя: {self.solver_type}")
        print(f"  Временной шаг: {sim_params.get('time_step_days', 1.0)} дней")
        
        if self.solver_type == 'fully_implicit':
            print(f"  Настройки решателя Ньютона:")
            print(f"    Макс. итераций: {sim_params.get('newton_max_iter', 20)}")
            print(f"    Допуск невязки: {sim_params.get('newton_tolerance', 1e-3)}")
            print(f"    Коэфф. демпфирования: {sim_params.get('damping_factor', 0.7)}")
            print(f"    Используется CUDA: {sim_params.get('use_cuda', False)}")
            print(f"    Адаптивный временной шаг: {sim_params.get('max_time_step_attempts', 1) > 1}")
        
        # Инициализация проводимостей для IMPES схемы
        if self.solver_type == 'impes':
            # Получаем размеры ячеек и проницаемости
            dx_mean, dy_mean, dz_mean = self.reservoir.grid_size
            k_x, k_y, k_z = self.reservoir.permeability_tensors
            nx, ny, nz = self.reservoir.dimensions
            
            # Вычисляем проводимости для каждого направления
            self.T_x = torch.zeros((nx-1, ny, nz), device=self.device)
            self.T_y = torch.zeros((nx, ny-1, nz), device=self.device)
            self.T_z = torch.zeros((nx, ny, nz-1), device=self.device)
            dx_vec = self.reservoir.dx_vector.to(self.device, dtype=torch.float64)
            dy_vec = self.reservoir.dy_vector.to(self.device, dtype=torch.float64)
            dz_vec = self.reservoir.dz_vector.to(self.device, dtype=torch.float64)
            dx_face = self.reservoir.dx_face.to(self.device, dtype=torch.float64)
            dy_face = self.reservoir.dy_face.to(self.device, dtype=torch.float64)
            dz_face = self.reservoir.dz_face.to(self.device, dtype=torch.float64)

            # Расчет проводимостей
            if nx > 1:
                area_yz = (dy_vec.view(1, ny, 1) * dz_vec.view(1, 1, nz)).to(self.device)
                kx_local = k_x.to(self.device)
                k_harmonic = 2 * kx_local[:-1, :, :] * kx_local[1:, :, :] / (kx_local[:-1, :, :] + kx_local[1:, :, :] + 1e-15)
                self.T_x = k_harmonic * area_yz / (dx_face.view(-1, 1, 1) + 1e-15)
            if ny > 1:
                area_xz = (dx_vec.view(nx, 1, 1) * dz_vec.view(1, 1, nz)).to(self.device)
                ky_local = k_y.to(self.device)
                k_harmonic = 2 * ky_local[:, :-1, :] * ky_local[:, 1:, :] / (ky_local[:, :-1, :] + ky_local[:, 1:, :] + 1e-15)
                self.T_y = k_harmonic * area_xz / (dy_face.view(1, -1, 1) + 1e-15)
            if nz > 1:
                area_xy = (dx_vec.view(nx, 1, 1) * dy_vec.view(1, ny, 1)).to(self.device)
                kz_local = k_z.to(self.device)
                k_harmonic = 2 * kz_local[:, :, :-1] * kz_local[:, :, 1:] / (kz_local[:, :, :-1] + kz_local[:, :, 1:] + 1e-15)
                self.T_z = k_harmonic * area_xy / (dz_face.view(1, 1, -1) + 1e-15)
        
        # Объем пористой среды для расчетов IMPES
        self.porous_volume = self.reservoir.porous_volume
        
        # Гравитационная постоянная для IMPES
        self.g = 9.81

        self.use_capillary_potentials = bool(self.sim_params.get("use_capillary_potentials", False))
        
        # Трекер баланса масс по фазам (кг)
        self.mass_balance = {
            'water': {'in': 0.0, 'out': 0.0},
            'oil': {'in': 0.0, 'out': 0.0},
            'gas': {'in': 0.0, 'out': 0.0},
        }
        # Трекер компонентного баланса в surface-единицах (м3 при ст. условиях)
        self.component_balance = {
            'oil': {'in': 0.0, 'out': 0.0, 'accum': 0.0},
            'gas': {'in': 0.0, 'out': 0.0, 'accum': 0.0},
        }

        # Отладочное логирование компонентного баланса
        self.debug_component_balance = bool(
            sim_params.get('debug_component_balance', False)
            or os.environ.get('OIL_DEBUG_COMPONENT', '0') == '1'
        )
        if self.debug_component_balance:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            default_path = os.path.join('results', f'component_debug_{ts}.log')
            self.debug_log_path = sim_params.get('debug_log_path', default_path)
            os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
            with open(self.debug_log_path, 'w') as f:
                f.write('# Component balance debug log\n')
                f.write(f'# started {ts}\n')

        # Геометрия ячеек для AMG near-nullspace
        self._pressure_cell_centers = self._compute_pressure_cell_centers()
        self._pressure_near_nullspace = self._build_pressure_near_nullspace(self._pressure_cell_centers)
        
    def _move_data_to_device(self):
        """Переносит данные на текущее устройство (CPU или GPU)"""
        # Резервуар
        attrs = [
            "permeability_x",
            "permeability_y",
            "permeability_z",
            "porosity",
            "porous_volume",
            "grid_size",
            "dx_vector",
            "dy_vector",
            "dz_vector",
            "dx_face",
            "dy_face",
            "dz_face",
            "cell_volume",
            "cell_volume_flat",
            "x_centers",
            "y_centers",
            "z_centers",
            "domain_lengths",
        ]
        for name in attrs:
            tensor = getattr(self.reservoir, name, None)
            if tensor is not None and hasattr(tensor, "to"):
                setattr(self.reservoir, name, tensor.to(self.device))

        # Флюид
        fluid_attrs = [
            "pressure",
            "s_w",
            "s_o",
            "cf",
            "prev_pressure",
            "prev_sw",
            "prev_sg",
        ]
        for name in fluid_attrs:
            tensor = getattr(self.fluid, name, None)
            if tensor is not None and hasattr(tensor, "to"):
                setattr(self.fluid, name, tensor.to(self.device))
        if hasattr(self.fluid, "s_g") and self.fluid.s_g is not None:
            self.fluid.s_g = self.fluid.s_g.to(self.device)
        if hasattr(self.fluid, "prev_sg") and self.fluid.prev_sg is not None:
            self.fluid.prev_sg = self.fluid.prev_sg.to(self.device)
        if hasattr(self.fluid, "prev_oil_mass") and self.fluid.prev_oil_mass is not None:
            self.fluid.prev_oil_mass = self.fluid.prev_oil_mass.to(self.device)
        if hasattr(self.fluid, "prev_water_mass") and self.fluid.prev_water_mass is not None:
            self.fluid.prev_water_mass = self.fluid.prev_water_mass.to(self.device)
        if hasattr(self.fluid, "prev_gas_mass") and self.fluid.prev_gas_mass is not None:
            self.fluid.prev_gas_mass = self.fluid.prev_gas_mass.to(self.device)

        self.fluid.device = self.device
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

    def _fully_implicit_step(self, dt):
        """
        Выполняет один временной шаг с использованием полностью неявной схемы.
        
        Args:
            dt: Временной шаг в секундах
            
        Returns:
            Успешность выполнения шага (True/False)
        """
        # Сохраняем оригинальный временной шаг для возможного увеличения в будущем
        original_dt = dt
        current_dt = dt
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
                
                # Обновляем предыдущие состояния флюида
                self.fluid.prev_pressure = self.fluid.pressure.clone()
                self.fluid.prev_sw = self.fluid.s_w.clone()
                
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
        Максимально оптимизированная реализация с улучшенным методом line search.
        
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
        # Получаем настройки из параметров симуляции, если не указаны явно
        if max_iter is None:
            max_iter = self.sim_params.get("newton_max_iter", 20)
        if tol is None:
            tol = self.sim_params.get("newton_tolerance", 1e-3)
        if damping_factor is None:
            damping_factor = self.sim_params.get("damping_factor", 0.7)
        if jac_reg is None:
            jac_reg = self.sim_params.get("jacobian_regularization", 1e-7)
        if use_cuda is None:
            use_cuda = self.sim_params.get("use_cuda", False)
        
        # Сохраняем текущее состояние для возможного отката
        current_p = self.fluid.pressure.clone()
        current_sw = self.fluid.s_w.clone()
        
        # Инициализация параметров для оптимизации
        nx, ny, nz = self.reservoir.dimensions
        if not self.reservoir.is_uniform_grid:
            raise NotImplementedError("Fast residual assembly поддерживает только равномерную сетку")
        num_cells = nx * ny * nz
        
        # Устанавливаем устройство в зависимости от доступности CUDA
        if use_cuda and torch.cuda.is_available() and self.fluid.device.type == 'cuda':
            device = self.fluid.device
            device_cpu = torch.device('cpu')
            using_cuda = True
        else:
            device = self.fluid.device
            device_cpu = device
            using_cuda = False
        
        # Инициализация факторов для line search с более плавным убыванием для улучшения сходимости
        if line_search_factors is None:
            line_search_factors = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01]
        
        # Предварительно вычисляем параметры сетки
        dx, dy, dz = self.reservoir.grid_size
        
        # Флаг для отслеживания предыдущей невязки для адаптивной сходимости
        prev_residual_norm = float('inf')
        
        # Основной цикл метода Ньютона
        for iter_idx in range(max_iter):
            # Время начала итерации для профилирования
            start_time = time.time()
            
            # Расчет остаточной невязки и якобиана
            if using_cuda:
                # Для CUDA: создаем тензоры на CPU для Якобиана (более эффективное решение СЛАУ)
                residual = torch.zeros(2 * num_cells, device=device_cpu)
                jacobian = torch.zeros(2 * num_cells, 2 * num_cells, device=device_cpu)
            else:
                # Для CPU: создаем тензоры на том же устройстве
                residual = torch.zeros(2 * num_cells, device=device)
                jacobian = torch.zeros(2 * num_cells, 2 * num_cells, device=device)
            
            # Векторизованный расчет базовых величин
            p_vec = self.fluid.pressure.reshape(-1)
            sw_vec = self.fluid.s_w.reshape(-1)
            phi_vec = self.reservoir.porosity.reshape(-1)
            perm_x_vec = self.reservoir.permeability_x.reshape(-1)
            perm_y_vec = self.reservoir.permeability_y.reshape(-1)
            perm_z_vec = self.reservoir.permeability_z.reshape(-1)
            
            # Используем JIT-компиляцию для расчета плотностей и вязкостей, если доступно
            if hasattr(torch, 'jit') and not using_cuda:
                try:
                    # Определение JIT-функций (только если они еще не определены)
                    if not hasattr(self, '_jit_rho_w'):
                        @torch.jit.script
                        def calc_rho_w(p, rho_w_ref, c_w):
                            return rho_w_ref * (1.0 + c_w * (p - 1e5))
                        
                        @torch.jit.script
                        def calc_rho_o(p, rho_o_ref, c_o):
                            return rho_o_ref * (1.0 + c_o * (p - 1e5))
                        
                        self._jit_rho_w = calc_rho_w
                        self._jit_rho_o = calc_rho_o
                    
                    # Использование JIT-функций
                    rho_w = self._jit_rho_w(p_vec, self.fluid.rho_water_ref, self.fluid.water_compressibility)
                    rho_o = self._jit_rho_o(p_vec, self.fluid.rho_oil_ref, self.fluid.oil_compressibility)
                except Exception:
                    # Если JIT не работает, используем обычный расчет
                    rho_w = self.fluid.calc_water_density(p_vec)
                    rho_o = self.fluid.calc_oil_density(p_vec)
            else:
                # Стандартный расчет плотностей
                rho_w = self.fluid.calc_water_density(p_vec)
                rho_o = self.fluid.calc_oil_density(p_vec)
            
            # Вязкости (константы)
            mu_w = self.fluid.mu_water * torch.ones_like(p_vec)
            mu_o = self.fluid.mu_oil * torch.ones_like(p_vec)
            
            # Расчет относительных проницаемостей и их производных
            kr_w = self.fluid.calc_water_kr(sw_vec)
            kr_o = self.fluid.calc_oil_kr(sw_vec)
            
            # Расчет мобильностей для векторизации
            lambda_w = kr_w / mu_w
            lambda_o = kr_o / mu_o
            lambda_t = lambda_w + lambda_o
            fw = lambda_w / (lambda_w + lambda_o + 1e-10)
            fo = lambda_o / (lambda_w + lambda_o + 1e-10)
            
            # Расчет капиллярного давления и его производной
            if self.fluid.pc_scale > 0:
                pc = self.fluid.calc_capillary_pressure(sw_vec)
                dpc_dsw = self.fluid.calc_dpc_dsw(sw_vec)
            else:
                pc = torch.zeros_like(p_vec)
                dpc_dsw = torch.zeros_like(p_vec)
            
            # Сохраняем предыдущие массы флюидов, если еще не сохранены
            if iter_idx == 0:
                cell_volume = self.reservoir.cell_volume_flat.to(self.fluid.device)
                self.fluid.prev_water_mass = phi_vec * self.fluid.prev_sw.reshape(-1) * \
                                            self.fluid.calc_water_density(self.fluid.prev_pressure.reshape(-1)) * \
                                            cell_volume
                self.fluid.prev_oil_mass = phi_vec * (1 - self.fluid.prev_sw.reshape(-1)) * \
                                          self.fluid.calc_oil_density(self.fluid.prev_pressure.reshape(-1)) * \
                                          cell_volume
            
            # Векторизованная сборка невязки и якобиана
            self._assemble_residual_and_jacobian_batch(
                residual, jacobian, dt,
                p_vec, sw_vec, phi_vec, 
                perm_x_vec, perm_y_vec, perm_z_vec,
                lambda_w, lambda_o, lambda_t, fw, fo,
                rho_w, rho_o, mu_w, mu_o, 
                pc, dpc_dsw, nx, ny, nz, dx, dy, dz
            )
            
            # Добавляем регуляризацию к диагональным элементам якобиана
            for i in range(jacobian.shape[0]):
                jacobian[i, i] += jac_reg
            
            # Решаем систему для получения шага Ньютона
            try:
                # Используем оптимизированный солвер для разреженных матриц
                if jacobian.shape[0] > 1000:  # Для больших систем используем итеративные методы
                    from scipy.sparse import csr_matrix
                    from scipy.sparse.linalg import spsolve, bicgstab
                    
                    # Преобразуем в разреженный формат для быстрого решения
                    jacobian_np = jacobian.cpu().numpy()
                    residual_np = residual.cpu().numpy()
                    jacobian_sparse = csr_matrix(jacobian_np)
                    
                    # Пробуем решить с помощью прямого метода
                    try:
                        delta_np = spsolve(jacobian_sparse, -residual_np)
                        delta = torch.from_numpy(delta_np).to(device)
                    except Exception:
                        # Если прямой метод не работает, используем итеративный
                        delta_np, info = bicgstab(jacobian_sparse, -residual_np, tol=1e-6, maxiter=1000)
                        if info != 0:
                            print(f"  Предупреждение: Итеративный решатель не сошелся (код {info})")
                        delta = torch.from_numpy(delta_np).to(device)
                else:
                    # Для небольших систем используем прямой решатель
                    delta = torch.linalg.solve(jacobian, -residual)
            except RuntimeError as e:
                print(f"  Ошибка решения системы: {e}")
                # Восстанавливаем исходное состояние
                self.fluid.pressure = current_p.clone()
                self.fluid.s_w = current_sw.clone()
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
            
            # Проверка на стагнацию с более гибкими условиями
            residual_improvement = prev_residual_norm / (residual_norm + 1e-15)
            if iter_idx > 3:
                if residual_improvement < 1.05:
                    print(f"  Сходимость замедлилась (улучшение только в {residual_improvement:.2f} раз)")
                    if residual_norm < 20 * tol:
                        print(f"  Принимаем результат, так как невязка близка к допустимой")
                        return True, iter_idx + 1
                elif residual_norm < 5 * tol:
                    print(f"  Невязка достаточно мала для принятия результата")
                    return True, iter_idx + 1
            
            prev_residual_norm = residual_norm
            
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
                
                # Быстрый расчет невязки без сборки полного якобиана
                new_residual = self._compute_residual_fast(dt, nx, ny, nz, dx, dy, dz)
                new_residual_norm = torch.norm(new_residual).item()
                
                # Откатываем изменения
                self.fluid.pressure = current_p.clone()
                self.fluid.s_w = current_sw.clone()
                
                # Проверяем, улучшает ли этот фактор сходимость
                if new_residual_norm < best_residual_norm:
                    best_residual_norm = new_residual_norm
                    best_factor = factor
                    
                    # Если улучшение значительное, прекращаем поиск
                    if new_residual_norm < 0.7 * residual_norm:
                        break
            
            # Улучшенная обработка случая, когда line search не помог
            if best_factor is None or best_residual_norm >= residual_norm:
                # Используем самый маленький фактор для предотвращения дивергенции
                best_factor = min(line_search_factors)
                print(f"  Внимание: Line search не смог уменьшить невязку. Используем минимальный шаг {best_factor}.")
                
                # Если невязка достаточно мала или это одна из начальных итераций, продолжаем
                if residual_norm < 15 * tol or iter_idx < 3:
                    print(f"  Продолжаем итерации с минимальным шагом")
                else:
                    # Проверяем, была ли сходимость на предыдущих итерациях
                    stagnation_count = getattr(self, '_stagnation_count', 0) + 1
                    setattr(self, '_stagnation_count', stagnation_count)
                    
                    if stagnation_count > 2:
                        print(f"  Невязка слишком велика, итерации Ньютона не сходятся после нескольких попыток")
                        # Восстанавливаем исходное состояние
                        self.fluid.pressure = current_p.clone()
                        self.fluid.s_w = current_sw.clone()
                        setattr(self, '_stagnation_count', 0)
                        return False, iter_idx + 1
                    else:
                        print(f"  Попытка продолжить с минимальным шагом (попытка {stagnation_count})")
            else:
                # Сбрасываем счетчик стагнаций при успешном шаге
                setattr(self, '_stagnation_count', 0)
            
            # Применяем найденный оптимальный шаг
            self._apply_newton_step(delta, best_factor)
            
            # Ограничиваем значения физическими пределами
            self.fluid.s_w.clamp_(self.fluid.sw_cr, 1.0 - self.fluid.so_r)
            self.fluid.pressure.clamp_(1e5, 100e6)  # От 0.1 МПа до 100 МПа
            self.fluid.s_o = 1.0 - self.fluid.s_w  # Обновляем нефтенасыщенность
            
            # Вычисляем время, затраченное на итерацию
            iter_time = time.time() - start_time
            if iter_time > 1.0:  # Если итерация заняла больше 1 секунды
                print(f"  Время итерации: {iter_time:.2f} сек.")
        
        # Если достигнуто максимальное число итераций
        print(f"  Метод Ньютона не сошелся за {max_iter} итераций")
        if residual_norm < 20 * tol:
            print(f"  Невязка достаточно близка к допустимой, принимаем результат")
            return True, max_iter
        else:
            # Восстанавливаем исходное состояние
            self.fluid.pressure = current_p.clone()
            self.fluid.s_w = current_sw.clone()
            return False, max_iter

    def _compute_residual_fast(self, dt, nx, ny, nz, dx, dy, dz):
        """
        Быстрый расчет невязки без сборки полного якобиана для line search.
        
        Args:
            dt: Временной шаг в секундах
            nx, ny, nz: Размеры сетки
            dx, dy, dz: Размеры ячеек
            
        Returns:
            Вектор невязки
        """
        num_cells = nx * ny * nz
        device = self.fluid.device
        
        # Создаем только вектор невязки
        residual = torch.zeros(2 * num_cells, device=device)
        
        # Базовые величины
        p_vec = self.fluid.pressure.reshape(-1)
        sw_vec = self.fluid.s_w.reshape(-1)
        phi_vec = self.reservoir.porosity.reshape(-1)
        
        # Плотности
        rho_w = self.fluid.calc_water_density(p_vec)
        rho_o = self.fluid.calc_oil_density(p_vec)
        
        # Капиллярное давление (если используется)
        if self.fluid.pc_scale > 0:
            pc = self.fluid.calc_capillary_pressure(sw_vec)
        else:
            pc = torch.zeros_like(p_vec)
        
        # Вязкости
        mu_w = self.fluid.mu_water * torch.ones_like(p_vec)
        mu_o = self.fluid.mu_oil * torch.ones_like(p_vec)
        
        # Относительные проницаемости
        kr_w = self.fluid.calc_water_kr(sw_vec)
        kr_o = self.fluid.calc_oil_kr(sw_vec)
        
        # Мобильности
        lambda_w = kr_w / mu_w
        lambda_o = kr_o / mu_o
        
        # Объем ячейки
        cell_volume = dx * dy * dz
        
        # Расчет невязки для аккумуляции
        water_mass = phi_vec * sw_vec * rho_w * cell_volume
        oil_mass = phi_vec * (1 - sw_vec) * rho_o * cell_volume
        
        for idx in range(num_cells):
            residual[2*idx] = water_mass[idx] - self.fluid.prev_water_mass[idx]
            residual[2*idx+1] = oil_mass[idx] - self.fluid.prev_oil_mass[idx]
        
        # Проводимости
        tx_const = dt * dy * dz / dx
        ty_const = dt * dx * dz / dy
        tz_const = dt * dx * dy / dz
        
        # Проницаемости
        perm_x_vec = self.reservoir.permeability_x.reshape(-1)
        perm_y_vec = self.reservoir.permeability_y.reshape(-1)
        perm_z_vec = self.reservoir.permeability_z.reshape(-1)
        
        # Векторизованный расчет потоков для X-направления
        for i in range(nx-1):
            for j in range(ny):
                for k in range(nz):
                    idx1 = i + j * nx + k * nx * ny
                    idx2 = (i+1) + j * nx + k * nx * ny
                    
                    # Средние значения
                    avg_perm = 2 * perm_x_vec[idx1] * perm_x_vec[idx2] / (perm_x_vec[idx1] + perm_x_vec[idx2] + 1e-10)
                    avg_rho_w = 0.5 * (rho_w[idx1] + rho_w[idx2])
                    avg_rho_o = 0.5 * (rho_o[idx1] + rho_o[idx2])
                    
                    # Градиенты давления
                    dp = p_vec[idx2] - p_vec[idx1]
                    dp_cap = pc[idx2] - pc[idx1] if self.fluid.pc_scale > 0 else 0.0
                    
                    # Гравитационный член (нет для X)
                    gravity_term_w = 0.0
                    gravity_term_o = 0.0
                    
                    # Восходящие мобильности
                    lambda_w_up = lambda_w[idx1] if dp >= 0 else lambda_w[idx2]
                    lambda_o_up = lambda_o[idx1] if dp >= 0 else lambda_o[idx2]
                    
                    # Проводимость
                    trans = tx_const * avg_perm
                    
                    # Потоки
                    water_flux = trans * lambda_w_up * (dp - dp_cap + gravity_term_w)
                    oil_flux = trans * lambda_o_up * (dp + gravity_term_o)
                    
                    # Невязка
                    residual[2*idx1] -= water_flux
                    residual[2*idx1+1] -= oil_flux
                    residual[2*idx2] += water_flux
                    residual[2*idx2+1] += oil_flux
        
        # Аналогично для Y-направления
        for j in range(ny-1):
            for i in range(nx):
                for k in range(nz):
                    idx1 = i + j * nx + k * nx * ny
                    idx2 = i + (j+1) * nx + k * nx * ny
                    
                    avg_perm = 2 * perm_y_vec[idx1] * perm_y_vec[idx2] / (perm_y_vec[idx1] + perm_y_vec[idx2] + 1e-10)
                    avg_rho_w = 0.5 * (rho_w[idx1] + rho_w[idx2])
                    avg_rho_o = 0.5 * (rho_o[idx1] + rho_o[idx2])
                    
                    dp = p_vec[idx2] - p_vec[idx1]
                    dp_cap = pc[idx2] - pc[idx1] if self.fluid.pc_scale > 0 else 0.0
                    
                    # Гравитационный член (нет для Y)
                    gravity_term_w = 0.0
                    gravity_term_o = 0.0
                    
                    lambda_w_up = lambda_w[idx1] if dp >= 0 else lambda_w[idx2]
                    lambda_o_up = lambda_o[idx1] if dp >= 0 else lambda_o[idx2]
                    
                    trans = ty_const * avg_perm
                    
                    water_flux = trans * lambda_w_up * (dp - dp_cap + gravity_term_w)
                    oil_flux = trans * lambda_o_up * (dp + gravity_term_o)
                    
                    residual[2*idx1] -= water_flux
                    residual[2*idx1+1] -= oil_flux
                    residual[2*idx2] += water_flux
                    residual[2*idx2+1] += oil_flux
        
        # Z-направление (с гравитацией)
        gravity = torch.tensor([0.0, 0.0, -9.81], device=perm_z_vec.device)
        for k in range(nz-1):
            for i in range(nx):
                for j in range(ny):
                    idx1 = i + j * nx + k * nx * ny
                    idx2 = i + j * nx + (k+1) * nx * ny
                    
                    avg_perm = 2 * perm_z_vec[idx1] * perm_z_vec[idx2] / (perm_z_vec[idx1] + perm_z_vec[idx2] + 1e-10)
                    avg_rho_w = 0.5 * (rho_w[idx1] + rho_w[idx2])
                    avg_rho_o = 0.5 * (rho_o[idx1] + rho_o[idx2])
                    
                    dp = p_vec[idx2] - p_vec[idx1]
                    dp_cap = pc[idx2] - pc[idx1] if self.fluid.pc_scale > 0 else 0.0
                    
                    # Гравитационные члены для Z-направления
                    gravity_term_w = avg_rho_w * gravity[2] * dz
                    gravity_term_o = avg_rho_o * gravity[2] * dz
                    
                    lambda_w_up = lambda_w[idx1] if dp >= 0 else lambda_w[idx2]
                    lambda_o_up = lambda_o[idx1] if dp >= 0 else lambda_o[idx2]
                    
                    trans = tz_const * avg_perm
                    
                    water_flux = trans * lambda_w_up * (dp - dp_cap + gravity_term_w)
                    oil_flux = trans * lambda_o_up * (dp + gravity_term_o)
                    
                    residual[2*idx1] -= water_flux
                    residual[2*idx1+1] -= oil_flux
                    residual[2*idx2] += water_flux
                    residual[2*idx2+1] += oil_flux
        
        # Учитываем скважины
        self._add_wells_to_residual_fast(residual, dt)
        
        return residual

    def _add_wells_to_residual_fast(self, residual, dt):
        """
        Быстрое добавление вклада скважин только в невязку.
        
        Args:
            residual: Вектор невязки
            dt: Временной шаг в секундах
        """
        wells = self.well_manager.get_wells()
        
        for well in wells:
            idx = well.cell_index_flat
            p = self.fluid.pressure.reshape(-1)[idx]
            sw = self.fluid.s_w.reshape(-1)[idx]
            
            # Вычисляем подвижности
            mu_w = self.fluid.mu_water
            mu_o = self.fluid.mu_oil
            kr_w = self.fluid.calc_water_kr(sw)
            kr_o = self.fluid.calc_oil_kr(sw)
            lambda_w = kr_w / mu_w
            lambda_o = kr_o / mu_o
            lambda_t = lambda_w + lambda_o
            
            if well.control_type == 'rate':
                # Скважина с контролем по дебиту
                rate = well.control_value / 86400.0  # м³/с
                
                if well.type == 'injector':
                    # Нагнетательная скважина (вода)
                    q_w = rate
                    q_o = 0.0
                    
                    # Обновляем невязку
                    residual[2*idx] -= q_w
                    residual[2*idx+1] -= q_o
                    
                else:  # producer
                    # Добывающая скважина
                    fw = lambda_w / (lambda_t + 1e-10)
                    fo = lambda_o / (lambda_t + 1e-10)
                    
                    q_w = rate * fw
                    q_o = rate * fo
                    
                    # Обновляем невязку
                    residual[2*idx] -= q_w
                    residual[2*idx+1] -= q_o
                    
            elif well.control_type == 'bhp':
                # Скважина с контролем забойного давления
                bhp = well.control_value * 1e6  # МПа -> Па
                
                # Дебиты
                q_w = well.well_index * lambda_w * (p - bhp)
                q_o = well.well_index * lambda_o * (p - bhp)
                
                # Обновляем невязку
                residual[2*idx] -= q_w
                residual[2*idx+1] -= q_o

    def _apply_newton_step(self, delta, factor):
        """
        Применяет шаг метода Ньютона с заданным фактором и строгими ограничениями на изменения.
        
        Args:
            delta: Вектор приращений решения
            factor: Коэффициент для шага
        """
        nx, ny, nz = self.reservoir.dimensions
        num_cells = nx * ny * nz
        
        # Делаем копию параметров для сравнения
        old_p = self.fluid.pressure.clone().reshape(-1)
        old_sw = self.fluid.s_w.clone().reshape(-1)
        
        # Применяем приращения с заданным фактором
        p_delta_raw = delta[:num_cells].reshape(-1) * factor
        sw_delta_raw = delta[num_cells:].reshape(-1) * factor
        
        # Ограничиваем изменения давления (не более 10% от текущего значения и не более 2 МПа)
        max_p_change_rel = 0.1 * torch.abs(old_p)
        max_p_change_abs = 2e6 * torch.ones_like(old_p)  # 2 МПа
        max_p_change = torch.minimum(max_p_change_rel, max_p_change_abs)
        p_delta = torch.clamp(p_delta_raw, -max_p_change, max_p_change)
        
        # Ограничиваем изменения насыщенности (не более 0.1 за шаг)
        max_sw_change = 0.1
        sw_delta = torch.clamp(sw_delta_raw, -max_sw_change, max_sw_change)
        
        # Применяем обновления к давлению и насыщенности
        self.fluid.pressure = (old_p + p_delta).reshape(nx, ny, nz)
        self.fluid.s_w = (old_sw + sw_delta).reshape(nx, ny, nz)
        
        # Ограничиваем физическими пределами
        self.fluid.pressure.clamp_(1e5, 100e6)  # От 0.1 МПа до 100 МПа
        self.fluid.s_w.clamp_(self.fluid.sw_cr, 1.0 - self.fluid.so_r)
        
        # Обновляем также нефтенасыщенность
        self.fluid.s_o = 1.0 - self.fluid.s_w
        
        # Подсчитываем количество ограниченных значений
        p_limited = torch.sum(p_delta != p_delta_raw).item()
        sw_limited = torch.sum(sw_delta != sw_delta_raw).item()
        
        # Выводим информацию о больших изменениях для отладки
        max_p_change = torch.max(torch.abs(p_delta)).item()
        max_sw_change = torch.max(torch.abs(sw_delta)).item()
        if max_p_change > 1e6 or max_sw_change > 0.1 or p_limited > 0 or sw_limited > 0:
            p_limited_percent = p_limited / num_cells * 100
            sw_limited_percent = sw_limited / num_cells * 100
            print(f"  Изменения: P_max={max_p_change/1e6:.3f} МПа, Sw_max={max_sw_change:.3f}. Ограничено: P={p_limited_percent:.1f}%, Sw={sw_limited_percent:.1f}%")

    def _idx_to_ijk(self, idx, nx, ny):
        """
        Преобразует линейный индекс в трехмерные индексы (i,j,k).
        
        Args:
            idx: Линейный индекс
            nx, ny: Размеры сетки по x и y
            
        Returns:
            Кортеж (i, j, k) - индексы в трехмерной сетке
        """
        # Предполагаем тот же порядок, что используется PyTorch при flatten():
        # idx = i * (ny * nz) + j * nz + k, где z-координата самая «быстрая».
        ny_nz = ny * self.reservoir.nz
        i = idx // ny_nz
        remainder = idx % ny_nz
        j = remainder // self.reservoir.nz
        k = remainder % self.reservoir.nz
        return i, j, k

    def _ijk_to_idx(self, i, j, k, nx, ny):
        """
        Преобразует трехмерные индексы (i,j,k) в линейный индекс.
        
        Args:
            i, j, k: Индексы в трехмерной сетке
            nx, ny: Размеры сетки по x и y
            
        Returns:
            Линейный индекс
        """
        # Используем тот же порядок, что и при flatten(): z – самая быстрая координата
        return (i * ny + j) * self.reservoir.nz + k

    def _compute_pressure_cell_centers(self) -> torch.Tensor:
        """
        Вычисляет координаты центров ячеек (нормированные на [0,1]) в порядке flatten().
        """
        nx, ny, nz = self.reservoir.dimensions
        x_centers = self.reservoir.x_centers.detach().cpu().double()
        y_centers = self.reservoir.y_centers.detach().cpu().double()
        z_centers = self.reservoir.z_centers.detach().cpu().double()

        lengths = self.reservoir.domain_lengths.detach().cpu().double()
        lx = max(lengths[0].item(), 1e-12)
        ly = max(lengths[1].item(), 1e-12)
        lz = max(lengths[2].item(), 1e-12)

        x_coords = x_centers / lx
        y_coords = y_centers / ly
        z_coords = z_centers / lz

        grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        coords = torch.stack(grid, dim=-1).reshape(-1, 3)
        return coords

    def _build_pressure_near_nullspace(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Формирует near-nullspace (константа + линейные координаты) для AMG.
        """
        n = coords.size(0)
        columns = [torch.ones(n, 1, dtype=torch.float64)]
        for dim in range(coords.size(1)):
            column = coords[:, dim:dim+1]
            if (column.max() - column.min()) > 1e-8:
                centered = column - column.mean()
                norm = centered.norm()
                if norm > 1e-10:
                    columns.append(centered / norm)
        if len(columns) == 1:
            columns.append(torch.arange(n, dtype=torch.float64).unsqueeze(1) / max(n - 1, 1))
        return torch.cat(columns, dim=1)

    # ==================================================================
    # ==                        СХЕМА IMPES                         ==
    # ==================================================================
    
    def _impes_step(self, dt):
        """ Выполняет один временной шаг с использованием схемы IMPES с адаптивным dt. """
        original_dt = dt
        current_dt = dt
        max_attempts = self.sim_params.get("max_time_step_attempts", 5)
        dt_reduction_factor = self.sim_params.get("dt_reduction_factor", 2.0)
        dt_increase_factor = self.sim_params.get("dt_increase_factor", 1.25)
        cfl_safety_factor = float(self.sim_params.get("cfl_safety_factor", 0.9))
        cfl_safety_factor = min(max(cfl_safety_factor, 0.1), 0.99)
        cfl_retry_cap = float(self.sim_params.get("cfl_retry_cap", 50.0))
        cfl_retry_cap = max(cfl_retry_cap, 1.0)
        mass_tol_rel = float(self.sim_params.get("mass_balance_tolerance", 1e-3))
        mass_tol_abs = float(self.sim_params.get("mass_balance_tolerance_abs", 1e-2))

        consecutive_success = 0
        last_dt_increased = False

        for attempt in range(max_attempts):
            print(f"Попытка шага IMPES с dt = {current_dt/86400:.2f} дней (Попытка {attempt+1}/{max_attempts})")

            P_new, converged = self._impes_pressure_step(current_dt)

            if converged:
                # Обновляем давление и выполняем шаг насыщенности
                prev_pressure_state = self.fluid.pressure.clone()
                prev_sw_state = self.fluid.s_w.clone()
                prev_sg_state = self.fluid.s_g.clone()
                prev_so_state = self.fluid.s_o.clone()
                prev_mass_balance = copy.deepcopy(self.mass_balance)
                prev_component_balance = copy.deepcopy(self.component_balance)

                self.fluid.pressure = P_new
                sat_info = self._impes_saturation_step(P_new, current_dt)

                for msg in sat_info.get("log_messages", []):
                    print(msg)

                rat_max = sat_info.get("rat_max", 1.0)
                max_substeps = sat_info.get("max_substeps", 20)
                if rat_max > max_substeps:
                    print(f"  ⚠ Насыщенность потребовала > max_substeps ({rat_max:.2f} > {sat_info.get('max_substeps', 20)}).")

                # CFL адаптивный шаг - опционально, по умолчанию выключен для обратной совместимости
                use_cfl_adaptive = self.sim_params.get("use_cfl_adaptive", False)
                if use_cfl_adaptive:
                    recommended_dt = sat_info.get("recommended_dt", current_dt)
                    if recommended_dt <= 0:
                        recommended_dt = current_dt / dt_reduction_factor
                    need_retry = (
                        rat_max > max_substeps
                        and rat_max < max_substeps * cfl_retry_cap
                        and recommended_dt < current_dt * cfl_safety_factor
                    )
                    if need_retry:
                        print(f"  ⚠ CFL: рекомендуемый шаг {recommended_dt/86400:.3f} дн., текущий {current_dt/86400:.3f} дн. Повторяем шаг с меньшим dt.")
                        self.fluid.pressure = prev_pressure_state
                        self.fluid.s_w = prev_sw_state
                        self.fluid.s_g = prev_sg_state
                        self.fluid.s_o = prev_so_state
                        self.mass_balance = prev_mass_balance
                        self.component_balance = prev_component_balance
                        current_dt = max(recommended_dt, current_dt / dt_reduction_factor)
                        consecutive_success = 0
                        last_dt_increased = False
                        continue

                # Проверяем насыщенности на клампы
                clamp_counts = sat_info.get("clamp_counts", {})
                if clamp_counts.get("sw_low", 0) or clamp_counts.get("sw_high", 0) or clamp_counts.get("sg_low", 0) or clamp_counts.get("sg_high", 0):
                    print(f"  ℹ Клампы насыщенности: Sw[{clamp_counts.get('sw_low',0)} низ / {clamp_counts.get('sw_high',0)} верх], "
                          f"Sg[{clamp_counts.get('sg_low',0)} низ / {clamp_counts.get('sg_high',0)} верх]")

                # Анализ баланса масс
                mass_report = sat_info.get("mass_balance", {})
                for phase, info in mass_report.items():
                    imbalance = abs(info.get("imbalance", 0.0))
                    scale = abs(info.get("in", 0.0)) + abs(info.get("out", 0.0)) + abs(info.get("accum", 0.0))
                    threshold = max(mass_tol_abs, mass_tol_rel * max(scale, 1.0))
                    if imbalance > threshold:
                        print(f"  ⚠ Массовый баланс ({phase}): in={info.get('in',0):.4e}, out={info.get('out',0):.4e}, "
                              f"accum={info.get('accum',0):.4e}, imbalance={info.get('imbalance',0):.4e} > {threshold:.2e}")

                # Сохраняем предыдущие состояния для следующего шага
                self.fluid.prev_pressure = self.fluid.pressure.clone()
                self.fluid.prev_sw = self.fluid.s_w.clone()

                consecutive_success += 1

                # Попробуем увеличить dt, если успешно несколько раз подряд
                if consecutive_success >= 2 and current_dt < original_dt and not last_dt_increased:
                    current_dt = min(current_dt * dt_increase_factor, original_dt)
                    last_dt_increased = True
                else:
                    last_dt_increased = False

                return True

            # если не сошлось, уменьшаем шаг
            print("  IMPES не сошелся, уменьшаем dt")
            current_dt /= dt_reduction_factor
            consecutive_success = 0
            last_dt_increased = False

        print("IMPES не сошелся даже с минимальным dt, прекращаем симуляцию.")
        return False

    def _impes_pressure_step(self, dt):
        """ Неявный шаг для расчета давления в схеме IMPES. """
        # 1. Подготовка коэффициентов
        P_prev = self.fluid.pressure
        S_w = self.fluid.s_w

        kro, krw, krg = self.fluid.get_rel_perms(S_w)
        # Вязкости фаз из PVT (если доступны) при P_prev
        mu_o_pas = self.fluid._eval_pvt(P_prev, 'mu_o') * 1e-3 if self.fluid.pvt else self.fluid.mu_oil
        mu_w_pas = self.fluid._eval_pvt(P_prev, 'mu_w') * 1e-3 if self.fluid.pvt else self.fluid.mu_water
        mu_g_pas = self.fluid._eval_pvt(P_prev, 'mu_g') * 1e-3 if self.fluid.pvt else self.fluid.mu_gas

        mob_w = krw / mu_w_pas
        mob_o = kro / mu_o_pas
        mob_g = krg / mu_g_pas
        mob_t = mob_w + mob_o + mob_g

        # 2. Трансмиссивности с учётом апстрима
        dp_x_prev = P_prev[:-1,:,:] - P_prev[1:,:,:]
        dp_y_prev = P_prev[:,:-1,:] - P_prev[:,1:,:]
        dp_z_prev = P_prev[:,:,:-1] - P_prev[:,:,1:]

        mob_t_x = torch.where(dp_x_prev > 0, mob_t[:-1,:,:], mob_t[1:,:,:])
        mob_t_y = torch.where(dp_y_prev > 0, mob_t[:,:-1,:], mob_t[:,1:,:])
        mob_t_z = torch.where(dp_z_prev > 0, mob_t[:,:,:-1], mob_t[:,:,1:])

        Tx_t = self.T_x * mob_t_x
        Ty_t = self.T_y * mob_t_y
        Tz_t = self.T_z * mob_t_z

        # 3. Скважины
        q_wells, well_bhp_terms = self._calculate_well_terms(mob_t, P_prev)

        # 4. Сборка матрицы и RHS
        A, A_diag = self._build_pressure_matrix_vectorized(Tx_t, Ty_t, Tz_t, dt, well_bhp_terms)
        Q = self._build_pressure_rhs(dt, P_prev, mob_w, mob_o, mob_g, q_wells, dp_x_prev, dp_y_prev, dp_z_prev)

        # 5. Выбор решателя давления: CG (по умолчанию) или AMG
        pressure_solver = self.sim_params.get("pressure_solver", "cg")
        if pressure_solver == "amg":
            # Параметры AMG
            amg_cfg = self.sim_params.get("amg", {}) or {}
            amg_tol = float(amg_cfg.get("tol", self.sim_params.get("cg_tolerance", 1e-6)))
            amg_max_cycles = int(amg_cfg.get("max_cycles", 20))
            amg_theta = float(amg_cfg.get("theta", 0.25))
            amg_max_levels = int(amg_cfg.get("max_levels", 10))
            amg_coarsest = int(amg_cfg.get("coarsest_size", 200))
            amg_device = amg_cfg.get("device", "auto")
            amg_mixed_precision = bool(amg_cfg.get("mixed_precision", False))
            amg_mixed_start = int(amg_cfg.get("mixed_start_level", 2))
            amg_cpu_offload = bool(amg_cfg.get("cpu_offload", False))
            amg_offload_level = max(0, int(amg_cfg.get("offload_level", max(3, amg_mixed_start))))
            # Импорт локально, чтобы не тянуть зависимости при CG
            from solver.pressure_amg import amg_solve
            # Решаем напрямую AMG V-cycles (внутри float64 и auto device)
            P_new_flat = amg_solve(
                A, Q,
                tol=amg_tol,
                max_cycles=amg_max_cycles,
                theta=amg_theta,
                max_levels=amg_max_levels,
                coarsest_size=amg_coarsest,
                device=amg_device,
                near_nullspace=self._pressure_near_nullspace,
                node_coords=self._pressure_cell_centers,
                mixed_precision=amg_mixed_precision,
                mixed_start_level=amg_mixed_start,
                cpu_offload=amg_cpu_offload,
                offload_level=amg_offload_level,
            )
            converged = True
            P_new = P_new_flat.view(self.reservoir.dimensions)
            return P_new, converged

        # 5. Параметры CG из конфигурации (по умолчанию)
        cg_tol_base = self.sim_params.get("cg_tolerance", 1e-6)
        cg_max_iter_base = self.sim_params.get("cg_max_iter", 500)

        # 6. Первая попытка решения CG
        if bool(self.sim_params.get('pressure_float64', False)):
            A64 = torch.sparse_coo_tensor(A.indices(), A.values().double(), A.size(), device=A.device).coalesce()
            Q64 = Q.double()
            M64 = A_diag.double() if A_diag is not None else None
            x64, converged = self._solve_pressure_cg_pytorch(A64, Q64, M_diag=M64, tol=cg_tol_base, max_iter=cg_max_iter_base)
            P_new_flat = x64.float()
        else:
            P_new_flat, converged = self._solve_pressure_cg_pytorch(A, Q, M_diag=A_diag, tol=cg_tol_base, max_iter=cg_max_iter_base)

        # 7. При неуспехе пробуем ещё раз с расслабленными параметрами
        if not converged:
            print("  CG не сошёлся: увеличиваем max_iter и ослабляем tol")
            if bool(self.sim_params.get('pressure_float64', False)):
                A64 = torch.sparse_coo_tensor(A.indices(), A.values().double(), A.size(), device=A.device).coalesce()
                Q64 = Q.double()
                M64 = A_diag.double() if A_diag is not None else None
                x64, converged = self._solve_pressure_cg_pytorch(
                    A64, Q64, M_diag=M64,
                    tol=cg_tol_base * 10.0,
                    max_iter=cg_max_iter_base * 4
                )
                P_new_flat = x64.float()
            else:
                P_new_flat, converged = self._solve_pressure_cg_pytorch(
                    A, Q, M_diag=A_diag,
                    tol=cg_tol_base * 10.0,
                    max_iter=cg_max_iter_base * 4
                )

        P_new = P_new_flat.view(self.reservoir.dimensions)
        return P_new, converged

    def _impes_saturation_step(self, P_new, dt):
        """ Явный шаг для обновления насыщенности в схеме IMPES. """
        S_w_old = self.fluid.s_w
        S_g_old = self.fluid.s_g
        log_messages: list[str] = []
        mass_before = copy.deepcopy(self.mass_balance)

        kro, krw, krg = self.fluid.get_rel_perms(S_w_old)
        # Вязкости фаз при новом давлении (для флюксов на шаге насыщенности)
        mu_o_pas = self.fluid._eval_pvt(P_new, 'mu_o') * 1e-3 if self.fluid.pvt else self.fluid.mu_oil
        mu_w_pas = self.fluid._eval_pvt(P_new, 'mu_w') * 1e-3 if self.fluid.pvt else self.fluid.mu_water
        mu_g_pas = self.fluid._eval_pvt(P_new, 'mu_g') * 1e-3 if self.fluid.pvt else self.fluid.mu_gas

        mob_w = krw / mu_w_pas
        mob_o = kro / mu_o_pas
        mob_g = krg / mu_g_pas
        mob_t = mob_w + mob_o + mob_g

        # 1. Градиенты давления и апстрим мобильностей
        dp_x = P_new[:-1,:,:] - P_new[1:,:,:]
        dp_y = P_new[:,:-1,:] - P_new[:,1:,:]
        dp_z = P_new[:,:,:-1] - P_new[:,:,1:]

        if self.use_capillary_potentials:
            pcow = self.fluid.get_capillary_pressure(S_w_old) if hasattr(self.fluid, 'get_capillary_pressure') else torch.zeros_like(S_w_old)
            dpcow_x = pcow[1:,:,:] - pcow[:-1,:,:]
            dpcow_y = pcow[:,1:,:] - pcow[:,:-1,:]
            dpcow_z = pcow[:,:,1:] - pcow[:,:,:-1]
            pot_w_x = dp_x - dpcow_x
            pot_w_y = dp_y - dpcow_y
            pcog = self.fluid.get_capillary_pressure_og(self.fluid.s_g) if hasattr(self.fluid, 'get_capillary_pressure_og') else torch.zeros_like(self.fluid.s_g)
            dpcog_x = pcog[1:,:,:] - pcog[:-1,:,:]
            dpcog_y = pcog[:,1:,:] - pcog[:,:-1,:]
            dpcog_z = pcog[:,:,1:] - pcog[:,:,:-1]
            pot_g_x = dp_x - dpcog_x
            pot_g_y = dp_y - dpcog_y

            if self.reservoir.nz > 1:
                rho_w_avg = 0.5 * (self.fluid.rho_w[:,:,:-1] + self.fluid.rho_w[:,:,1:])
                rho_g_avg = 0.5 * (self.fluid.rho_g[:,:,:-1] + self.fluid.rho_g[:,:,1:])
                dz_face = self.reservoir.dz_face.to(self.device, dtype=torch.float64).view(1, 1, -1)
                pot_w_z = dp_z - dpcow_z + self.g * rho_w_avg * dz_face
                pot_g_z = dp_z - dpcog_z + self.g * rho_g_avg * dz_face
            else:
                pot_w_z = dp_z - dpcow_z
                pot_g_z = dp_z - dpcog_z

            mob_w_x = torch.where(pot_w_x > 0, mob_w[:-1,:,:], mob_w[1:,:,:])
            mob_w_y = torch.where(pot_w_y > 0, mob_w[:,:-1,:], mob_w[:,1:,:])
            mob_w_z = torch.where(pot_w_z > 0, mob_w[:,:,:-1], mob_w[:,:,1:])

            mob_g_x = torch.where(pot_g_x > 0, mob_g[:-1,:,:], mob_g[1:,:,:])
            mob_g_y = torch.where(pot_g_y > 0, mob_g[:,:-1,:], mob_g[:,1:,:])
            mob_g_z = torch.where(pot_g_z > 0, mob_g[:,:,:-1], mob_g[:,:,1:])

            flow_w_x = self.T_x * mob_w_x * pot_w_x
            flow_w_y = self.T_y * mob_w_y * pot_w_y
            flow_w_z = self.T_z * mob_w_z * pot_w_z

            flow_g_x = self.T_x * mob_g_x * pot_g_x
            flow_g_y = self.T_y * mob_g_y * pot_g_y
            flow_g_z = self.T_z * mob_g_z * pot_g_z
        else:
            mob_w_x = torch.where(dp_x > 0, mob_w[:-1,:,:], mob_w[1:,:,:])
            mob_w_y = torch.where(dp_y > 0, mob_w[:,:-1,:], mob_w[:,1:,:])
            mob_w_z = torch.where(dp_z > 0, mob_w[:,:,:-1], mob_w[:,:,1:])

            mob_g_x = torch.where(dp_x > 0, mob_g[:-1,:,:], mob_g[1:,:,:])
            mob_g_y = torch.where(dp_y > 0, mob_g[:,:-1,:], mob_g[:,1:,:])
            mob_g_z = torch.where(dp_z > 0, mob_g[:,:,:-1], mob_g[:,:,1:])

            if self.reservoir.nz > 1:
                rho_w_avg = 0.5 * (self.fluid.rho_w[:,:,:-1] + self.fluid.rho_w[:,:,1:])
                rho_g_avg = 0.5 * (self.fluid.rho_g[:,:,:-1] + self.fluid.rho_g[:,:,1:])
                dz_face = self.reservoir.dz_face.to(self.device, dtype=torch.float64).view(1, 1, -1)
                pot_w_z = dp_z + self.g * rho_w_avg * dz_face
                pot_g_z = dp_z + self.g * rho_g_avg * dz_face
            else:
                pot_w_z = dp_z
                pot_g_z = dp_z

            flow_w_x = self.T_x * mob_w_x * dp_x
            flow_w_y = self.T_y * mob_w_y * dp_y
            flow_w_z = self.T_z * mob_w_z * pot_w_z

            if getattr(self.fluid, 'pc_og_scale', 0.0) > 0 or (self.fluid.relperm_model == 'table' and 'sgof' in getattr(self.fluid, 'relperm_tables', {})):
                pcog = self.fluid.get_capillary_pressure_og(self.fluid.s_g)
                dpcg_x = pcog[1:,:,:] - pcog[:-1,:,:]
                dpcg_y = pcog[:,1:,:] - pcog[:,:-1,:]
                dpcg_z = pcog[:,:,1:] - pcog[:,:,:-1]
                flow_g_x = self.T_x * mob_g_x * (dp_x - dpcg_x)
                flow_g_y = self.T_y * mob_g_y * (dp_y - dpcg_y)
                flow_g_z = self.T_z * mob_g_z * (pot_g_z - dpcg_z)
            else:
                flow_g_x = self.T_x * mob_g_x * dp_x
                flow_g_y = self.T_y * mob_g_y * dp_y
                flow_g_z = self.T_z * mob_g_z * pot_g_z

        # 4. Дивергенция
        div_w = torch.zeros_like(S_w_old)
        div_w[:-1, :, :] += flow_w_x
        div_w[1:, :, :]  -= flow_w_x
        div_w[:, :-1, :] += flow_w_y
        div_w[:, 1:, :]  -= flow_w_y
        div_w[:, :, :-1] += flow_w_z
        div_w[:, :, 1:]  -= flow_w_z

        div_g = torch.zeros_like(S_g_old)
        div_g[:-1, :, :] += flow_g_x
        div_g[1:, :, :]  -= flow_g_x
        div_g[:, :-1, :] += flow_g_y
        div_g[:, 1:, :]  -= flow_g_y
        div_g[:, :, :-1] += flow_g_z
        div_g[:, :, 1:]  -= flow_g_z

        # 5. Источники/стоки воды от скважин
        q_w = torch.zeros_like(S_w_old)
        q_g = torch.zeros_like(S_g_old)
        q_o = torch.zeros_like(S_w_old)
        fw = mob_w / (mob_t + 1e-10)
        fg = mob_g / (mob_t + 1e-10)
        fo = mob_o / (mob_t + 1e-10)
        for well in self.well_manager.get_wells():
            i, j, k = well.i, well.j, well.k
            if i >= self.reservoir.nx or j >= self.reservoir.ny or k >= self.reservoir.nz:
                continue

            if well.control_type == 'rate':
                # Базовый дебит в м3/с
                rate_scale = float(self.sim_params.get('global_rate_scale', 1.0))
                q_base = (well.control_value * rate_scale) / 86400.0
                # Конверсия surface->reservoir для инжектора
                if well.type == 'injector' and getattr(well, 'rate_type', 'reservoir') == 'surface' and self.fluid.pvt is not None:
                    if well.injected_phase == 'gas':
                        Bg_cell = float(self.fluid._eval_pvt(P_new, 'Bg')[i, j, k])
                        q_total = q_base * Bg_cell
                    else:
                        Bw_cell = float(self.fluid._eval_pvt(P_new, 'Bw')[i, j, k])
                        q_total = q_base * Bw_cell
                else:
                    q_total = q_base
                q_total *= (1 if well.type == 'injector' else -1)
                if well.type == 'injector':
                    injected_phase = getattr(well, 'injected_phase', 'water')
                    if injected_phase == 'gas':
                        # Availability-choke для газа
                        PV_cell = float(self.porous_volume[i, j, k])
                        sg = float(S_g_old[i, j, k]); sw = float(S_w_old[i, j, k])
                        so_r = float(self.fluid.so_r); sg_min = float(self.fluid.sg_cr)
                        sg_up = max(0.0, (1.0 - so_r - sw) - sg)
                        dSg_well = (dt / max(PV_cell, 1e-12)) * q_total
                        alpha = 1.0
                        if dSg_well > 0.0:
                            alpha = min(alpha, sg_up / max(dSg_well, 1e-12))
                        else:
                            alpha = min(alpha, (sg - sg_min) / max(abs(dSg_well), 1e-12))
                        alpha = max(0.0, min(1.0, alpha))
                        if alpha < 0.999:
                            log_messages.append(
                                f"  ℹ Availability choke: {well.name} inj_gas alpha={alpha:.3f} ячейка ({i},{j},{k})"
                            )
                        q_total *= alpha
                        q_g[i, j, k] += q_total
                        # Компоненты (surface) — считаем по фактическому q_total после choke
                        if self.fluid.pvt is not None:
                            Bg_cell = float(self.fluid._eval_pvt(P_new, 'Bg')[i, j, k])
                            qg_surf = max(q_total, 0.0) / max(Bg_cell, 1e-12) * 86400.0
                            try:
                                Rv_cell = float(self.fluid._eval_pvt(P_new, 'Rv')[i, j, k])
                            except Exception:
                                Rv_cell = 0.0
                            scale = float(dt / 86400.0)
                            self.component_balance['gas']['in'] += float(qg_surf * scale)
                            self.component_balance['oil']['in'] += float(Rv_cell * qg_surf * scale)
                            self._dbg(
                                f"WELL_IN name={well.name} mode=inj_gas cell=({i},{j},{k}) q_total={q_total:.6e}",
                                f"Bg={Bg_cell:.4g} Rv={Rv_cell:.4g} qg_surf={qg_surf:.6e} oil_in+= {Rv_cell*qg_surf*scale:.6e} gas_in+= {qg_surf*scale:.6e}"
                            )
                    else:
                        # Availability-choke для воды
                        PV_cell = float(self.porous_volume[i, j, k])
                        sw = float(S_w_old[i, j, k]); sg = float(S_g_old[i, j, k])
                        sw_min = float(self.fluid.sw_cr); so_r = float(self.fluid.so_r)
                        sw_up = max(0.0, (1.0 - so_r - sg) - sw)
                        dSw_well = (dt / max(PV_cell, 1e-12)) * q_total
                        alpha = 1.0
                        if dSw_well > 0.0:
                            alpha = min(alpha, sw_up / max(dSw_well, 1e-12))
                        else:
                            alpha = min(alpha, (sw - sw_min) / max(abs(dSw_well), 1e-12))
                        alpha = max(0.0, min(1.0, alpha))
                        if alpha < 0.999:
                            log_messages.append(
                                f"  ℹ Availability choke: {well.name} inj_water alpha={alpha:.3f} ячейка ({i},{j},{k})"
                            )
                        q_total *= alpha
                    q_w[i, j, k] += q_total
                else:
                    # Producer
                    if getattr(well, 'rate_type', 'reservoir') == 'surface' and self.fluid.pvt is not None:
                        sp = (well.surface_phase or 'liquid').lower()
                        Bo_cell = float(self.fluid._eval_pvt(P_new, 'Bo')[i, j, k])
                        Bw_cell = float(self.fluid._eval_pvt(P_new, 'Bw')[i, j, k])
                        Bg_cell = float(self.fluid._eval_pvt(P_new, 'Bg')[i, j, k])
                        Rs_cell = float(self.fluid._eval_pvt(P_new, 'Rs')[i, j, k])
                        # Rv может отсутствовать — в этом случае считаем 0
                        try:
                            Rv_cell = float(self.fluid._eval_pvt(P_new, 'Rv')[i, j, k])
                        except Exception:
                            Rv_cell = 0.0
                        fo_ = fo[i, j, k].item()
                        fw_ = fw[i, j, k].item()
                        fg_ = fg[i, j, k].item()
                        if sp == 'oil':
                            # q_o_surf = q_total*(fo/Bo + fg*Rv/Bg)
                            denom = (fo_ / max(Bo_cell, 1e-12) + fg_ * Rv_cell / max(Bg_cell, 1e-12))
                            q_total = - (q_base / max(denom, 1e-12))
                        elif sp == 'water':
                            q_w_res = q_base * Bw_cell
                            q_total = - q_w_res / max(fw_, 1e-8)
                        elif sp == 'gas':
                            # q_g_surf = q_g_res_free/Bg + Rs * q_o_res/Bo_surf? В surface единицах: q_g_surf = q_total*(fg/Bg + fo*Rs/Bo)
                            denom = (fg_ / max(Bg_cell, 1e-12) + fo_ * Rs_cell / max(Bo_cell, 1e-12))
                            q_total = - (q_base / max(denom, 1e-12))
                        else:  # 'liquid' (oil+water)
                            # q_liq_surf = q_total*(fo/Bo + fg*Rv/Bg + fw/Bw)
                            denom = (fo_ / max(Bo_cell, 1e-12) + fg_ * Rv_cell / max(Bg_cell, 1e-12) + fw_ / max(Bw_cell, 1e-12))
                            q_total = - (q_base / max(denom, 1e-12))
                        # Availability-choke: масштабируем q_total по доступности фаз в ячейке (dS не выходит за пределы)
                        PV_cell = float(self.porous_volume[i, j, k])
                        fw_ = fw[i, j, k].item(); fg_ = fg[i, j, k].item()
                        sw = float(S_w_old[i, j, k]); sg = float(S_g_old[i, j, k])
                        sw_min = float(self.fluid.sw_cr); sg_min = float(self.fluid.sg_cr); so_r = float(self.fluid.so_r)
                        # доступные окна
                        sw_up = max(0.0, (1.0 - so_r - sg) - sw)
                        sw_dn = max(0.0, sw - sw_min)
                        sg_up = max(0.0, (1.0 - so_r - sw) - sg)
                        sg_dn = max(0.0, sg - sg_min)
                        # желаемые dS от скважины
                        dSw_well = (dt / max(PV_cell, 1e-12)) * (q_total * fw_)
                        dSg_well = (dt / max(PV_cell, 1e-12)) * (q_total * fg_)
                        alpha = 1.0
                        if dSw_well > 0.0:
                            alpha = min(alpha, sw_up / max(dSw_well, 1e-12))
                        else:
                            alpha = min(alpha, sw_dn / max(abs(dSw_well), 1e-12))
                        if dSg_well > 0.0:
                            alpha = min(alpha, sg_up / max(dSg_well, 1e-12))
                        else:
                            alpha = min(alpha, sg_dn / max(abs(dSg_well), 1e-12))
                        alpha = max(0.0, min(1.0, alpha))
                        if alpha < 0.999:
                            log_messages.append(
                                f"  ℹ Availability choke: {well.name} surface alpha={alpha:.3f} ячейка ({i},{j},{k})"
                            )
                        q_total *= alpha

                        # Фазовые лимиты в surface-единицах (м3/сут) — при необходимости троттлинг по |q_total|
                        lim = getattr(well, 'limits', None) or {}
                        if lim:
                            denom_o = fo_ / max(Bo_cell, 1e-12) + fg_ * Rv_cell / max(Bg_cell, 1e-12)
                            denom_w = fw_ / max(Bw_cell, 1e-12)
                            denom_g = fg_ / max(Bg_cell, 1e-12) + fo_ * Rs_cell / max(Bo_cell, 1e-12)
                            denom_l = denom_o + denom_w
                            candidates = []
                            if 'wopr' in lim and denom_o > 0:
                                candidates.append((lim['wopr'] / 86400.0) / denom_o)
                            if 'wlpr' in lim and denom_w > 0:
                                candidates.append((lim['wlpr'] / 86400.0) / denom_w)
                            if 'wgpr' in lim and denom_g > 0:
                                candidates.append((lim['wgpr'] / 86400.0) / denom_g)
                            if 'liqr' in lim and denom_l > 0:
                                candidates.append((lim['liqr'] / 86400.0) / denom_l)
                            if candidates:
                                max_abs_q = min(candidates)
                                if abs(q_total) > max_abs_q:
                                    log_messages.append(
                                        f"  ⚠ Скважина {well.name} ограничена лимитом surface {max_abs_q*86400:.2f} м³/сут (фазы: {list(lim.keys())})"
                                    )
                                    q_total = -max_abs_q
                        # Диагностика per-well: сохранить surface-скорости (м3/сут)
                        oil_surf = abs(q_total) * (fo_ / max(Bo_cell, 1e-12) + fg_ * Rv_cell / max(Bg_cell, 1e-12)) * 86400.0
                        wat_surf = abs(q_total) * (fw_ / max(Bw_cell, 1e-12)) * 86400.0
                        gas_surf = abs(q_total) * (fg_ / max(Bg_cell, 1e-12) + fo_ * Rs_cell / max(Bo_cell, 1e-12)) * 86400.0
                        liq_surf = oil_surf + wat_surf
                        well.last_surface_rates = {"oil": oil_surf, "water": wat_surf, "gas": gas_surf, "liquid": liq_surf}
                        well.last_q_total = float(q_total)
                        # Компоненты OUT по фактическим reservoir-дебитам после choke
                        # Np = ∫( -qo_loc/Bo + Rv * (-qg_loc)/Bg ) dt; Gp = ∫( -qg_loc/Bg + Rs * (-qo_loc)/Bo ) dt
                        scale_t = float(dt)
                        qo_res = float((-q_total * fo_))  # м3/с (положительно на отбор)
                        qg_res = float((-q_total * fg_))  # м3/с
                        oil_delta = scale_t * (max(qo_res, 0.0) / max(Bo_cell, 1e-12) + Rv_cell * max(qg_res, 0.0) / max(Bg_cell, 1e-12))
                        gas_delta = scale_t * (max(qg_res, 0.0) / max(Bg_cell, 1e-12) + Rs_cell * max(qo_res, 0.0) / max(Bo_cell, 1e-12))
                        self.component_balance['oil']['out'] += oil_delta
                        self.component_balance['gas']['out'] += gas_delta
                        self._dbg(
                            f"WELL_OUT name={well.name} mode=rate cell=({i},{j},{k}) q_total={q_total:.6e} fw={fw_:.3f} fg={fg_:.3f} fo={fo_:.3f}",
                            f"Bo={Bo_cell:.4g} Bw={Bw_cell:.4g} Bg={Bg_cell:.4g} Rs={Rs_cell:.4g} Rv={Rv_cell:.4g}",
                            f"oil_out+= {oil_delta:.6e} gas_out+= {gas_delta:.6e}"
                        )
                        qw_loc = q_total * fw[i, j, k]
                        qg_loc = q_total * fg[i, j, k]
                        qo_loc = q_total * fo[i, j, k]
                    else:
                        # Резервуарный total rate
                        # Возможные surface-лимиты также применим, если есть PVT
                        if self.fluid.pvt is not None:
                            Bo_cell = float(self.fluid._eval_pvt(P_new, 'Bo')[i, j, k])
                            Bw_cell = float(self.fluid._eval_pvt(P_new, 'Bw')[i, j, k])
                            Bg_cell = float(self.fluid._eval_pvt(P_new, 'Bg')[i, j, k])
                            Rs_cell = float(self.fluid._eval_pvt(P_new, 'Rs')[i, j, k])
                            try:
                                Rv_cell = float(self.fluid._eval_pvt(P_new, 'Rv')[i, j, k])
                            except Exception:
                                Rv_cell = 0.0
                            fo_ = fo[i, j, k].item(); fw_ = fw[i, j, k].item(); fg_ = fg[i, j, k].item()
                            # Availability-choke
                            PV_cell = float(self.porous_volume[i, j, k])
                            sw = float(S_w_old[i, j, k]); sg = float(S_g_old[i, j, k])
                            sw_min = float(self.fluid.sw_cr); sg_min = float(self.fluid.sg_cr); so_r = float(self.fluid.so_r)
                            sw_up = max(0.0, (1.0 - so_r - sg) - sw)
                            sw_dn = max(0.0, sw - sw_min)
                            sg_up = max(0.0, (1.0 - so_r - sw) - sg)
                            sg_dn = max(0.0, sg - sg_min)
                            dSw_well = (dt / max(PV_cell, 1e-12)) * (q_total * fw_)
                            dSg_well = (dt / max(PV_cell, 1e-12)) * (q_total * fg_)
                            alpha = 1.0
                            if dSw_well > 0.0:
                                alpha = min(alpha, sw_up / max(dSw_well, 1e-12))
                            else:
                                alpha = min(alpha, sw_dn / max(abs(dSw_well), 1e-12))
                            if dSg_well > 0.0:
                                alpha = min(alpha, sg_up / max(dSg_well, 1e-12))
                            else:
                                alpha = min(alpha, sg_dn / max(abs(dSg_well), 1e-12))
                            alpha = max(0.0, min(1.0, alpha))
                            if alpha < 0.999:
                                log_messages.append(
                                    f"  ℹ Availability choke: {well.name} reservoir alpha={alpha:.3f} ячейка ({i},{j},{k})"
                                )
                            q_total *= alpha
                            lim = getattr(well, 'limits', None) or {}
                            if lim:
                                denom_o = fo_ / max(Bo_cell, 1e-12) + fg_ * Rv_cell / max(Bg_cell, 1e-12)
                                denom_w = fw_ / max(Bw_cell, 1e-12)
                                denom_g = fg_ / max(Bg_cell, 1e-12) + fo_ * Rs_cell / max(Bo_cell, 1e-12)
                                denom_l = denom_o + denom_w
                                candidates = []
                                if 'wopr' in lim and denom_o > 0:
                                    candidates.append((lim['wopr'] / 86400.0) / denom_o)
                                if 'wlpr' in lim and denom_w > 0:
                                    candidates.append((lim['wlpr'] / 86400.0) / denom_w)
                                if 'wgpr' in lim and denom_g > 0:
                                    candidates.append((lim['wgpr'] / 86400.0) / denom_g)
                                if 'liqr' in lim and denom_l > 0:
                                    candidates.append((lim['liqr'] / 86400.0) / denom_l)
                                if candidates:
                                    max_abs_q = min(candidates)
                                    if abs(q_total) > max_abs_q:
                                        log_messages.append(
                                            f"  ⚠ Скважина {well.name} ограничена лимитом reservoir {max_abs_q*86400:.2f} м³/сут (фазы: {list(lim.keys())})"
                                        )
                                        q_total = -max_abs_q
                            oil_surf = abs(q_total) * (fo_ / max(Bo_cell, 1e-12) + fg_ * Rv_cell / max(Bg_cell, 1e-12)) * 86400.0
                            wat_surf = abs(q_total) * (fw_ / max(Bw_cell, 1e-12)) * 86400.0
                            gas_surf = abs(q_total) * (fg_ / max(Bg_cell, 1e-12) + fo_ * Rs_cell / max(Bo_cell, 1e-12)) * 86400.0
                            liq_surf = oil_surf + wat_surf
                            well.last_surface_rates = {"oil": oil_surf, "water": wat_surf, "gas": gas_surf, "liquid": liq_surf}
                            well.last_q_total = float(q_total)
                            # Компоненты OUT по фактическим reservoir-дебитам после choke
                            scale_t = float(dt)
                            qo_res = float((-q_total * fo_))
                            qg_res = float((-q_total * fg_))
                            oil_delta = scale_t * (max(qo_res, 0.0) / max(Bo_cell, 1e-12) + Rv_cell * max(qg_res, 0.0) / max(Bg_cell, 1e-12))
                            gas_delta = scale_t * (max(qg_res, 0.0) / max(Bg_cell, 1e-12) + Rs_cell * max(qo_res, 0.0) / max(Bo_cell, 1e-12))
                            self.component_balance['oil']['out'] += oil_delta
                            self.component_balance['gas']['out'] += gas_delta
                            self._dbg(
                                f"WELL_OUT name={well.name} mode=rate_res cell=({i},{j},{k}) q_total={q_total:.6e} fw={fw_:.3f} fg={fg_:.3f} fo={fo_:.3f}",
                                f"Bo={Bo_cell:.4g} Bw={Bw_cell:.4g} Bg={Bg_cell:.4g} Rs={Rs_cell:.4g} Rv={Rv_cell:.4g}",
                                f"oil_out+= {oil_delta:.6e} gas_out+= {gas_delta:.6e}"
                            )
                        qw_loc = q_total * fw[i, j, k]
                        qg_loc = q_total * fg[i, j, k]
                        qo_loc = q_total * fo[i, j, k]
                    q_w[i, j, k] += qw_loc
                    q_g[i, j, k] += qg_loc
                    q_o[i, j, k] += qo_loc
            elif well.control_type == 'bhp':
                p_bhp = well.control_value * 1e6
                p_block = P_new[i, j, k]
                q_total = well.well_index * mob_t[i, j, k] * (p_block - p_bhp)
                if well.type == 'injector':
                    injected_phase = getattr(well, 'injected_phase', 'water')
                    if injected_phase == 'gas':
                        q_g[i, j, k] += (-q_total)
                        if self.fluid.pvt is not None:
                            Bg_cell = float(self.fluid._eval_pvt(P_new, 'Bg')[i, j, k])
                            qg_surf = max(-q_total, 0.0) / max(Bg_cell, 1e-12) * 86400.0
                            try:
                                Rv_cell = float(self.fluid._eval_pvt(P_new, 'Rv')[i, j, k])
                            except Exception:
                                Rv_cell = 0.0
                            scale = float(dt / 86400.0)
                            self.component_balance['gas']['in'] += float(qg_surf * scale)
                            self.component_balance['oil']['in'] += float(Rv_cell * qg_surf * scale)
                    else:
                        q_w[i, j, k] += (-q_total)
                else:
                    qw_loc = (-q_total) * fw[i, j, k]
                    qg_loc = (-q_total) * fg[i, j, k]
                    qo_loc = (-q_total) * fo[i, j, k]
                    q_w[i, j, k] += qw_loc
                    q_g[i, j, k] += qg_loc
                    q_o[i, j, k] += qo_loc
                    # Компонентный отбор при BHP-контроле (через surface-конверсию)
                    if self.fluid.pvt is not None:
                        Bo_cell = float(self.fluid._eval_pvt(P_new, 'Bo')[i, j, k])
                        Bw_cell = float(self.fluid._eval_pvt(P_new, 'Bw')[i, j, k])
                        Bg_cell = float(self.fluid._eval_pvt(P_new, 'Bg')[i, j, k])
                        Rs_cell = float(self.fluid._eval_pvt(P_new, 'Rs')[i, j, k])
                        try:
                            Rv_cell = float(self.fluid._eval_pvt(P_new, 'Rv')[i, j, k])
                        except Exception:
                            Rv_cell = 0.0
                        fo_ = fo[i, j, k].item(); fw_ = fw[i, j, k].item(); fg_ = fg[i, j, k].item()
                        qabs = abs(q_total)
                        oil_surf = qabs * (fo_ / max(Bo_cell, 1e-12) + fg_ * Rv_cell / max(Bg_cell, 1e-12)) * 86400.0
                        gas_surf = qabs * (fg_ / max(Bg_cell, 1e-12) + fo_ * Rs_cell / max(Bo_cell, 1e-12)) * 86400.0
                        # Компоненты OUT по фактическим reservoir-дебитам при BHP
                        scale_t = float(dt)
                        qo_res = float((qabs * fo_))
                        qg_res = float((qabs * fg_))
                        oil_delta = scale_t * (qo_res / max(Bo_cell, 1e-12) + Rv_cell * qg_res / max(Bg_cell, 1e-12))
                        gas_delta = scale_t * (qg_res / max(Bg_cell, 1e-12) + Rs_cell * qo_res / max(Bo_cell, 1e-12))
                        self.component_balance['oil']['out'] += oil_delta
                        self.component_balance['gas']['out'] += gas_delta
                        self._dbg(
                            f"WELL_OUT name={well.name} mode=bhp cell=({i},{j},{k}) q_total={q_total:.6e} fw={fw_:.3f} fg={fg_:.3f} fo={fo_:.3f}",
                            f"Bo={Bo_cell:.4g} Bw={Bw_cell:.4g} Bg={Bg_cell:.4g} Rs={Rs_cell:.4g} Rv={Rv_cell:.4g}",
                            f"oil_out+= {oil_delta:.6e} gas_out+= {gas_delta:.6e}"
                        )

        # Агрегированный баланс масс будет посчитан после применения dS и обновления насыщенностей ниже

        # 6. Обновление насыщенности: без пост-клампов, через адаптивные подшаги
        dSw = (dt / self.porous_volume) * (q_w - div_w)
        dSg = (dt / self.porous_volume) * (q_g - div_g)
        max_dS = self.sim_params.get("max_saturation_change", 0.05)

        S_w_start = S_w_old.clone()
        S_g_start = S_g_old.clone()

        # Рассчитываем необходимое число подшагов N для соблюдения окон насыщенности и max_dS
        so_r = float(self.fluid.so_r)
        sw_cr = float(self.fluid.sw_cr)
        sg_cr = float(self.fluid.sg_cr)
        eps = 1e-12
        sw_up = torch.clamp(1.0 - so_r - S_g_old - S_w_old, min=0.0)
        sw_dn = torch.clamp(S_w_old - sw_cr, min=0.0)
        sg_up = torch.clamp(1.0 - so_r - S_w_old - S_g_old, min=0.0)
        sg_dn = torch.clamp(S_g_old - sg_cr, min=0.0)
        rat_w = torch.where(dSw > 0, dSw / torch.clamp(sw_up, min=eps), (-dSw) / torch.clamp(sw_dn, min=eps))
        rat_g = torch.where(dSg > 0, dSg / torch.clamp(sg_up, min=eps), (-dSg) / torch.clamp(sg_dn, min=eps))
        rat_lim = torch.maximum(torch.abs(dSw)/max_dS, torch.abs(dSg)/max_dS)
        rat = torch.maximum(torch.maximum(rat_w, rat_g), rat_lim)
        rat_max = torch.nan_to_num(rat.max(), nan=0.0, posinf=0.0, neginf=0.0)
        N = int(torch.ceil(rat_max).item())
        if N < 1:
            N = 1
        # Ограничение на число подшагов для стабильного времени расчёта
        max_substeps = int(self.sim_params.get("max_substeps", 20))
        if N > max_substeps:
            log_messages.append(
                f"  ⚠ Требуемые подшаги ({N}) превышают max_substeps={max_substeps}, применяется ограничение."
            )
            N = max_substeps

        dSw_sub = dSw / N
        dSg_sub = dSg / N
        S_w_cur = S_w_old
        S_g_cur = S_g_old
        for _ in range(N):
            S_w_cur = S_w_cur + dSw_sub
            S_g_cur = S_g_cur + dSg_sub
            # мягкие клампы на случай численной погрешности (min/max как тензоры одной формы)
            sw_min_t = torch.full_like(S_w_cur, float(sw_cr))
            sw_max_t = (1.0 - so_r - S_g_cur)
            sw_max_t = torch.maximum(sw_max_t, sw_min_t)
            S_w_cur = torch.clamp(S_w_cur, min=sw_min_t, max=sw_max_t)
            sg_min_t = torch.full_like(S_g_cur, float(sg_cr))
            sg_max_t = (1.0 - so_r - S_w_cur)
            sg_max_t = torch.maximum(sg_max_t, sg_min_t)
            S_g_cur = torch.clamp(S_g_cur, min=sg_min_t, max=sg_max_t)

            # Подшаговая интеграция компонентного баланса с учётом текущих S
            try:
                if self.fluid.pvt is not None:
                    # Обновляем временно насыщенности для расчёта кривых
                    self.fluid.s_w = S_w_cur
                    self.fluid.s_g = S_g_cur
                    self.fluid.s_o = 1.0 - self.fluid.s_w - self.fluid.s_g
                    # Мобилизации и доли
                    mu_w = self.fluid._eval_pvt(P_new, 'mu_w')
                    mu_o = self.fluid._eval_pvt(P_new, 'mu_o')
                    mu_g = self.fluid._eval_pvt(P_new, 'mu_g')
                    krw = self.fluid.calc_water_kr(self.fluid.s_w)
                    kro = self.fluid.calc_oil_kr(self.fluid.s_w, self.fluid.s_g)
                    krg = self.fluid.calc_gas_kr(self.fluid.s_g)
                    mob_w_cur = krw / (mu_w + 1e-12)
                    mob_o_cur = kro / (mu_o + 1e-12)
                    mob_g_cur = krg / (mu_g + 1e-12)
                    mob_t_cur = mob_w_cur + mob_o_cur + mob_g_cur
                    fw_cur = mob_w_cur / (mob_t_cur + 1e-12)
                    fg_cur = mob_g_cur / (mob_t_cur + 1e-12)
                    fo_cur = mob_o_cur / (mob_t_cur + 1e-12)
                    # PVT на текущем P
                    Bo = self.fluid._eval_pvt(P_new, 'Bo'); Bw = self.fluid._eval_pvt(P_new, 'Bw'); Bg = self.fluid._eval_pvt(P_new, 'Bg')
                    Rs = self.fluid._eval_pvt(P_new, 'Rs')
                    try:
                        Rv = self.fluid._eval_pvt(P_new, 'Rv')
                    except Exception:
                        Rv = torch.zeros_like(Bg)
                    dt_sub = dt / N
                    # Скважины: вклад за подшаг
                    for well in self.well_manager.get_wells():
                        i, j, k = well.i, well.j, well.k
                        if i >= self.reservoir.nx or j >= self.reservoir.ny or k >= self.reservoir.nz:
                            continue
                        Bo_cell = float(Bo[i, j, k]); Bw_cell = float(Bw[i, j, k]); Bg_cell = float(Bg[i, j, k])
                        Rs_cell = float(Rs[i, j, k])
                        try:
                            Rv_cell = float(Rv[i, j, k])
                        except Exception:
                            Rv_cell = 0.0
                        fo_ = float(fo_cur[i, j, k]); fw_ = float(fw_cur[i, j, k]); fg_ = float(fg_cur[i, j, k])
                        # Вычисляем q_total для подшага по текущим долям
                        if well.control_type == 'rate':
                            rate_scale = float(self.sim_params.get('global_rate_scale', 1.0))
                            q_base = (well.control_value * rate_scale) / 86400.0
                            if well.type == 'injector':
                                if getattr(well, 'rate_type', 'reservoir') == 'surface':
                                    if getattr(well, 'injected_phase', 'water') == 'gas':
                                        q_total_sub = q_base * Bg_cell
                                    else:
                                        q_total_sub = q_base * Bw_cell
                                else:
                                    q_total_sub = q_base
                                q_total_sub *= 1.0
                                # IN для газа/воды
                                if getattr(well, 'injected_phase', 'water') == 'gas':
                                    qg_surf = max(q_total_sub, 0.0) / max(Bg_cell, 1e-12) * 86400.0
                                    self.component_balance['gas']['in'] += float(qg_surf * (dt_sub / 86400.0))
                                    self.component_balance['oil']['in'] += float(Rv_cell * qg_surf * (dt_sub / 86400.0))
                                else:
                                    qw_surf = max(q_total_sub, 0.0) / max(Bw_cell, 1e-12) * 86400.0
                                    self.component_balance['oil']['in'] += 0.0
                                    self.component_balance['gas']['in'] += 0.0
                            else:
                                # producer
                                if getattr(well, 'rate_type', 'reservoir') == 'surface':
                                    sp = (well.surface_phase or 'liquid').lower()
                                    if sp == 'oil':
                                        denom = (fo_ / max(Bo_cell, 1e-12) + fg_ * Rv_cell / max(Bg_cell, 1e-12))
                                        q_total_sub = - (q_base / max(denom, 1e-12))
                                    elif sp == 'water':
                                        q_w_res = q_base * Bw_cell
                                        q_total_sub = - q_w_res / max(fw_, 1e-12)
                                    elif sp == 'gas':
                                        denom = (fg_ / max(Bg_cell, 1e-12) + fo_ * Rs_cell / max(Bo_cell, 1e-12))
                                        q_total_sub = - (q_base / max(denom, 1e-12))
                                    else:
                                        denom = (fo_ / max(Bo_cell, 1e-12) + fg_ * Rv_cell / max(Bg_cell, 1e-12) + fw_ / max(Bw_cell, 1e-12))
                                        q_total_sub = - (q_base / max(denom, 1e-12))
                                else:
                                    q_total_sub = -q_base
                                qo_res = max((-q_total_sub * fo_), 0.0)
                                qg_res = max((-q_total_sub * fg_), 0.0)
                                oil_delta = dt_sub * (qo_res / max(Bo_cell, 1e-12) + Rv_cell * qg_res / max(Bg_cell, 1e-12))
                                gas_delta = dt_sub * (qg_res / max(Bg_cell, 1e-12) + Rs_cell * qo_res / max(Bo_cell, 1e-12))
                                self.component_balance['oil']['out'] += float(oil_delta)
                                self.component_balance['gas']['out'] += float(gas_delta)
                        elif well.control_type == 'bhp':
                            # оценим total по текущим долям (Mobility-weighted WI уже учтён на шаге давления)
                            wi = well.well_index
                            p_bhp = well.control_value * 1e6
                            p_block = float(P_new[i, j, k])
                            # приблизим mob_t по текущим долям (линейно): используем mob_t_cur
                            mob_t_cell = float(mob_t_cur[i, j, k])
                            q_total_sub = wi * mob_t_cell * (p_block - p_bhp)
                            if well.type == 'injector':
                                if getattr(well, 'injected_phase', 'water') == 'gas':
                                    qg_surf = max(-q_total_sub, 0.0) / max(Bg_cell, 1e-12) * 86400.0
                                    self.component_balance['gas']['in'] += float(qg_surf * (dt_sub / 86400.0))
                                    self.component_balance['oil']['in'] += float(Rv_cell * qg_surf * (dt_sub / 86400.0))
                            else:
                                qabs = abs(q_total_sub)
                                qo_res = qabs * fo_
                                qg_res = qabs * fg_
                                oil_delta = dt_sub * (qo_res / max(Bo_cell, 1e-12) + Rv_cell * qg_res / max(Bg_cell, 1e-12))
                                gas_delta = dt_sub * (qg_res / max(Bg_cell, 1e-12) + Rs_cell * qo_res / max(Bo_cell, 1e-12))
                                self.component_balance['oil']['out'] += float(oil_delta)
                                self.component_balance['gas']['out'] += float(gas_delta)
            except Exception:
                pass

        self.fluid.s_w = S_w_cur
        self.fluid.s_g = S_g_cur
        self.fluid.s_o = 1.0 - self.fluid.s_w - self.fluid.s_g

        # Агрегированный баланс масс по фактическим dS (без пост-клампов)
        dSw_eff = self.fluid.s_w - S_w_start
        dSg_eff = self.fluid.s_g - S_g_start
        rho_w_now = self.fluid.rho_w
        rho_g_now = self.fluid.rho_g
        rho_o_now = self.fluid.rho_o
        mw_eff = dSw_eff * self.porous_volume * rho_w_now
        mg_eff = dSg_eff * self.porous_volume * rho_g_now
        mo_eff = (-(dSw_eff + dSg_eff)) * self.porous_volume * rho_o_now
        self.mass_balance['water']['in'] += float(mw_eff.clamp(min=0).sum().item())
        self.mass_balance['water']['out'] += float((-mw_eff.clamp(max=0)).sum().item())
        self.mass_balance['gas']['in'] += float(mg_eff.clamp(min=0).sum().item())
        self.mass_balance['gas']['out'] += float((-mg_eff.clamp(max=0)).sum().item())
        self.mass_balance['oil']['in'] += float(mo_eff.clamp(min=0).sum().item())
        self.mass_balance['oil']['out'] += float((-mo_eff.clamp(max=0)).sum().item())

        # Компонентный баланс (surface-единицы): накопление = in - out за шаг
        self.component_balance['oil']['accum'] = self.component_balance['oil']['in'] - self.component_balance['oil']['out']
        self.component_balance['gas']['accum'] = self.component_balance['gas']['in'] - self.component_balance['gas']['out']

        affected_cells = torch.sum(torch.abs(dSw) > 1e-8).item()
        print(
            f"P̄ = {P_new.mean()/1e6:.2f} МПа, Sw(min/max) = {self.fluid.s_w.min():.3f}/{self.fluid.s_w.max():.3f}, "
            f"Sg(min/max) = {self.fluid.s_g.min():.3f}/{self.fluid.s_g.max():.3f}, substeps N={N}, ячеек изм.: {affected_cells}",
            flush=True
        )
        if getattr(self, 'debug_component_balance', False):
            cb = self.component_balance
            oil_res = (cb['oil']['in'] - cb['oil']['out']) - cb['oil']['accum']
            gas_res = (cb['gas']['in'] - cb['gas']['out']) - cb['gas']['accum']
            self._dbg(
                f"SUMMARY OIL in={cb['oil']['in']:.6e} out={cb['oil']['out']:.6e} accum={cb['oil']['accum']:.6e} residual={oil_res:.6e}",
                f"GAS in={cb['gas']['in']:.6e} out={cb['gas']['out']:.6e} accum={cb['gas']['accum']:.6e} residual={gas_res:.6e}"
        )

        clamp_counts = {
            "sw_low": int((self.fluid.s_w <= float(sw_cr) + 1e-6).sum().item()),
            "sw_high": int((self.fluid.s_w >= (1.0 - float(so_r) - self.fluid.s_g) - 1e-6).sum().item()),
            "sg_low": int((self.fluid.s_g <= float(sg_cr) + 1e-6).sum().item()),
            "sg_high": int((self.fluid.s_g >= (1.0 - float(so_r) - self.fluid.s_w) - 1e-6).sum().item()),
        }

        recommended_dt = dt if rat_max <= 1.0 else dt / max(rat_max, 1e-12)

        accum_values = {
            "water": float(mw_eff.sum().item()),
            "gas": float(mg_eff.sum().item()),
            "oil": float(mo_eff.sum().item()),
        }
        mass_report = {}
        for phase in ("water", "gas", "oil"):
            in_step = self.mass_balance[phase]['in'] - mass_before[phase]['in']
            out_step = self.mass_balance[phase]['out'] - mass_before[phase]['out']
            accum = accum_values[phase]
            mass_report[phase] = {
                "in": float(in_step),
                "out": float(out_step),
                "accum": accum,
                "imbalance": float(in_step - out_step - accum),
            }

        return {
            "recommended_dt": recommended_dt,
            "rat_max": float(rat_max),
            "max_substeps": max_substeps,
            "clamp_counts": clamp_counts,
            "mass_balance": mass_report,
            "log_messages": log_messages,
        }

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
        # Совокупная сжимаемость из PVT (если доступна)
        try:
            ct_tensor = self.fluid.calc_total_compressibility(self.fluid.pressure, self.fluid.s_w, self.fluid.s_g)
            acc_term = (self.porous_volume.view(-1) * ct_tensor.view(-1) / dt)
        except Exception:
            acc_term = (self.porous_volume.view(-1) * self.fluid.cf.view(-1) / dt)
        diag_dtype = vals.dtype if vals.numel() > 0 else torch.float64
        diag_vals = torch.zeros(N, device=self.device, dtype=diag_dtype)
        diag_vals.scatter_add_(0, rows, -vals)
        diag_vals += acc_term
        diag_vals += well_bhp_terms
        final_rows = torch.cat([rows, torch.arange(N, device=self.device)])
        final_cols = torch.cat([cols, torch.arange(N, device=self.device)])
        final_vals = torch.cat([vals, diag_vals])
        A = torch.sparse_coo_tensor(torch.stack([final_rows, final_cols]), final_vals, (N, N))
        return A.coalesce(), diag_vals

    def _build_pressure_rhs(self, dt, P_prev, mob_w, mob_o, mob_g, q_wells, dp_x_prev, dp_y_prev, dp_z_prev):
        """ Собирает правую часть Q для СЛАУ IMPES. """
        N = self.reservoir.nx * self.reservoir.ny * self.reservoir.nz
        # ВАЖНО: использовать тот же коэффициент аккумуляции, что и в матрице:
        # ct(P,Sw,Sg) из PVT (или fallback на self.fluid.cf), иначе возможен дрейф среднего давления.
        try:
            ct_tensor_rhs = self.fluid.calc_total_compressibility(self.fluid.pressure, self.fluid.s_w, self.fluid.s_g)
        except Exception:
            ct_tensor_rhs = self.fluid.cf
        compressibility_term = (self.porous_volume.view(-1) * ct_tensor_rhs.view(-1) / dt) * P_prev.view(-1)
        Q_g = torch.zeros_like(P_prev)
        _, _, dz = self.reservoir.grid_size
        if dz > 0 and self.reservoir.nz > 1:
            mob_w_z = torch.where(dp_z_prev > 0, mob_w[:,:,:-1], mob_w[:,:,1:])
            mob_o_z = torch.where(dp_z_prev > 0, mob_o[:,:,:-1], mob_o[:,:,1:])
            mob_g_z = torch.where(dp_z_prev > 0, mob_g[:,:,:-1], mob_g[:,:,1:])
            rho_w_z = torch.where(dp_z_prev > 0, self.fluid.rho_w[:,:,:-1], self.fluid.rho_w[:,:,1:])
            rho_o_z = torch.where(dp_z_prev > 0, self.fluid.rho_o[:,:,:-1], self.fluid.rho_o[:,:,1:])
            rho_g_z = torch.where(dp_z_prev > 0, self.fluid.rho_g[:,:,:-1], self.fluid.rho_g[:,:,1:])
            grav_flow = self.T_z * self.g * dz * (mob_w_z * rho_w_z + mob_o_z * rho_o_z + mob_g_z * rho_g_z)
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
            i, j, k = well.i, well.j, well.k
            idx = self._ijk_to_idx(i, j, k, self.reservoir.nx, self.reservoir.ny)
            well.cell_index_flat = idx
            if well.control_type == 'rate':
                # Знак источника зависит от типа скважины: положительный для инжектора, отрицательный для продуктора
                rate_scale = float(self.sim_params.get('global_rate_scale', 1.0))
                rate_si = (well.control_value * rate_scale) / 86400.0 * (1 if well.type == 'injector' else -1)
                # Авто-переключение на BHP при ограничении bhp_min (для продакшенов)
                if well.type == 'producer' and well.bhp_min is not None:
                    p_block = P_prev.view(-1)[idx]
                    mob_t_well = mob_t.view(-1)[idx]
                    wi = well.well_index
                    # Требуемый BHP для обеспечения заданного rate
                    # q = wi * mob_t * (p_block - p_bhp) => p_bhp_req = p_block - q/(wi*mob_t)
                    denom = wi * mob_t_well + 1e-12
                    p_bhp_req = p_block - rate_si / denom
                    p_bhp_min = well.bhp_min * 1e6
                    if p_bhp_req < p_bhp_min:
                        # Переключаемся на BHP-контроль на этом шаге
                        print(
                            f"  ℹ Скважина {well.name}: авто-переключение на BHP ({p_bhp_req/1e6:.2f} < bhp_min {well.bhp_min:.2f} МПа)"
                        )
                        well.last_mode = 'bhp'
                        well_bhp_terms[idx] += wi * mob_t_well
                        q_wells[idx] += wi * mob_t_well * p_bhp_min
                        continue
                well.last_mode = 'rate'
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
        else:  # цикл завершился без break, не сошлось
            return x, False

        return x, True

    def _assemble_residual_and_jacobian_batch(self, residual, jacobian, dt,
                                              p_vec, sw_vec, phi_vec, 
                                              perm_x_vec, perm_y_vec, perm_z_vec,
                                              lambda_w, lambda_o, lambda_t, fw, fo,
                                              rho_w, rho_o, mu_w, mu_o, 
                                              pc, dpc_dsw, nx, ny, nz, dx, dy, dz):
        """
        Векторизованная сборка остаточной невязки и якобиана для полностью неявной схемы.
        Оптимизированная версия с максимальным использованием векторизации.
        """
        if not self.reservoir.is_uniform_grid:
            raise NotImplementedError("Fully implicit схема поддерживает только равномерную сетку")
        num_cells = nx * ny * nz
        cell_volume = dx * dy * dz
        device = self.fluid.device
        
        # Проводимости для X, Y, Z направлений
        tx_const = dt * dy * dz / dx
        ty_const = dt * dx * dz / dy
        tz_const = dt * dx * dy / dz
        
        # Вектор гравитации
        gravity = torch.tensor([0.0, 0.0, -9.81], device=device)
        
        # Вычисляем массы для невязки аккумуляции
        water_mass = phi_vec * sw_vec * rho_w * cell_volume
        oil_mass = phi_vec * (1 - sw_vec) * rho_o * cell_volume
        
        # Заполняем невязки для аккумуляции (векторизовано)
        for idx in range(num_cells):
            residual[2*idx] = water_mass[idx] - self.fluid.prev_water_mass[idx]
            residual[2*idx+1] = oil_mass[idx] - self.fluid.prev_oil_mass[idx]
            
            # Производные для якобиана
            dphi_dp = self.reservoir.rock_compressibility * phi_vec[idx]
            drho_w_dp = self.fluid.water_compressibility * rho_w[idx]
            drho_o_dp = self.fluid.oil_compressibility * rho_o[idx]
            
            # Якобиан для воды
            jacobian[2*idx, 2*idx] = dphi_dp * sw_vec[idx] * rho_w[idx] * cell_volume + \
                                    phi_vec[idx] * sw_vec[idx] * drho_w_dp * cell_volume
            jacobian[2*idx, 2*idx+1] = phi_vec[idx] * rho_w[idx] * cell_volume
            
            # Якобиан для нефти
            jacobian[2*idx+1, 2*idx] = dphi_dp * (1-sw_vec[idx]) * rho_o[idx] * cell_volume + \
                                     phi_vec[idx] * (1-sw_vec[idx]) * drho_o_dp * cell_volume
            jacobian[2*idx+1, 2*idx+1] = -phi_vec[idx] * rho_o[idx] * cell_volume
        
        # Вычисляем и кэшируем производные относительных проницаемостей
        dkr_w_dsw = self.fluid.calc_dkrw_dsw(sw_vec)
        dkr_o_dsw = self.fluid.calc_dkro_dsw(sw_vec)
        dlambda_w_dsw = dkr_w_dsw / mu_w
        dlambda_o_dsw = dkr_o_dsw / mu_o
        
        # Векторизованный расчет потоков для X-направления
        for i in range(nx-1):
            for j in range(ny):
                for k in range(nz):
                    idx1 = i + j * nx + k * nx * ny
                    idx2 = (i+1) + j * nx + k * nx * ny
                    
                    # Средние значения
                    avg_perm = 2 * perm_x_vec[idx1] * perm_x_vec[idx2] / (perm_x_vec[idx1] + perm_x_vec[idx2] + 1e-10)
                    avg_rho_w = 0.5 * (rho_w[idx1] + rho_w[idx2])
                    avg_rho_o = 0.5 * (rho_o[idx1] + rho_o[idx2])
                    
                    # Градиенты давления
                    dp = p_vec[idx2] - p_vec[idx1]
                    dp_cap = pc[idx2] - pc[idx1] if self.fluid.pc_scale > 0 else 0.0
                    
                    # Гравитационный член (нет для X)
                    gravity_term_w = 0.0
                    gravity_term_o = 0.0
                    
                    # Восходящие мобильности
                    lambda_w_up = lambda_w[idx1] if dp >= 0 else lambda_w[idx2]
                    lambda_o_up = lambda_o[idx1] if dp >= 0 else lambda_o[idx2]
                    
                    # Проводимость
                    trans = tx_const * avg_perm
                    
                    # Потоки
                    water_flux = trans * lambda_w_up * (dp - dp_cap + gravity_term_w)
                    oil_flux = trans * lambda_o_up * (dp + gravity_term_o)
                    
                    # Невязка
                    residual[2*idx1] -= water_flux
                    residual[2*idx1+1] -= oil_flux
                    residual[2*idx2] += water_flux
                    residual[2*idx2+1] += oil_flux
                    
                    # Производные для якобиана
                    # Производные по давлению
                    dfw_dp1 = trans * lambda_w_up * (-1.0)
                    dfw_dp2 = trans * lambda_w_up * (1.0)
                    dfo_dp1 = trans * lambda_o_up * (-1.0)
                    dfo_dp2 = trans * lambda_o_up * (1.0)
                    
                    # Добавляем в якобиан
                    jacobian[2*idx1, 2*idx1] -= dfw_dp1
                    jacobian[2*idx1, 2*idx2] -= dfw_dp2
                    jacobian[2*idx1+1, 2*idx1] -= dfo_dp1
                    jacobian[2*idx1+1, 2*idx2] -= dfo_dp2
                    
                    jacobian[2*idx2, 2*idx1] += dfw_dp1
                    jacobian[2*idx2, 2*idx2] += dfw_dp2
                    jacobian[2*idx2+1, 2*idx1] += dfo_dp1
                    jacobian[2*idx2+1, 2*idx2] += dfo_dp2
                    
                    # Производные по насыщенности
                    if dp >= 0:  # Восходящая схема слева
                        dfw_dsw1 = trans * dlambda_w_dsw[idx1] * (dp - dp_cap + gravity_term_w)
                        dfo_dsw1 = trans * dlambda_o_dsw[idx1] * (dp + gravity_term_o)
                        
                        if self.fluid.pc_scale > 0:
                            dfw_dsw1 += trans * lambda_w_up * (-dpc_dsw[idx1])
                            dfw_dsw2 = trans * lambda_w_up * dpc_dsw[idx2]
                            
                            jacobian[2*idx1, 2*idx1+1] -= dfw_dsw1
                            jacobian[2*idx1, 2*idx2+1] -= dfw_dsw2
                            jacobian[2*idx2, 2*idx1+1] += dfw_dsw1
                            jacobian[2*idx2, 2*idx2+1] += dfw_dsw2
                        else:
                            jacobian[2*idx1, 2*idx1+1] -= dfw_dsw1
                            jacobian[2*idx2, 2*idx1+1] += dfw_dsw1
                        
                        jacobian[2*idx1+1, 2*idx1+1] -= dfo_dsw1
                        jacobian[2*idx2+1, 2*idx1+1] += dfo_dsw1
                        
                    else:  # Восходящая схема справа
                        dfw_dsw2 = trans * dlambda_w_dsw[idx2] * (dp - dp_cap + gravity_term_w)
                        dfo_dsw2 = trans * dlambda_o_dsw[idx2] * (dp + gravity_term_o)
                        
                        if self.fluid.pc_scale > 0:
                            dfw_dsw1 = trans * lambda_w_up * (-dpc_dsw[idx1])
                            dfw_dsw2 += trans * lambda_w_up * dpc_dsw[idx2]
                            
                            jacobian[2*idx1, 2*idx1+1] -= dfw_dsw1
                            jacobian[2*idx1, 2*idx2+1] -= dfw_dsw2
                            jacobian[2*idx2, 2*idx1+1] += dfw_dsw1
                            jacobian[2*idx2, 2*idx2+1] += dfw_dsw2
                        else:
                            jacobian[2*idx1, 2*idx2+1] -= dfw_dsw2
                            jacobian[2*idx2, 2*idx2+1] += dfw_dsw2
                        
                        jacobian[2*idx1+1, 2*idx2+1] -= dfo_dsw2
                        jacobian[2*idx2+1, 2*idx2+1] += dfo_dsw2
        
        # Аналогично для Y-направления
        for j in range(ny-1):
            for i in range(nx):
                for k in range(nz):
                    idx1 = i + j * nx + k * nx * ny
                    idx2 = i + (j+1) * nx + k * nx * ny
                    
                    avg_perm = 2 * perm_y_vec[idx1] * perm_y_vec[idx2] / (perm_y_vec[idx1] + perm_y_vec[idx2] + 1e-10)
                    avg_rho_w = 0.5 * (rho_w[idx1] + rho_w[idx2])
                    avg_rho_o = 0.5 * (rho_o[idx1] + rho_o[idx2])
                    
                    dp = p_vec[idx2] - p_vec[idx1]
                    dp_cap = pc[idx2] - pc[idx1] if self.fluid.pc_scale > 0 else 0.0
                    
                    # Гравитационный член (нет для Y)
                    gravity_term_w = 0.0
                    gravity_term_o = 0.0
                    
                    # Восходящие мобильности
                    lambda_w_up = lambda_w[idx1] if dp >= 0 else lambda_w[idx2]
                    lambda_o_up = lambda_o[idx1] if dp >= 0 else lambda_o[idx2]
                    
                    # Проводимость
                    trans = ty_const * avg_perm
                    
                    # Потоки
                    water_flux = trans * lambda_w_up * (dp - dp_cap + gravity_term_w)
                    oil_flux = trans * lambda_o_up * (dp + gravity_term_o)
                    
                    # Невязка
                    residual[2*idx1] -= water_flux
                    residual[2*idx1+1] -= oil_flux
                    residual[2*idx2] += water_flux
                    residual[2*idx2+1] += oil_flux
                    
                    # Производные для якобиана
                    # Производные по давлению
                    dfw_dp1 = trans * lambda_w_up * (-1.0)
                    dfw_dp2 = trans * lambda_w_up * (1.0)
                    dfo_dp1 = trans * lambda_o_up * (-1.0)
                    dfo_dp2 = trans * lambda_o_up * (1.0)
                    
                    # Добавляем в якобиан
                    jacobian[2*idx1, 2*idx1] -= dfw_dp1
                    jacobian[2*idx1, 2*idx2] -= dfw_dp2
                    jacobian[2*idx1+1, 2*idx1] -= dfo_dp1
                    jacobian[2*idx1+1, 2*idx2] -= dfo_dp2
                    
                    jacobian[2*idx2, 2*idx1] += dfw_dp1
                    jacobian[2*idx2, 2*idx2] += dfw_dp2
                    jacobian[2*idx2+1, 2*idx1] += dfo_dp1
                    jacobian[2*idx2+1, 2*idx2] += dfo_dp2
                    
                    # Производные по насыщенности (аналогично X-направлению)
                    if dp >= 0:
                        dfw_dsw1 = trans * dlambda_w_dsw[idx1] * (dp - dp_cap + gravity_term_w)
                        dfo_dsw1 = trans * dlambda_o_dsw[idx1] * (dp + gravity_term_o)
                        
                        if self.fluid.pc_scale > 0:
                            dfw_dsw1 += trans * lambda_w_up * (-dpc_dsw[idx1])
                            dfw_dsw2 = trans * lambda_w_up * dpc_dsw[idx2]
                            
                            jacobian[2*idx1, 2*idx1+1] -= dfw_dsw1
                            jacobian[2*idx1, 2*idx2+1] -= dfw_dsw2
                            jacobian[2*idx2, 2*idx1+1] += dfw_dsw1
                            jacobian[2*idx2, 2*idx2+1] += dfw_dsw2
                        else:
                            jacobian[2*idx1, 2*idx1+1] -= dfw_dsw1
                            jacobian[2*idx2, 2*idx1+1] += dfw_dsw1
                        
                        jacobian[2*idx1+1, 2*idx1+1] -= dfo_dsw1
                        jacobian[2*idx2+1, 2*idx1+1] += dfo_dsw1
                        
                    else:
                        dfw_dsw2 = trans * dlambda_w_dsw[idx2] * (dp - dp_cap + gravity_term_w)
                        
                        if self.fluid.pc_scale > 0:
                            dfw_dsw1 = trans * lambda_w_up * (-dpc_dsw[idx1])
                            dfw_dsw2 += trans * lambda_w_up * dpc_dsw[idx2]
                            
                            jacobian[2*idx1, 2*idx1+1] -= dfw_dsw1
                            jacobian[2*idx1, 2*idx2+1] -= dfw_dsw2
                            jacobian[2*idx2, 2*idx1+1] += dfw_dsw1
                            jacobian[2*idx2, 2*idx2+1] += dfw_dsw2
                        else:
                            jacobian[2*idx1, 2*idx2+1] -= dfw_dsw2
                            jacobian[2*idx2, 2*idx2+1] += dfw_dsw2
                        
                        jacobian[2*idx1+1, 2*idx2+1] -= dfo_dsw2
                        jacobian[2*idx2+1, 2*idx2+1] += dfo_dsw2
        
        # Z-направление с учетом гравитации
        for k in range(nz-1):
            for i in range(nx):
                for j in range(ny):
                    idx1 = i + j * nx + k * nx * ny
                    idx2 = i + j * nx + (k+1) * nx * ny
                    
                    avg_perm = 2 * perm_z_vec[idx1] * perm_z_vec[idx2] / (perm_z_vec[idx1] + perm_z_vec[idx2] + 1e-10)
                    avg_rho_w = 0.5 * (rho_w[idx1] + rho_w[idx2])
                    avg_rho_o = 0.5 * (rho_o[idx1] + rho_o[idx2])
                    
                    dp = p_vec[idx2] - p_vec[idx1]
                    dp_cap = pc[idx2] - pc[idx1] if self.fluid.pc_scale > 0 else 0.0
                    
                    # Гравитационные члены для Z-направления
                    gravity_term_w = avg_rho_w * gravity[2] * dz
                    gravity_term_o = avg_rho_o * gravity[2] * dz
                    
                    # Восходящие мобильности
                    lambda_w_up = lambda_w[idx1] if dp - dp_cap + gravity_term_w >= 0 else lambda_w[idx2]
                    lambda_o_up = lambda_o[idx1] if dp + gravity_term_o >= 0 else lambda_o[idx2]
                    
                    # Проводимость
                    trans = tz_const * avg_perm
                    
                    # Потоки
                    water_flux = trans * lambda_w_up * (dp - dp_cap + gravity_term_w)
                    oil_flux = trans * lambda_o_up * (dp + gravity_term_o)
                    
                    # Невязка
                    residual[2*idx1] -= water_flux
                    residual[2*idx1+1] -= oil_flux
                    residual[2*idx2] += water_flux
                    residual[2*idx2+1] += oil_flux
                    
                    # Производные для якобиана
                    # Производные по давлению
                    dfw_dp1 = trans * lambda_w_up * (-1.0)
                    dfw_dp2 = trans * lambda_w_up * (1.0)
                    dfo_dp1 = trans * lambda_o_up * (-1.0)
                    dfo_dp2 = trans * lambda_o_up * (1.0)
                    
                    # Добавляем в якобиан
                    jacobian[2*idx1, 2*idx1] -= dfw_dp1
                    jacobian[2*idx1, 2*idx2] -= dfw_dp2
                    jacobian[2*idx1+1, 2*idx1] -= dfo_dp1
                    jacobian[2*idx1+1, 2*idx2] -= dfo_dp2
                    
                    jacobian[2*idx2, 2*idx1] += dfw_dp1
                    jacobian[2*idx2, 2*idx2] += dfw_dp2
                    jacobian[2*idx2+1, 2*idx1] += dfo_dp1
                    jacobian[2*idx2+1, 2*idx2] += dfo_dp2
                    
                    # Производные по насыщенности с учетом гравитации
                    if dp - dp_cap + gravity_term_w >= 0:
                        dfw_dsw1 = trans * dlambda_w_dsw[idx1] * (dp - dp_cap + gravity_term_w)
                        
                        if self.fluid.pc_scale > 0:
                            dfw_dsw1 += trans * lambda_w_up * (-dpc_dsw[idx1])
                            dfw_dsw2 = trans * lambda_w_up * dpc_dsw[idx2]
                            
                            jacobian[2*idx1, 2*idx1+1] -= dfw_dsw1
                            jacobian[2*idx1, 2*idx2+1] -= dfw_dsw2
                            jacobian[2*idx2, 2*idx1+1] += dfw_dsw1
                            jacobian[2*idx2, 2*idx2+1] += dfw_dsw2
                        else:
                            jacobian[2*idx1, 2*idx1+1] -= dfw_dsw1
                            jacobian[2*idx2, 2*idx1+1] += dfw_dsw1
                    else:
                        dfw_dsw2 = trans * dlambda_w_dsw[idx2] * (dp - dp_cap + gravity_term_w)
                        
                        if self.fluid.pc_scale > 0:
                            dfw_dsw1 = trans * lambda_w_up * (-dpc_dsw[idx1])
                            dfw_dsw2 += trans * lambda_w_up * dpc_dsw[idx2]
                            
                            jacobian[2*idx1, 2*idx1+1] -= dfw_dsw1
                            jacobian[2*idx1, 2*idx2+1] -= dfw_dsw2
                            jacobian[2*idx2, 2*idx1+1] += dfw_dsw1
                            jacobian[2*idx2, 2*idx2+1] += dfw_dsw2
                        else:
                            jacobian[2*idx1, 2*idx2+1] -= dfw_dsw2
                            jacobian[2*idx2, 2*idx2+1] += dfw_dsw2
                    
                    # Для нефти
                    if dp + gravity_term_o >= 0:
                        dfo_dsw1 = trans * dlambda_o_dsw[idx1] * (dp + gravity_term_o)
                        jacobian[2*idx1+1, 2*idx1+1] -= dfo_dsw1
                        jacobian[2*idx2+1, 2*idx1+1] += dfo_dsw1
                    else:
                        dfo_dsw2 = trans * dlambda_o_dsw[idx2] * (dp + gravity_term_o)
                        jacobian[2*idx1+1, 2*idx2+1] -= dfo_dsw2
                        jacobian[2*idx2+1, 2*idx2+1] += dfo_dsw2
        
        # Добавляем вклад скважин
        self._add_wells_to_system(residual, jacobian, dt)

    def _add_wells_to_system(self, residual, jacobian, dt):
        """
        Добавляет вклад скважин в систему (невязку и якобиан).
        Оптимизированная версия.
        """
        wells = self.well_manager.get_wells()
        
        for well in wells:
            idx = well.cell_index_flat
            p = self.fluid.pressure.reshape(-1)[idx]
            sw = self.fluid.s_w.reshape(-1)[idx]
            
            # Вычисляем подвижности в ячейке со скважиной
            mu_w = self.fluid.mu_water
            mu_o = self.fluid.mu_oil
            kr_w = self.fluid.calc_water_kr(sw)
            kr_o = self.fluid.calc_oil_kr(sw)
            lambda_w = kr_w / mu_w
            lambda_o = kr_o / mu_o
            lambda_t = lambda_w + lambda_o
            
            # Вычисляем производные подвижностей
            dkr_w_dsw = self.fluid.calc_dkrw_dsw(sw)
            dkr_o_dsw = self.fluid.calc_dkro_dsw(sw)
            dlambda_w_dsw = dkr_w_dsw / mu_w
            dlambda_o_dsw = dkr_o_dsw / mu_o
            
            if well.control_type == 'rate':
                # Скважина с контролем по дебиту
                rate = well.control_value / 86400.0  # м³/с
                
                if well.type == 'injector':
                    # Нагнетательная скважина (вода)
                    q_w = rate
                    q_o = 0.0
                    
                    # Обновляем невязку
                    residual[2*idx] -= q_w
                    residual[2*idx+1] -= q_o
                    
                else:  # producer
                    # Добывающая скважина
                    fw = lambda_w / (lambda_t + 1e-10)
                    fo = lambda_o / (lambda_t + 1e-10)
                    
                    q_w = rate * fw
                    q_o = rate * fo
                    
                    # Производные дебитов по насыщенности
                    dfw_dsw = (dlambda_w_dsw * lambda_t - lambda_w * (dlambda_w_dsw + dlambda_o_dsw)) / (lambda_t**2 + 1e-10)
                    dfo_dsw = -dfw_dsw
                    
                    dq_w_dsw = rate * dfw_dsw
                    dq_o_dsw = rate * dfo_dsw
                    
                    # Обновляем невязку
                    residual[2*idx] -= q_w
                    residual[2*idx+1] -= q_o
                    
                    # Обновляем якобиан
                    jacobian[2*idx, 2*idx+1] -= dq_w_dsw
                    jacobian[2*idx+1, 2*idx+1] -= dq_o_dsw
                    
            elif well.control_type == 'bhp':
                # Скважина с контролем забойного давления
                bhp = well.control_value * 1e6  # МПа -> Па
                
                # Дебиты
                q_w = well.well_index * lambda_w * (p - bhp)
                q_o = well.well_index * lambda_o * (p - bhp)
                
                # Обновляем невязку
                residual[2*idx] -= q_w
                residual[2*idx+1] -= q_o
                
                # Производные по давлению
                dq_w_dp = well.well_index * lambda_w
                dq_o_dp = well.well_index * lambda_o
                
                # Производные по насыщенности
                dq_w_dsw = well.well_index * dlambda_w_dsw * (p - bhp)
                dq_o_dsw = well.well_index * dlambda_o_dsw * (p - bhp)
                
                # Обновляем якобиан
                jacobian[2*idx, 2*idx] -= dq_w_dp
                jacobian[2*idx, 2*idx+1] -= dq_w_dsw
                jacobian[2*idx+1, 2*idx] -= dq_o_dp
                jacobian[2*idx+1, 2*idx+1] -= dq_o_dsw

    def run(self, output_filename, save_vtk=False, save_vtk_intermediate=False, save_3d_visualization=False):
        """
        Запускает полную симуляцию.
        
        Args:
            output_filename: Имя файла для сохранения результатов
            save_vtk: Флаг для сохранения результатов в формате VTK (финальный)
            save_vtk_intermediate: Сохранять VTK на промежуточных шагах
            save_3d_visualization: Сохранять интерактивные 3D визуализации через PyVista
        """
        # Импорт необходимых модулей
        import os
        import numpy as np
        from tqdm import tqdm
        try:
            from ..plotting.plotter import Plotter
        except ImportError:
            # Если относительный импорт не работает, используем абсолютный
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from plotting.plotter import Plotter
        
        # Получаем параметры симуляции
        total_time_days = self.sim_params.get('total_time_days', 100)
        time_step_days = self.sim_params.get('time_step_days', 1.0)
        time_step_sec = time_step_days * 86400
        save_interval = self.sim_params.get('save_interval', 10)
        animation_fps = self.sim_params.get('animation_fps', 5)
        
        # Рассчитываем количество шагов
        num_steps = int(total_time_days / time_step_days)
        
        # Создаем уникальную директорию для данного запуска
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("results", f"{output_filename}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        intermediate_results_dir = os.path.join(results_dir, "intermediate")
        if save_interval < num_steps:
            os.makedirs(intermediate_results_dir, exist_ok=True)
        
        # Выводим информацию о симуляции
        print(f"\nЗапуск симуляции на {num_steps} шагов по {time_step_days} дней...")
        print(f"Результаты будут сохраняться каждые {save_interval} шагов.")
        
        # Создаем объект для визуализации
        plotter = Plotter(self.reservoir)
        
        # save_3d_visualization больше не используется - используем только VTK файлы
        # VTK файлы можно открыть в ParaView для просмотра
        
        # Основной цикл симуляции
        for i in tqdm(range(num_steps), desc="Симуляция"):
            # Выполняем один шаг
            success = self.run_step(dt=time_step_sec)
            
            # Если шаг не сошелся, пробуем с меньшим шагом
            if not success and self.solver_type == 'fully_implicit':
                print(f"  Шаг {i+1} не сошелся. Пробуем с меньшим временным шагом.")
                reduced_dt = time_step_sec / 2
                success = self.run_step(dt=reduced_dt)
                
                if not success:
                    print(f"  Шаг {i+1} не сошелся даже с уменьшенным шагом. Завершаем симуляцию.")
                    break
            
            # Сохранение промежуточных результатов
            if (i + 1) % save_interval == 0 and save_interval < num_steps:
                p_current = self.fluid.pressure.cpu().numpy()
                sw_current = self.fluid.s_w.cpu().numpy()
                sg_current = self.fluid.s_g.cpu().numpy()
                
                time_info = f"День {int((i + 1) * time_step_days)}"
                filename = f"{output_filename}_step_{i+1}.png"
                filepath = os.path.join(intermediate_results_dir, filename)
                
                plotter.save_plots(p_current, sw_current, filepath, time_info=time_info, gas_saturation=sg_current)
                
                # VTK промежуточные шаги
                if save_vtk_intermediate:
                    try:
                        import sys
                        import os as os_module
                        # Пробуем разные способы импорта
                        try:
                            from ..output.vtk_writer import save_to_vtk
                        except (ImportError, ValueError):
                            # Fallback для случаев, когда относительный импорт не работает
                            base_path = os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__)))
                            vtk_path = os_module.path.join(base_path, 'output', 'vtk_writer.py')
                            if base_path not in sys.path:
                                sys.path.insert(0, base_path)
                            from output.vtk_writer import save_to_vtk
                        
                        vtk_filename = os.path.join(intermediate_results_dir, f"{output_filename}_step_{i+1}")
                        save_to_vtk(self.reservoir, self.fluid, vtk_filename)
                    except Exception as e:
                        print(f"  ⚠ Не удалось сохранить VTK для шага {i+1}: {e}")
                
                # Сохранение данных для 3D визуализации (VTK файлы уже создаются выше)
                # VTK файлы можно открыть в ParaView для просмотра
        
        print("\nСимуляция завершена.")
        
        # Сохранение и визуализация финальных результатов
        p_final = self.fluid.pressure.cpu().numpy()
        sw_final = self.fluid.s_w.cpu().numpy()
        sg_final = self.fluid.s_g.cpu().numpy()
        so_final = self.fluid.s_o.cpu().numpy()
        
        print(f"Итоговое давление: Мин={p_final.min()/1e6:.2f} МПа, Макс={p_final.max()/1e6:.2f} МПа")
        print(f"Итоговая водонасыщенность: Мин={sw_final.min():.4f}, Макс={sw_final.max():.4f}")
        print(f"Итоговая газонасыщенность: Мин={sg_final.min():.4f}, Макс={sg_final.max():.4f}")
        print(f"Итоговая нефтенасыщенность: Мин={so_final.min():.4f}, Макс={so_final.max():.4f}")
        
        # Сохранение числовых данных (сводка и опционально полные массивы)
        results_txt_path = os.path.join(results_dir, f"{output_filename}.txt")
        write_full_arrays_txt = self.sim_params.get('write_full_arrays_txt', False)
        save_npz = self.sim_params.get('save_npz', True)

        with open(results_txt_path, 'w') as f:
            # Сводные метрики
            f.write("=== SUMMARY (final) ===\n")
            f.write(f"Pressure MPa: min={p_final.min()/1e6:.2f}, max={p_final.max()/1e6:.2f}, mean={p_final.mean()/1e6:.2f}\n")
            f.write(f"Sw: min={sw_final.min():.4f}, max={sw_final.max():.4f}, mean={sw_final.mean():.4f}\n")
            f.write(f"Sg: min={sg_final.min():.4f}, max={sg_final.max():.4f}, mean={sg_final.mean():.4f}\n")
            f.write(f"So: min={so_final.min():.4f}, max={so_final.max():.4f}, mean={so_final.mean():.4f}\n")

            # Баланс масс
            mb = self.mass_balance
            net_w = mb['water']['in'] - mb['water']['out']
            net_g = mb['gas']['in'] - mb['gas']['out']
            net_o = mb['oil']['in'] - mb['oil']['out']
            f.write("\n=== MASS BALANCE (kg) ===\n")
            f.write(f"water: in={mb['water']['in']:.6e}, out={mb['water']['out']:.6e}, net={net_w:.6e}\n")
            f.write(f"gas:   in={mb['gas']['in']:.6e}, out={mb['gas']['out']:.6e}, net={net_g:.6e}\n")
            f.write(f"oil:   in={mb['oil']['in']:.6e}, out={mb['oil']['out']:.6e}, net={net_o:.6e}\n")
            f.write(f"total net={net_w+net_g+net_o:.6e}\n")

            # Полные массивы (опционально в txt — крайне большие)
            if write_full_arrays_txt:
                f.write("\n\nFinal Pressure (MPa):\n")
                f.write(np.array2string(p_final/1e6, threshold=np.inf, formatter={'float_kind':lambda x: "%.2f" % x}))
                f.write("\n\nFinal Water Saturation:\n")
                f.write(np.array2string(sw_final, threshold=np.inf, formatter={'float_kind':lambda x: "%.4f" % x}))
                f.write("\n\nFinal Gas Saturation:\n")
                f.write(np.array2string(sg_final, threshold=np.inf, formatter={'float_kind':lambda x: "%.4f" % x}))
                f.write("\n\nFinal Oil Saturation:\n")
                f.write(np.array2string(so_final, threshold=np.inf, formatter={'float_kind':lambda x: "%.4f" % x}))

        print(f"Числовые результаты сохранены в файл {results_txt_path}")

        # Сохранение полных массивов в сжатом формате
        if save_npz:
            npz_path = os.path.join(results_dir, f"{output_filename}.npz")
            np.savez_compressed(npz_path, pressure=p_final, sw=sw_final, sg=sg_final, so=so_final)
            print(f"Полные поля сохранены в {npz_path} (npz)")
        
        # Сохранение финальных графиков
        final_plot_path = os.path.join(results_dir, f"{output_filename}_final.png")
        plotter.save_plots(p_final, sw_final, final_plot_path, time_info=f"День {total_time_days} (Final)", gas_saturation=sg_final)
        print(f"Финальные графики сохранены в файл {final_plot_path}")
        
        # Сохранение результатов в VTK (если указано)
        if save_vtk:
            try:
                from ..output.vtk_writer import save_to_vtk
                vtk_path = os.path.join(results_dir, output_filename)
                save_to_vtk(self.reservoir, self.fluid, vtk_path)
            except Exception as e:
                print(f"Не удалось сохранить в формате VTK: {e}")
        
        # Информация о просмотре VTK файлов
        if save_vtk or save_vtk_intermediate:
            print(f"\n📖 Для просмотра 3D визуализации в ParaView:")
            print(f"   1. Установите ParaView: https://www.paraview.org/")
            print(f"   2. Откройте ParaView → File → Open → выберите .vtr файлы из:")
            if save_vtk_intermediate:
                print(f"      {intermediate_results_dir}/")
            if save_vtk:
                print(f"      {results_dir}/")
            print(f"   3. ⚠ Если видите коричневый экран:")
            print(f"      - Нажмите Apply в панели Properties")
            print(f"      - В панели Coloring выберите: Color by = Pressure_MPa")
            print(f"      - Representation = Surface (вместо Outline)")
            print(f"   4. Или создайте срез: Filters → Slice → Normal = [0,0,1]")
            print(f"   5. Подробная инструкция: см. PARAVIEW_INSTRUCTIONS.md")
        
        # Создание анимации (если нужно)
        if save_interval < num_steps and animation_fps > 0:
            try:
                gif_path = os.path.join(results_dir, f"{output_filename}.gif")
                print(f"\nСоздание анимации с {animation_fps} FPS...")
                self._create_animation(intermediate_results_dir, gif_path, fps=animation_fps)
            except ImportError:
                print("Не удалось создать анимацию: отсутствует модуль imageio.")

    def _save_to_vtk(self, output_filename):
        """
        Сохраняет результаты в формате VTK.
        
        Args:
            output_filename: Имя файла для сохранения
        """
        try:
            import vtk
            from vtkmodules.util import numpy_support
            # Создаем VTK-объекты
            grid = vtk.vtkStructuredGrid()
            
            nx, ny, nz = self.reservoir.dimensions
            grid.SetDimensions(nx, ny, nz)
            
            # Создаем точки сетки
            points = vtk.vtkPoints()
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        points.InsertNextPoint(i, j, k)
            grid.SetPoints(points)
            
            # Добавляем данные давления
            pressure_array = self.fluid.pressure.cpu().numpy().flatten()
            vtk_pressure = numpy_support.numpy_to_vtk(pressure_array)
            vtk_pressure.SetName("Pressure")
            grid.GetPointData().AddArray(vtk_pressure)
            
            # Добавляем данные насыщенности
            sw_array = self.fluid.s_w.cpu().numpy().flatten()
            vtk_sw = numpy_support.numpy_to_vtk(sw_array)
            vtk_sw.SetName("Water_Saturation")
            grid.GetPointData().AddArray(vtk_sw)
            
            # Записываем в файл
            writer = vtk.vtkXMLStructuredGridWriter()
            writer.SetFileName(os.path.join(results_dir, f"{output_filename}.vts"))
            writer.SetInputData(grid)
            writer.Write()
            
            print(f"Результаты сохранены в формате VTK.")
        except ImportError:
            print("Не удалось сохранить в формате VTK: отсутствуют необходимые модули.")

    def _create_animation(self, images_dir, output_path, fps=5):
        """
        Создает анимацию из набора изображений.
        
        Args:
            images_dir: Директория с изображениями
            output_path: Путь для сохранения анимации
            fps: Кадров в секунду
        """
        # Локальные импорты, чтобы не создавать зависимость при импорте модуля
        import glob
        import imageio
        import re
        # Получаем список файлов изображений и сортируем их по номеру шага,
        # а не лексикографически, чтобы кадры шли в правильном хронологическом порядке
        regex = re.compile(r'_step_(\d+)\.png$')
        step_files = []
        for path in glob.glob(os.path.join(images_dir, "*_step_*.png")):
            m = regex.search(os.path.basename(path))
            if m:
                step_files.append((int(m.group(1)), path))

        # Сортируем по номеру шага
        step_files.sort(key=lambda t: t[0])
        images = [p for _, p in step_files]
        
        # Добавляем финальный кадр, если существует (лежит в родительской директории)
        final_candidates = glob.glob(os.path.join(os.path.dirname(images_dir), f"{os.path.basename(images_dir).split(os.sep)[-1].replace('intermediate','')}*_final.png"))
        if final_candidates:
            images.append(final_candidates[0])
        
        # Создаем анимацию
        with imageio.get_writer(output_path, mode='I', fps=fps) as writer:
            for image_path in images:
                image = imageio.imread(image_path)
                writer.append_data(image)
        
        print(f"Анимация сохранена в файл {output_path}")

    def _dbg(self, *args):
        if not getattr(self, 'debug_component_balance', False):
            return
        try:
            line = " ".join(str(x) for x in args)
            # Печать в консоль
            print(line, flush=True)
            with open(self.debug_log_path, 'a') as f:
                f.write(line + "\n")
        except Exception:
            pass
