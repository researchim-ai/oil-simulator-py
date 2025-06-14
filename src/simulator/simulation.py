import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, diags, bmat
from scipy.sparse.linalg import cg, LinearOperator, bicgstab, spsolve
import time
import os
import datetime

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
            dx, dy, dz = self.reservoir.grid_size
            k_x, k_y, k_z = self.reservoir.permeability_tensors
            nx, ny, nz = self.reservoir.dimensions
            
            # Вычисляем проводимости для каждого направления
            self.T_x = torch.zeros((nx-1, ny, nz), device=self.device)
            self.T_y = torch.zeros((nx, ny-1, nz), device=self.device)
            self.T_z = torch.zeros((nx, ny, nz-1), device=self.device)
            
            # Расчет проводимостей
            for i in range(nx-1):
                k_harmonic = 2 * k_x[i, :, :] * k_x[i+1, :, :] / (k_x[i, :, :] + k_x[i+1, :, :] + 1e-15)
                self.T_x[i, :, :] = k_harmonic * dy * dz / dx
            
            for j in range(ny-1):
                k_harmonic = 2 * k_y[:, j, :] * k_y[:, j+1, :] / (k_y[:, j, :] + k_y[:, j+1, :] + 1e-15)
                self.T_y[:, j, :] = k_harmonic * dx * dz / dy
            
            for k in range(nz-1):
                k_harmonic = 2 * k_z[:, :, k] * k_z[:, :, k+1] / (k_z[:, :, k] + k_z[:, :, k+1] + 1e-15)
                self.T_z[:, :, k] = k_harmonic * dx * dy / dz
        
        # Объем пористой среды для расчетов IMPES
        self.porous_volume = self.reservoir.porous_volume
        
        # Гравитационная постоянная для IMPES
        self.g = 9.81
        
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
                cell_volume = dx * dy * dz
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

        consecutive_success = 0
        last_dt_increased = False

        for attempt in range(max_attempts):
            print(f"Попытка шага IMPES с dt = {current_dt/86400:.2f} дней (Попытка {attempt+1}/{max_attempts})")

            P_new, converged = self._impes_pressure_step(current_dt)

            if converged:
                # Обновляем давление и выполняем шаг насыщенности
                self.fluid.pressure = P_new
                self._impes_saturation_step(P_new, current_dt)

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

        kro, krw = self.fluid.get_rel_perms(S_w)
        mu_o_pas = self.fluid.mu_oil
        mu_w_pas = self.fluid.mu_water

        mob_w = krw / mu_w_pas
        mob_o = kro / mu_o_pas
        mob_t = mob_w + mob_o

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
        Q = self._build_pressure_rhs(dt, P_prev, mob_w, mob_o, q_wells, dp_x_prev, dp_y_prev, dp_z_prev)

        # 5. Параметры CG из конфигурации
        cg_tol_base = self.sim_params.get("cg_tolerance", 1e-6)
        cg_max_iter_base = self.sim_params.get("cg_max_iter", 500)

        # 6. Первая попытка решения CG
        P_new_flat, converged = self._solve_pressure_cg_pytorch(A, Q, M_diag=A_diag, tol=cg_tol_base, max_iter=cg_max_iter_base)

        # 7. При неуспехе пробуем ещё раз с расслабленными параметрами
        if not converged:
            print("  CG не сошёлся: увеличиваем max_iter и ослабляем tol")
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

        kro, krw = self.fluid.get_rel_perms(S_w_old)
        mu_o_pas = self.fluid.mu_oil
        mu_w_pas = self.fluid.mu_water

        mob_w = krw / mu_w_pas
        mob_o = kro / mu_o_pas
        mob_t = mob_w + mob_o

        # 1. Градиенты давления и апстрим мобильностей
        dp_x = P_new[:-1,:,:] - P_new[1:,:,:]
        dp_y = P_new[:,:-1,:] - P_new[:,1:,:]
        dp_z = P_new[:,:,:-1] - P_new[:,:,1:]

        mob_w_x = torch.where(dp_x > 0, mob_w[:-1,:,:], mob_w[1:,:,:])
        mob_w_y = torch.where(dp_y > 0, mob_w[:,:-1,:], mob_w[:,1:,:])
        mob_w_z = torch.where(dp_z > 0, mob_w[:,:,:-1], mob_w[:,:,1:])

        # 2. Потенциалы с учётом гравитации
        _, _, dz = self.reservoir.grid_size
        if dz > 0 and self.reservoir.nz > 1:
            rho_w_avg = 0.5 * (self.fluid.rho_w[:,:,:-1] + self.fluid.rho_w[:,:,1:])
            pot_z = dp_z + self.g * rho_w_avg * dz
        else:
            pot_z = dp_z

        # 3. Расходы воды
        flow_w_x = self.T_x * mob_w_x * dp_x
        flow_w_y = self.T_y * mob_w_y * dp_y
        flow_w_z = self.T_z * mob_w_z * pot_z

        # 4. Дивергенция
        div_flow = torch.zeros_like(S_w_old)
        div_flow[:-1, :, :] += flow_w_x
        div_flow[1:, :, :]  -= flow_w_x
        div_flow[:, :-1, :] += flow_w_y
        div_flow[:, 1:, :]  -= flow_w_y
        div_flow[:, :, :-1] += flow_w_z
        div_flow[:, :, 1:]  -= flow_w_z

        # 5. Источники/стоки воды от скважин
        q_w = torch.zeros_like(S_w_old)
        fw = mob_w / (mob_t + 1e-10)
        for well in self.well_manager.get_wells():
            i, j, k = well.i, well.j, well.k
            if i >= self.reservoir.nx or j >= self.reservoir.ny or k >= self.reservoir.nz:
                continue

            if well.control_type == 'rate':
                q_total = well.control_value / 86400.0 * (1 if well.type == 'injector' else -1)
                q_w[i, j, k] += q_total if well.type == 'injector' else q_total * fw[i, j, k]
            elif well.control_type == 'bhp':
                p_bhp = well.control_value * 1e6
                p_block = P_new[i, j, k]
                q_total = well.well_index * mob_t[i, j, k] * (p_block - p_bhp)
                q_w[i, j, k] += (-q_total) if well.type == 'injector' else (-q_total * fw[i, j, k])

        # 6. Обновление насыщенности с ограничением максимального изменения
        dSw = (dt / self.porous_volume) * (q_w - div_flow)
        max_dSw = self.sim_params.get("max_saturation_change", 0.05)
        dSw_clamped = dSw.clamp(-max_dSw, max_dSw)

        S_w_new = (S_w_old + dSw_clamped).clamp(self.fluid.sw_cr, 1.0 - self.fluid.so_r)

        self.fluid.s_w = S_w_new
        self.fluid.s_o = 1.0 - self.fluid.s_w

        affected_cells = torch.sum(torch.abs(dSw) > 1e-8).item()
        print(
            f"P̄ = {P_new.mean()/1e6:.2f} МПа, Sw(min/max) = {self.fluid.s_w.min():.3f}/{self.fluid.s_w.max():.3f}, ΔSw ограничено до ±{max_dSw}, ячеек изм.: {affected_cells}"
        )

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
            rho_w_z = torch.where(dp_z_prev > 0, self.fluid.rho_w[:,:,:-1], self.fluid.rho_w[:,:,1:])
            rho_o_z = torch.where(dp_z_prev > 0, self.fluid.rho_o[:,:,:-1], self.fluid.rho_o[:,:,1:])
            grav_flow = self.T_z * self.g * dz * (mob_w_z * rho_w_z + mob_o_z * rho_o_z)
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
                rate_si = well.control_value / 86400.0 * (1 if well.type == 'injector' else -1)
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

    def run(self, output_filename, save_vtk=False):
        """
        Запускает полную симуляцию.
        
        Args:
            output_filename: Имя файла для сохранения результатов
            save_vtk: Флаг для сохранения результатов в формате VTK
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
                
                time_info = f"День {int((i + 1) * time_step_days)}"
                filename = f"{output_filename}_step_{i+1}.png"
                filepath = os.path.join(intermediate_results_dir, filename)
                
                plotter.save_plots(p_current, sw_current, filepath, time_info=time_info)
        
        print("\nСимуляция завершена.")
        
        # Сохранение и визуализация финальных результатов
        p_final = self.fluid.pressure.cpu().numpy()
        sw_final = self.fluid.s_w.cpu().numpy()
        
        print(f"Итоговое давление: Мин={p_final.min()/1e6:.2f} МПа, Макс={p_final.max()/1e6:.2f} МПа")
        print(f"Итоговая водонасыщенность: Мин={sw_final.min():.4f}, Макс={sw_final.max():.4f}")
        
        # Сохранение числовых данных
        results_txt_path = os.path.join(results_dir, f"{output_filename}.txt")
        with open(results_txt_path, 'w') as f:
            f.write("Final Pressure (MPa):\n")
            f.write(np.array2string(p_final/1e6, threshold=np.inf, formatter={'float_kind':lambda x: "%.2f" % x}))
            f.write("\n\nFinal Water Saturation:\n")
            f.write(np.array2string(sw_final, threshold=np.inf, formatter={'float_kind':lambda x: "%.4f" % x}))
        print(f"Числовые результаты сохранены в файл {results_txt_path}")
        
        # Сохранение финальных графиков
        final_plot_path = os.path.join(results_dir, f"{output_filename}_final.png")
        plotter.save_plots(p_final, sw_final, final_plot_path, time_info=f"День {total_time_days} (Final)")
        print(f"Финальные графики сохранены в файл {final_plot_path}")
        
        # Сохранение результатов в VTK (если указано)
        if save_vtk:
            try:
                self._save_to_vtk(output_filename)
            except ImportError:
                print("Не удалось сохранить в формате VTK: отсутствуют необходимые модули.")
        
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
