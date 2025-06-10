import torch
import numpy as np
from scipy.sparse import csc_matrix, diags, bmat
from scipy.sparse.linalg import cg, LinearOperator, bicgstab, spsolve

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
            converged = self._newton_loop(current_dt)
            
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
        
    def _newton_loop(self, dt):
        """Одна попытка сделать шаг dt с циклом Ньютона."""
        P_old = self.fluid.pressure.clone()
        S_w_old = self.fluid.s_w.clone()
        
        P_k = P_old.clone()
        S_w_k = S_w_old.clone()
        
        newton_max_iter = self.sim_params.get("newton_max_iter", 15)
        newton_tol = self.sim_params.get("newton_tolerance", 1e-4)
        # Параметр регуляризации для плохо обусловленных матриц
        regularization = self.sim_params.get("jacobian_regularization", 1e-8)
        # Параметр релаксации для стабильности решения
        damping_factor = self.sim_params.get("damping_factor", 0.7)

        for k in range(newton_max_iter):
            # Строим систему уравнений
            R, J = self._build_residual_and_jacobian_sparse(P_k, S_w_k, P_old, S_w_old, dt)
            res_norm = np.linalg.norm(R)

            if k == 0: initial_res_norm = max(res_norm, 1e-10)
            relative_res = res_norm / initial_res_norm
            
            print(f"  Итерация Ньютона {k+1}: Невязка = {res_norm:.4e}, Отн. невязка = {relative_res:.4e}")
            
            # Проверка сходимости
            if relative_res < newton_tol or res_norm < 1e-6:
                print(f"  Сходимость достигнута за {k+1} итераций")
                self.fluid.pressure, self.fluid.s_w = P_k, S_w_k
                self.fluid.s_o = 1.0 - self.fluid.s_w  # Обновляем нефтенасыщенность
                return True

            # Регуляризация якобиана для лучшей обусловленности
            N = self.reservoir.nx * self.reservoir.ny * self.reservoir.nz
            if regularization > 0:
                # Добавляем небольшое значение к диагональным элементам
                J.setdiag(J.diagonal() + regularization * (1.0 + k/2) * np.ones(2*N))
            
            # Решаем систему линейных уравнений с увеличенной регуляризацией при необходимости
            try:
                dx = spsolve(J, -R)
            except Exception as e:
                print(f"  Ошибка решения СЛАУ: {e}")
                # Увеличиваем регуляризацию и пробуем еще раз
                J.setdiag(J.diagonal() + 100 * regularization * np.ones(2*N))
                try:
                    dx = spsolve(J, -R)
                except:
                    print("  Не удалось решить СЛАУ даже с усиленной регуляризацией")
                    return False
            
            if np.any(np.isnan(dx)):
                print("  Внимание: Решатель вернул NaN. Прерываем итерации.")
                return False
            
            # Извлекаем приращения для давления и насыщенности
            dp = torch.from_numpy(dx[:N]).view(self.reservoir.dimensions).to(self.device)
            dsw = torch.from_numpy(dx[N:]).view(self.reservoir.dimensions).to(self.device)
            
            # Применяем глобальное демпфирование для стабильности решения на начальных итерациях
            if k < 3:
                dp *= damping_factor
                dsw *= damping_factor
                print(f"  Применено демпфирование с коэффициентом {damping_factor}")

            # --- Улучшенный Line Search для выбора оптимального шага alpha ---
            best_alpha = None
            best_res_norm = float('inf')
            
            # Набор значений alpha для проверки (больше значений для лучшего поиска)
            alpha_values = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01]
            
            for alpha in alpha_values:
                P_k_new = P_k + alpha * dp
                S_w_k_new = S_w_k + alpha * dsw
                
                # Ограничиваем насыщенность физическими пределами
                S_w_k_new = torch.clamp(S_w_k_new, self.fluid.sw_cr, 1.0 - self.fluid.so_r)
                
                # Проверяем невязку с новыми значениями
                R_new, _ = self._build_residual_and_jacobian_sparse(P_k_new, S_w_k_new, P_old, S_w_old, dt)
                res_new_norm = np.linalg.norm(R_new)
                
                # Сохраняем лучший результат
                if res_new_norm < best_res_norm:
                    best_res_norm = res_new_norm
                    best_alpha = alpha
                
                # Если достигнуто достаточное уменьшение невязки, прерываем поиск
                if res_new_norm < 0.85 * res_norm:
                    break
            
            if best_alpha is None or best_res_norm >= res_norm:
                print("  Внимание: Line search не смог уменьшить невязку. Пробуем меньший шаг.")
                # Очень маленький шаг как последняя попытка
                best_alpha = 0.001
                P_k_new = P_k + best_alpha * dp
                S_w_k_new = S_w_k + best_alpha * dsw
                S_w_k_new = torch.clamp(S_w_k_new, self.fluid.sw_cr, 1.0 - self.fluid.so_r)
                
                R_new, _ = self._build_residual_and_jacobian_sparse(P_k_new, S_w_k_new, P_old, S_w_old, dt)
                best_res_norm = np.linalg.norm(R_new)
                
                if best_res_norm >= res_norm:
                    # Если мы близки к решению, но line search не помогает, попробуем принять решение
                    if res_norm < 1e-2:
                        print("  Невязка достаточно мала, принимаем текущее решение несмотря на трудности в line search")
                        self.fluid.pressure, self.fluid.s_w = P_k, S_w_k
                        self.fluid.s_o = 1.0 - self.fluid.s_w
                        return True
                    
                    print("  Не удалось уменьшить невязку даже с минимальным шагом. Итерации прекращены.")
                    return False
            
            # Принимаем новые значения
            P_k = P_k + best_alpha * dp
            S_w_k = torch.clamp(S_w_k + best_alpha * dsw, self.fluid.sw_cr, 1.0 - self.fluid.so_r)
            
            print(f"  Выбран шаг alpha = {best_alpha}, новая невязка = {best_res_norm:.4e}")
            
            # Прекращаем итерации, если улучшение слишком малое, но все же приближаемся к решению
            if k > 5 and best_res_norm > 0.95 * res_norm:
                print("  Прогресс слишком медленный. Проверяем близость к решению.")
                if best_res_norm < 5 * newton_tol * initial_res_norm:
                    # Принимаем результат, если невязка достаточно близка к допустимой
                    print("  Принимаем результат, несмотря на медленную сходимость.")
                    self.fluid.pressure, self.fluid.s_w = P_k, S_w_k
                    self.fluid.s_o = 1.0 - self.fluid.s_w
                    return True
                return False

        # Достигнуто максимальное число итераций
        if res_norm < 5 * newton_tol * initial_res_norm:
            # Принимаем результат, если невязка достаточно близка к допустимой
            print(f"  Принимаем результат после {newton_max_iter} итераций.")
            self.fluid.pressure, self.fluid.s_w = P_k, S_w_k
            self.fluid.s_o = 1.0 - self.fluid.s_w
            return True
            
        return False

    def _build_residual_and_jacobian_sparse(self, P, S_w, P_old, S_w_old, dt):
        nx, ny, nz = self.reservoir.dimensions
        N = nx * ny * nz
        
        # --- Свойства флюида и их производные (векторизованно) ---
        mu_w_pas, mu_o_pas = self.fluid.mu_w * 1e-3, self.fluid.mu_o * 1e-3
        kro, krw = self.fluid.get_rel_perms(S_w)
        mob_w, mob_o = krw / mu_w_pas, kro / mu_o_pas
        pc = self.fluid.get_capillary_pressure(S_w)
        
        # Вычисляем производные заранее
        dkro_dsw, dkrw_dsw = self.fluid.get_rel_perms_derivatives(S_w)
        dmobw_dsw, dmobo_dsw = dkrw_dsw / mu_w_pas, dkro_dsw / mu_w_pas
        dpc_dsw = self.fluid.get_capillary_pressure_derivative(S_w)
        
        # Переводим в numpy для быстрой обработки
        P_np = P.cpu().numpy()
        P_old_np = P_old.cpu().numpy()
        S_w_np = S_w.cpu().numpy()
        S_w_old_np = S_w_old.cpu().numpy()
        mob_w_np, mob_o_np = mob_w.cpu().numpy(), mob_o.cpu().numpy()
        dmobw_dsw_np, dmobo_dsw_np = dmobw_dsw.cpu().numpy(), dmobo_dsw.cpu().numpy()
        dpc_dsw_np = dpc_dsw.cpu().numpy()
        pc_np = pc.cpu().numpy()
        
        # --- Накопление и невязки ---
        # Предварительно вычисляем объём порового пространства разделенный на dt
        Vp_dt = (self.porous_volume / dt).cpu().numpy().flatten()
        # Вычисляем невязки для воды и нефти
        Rw = Vp_dt * (S_w_np.flatten() - S_w_old_np.flatten())
        Ro = Vp_dt * ((1 - S_w_np.flatten()) - (1 - S_w_old_np.flatten()))
        
        # --- Якобиан - накопление ---
        J_ws_diag = Vp_dt.copy()
        J_os_diag = -Vp_dt.copy()
        
        # Заготовки для диагональных элементов и внедиагональных блоков
        J_ww_diag = np.zeros(N)
        J_ow_diag = np.zeros(N)
        J_ww_off, J_ow_off, J_ws_off, J_os_off = [], [], [], []
        
        # Создаем индексные маски для всех направлений сразу
        nx_range = np.arange(nx).reshape(-1, 1, 1)
        ny_range = np.arange(ny).reshape(1, -1, 1)
        nz_range = np.arange(nz).reshape(1, 1, -1)
        
        # Для каждой оси обрабатываем потоки и якобиан
        for axis_idx, (T_ax, axis_size) in enumerate(zip([self.T_x, self.T_y, self.T_z], [nx, ny, nz])):
            if axis_size <= 1:  # Пропускаем оси размера 1
                continue
                
            T_ax_np = T_ax.cpu().numpy()
            # Создаем маску для границ
            slices_i = tuple(slice(None, -1) if i == axis_idx else slice(None) for i in range(3))
            slices_j = tuple(slice(1, None) if i == axis_idx else slice(None) for i in range(3))
            
            # Получаем плоские индексы для i и j ячеек
            grid_indices = np.stack(np.meshgrid(nx_range, ny_range, nz_range, indexing='ij'), axis=3)
            grid_indices = grid_indices.reshape(-1, 3)
            flat_indices = grid_indices[:, 0] * ny * nz + grid_indices[:, 1] * nz + grid_indices[:, 2]
            flat_indices_i = flat_indices.reshape(nx, ny, nz)[slices_i].flatten()
            flat_indices_j = flat_indices.reshape(nx, ny, nz)[slices_j].flatten()

            # Вычисляем разность давлений для воды и нефти
            dp_w = (P_np[slices_i] - P_np[slices_j]).flatten()
            dp_o = ((P_np + pc_np)[slices_i] - (P_np + pc_np)[slices_j]).flatten()
            
            # Определяем восходящие направления потоков
            up_w = dp_w > 0
            up_o = dp_o > 0
            
            # Вычисляем подвижности в восходящем направлении
            mob_w_up = np.where(up_w, mob_w_np[slices_i].flatten(), mob_w_np[slices_j].flatten())
            mob_o_up = np.where(up_o, mob_o_np[slices_i].flatten(), mob_o_np[slices_j].flatten())
            
            # Вычисляем потоки
            T_flat = T_ax_np.flatten()
            flow_w = T_flat * mob_w_up * dp_w
            flow_o = T_flat * mob_o_up * dp_o
            
            # Обновляем невязки (аккумулируем потоки)
            np.add.at(Rw, flat_indices_i, -flow_w)
            np.add.at(Rw, flat_indices_j, flow_w)
            np.add.at(Ro, flat_indices_i, -flow_o)
            np.add.at(Ro, flat_indices_j, flow_o)
            
            # --- Якобиан - вода/давление ---
            dFw_dPi = T_flat * mob_w_up
            dFw_dPj = -dFw_dPi
            np.add.at(J_ww_diag, flat_indices_i, -dFw_dPi)
            np.add.at(J_ww_diag, flat_indices_j, -dFw_dPj)
            J_ww_off.append((dFw_dPj, flat_indices_i, flat_indices_j))
            J_ww_off.append((dFw_dPi, flat_indices_j, flat_indices_i))
            
            # --- Якобиан - нефть/давление ---
            dFo_dPi = T_flat * mob_o_up
            dFo_dPj = -dFo_dPi
            np.add.at(J_ow_diag, flat_indices_i, -dFo_dPi)
            np.add.at(J_ow_diag, flat_indices_j, -dFo_dPj)
            J_ow_off.append((dFo_dPj, flat_indices_i, flat_indices_j))
            J_ow_off.append((dFo_dPi, flat_indices_j, flat_indices_i))
            
            # --- Якобиан - вода/насыщенность ---
            # Вычисляем производные подвижности воды по насыщенности с учетом восходящего направления
            dmobw_up_i = np.where(up_w, dmobw_dsw_np[slices_i].flatten(), 0)
            dmobw_up_j = np.where(up_w, 0, dmobw_dsw_np[slices_j].flatten())
            dFw_dSwi = T_flat * dmobw_up_i * dp_w
            dFw_dSwj = T_flat * dmobw_up_j * dp_w
            np.add.at(J_ws_diag, flat_indices_i, -dFw_dSwi)
            np.add.at(J_ws_diag, flat_indices_j, -dFw_dSwj)
            J_ws_off.append((dFw_dSwj, flat_indices_i, flat_indices_j))
            J_ws_off.append((dFw_dSwi, flat_indices_j, flat_indices_i))

            # --- Якобиан - нефть/насыщенность ---
            # Вычисляем производные подвижности нефти по насыщенности с учетом восходящего направления
            dmobo_up_i = np.where(up_o, dmobo_dsw_np[slices_i].flatten(), 0)
            dmobo_up_j = np.where(up_o, 0, dmobo_dsw_np[slices_j].flatten())
            
            # Важно: производная капиллярного давления имеет ОТРИЦАТЕЛЬНЫЙ знак
            dFo_dSwi = T_flat * (dmobo_up_i * dp_o + mob_o_up * dpc_dsw_np[slices_i].flatten())
            dFo_dSwj = T_flat * (dmobo_up_j * dp_o - mob_o_up * dpc_dsw_np[slices_j].flatten())
            np.add.at(J_os_diag, flat_indices_i, -dFo_dSwi)
            np.add.at(J_os_diag, flat_indices_j, -dFo_dSwj)
            J_os_off.append((dFo_dSwj, flat_indices_i, flat_indices_j))
            J_os_off.append((dFo_dSwi, flat_indices_j, flat_indices_i))

        # --- Скважины ---
        q_w, q_o, dqw_dp, dqw_dsw, dqo_dp, dqo_dsw = self._calculate_well_terms_fully_implicit(P, S_w, mob_w, mob_o, dmobw_dsw, dmobo_dsw)
        well_indices = self.well_manager.get_well_indices_flat().cpu().numpy()
        
        Rw[well_indices] -= q_w.cpu().numpy()
        Ro[well_indices] -= q_o.cpu().numpy()
        J_ww_diag[well_indices] -= dqw_dp.cpu().numpy()
        J_ws_diag[well_indices] -= dqw_dsw.cpu().numpy()
        J_ow_diag[well_indices] -= dqo_dp.cpu().numpy()
        J_os_diag[well_indices] -= dqo_dsw.cpu().numpy()

        # --- Сборка разреженной матрицы Якобиана ---
        def to_coo(diags_data, off_diags_data, shape):
            data, row, col = [diags_data], [np.arange(shape[0])], [np.arange(shape[1])]
            for d, i, j in off_diags_data:
                if len(d) > 0:  # Пропускаем пустые массивы
                    data.append(d)
                    row.append(i)
                    col.append(j)
            
            if len(data) > 1:
                return csc_matrix((np.concatenate(data), (np.concatenate(row), np.concatenate(col))), shape=shape)
            else:
                # Если только диагональные элементы
                return csc_matrix((data[0], (row[0], col[0])), shape=shape)

        # Сборка блоков Якобиана
        Jww = to_coo(J_ww_diag, J_ww_off, (N, N))
        Jws = to_coo(J_ws_diag, J_ws_off, (N, N))
        Jow = to_coo(J_ow_diag, J_ow_off, (N, N))
        Jos = to_coo(J_os_diag, J_os_off, (N, N))

        # Сборка полной матрицы Якобиана из блоков
        J = bmat([[Jww, Jws], [Jow, Jos]], format='csr')
        R = np.concatenate([Rw, Ro])
        
        return R, J

    def _calculate_well_terms_fully_implicit(self, P, S_w, mob_w, mob_o, dmobw_dsw, dmobo_dsw):
        """ 
        Рассчитывает источниковые члены и их производные от скважин для неявной схемы.
        Оптимизированная версия с векторизацией операций где возможно.
        """
        # Получаем общее количество скважин
        num_wells = self.well_manager.num_wells
        
        # Создаем тензоры для хранения результатов
        q_w = torch.zeros(num_wells, device=self.device)
        q_o = torch.zeros(num_wells, device=self.device)
        dqw_dp = torch.zeros(num_wells, device=self.device)
        dqw_dsw = torch.zeros(num_wells, device=self.device)
        dqo_dp = torch.zeros(num_wells, device=self.device)
        dqo_dsw = torch.zeros(num_wells, device=self.device)

        # Преобразуем тензоры в плоский вид для более быстрого доступа
        P_flat = P.view(-1)
        S_w_flat = S_w.view(-1)
        mob_w_flat = mob_w.view(-1)
        mob_o_flat = mob_o.view(-1)
        dmobw_dsw_flat = dmobw_dsw.view(-1)
        dmobo_dsw_flat = dmobo_dsw.view(-1)
        
        # Получаем списки скважин и их индексы
        wells = self.well_manager.get_wells()
        well_indices = [well.cell_index_flat for well in wells]
        
        # Предварительно извлекаем все значения для скважин
        p_wells = P_flat[well_indices]
        sw_wells = S_w_flat[well_indices]
        mob_w_wells = mob_w_flat[well_indices]
        mob_o_wells = mob_o_flat[well_indices]
        dmobw_dsw_wells = dmobw_dsw_flat[well_indices]
        dmobo_dsw_wells = dmobo_dsw_flat[well_indices]
        
        # Получаем типы скважин и их контроля для использования в векторизации
        well_types = [well.type for well in wells]
        control_types = [well.control_type for well in wells]
        control_values = torch.tensor([well.control_value for well in wells], device=self.device)
        well_indices_tensor = torch.tensor(well_indices, device=self.device)
        
        # Обрабатываем каждую скважину
        for i, well in enumerate(wells):
            if well.control_type == 'rate':
                # Конвертируем суточную скорость в м³/с
                rate_si = well.control_value / 86400.0
                
                if well.type == 'injector':
                    # Нагнетательная скважина: вся скорость идет на воду
                    q_w[i] = rate_si
                    q_o[i] = 0.0
                else:  # producer
                    # Добывающая скважина: распределяем поток по фазам
                    mob_t = mob_w_wells[i] + mob_o_wells[i]
                    fw = mob_w_wells[i] / (mob_t + 1e-9)  # Доля воды
                    q_w[i] = rate_si * fw
                    q_o[i] = rate_si * (1-fw)
                    
                    # Производные дебитов по насыщенности
                    dfw_dsw = (dmobw_dsw_wells[i] * mob_o_wells[i] - mob_w_wells[i] * dmobo_dsw_wells[i]) / (mob_t**2)
                    dqw_dsw[i] = rate_si * dfw_dsw
                    dqo_dsw[i] = -rate_si * dfw_dsw
                    
            elif well.control_type == 'bhp':
                # Скважина с контролем забойного давления
                p_bhp = well.control_value * 1e6  # МПа -> Па
                dp_val = p_wells[i] - p_bhp
                
                # Расчет дебитов фаз
                q_w[i] = well.well_index * mob_w_wells[i] * dp_val
                q_o[i] = well.well_index * mob_o_wells[i] * dp_val
                
                # Производные дебитов по давлению
                dqw_dp[i] = well.well_index * mob_w_wells[i]
                dqo_dp[i] = well.well_index * mob_o_wells[i]
                
                # Производные дебитов по насыщенности
                dqw_dsw[i] = well.well_index * dmobw_dsw_wells[i] * dp_val
                dqo_dsw[i] = well.well_index * dmobo_dsw_wells[i] * dp_val
                
        return q_w, q_o, dqw_dp, dqw_dsw, dqo_dp, dqo_dsw

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
