import torch
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg, LinearOperator

class Simulator:
    """
    Основной класс симулятора, отвечающий за выполнение расчетов по схеме IMPES
    (Implicit Pressure, Explicit Saturation).
    """
    def __init__(self, reservoir, fluid_system, well_manager):
        """ Инициализация симулятора """
        self.reservoir = reservoir
        self.fluid = fluid_system
        self.well_manager = well_manager
        self.device = reservoir.device

        # Конвертируем проницаемость из мД в м^2 (1 мД = 1e-15 м^2)
        perm_m2 = reservoir.permeability * 1e-15
        
        # Рассчитываем геометрический фактор и гармоническое среднее проницаемости
        dx, dy, dz = reservoir.grid_size
        ax, ay, az = dy*dz, dx*dz, dx*dy
        
        k_x_h = 2 * perm_m2[1:,:,:] * perm_m2[:-1,:,:] / (perm_m2[1:,:,:] + perm_m2[:-1,:,:] + 1e-20)
        k_y_h = 2 * perm_m2[:,1:,:] * perm_m2[:,:-1,:] / (perm_m2[:,1:,:] + perm_m2[:,:-1,:] + 1e-20)
        k_z_h = 2 * perm_m2[:,:,1:] * perm_m2[:,:,:-1] / (perm_m2[:,:,1:] + perm_m2[:,:,:-1] + 1e-20)

        # Рассчитываем проводимость (трансмиссивность) по осям
        self.T_x = (ax / dx) * k_x_h
        self.T_y = (ay / dy) * k_y_h
        self.T_z = (az / dz) * k_z_h

        # Сохраняем пористый объем (тензор)
        self.porous_volume = reservoir.porous_volume
        self.g = 9.81 # Ускорение свободного падения

    def run_step(self, dt):
        """ Выполняет один временной шаг симуляции """
        P_new, converged = self._implicit_pressure_step(dt)
        assert converged, "Решатель давления не сошелся!"
        self._explicit_saturation_step(P_new, dt)

    def _implicit_pressure_step(self, dt):
        """ Неявный шаг для расчета давления """
        P_prev = self.fluid.pressure
        S_w = self.fluid.s_w

        # 1. Рассчитываем ОФП и подвижности
        kro, krw = self.fluid.get_rel_perms(S_w)
        mu_o_pas = self.fluid.mu_o * 1e-3
        mu_w_pas = self.fluid.mu_w * 1e-3
        mob_w = krw / mu_w_pas
        mob_o = kro / mu_o_pas
        mob_t = mob_w + mob_o

        # 2. Рассчитываем проводимости с upwind
        dp_x_prev = P_prev[:-1,:,:] - P_prev[1:,:,:]
        dp_y_prev = P_prev[:,:-1,:] - P_prev[:,1:,:]
        dp_z_prev = P_prev[:,:,:-1] - P_prev[:,:,1:]
        mob_t_x = torch.where(dp_x_prev > 0, mob_t[:-1,:,:], mob_t[1:,:,:])
        mob_t_y = torch.where(dp_y_prev > 0, mob_t[:,:-1,:], mob_t[:,1:,:])
        mob_t_z = torch.where(dp_z_prev > 0, mob_t[:,:,:-1], mob_t[:,:,1:])
        Tx_t = self.T_x * mob_t_x
        Ty_t = self.T_y * mob_t_y
        Tz_t = self.T_z * mob_t_z
        
        # 3. Собираем матрицу A и правую часть Q
        A, A_diag = self._build_pressure_matrix_vectorized(Tx_t, Ty_t, Tz_t, dt)
        Q = self._build_pressure_rhs(dt, P_prev)
        
        # 4. Решаем СЛАУ
        P_new_flat, converged = self._solve_pressure_pcg(A, A_diag, Q)
        P_new = P_new_flat.view(self.reservoir.dimensions)
        self.fluid.pressure = P_new
        return P_new, converged

    def _explicit_saturation_step(self, P_new, dt):
        """ Явный шаг для обновления насыщенности """
        S_w = self.fluid.s_w
        nx, ny, nz = self.reservoir.dimensions

        # Рассчитываем подвижности заново, так как они нужны для fw
        kro, krw = self.fluid.get_rel_perms(S_w)
        mu_o_pas = self.fluid.mu_o * 1e-3
        mu_w_pas = self.fluid.mu_w * 1e-3
        mob_w = krw / mu_w_pas
        mob_o = kro / mu_o_pas
        mob_t = mob_w + mob_o
        
        # Рассчитываем потоки воды
        dp_x = P_new[:-1,:,:] - P_new[1:,:,:]
        dp_y = P_new[:,:-1,:] - P_new[:,1:,:]
        dp_z = P_new[:,:,:-1] - P_new[:,:,1:]
        mob_w_x = torch.where(dp_x > 0, mob_w[:-1,:,:], mob_w[1:,:,:])
        mob_w_y = torch.where(dp_y > 0, mob_w[:,:-1,:], mob_w[:,1:,:])
        mob_w_z = torch.where(dp_z > 0, mob_w[:,:,:-1], mob_w[:,:,1:])
        flow_w_x = self.T_x * mob_w_x * dp_x
        flow_w_y = self.T_y * mob_w_y * dp_y
        flow_w_z = self.T_z * mob_w_z * dp_z

        # Рассчитываем дивергенцию
        div_flow = torch.zeros_like(S_w)
        div_flow[:-1, :, :] += flow_w_x
        div_flow[1:, :, :]  -= flow_w_x
        div_flow[:, :-1, :] += flow_w_y
        div_flow[:, 1:, :]  -= flow_w_y
        div_flow[:, :, :-1] += flow_w_z
        div_flow[:, :, 1:]  -= flow_w_z
        
        # Учет скважин
        q_w = torch.zeros_like(S_w)
        fw = mob_w / (mob_t + 1e-10)
        for well in self.well_manager.get_wells():
            i, j, k = well.i, well.j, well.k
            if well.well_type == 'injector':
                q_w[i,j,k] += well.rate_si
            elif well.well_type == 'producer':
                q_w[i,j,k] += well.rate_si * fw[i, j, k]

        # Обновление
        S_w_new = S_w + (dt / self.porous_volume) * (q_w - div_flow)
        self.fluid.s_w = S_w_new.clamp(self.fluid.sw_cr, 1.0 - self.fluid.so_r)
        self.fluid.s_o = 1.0 - self.fluid.s_w

        # Логирование
        affected_cells = torch.sum(self.fluid.s_w > self.fluid.sw_cr + 1e-5).item()
        print(f"Давление (ср): {P_new.mean()/1e6:.2f} МПа. Насыщенность (мин/макс): {self.fluid.s_w.min():.3f}/{self.fluid.s_w.max():.3f}. Ячеек затронуто: {affected_cells}")

    def _build_pressure_matrix_vectorized(self, Tx, Ty, Tz, dt):
        """ Векторизованная сборка матрицы давления """
        nx, ny, nz = self.reservoir.dimensions
        N = nx * ny * nz
        
        # --- Собираем внедиагональные элементы ---

        # Общий массив индексов для создания масок
        row_indices_all = torch.arange(N, device=self.device)

        # Связи по X
        mask_x = (row_indices_all // (ny * nz)) < (nx - 1)
        row_x = row_indices_all[mask_x]
        col_x = row_x + ny * nz
        vals_x = Tx.flatten() # Tx уже имеет правильный размер (nx-1, ny, nz)
        
        # Связи по Y
        mask_y = (row_indices_all // nz) % ny < (ny - 1)
        row_y = row_indices_all[mask_y]
        col_y = row_y + nz
        vals_y = Ty.flatten() # Ty уже имеет правильный размер (nx, ny-1, nz)

        # Связи по Z
        mask_z = (row_indices_all % nz) < (nz - 1)
        row_z = row_indices_all[mask_z]
        col_z = row_z + 1
        vals_z = Tz.flatten() # Tz уже имеет правильный размер (nx, ny, nz-1)

        # Собираем индексы и значения для внедиагональных элементов (верхний и нижний треугольники)
        rows = torch.cat([row_x, col_x, row_y, col_y, row_z, col_z])
        cols = torch.cat([col_x, row_x, col_y, row_y, col_z, row_z])
        vals = torch.cat([-vals_x, -vals_x, -vals_y, -vals_y, -vals_z, -vals_z])

        # --- Собираем диагональные элементы ---
        # Член накопления C = Vp*c/dt
        acc_term = (self.porous_volume.view(-1) * self.fluid.cf.view(-1) / dt)
        
        # Собираем диагональ
        diag_vals = torch.zeros(N, device=self.device)
        diag_vals.scatter_add_(0, rows, -vals) # Сумма проводимостей от соседей
        diag_vals += acc_term # Добавляем член накопления

        # --- Собираем итоговую матрицу ---
        
        # Добавляем диагональ к остальным элементам
        final_rows = torch.cat([rows, torch.arange(N, device=self.device)])
        final_cols = torch.cat([cols, torch.arange(N, device=self.device)])
        final_vals = torch.cat([vals, diag_vals])

        return torch.sparse_coo_tensor(torch.stack([final_rows, final_cols]), final_vals, (N, N)), diag_vals

    def _build_pressure_rhs(self, dt, P_prev):
        """ Собирает правую часть Q для СЛАУ A*P_new = Q """
        nx, ny, nz = self.reservoir.dimensions
        N = nx * ny * nz

        # 1. Член сжимаемости (теперь использует тензор пористого объема)
        compressibility_term = (self.porous_volume.view(-1) * self.fluid.cf.view(-1) / dt) * P_prev.view(-1)

        # 2. Член скважин
        q_wells = torch.zeros(N, device=self.device)
        for well in self.well_manager.get_wells():
            q_wells[well.cell_idx] += well.rate_si # Просто прибавляем дебит (он уже со знаком)
        
        # Правая часть Q = q_wells + acc_term * P_old
        return q_wells + compressibility_term

    def _solve_pressure_pcg(self, A, A_diag, b, max_iter=1000, tol=1e-5):
        """
        Решает СЛАУ Ax=b с помощью метода сопряженных градиентов из SciPy.
        """
        # 0. Объединяем дублирующиеся индексы в разреженном тензоре
        A = A.coalesce()
        
        # 1. Переносим данные с GPU на CPU и конвертируем в NumPy/SciPy формат
        A_cpu = A.cpu()
        indices = A_cpu.indices().numpy()
        values = A_cpu.values().numpy()
        n = A_cpu.shape[0]
        
        A_scipy = csc_matrix((values, (indices[0], indices[1])), shape=(n, n))
        b_numpy = b.cpu().numpy()

        # Диагональный предобуславливатель для SciPy
        M_inv_numpy = 1.0 / A_diag.cpu().numpy()
        preconditioner = LinearOperator((n, n), matvec=lambda v: M_inv_numpy * v)
        
        # 2. Вызываем решатель SciPy
        x_numpy, exit_code = cg(A_scipy, b_numpy, M=preconditioner, atol=tol, maxiter=max_iter)
        
        if exit_code == 0:
            print(f"  SciPy CG решатель сошелся.")
        else:
            print(f"  SciPy CG решатель не сошелся (код: {exit_code}).")

        # 3. Конвертируем решение обратно в тензор PyTorch и возвращаем результат
        P_new_flat = torch.from_numpy(x_numpy).to(self.device)
        
        return P_new_flat, exit_code == 0
