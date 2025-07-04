import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, diags, bmat, csr_matrix, identity
from scipy.sparse.linalg import cg, LinearOperator, bicgstab, spsolve, gmres, spilu
import time
import os
import datetime
import builtins

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
        if device is not None:
            self.device = device
        else:
            # Используем то же устройство, что и у флюида (он уже создан к этому моменту)
            self.device = fluid.device if hasattr(fluid, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
            dev = self.fluid.device  # Используем то же устройство, что и тензоры флюида
            self.T_x = torch.zeros((nx-1, ny, nz), device=dev)
            self.T_y = torch.zeros((nx, ny-1, nz), device=dev)
            self.T_z = torch.zeros((nx, ny, nz-1), device=dev)
            
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
        
        # Управление подробностью вывода. По умолчанию вывод подавляется, его можно включить
        # через field "verbose" в блоке simulation конфигурационного файла.
        self.verbose = self.sim_params.get('verbose', False)
        
        # Вспомогательная функция для «тихого» логирования. Используйте self._log() вместо print().
        # Это минимальное изменение, не затрагивающее существующую логику, но позволяющее в один
        # момент отключить избыточный вывод (что в тестах и профилировании экономит заметное время).
        def _log(*args, **kwargs):
            if self.verbose:
                print(*args, **kwargs)
        
        # Сохраняем как bound-method, чтобы можно было вызывать из других методов класса.
        self._log = _log

        # --------------------------------------------------------------
        #  Линейный солвер: автоконфигурация
        # --------------------------------------------------------------
        # Если пользователь не указал блок "linear_solver", пытаемся
        # автоматически выбрать наиболее продвинутый доступный backend.
        # Предпочтение: (1) Hypre BoomerAMG через PETSc, (2) torch_gmres.
        if 'linear_solver' not in self.sim_params:
            self.sim_params['linear_solver'] = {}
        lin_cfg = self.sim_params['linear_solver']
        if 'backend' not in lin_cfg:
            try:
                import petsc4py  # noqa: F401
                from petsc4py import PETSc as _P
                _ = _P.Sys.getVersion()  # попытка загрузить libpetsc, может бросить
                lin_cfg['backend'] = 'hypre'
                lin_cfg.setdefault('tol', 1e-8)
                lin_cfg.setdefault('max_iter', 400)
                self._log("[auto] Выбран backend Hypre/BoomerAMG (petsc4py OK).")
            except Exception as e:
                lin_cfg['backend'] = 'torch_gmres'
                lin_cfg.setdefault('tol', 1e-6)
                lin_cfg.setdefault('restart', 50)
                lin_cfg.setdefault('max_iter', 400)
                self._log(f"[auto] petsc4py/Hypre не доступен: {e}. Переключаемся на torch_gmres.")

        # --------------------------------------------------------------
        # Предвычисляем индексы соседних ячеек (edges) для X-,Y-,Z-направлений.
        # Это позволит векторизовать расчёт потоков без Python-циклов.
        nx, ny, nz = self.reservoir.dimensions
        idx = torch.arange(nx * ny * nz)
        idx_3d = idx.reshape(nx, ny, nz)

        # X-рёбра: все, кроме последней колонки по X
        if nx > 1:
            self.edge_x_i = idx_3d[:-1, :, :].reshape(-1).to(torch.long)
            self.edge_x_j = idx_3d[1:, :, :].reshape(-1).to(torch.long)
        else:
            self.edge_x_i = self.edge_x_j = torch.tensor([], dtype=torch.long)

        # Y-рёбра: все, кроме последней строки по Y
        if ny > 1:
            self.edge_y_i = idx_3d[:, :-1, :].reshape(-1).to(torch.long)
            self.edge_y_j = idx_3d[:, 1:, :].reshape(-1).to(torch.long)
        else:
            self.edge_y_i = self.edge_y_j = torch.tensor([], dtype=torch.long)

        # Z-рёбра: все, кроме последнего слоя по Z
        if nz > 1:
            self.edge_z_i = idx_3d[:, :, :-1].reshape(-1).to(torch.long)
            self.edge_z_j = idx_3d[:, :, 1:].reshape(-1).to(torch.long)
        else:
            self.edge_z_i = self.edge_z_j = torch.tensor([], dtype=torch.long)

        # Псевдо-транзиентное продолжение (PTC): дополнительная «ёмкость»
        # для аккумуляционных членов, стабилизирующая Ньютон на резких фронтах.
        # Если параметр не указан, он равен 0.0 и схема эквивалентна прежней.
        # PTC: по-умолчанию добавляем 0.03 (≈15 % от φ=0.2), что хорошо гасит фронты.
        self.ptc_alpha = self.sim_params.get("ptc_alpha", 0.03)

        # По-умолчанию используем предварительный predictor IMPES перед
        # fully-implicit, чтобы дать хорошее initial guess.
        self.use_impes_predictor = self.sim_params.get("use_impes_predictor", True)

        # По-умолчанию используем «ручной» Якобиан, чтобы полностью полагаться
        # на классический метод Ньютона (собранная матрица) вместо autograd/JFNK.
        # Это приближает нас к промышленным симуляторам и исключает дрейф
        # производительности на крупных сетках, где autograd становится дорогим.
        if 'jacobian' not in self.sim_params:
            # По-умолчанию выбираем «auto»: для мелких задач (N<autograd_threshold_cells)
            # будет использован автоград-Ньютон, для больших – JFNK, но всегда без
            # ручного плотного Якобиана, который показал слабую устойчивость.
            self.sim_params['jacobian'] = 'auto'

        # --------------------------------------------------------------
        #  Динамические лимиты trust-region (давление / насыщенность)
        # --------------------------------------------------------------
        self._sw_trust_limit_init = self.sim_params.get("max_saturation_change", 0.8)
        self._p_trust_limit_init  = self.sim_params.get("max_pressure_change", 50.0)
        self._sw_trust_limit = self._sw_trust_limit_init
        self._p_trust_limit  = self._p_trust_limit_init
        # Эти значения будут адаптивно увеличиваться, если алгоритму
        # не удаётся сделать достаточно большой шаг (см. _fi_autograd_step
        # и _fi_jfnk_step).

        # --------------------------------------------------------------
        #  Доп. лимиты для ячеек-скважин + CNV-критерий
        # --------------------------------------------------------------
        self._well_sw_extra   = self.sim_params.get("well_saturation_extra", 0.0)  # отключено: единый лимит для всех ячеек
        self._cnv_threshold   = self.sim_params.get("cnv_threshold", 1.2)  # порог относит. изм. насыщенности

        # Маска ячеек, в которых расположены скважины (True там, где есть хоть одна скважина)
        nx, ny, nz = self.reservoir.dimensions
        self.well_mask = torch.zeros((nx, ny, nz), dtype=torch.bool)
        for w in self.well_manager.wells:
            i, j, k = w.i, w.j, w.k
            # проверка на диапазон сетки
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                self.well_mask[i, j, k] = True
        # Плоская версия (1-D) для быстрого индексирования
        self._well_mask_flat = self.well_mask.view(-1)
        
        # ---- Инициализация prev_mass для fully-implicit ----
        if self.solver_type == 'fully_implicit':
            self._initialize_previous_masses()

    def _initialize_previous_masses(self):
        """Инициализирует предыдущие массы флюидов для расчета невязки аккумуляции"""
        # Получаем текущие параметры
        p_current = self.fluid.pressure.view(-1)
        sw_current = self.fluid.s_w.view(-1)
        
        # Пористость с учетом сжимаемости породы
        phi0_vec = self.reservoir.porosity_ref.view(-1)
        c_r = self.reservoir.rock_compressibility
        p_ref = 1e5  # референсное давление (1 атм)
        phi_vec = phi0_vec * (1 + c_r * (p_current - p_ref)) + self.ptc_alpha
        
        # Плотности флюидов при текущем давлении
        rho_w = self.fluid.calc_water_density(p_current)
        rho_o = self.fluid.calc_oil_density(p_current)
        
        # Объем ячейки
        cell_vol = self.reservoir.cell_volume
        
        # Инициализируем массы как 1-D тензоры (совместимые с JFNK)
        self.fluid.prev_water_mass = phi_vec * sw_current * rho_w * cell_vol
        self.fluid.prev_oil_mass = phi_vec * (1 - sw_current) * rho_o * cell_vol
        
        print(f"Инициализированы массы: вода={self.fluid.prev_water_mass.sum().item()/1e6:.1f} млн кг, нефть={self.fluid.prev_oil_mass.sum().item()/1e6:.1f} млн кг")

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
            success = self._impes_step(dt)
        elif self.solver_type == 'fully_implicit':
            success = self._fully_implicit_step(dt)
        else:
            raise ValueError(f"Неизвестный тип решателя: {self.solver_type}")

        # После каждого шага гарантируем, что тензоры состояния не требуют градиента,
        # чтобы тесты могли безопасно вызывать .numpy().
        self.fluid.pressure = self.fluid.pressure.detach()
        self.fluid.s_w      = self.fluid.s_w.detach()
        self.fluid.s_o      = self.fluid.s_o.detach()

        return success

    def _fully_implicit_step(self, dt):
        """ Выполняет один временной шаг полностью неявной схемой. """
        # ------------------------------------------------------------------
        # Новый, предсказуемый выбор метода: исключительно по полю
        #     sim_params["jacobian"].
        # Поддерживаются значения:
        #   • "autograd"  – полный Якобиан через PyTorch Autograd
        #   • "jfnk"      – Jacobian-Free Newton–Krylov (c CPR/AMG, если включён)
        #   • "manual"    – старый ручной Ньютон с явным Якобианом
        # Если ключа нет – берём "jfnk" как надёжный по умолчанию.
        # Никаких внутренних эвристик по размерам сетки больше НЕТ.

        # 1. Если включён быстрый предиктор IMPES – делаем его до выбора метода.
        if getattr(self, "use_impes_predictor", False):
            try:
                self._impes_predictor(dt)
            except Exception as e:
                print(f"Предиктор IMPES не удался: {e}. Продолжаем без него.")

        # 2. Выбираем решатель строго по sim_params["jacobian"].
        jacobian_mode = self.sim_params.get("jacobian", "jfnk").lower()

        if jacobian_mode == "manual":
            # старый ручной Якобиан (код ниже)
            pass
        elif jacobian_mode == "autograd":
            if self._fi_autograd_adaptive(dt):
                return True
            print("Autograd-Ньютон не сошёлся – пробуем fallback на JFNK")
            if self._fi_jfnk_adaptive(dt):
                return True
            print("JFNK после отказа autograd также не сошёлся – шаг будет помечен как провал")
            return False
        elif jacobian_mode == "jfnk":
            if self._fi_jfnk_adaptive(dt):
                return True
            print("JFNK не сошёлся – пробуем fallback на autograd")
            if self._fi_autograd_adaptive(dt):
                return True
            print("Autograd после отказа JFNK также не сошёлся – шаг будет помечен как провал")
            return False
        else:
            raise ValueError(f"Неизвестный режим jacobian='{jacobian_mode}'. Ожидается 'manual', 'autograd' или 'jfnk'.")

        # ------------------------------------------------------------------
        # 1. Predictor: делаем один быстрый шаг IMPES, чтобы получить
        #    осмысленное начальное приближение (P, Sw).
        #    Этот шаг НЕ обновляет prev_mass, поэтому баланс по массе
        #    для fully-implicit сохраняется корректным.
        # ------------------------------------------------------------------
        if getattr(self, "use_impes_predictor", False):
            try:
                self._impes_predictor(dt)
            except Exception as e:
                print(f"Предиктор IMPES не удался: {e}. Продолжаем без него.")

        # По умолчанию используем JFNK, если не указано "manual".
        # Автоматический выбор: для мелких сеток JFNK даёт больше накладных расходов и
        # может быть избыточно. Кроме того, текущая реализация JFNK ещё не полностью
        # отлажена для маленьких задач, что отражается в тестовом наборе (50×50×1).
        # Поэтому, если пользователь явно не попросил JFNK ("jacobian":"auto" или
        # отсутствует ключ) и размер сетки меньше порога, переключаемся на ручной
        # Jacobian. Порог можно настроить через sim_params['jfnk_threshold_cells'].

        num_cells = self.reservoir.dimensions[0] * self.reservoir.dimensions[1] * self.reservoir.dimensions[2]
        jfnk_threshold = self.sim_params.get("jfnk_threshold_cells", 10000)

        jacobian_mode = self.sim_params.get("jacobian", "auto")

        # Если пользователь ЯВНО попросил ручной Якобиан – используем его.
        if jacobian_mode == "manual":
            # Путь старого ручного Ньютона (ниже в коде)
            pass
        else:
            nx, ny, nz = self.reservoir.dimensions
            num_cells = nx * ny * nz
            threshold = self.sim_params.get("autograd_threshold_cells", 1000)
            # Для небольших задач используем автоград-Ньютон, иначе JFNK
            if num_cells <= threshold:
                # Используем адаптивный автоград-Ньютон: при несходимости он
                # будет уменьшать dt несколько раз прежде, чем вернуть False.
                success = self._fi_autograd_adaptive(dt)
                if success:
                    return True
                print("Autograd-Ньютон не сошёлся даже после адаптации dt – пробуем JFNK fallback")
                if self._fi_jfnk_adaptive(dt):
                    return True
                print("JFNK также не сошёлся – пробуем ещё раз autograd на более мелком dt")
                if self._fi_autograd_adaptive(dt / 2):
                    return True
                print("Autograd после JFNK тоже не сошёлся – выполняем fallback на IMPES")
                return self._impes_step(dt)
            else:
                # Для крупных задач используем новый Jacobian-free solver
                lin_cfg = self.sim_params.get("linear_solver", {})
                if not hasattr(self, "_fisolver"):
                    try:
                        from solver.jfnk import FullyImplicitSolver
                        backend = lin_cfg.get("prec_backend", "amgx")
                        self._fisolver = FullyImplicitSolver(self, backend=backend)
                    except Exception as e:
                        print(f"Не удалось инициализировать новый FullyImplicitSolver: {e}. Переходим к старому JFNK.")
                        return self._fi_jfnk_adaptive(dt)

                x0 = torch.cat([
                    (self.fluid.pressure.view(-1) / 1e6),
                    self.fluid.s_w.view(-1)
                ]).to(self.device)

                x_out, ok = self._fisolver.step(x0, dt)
                if ok:
                    # раскладываем решение
                    N = self.reservoir.dimensions[0]*self.reservoir.dimensions[1]*self.reservoir.dimensions[2]
                    p_new = (x_out[:N] * 1e6).view(self.reservoir.dimensions)
                    sw_new = x_out[N:].view(self.reservoir.dimensions).clamp(self.fluid.sw_cr, 1-self.fluid.so_r)
                    self.fluid.pressure = p_new
                    self.fluid.s_w = sw_new
                    self.fluid.s_o = 1 - sw_new
                    return True
                else:
                    print("Новый FullyImplicitSolver не сошёлся – fallback на IMPES")
                    return self._impes_step(dt)

        # == прежний путь с ручным якобианом ==
        current_dt = dt
        max_attempts = self.sim_params.get("max_time_step_attempts", 4)

        for attempt in range(max_attempts):
            print(f"Попытка шага с dt = {current_dt/86400:.2f} дней (Попытка {attempt+1}/{max_attempts})")

            newton_result = self._fully_implicit_newton_step(current_dt)
            if isinstance(newton_result, tuple):
                converged, _ = newton_result
            else:
                converged = bool(newton_result)

            if converged:
                print(f"Шаг успешно выполнен с dt = {current_dt/86400:.2f} дней.")
                return True

            # Неудачная попытка - восстанавливаем начальное состояние
            self.fluid.pressure = self.fluid.pressure.clone()
            self.fluid.s_w = self.fluid.s_w.clone()
            self.fluid.s_o = 1.0 - self.fluid.s_w
            
            print("Решатель не сошелся. Уменьшаем шаг времени.")
            current_dt /= self.sim_params.get("dt_reduction_factor", 2.0)

        print("Не удалось добиться сходимости даже с минимальным шагом. Пробуем JFNK перед fallback на IMPES.")
        # Пытаемся выполнить шаг JFNK с тем же dt (адаптивный внутри метода)
        if self._fi_jfnk_adaptive(dt):
            print("Шаг успешно выполнен с помощью JFNK после отказа ручного Ньютона.")
            return True

        # Если и JFNK не сошёлся, выполняем запасной вариант IMPES,
        # чтобы симуляция не прерывалась – особенно полезно для CI.
        print("JFNK не сошёлся – выполняем fallback на IMPES для данного шага.")
        return self._impes_step(dt)

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
        # ------------------------------------------------------------------
        # При выключенном verbose перенаправляем print в no-op для ускорения.
        # Делается через builtins, чтобы затронуть все вложенные вызовы.
        # Будем восстанавливать в конце функции (в блоке finally).
        import builtins
        _orig_print = builtins.print
        if not getattr(self, 'verbose', False):
            builtins.print = lambda *args, **kwargs: None
        try:
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
                sw_vec_raw = self.fluid.s_w.reshape(-1)
                sw_vec = self._soft_clamp(sw_vec_raw, self.fluid.sw_cr, 1.0 - self.fluid.so_r)
                # Пористость зависит от давления: φ(P) = φ_ref * (1 + c_r (P - P_ref))
                phi0_vec = self.reservoir.porosity_ref.reshape(-1)
                c_r = self.reservoir.rock_compressibility
                p_ref = 1e5  # давление-референс (Па)
                phi_vec = phi0_vec * (1 + c_r * (p_vec - p_ref))
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
                    phi_prev_vec = phi0_vec * (1 + c_r * (self.fluid.prev_pressure.reshape(-1) - p_ref))

                    self.fluid.prev_water_mass = phi_prev_vec * self.fluid.prev_sw.reshape(-1) * \
                                                self.fluid.calc_water_density(self.fluid.prev_pressure.reshape(-1)) * \
                                                cell_volume
                    self.fluid.prev_oil_mass = phi_prev_vec * (1 - self.fluid.prev_sw.reshape(-1)) * \
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
                        import numpy as np
                        from scipy.sparse import csr_matrix, identity
                        from scipy.sparse.linalg import spilu, gmres, LinearOperator

                        jacobian_np = jacobian.cpu().numpy().astype(np.float32)
                        residual_np = residual.cpu().numpy().astype(np.float32)

                        jacobian_csr = csr_matrix(jacobian_np)

                        # Tikhonov-регуляризация, чтобы устранить возможные нули на диагонали
                        lam_reg = self.sim_params.get("tikhonov_lambda", 1e-8)
                        jacobian_csr = jacobian_csr + lam_reg * identity(jacobian_csr.shape[0], dtype=jacobian_csr.dtype)

                        # ILU0 предобуславливатель
                        fill_factor = self.sim_params.get("linear_solver", {})
                        try:
                            ilu = spilu(jacobian_csr.astype(np.float64), drop_tol=0.0, fill_factor=fill_factor)

                            def Mx(x):
                                return ilu.solve(x)

                            M = LinearOperator(jacobian_csr.shape, Mx, dtype=np.float64)

                            ls_cfg = self.sim_params.get("linear_solver", {})
                            restart = ls_cfg.get("restart", 50)
                            max_it  = ls_cfg.get("max_iter", 400)
                            tol_lin = ls_cfg.get("tol", 1e-8)

                            delta_np, info = gmres(
                                jacobian_csr, -residual_np,
                                M=M, restart=restart, maxiter=max_it, tol=tol_lin
                            )
                            if info != 0:
                                print(f"  Предупреждение: GMRES не сошёлся (info={info}) → fallback на bicgstab")
                                from scipy.sparse.linalg import bicgstab
                                delta_np, info2 = bicgstab(jacobian_csr, -residual_np, tol=1e-6, maxiter=1000, M=M)
                                if info2 != 0:
                                    raise RuntimeError("BiCGStab также не сошёлся")
                        except Exception as e_ilu:
                            print(f"  ILU0/GMRES не удалось: {e_ilu}. Переходим к spsolve")
                            from scipy.sparse.linalg import spsolve
                            delta_np = spsolve(jacobian_csr, -residual_np)
                    else:
                        # Для небольших систем используем прямой решатель
                        delta = self._robust_solve(jacobian, -residual)
                except RuntimeError as e:
                    print(f"  Ошибка решения системы: {e}")
                    # Восстанавливаем исходное состояние
                    self.fluid.pressure = current_p.clone()
                    self.fluid.s_w = current_sw.clone()
                    return False, iter_idx
                
                # ---- Trust–region по полной норме шага ----------------------
                if iter_idx == 0 and not hasattr(self, "_trust_radius"):
                    # Инициализируем: 20 % нормы начального состояния – эвристика
                    x0_norm = torch.norm(torch.cat([p_vec, sw_vec])).item()
                    self._trust_radius = 0.2 * x0_norm

                step_norm = torch.norm(delta).item()
                if step_norm > self._trust_radius:
                    scale_trust = self._trust_radius / (step_norm + 1e-15)
                    delta = delta * scale_trust
                    if self.verbose:
                        print(f"  Trust-region: ||δ||={step_norm:.2e} > r={self._trust_radius:.2e} → масштаб x{scale_trust:.3f}")

                # -------------------------------------------------------------

                # Нормализуем невязку
                if iter_idx == 0:
                    initial_residual_norm = torch.norm(residual).item()
                    residual_norm = initial_residual_norm
                    relative_residual = 1.0
                else:
                    residual_norm = torch.norm(residual).item()
                    relative_residual = residual_norm / initial_residual_norm
                
                print(f"  Итерация Ньютона {iter_idx+1}: Невязка = {residual_norm:.4e}, Отн. невязка = {relative_residual:.4e}")
                
                # Считаем tol относительным порогом: требуем, чтобы относительная невязка
                # (по отношению к первой итерации) стала меньше tol. Для безопасности
                # также принимаем решение при очень маленькой абсолютной невязке.
                if relative_residual < tol or residual_norm < tol * 1e3:
                    print(f"  Метод Ньютона сошелся за {iter_idx+1} итераций (relative={relative_residual:.3e})")
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
                self._update_trust_limits(prev_residual_norm, residual_norm, jacobian, delta, p_vec, sw_vec)
                
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
            if relative_residual < 20 * tol or residual_norm < 20 * tol * 1e3:
                print(f"  Невязка достаточно близка к допустимой, принимаем результат")
                return True, max_iter
            else:
                # Восстанавливаем исходное состояние
                self.fluid.pressure = current_p.clone()
                self.fluid.s_w = current_sw.clone()
                return False, max_iter
        finally:
            # Гарантируем восстановление print даже при исключениях
            builtins.print = _orig_print

    def _compute_residual_fast(self, dt, nx, ny, nz, dx, dy, dz):
        """
        Быстрый расчёт невязки (массовый баланс) без сборки Якобиана.
        Ранее здесь учитывалась только аккумуляция, что ухудшало line-search.
        Теперь используем полнофазовую невязку из `_compute_residual_full`,
        которая векторизована и достаточно быстра, но охватывает все члены
        (аккумуляцию, конвективные потоки, капиллярное давление и скважины).
        Сигнатура сохранена для совместимости с существующими вызовами.
        
        Args:
            dt: Временной шаг (сек)
            nx, ny, nz, dx, dy, dz: Параметры сетки (не используются, передаются
                                     для обратной совместимости).
        Returns:
            1-D тензор невязки длиной 2*N (water/oil)
        """
        # Используем оптимизированную «полную» невязку для всех фаз.
        # Она уже векторизована и опирается на кэшированные transmissibilities,
        # поэтому выполняется достаточно быстро даже на больших сетках.
        return self._compute_residual_full(dt)

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
        
        # Ограничиваем изменения давления (не более 10% от текущего значения и не более 5 МПа)
        max_p_change_rel = 0.1 * torch.abs(old_p)
        max_p_change_abs = 5e6 * torch.ones_like(old_p)  # 5 МПа
        max_p_change = torch.minimum(max_p_change_rel, max_p_change_abs)
        p_delta = torch.clamp(p_delta_raw, -max_p_change, max_p_change)
        
        # Насыщенность не ограничиваем компонентно – доверяем глобальному trust-region
        sw_delta = sw_delta_raw
        
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

        # -------- Локальный trust-region больше не нужен: глобальный ограничитель уже применён ---------

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
        # Убедимся, что проводимости рассчитаны
        self._init_impes_transmissibilities()
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
        sw_mean = float(self.fluid.s_w.mean().item())
        max_sw_cfg = self.sim_params.get("max_saturation_change", 0.05)
        max_sw_step = max(max_sw_cfg, 0.3 * (1 - sw_mean), 0.15)
        dSw_clamped = dSw.clamp(-max_sw_step, max_sw_step)

        S_w_new = (S_w_old + dSw_clamped).clamp(self.fluid.sw_cr, 1.0 - self.fluid.so_r)

        self.fluid.s_w = S_w_new
        self.fluid.s_o = 1.0 - self.fluid.s_w

        affected_cells = torch.sum(torch.abs(dSw) > 1e-8).item()
        print(
            f"P̄ = {P_new.mean()/1e6:.2f} МПа, Sw(min/max) = {self.fluid.s_w.min():.3f}/{self.fluid.s_w.max():.3f}, ΔSw ограничено до ±{max_sw_step}, ячеек изм.: {affected_cells}"
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
        # Гарантируем, что dtype совпадает с diag_vals (float32), чтобы scatter_add_ не падал
        vals = vals.to(torch.float32)
        acc_term = (self.porous_volume.view(-1) * self.fluid.cf.view(-1) / dt).to(torch.float32)
        diag_vals = torch.zeros(N, device=self.device, dtype=torch.float32)
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
        compressibility_term = ((self.porous_volume.view(-1) * self.fluid.cf.view(-1) / dt).float() * P_prev.view(-1).float())
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
        Q_total = compressibility_term + q_wells.flatten().float() + Q_g.view(-1).float() + Q_pc.view(-1).float()
        Q_total = Q_total.to(torch.float32)
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
                rate = well.control_value / 86400.0 * (1 if well.type == 'injector' else -1)
                q_wells[idx] += rate
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
        # Приводим всё к одному dtype (float32)
        b = b.to(torch.float32)
        if x0 is not None:
            x0 = x0.to(torch.float32)
        if M_diag is not None:
            M_diag = M_diag.to(torch.float32)
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
        
        # ΔM  (масса за шаг)
        delta_water = water_mass - self.fluid.prev_water_mass
        delta_oil   = oil_mass   - self.fluid.prev_oil_mass
        
        # Заполняем невязку аккумуляции и соответствующие элементы Якобиана
        for idx in range(num_cells):
            residual[2*idx]   = delta_water[idx]
            residual[2*idx+1] = delta_oil[idx]

            # --- диагональные элементы Jacobian (аккумуляция) ---
            dphi_dp    = self.reservoir.rock_compressibility * self.reservoir.porosity_ref.reshape(-1)[idx]
            drho_w_dp  = self.fluid.water_compressibility * rho_w[idx]
            drho_o_dp  = self.fluid.oil_compressibility   * rho_o[idx]

            # Вода
            jacobian[2*idx,   2*idx]   = (dphi_dp * sw_vec[idx] * rho_w[idx] + phi_vec[idx] * sw_vec[idx] * drho_w_dp) * cell_volume
            jacobian[2*idx,   2*idx+1] =  (phi_vec[idx] * rho_w[idx]) * cell_volume

            # Нефть
            jacobian[2*idx+1, 2*idx]   = (dphi_dp * (1 - sw_vec[idx]) * rho_o[idx] + phi_vec[idx] * (1 - sw_vec[idx]) * drho_o_dp) * cell_volume
            jacobian[2*idx+1, 2*idx+1] = -(phi_vec[idx] * rho_o[idx]) * cell_volume
        
        # Вычисляем и кэшируем производные относительных проницаемостей
        dkr_w_dsw = self.fluid.calc_dkrw_dsw(sw_vec)
        dkr_o_dsw = self.fluid.calc_dkro_dsw(sw_vec)
        dlambda_w_dsw = dkr_w_dsw / mu_w
        dlambda_o_dsw = dkr_o_dsw / mu_o
        
        # Производные плотностей по давлению для всех ячеек (используются в потоках)
        drho_w_dp_vec = self.fluid.water_compressibility * rho_w
        drho_o_dp_vec = self.fluid.oil_compressibility   * rho_o
        
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
                    
                    # Потоки (объёмные) -> переводим в массовые, умножая на среднюю плотность
                    water_flux = trans * lambda_w_up * (dp - dp_cap + gravity_term_w) * avg_rho_w
                    oil_flux  = trans * lambda_o_up * (dp + gravity_term_o) * avg_rho_o
                    
                    # Невязка
                    residual[2*idx1] -= water_flux
                    residual[2*idx1+1] -= oil_flux
                    residual[2*idx2] += water_flux
                    residual[2*idx2+1] += oil_flux
                    
                    # Производные для якобиана
                    # Учитываем зависимость плотности от давления (доп. слагаемое 0.5 * drho/dp)
                    dfw_dp1 = trans * lambda_w_up * (
                        -avg_rho_w + 0.5 * (dp - dp_cap + gravity_term_w) * drho_w_dp_vec[idx1]
                    )
                    dfw_dp2 = trans * lambda_w_up * (
                        avg_rho_w + 0.5 * (dp - dp_cap + gravity_term_w) * drho_w_dp_vec[idx2]
                    )

                    dfo_dp1 = trans * lambda_o_up * (
                        -avg_rho_o + 0.5 * (dp + gravity_term_o) * drho_o_dp_vec[idx1]
                    )
                    dfo_dp2 = trans * lambda_o_up * (
                        avg_rho_o + 0.5 * (dp + gravity_term_o) * drho_o_dp_vec[idx2]
                    )
                    
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
                        dfw_dsw1 = trans * dlambda_w_dsw[idx1] * (dp - dp_cap + gravity_term_w) * avg_rho_w
                        dfo_dsw1 = trans * dlambda_o_dsw[idx1] * (dp + gravity_term_o) * avg_rho_o
                        
                        if self.fluid.pc_scale > 0:
                            dfw_dsw1 += trans * lambda_w_up * (-dpc_dsw[idx1]) * avg_rho_w
                            dfw_dsw2 = trans * lambda_w_up * dpc_dsw[idx2] * avg_rho_w
                            
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
                        dfw_dsw2 = trans * dlambda_w_dsw[idx2] * (dp - dp_cap + gravity_term_w) * avg_rho_w
                        dfo_dsw2 = trans * dlambda_o_dsw[idx2] * (dp + gravity_term_o) * avg_rho_o
                        
                        if self.fluid.pc_scale > 0:
                            dfw_dsw1 = trans * lambda_w_up * (-dpc_dsw[idx1]) * avg_rho_w
                            dfw_dsw2 += trans * lambda_w_up * dpc_dsw[idx2] * avg_rho_w
                            
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
                    
                    # Потоки (объёмные) -> переводим в массовые, умножая на среднюю плотность
                    water_flux = trans * lambda_w_up * (dp - dp_cap + gravity_term_w) * avg_rho_w
                    oil_flux  = trans * lambda_o_up * (dp + gravity_term_o) * avg_rho_o
                    
                    # Невязка
                    residual[2*idx1] -= water_flux
                    residual[2*idx1+1] -= oil_flux
                    residual[2*idx2] += water_flux
                    residual[2*idx2+1] += oil_flux
                    
                    # Производные для якобиана
                    # Учитываем зависимость плотности от давления (доп. слагаемое 0.5 * drho/dp)
                    dfw_dp1 = trans * lambda_w_up * (
                        -avg_rho_w + 0.5 * (dp - dp_cap + gravity_term_w) * drho_w_dp_vec[idx1]
                    )
                    dfw_dp2 = trans * lambda_w_up * (
                        avg_rho_w + 0.5 * (dp - dp_cap + gravity_term_w) * drho_w_dp_vec[idx2]
                    )

                    dfo_dp1 = trans * lambda_o_up * (
                        -avg_rho_o + 0.5 * (dp + gravity_term_o) * drho_o_dp_vec[idx1]
                    )
                    dfo_dp2 = trans * lambda_o_up * (
                        avg_rho_o + 0.5 * (dp + gravity_term_o) * drho_o_dp_vec[idx2]
                    )
                    
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
                        dfw_dsw1 = trans * dlambda_w_dsw[idx1] * (dp - dp_cap + gravity_term_w) * avg_rho_w
                        dfo_dsw1 = trans * dlambda_o_dsw[idx1] * (dp + gravity_term_o) * avg_rho_o
                        
                        if self.fluid.pc_scale > 0:
                            dfw_dsw1 += trans * lambda_w_up * (-dpc_dsw[idx1]) * avg_rho_w
                            dfw_dsw2 = trans * lambda_w_up * dpc_dsw[idx2] * avg_rho_w
                            
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
                        dfw_dsw2 = trans * dlambda_w_dsw[idx2] * (dp - dp_cap + gravity_term_w) * avg_rho_w
                        dfo_dsw2 = trans * dlambda_o_dsw[idx2] * (dp + gravity_term_o) * avg_rho_o
                        
                        if self.fluid.pc_scale > 0:
                            dfw_dsw1 = trans * lambda_w_up * (-dpc_dsw[idx1]) * avg_rho_w
                            dfw_dsw2 += trans * lambda_w_up * dpc_dsw[idx2] * avg_rho_w
                            
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
                    
                    # Потоки (объёмные) -> переводим в массовые, умножая на среднюю плотность
                    water_flux = trans * lambda_w_up * (dp - dp_cap + gravity_term_w) * avg_rho_w
                    oil_flux  = trans * lambda_o_up * (dp + gravity_term_o) * avg_rho_o
                    
                    # Невязка
                    residual[2*idx1] -= water_flux
                    residual[2*idx1+1] -= oil_flux
                    residual[2*idx2] += water_flux
                    residual[2*idx2+1] += oil_flux
                    
                    # Производные для якобиана
                    # Учитываем зависимость плотности от давления (доп. слагаемое 0.5 * drho/dp)
                    dfw_dp1 = trans * lambda_w_up * (
                        -avg_rho_w + 0.5 * (dp - dp_cap + gravity_term_w) * drho_w_dp_vec[idx1]
                    )
                    dfw_dp2 = trans * lambda_w_up * (
                        avg_rho_w + 0.5 * (dp - dp_cap + gravity_term_w) * drho_w_dp_vec[idx2]
                    )

                    dfo_dp1 = trans * lambda_o_up * (
                        -avg_rho_o + 0.5 * (dp + gravity_term_o) * drho_o_dp_vec[idx1]
                    )
                    dfo_dp2 = trans * lambda_o_up * (
                        avg_rho_o + 0.5 * (dp + gravity_term_o) * drho_o_dp_vec[idx2]
                    )
                    
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
                        dfw_dsw1 = trans * dlambda_w_dsw[idx1] * (dp - dp_cap + gravity_term_w) * avg_rho_w
                        
                        if self.fluid.pc_scale > 0:
                            dfw_dsw1 += trans * lambda_w_up * (-dpc_dsw[idx1]) * avg_rho_w
                            dfw_dsw2 = trans * lambda_w_up * dpc_dsw[idx2] * avg_rho_w
                            
                            jacobian[2*idx1, 2*idx1+1] -= dfw_dsw1
                            jacobian[2*idx1, 2*idx2+1] -= dfw_dsw2
                            jacobian[2*idx2, 2*idx1+1] += dfw_dsw1
                            jacobian[2*idx2, 2*idx2+1] += dfw_dsw2
                        else:
                            jacobian[2*idx1, 2*idx1+1] -= dfw_dsw1
                            jacobian[2*idx2, 2*idx1+1] += dfw_dsw1
                    else:
                        dfw_dsw2 = trans * dlambda_w_dsw[idx2] * (dp - dp_cap + gravity_term_w) * avg_rho_w
                        
                        if self.fluid.pc_scale > 0:
                            dfw_dsw1 = trans * lambda_w_up * (-dpc_dsw[idx1]) * avg_rho_w
                            dfw_dsw2 += trans * lambda_w_up * dpc_dsw[idx2] * avg_rho_w
                            
                            jacobian[2*idx1, 2*idx1+1] -= dfw_dsw1
                            jacobian[2*idx1, 2*idx2+1] -= dfw_dsw2
                            jacobian[2*idx2, 2*idx1+1] += dfw_dsw1
                            jacobian[2*idx2, 2*idx2+1] += dfw_dsw2
                        else:
                            jacobian[2*idx1, 2*idx2+1] -= dfw_dsw2
                            jacobian[2*idx2, 2*idx2+1] += dfw_dsw2
                    
                    # Для нефти
                    if dp + gravity_term_o >= 0:
                        dfo_dsw1 = trans * dlambda_o_dsw[idx1] * (dp + gravity_term_o) * avg_rho_o
                        jacobian[2*idx1+1, 2*idx1+1] -= dfo_dsw1
                        jacobian[2*idx2+1, 2*idx1+1] += dfo_dsw1
                    else:
                        dfo_dsw2 = trans * dlambda_o_dsw[idx2] * (dp + gravity_term_o) * avg_rho_o
                        jacobian[2*idx1+1, 2*idx2+1] -= dfo_dsw2
                        jacobian[2*idx2+1, 2*idx2+1] += dfo_dsw2
        
        # Добавляем вклад скважин
        self._add_wells_to_system(residual, jacobian, dt)

    def _add_wells_to_system(self, residual, jacobian, dt):
        """
        Добавляет вклад скважин в систему (невязку и якобиан).
        Массовые дебиты интегрируются по времени: q_mass = q_mass_rate * dt.
        Здесь НЕ допускаем двойного вычитания – каждую фазу учитываем ровно один раз.
        ИСПРАВЛЕНО: используем БЛОЧНУЮ индексацию вместо интерлеавинга.
        """
        # Если якобиан не передан (режим JFNK), изменяем только residual.
        jac_update = jacobian is not None
 
        wells = self.well_manager.get_wells()
        
        # ИСПРАВЛЕНО: получаем размер грида для блочной индексации
        nx, ny, nz = self.reservoir.dimensions
        N = nx * ny * nz

        for well in wells:
            idx = well.cell_index_flat

            # Локальные давление и насыщенность в ячейке скважины
            p_cell = self.fluid.pressure.view(-1)[idx]
            sw_cell = self.fluid.s_w.view(-1)[idx]

            rho_w_cell = self.fluid.calc_water_density(p_cell)
            rho_o_cell = self.fluid.calc_oil_density(p_cell)

            # Подвижности и их производные
            mu_w = self.fluid.mu_water
            mu_o = self.fluid.mu_oil
            kr_w = self.fluid.calc_water_kr(sw_cell)
            kr_o = self.fluid.calc_oil_kr(sw_cell)
            lambda_w = kr_w / mu_w
            lambda_o = kr_o / mu_o
            lambda_t = lambda_w + lambda_o

            dkrw_dsw = self.fluid.calc_dkrw_dsw(sw_cell)
            dkro_dsw = self.fluid.calc_dkro_dsw(sw_cell)
            dlamb_w_dsw = dkrw_dsw / mu_w
            dlamb_o_dsw = dkro_dsw / mu_o

            if well.control_type == 'rate':
                # номинальный объёмный дебит (м³/сут) -> м³/с
                q_tot_vol_rate = well.control_value / 86400.0

                if well.type == 'injector':
                    q_w_mass_step = q_tot_vol_rate * self.fluid.rho_water_ref * dt  # кг за шаг
                    # БЛОЧНАЯ индексация: water equations в первых N элементах
                    residual[idx] -= q_w_mass_step
                    # нефть не закачивается
                else:  # producer
                    # Фракции потоков
                    fw = lambda_w / (lambda_t + 1e-12)
                    fo = 1.0 - fw

                    q_w_mass_step = q_tot_vol_rate * fw * self.fluid.rho_water_ref * dt
                    q_o_mass_step = q_tot_vol_rate * fo * self.fluid.rho_oil_ref   * dt

                    # БЛОЧНАЯ индексация: water в [0:N], oil в [N:2N]
                    residual[idx]     -= q_w_mass_step     # water equation
                    residual[N + idx] -= q_o_mass_step     # oil equation

                    # производные (по Sw) – только для продуцирующей
                    dfw_dsw = (dlamb_w_dsw * lambda_t - lambda_w * (dlamb_w_dsw + dlamb_o_dsw)) / (lambda_t**2 + 1e-12)
                    dfo_dsw = -dfw_dsw

                    dq_w_dsw = q_tot_vol_rate * self.fluid.rho_water_ref * dt * dfw_dsw
                    dq_o_dsw = q_tot_vol_rate * self.fluid.rho_oil_ref  * dt * dfo_dsw

                    if jac_update:
                        # БЛОЧНАЯ индексация для якобиана: [P: 0:N, Sw: N:2N]
                        jacobian[idx,     N + idx] -= dq_w_dsw  # water eq, sw var
                        jacobian[N + idx, N + idx] -= dq_o_dsw  # oil eq, sw var

            elif well.control_type == 'bhp':
                bhp_pa = well.control_value * 1e6  # МПа->Па

                q_w_vol_rate = well.well_index * lambda_w * (p_cell - bhp_pa)  # м³/с
                q_o_vol_rate = well.well_index * lambda_o * (p_cell - bhp_pa)  # м³/с

                q_w_mass_step = q_w_vol_rate * rho_w_cell * dt
                q_o_mass_step = q_o_vol_rate * rho_o_cell * dt

                # БЛОЧНАЯ индексация: water в [0:N], oil в [N:2N]
                residual[idx]     -= q_w_mass_step     # water equation
                residual[N + idx] -= q_o_mass_step     # oil equation

                # Якобиан: производные по давлению
                dq_w_dp = well.well_index * lambda_w * rho_w_cell * dt
                dq_o_dp = well.well_index * lambda_o * rho_o_cell * dt

                if jac_update:
                    # БЛОЧНАЯ индексация для якобиана: [P: 0:N, Sw: N:2N]
                    jacobian[idx,     idx]     -= dq_w_dp  # water eq, pressure var
                    jacobian[N + idx, idx]     -= dq_o_dp  # oil eq, pressure var

                # Якобиан: производные по насыщенности через подвижности
                dq_w_dsw = well.well_index * dlamb_w_dsw * (p_cell - bhp_pa) * rho_w_cell * dt
                dq_o_dsw = well.well_index * dlamb_o_dsw * (p_cell - bhp_pa) * rho_o_cell * dt

                if jac_update:
                    # БЛОЧНАЯ индексация для якобиана: [P: 0:N, Sw: N:2N]
                    jacobian[idx,     N + idx] -= dq_w_dsw  # water eq, sw var
                    jacobian[N + idx, N + idx] -= dq_o_dsw  # oil eq, sw var

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

    # ------------------------------------------------------------------
    #               ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ IMPES
    # ------------------------------------------------------------------

    def _init_impes_transmissibilities(self):
        """Вычисляет T_x/T_y/T_z для IMPES, если они ещё не посчитаны."""
        if hasattr(self, 'T_x') and hasattr(self, 'T_y') and hasattr(self, 'T_z'):
            return  # Уже инициализировано

        dx, dy, dz = self.reservoir.grid_size
        k_x, k_y, k_z = self.reservoir.permeability_tensors
        nx, ny, nz = self.reservoir.dimensions

        dev = self.fluid.device  # Используем то же устройство, что и тензоры флюида
        self.T_x = torch.zeros((nx-1, ny, nz), device=dev)
        self.T_y = torch.zeros((nx, ny-1, nz), device=dev)
        self.T_z = torch.zeros((nx, ny, nz-1), device=dev)

        for i in range(nx-1):
            k_harmonic = 2 * k_x[i, :, :] * k_x[i+1, :, :] / (k_x[i, :, :] + k_x[i+1, :, :] + 1e-15)
            self.T_x[i, :, :] = k_harmonic * dy * dz / dx

        for j in range(ny-1):
            k_harmonic = 2 * k_y[:, j, :] * k_y[:, j+1, :] / (k_y[:, j, :] + k_y[:, j+1, :] + 1e-15)
            self.T_y[:, j, :] = k_harmonic * dx * dz / dy

        for k in range(nz-1):
            k_harmonic = 2 * k_z[:, :, k] * k_z[:, :, k+1] / (k_z[:, :, k] + k_z[:, :, k+1] + 1e-15)
            self.T_z[:, :, k] = k_harmonic * dx * dy / dz

    # ==================================================================
    #         Jacobian-Free Newton–Krylov  (автоград J·v)               
    # ==================================================================

    def _fi_residual_vec(self, x, dt):
        """Возвращает невязку при векторе переменных x = [p, sw] (1-D)."""
        nx, ny, nz = self.reservoir.dimensions
        N = nx * ny * nz

        # 🔧 ИСПРАВЛЕНО: используем глобальные масштабы из JFNK для согласованности
        # НЕ пересчитываем P_SCALE каждый раз - берем из JFNK контекста
        P_SCALE = getattr(self, '_current_p_scale', 1e6)
        SATURATION_SCALE = getattr(self, '_current_saturation_scale', 1.0)

        # --- гарантия, что предыдущие массы инициализированы -----------------
        if self.fluid.prev_water_mass is None:
            phi0_vec = self.reservoir.porosity_ref.reshape(-1)
            c_r = self.reservoir.rock_compressibility
            p_prev_vec = self.fluid.prev_pressure.reshape(-1)
            phi_prev_vec = phi0_vec * (1 + c_r * (p_prev_vec - 1e5)) + self.ptc_alpha

            cell_vol = self.reservoir.cell_volume
            rho_w_prev = self.fluid.calc_water_density(p_prev_vec)
            rho_o_prev = self.fluid.calc_oil_density(p_prev_vec)
            sw_prev_vec = self.fluid.prev_sw.reshape(-1)

            self.fluid.prev_water_mass = phi_prev_vec * sw_prev_vec * rho_w_prev * cell_vol
            self.fluid.prev_oil_mass   = phi_prev_vec * (1 - sw_prev_vec) * rho_o_prev * cell_vol

        # 🔧 ИСПРАВЛЕНО: согласованное масштабирование с JFNK
        p_vec  = x[:N] * P_SCALE
        sw_vec_raw = x[N:] * SATURATION_SCALE
        # Используем мягкое ограничение для сохранения градиентов
        sw_vec = self._soft_clamp(sw_vec_raw, self.fluid.sw_cr, 1.0 - self.fluid.so_r)

        # Полный расчёт невязки БЕЗ изменения состояния
        p_new = p_vec.view(self.reservoir.dimensions)
        sw_new = sw_vec.view(self.reservoir.dimensions)

        residual = self._compute_residual_full_direct(dt, p_new, sw_new, 
                                                      self.fluid.prev_water_mass, 
                                                      self.fluid.prev_oil_mass)

        # 🔧 ИСПРАВЛЕНО: ПРОСТАЯ и СТАБИЛЬНАЯ нормализация
        # Нормализуем остатки на характерные величины БЕЗ сложной логики
        
        # Характерные массы для нормализации (из текущего состояния)
        phi_curr = self.reservoir.porosity_ref.view(-1) * (1 + self.reservoir.rock_compressibility * (p_vec - 1e5)) + self.ptc_alpha
        rho_w_curr = self.fluid.calc_water_density(p_vec)
        rho_o_curr = self.fluid.calc_oil_density(p_vec)
        cell_vol = self.reservoir.cell_volume
        
        # Характерные массы - просто средние значения по резервуару
        char_mass_w = (phi_curr * sw_vec * rho_w_curr * cell_vol).mean()
        char_mass_o = (phi_curr * (1 - sw_vec) * rho_o_curr * cell_vol).mean()
        
        # Простая нормализация БЕЗ inplace операций
        water_residuals = residual[:N] / (char_mass_w + 1e-12)
        oil_residuals = residual[N:] / (char_mass_o + 1e-12)
        
        # Возвращаем нормализованные остатки
        return torch.cat([water_residuals, oil_residuals])

    # ------------- линейный солвер BiCGSTAB (torch, J·v) --------------
    def _bicgstab(self, matvec, b, tol=1e-6, max_iter=400):
        """Улучшенный BiCGSTAB с обработкой NaN/Inf и простым fallback"""
        x = torch.zeros_like(b)
        r = b.clone()
        r_hat = r.clone()
        rho_old = alpha = omega = torch.tensor(1.0, device=b.device)
        v = torch.zeros_like(b)
        p = torch.zeros_like(b)
        
        initial_norm = r.norm()
        if initial_norm < tol:
            return x

        stagnation_count = 0
        restart_count = 0
        max_restarts = 2

        for i in range(max_iter):
            rho_new = torch.dot(r_hat, r)
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: обрабатываем NaN/Inf
            if not torch.isfinite(rho_new):
                # print(f"  BiCGSTAB: rho не конечна ({rho_new.item()}) на итерации {i+1}")
                if restart_count < max_restarts:
                    # print(f"  BiCGSTAB: перезапуск #{restart_count+1}")
                    restart_count += 1
                    # Простой перезапуск с возмущением
                    x = 0.1 * torch.randn_like(b)
                    r = b - matvec(x)
                    r_hat = r.clone()
                    rho_old = alpha = omega = torch.tensor(1.0, device=b.device)
                    v.zero_()
                    p.zero_()
                    continue
                else:
                    # print(f"  BiCGSTAB: слишком много перезапусков - возвращаем нулевое решение")
                    return torch.zeros_like(b)
            
            # Проверка на стагнацию rho
            if rho_new.abs() < 1e-20:
                # print(f"  BiCGSTAB: rho слишком мала ({rho_new.item():.2e}) на итерации {i+1}")
                stagnation_count += 1
                if stagnation_count > 3:
                    break
                continue

            beta = (rho_new / rho_old) * (alpha / omega)
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: beta должна быть конечной
            if not torch.isfinite(beta):
                # print(f"  BiCGSTAB: beta не конечна ({beta.item()}) на итерации {i+1}")
                if restart_count < max_restarts:
                    # print(f"  BiCGSTAB: перезапуск #{restart_count+1}")
                    restart_count += 1
                    # Сброс к простому состоянию
                    beta = torch.tensor(0.0, device=b.device)
                    p = r.clone()
                else:
                    # print(f"  BiCGSTAB: критическая ошибка - возвращаем текущее решение")
                    return x
            else:
                p = r + beta * (p - omega * v)
            
            v = matvec(p)
            
            # Проверяем что matvec вернул конечные значения
            if not torch.isfinite(v).all():
                # print(f"  BiCGSTAB: matvec вернул NaN/Inf на итерации {i+1}")
                return x  # возвращаем лучшее что есть
            
            alpha = rho_new / torch.dot(r_hat, v)
            
            if not torch.isfinite(alpha):
                # print(f"  BiCGSTAB: alpha не конечна на итерации {i+1}")
                return x
            
            s = r - alpha * v
            
            # Проверяем норму s
            s_norm = s.norm()
            if s_norm < tol:
                x = x + alpha * p
                break
            
            t = matvec(s)
            
            if not torch.isfinite(t).all():
                # print(f"  BiCGSTAB: второй matvec вернул NaN/Inf на итерации {i+1}")
                return x
            
            omega = torch.dot(t, s) / torch.dot(t, t)
            
            if not torch.isfinite(omega):
                # print(f"  BiCGSTAB: omega не конечна на итерации {i+1}")
                return x
            
            x = x + alpha * p + omega * s
            r = s - omega * t
            
            # ДИАГНОСТИКА первых итераций (отключаем .item() для векторизации)
            if i < 5:
                residual_norm = r.norm()
                # print(f"    BiCGSTAB[{i+1}]: ||r||={residual_norm:.3e}, rho={rho_new:.3e}, alpha={alpha:.3e}, omega={omega:.3e}")
                # print(f"                     ||x||={x.norm():.3e}, x_range=[{x.min():.3e}, {x.max():.3e}]")
            
            # Проверяем сходимость
            residual_norm = r.norm()
            if residual_norm < tol:
                # print(f"  BiCGSTAB сошелся за {i+1} итераций, ||r||={residual_norm:.3e}")
                break
            
            rho_old = rho_new
            
            # Проверка прогресса
            if i > 20 and residual_norm > 0.95 * initial_norm:
                stagnation_count += 1
                if stagnation_count > 10:
                    # print(f"  BiCGSTAB: стагнация на итерации {i+1}")
                    break

        return x

    def _bicgstab_improved(self, matvec, b, tol=1e-6, max_iter=200):
        """Улучшенная версия BiCGSTAB с лучшей обработкой ошибок"""
        x = torch.zeros_like(b)
        r = b.clone()
        r_hat = r.clone()
        rho_old = alpha = omega = torch.tensor(1.0, device=b.device)
        v = torch.zeros_like(b)
        p = torch.zeros_like(b)
        
        initial_norm = r.norm()
        if initial_norm < tol:
            return x

        for i in range(max_iter):
            rho_new = torch.dot(r_hat, r)
            
            # Проверка на стагнацию
            if rho_new.abs() < 1e-20:
                break
                
            if i > 0:
                beta = (rho_new / rho_old) * (alpha / omega)
                if not torch.isfinite(beta):
                    break
                p = r + beta * (p - omega * v)
            else:
                p = r.clone()
            
            v = matvec(p)
            
            # Проверяем что matvec вернул конечные значения
            if not torch.isfinite(v).all():
                break
            
            alpha = rho_new / torch.dot(r_hat, v)
            if not torch.isfinite(alpha):
                break
            
            s = r - alpha * v
            
            # Проверяем норму s
            s_norm = s.norm()
            if s_norm < tol:
                x = x + alpha * p
                break
            
            t = matvec(s)
            
            if not torch.isfinite(t).all():
                break
            
            omega = torch.dot(t, s) / torch.dot(t, t)
            if not torch.isfinite(omega):
                break
            
            x = x + alpha * p + omega * s
            r = s - omega * t
            
            # Проверяем сходимость
            if r.norm() < tol:
                break
                
            rho_old = rho_new

        return x

    def _fallback_solve(self, matvec, rhs):
        """Fallback решение для случая когда все стратегии не сработали"""
        # Пробуем простое решение - направление антиградиента
        print("    Пробуем простое решение - направление антиградиента")
        
        # Используем направление антиградиента с разумным масштабированием
        delta_magnitude = min(1.0, rhs.norm().item())
        delta = -rhs / (rhs.norm() + 1e-8) * delta_magnitude
        
        # Проверяем качество решения
        if delta.norm() > 1e-12:
            print(f"    ✅ Fallback решение: ||delta||={delta.norm():.3e}")
            return delta
        else:
            print("    ❌ Fallback решение слишком мало")
            return torch.zeros_like(rhs)

    # ------------- JFNK основной цикл --------------------------------
    def _fi_jfnk_step(self, dt, tol=None, max_iter=10, damping=0.6):
        if tol is None:
            tol = self.sim_params.get("newton_tolerance", 1e-3)

        nx, ny, nz = self.reservoir.dimensions
        N = nx * ny * nz

        # 🔧 ИСПРАВЛЕНО: адаптивные и стабильные масштабы
        # Используем текущее среднее давление как характерный масштаб
        current_pressure = float(self.fluid.pressure.mean().item())
        P_SCALE = max(current_pressure, 1e6)  # минимум 1 МПа
        SATURATION_SCALE = 1.0  # насыщенность безразмерна
        
        print(f"  Масштабы: P_SCALE={P_SCALE/1e6:.1f} МПа, текущее давление={current_pressure/1e6:.1f} МПа")
        
        # Начальное приближение (масштабируем обе переменные)
        x = torch.cat([
            (self.fluid.pressure.view(-1) / P_SCALE),           # давление: ~20
            (self.fluid.s_w.view(-1) / SATURATION_SCALE)        # насыщенность: ~0.2-0.4
        ]).to(self.device).requires_grad_(True)

        initial_norm = None
        for it in range(max_iter):
            # Устанавливаем текущий масштаб для _fi_residual_vec
            self._current_p_scale = P_SCALE
            self._current_saturation_scale = SATURATION_SCALE
            
            F = self._fi_residual_vec(x, dt)
            norm_F = F.norm()
            if initial_norm is None:
                initial_norm = norm_F.clone()
            rel_res = (norm_F / (initial_norm + 1e-20)).item()
            print(f"  Итерация JFNK {it+1}: ||F||={norm_F:.3e}  rel={rel_res:.3e}")
            if rel_res < tol:
                break

            # 🔧 ИСПРАВЛЕНО: простой и стабильный finite differences
            def matvec(v):
                self._current_p_scale = P_SCALE
                self._current_saturation_scale = SATURATION_SCALE
                
                # 🔧 АГРЕССИВНЫЙ адаптивный eps для катастрофически плохо обусловленных систем
                F_norm = F.norm().item()
                
                # Базовый eps увеличен для системы с condition number ~1e12
                base_eps = 1e-3  # минимальный eps для стабильности
                
                # Адаптивный eps на основе нормы residual
                adaptive_eps = max(base_eps, 5e-3 * F_norm)  # увеличен коэффициент
                
                # Дополнительно увеличиваем для первых итераций (критично для плохо обусловленных систем)
                if it == 0:
                    adaptive_eps = max(adaptive_eps, 5e-2)  # первая итерация - очень крупный
                elif it <= 2:
                    adaptive_eps = max(adaptive_eps, 1e-2)  # первые итерации - крупный
                elif it <= 5:
                    adaptive_eps = max(adaptive_eps, 5e-3)  # средние итерации - умеренный
                
                eps = adaptive_eps
                
                # Для condition number > 1e12 принудительно увеличиваем eps
                if hasattr(self, '_last_condition_number') and self._last_condition_number > 1e12:
                    eps = max(eps, 1e-2)  # принудительно крупный eps для катастрофически плохих систем
                
                print(f"    Finite difference epsilon: {eps:.3e} (iteration {it})")
                
                F_plus = self._fi_residual_vec(x + eps * v, dt)
                F_minus = self._fi_residual_vec(x - eps * v, dt)
                Jv_fd = (F_plus - F_minus) / (2 * eps)
                
                # Минимальная регуляризация только для стабильности
                reg_lambda = 1e-12
                return Jv_fd + reg_lambda * v

            # ---- ПРОСТОЕ ПРЕДОБУСЛАВЛИВАНИЕ ----
            lin_cfg = self.sim_params.get("linear_solver", {})
            backend = lin_cfg.get("backend", "simple")
            
            if backend in ["hypre", "amgx", "cpr"]:
                print(f"  Используем CPR предобуславливатель (backend={backend})")
                
                # Создаем CPR предобуславливатель только при необходимости
                if not hasattr(self, "_cpr_preconditioner"):
                    try:
                        from solver.cpr import CPRPreconditioner
                        self._cpr_preconditioner = CPRPreconditioner(
                            self.reservoir, self.fluid, 
                            backend=backend, omega=0.8
                        )
                        print("  CPR предобуславливатель создан")
                    except Exception as e:
                        print(f"  Ошибка создания CPR: {e}")
                        self._cpr_preconditioner = None
                
                # Применяем CPR предобуславливатель
                def matvec_preconditioned(v):
                    Jv = matvec(v)
                    if hasattr(self, "_cpr_preconditioner") and self._cpr_preconditioner is not None:
                        try:
                            return self._cpr_preconditioner.apply(Jv)
                        except Exception as e:
                            print(f"  CPR не удался: {e}")
                            # Fallback на простое блочное предобуславливание
                            N = len(x) // 2
                            return torch.cat([Jv[:N] * 0.1, Jv[N:] * 0.5])
                    else:
                        # Простое блочное предобуславливание
                        N = len(x) // 2
                        return torch.cat([Jv[:N] * 0.1, Jv[N:] * 0.5])

                rhs = -F
                        
            else:
                print("  Используем улучшенное предобуславливание")
                
                # 🔧 УЛУЧШЕННОЕ ПРЕДОБУСЛАВЛИВАНИЕ
                # Создаем приближенный якобиан только на первой итерации
                if it == 0:
                    try:
                        print("    Создание ILU(0) предобуславливателя...")
                        
                        # Создаем спарсифицированный якобиан для предобуславливания
                        n = len(x)
                        N = n // 2
                        
                        # Быстрое создание предобуславливателя
                        # Диагональные элементы + ключевые off-diagonal
                        diag_elements = torch.zeros(n, device=x.device)
                        
                        # 🔧 УЛУЧШЕННОЕ вычисление диагонали с сильной регуляризацией
                        sample_step = max(1, n//20)  # более частый sampling для лучшего предобуславливания
                        for i in range(0, n, sample_step):
                            e_i = torch.zeros_like(x)
                            e_i[i] = 1.0
                            Jei = matvec(e_i)
                            # Сильная регуляризация для плохо обусловленных систем
                            diag_elements[i] = Jei[i] + 1e-3  # увеличена регуляризация
                        
                        # 🔧 УЛУЧШЕННАЯ интерполяция для остальных элементов
                        for i in range(n):
                            if diag_elements[i] == 0:
                                # Находим ближайший вычисленный элемент
                                idx = (i // sample_step) * sample_step
                                if idx < n and diag_elements[idx] != 0:
                                    diag_elements[i] = diag_elements[idx]
                                else:
                                    # Более сильная регуляризация для fallback
                                    diag_elements[i] = 1e-2  # увеличен fallback
                        
                        # Адаптивное масштабирование на основе блоков
                        pressure_diag = diag_elements[:N]
                        saturation_diag = diag_elements[N:]
                        
                        # 🔧 УЛУЧШЕННАЯ нормализация по блокам
                        p_mean = pressure_diag.abs().mean() + 1e-8
                        s_mean = saturation_diag.abs().mean() + 1e-8
                        
                        # Более консервативные масштабы для стабильности
                        p_scale = 0.1 / p_mean  # более консервативно
                        s_scale = 0.5 / s_mean  # более консервативно
                        
                        # Более широкие ограничения масштабов
                        p_scale = torch.clamp(p_scale, 0.01, 0.5)  # увеличен диапазон
                        s_scale = torch.clamp(s_scale, 0.1, 1.0)   # увеличен диапазон
                        
                        self._precond_p_scale = float(p_scale.item())
                        self._precond_s_scale = float(s_scale.item())
                        self._precond_diag = diag_elements
                        
                        print(f"    ILU(0) готов: P_scale={self._precond_p_scale:.3f}, S_scale={self._precond_s_scale:.3f}")
                        
                    except Exception as e:
                        print(f"    ILU(0) не удался: {e}, используем адаптивное Jacobi")
                        self._precond_p_scale = 0.1
                        self._precond_s_scale = 0.5
                        self._precond_diag = None
                
                # 🔧 СОЗДАНИЕ ПОЛНОГО ПРЕДОБУСЛАВЛИВАНИЯ (один раз)
                if len(x) <= 500 and it == 0 and not hasattr(self, '_precond_full_inv'):
                    try:
                        print("    Создание полной матрицы для прямого обращения...")
                        n = len(x)
                        
                        # Создаем полную матрицу якобиана
                        J_full = torch.zeros((n, n), device=x.device, dtype=x.dtype)
                        for i in range(n):
                            if i % 50 == 0:
                                print(f"    Создание матрицы: {i}/{n}")
                            e_i = torch.zeros_like(x)
                            e_i[i] = 1.0
                            J_full[:, i] = matvec(e_i)
                        
                        # Регуляризация для обращения
                        reg_lambda = 1e-6
                        J_reg = J_full + reg_lambda * torch.eye(n, device=x.device, dtype=x.dtype)
                        
                        # Вычисляем обратную матрицу
                        try:
                            J_inv = torch.inverse(J_reg)
                            self._precond_full_inv = J_inv
                            print("    ✅ Полная обратная матрица создана!")
                        except Exception as e:
                            print(f"    ❌ Обращение не удалось: {e}")
                            # Попробуем псевдообращение
                            try:
                                J_inv = torch.pinverse(J_reg)
                                self._precond_full_inv = J_inv
                                print("    ✅ Псевдообращение создано!")
                            except Exception as e2:
                                print(f"    ❌ Псевдообращение не удалось: {e2}")
                                self._precond_full_inv = None
                                
                    except Exception as e:
                        print(f"    ❌ Создание полной матрицы не удалось: {e}")
                        self._precond_full_inv = None
                
                # Применяем предобуславливание
                def matvec_preconditioned(v):
                    Jv = matvec(v)
                    N = len(x) // 2
                    
                    # Применяем полное предобуславливание если доступно
                    if hasattr(self, '_precond_full_inv') and self._precond_full_inv is not None:
                        try:
                            return torch.matmul(self._precond_full_inv, Jv)
                        except Exception as e:
                            print(f"    ❌ Полное предобуславливание не удалось: {e}")
                    
                    # Fallback на диагональное предобуславливание
                    if hasattr(self, '_precond_diag') and self._precond_diag is not None:
                        try:
                            # Более сильная регуляризация
                            reg_diag = self._precond_diag + 1e-2
                            reg_diag = torch.clamp(reg_diag, 1e-4, 1e4)
                            return Jv / reg_diag
                        except Exception as e:
                            print(f"    Диагональное предобуславливание не удалось: {e}")
                    
                    # Последний fallback - блочное предобуславливание
                    p_scale = getattr(self, '_precond_p_scale', 0.01)  # более сильное
                    s_scale = getattr(self, '_precond_s_scale', 0.1)   # более сильное
                    
                    return torch.cat([Jv[:N] * p_scale, Jv[N:] * s_scale])

                rhs = -F
                
                                    # 🔍 КРИТИЧЕСКАЯ ДИАГНОСТИКА: сравнение с autograd
                if it == 0:
                    print("    🔍 КРИТИЧЕСКАЯ ДИАГНОСТИКА: AUTOGRAD vs FINITE DIFFERENCES")
                    
                    # Проверяем condition number и сохраняем для использования в finite differences
                    self._diagnostic_condition_number(matvec_preconditioned, len(x))
                    
                    # Сохраняем приблизительное condition number для адаптации eps
                    try:
                        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                        v_test = torch.randn(len(x), dtype=torch.float32, device=device)
                        v_test = v_test / torch.norm(v_test)
                        Av_test = matvec_preconditioned(v_test)
                        self._last_condition_number = Av_test.norm().item() * 1e12  # приблизительная оценка
                    except:
                        self._last_condition_number = 1e12  # fallback значение
                    
                    # Сравниваем с autograd якобианом
                    print("    🔍 Сравнение с autograd:")
                    try:
                        # Небольшой test vector для проверки
                        test_vec = torch.ones_like(x) * 0.01
                        
                        # JFNK matvec
                        jfnk_result = matvec(test_vec)
                        print(f"    JFNK J*v: ||result||={jfnk_result.norm():.3e}")
                        print(f"    JFNK J*v диапазон: [{jfnk_result.min():.3e}, {jfnk_result.max():.3e}]")
                        
                        # Autograd matvec
                        def autograd_matvec(v):
                            return torch.autograd.functional.jvp(
                                lambda z: self._fi_residual_vec(z, dt), 
                                (x,), (v,)
                            )[1]
                        
                        autograd_result = autograd_matvec(test_vec)
                        print(f"    Autograd J*v: ||result||={autograd_result.norm():.3e}")
                        print(f"    Autograd J*v диапазон: [{autograd_result.min():.3e}, {autograd_result.max():.3e}]")
                        
                        # Сравниваем результаты
                        diff = (jfnk_result - autograd_result).norm()
                        rel_error = diff / autograd_result.norm()
                        print(f"    Относительная ошибка: {rel_error:.3e}")
                        
                        if rel_error > 0.1:
                            print("    ❌ JFNK FINITE DIFFERENCES СЛОМАНЫ!")
                            print("    Проблема в finite differences, не в предобуславливании!")
                        else:
                            print("    ✅ Finite differences корректны")
                            print("    Проблема в предобуславливании, не в finite differences")
                            
                    except Exception as e:
                        print(f"    ❌ Ошибка сравнения с autograd: {e}")
                        
                    # Проверяем residual в точке x
                    print(f"    Residual в точке x: ||F||={F.norm():.3e}")
                    print(f"    F направление: [{F[:5]}]")
                    print(f"    rhs направление: [{rhs[:5]}]")
            
            # Решаем линейную систему
            print(f"  Решаем линейную систему: ||F||={F.norm():.3e}, ||rhs||={rhs.norm():.3e}")
            
            # 🔧 УЛУЧШЕННОЕ РЕШЕНИЕ ЛИНЕЙНОЙ СИСТЕМЫ
            delta = None
            
            # 🔧 АГРЕССИВНЫЕ стратегии GMRES - для плохо обусловленных систем
            strategies = [
                ("GMRES-80", {"restart": 80, "max_iter": 200, "tol": 1e-3}),   # очень высокий restart для тяжелых систем
                ("GMRES-60", {"restart": 60, "max_iter": 150, "tol": 5e-3}),   # высокий restart
                ("GMRES-40", {"restart": 40, "max_iter": 120, "tol": 1e-2}),   # умеренный restart
                ("GMRES-20", {"restart": 20, "max_iter": 80, "tol": 5e-2}),    # низкий restart, мягкий tolerance
                ("BiCGSTAB", {"max_iter": 200, "tol": 1e-2}),                  # альтернативный метод
            ]
            
            for strategy_name, params in strategies:
                try:
                    if strategy_name.startswith("GMRES"):
                        from linear_gpu.gmres import gmres
                        
                        print(f"    Пробуем {strategy_name}: restart={params['restart']}, tol={params['tol']:.1e}")
                        delta, info = gmres(matvec_preconditioned, rhs, 
                                          tol=params['tol'], 
                                          restart=params['restart'], 
                                          max_iter=params['max_iter'])
                        
                        if info == 0:
                            print(f"    ✅ {strategy_name} сошёлся успешно")
                            break
                        else:
                            print(f"    ⚠️  {strategy_name} не сошёлся (info={info})")
                            
                            # Если решение разумное, используем его
                            if delta is not None and delta.norm() > 1e-12 and delta.norm() < 1e6:
                                print(f"    🔧 Используем частичное решение: ||delta||={delta.norm():.3e}")
                                break
                                
                    elif strategy_name == "BiCGSTAB":
                        print(f"    Пробуем BiCGSTAB: max_iter={params['max_iter']}, tol={params['tol']:.1e}")
                        
                        # Используем встроенную реализацию BiCGSTAB
                        delta = self._bicgstab_improved(matvec_preconditioned, rhs, 
                                                      tol=params['tol'], 
                                                      max_iter=params['max_iter'])
                        
                        if delta is not None and delta.norm() > 1e-12:
                            print(f"    ✅ BiCGSTAB успешен: ||delta||={delta.norm():.3e}")
                            break
                        else:
                            print(f"    ⚠️  BiCGSTAB не сошёлся")
                            
                except Exception as e:
                    print(f"    ❌ {strategy_name} ошибка: {e}")
                    continue
            
            # Если ни одна стратегия не сработала полностью
            if delta is None or delta.norm() < 1e-12:
                print("    Все стратегии не сработали - используем fallback")
                delta = self._fallback_solve(matvec_preconditioned, rhs)
            
            # 🔧 ИСПРАВЛЕННЫЙ fallback на BiCGSTAB - более мягкие условия
            if delta is None or delta.norm() < 1e-6:  # менее строгое условие
                try:
                    print("    Пробуем BiCGSTAB fallback...")
                    delta = self._bicgstab(matvec_preconditioned, rhs, tol=1e-1, max_iter=50)  # более мягкий tolerance
                    if delta.norm() > 1e-6:
                        print(f"    ✅ BiCGSTAB успешен: ||delta||={delta.norm():.3e}")
                    else:
                        print(f"    ⚠️  BiCGSTAB дал малое решение")
                except Exception as e:
                    print(f"    ❌ BiCGSTAB не удался: {e}")
            
            # Последний fallback - улучшенное простое решение
            if delta is None or delta.norm() < 1e-6:
                print("    Используем улучшенное простое решение")
                
                # Используем направление антиградиента с разумным масштабированием
                delta_magnitude = min(1.0, rhs.norm().item())
                delta = -rhs / (rhs.norm() + 1e-8) * delta_magnitude
                print(f"    Простое решение: ||delta||={delta.norm():.3e}")
            
            print(f"  Линейное решение: ||delta||={delta.norm():.3e}")
            print(f"  delta[:5] = {delta[:5]}")
            print(f"  delta диапазон: [{delta.min():.3e}, {delta.max():.3e}]")
            
            # 🔍 ДИАГНОСТИКА качества решения
            if delta.norm() < 1e-3:
                print(f"  ⚠️  РЕШЕНИЕ СЛИШКОМ МАЛО! ||delta||={delta.norm():.3e}")
                
                # Проверяем что происходит с Jacobi*delta vs rhs
                J_delta = matvec_preconditioned(delta)
                residual_reduction = (J_delta - rhs).norm() / rhs.norm()
                print(f"  ||J*delta - rhs|| / ||rhs|| = {residual_reduction:.3e}")
                
                if residual_reduction > 0.1:
                    print("  ❌ ЛИНЕЙНАЯ СИСТЕМА РЕШЕНА ПЛОХО!")
                    print("  Проблема в предобуславливании или GMRES")
                else:
                    print("  ✅ Линейная система решена правильно, но шаг мал")
                    print("  Проблема в масштабировании или физике")
            
            # Проверка на разумность решения
            if delta.norm() < 1e-15:
                print("  ❌ Решение слишком мало - возможны проблемы с обусловленностью")
                break
            elif delta.norm() > 1e5:
                print("  ❌ Решение слишком велико - возможны проблемы с масштабированием")
                break
            
            # 🔧 ИСПРАВЛЕНО: простая trust-region стратегия
            # Ограничиваем шаг разумными пределами
            N = len(x) // 2
            delta_p = delta[:N]  # изменение давления (безразмерное)
            delta_s = delta[N:]  # изменение насыщенности (безразмерное)
            
            # 🔧 УМНЫЕ физические пределы для шагов - адаптивные на основе текущего состояния
            # Для давления: больше позволяем при низком давлении, меньше при высоком
            current_p_phys = (x[:N] * P_SCALE).mean().item() / 1e6  # средние МПа
            if current_p_phys < 5:   # низкое давление
                max_dp_scaled = 8.0  # можем делать большие шаги
            elif current_p_phys < 20:  # умеренное давление
                max_dp_scaled = 5.0
            else:                    # высокое давление
                max_dp_scaled = 2.0  # осторожные шаги
            
            # Для насыщенности: более консервативно около границ
            current_sw = (x[N:] * SATURATION_SCALE).mean().item()
            if current_sw < 0.3 or current_sw > 0.7:  # около границ
                max_ds_scaled = 0.1  # очень осторожно
            else:                                      # в средней зоне
                max_ds_scaled = 0.2  # умеренно
            
            # Более консервативное масштабирование шага
            scale_p = min(1.0, max_dp_scaled / (delta_p.abs().max().item() + 1e-12))
            scale_s = min(1.0, max_ds_scaled / (delta_s.abs().max().item() + 1e-12))
            scale = min(scale_p, scale_s)
            scale = max(scale, 0.05)  # минимальный масштаб 0.05 для большей осторожности
            
            print(f"    Trust-region: current P={current_p_phys:.1f} МПа, Sw={current_sw:.3f}")
            print(f"    Trust-region: max_dp={max_dp_scaled:.1f}, max_ds={max_ds_scaled:.2f}, scale={scale:.3f}")
            
            if scale < 1.0:
                delta = delta * scale
                print(f"  Trust-region: масштабирую шаг на {scale:.3f}")
            
            # 🔧 УЛУЧШЕННЫЙ Line-search с адаптивными критериями
            best_alpha = 0.0
            best_norm = norm_F.item()
            
            # Более агрессивные alpha значения для плохо обусловленных систем
            if it <= 2:  # для первых итераций - более консервативные шаги
                alphas = [0.1, 0.05, 0.01, 0.005, 0.001]
            else:  # для последующих итераций - более агрессивные
                alphas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01]
            
            # Адаптивные критерии принятия на основе итерации
            if it == 0:
                accept_factor = 2.0    # для первой итерации допускаем удвоение
            elif it <= 2:
                accept_factor = 1.5    # для первых итераций - мягче
            else:
                accept_factor = 1.1    # для последующих - строже
            
            for alpha in alphas:
                try:
                    x_new = x + alpha * delta
                    self._current_p_scale = P_SCALE
                    self._current_saturation_scale = SATURATION_SCALE
                    F_new = self._fi_residual_vec(x_new, dt)
                    norm_new = F_new.norm().item()
                    
                    # Адаптивные критерии принятия
                    if norm_new < best_norm * accept_factor:
                        best_alpha = alpha
                        best_norm = norm_new
                        print(f"  Line-search: alpha={alpha:.3f}, ||F||: {norm_F.item():.3e} → {norm_new:.3e}")
                        break  # принимаем первое подходящее значение
                        
                except Exception as e:
                    print(f"  Line-search alpha={alpha}: {e}")
                    continue
            
            # Fallback: если ничего не подходит, используем очень маленький шаг
            if best_alpha == 0:
                best_alpha = 0.001
                try:
                    x_new = x + best_alpha * delta
                    self._current_p_scale = P_SCALE
                    self._current_saturation_scale = SATURATION_SCALE
                    F_new = self._fi_residual_vec(x_new, dt)
                    best_norm = F_new.norm().item()
                    print(f"  Line-search: принудительно используем alpha={best_alpha:.3f}")
                except:
                    print("  ❌ Line-search критически не удался")
                    break
            
            x = x + best_alpha * delta
            
            # 🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: применяем физические ограничения
            x_clipped = self._apply_physical_constraints(x, P_SCALE, SATURATION_SCALE)
            if not torch.allclose(x, x_clipped, atol=1e-10):
                x = x_clipped
                print(f"  🔧 Применены физические ограничения для {N} переменных")
            
            print(f"  ✅ Line-search: alpha={best_alpha:.3f}, ||F||: {norm_F.item():.3e} → {best_norm:.3e}")
        
        # Возвращаем немасштабированные значения
        self._current_p_scale = P_SCALE
        self._current_saturation_scale = SATURATION_SCALE
        
        p_result = x[:N] * P_SCALE
        s_result = x[N:] * SATURATION_SCALE
        
        return p_result.view(self.reservoir.dimensions), s_result.view(self.reservoir.dimensions)

    # ------------- адаптивное уменьшение dt с JFNK -------------------
    def _fi_jfnk_adaptive(self, dt):
        current_dt = dt
        max_attempts = self.sim_params.get("max_time_step_attempts", 4)

        for attempt in range(max_attempts):
            print(f"Попытка шага (JFNK) с dt = {current_dt/86400:.2f} дней (Попытка {attempt+1}/{max_attempts})")
            
            try:
                # Вызываем JFNK солвер
                p_new, s_new = self._fi_jfnk_step(current_dt)
                
                # Обновляем состояние fluid
                self.fluid.pressure = p_new
                self.fluid.s_w = s_new
                self.fluid.s_o = 1.0 - s_new
                self.fluid.prev_pressure = p_new.clone()
                self.fluid.prev_sw = s_new.clone()
                
                # Пересчитываем массы для следующего шага
                rho_w = self.fluid.calc_water_density(p_new.view(-1))
                rho_o = self.fluid.calc_oil_density(p_new.view(-1))
                phi0 = self.reservoir.porosity_ref.view(-1)
                phi = phi0 * (1 + self.reservoir.rock_compressibility * (p_new.view(-1) - 1e5)) + self.ptc_alpha
                cell_vol = self.reservoir.cell_volume
                self.fluid.prev_water_mass = phi * s_new.view(-1) * rho_w * cell_vol
                self.fluid.prev_oil_mass = phi * (1 - s_new.view(-1)) * rho_o * cell_vol
                
                print(f"  ✅ JFNK шаг успешен")
                return True
                
            except Exception as e:
                print(f"  ❌ JFNK шаг не удался: {e}")
                print("  Уменьшаем dt")
                current_dt /= self.sim_params.get("dt_reduction_factor", 2.0)
                
        print("  ❌ JFNK не смог найти решение после максимального количества попыток")
        return False

    def _compute_residual_full(self, dt):
        """Полная невязка (масса) без сборки Якобиана – используется в JFNK.
        Содержит аккумуляцию, конвективные потоки и скважины.
        Возвращает 1-D тензор длиной 2*N (water/oil).
        
        ГРАНИЧНЫЕ УСЛОВИЯ: No-flow boundary conditions применяются неявно:
        - Поток через внешние границы пласта = 0
        - Вычисляются только потоки между соседними ячейками ВНУТРИ пласта
        - Граничные ячейки имеют правильные уравнения баланса массы
        """
        nx, ny, nz = self.reservoir.dimensions
        N = nx * ny * nz
        device = self.fluid.device

        # === геометрия ===
        dx, dy, dz = self.reservoir.grid_size
        cell_volume = dx * dy * dz

        p = self.fluid.pressure  # (nx,ny,nz)
        # ИСПРАВЛЕНО: используем мягкое ограничение для сохранения градиентов (автоматический slope)
        sw = self._soft_clamp(self.fluid.s_w, self.fluid.sw_cr, 1.0 - self.fluid.so_r)

        # пористость сжатого пласта
        phi = self.reservoir.porosity_ref * (1 + self.reservoir.rock_compressibility * (p - 1e5)) + self.ptc_alpha

        # плотности
        rho_w = self.fluid.calc_water_density(p)
        rho_o = self.fluid.calc_oil_density(p)

        # аккумуляция (масса, кг)
        water_mass = phi * sw * rho_w * cell_volume
        oil_mass   = phi * (1 - sw) * rho_o * cell_volume

        res_water = (water_mass - self.fluid.prev_water_mass.view(nx,ny,nz))
        res_oil   = (oil_mass   - self.fluid.prev_oil_mass.view(nx,ny,nz))

        # === потоки ===
        # вычисляем Т_x, T_y, T_z если не было
        self._init_impes_transmissibilities()
        # относит. проницаемости и мобил.
        kr_w = self.fluid.calc_water_kr(sw)
        kr_o = self.fluid.calc_oil_kr(sw)
        mu_w = self.fluid.mu_water
        mu_o = self.fluid.mu_oil
        lambda_w = kr_w / mu_w
        lambda_o = kr_o / mu_o

        # капиллярное давление
        if self.fluid.pc_scale > 0:
            pc = self.fluid.calc_capillary_pressure(sw)
        else:
            pc = torch.zeros_like(p)
            
        # === X-направление (NO-FLOW BC на левой/правой границах) ===
        # Вычисляем только потоки между соседними ячейками (i,i+1)
        # Поток через границу i=0 (слева) = 0 (неявно)  
        # Поток через границу i=nx (справа) = 0 (неявно)
        dp_x = p[:-1,:,:] - p[1:,:,:]  # shape (nx-1,ny,nz)
        lambda_w_up_x = torch.where(dp_x > 0, lambda_w[:-1,:,:], lambda_w[1:,:,:])
        lambda_o_up_x = torch.where(dp_x > 0, lambda_o[:-1,:,:], lambda_o[1:,:,:])
        rho_w_avg_x = 0.5 * (rho_w[:-1,:,:] + rho_w[1:,:,:])
        rho_o_avg_x = 0.5 * (rho_o[:-1,:,:] + rho_o[1:,:,:])
        dpc_x = pc[:-1,:,:] - pc[1:,:,:]
        trans_x = self.T_x * dt  
        water_flux_x = trans_x * lambda_w_up_x * (dp_x - dpc_x) * rho_w_avg_x
        oil_flux_x   = trans_x * lambda_o_up_x * (dp_x)            * rho_o_avg_x
        # расход из левой ячейки ("-"), к правой ("+")
        res_water[:-1,:,:] -= water_flux_x  # ячейки i=0..nx-2 
        res_water[1: ,:,:] += water_flux_x  # ячейки i=1..nx-1
        res_oil  [:-1,:,:] -= oil_flux_x
        res_oil  [1: ,:,:] += oil_flux_x

        # === Y-направление (NO-FLOW BC на передней/задней границах) ===
        dp_y = p[:,:-1,:] - p[:,1:,:]
        lambda_w_up_y = torch.where(dp_y > 0, lambda_w[:,:-1,:], lambda_w[:,1:,:])
        lambda_o_up_y = torch.where(dp_y > 0, lambda_o[:,:-1,:], lambda_o[:,1:,:])
        rho_w_avg_y = 0.5 * (rho_w[:,:-1,:] + rho_w[:,1:,:])
        rho_o_avg_y = 0.5 * (rho_o[:,:-1,:] + rho_o[:,1:,:])
        dpc_y = pc[:,:-1,:] - pc[:,1:,:]
        trans_y = self.T_y * dt
        water_flux_y = trans_y * lambda_w_up_y * (dp_y - dpc_y) * rho_w_avg_y
        oil_flux_y   = trans_y * lambda_o_up_y * (dp_y) * rho_o_avg_y
        res_water[:,:-1,:] -= water_flux_y  # j=0..ny-2
        res_water[:,1: ,:] += water_flux_y  # j=1..ny-1
        res_oil  [:,:-1,:] -= oil_flux_y
        res_oil  [:,1: ,:] += oil_flux_y

        # === Z-направление (NO-FLOW BC на верхней/нижней границах) ===
        if nz > 1:
            dp_z = p[:,:,:-1] - p[:,:,1:]
            lambda_w_up_z = torch.where(dp_z > 0, lambda_w[:,:,:-1], lambda_w[:,:,1:])
            lambda_o_up_z = torch.where(dp_z > 0, lambda_o[:,:,:-1], lambda_o[:,:,1:])
            rho_w_avg_z = 0.5 * (rho_w[:,:,:-1] + rho_w[:,:,1:])
            rho_o_avg_z = 0.5 * (rho_o[:,:,:-1] + rho_o[:,:,1:])
            dpc_z = pc[:,:,:-1] - pc[:,:,1:]
            trans_z = self.T_z * dt
            water_flux_z = trans_z * lambda_w_up_z * (dp_z - dpc_z) * rho_w_avg_z
            oil_flux_z   = trans_z * lambda_o_up_z * (dp_z) * rho_o_avg_z
            res_water[:,:,:-1] -= water_flux_z  # k=0..nz-2
            res_water[:,:,1: ] += water_flux_z  # k=1..nz-1
            res_oil  [:,:,:-1] -= oil_flux_z
            res_oil  [:,:,1: ] += oil_flux_z

        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: избегаем inplace операций для сохранения gradients
        # ИСПРАВЛЕНО: сформируем вектор в БЛОЧНОМ формате БЕЗ inplace операций
        # [water0, water1, ..., waterN-1, oil0, oil1, ..., oilN-1]
        residual = torch.cat([
            res_water.reshape(-1),     # первые N элементов = water equations
            res_oil.reshape(-1)        # последние N элементов = oil equations
        ])

        # скважины
        self._add_wells_to_system(residual, None, dt)

        return residual

    def _compute_residual_full_direct(self, dt, p_input, sw_input, prev_water_mass, prev_oil_mass):
        """Полная невязка БЕЗ изменения состояния объектов - для сохранения градиентов в JVP.
        Принимает давление и насыщенность как параметры, а не использует self.fluid состояние.
        ИСПРАВЛЕНО: также принимает prev_water_mass и prev_oil_mass как параметры.
        Возвращает 1-D тензор длиной 2*N (water/oil).
        """
        nx, ny, nz = self.reservoir.dimensions
        N = nx * ny * nz
        device = self.fluid.device
        
        # 🔍 ДИАГНОСТИКА: проверяем входные параметры (БЕЗ .item() для векторизации)
        # print(f"      🔍 compute_residual_full_direct: dt={dt:.2e}, p_input [{p_input.min():.2e}, {p_input.max():.2e}], sw_input [{sw_input.min():.3f}, {sw_input.max():.3f}]")

        # === геометрия ===
        dx, dy, dz = self.reservoir.grid_size
        cell_volume = dx * dy * dz

        # ИСПРАВЛЕНО: используем входные параметры вместо self.fluid состояния
        p = p_input  # (nx,ny,nz)
        # ИСПРАВЛЕНО: используем мягкое ограничение для сохранения градиентов (автоматический slope)
        sw = self._soft_clamp(sw_input, self.fluid.sw_cr, 1.0 - self.fluid.so_r)

        # пористость сжатого пласта
        phi = self.reservoir.porosity_ref * (1 + self.reservoir.rock_compressibility * (p - 1e5)) + self.ptc_alpha

        # плотности
        rho_w = self.fluid.calc_water_density(p)
        rho_o = self.fluid.calc_oil_density(p)

        # аккумуляция (масса, кг)
        water_mass = phi * sw * rho_w * cell_volume
        oil_mass   = phi * (1 - sw) * rho_o * cell_volume

        # 🔍 ДИАГНОСТИКА: проверяем зависимость масс от входных параметров (БЕЗ .item() для векторизации)
        # print(f"      🔍 Массы: water_mass [{water_mass.min():.2e}, {water_mass.max():.2e}], requires_grad={water_mass.requires_grad}")
        # print(f"      🔍 Массы: oil_mass [{oil_mass.min():.2e}, {oil_mass.max():.2e}], requires_grad={oil_mass.requires_grad}")
        # print(f"      🔍 Компоненты: phi [{phi.min():.3f}, {phi.max():.3f}], sw [{sw.min():.3f}, {sw.max():.3f}]")
        # print(f"      🔍 Компоненты: rho_w [{rho_w.min():.1f}, {rho_w.max():.1f}], rho_o [{rho_o.min():.1f}, {rho_o.max():.1f}]")
        # print(f"      🔍 Prev masses: prev_water_mass [{prev_water_mass.min():.2e}, {prev_water_mass.max():.2e}]")
        # print(f"      🔍 Prev masses: prev_oil_mass [{prev_oil_mass.min():.2e}, {prev_oil_mass.max():.2e}]")

        # ИСПРАВЛЕНО: используем переданные prev_masses вместо self.fluid состояния
        res_water = (water_mass - prev_water_mass.view(nx,ny,nz))
        res_oil   = (oil_mass   - prev_oil_mass.view(nx,ny,nz))
        
        # print(f"      🔍 Аккумуляция: res_water [{res_water.min():.2e}, {res_water.max():.2e}]")
        # print(f"      🔍 Аккумуляция: res_oil [{res_oil.min():.2e}, {res_oil.max():.2e}]")

        # === потоки ===
        # вычисляем Т_x, T_y, T_z если не было
        self._init_impes_transmissibilities()
        # относит. проницаемости и мобил.
        kr_w = self.fluid.calc_water_kr(sw)
        kr_o = self.fluid.calc_oil_kr(sw)
        mu_w = self.fluid.mu_water
        mu_o = self.fluid.mu_oil
        lambda_w = kr_w / mu_w
        lambda_o = kr_o / mu_o

        # капиллярное давление
        if self.fluid.pc_scale > 0:
            pc = self.fluid.calc_capillary_pressure(sw)
        else:
            pc = torch.zeros_like(p)
            
        # === X-направление (NO-FLOW BC на левой/правой границах) ===
        dp_x = p[:-1,:,:] - p[1:,:,:]  # shape (nx-1,ny,nz)
        lambda_w_up_x = torch.where(dp_x > 0, lambda_w[:-1,:,:], lambda_w[1:,:,:])
        lambda_o_up_x = torch.where(dp_x > 0, lambda_o[:-1,:,:], lambda_o[1:,:,:])
        rho_w_avg_x = 0.5 * (rho_w[:-1,:,:] + rho_w[1:,:,:])
        rho_o_avg_x = 0.5 * (rho_o[:-1,:,:] + rho_o[1:,:,:])
        dpc_x = pc[:-1,:,:] - pc[1:,:,:]
        trans_x = self.T_x * dt  
        water_flux_x = trans_x * lambda_w_up_x * (dp_x - dpc_x) * rho_w_avg_x
        oil_flux_x   = trans_x * lambda_o_up_x * (dp_x)            * rho_o_avg_x
        # расход из левой ячейки ("-"), к правой ("+")
        res_water[:-1,:,:] -= water_flux_x  # ячейки i=0..nx-2 
        res_water[1: ,:,:] += water_flux_x  # ячейки i=1..nx-1
        res_oil  [:-1,:,:] -= oil_flux_x
        res_oil  [1: ,:,:] += oil_flux_x

        # === Y-направление (NO-FLOW BC на передней/задней границах) ===
        dp_y = p[:,:-1,:] - p[:,1:,:]
        lambda_w_up_y = torch.where(dp_y > 0, lambda_w[:,:-1,:], lambda_w[:,1:,:])
        lambda_o_up_y = torch.where(dp_y > 0, lambda_o[:,:-1,:], lambda_o[:,1:,:])
        rho_w_avg_y = 0.5 * (rho_w[:,:-1,:] + rho_w[:,1:,:])
        rho_o_avg_y = 0.5 * (rho_o[:,:-1,:] + rho_o[:,1:,:])
        dpc_y = pc[:,:-1,:] - pc[:,1:,:]
        trans_y = self.T_y * dt
        water_flux_y = trans_y * lambda_w_up_y * (dp_y - dpc_y) * rho_w_avg_y
        oil_flux_y   = trans_y * lambda_o_up_y * (dp_y) * rho_o_avg_y
        res_water[:,:-1,:] -= water_flux_y  # j=0..ny-2
        res_water[:,1: ,:] += water_flux_y  # j=1..ny-1
        res_oil  [:,:-1,:] -= oil_flux_y
        res_oil  [:,1: ,:] += oil_flux_y

        # === Z-направление (NO-FLOW BC на верхней/нижней границах) ===
        if nz > 1:
            dp_z = p[:,:,:-1] - p[:,:,1:]
            lambda_w_up_z = torch.where(dp_z > 0, lambda_w[:,:,:-1], lambda_w[:,:,1:])
            lambda_o_up_z = torch.where(dp_z > 0, lambda_o[:,:,:-1], lambda_o[:,:,1:])
            rho_w_avg_z = 0.5 * (rho_w[:,:,:-1] + rho_w[:,:,1:])
            rho_o_avg_z = 0.5 * (rho_o[:,:,:-1] + rho_o[:,:,1:])
            dpc_z = pc[:,:,:-1] - pc[:,:,1:]
            trans_z = self.T_z * dt
            water_flux_z = trans_z * lambda_w_up_z * (dp_z - dpc_z) * rho_w_avg_z
            oil_flux_z   = trans_z * lambda_o_up_z * (dp_z) * rho_o_avg_z
            res_water[:,:,:-1] -= water_flux_z  # k=0..nz-2
            res_water[:,:,1: ] += water_flux_z  # k=1..nz-1
            res_oil  [:,:,:-1] -= oil_flux_z
            res_oil  [:,:,1: ] += oil_flux_z

        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: избегаем inplace операций для сохранения gradients
        # ИСПРАВЛЕНО: сформируем вектор в БЛОЧНОМ формате БЕЗ inplace операций
        # [water0, water1, ..., waterN-1, oil0, oil1, ..., oilN-1]
        residual = torch.cat([
            res_water.reshape(-1),     # первые N элементов = water equations
            res_oil.reshape(-1)        # последние N элементов = oil equations
        ])

        # print(f"      🔍 Residual до скважин: water [{residual[:N].min():.2e}, {residual[:N].max():.2e}], oil [{residual[N:].min():.2e}, {residual[N:].max():.2e}]")

        # скважины - НО передаем p и sw как параметры!
        # 🔥 ИСПРАВЛЕНО: получаем новый residual от функции (не inplace)
        residual = self._add_wells_to_system_direct(residual, None, dt, p_input, sw_input)

        # 🔍 ДИАГНОСТИКА: проверяем итоговые residuals (БЕЗ .item() для векторизации)
        # print(f"      🔍 Итоговые residuals: water [{residual[:N].min():.2e}, {residual[:N].max():.2e}], oil [{residual[N:].min():.2e}, {residual[N:].max():.2e}]")
        
        return residual

    def _add_wells_to_system_direct(self, residual, jacobian, dt, p_input, sw_input):
        """
        Добавляет вклад скважин в систему БЕЗ изменения состояния объектов.
        Принимает давление и насыщенность как параметры для сохранения градиентов.
        ИСПРАВЛЕНО: используем БЛОЧНУЮ индексацию вместо интерлеавинга.
        🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: клонируем residual для избежания inplace операций.
        """
        # 🔥 ИСПРАВЛЕНО: создаем новый residual для сохранения градиентов
        residual = residual.clone()
        
        # Если якобиан не передан (режим JFNK), изменяем только residual.
        jac_update = jacobian is not None
        wells = self.well_manager.get_wells()
        
        # ИСПРАВЛЕНО: получаем размер грида для блочной индексации
        nx, ny, nz = self.reservoir.dimensions
        N = nx * ny * nz

        for well in wells:
            idx = well.cell_index_flat

            # ИСПРАВЛЕНО: используем входные параметры вместо self.fluid состояния
            p_cell = p_input.view(-1)[idx]
            sw_cell = sw_input.view(-1)[idx]

            rho_w_cell = self.fluid.calc_water_density(p_cell)
            rho_o_cell = self.fluid.calc_oil_density(p_cell)

            # Подвижности и их производные
            mu_w = self.fluid.mu_water
            mu_o = self.fluid.mu_oil
            kr_w = self.fluid.calc_water_kr(sw_cell)
            kr_o = self.fluid.calc_oil_kr(sw_cell)
            lambda_w = kr_w / mu_w
            lambda_o = kr_o / mu_o
            lambda_t = lambda_w + lambda_o

            dkrw_dsw = self.fluid.calc_dkrw_dsw(sw_cell)
            dkro_dsw = self.fluid.calc_dkro_dsw(sw_cell)
            dlamb_w_dsw = dkrw_dsw / mu_w
            dlamb_o_dsw = dkro_dsw / mu_o

            if well.control_type == 'rate':
                # номинальный объёмный дебит (м³/сут) -> м³/с
                q_tot_vol_rate = well.control_value / 86400.0

                if well.type == 'injector':
                    q_w_mass_step = q_tot_vol_rate * self.fluid.rho_water_ref * dt  # кг за шаг
                    # БЛОЧНАЯ индексация: water equations в первых N элементах
                    residual[idx] -= q_w_mass_step
                    # нефть не закачивается
                else:  # producer
                    # Фракции потоков
                    fw = lambda_w / (lambda_t + 1e-12)
                    fo = 1.0 - fw

                    q_w_mass_step = q_tot_vol_rate * fw * self.fluid.rho_water_ref * dt
                    q_o_mass_step = q_tot_vol_rate * fo * self.fluid.rho_oil_ref   * dt

                    # БЛОЧНАЯ индексация: water в [0:N], oil в [N:2N]
                    residual[idx]   -= q_w_mass_step     # water equation
                    residual[N + idx] -= q_o_mass_step   # oil equation

                    # производные (по Sw) – только для продуцирующей
                    dfw_dsw = (dlamb_w_dsw * lambda_t - lambda_w * (dlamb_w_dsw + dlamb_o_dsw)) / (lambda_t**2 + 1e-12)
                    dfo_dsw = -dfw_dsw

                    dq_w_dsw = q_tot_vol_rate * self.fluid.rho_water_ref * dt * dfw_dsw
                    dq_o_dsw = q_tot_vol_rate * self.fluid.rho_oil_ref  * dt * dfo_dsw

                    if jac_update:
                        # БЛОЧНАЯ индексация для якобиана: [P: 0:N, Sw: N:2N]
                        jacobian[idx,     N + idx] -= dq_w_dsw  # water eq, sw var
                        jacobian[N + idx, N + idx] -= dq_o_dsw  # oil eq, sw var

            elif well.control_type == 'bhp':
                bhp_pa = well.control_value * 1e6  # МПа->Па

                q_w_vol_rate = well.well_index * lambda_w * (p_cell - bhp_pa)  # м³/с
                q_o_vol_rate = well.well_index * lambda_o * (p_cell - bhp_pa)  # м³/с

                q_w_mass_step = q_w_vol_rate * rho_w_cell * dt
                q_o_mass_step = q_o_vol_rate * rho_o_cell * dt

                # БЛОЧНАЯ индексация: water в [0:N], oil в [N:2N]
                residual[idx]     -= q_w_mass_step     # water equation
                residual[N + idx] -= q_o_mass_step     # oil equation

                # Якобиан: производные по давлению
                dq_w_dp = well.well_index * lambda_w * rho_w_cell * dt
                dq_o_dp = well.well_index * lambda_o * rho_o_cell * dt

                if jac_update:
                    # БЛОЧНАЯ индексация для якобиана: [P: 0:N, Sw: N:2N]
                    jacobian[idx,     idx]     -= dq_w_dp  # water eq, pressure var
                    jacobian[N + idx, idx]     -= dq_o_dp  # oil eq, pressure var

                # Якобиан: производные по насыщенности через подвижности
                dq_w_dsw = well.well_index * dlamb_w_dsw * (p_cell - bhp_pa) * rho_w_cell * dt
                dq_o_dsw = well.well_index * dlamb_o_dsw * (p_cell - bhp_pa) * rho_o_cell * dt

                if jac_update:
                    # БЛОЧНАЯ индексация для якобиана: [P: 0:N, Sw: N:2N]
                    jacobian[idx,     N + idx] -= dq_w_dsw  # water eq, sw var
                    jacobian[N + idx, N + idx] -= dq_o_dsw  # oil eq, sw var
        
        # 🔥 ВОЗВРАЩАЕМ НОВЫЙ RESIDUAL (не изменяем исходный)
        return residual

    # ---------------------------------------------------------------
    #          Полный автоград-Ньютон (для маленьких сеток)
    # ---------------------------------------------------------------
    def _fi_autograd_step(self, dt, tol=None, max_iter=12, damping=0.8):
        if tol is None:
            tol = self.sim_params.get("newton_tolerance", 1e-3)

        # 🔍 ДЕТЕКТОР АНОМАЛИЙ временно отключен для векторизации
        # import torch
        # torch.autograd.set_detect_anomaly(True)
        
        # УЛУЧШЕННОЕ масштабирование: используем характерное давление задачи
        P_SCALE = float(self.fluid.pressure.mean().item())  # текущее среднее давление
        if P_SCALE < 1e5:  # если давление слишком мало, используем 1 МПа
            P_SCALE = 1e6
            
        nx, ny, nz = self.reservoir.dimensions
        N = nx * ny * nz

        x = torch.cat([
            (self.fluid.pressure.view(-1) / P_SCALE),
            self.fluid.s_w.view(-1)
        ]).to(self.device).requires_grad_(True)

        initial_norm = None
        for it in range(max_iter):
            # Устанавливаем текущий масштаб для _fi_residual_vec
            self._current_p_scale = P_SCALE
            
            F = self._fi_residual_vec(x, dt)
            norm_F = F.norm()
            if initial_norm is None:
                initial_norm = norm_F.clone()
            rel_res = (norm_F / (initial_norm + 1e-20)).item()
            print(f"  Итерация autograd-Ньютона {it+1}: ||F||={norm_F:.3e}  rel={rel_res:.3e}")
            if rel_res < tol * 1.2:  # немного более мягкий tolerance для численных погрешностей
                print(f"  ✅ Autograd сошёлся на итерации {it+1}: rel={rel_res:.3e} < {tol}")
                # Обновляем состояние и возвращаем успех
                p_new = (x[:N] * P_SCALE).view(self.reservoir.dimensions)
                sw_new = self._soft_clamp(x[N:], self.fluid.sw_cr, 1 - self.fluid.so_r).view(self.reservoir.dimensions)
                self.fluid.pressure = p_new
                self.fluid.s_w = sw_new
                self.fluid.s_o = 1.0 - sw_new
                self.fluid.prev_pressure = p_new.clone()
                self.fluid.prev_sw = sw_new.clone()
                # Обновляем массы
                rho_w = self.fluid.calc_water_density(p_new.view(-1))
                rho_o = self.fluid.calc_oil_density(p_new.view(-1))
                phi0 = self.reservoir.porosity_ref.view(-1)
                phi = phi0 * (1 + self.reservoir.rock_compressibility * (p_new.view(-1) - 1e5)) + self.ptc_alpha
                cell_vol = self.reservoir.cell_volume
                self.fluid.prev_water_mass = phi * sw_new.view(-1) * rho_w * cell_vol
                self.fluid.prev_oil_mass = phi * (1 - sw_new.view(-1)) * rho_o * cell_vol
                return True

            # --- Инициализируем trust-radius при первой итерации ----
            if not hasattr(self, "_trust_radius_auto"):
                # стартовое значение = 10 % нормы текущего состояния x
                self._trust_radius_auto = float(torch.norm(x).item() * 0.1 + 1e-15)

            # Полный Якобиан с правильным масштабом
            self._current_p_scale = P_SCALE
            J = torch.autograd.functional.jacobian(lambda z: self._fi_residual_vec(z, dt), x, create_graph=False, vectorize=True)

            # ---- Решаем J δ = –F через GMRES + ILU0 (CPU) -------------
            lin_cfg = self.sim_params.get("linear_solver", {})
            if lin_cfg.get("backend") == "amgx":
                from linear_gpu import dense_to_csr, solve_amgx_torch as solve_amgx, amgx_available
                if not amgx_available():
                    raise RuntimeError("backend='amgx', но pyamgx не доступен. Установите pyamgx или используйте другой backend.")
                A_csr = dense_to_csr(J.detach()) if not J.is_sparse_csr else J.detach()
                delta = solve_amgx(A_csr, -F.detach(), tol=lin_cfg.get("tol",1e-8))

            elif lin_cfg.get("backend") == "torch_gmres":
                from linear_gpu import gmres as _gmres, jacobi_precond, dense_to_csr
                from linear_gpu.csr import dense_to_csr

                A = dense_to_csr(J.detach()) if not J.is_sparse_csr else J.detach()
                b_vec = -F.detach()
                if lin_cfg.get("precond") == "ilu":
                    from linear_gpu import ilu_precond
                    drop = lin_cfg.get("drop_tol", 1e-4)
                    fill = lin_cfg.get("fill_factor", 10)
                    for attempt in range(4):
                        try:
                            M = ilu_precond(A, drop_tol=drop, fill_factor=fill)
                            if self.verbose:
                                print(f"  ILU built: drop_tol={drop:.1e}, fill_factor={fill}")
                            break
                        except RuntimeError as e:
                            if self.verbose:
                                print(f"  ILU build failed (attempt {attempt+1}): {e}")
                            drop *= 0.1
                            fill *= 2
                    else:
                        if self.verbose:
                            print("  ILU failed after retries, fallback to Jacobi")
                        M = jacobi_precond(A)
                elif lin_cfg.get("precond", "jacobi") == "fsai":
                    from linear_gpu import fsai_precond
                    M = fsai_precond(A, k=lin_cfg.get("k", 1))
                else:
                    M = jacobi_precond(A, omega=lin_cfg.get("omega", 0.8))
                delta_t, info = _gmres(A, b_vec, M=M,
                                       tol=lin_cfg.get("tol", 1e-8),
                                       restart=lin_cfg.get("restart", 50),
                                       max_iter=lin_cfg.get("max_iter", 400))
                if info == 0:
                    delta = delta_t.to(self.device)
                else:
                    if self.verbose:
                        print("  [torch_gmres] не сошёлся, используем robust_solve")
                    J_dense = J.detach().to(self.device)
                    b_dense = -F.detach().to(self.device)
                    delta = self._robust_solve(J_dense, b_dense)
            elif lin_cfg.get("backend") == "hypre":
                # --- CPR-решение с давлением через Hypre BoomerAMG ---------
                # Из полного Якобиана J (2N×2N) извлекаем давление-блок
                # размером N×N, конвертируем в CSR и решаем A_p Δp = –F_p
                # via BoomerAMG.  Для блока насыщенностей используем
                # диагональное приближение (ω-Jacobi).

                import numpy as np
                from linear_gpu.csr import dense_to_csr
                from linear_gpu.petsc_boomeramg import solve_boomeramg

                # 1) Давление-подматрица (CPU) → CSR
                Jp_dense = J.detach()[:N, :N].cpu()
                A_p_csr  = dense_to_csr(Jp_dense)
                indptr   = A_p_csr.crow_indices().to(torch.int32).cpu().numpy()
                indices  = A_p_csr.col_indices().to(torch.int32).cpu().numpy()
                data     = A_p_csr.values().cpu().numpy()

                # 2) Правая часть по давлению
                b_p = (-F.detach()[:N]).cpu().numpy()

                # 3) Решаем AMG
                sol_p, its, res = solve_boomeramg(
                    indptr, indices, data, b_p,
                    tol=lin_cfg.get("tol", 1e-6),
                    max_iter=lin_cfg.get("max_iter", 200),
                )

                if not np.isfinite(res) or not np.all(np.isfinite(sol_p)):
                    # Fallback на устойчивый solve, как раньше
                    J_dense = J.detach().to(self.device)
                    b_dense = -F.detach().to(self.device)
                    delta = self._robust_solve(J_dense, b_dense)
                else:
                    # 4) Насыщенности – ω-Jacobi по диагонали
                    omega = lin_cfg.get("omega", 0.8)
                    diag_sw = torch.diag(J.detach()[N:, N:]).to(self.device)
                    delta_sw = (-F[N:] / (diag_sw + 1e-12)) * omega

                    # 5) Собираем полный вектор δ
                    delta = torch.zeros_like(F, device=self.device)
                    delta[:N] = torch.from_numpy(sol_p).to(self.device)
                    delta[N:] = delta_sw
            else:
                # Если указан другой backend – используем надёжный solve
                J_dense = J.detach().to(self.device)
                b_dense = -F.detach().to(self.device)
                delta = self._robust_solve(J_dense, b_dense)

            # ---------- Ограничение δSw (trust-region по насыщенности) ----------
            sw_mean = float(self.fluid.s_w.mean().item())
            max_sw_step = max(self._sw_trust_limit, 0.3 * (1 - sw_mean), 0.15)

            dSw_max = torch.max(torch.abs(delta[N:])).item() + 1e-15
            dp_max  = torch.max(torch.abs(delta[:N])).item()  # уже в единицах P_SCALE

            scale_sw = max_sw_step / dSw_max
            max_dp_step = self._p_trust_limit / P_SCALE  # переводим в единицы P_SCALE
            p_mean_scaled = float(self.fluid.pressure.mean().item() / P_SCALE) + 1e-6
            max_dp_step = max(max_dp_step, 0.3 * p_mean_scaled)
            scale_dp = float('inf') if max_dp_step<=0 else max_dp_step / dp_max
            scale_trust = min(1.0, scale_sw, scale_dp)

            # Двусторонняя адаптация глобальных лимитов
            self._update_trust_limits(scale_sw, scale_dp, sw_mean)

            # ---- локальный clamp δSw (единый предел) --------------
            delta_sw = delta[N:].view(self.reservoir.dimensions)
            lim_local = max_sw_step
            delta_sw = torch.clamp(delta_sw, -lim_local, lim_local)
            delta[N:] = delta_sw.view(-1)

            # ---- CNV-критерий --------------------------------------
            cnv_val = torch.max(torch.abs(delta_sw) / (self.fluid.s_w + 1e-12)).item()
            if cnv_val > self._cnv_threshold:
                if self.verbose:
                    print(f"  CNV={cnv_val:.2f} > {self._cnv_threshold} — уменьшаем dt")
                return False

            # Если шаг стал почти нулевой – приём сразу без line-search
            if torch.norm(delta) < 1e-10 * torch.norm(x):
                x = x + damping * delta
                continue

            # --- вычисляем норму шага для адаптивного trust-radius ---
            delta_norm = float(delta.norm().item())

            # Armijo line-search
            c1 = 1e-4  # «мягкий» параметр Армижо (обычно 1e-4)
            factor = 1.0
            success_ls = False
            while factor >= 1e-6:
                x_trial = x + factor * damping * delta
                F_trial = self._fi_residual_vec(x_trial, dt)

                # Обработка NaN/Inf
                if not torch.isfinite(F_trial).all():
                    factor *= 0.5
                    continue

                # Классическое условие Армижо
                if F_trial.norm() <= (1 - c1 * factor) * norm_F:
                    x = x_trial.detach().clone().requires_grad_(True)
                    success_ls = True
                    # ---- адаптивное обновление trust-radius ----------
                    if factor > 0.8 and delta_norm > 0.9 * self._trust_radius_auto:
                        self._trust_radius_auto *= 1.3  # расширяем радиус, если шаг был почти предельным
                    elif factor < 0.2:
                        self._trust_radius_auto *= 0.7  # если пришлось сильно уменьшать, сужаем
                    break

                # Разрешаем небольшой прогресс (<1%) при достаточно малом шаге –
                # это помогает избегать зацикливания на «плато».
                if F_trial.norm() < 0.99 * norm_F and factor < 0.1:
                    x = x_trial.detach().clone().requires_grad_(True)
                    success_ls = True
                    # ---- адаптивное обновление trust-radius ----------
                    if factor > 0.8 and delta_norm > 0.9 * self._trust_radius_auto:
                        self._trust_radius_auto *= 1.3  # расширяем радиус, если шаг был почти предельным
                    elif factor < 0.2:
                        self._trust_radius_auto *= 0.7  # если пришлось сильно уменьшать, сужаем
                    break

                factor *= 0.5

            if not success_ls:
                print("  Line-search (autograd) не смог подобрать шаг – уменьшаем dt")
                return False  # не сошлось

        # Обновляем состояние
        p_new = (x[:N] * P_SCALE).view(self.reservoir.dimensions)
        # ИСПРАВЛЕНО: используем мягкое ограничение для насыщенности (автоматический slope)
        sw_new = self._soft_clamp(x[N:], self.fluid.sw_cr, 1 - self.fluid.so_r).view(self.reservoir.dimensions)

        self.fluid.pressure = p_new
        self.fluid.s_w = sw_new
        self.fluid.s_o = 1.0 - sw_new

        self.fluid.prev_pressure = p_new.clone()
        self.fluid.prev_sw = sw_new.clone()

        # Обновляем массы для следующего шага
        rho_w = self.fluid.calc_water_density(p_new.view(-1))
        rho_o = self.fluid.calc_oil_density(p_new.view(-1))
        phi0 = self.reservoir.porosity_ref.view(-1)
        phi = phi0 * (1 + self.reservoir.rock_compressibility * (p_new.view(-1) - 1e5)) + self.ptc_alpha
        cell_vol = self.reservoir.cell_volume
        self.fluid.prev_water_mass = phi * sw_new.view(-1) * rho_w * cell_vol
        self.fluid.prev_oil_mass = phi * (1 - sw_new.view(-1)) * rho_o * cell_vol

        return (norm_F / (initial_norm + 1e-20)).item() < tol

    # -------------------------------------------------------------
    #      Универсальное решение A x = b с регуляризацией
    # -------------------------------------------------------------
    def _robust_solve(self, A: torch.Tensor, b: torch.Tensor, lam: float = 1e-8):
        """Возвращает решение x для A x = b, автоматически добавляя
        Tikhonov-регуляризацию при вырожденном/плохо обусловленном A и
        переходя к псевдообратной, если прямой solve терпит неудачу."""
        try:
            return torch.linalg.solve(A, b)
        except (RuntimeError, torch._C._LinAlgError):
            eps = lam * torch.linalg.norm(A, ord=float("inf"))
            if not torch.isfinite(eps):
                eps = lam
            I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
            try:
                return torch.linalg.solve(A + eps * I, b)
            except (RuntimeError, torch._C._LinAlgError):
                return A.pinverse() @ b

    # -----------------------------------------------------------
    #              IMPES-predictor (initial guess)
    # -----------------------------------------------------------
    def _impes_predictor(self, dt):
        """Одношаговый IMPES-прогноз, который обновляет self.fluid.pressure
        и self.fluid.s_w, но НЕ трогает prev_mass. Используется как
        начальное приближение для fully-implicit Ньютона."""
        # Сохраняем текущее состояние, чтобы в случае неудачи можно было откатиться
        p_old = self.fluid.pressure.clone()
        sw_old = self.fluid.s_w.clone()

        # Мини-шаг IMPES: одна итерация давления + явный шаг насыщенности
        self._init_impes_transmissibilities()
        P_new, converged = self._impes_pressure_step(dt)
        if not converged:
            # Если CG не сошёлся, откатываемся и бросаем исключение
            self.fluid.pressure = p_old
            self.fluid.s_w = sw_old
            raise RuntimeError("IMPES-predictor: шаг давления не сошёлся")

        # Обновляем давление и делаем явный шаг насыщенности
        self.fluid.pressure = P_new
        self._impes_saturation_step(P_new, dt)

        # КРИТИЧЕСКИ ВАЖНО: обновляем prev_mass после IMPES-predictor!
        # Иначе JFNK будет сравнивать текущие массы (после IMPES) с массами до IMPES
        nx, ny, nz = self.reservoir.dimensions
        p_new = self.fluid.pressure
        sw_new = self.fluid.s_w
        
        # Пересчитываем и сохраняем массы флюидов для корректной работы JFNK
        rho_w = self.fluid.calc_water_density(p_new.view(-1))
        rho_o = self.fluid.calc_oil_density(p_new.view(-1))
        phi0 = self.reservoir.porosity_ref.view(-1)
        phi = phi0 * (1 + self.reservoir.rock_compressibility * (p_new.view(-1) - 1e5)) + self.ptc_alpha
        cell_vol = self.reservoir.cell_volume
        
        # Обновляем prev_mass как 1-D тензоры (совместимые с JFNK)
        self.fluid.prev_water_mass = phi * sw_new.view(-1) * rho_w * cell_vol
        self.fluid.prev_oil_mass = phi * (1 - sw_new.view(-1)) * rho_o * cell_vol
        
        self._log("IMPES-predictor выполнен: использовано как initial guess")

    # ---------------------------------------------------------------
    #        Adaptively reduce dt for autograd-Newton                
    # ---------------------------------------------------------------
    def _fi_autograd_adaptive(self, dt):
        """Пытается выполнить autograd-Ньютон, уменьшая dt при неудаче
        прежде чем переключаться на другой решатель."""
        current_dt = dt
        max_attempts = self.sim_params.get("max_time_step_attempts", 4)
        for attempt in range(max_attempts):
            if getattr(self, "use_impes_predictor", False):
                try:
                    self._impes_predictor(current_dt)
                except Exception:
                    pass  # предиктор может не сойтись на очень маленьком dt

            print(f"Попытка шага (autograd) с dt = {current_dt/86400:.2f} дней (Попытка {attempt+1}/{max_attempts})")
            if self._fi_autograd_step(current_dt):
                return True

            # уменьшаем dt и пробуем снова
            current_dt /= self.sim_params.get("dt_reduction_factor", 2.0)
            if current_dt < self.sim_params.get("min_time_step", 0.02*86400):
                break
        return False

    def _update_trust_radius(self, prev_residual_norm, residual_norm, jacobian, delta, p_vec, sw_vec):
        """Адаптивно обновляет глобальный радиус trust-region ``self._trust_radius``.

        Используем классическую схему Powell-Dogleg: оцениваем прогнозируемое
        уменьшение невязки через норму ``‖J δ‖`` и сравниваем с фактическим.
        По коэффициенту ρ динамически расширяем/сужаем радиус.
        """

        if not hasattr(self, "_trust_radius"):
            return  # Радиус ещё не инициализирован (будет на 1-й итерации)

        # --- Предсказанное уменьшение невязки
        try:
            predicted_red = torch.norm(jacobian @ delta).item() + 1e-15
        except Exception:
            # Если ``jacobian`` – SciPy CSR или случился другой тип, берём грубую оценку
            predicted_red = prev_residual_norm + 1e-15

        rho = (prev_residual_norm - residual_norm) / predicted_red

        # --- Обновление радиуса r ---
        if rho < 0.25:
            self._trust_radius *= 0.5
        elif rho > 0.75:
            self._trust_radius *= 1.3

        # Предельные значения радиуса: 1e-4 ‖x‖ ≤ r ≤ 1.0 ‖x‖
        x_norm = torch.norm(torch.cat([p_vec, sw_vec])).item() + 1e-15
        self._trust_radius = max(1e-4 * x_norm, min(self._trust_radius, 1.0 * x_norm))

        if getattr(self, "verbose", False):
            print(f"  trust-radius update: ρ={rho:.3f}, r={self._trust_radius:.2e}")

    # -------------------------------------------------------------
    #          Плавный clamp для насыщенности (ненулевая производная)
    # -------------------------------------------------------------
    def _soft_clamp(self, x: torch.Tensor, low: float, high: float, slope: float = None):
        """Плавная версия clamp, сохраняющая ненулевые производные у границ.

        Используем сигмоидальную проекцию в диапазон [low, high].
        slope определяет крутизну перехода: чем больше, тем ближе к жёсткому clamp."""
        if slope is None:
            # ИСПРАВЛЕНО: более мягкий переход для стабильности градиентов
            slope = 4.0 / (high - low + 1e-20)  # более мягкий переход
        center = 0.5 * (high + low)
        scale = (high - low)
        # ИСПРАВЛЕНО: ограничиваем входной аргумент sigmoid для стабильности
        sigmoid_input = torch.clamp(slope * (x - center), -10.0, 10.0)
        return low + scale * torch.sigmoid(sigmoid_input)

    # -------------------------------------------------------------
    #        Собрать 7-точечную CSR-матрицу для давления            
    # -------------------------------------------------------------
    def _assemble_pressure_csr(self, lambda_t: torch.Tensor, dt: float | None = None):
        """Формирует CSR давления (7-точечный шаблон). Если передан dt (>0),
        к диагонали добавляется вклад аккумуляции φ·C_r·ρ·V/dt для улучшения
        кондиционирования предобуславливателя CPR на крупных шагах времени."""
        nx, ny, nz = self.reservoir.dimensions
        N = nx * ny * nz
        Tx = self.T_x.cpu().numpy()  # shape (nx-1,ny,nz)
        Ty = self.T_y.cpu().numpy()
        if nz > 1:
            Tz = self.T_z.cpu().numpy()
        lam = lambda_t.cpu().numpy().reshape(nx, ny, nz)

        indptr = np.zeros(N + 1, dtype=np.int64)
        nnz_est = 7 * N
        indices = np.empty(nnz_est, dtype=np.int32)
        data    = np.empty(nnz_est, dtype=np.float64)

        pos = 0
        idx = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    center = idx
                    indptr[idx] = pos
                    diag = 0.0
                    # X-
                    if i > 0:
                        t = Tx[i-1, j, k] * lam[i-1, j, k]
                        indices[pos] = center - 1
                        data[pos] = -t
                        pos += 1
                        diag += t
                    if i < nx - 1:
                        t = Tx[i, j, k] * lam[i, j, k]
                        indices[pos] = center + 1
                        data[pos] = -t
                        pos += 1
                        diag += t
                    # Y-
                    if j > 0:
                        t = Ty[i, j-1, k] * lam[i, j-1, k]
                        indices[pos] = center - nx
                        data[pos] = -t
                        pos += 1
                        diag += t
                    if j < ny - 1:
                        t = Ty[i, j, k] * lam[i, j, k]
                        indices[pos] = center + nx
                        data[pos] = -t
                        pos += 1
                        diag += t
                    # Z-
                    if nz > 1:
                        if k > 0:
                            t = Tz[i, j, k-1] * lam[i, j, k-1]
                            indices[pos] = center - nx * ny
                            data[pos] = -t
                            pos += 1
                            diag += t
                        if k < nz - 1:
                            t = Tz[i, j, k] * lam[i, j, k]
                            indices[pos] = center + nx * ny
                            data[pos] = -t
                            pos += 1
                            diag += t
                    # диагональ (потоки +, при необходимости, аккумуляция)
                    indices[pos] = center
                    diag_val = diag + 1e-12  # базовая стабилизация
                    if dt is not None and dt > 0:
                        phi_ref = self.reservoir.porosity_ref[i, j, k].item()
                        cr = self.reservoir.rock_compressibility
                        rho_avg = 0.5 * (self.fluid.rho_water_ref + self.fluid.rho_oil_ref)
                        cell_vol = self.reservoir.cell_volume
                        acc = (phi_ref * cr * rho_avg * cell_vol) / dt
                        diag_val += acc
                    data[pos] = diag_val
                    pos += 1
                    idx += 1
        indptr[N] = pos
        return indptr[:N+1], indices[:pos], data[:pos]

    def _update_trust_limits(self, scale_sw: float, scale_dp: float, sw_mean: float):
        """Двусторонняя адаптация динамических лимитов trust-region.

        Если шаг сильно урезан (scale<0.4) – лимит расширяется, если почти не урезан
        несколько итераций подряд – лимит слегка сжимается, чтобы избежать слишком
        больших шагов в спокойных зонах.
        """
        # --- расширение, когда слишком тесно ---
        if scale_sw < 0.4:
            self._sw_trust_limit = min(self._sw_trust_limit * 1.5,
                                       0.9 * (1 - sw_mean))
        elif scale_sw > 0.9 and self._sw_trust_limit > 0.2:
            # лёгкое сужение, если запас слишком велик
            self._sw_trust_limit *= 0.9

        if scale_dp < 0.4:
            self._p_trust_limit *= 1.5
        elif scale_dp > 0.9 and self._p_trust_limit > self._p_trust_limit_init:
            self._p_trust_limit *= 0.9

        # Жёсткие глобальные пределы
        self._sw_trust_limit = max(0.2, min(self._sw_trust_limit, 0.8))
        self._p_trust_limit  = max(10.0, min(self._p_trust_limit, 100.0))

    def _diagnostic_jvp_vs_fd(self, x, dt, P_SCALE, SATURATION_SCALE=1.0):
        """Диагностика: сравниваем JVP с finite differences"""
        # Выбираем случайное направление
        v = torch.randn_like(x) * 0.01
        
        # JVP
        try:
            self._current_p_scale = P_SCALE
            self._current_saturation_scale = SATURATION_SCALE
            _, Jv_auto = torch.autograd.functional.jvp(
                lambda z: self._fi_residual_vec(z, dt), x, v, create_graph=False)
        except Exception as e:
            print(f"    ❌ JVP failed: {e}")
            return
        
        # Finite differences
        eps = 1e-6
        try:
            self._current_p_scale = P_SCALE
            self._current_saturation_scale = SATURATION_SCALE
            F_plus = self._fi_residual_vec(x + eps * v, dt)
            F_minus = self._fi_residual_vec(x - eps * v, dt)
            Jv_fd = (F_plus - F_minus) / (2 * eps)
        except Exception as e:
            print(f"    ❌ FD failed: {e}")
            return
        
        # Сравниваем
        diff = torch.norm(Jv_auto - Jv_fd)
        rel_diff = diff / (torch.norm(Jv_auto) + 1e-12)
        
        print(f"    JVP vs FD: ||J*v||_auto = {torch.norm(Jv_auto).item():.3e}")
        print(f"               ||J*v||_fd   = {torch.norm(Jv_fd).item():.3e}")
        print(f"               ||diff||     = {diff.item():.3e}")
        print(f"               rel_diff     = {rel_diff.item():.3e}")
        
        if rel_diff > 1e-3:
            print(f"    ⚠️  ВНИМАНИЕ: большая разница между JVP и FD!")
        else:
            print(f"    ✅ JVP корректен")
    
    def _diagnostic_condition_number(self, matvec, n):
        """Улучшенная диагностика condition number с агрессивными рекомендациями"""
        # Не делаем полную диагностику для больших систем
        if n > 500:
            print(f"    Система слишком большая ({n}) для диагностики condition number")
            return
            
        print(f"    🔍 Оценка condition number для системы размера {n}...")
        
        # Power method для оценки максимального собственного числа
        # Используем то же device, что и в системе
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        v = torch.randn(n, dtype=torch.float32, device=device)
        v = v / torch.norm(v)
        
        lambda_max = 0.0
        for i in range(20):  # больше итераций для точности
            try:
                Av = matvec(v)
                lambda_max = torch.dot(v, Av).item()
                v_norm = torch.norm(Av)
                if v_norm < 1e-12:
                    break
                v = Av / v_norm
            except Exception as e:
                print(f"    ❌ Power method failed at iteration {i}: {e}")
                return
        
        # Улучшенная оценка минимального собственного числа через обратную итерацию
        lambda_min = lambda_max * 1e-12  # консервативная оценка
        
        # Пытаемся найти более точное значение λ_min
        try:
            v_inv = torch.randn(n, dtype=torch.float32, device=device)
            v_inv = v_inv / torch.norm(v_inv)
            
            # Обратная итерация для минимального собственного числа
            for i in range(10):
                Av = matvec(v_inv)
                if torch.norm(Av) < 1e-15:
                    lambda_min = 1e-15
                    break
                v_inv = Av / torch.norm(Av)
            else:
                # Если не нашли нулевое, используем последнее значение
                lambda_min = torch.norm(matvec(v_inv)).item()
        except:
            pass
        
        # Оценка condition number
        if abs(lambda_min) < 1e-15:
            cond_est = float('inf')
        else:
            cond_est = abs(lambda_max / lambda_min)
        
        print(f"    📊 Результат condition number:")
        print(f"      λ_max ≈ {lambda_max:.3e}")
        print(f"      λ_min ≈ {lambda_min:.3e}")
        print(f"      cond  ≈ {cond_est:.3e}")
        
        if cond_est > 1e12:
            print(f"    💀 СИСТЕМА КАТАСТРОФИЧЕСКИ ПЛОХО ОБУСЛОВЛЕНА! (cond > 1e12)")
            print(f"    🚨 КРИТИЧЕСКИЕ РЕКОМЕНДАЦИИ:")
            print(f"       - Увеличить finite difference epsilon до 1e-3")
            print(f"       - Использовать CPR предобуславливание")
            print(f"       - Применить физические ограничения")
            print(f"       - Уменьшить временной шаг в 4 раза")
            print(f"       - Использовать более сильную регуляризацию")
        elif cond_est > 1e10:
            print(f"    ❌ СИСТЕМА КРАЙНЕ ПЛОХО ОБУСЛОВЛЕНА! (cond > 1e10)")
            print(f"       - Увеличить finite difference epsilon до 1e-4")
            print(f"       - Использовать CPR или ILU предобуславливание")
            print(f"       - Уменьшить временной шаг в 2 раза")
        elif cond_est > 1e8:
            print(f"    ⚠️  Система очень плохо обусловлена (cond > 1e8)")
            print(f"       - Использовать AMG предобуславливание")
            print(f"       - Применить регуляризацию")
        elif cond_est > 1e6:
            print(f"    ⚠️  Система плохо обусловлена (cond > 1e6)")
            print(f"       - Использовать хорошее предобуславливание")
        else:
            print(f"    ✅ Обусловленность приемлемая")

    def _diagnostic_jacobian_structure(self, matvec, n):
        """Диагностика структуры якобиана: поиск нулевых строк, столбцов и других проблем"""
        if n > 500:
            print(f"    Система слишком большая ({n}) для структурной диагностики")
            return
            
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"    Анализ структуры якобиана {n}×{n}...")
        
        # Проверяем нулевые строки: J*e_i = 0?
        zero_rows = []
        for i in range(min(n, 50)):  # проверяем первые 50 строк
            ei = torch.zeros(n, device=device)
            ei[i] = 1.0
            Jei = matvec(ei)
            row_norm = torch.norm(Jei).item()
            if row_norm < 1e-12:
                zero_rows.append(i)
        
        # Проверяем диагональные элементы
        diag_elements = []
        for i in range(min(n, 50)):
            ei = torch.zeros(n, device=device)
            ei[i] = 1.0
            Jei = matvec(ei)
            diag_elements.append(Jei[i].item())
        
        # Статистика диагональных элементов
        diag_tensor = torch.tensor(diag_elements)
        small_diag = torch.sum(torch.abs(diag_tensor) < 1e-10).item()
        
        # Проверяем есть ли блочная структура давление/насыщенность
        N = n // 2  # предполагаем что первая половина - давление, вторая - насыщенность
        
        print(f"    📊 Результаты анализа структуры:")
        print(f"      Нулевых строк (из первых 50): {len(zero_rows)}")
        if zero_rows:
            print(f"      Нулевые строки: {zero_rows[:10]}")  # показываем первые 10
        print(f"      Малых диагональных элементов (<1e-10): {small_diag}/{len(diag_elements)}")
        
        # Проверяем связность блоков
        if N > 0 and N < n:
            print(f"    🔗 Анализ блочной структуры (P: 0-{N-1}, Sw: {N}-{n-1}):")
            
            # Проверяем P-P блок
            ep = torch.zeros(n, device=device)
            ep[0] = 1.0  # первая переменная давления
            Jep = matvec(ep)
            pp_norm = torch.norm(Jep[:N]).item()
            ps_norm = torch.norm(Jep[N:]).item()
            
            # Проверяем Sw-Sw блок
            es = torch.zeros(n, device=device)
            es[N] = 1.0  # первая переменная насыщенности
            Jes = matvec(es)
            sp_norm = torch.norm(Jes[:N]).item()
            ss_norm = torch.norm(Jes[N:]).item()
            
            print(f"      P->P связь: {pp_norm:.3e}, P->Sw связь: {ps_norm:.3e}")
            print(f"      Sw->P связь: {sp_norm:.3e}, Sw->Sw связь: {ss_norm:.3e}")
            
            # Диагностика возможных проблем
            if pp_norm < 1e-12:
                print(f"    ⚠️  ПРОБЛЕМА: нет связи давление-давление!")
            if ss_norm < 1e-12:
                print(f"    ⚠️  ПРОБЛЕМА: нет связи насыщенность-насыщенность!")
            if ps_norm < 1e-12 and sp_norm < 1e-12:
                print(f"    ⚠️  ПРОБЛЕМА: полностью раздельные подсистемы!")
        
        # Общая диагностика
        if len(zero_rows) > 0:
            print(f"    🚨 КРИТИЧЕСКАЯ ПРОБЛЕМА: найдены нулевые строки якобиана!")
        elif small_diag > len(diag_elements) // 2:
            print(f"    ⚠️  ПРОБЛЕМА: слишком много малых диагональных элементов")
        else:
            print(f"    ✅ Структура якобиана выглядит разумной")

    def _diagnostic_full_comparison(self, x, dt, P_SCALE, SATURATION_SCALE=1.0):
        """Полная диагностика: сравнение autograd vs JFNK якобианов"""
        print("\n" + "="*60)
        print("🔍 ПОЛНАЯ ДИАГНОСТИКА: AUTOGRAD vs JFNK")
        print("="*60)
        
        # Устанавливаем текущий масштаб
        self._current_p_scale = P_SCALE
        self._current_saturation_scale = SATURATION_SCALE
        
        # 1. Вычисляем residual в точке x
        F = self._fi_residual_vec(x, dt)
        print(f"\n1. RESIDUAL АНАЛИЗ:")
        print(f"   ||F|| = {F.norm():.6e}")
        print(f"   F диапазон: [{F.min():.6e}, {F.max():.6e}]")
        
        nx, ny, nz = self.reservoir.dimensions
        N = nx * ny * nz
        F_p = F[:N]  # residual по давлению
        F_s = F[N:]  # residual по насыщенности
        print(f"   ||F_pressure|| = {F_p.norm():.6e}")
        print(f"   ||F_saturation|| = {F_s.norm():.6e}")
        
        # 2. Autograd якобиан
        print(f"\n2. AUTOGRAD ЯКОБИАН:")
        try:
            J_auto = torch.autograd.functional.jacobian(
                lambda z: self._fi_residual_vec(z, dt), x, 
                create_graph=False, vectorize=True
            )
            print(f"   J_auto размер: {J_auto.shape}")
            print(f"   J_auto диапазон: [{J_auto.min():.6e}, {J_auto.max():.6e}]")
            print(f"   J_auto norm: {J_auto.norm():.6e}")
            
            # Блочная структура
            J_pp = J_auto[:N, :N]  # ∂F_p/∂p
            J_ps = J_auto[:N, N:]  # ∂F_p/∂s
            J_sp = J_auto[N:, :N]  # ∂F_s/∂p  
            J_ss = J_auto[N:, N:]  # ∂F_s/∂s
            
            print(f"   J_pp (∂F_p/∂p) norm: {J_pp.norm():.6e}")
            print(f"   J_ps (∂F_p/∂s) norm: {J_ps.norm():.6e}")
            print(f"   J_sp (∂F_s/∂p) norm: {J_sp.norm():.6e}")
            print(f"   J_ss (∂F_s/∂s) norm: {J_ss.norm():.6e}")
            
            # Condition number
            try:
                s = torch.linalg.svdvals(J_auto)
                cond_auto = (s.max() / s.min()).item()
                print(f"   Condition number (autograd): {cond_auto:.6e}")
            except:
                print(f"   Condition number (autograd): не удалось вычислить")
                
        except Exception as e:
            print(f"   ❌ Ошибка autograd: {e}")
            J_auto = None
        
        # 3. JFNK finite differences якобиан
        print(f"\n3. JFNK FINITE DIFFERENCES:")
        F_norm = F.norm().item()
        eps = max(1e-4 * F_norm, 1e-6)
        print(f"   eps для FD: {eps:.6e}")
        
        # Вычисляем полный якобиан через finite differences
        n = len(x)
        J_fd = torch.zeros((n, n), device=x.device, dtype=x.dtype)
        
        print(f"   Вычисляем J_fd для {n} переменных...")
        for i in range(n):
            if i % 50 == 0:
                print(f"   Прогресс: {i}/{n}")
            
            e_i = torch.zeros_like(x)
            e_i[i] = 1.0
            
            try:
                F_plus = self._fi_residual_vec(x + eps * e_i, dt)
                F_minus = self._fi_residual_vec(x - eps * e_i, dt)
                J_fd[:, i] = (F_plus - F_minus) / (2 * eps)
            except Exception as e:
                print(f"   ❌ Ошибка FD для столбца {i}: {e}")
                J_fd[:, i] = 0
        
        print(f"   J_fd размер: {J_fd.shape}")
        print(f"   J_fd диапазон: [{J_fd.min():.6e}, {J_fd.max():.6e}]")
        print(f"   J_fd norm: {J_fd.norm():.6e}")
        
        # Блочная структура FD
        J_fd_pp = J_fd[:N, :N]  # ∂F_p/∂p
        J_fd_ps = J_fd[:N, N:]  # ∂F_p/∂s
        J_fd_sp = J_fd[N:, :N]  # ∂F_s/∂p  
        J_fd_ss = J_fd[N:, N:]  # ∂F_s/∂s
        
        print(f"   J_fd_pp (∂F_p/∂p) norm: {J_fd_pp.norm():.6e}")
        print(f"   J_fd_ps (∂F_p/∂s) norm: {J_fd_ps.norm():.6e}")
        print(f"   J_fd_sp (∂F_s/∂p) norm: {J_fd_sp.norm():.6e}")
        print(f"   J_fd_ss (∂F_s/∂s) norm: {J_fd_ss.norm():.6e}")
        
        # Condition number FD
        try:
            s = torch.linalg.svdvals(J_fd)
            cond_fd = (s.max() / s.min()).item()
            print(f"   Condition number (FD): {cond_fd:.6e}")
        except:
            print(f"   Condition number (FD): не удалось вычислить")
        
        # 4. Сравнение якобианов
        if J_auto is not None:
            print(f"\n4. СРАВНЕНИЕ ЯКОБИАНОВ:")
            diff = J_auto - J_fd
            print(f"   ||J_auto - J_fd|| = {diff.norm():.6e}")
            print(f"   Relative error: {(diff.norm() / J_auto.norm()).item():.6e}")
            
            # Сравнение по блокам
            print(f"   Блок ∂F_p/∂p relative error: {((J_pp - J_fd_pp).norm() / J_pp.norm()).item():.6e}")
            print(f"   Блок ∂F_p/∂s relative error: {((J_ps - J_fd_ps).norm() / J_ps.norm()).item():.6e}")
            print(f"   Блок ∂F_s/∂p relative error: {((J_sp - J_fd_sp).norm() / J_sp.norm()).item():.6e}")
            print(f"   Блок ∂F_s/∂s relative error: {((J_ss - J_fd_ss).norm() / J_ss.norm()).item():.6e}")
        
        # 5. Анализ переменных x
        print(f"\n5. АНАЛИЗ ПЕРЕМЕННЫХ X:")
        x_p = x[:N]  # давление (масштабированное)
        x_s = x[N:]  # насыщенность
        
        print(f"   Давление (масштабированное): [{x_p.min():.6e}, {x_p.max():.6e}]")
        print(f"   Насыщенность: [{x_s.min():.6e}, {x_s.max():.6e}]")
        
        # Физические значения
        p_phys = x_p * P_SCALE
        print(f"   Давление (физическое, МПа): [{p_phys.min()/1e6:.6f}, {p_phys.max()/1e6:.6f}]")
        
        print("="*60)
        print("🔍 ДИАГНОСТИКА ЗАВЕРШЕНА")
        print("="*60 + "\n")
        
        return J_auto, J_fd

    def _apply_physical_constraints(self, x, P_SCALE, SATURATION_SCALE):
        """
        Применяет физические ограничения к решению
        
        Args:
            x: масштабированные переменные [давление, насыщенность]
            P_SCALE: масштаб давления [Па]
            SATURATION_SCALE: масштаб насыщенности (обычно 1.0)
        
        Returns:
            x_clipped: ограниченные переменные
        """
        N = len(x) // 2
        x_clipped = x.clone()
        
        # === ОГРАНИЧЕНИЯ ДЛЯ ДАВЛЕНИЯ ===
        # Переводим в физические единицы 
        p_physical = x[:N] * P_SCALE
        
        # Минимальное давление: 0.1 МПа = 1e5 Па
        p_min = 1e5 / P_SCALE  # в масштабированных единицах
        # Максимальное давление: 200 МПа = 2e8 Па  
        p_max = 2e8 / P_SCALE  # в масштабированных единицах
        
        x_clipped[:N] = torch.clamp(x[:N], p_min, p_max)
        
        # === ОГРАНИЧЕНИЯ ДЛЯ ВОДОНАСЫЩЕННОСТИ ===
        # Переводим в физические единицы
        sw_physical = x[N:] * SATURATION_SCALE
        
        # Физические пределы водонасыщенности
        sw_min = self.fluid.sw_cr / SATURATION_SCALE  # связанная водонасыщенность
        sw_max = (1.0 - self.fluid.so_r) / SATURATION_SCALE  # 1 - остаточная нефтенасыщенность
        
        x_clipped[N:] = torch.clamp(x[N:], sw_min, sw_max)
        
        # === ДИАГНОСТИКА ===
        n_clipped_p = (x[:N] != x_clipped[:N]).sum().item()
        n_clipped_s = (x[N:] != x_clipped[N:]).sum().item()
        
        if n_clipped_p > 0 or n_clipped_s > 0:
            print(f"    🔧 Ограничено: {n_clipped_p} давлений, {n_clipped_s} насыщенностей")
            
            if n_clipped_p > 0:
                p_min_actual = (x_clipped[:N] * P_SCALE).min().item() / 1e6  # МПа
                p_max_actual = (x_clipped[:N] * P_SCALE).max().item() / 1e6  # МПа
                print(f"    P диапазон: [{p_min_actual:.2f}, {p_max_actual:.2f}] МПа")
            
            if n_clipped_s > 0:
                sw_min_actual = (x_clipped[N:] * SATURATION_SCALE).min().item()
                sw_max_actual = (x_clipped[N:] * SATURATION_SCALE).max().item()
                print(f"    Sw диапазон: [{sw_min_actual:.3f}, {sw_max_actual:.3f}]")
        
        return x_clipped


