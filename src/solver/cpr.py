import torch, numpy as np
from .amg import BoomerSolver, AmgXSolver
from .geom_amg import GeoSolver
from typing import Optional

class CPRPreconditioner:
    def __init__(self, reservoir, fluid, backend="amgx", omega=0.3, smoother: str = "jacobi"):
        self.backend = backend
        self.omega = omega
        self.failed_amg = False  # Флаг провала AMG
        
        print(f"🔧 CPR: Инициализация с backend='{backend}'")

        # Сохраняем ссылку на reservoir для последующей возможной
        # переинициализации AMG (например, смена сглаживателя).
        self.reservoir = reservoir

        indptr, ind, data = self._assemble_pressure_csr(reservoir, fluid)
        print(f"🔧 CPR: Построена pressure матрица размера {len(indptr)-1}x{len(indptr)-1}, nnz={len(data)}")
        
        # Сохраняем диагональ для Jacobi fallback
        self.diag_inv = self._extract_diagonal_inverse(indptr, ind, data)
        print(f"🔧 CPR: Диагональ для fallback готова")
        
        if backend == "amgx" and AmgXSolver is not None:
            try:
                print(f"🔧 CPR: Пытаемся инициализировать AmgX...")
                self.solver = AmgXSolver(indptr, ind, data)
                print(f"✅ CPR: AmgX инициализирован успешно")
            except Exception as e:
                print(f"❌ CPR: Ошибка инициализации AmgX: {e}")
                self.solver = None
                self.failed_amg = True
        elif backend == "geo":
            try:
                print(f"🔧 CPR: Используем собственный геометрический AMG (GeoSolver, smoother='{smoother}')...")
                self.solver = GeoSolver(reservoir, smoother=smoother)
                # Alias для обратной совместимости
                self.geo_solver = self.solver
                print("✅ CPR: GeoSolver инициализирован успешно")
            except Exception as e:
                print(f"❌ CPR: Ошибка GeoSolver: {e}")
                self.solver = None
                self.failed_amg = True
        elif backend in ("hypre", "boomer", "cpu"):  # BoomerAMG на CPU
            try:
                print(f"🔧 CPR: Пытаемся инициализировать BoomerAMG...")
                print(f"🔧 CPR: CSR matrix: shape=({len(indptr)-1}x{len(indptr)-1}), nnz={len(data)}")
                print(f"🔧 CPR: Matrix range: min={np.min(data):.3e}, max={np.max(data):.3e}")
                
                self.solver = BoomerSolver(indptr, ind, data)
                print(f"✅ CPR: BoomerAMG инициализирован успешно")
            except Exception as e:
                print(f"❌ CPR: Ошибка инициализации BoomerAMG: {e}")
                import traceback
                print(f"❌ CPR: Полный трейс ошибки:")
                traceback.print_exc()
                self.solver = None
                self.failed_amg = True
        else:
            # 'jacobi' или 'none' – не используем AMG
            print(f"🔧 CPR: Использование диагонального предобуславливания (backend='{backend}')")
            self.solver = None
        
        if self.solver is None:
            print(f"⚠️  CPR: Будет использоваться диагональное предобуславливание")
        else:
            print(f"✅ CPR: AMG предобуславливание готово")

    def _extract_diagonal_inverse(self, indptr, indices, data):
        """Извлекает обратную диагональ из CSR матрицы"""
        n = len(indptr) - 1
        diag = np.ones(n)
        
        for i in range(n):
            start, end = indptr[i], indptr[i+1]
            for j in range(start, end):
                if indices[j] == i:  # диагональный элемент
                    diag[i] = 1.0 / max(abs(data[j]), 1e-12)
                    break
        return diag

    def _assemble_pressure_csr(self, reservoir, fluid):
        """Формирует CSR-матрицу (indptr, indices, data) для уравнения
        давления по классическому 7-точечному шаблону.

        Используем гармонические средние проницаемостей для трансмис-
        сибилизаторов и предполагаем постоянную суммарную мобильность
        λ_t = 1/μ_w + 1/μ_o. Такого приближения достаточно для
        предобуславливателя CPR: матрица отражает геометрию сетки и
        контраст проницаемостей, а обновлять её каждый шаг не требуется.
        """

        # --- параметры сетки и проницаемости ---
        nx, ny, nz = reservoir.dimensions
        dx, dy, dz = reservoir.grid_size

        # Переводим из тензоров CUDA/CPU в numpy
        kx = reservoir.permeability_x.detach().cpu().numpy()
        ky = reservoir.permeability_y.detach().cpu().numpy()
        kz = reservoir.permeability_z.detach().cpu().numpy()

        dx = float(dx); dy = float(dy); dz = float(dz)

        # --- transmissibilities по граням ---
        Tx = np.zeros((nx-1, ny, nz), dtype=np.float64)
        for i in range(nx-1):
            k_harm = 2 * kx[i] * kx[i+1] / (kx[i] + kx[i+1] + 1e-15)
            Tx[i] = k_harm * dy * dz / dx

        Ty = np.zeros((nx, ny-1, nz), dtype=np.float64)
        for j in range(ny-1):
            k_harm = 2 * ky[:, j, :] * ky[:, j+1, :] / (ky[:, j, :] + ky[:, j+1, :] + 1e-15)
            Ty[:, j, :] = k_harm * dx * dz / dy

        Tz = np.zeros((nx, ny, nz-1), dtype=np.float64)
        if nz > 1:
            for k in range(nz-1):
                k_harm = 2 * kz[:, :, k] * kz[:, :, k+1] / (kz[:, :, k] + kz[:, :, k+1] + 1e-15)
                Tz[:, :, k] = k_harm * dx * dy / dz

        # --- суммарная мобильность (константа для CPR) ---
        lam_t = 1.0 / fluid.mu_water + 1.0 / fluid.mu_oil  # 1/Па·с
        lam = lam_t  # скаляр
        self.lam_const = lam  # сохраняем для масштабирования AMG результата

        # 🎯 УЛУЧШЕННОЕ МАСШТАБИРОВАНИЕ для высокой сжимаемости
        # Типичная transmissibility
        typical_T = np.mean(kx) * dy * dz / dx * lam
        
        # 🔧 КРИТИЧЕСКОЕ УЛУЧШЕНИЕ: учитываем сжимаемость
        # Получаем характерные значения сжимаемости
        max_compress = max(
            getattr(fluid, 'oil_compressibility', 1e-9),
            getattr(fluid, 'water_compressibility', 1e-9),
            getattr(reservoir, 'rock_compressibility', 1e-9)
        )
        
        # Для высокой сжимаемости нужно более агрессивное масштабирование
        compressibility_factor = max_compress / 1e-9  # Нормализуем к 1e-9
        
        # FIX: отказались от искусственного масштабирования — ставим 1.0.
        # Чрезмерное "растягивание" матрицы приводило к гигантским поправкам δp
        # и к фактическому «заглушению» шагов Ньютона. Более корректно оставить
        # физический масштаб коэффициентов и позволить AMG обрабатывать
        # плохо обусловленную, но реалистичную матрицу.

        matrix_scale = 1.0
        
        print(f"🎯 CPR: Типичная transmissibility: {typical_T:.3e}")
        print(f"🎯 CPR: Максимальная сжимаемость: {max_compress:.3e}")
        print(f"🎯 CPR: Фактор сжимаемости: {compressibility_factor:.3e}")
        print(f"🎯 CPR: Масштаб матрицы: {matrix_scale:.3e} (физический масштаб, без искусственного множителя)")
        
        # Сохраняем масштаб для восстановления решения
        self.matrix_scale = matrix_scale
        self.compressibility_factor = compressibility_factor

        # --- предварительное выделение памяти под CSR ---
        N = nx * ny * nz
        nnz_est = 7 * N
        indptr = np.zeros(N + 1, dtype=np.int64)
        indices = np.empty(nnz_est, dtype=np.int32)
        data = np.empty(nnz_est, dtype=np.float64)

        # Сохраняем диагональные элементы для последующего автомасштабирования
        diag_vals = []

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
                        t = Tx[i-1, j, k] * lam  # self.matrix_scale =1
                        indices[pos] = center - 1
                        data[pos] = -t
                        pos += 1
                        diag += t
                    # X+
                    if i < nx - 1:
                        t = Tx[i, j, k] * lam
                        indices[pos] = center + 1
                        data[pos] = -t
                        pos += 1
                        diag += t
                    # Y-
                    if j > 0:
                        t = Ty[i, j-1, k] * lam
                        indices[pos] = center - nx
                        data[pos] = -t
                        pos += 1
                        diag += t
                    # Y+
                    if j < ny - 1:
                        t = Ty[i, j, k] * lam
                        indices[pos] = center + nx
                        data[pos] = -t
                        pos += 1
                        diag += t
                    # Z-/Z+
                    if nz > 1:
                        if k > 0:
                            t = Tz[i, j, k-1] * lam
                            indices[pos] = center - nx * ny
                            data[pos] = -t
                            pos += 1
                            diag += t
                        if k < nz - 1:
                            t = Tz[i, j, k] * lam
                            indices[pos] = center + nx * ny
                            data[pos] = -t
                            pos += 1
                            diag += t

                    # 🔧 КРИТИЧЕСКОЕ УЛУЧШЕНИЕ: адаптивный стабилизационный сдвиг
                    # Для высокой сжимаемости нужен больший сдвиг
                    base_shift = 1e-12
                    if hasattr(self, 'compressibility_factor'):
                        # Увеличиваем сдвиг пропорционально сжимаемости
                        adaptive_shift = base_shift * max(1.0, self.compressibility_factor ** 0.5)
                    else:
                        adaptive_shift = base_shift
                    
                    # Диагональный элемент
                    indices[pos] = center
                    diag_entry = diag + adaptive_shift  # пока без масштабирования
                    data[pos] = diag_entry
                    diag_vals.append(abs(diag_entry))
                    pos += 1
                    idx += 1

        indptr[N] = pos

        # --- АВТОМАТИЧЕСКОЕ МАСШТАБИРОВАНИЕ МАТРИЦЫ ---
        diag_median = np.median(diag_vals) if diag_vals else 1.0
        # Гарантируем ненулевую диагональ
        if diag_median < 1e-20:
            diag_median = 1e-20
        scale = 1.0 / diag_median
        data[:pos] *= scale  # нормализуем матрицу

        # Сохраняем коэффициент для последующего восстановления решения
        self.matrix_scale = scale

        print(f"🎯 CPR: Автомасштабирование — median(|diag|)={diag_median:.3e}, scale={scale:.3e}")
        print(f"🎯 CPR: Диапазон элементов после масштабирования: min={data[:pos].min():.3e}, max={data[:pos].max():.3e}")

        return indptr[:N+1], indices[:pos], data[:pos]

    def apply(self, vec: torch.Tensor) -> torch.Tensor:
        """🎯 ROBUST CPR предобуславливание с автоматическим масштабированием
        
        CPR применяется как:
        1. Решаем уравнение давления через AMG: A_p * delta_p = rhs_p  
        2. Насыщенность обрабатываем через простое Jacobi масштабирование
        3. Автоматическое масштабирование для робастности
        """
        # Количество ячеек
        if not hasattr(self, "_n_cells"):
            nx, ny, nz = self.reservoir_dims if hasattr(self, "reservoir_dims") else (None, None, None)
        n_cells = getattr(self, "_n_cells", None)
        if n_cells is None:
            # лениво сохраняем на первом вызове
            from math import prod
            try:
                import builtins  # avoid circular if reservoir not passed
                dims = builtins.__dict__.get("_cpr_cached_dims", None)
            except Exception:
                dims = None
            # safest way: infer from diag_inv length
            n_cells = self.diag_inv.shape[0]
            self._n_cells = n_cells

        vars_per_cell = vec.shape[0] // n_cells
        if vars_per_cell not in (2, 3):
            raise ValueError(f"CPRPreconditioner: unsupported vars_per_cell={vars_per_cell} (expected 2 or 3)")

        # Давление — первые n_cells компонентов
        n = n_cells
        
        # 🔧 ИСПРАВЛЕНО: правильная обработка gradients
        if vec.requires_grad:
            rhs_p = vec[:n].detach().cpu().numpy()
        else:
            rhs_p = vec[:n].cpu().numpy()

        # 🎯 АВТОМАТИЧЕСКОЕ МАСШТАБИРОВАНИЕ для робастности
        rhs_norm = np.linalg.norm(rhs_p)
        if rhs_norm < 1e-15:
            # Нулевая правая часть - возвращаем нуль
            out = torch.zeros_like(vec, dtype=vec.dtype, device=vec.device, requires_grad=False)
            return out
        
        # Нормализуем RHS к разумному масштабу
        rhs_scale = max(1e-6, min(1e6, rhs_norm))  # Клампим между 1e-6 и 1e6
        rhs_scaled = rhs_p * (1.0 / rhs_scale)

        # Решаем давление через AMG или Jacobi
        if self.solver is None or self.failed_amg:
            # Fallback к диагональному предобуславливателю
            print(f"    CPR: Используем диагональное предобуславливание")
            delta_p_scaled = self.diag_inv * rhs_scaled
        else:
            try:
                print(f"    CPR: Используем AMG решение (RHS масштаб: {rhs_scale:.2e})")
                delta_p_geom = self.solver.solve(rhs_scaled, tol=1e-8, max_iter=200)
                # --- ВОССТАНОВЛЕНИЕ РЕШЕНИЯ ---
                # Теперь матрица собирается без искусственного множителя, поэтому
                # решение AMG непосредственно связано с физическим через
                #   δ_geom = δ_true / rhs_scale  ⇒  δ_true = δ_geom · rhs_scale.
                # Пока откладываем умножение на rhs_scale до финального шага.
                delta_p_scaled = delta_p_geom  # matrix_scale = 1.0
                
                # Проверяем результат на NaN/Inf
                if np.any(np.isnan(delta_p_scaled)) or np.any(np.isinf(delta_p_scaled)):
                    print("    CPR: AMG вернул NaN/Inf, переключаемся на Jacobi")
                    self.failed_amg = True
                    delta_p_scaled = self.diag_inv * rhs_scaled
                else:
                    delta_p_norm = np.linalg.norm(delta_p_scaled)
                    print(f"    CPR: AMG решение успешно, ||delta_p||={delta_p_norm:.3e}")

                    # 🎯 ROBUST проверка: сравниваем относительную величину решения
                    rel_ratio = delta_p_norm / (rhs_norm + 1e-30)
                    # Если решение слишком велико ( >1e8 раз RHS), считаем AMG нестабильным
                    if rel_ratio > 1e8:
                        print(f"    CPR: AMG решение УТРАТИЛО надёжность (||δp||/||rhs||={rel_ratio:.2e});")
                        if self.backend == "geo" and getattr(self.solver, "smoother", "") != "jacobi":
                            print("    CPR: Переключаем GeoSolver на Jacobi-сглаживатель и пробуем ещё раз...")
                            try:
                                self.solver = self.solver.__class__(self.reservoir, smoother="jacobi")
                                print("✅ CPR: GeoSolver переинициализирован на Jacobi-сглаживатель")
                                delta_p_geom = self.solver.solve(rhs_scaled, tol=1e-8, max_iter=200)
                                delta_p_scaled = delta_p_geom
                                print(f"✅ CPR: GeoSolver успешно решил AMG (Jacobi)")
                                # Пересчитываем норму и ratio после повторной попытки
                                delta_p_norm = np.linalg.norm(delta_p_scaled)
                                rel_ratio = delta_p_norm / (rhs_norm + 1e-30)
                            except Exception as e:
                                print(f"❌ CPR: Ошибка повторного AMG (Jacobi): {e}")
                                rel_ratio = 1e20  # форсируем откат
                        # Если всё ещё слишком велико — окончательный откат на Jacobi
                        if rel_ratio > 1e8:
                            print("❌ CPR: Даже после смены сглаживателя решение остаётся нестабильным; окончательно отключаем AMG")
                            self.failed_amg = True
                            delta_p_scaled = self.diag_inv * rhs_scaled
                    elif rel_ratio > 1e6:
                        print(f"    CPR: AMG решение выглядит подозрительно (||δp||/||rhs||={rel_ratio:.2e}), но продолжаем использовать")
                
            except Exception as e:
                print(f"    CPR: Ошибка в AMG решателе: {e}, переключаемся на Jacobi")
                self.failed_amg = True
                delta_p_scaled = self.diag_inv * rhs_scaled

        # 🎯 ПРАВИЛЬНОЕ ВОССТАНОВЛЕНИЕ МАСШТАБА
        #   δ_geom = δ_true / rhs_scale      (для GeoSolver)
        #   δ_geom = δ_true / (rhs_scale·matrix_scale)  (для AMGX/Boomer)
        if self.backend == "geo":
            delta_p = delta_p_scaled * rhs_scale
        else:
            delta_p = delta_p_scaled * rhs_scale * self.matrix_scale
        
        # 🔧 ДОПОЛНИТЕЛЬНЫЙ DEBUG
        print(f"    CPR: Восстановление масштаба: rhs_scale={rhs_scale:.3e}, matrix_scale={self.matrix_scale:.3e} (backend={self.backend})")
        print(f"    CPR: ||delta_p|| до восстановления: {np.linalg.norm(delta_p_scaled):.3e}")
        print(f"    CPR: ||delta_p|| после восстановления: {np.linalg.norm(delta_p):.3e}")
        
        # ❌ УБРАНО: delta_p = delta_p / self.matrix_scale (двойное восстановление!)

        # 🔧 ИСПРАВЛЕНО: правильная сборка результата
        out = torch.zeros_like(vec, dtype=vec.dtype, device=vec.device, requires_grad=False)

        # Давление: результат AMG решения
        pressure_result = torch.from_numpy(delta_p).to(device=vec.device, dtype=vec.dtype)

        # 🎯 ДОПОЛНИТЕЛЬНАЯ защита от экстремальных значений
        pressure_norm = pressure_result.norm()
        vec_norm = vec[:n].norm()
        if vec_norm > 1e-15:
            # Разрешаем значительно более крупные поправки (до 1e12 раз RHS).
            # Линейный поиск позаботится о корректном демпфировании.
            max_ratio = 1e12
            if pressure_norm > max_ratio * vec_norm:
                scale_factor = (max_ratio * vec_norm) / (pressure_norm + 1e-30)
                pressure_result = pressure_result * scale_factor
                print(f"    CPR: Ограничили решение фактором {scale_factor:.3e}")

        out[:n] = pressure_result

        # Насыщенности (любой фазы): Jacobi scaling ω
        out[n:] = self.omega * vec[n:]
        
        return out 