import torch, numpy as np
from .amg import BoomerSolver, AmgXSolver
from .geom_amg import GeoSolver
from typing import Optional

class CPRPreconditioner:
    def __init__(self, reservoir, fluid, backend="amgx", omega=0.3, smoother: str = "chebyshev", scaler=None):
        self.backend = backend
        # VariableScaler для согласованного column-scale (давление)
        self.scaler = scaler
        if scaler is not None:
            self.p_scale = scaler.p_scale
            self.inv_p_scale = scaler.inv_p_scale
            # Массив масштабов для насыщенностей (Sw, опц. Sg)
            self.s_scales = getattr(scaler, 's_scales', [1.0])
            self.inv_s_scales = getattr(scaler, 'inv_s_scales', [1.0])
        else:
            self.p_scale = 1.0
            self.inv_p_scale = 1.0
            self.s_scales = [1.0]
            self.inv_s_scales = [1.0]
        self.omega = omega
        self.failed_amg = False  # Флаг провала AMG
        
        print(f"🔧 CPR: Инициализация с backend='{backend}'")

        # Сохраняем ссылку на reservoir для последующей возможной
        # переинициализации AMG (например, смена сглаживателя).
        self.reservoir = reservoir

        indptr, ind, data = self._assemble_pressure_csr(reservoir, fluid)
        print(f"🔧 CPR: Построена pressure матрица размера {len(indptr)-1}x{len(indptr)-1}, nnz={len(data)}")

        # --------------------------------------------------------------
        # Защита от чрезмерного масштабирования матрицы
        # --------------------------------------------------------------
        # Для мелких моделей (2-D, тонкие пласты) диагональ может быть
        # ~1e-9, что приводит к scale~1e+9 и, как следствие, к гигантским
        # поправкам δp после восстановления.  Ограничиваем коэффициент
        # сверху разумным значением (1e4) для backends, использующих
        # численные AMG (Boomer/Hypre) – там и без дополнительного 
        # масштабирования условность приемлема.

        if hasattr(self, "matrix_scale") and self.matrix_scale > 1e8 and backend in ("hypre", "boomer", "cpu", "amgx"):
            # Для AMG backends на CPU/GPU слишком большой scale ухудшает устойчивость;
            # однако объёмная форма требует scale до 1e8. Ограничиваем более мягко.
            LIMIT = 1e8
            if self.matrix_scale > LIMIT:
                print(f"⚠️  CPR: matrix_scale={self.matrix_scale:.3e} > {LIMIT:.1e}; клампим")
                self.matrix_scale = LIMIT
        
        # Сохраняем диагональ для Jacobi fallback
        self.diag_inv = self._extract_diagonal_inverse(indptr, ind, data)
        print(f"�� CPR: Диагональ для fallback готова")
        
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
                self.solver = GeoSolver(reservoir, smoother=smoother or "chebyshev")
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

        # Для row-scaling понадобится сохранить макс.|row| после сборки
        row_abs_max = np.zeros(N, dtype=np.float64)
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
                        t = Tx[i-1, j, k] * lam
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
                        adaptive_shift = base_shift * max(1.0, self.compressibility_factor ** 0.5)
                    else:
                        adaptive_shift = base_shift
                    
                    # Диагональный элемент
                    indices[pos] = center
                    diag_entry = diag + adaptive_shift  # already in scaled units
                    data[pos] = diag_entry
                    pos += 1
                    diag_vals.append(abs(diag_entry))

                    # ---- Row max abs value (для последующей нормализации) ----
                    row_start = indptr[idx]
                    row_end   = pos
                    row_abs_max[idx] = np.max(np.abs(data[row_start:row_end]))
                    idx += 1

        indptr[N] = pos

        # --- АВТОМАТИЧЕСКОЕ МАСШТАБИРОВАНИЕ МАТРИЦЫ ---
        diag_median = np.median(diag_vals) if diag_vals else 1.0
        # Гарантируем ненулевую диагональ
        if diag_median < 1e-20:
            diag_median = 1e-20
        scale_raw = 1.0 / diag_median
        # 💡 Ограничиваем scale, иначе Geo-AMG/Chebyshev взрываются при 1e8…1e9
        MAX_SCALE = 1e8
        N_cells = nx * ny * nz

        # 🔧 НОВОЕ: для микросеток (<100 ячеек) полностью отключаем scale,
        # чтобы избежать гигантских δp после восстановления.
        if N_cells <= 100:
            print("⚙️  CPR: микромодель (≤100 ячеек) — отключаем matrix_scale")
            scale = 1.0
        else:
            if scale_raw > MAX_SCALE:
                print(
                    f"⚠️  CPR: scale={scale_raw:.3e} слишком велик (N={N_cells}), обрезаем до {MAX_SCALE:.1e}"
                )
                scale = MAX_SCALE
            else:
                scale = scale_raw

        data[:pos] *= scale  # нормализуем матрицу с учётом клипа

        # ----- ROW SCALING ---------------------------------------------------
        # d_i = 1 / max|row_i|  ⇒  D A x = D b
        # Row-scaling должен опираться на уже отмасштабированную матрицу,
        # иначе мы фактически дублируем коэффициент «scale» и получаем
        # гигантские величины δp.  Используем m_scaled = row_abs_max * scale.

        row_scale = np.ones(N, dtype=np.float64)
        for i in range(N):
            m_scaled = row_abs_max[i] * scale
            if m_scaled > 1e-30:
                row_scale[i] = 1.0 / m_scaled
            else:
                row_scale[i] = 1.0

        # Применяем масштаб к данным матрицы
        for i in range(N):
            s = row_scale[i]
            start = indptr[i]
            end   = indptr[i+1] if i < N-1 else pos
            data[start:end] *= s

        # Диагональ для Jacobi должна отражать ту же нормализацию
        diag_inv_scaled = np.zeros(N, dtype=np.float64)
        for i in range(N):
            start = indptr[i]
            end   = indptr[i+1] if i < N-1 else pos
            # после масштабирования diag находится внутри этого среза
            for j in range(start, end):
                if indices[j] == i:
                    val = data[j]
                    diag_inv_scaled[i] = 1.0 / max(abs(val), 1e-12)
                    break

        self.row_scale = row_scale          # numpy 1-D (length N)
        self.diag_inv = diag_inv_scaled     # заменяем предыдущую

        # Матрица теперь отмасштабирована → дополнительный matrix_scale не нужен
        self.matrix_scale = 1.0

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

        # Используем Normalizer для безразмерного RHS
        rhs_hat_torch = self.scaler.scale_vec(vec)[:n]
        rhs_p = rhs_hat_torch.detach().cpu().numpy()

        # применяем row-scale, рассчитанный для той же матрицы
        rhs_p = rhs_p * self.row_scale

        # 🎯 АВТОМАТИЧЕСКОЕ МАСШТАБИРОВАНИЕ для робастности
        rhs_norm = np.linalg.norm(rhs_p)
        if rhs_norm < 1e-15:
            # Нулевая правая часть - возвращаем нуль
            out = torch.zeros_like(vec, dtype=vec.dtype, device=vec.device, requires_grad=False)
            return out
        
        # Новая стратегия: после row-scaling система уже хорошо нормирована,
        # дополнительный rhs_scale только искажает итоговую δp.
        rhs_scale = 1.0

        # --------------------------------------------------------------
        # 🔒 ДОП. ЗАЩИТА: ограничиваем одновременный масштаб
        #          matrix_scale · rhs_scale ≤ 1e6
        # --------------------------------------------------------------
        MAX_COMBINED_SCALE = 1e6
        prod_scale = self.matrix_scale * rhs_scale
        if prod_scale > MAX_COMBINED_SCALE:
            # Уменьшаем rhs_scale, сохраняя нижний порог 1e-6
            rhs_scale_new = MAX_COMBINED_SCALE / max(self.matrix_scale, 1e-30)
            rhs_scale_new = max(rhs_scale_new, 1e-6)
            print(
                f"    CPR: Ограничиваем масштаб: matrix_scale·rhs_scale={prod_scale:.2e} » {MAX_COMBINED_SCALE:.1e}; "
                f"rhs_scale: {rhs_scale:.2e} → {rhs_scale_new:.2e}"
            )
            rhs_scale = rhs_scale_new

        # после row-scaling зачастую rhs_norm уже O(1);
        # оставляем прежний rhs_scale но используем более мягкий лимит
        rhs_scaled = rhs_p  # rhs_scale == 1

        # Решаем давление через AMG или Jacobi
        if self.solver is None or self.failed_amg:
            # Fallback к диагональному предобуславливателю
            print(f"    CPR: Используем диагональное предобуславливание")
            delta_p_scaled = self.diag_inv * rhs_scaled
        else:
            try:
                print(f"    CPR: Используем AMG решение (RHS масштаб: {rhs_scale:.2e})")
                gmres_tol = 1e-6 if n_cells <= 500 else 1e-8
                delta_p_geom = self.solver.solve(rhs_scaled, tol=gmres_tol, max_iter=200)
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
                    delta_p_norm_scaled = np.linalg.norm(delta_p_scaled)
                    print(f"    CPR: AMG решение успешно, ||delta_p_scaled||={delta_p_norm_scaled:.3e}")

                    if self.backend != "geo":
                        # --- ROBUST проверка для численных AMG ---
                        delta_p_phys_norm = delta_p_norm_scaled * self.matrix_scale
                        rel_ratio = delta_p_phys_norm / (rhs_norm + 1e-30)

                        # Если решение слишком велико (>1e8 раз RHS) – считаем AMG нестабильным
                        if n_cells > 500 and rhs_norm > 1e-6 and rel_ratio > 1e8:
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
                        elif n_cells > 500 and rhs_norm > 1e-6 and rel_ratio > 1e6:
                            print(f"    CPR: AMG решение выглядит подозрительно (||δp||/||rhs||={rel_ratio:.2e}), но продолжаем использовать")
                
            except Exception as e:
                print(f"    CPR: Ошибка в AMG решателе: {e}, переключаемся на Jacobi")
                self.failed_amg = True
                delta_p_scaled = self.diag_inv * rhs_scaled

        # --------------------------------------------------------------
        # ПРАВИЛЬНОЕ восстановление физических поправок давления
        # --------------------------------------------------------------
        # --- ВОССТАНОВЛЕНИЕ ФИЗИЧЕСКОЙ Δp ---------------------------------
        # Общий вывод (см. детальное описание чуть выше):
        #   δ_true = δ_geom · rhs_scale · matrix_scale
        # Однако в backend="geo" сама GeoSolver собирает A_orig **уже
        # умноженной** на matrix_scale, а свой внутренний solve() возвращает
        # δ_geom, фактически равный δ_true / matrix_scale.  Поэтому здесь
        # нужно вернуть исходный масштаб ЧЕРЕЗ деление, а не умножение.

        # --------------------------------------------------------------
        # Обратный column-scale: возвращаем давление в физических Па
        # --------------------------------------------------------------
        delta_p_hat = delta_p_scaled  # matrix_scale =1, уже в hat-единицах
        print(f"    CPR: ||delta_p_hat||={np.linalg.norm(delta_p_hat):.3e}")

        # --------------------------------------------------------------
        # Безопасный кламп давления в hat-пространстве
        # --------------------------------------------------------------
        MAX_DP_HAT = 1e4  # 10 кМПа при p_scale=1 МПа
        delta_p_hat = np.clip(delta_p_hat, -MAX_DP_HAT, MAX_DP_HAT)
        print(f"    CPR: ||delta_p_hat(clamped)||={np.linalg.norm(delta_p_hat):.3e}")

        # --- восстановление физических Па через Normalizer ------------
        delta_hat_full = torch.zeros_like(vec, dtype=vec.dtype, device=vec.device)
        delta_hat_full[:n] = torch.from_numpy(delta_p_hat).to(device=vec.device, dtype=vec.dtype)
        delta_phys_full = self.scaler.unscale_vec(delta_hat_full)
        pressure_result = delta_phys_full[:n]

        # В будущем, если появятся бекенды, где матрица не масштабируется,
        # достаточно выставлять self.matrix_scale = 1.0 во время сборки.
        
        # 🔧 ДОПОЛНИТЕЛЬНЫЙ DEBUG
        print(f"    CPR: ||delta_p_phys||={pressure_result.norm():.3e}")
        
        # ❌ УБРАНО: delta_p = delta_p / self.matrix_scale (двойное восстановление!)

        # 🔧 ИСПРАВЛЕНО: правильная сборка результата
        out = torch.zeros_like(vec, dtype=vec.dtype, device=vec.device, requires_grad=False)

        # Давление уже записано в pressure_result (Па)

        # --------------------------------------------------------------
        # Saturation block — нормализация и Jacobi
        # --------------------------------------------------------------
        rhs_hat_full = self.scaler.scale_vec(vec)
        sat_hat = rhs_hat_full[n:]
        sat_norm = sat_hat.norm().item()

        # Адаптивное ω: меньше при очень больших невязках
        base_omega = self.omega
        omega_eff = min(base_omega, 0.1) if sat_norm > 100.0 else base_omega

        # Нормализуем, применяем Jacobi, масштаб НЕ возвращаем
        scale_s = sat_norm if sat_norm > 1.0 else 1.0
        delta_s_hat = omega_eff * (sat_hat / scale_s)
        delta_s_phys = delta_s_hat  # насыщенности безразмерны

        # --------------------------------------------------------------
        # Финальная защита: если даже после всех клампов поправка давления
        # остаётся огромной (||δp|| > 1e9 × ||rhs||) – обнуляем, чтобы не
        # испортить line-search.  JFNK при необходимости скорректирует шаг.
        # --------------------------------------------------------------

        # после корректной нормализации необходимость дополнительных клипов резко падает;
        # однако оставляем проверку NaN/Inf на всякий случай
        final_norm = pressure_result.norm().item()
        rhs_norm_torch = vec[:n].norm().item() + 1e-30
        if self.backend != "geo" and n_cells > 500 and rhs_norm_torch > 1e-6 and final_norm > 1e9 * rhs_norm_torch:
            print(f"    CPR: Δp всё ещё экстремально велико (||δp||/||rhs||={final_norm/rhs_norm_torch:.2e}); обнуляем результат")
            pressure_result.zero_()

        # 🎯 ДОПОЛНИТЕЛЬНАЯ защита от экстремальных значений
        pressure_norm = pressure_result.norm()
        vec_norm = vec[:n].norm()
        if vec_norm > 1e-15:
            # Разрешаем значительно более крупные поправки (до 1e12 раз RHS).
            # Линейный поиск позаботится о корректном демпфировании.
            max_ratio = 1e12
            if self.backend != "geo" and vec_norm > 1e-6 and pressure_norm > max_ratio * vec_norm:
                scale_factor = (max_ratio * vec_norm) / (pressure_norm + 1e-30)
                pressure_result = pressure_result * scale_factor
                print(f"    CPR: Ограничили решение фактором {scale_factor:.3e}")

        out[:n] = pressure_result

        # Насыщенности (все фазы): простое Jacobi damping ω
        out[n:] = delta_s_phys

        return out 