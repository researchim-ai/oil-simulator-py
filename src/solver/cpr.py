import torch, numpy as np
import math
from .amg import BoomerSolver, AmgXSolver
from .geom_amg import GeoSolver
from typing import Optional, Dict
import os

class CPRPreconditioner:
    def __init__(self, *args,
                 backend: str = "amgx",
                 omega: float = 0.3,
                 smoother: str = "chebyshev",
                 scaler=None,
                 geo_params: Optional[dict] = None):
        """CPR предобуславливатель.

        Поддерживаются два способа вызова:
        1. CPRPreconditioner(simulator, backend="geo", ...)
           – новый интерфейс, где передаём симулятор.
        2. CPRPreconditioner(reservoir, fluid, backend="geo", ...)
           – старый интерфейс, используется для обратной совместимости
             (симулятор может быть None, тогда Stage-2 недоступен).
        """

        # --------------------------------------------------------------
        # Разбор positional args для обратной совместимости
        # --------------------------------------------------------------
        if len(args) == 1:
            # Новый интерфейс: только simulator
            simulator = args[0]
            from simulator.simulation import Simulator as _Sim
            if not isinstance(simulator, _Sim):
                raise TypeError("CPRPreconditioner: ожидается объект Simulator либо (reservoir, fluid)")
            reservoir = simulator.reservoir
            fluid = simulator.fluid
            self.simulator = simulator
        elif len(args) >= 2:
            # Старый интерфейс
            reservoir, fluid = args[0], args[1]
            simulator = None if len(args) == 2 else args[2]
            self.simulator = simulator
        else:
            raise TypeError("CPRPreconditioner: неверные позиционные аргументы")

        self.backend = backend
        # --------------------------------------------------------------
        # VariableScaler: если не передан – используем единичный
        # --------------------------------------------------------------
        if scaler is None:
            class _IdentityScaler:
                p_scale = 1.0
                inv_p_scale = 1.0
                s_scales = [1.0]
                inv_s_scales = [1.0]

                def scale_vec(self, v):
                    return v

                def unscale_vec(self, v):
                    return v

                def p_to_hat(self, p):
                    # Давление Pa → оставляем как есть
                    return p

                n_cells = 0  # будет переписано позже

            scaler = _IdentityScaler()

        self.scaler = scaler
        if hasattr(reservoir, "dimensions"):
            n_cells_tot = reservoir.dimensions[0] * reservoir.dimensions[1] * reservoir.dimensions[2]
            # Обновим n_cells для scaler, если вдруг
            try:
                setattr(self.scaler, "n_cells", n_cells_tot)
            except Exception:
                pass

        # Масштаб давления (Па → hat) для безразмеризации.
        # Нужен только для геометрического AMG v2, но сохраняем всегда.
        self.p_scale    = getattr(self.scaler, "p_scale", 1.0)
        self.inv_p_scale = getattr(self.scaler, "inv_p_scale", 1.0)

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
        # Сохраняем CSR блока давления
        self._indptr_p = indptr
        self._indices_p = ind
        self._data_p = data

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
            # Автопереключение сглаживателя для крупных сеток
            n_cells_geo = reservoir.dimensions[0] * reservoir.dimensions[1] * reservoir.dimensions[2]
            if n_cells_geo > 50000 and smoother in ("chebyshev", "jacobi", None):
                print("⚙️  CPR: GeoSolver – крупная сетка, переключаем smoother на 'l1gs'")
                smoother = "l1gs"
            try:
                print(f"🔧 CPR: Используем собственный геометрический AMG (GeoSolver, smoother='{smoother}')...")
                # Если параметры не заданы – ставим лёгкий режим (cycles=2, pre/post=2, levels=6)
                geo_params = geo_params or {}
                if "cycles_per_call" not in geo_params:
                    geo_params["cycles_per_call"] = 2  # избежать strong-режима
                if "pre_smooth" not in geo_params:
                    geo_params["pre_smooth"] = 2
                if "post_smooth" not in geo_params:
                    geo_params["post_smooth"] = 2
                # Избегаем авто-"strong" режима GeoSolver: если cycles=1 и pre=2 –
                # поменяем pre/post на 3, что незначительно увеличит работу, но
                # не вызовет усиление до cycles=3 pre=8.
                if geo_params["cycles_per_call"] == 1 and geo_params["pre_smooth"] == 2:
                    geo_params["pre_smooth"] = geo_params["post_smooth"] = 3
                if "max_levels" not in geo_params:
                    geo_params["max_levels"] = 6
                self.solver = GeoSolver(reservoir, smoother=smoother or "chebyshev", **geo_params)
                # Alias для обратной совместимости
                self.geo_solver = self.solver
                print("✅ CPR: GeoSolver инициализирован успешно")
            except Exception as e:
                print(f"❌ CPR: Ошибка GeoSolver: {e}")
                self.solver = None
                self.failed_amg = True
        elif backend == "geo2":
            from solver.geo_solver_v2 import GeoSolverV2
            # Передаём пользовательские параметры (если заданы), оставляя только
            # те, которые реально присутствуют в сигнатуре GeoSolverV2.
            geo_params = geo_params or {}
            allowed_geo2_keys = {
                "omega", "max_coarse_ratio", "device", "cycle_type",
                "cycles_per_call", "pre_smooth", "post_smooth",
                "omega_fine", "smoother_fine", "cheby_tail",
                "delta_clip_factor", "clip_kappa", "debug",
            }
            geo2_kwargs = {k: v for k, v in geo_params.items() if k in allowed_geo2_keys}
            if geo2_kwargs:
                print(f"🔧 CPR: GeoSolverV2 с пользовательскими параметрами: {geo2_kwargs}")
            self.solver = GeoSolverV2(reservoir, **geo2_kwargs)
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
        # ----- Безразмеризация: переводим коэффициенты в hat-пространство ----
        inv_p_scale = getattr(self, "inv_p_scale", 1.0)
        lam = lam_t * inv_p_scale  # скаляр в hat-единицах (1/hat·s)
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

        # ------------------------------------------------------------------
        # 🚩 ЗАЗЕМЛЯЕМ ДАВЛЕНИЕ (ANCHOR ROW)
        # Промышленные симуляторы убирают нулевой режим «P = const» фиксируя
        # одну опорную ячейку.  Здесь выбираем ячейку 0.  Её строку в CSR
        # заменяем на единичную диагональ: A[0,0] = 1, остальные элементы 0.
        # Это делает систему невырожденной и улучшает сходимость AMG.
        # ------------------------------------------------------------------
        anchor = 0  # индекс опорной ячейки
        start, end = indptr[anchor], indptr[anchor + 1]

        # Если в строке нет места (теоретически не должно быть), расширять
        # массивы не будем – вместо этого просто перезапишем первую позицию.
        # Обнуляем значения строки
        data[start:end] = 0.0

        # Гарантируем хотя бы один элемент (диагональ) – записываем в первую
        # позицию текущего диапазона.  Если diag уже там, indices[start] уже
        # равно anchor; если нет – всё равно перезаписываем.
        indices[start] = anchor
        data[start] = 1.0  # единичная диагональ (масштабируется далее вместе со всеми)

        # Для корректности row_abs_max переопределяем для anchor
        row_abs_max[anchor] = 1.0

        # --- АВТОМАТИЧЕСКОЕ МАСШТАБИРОВАНИЕ МАТРИЦЫ ---
        diag_median = np.median(diag_vals) if diag_vals else 1.0
        # Гарантируем ненулевую диагональ
        if diag_median < 1e-20:
            diag_median = 1e-20
        scale_raw = 1.0 / diag_median
        # 💡 Ограничиваем scale, иначе Geo-AMG/Chebyshev взрываются при 1e8…1e9
        # Более жёсткий потолок для matrix-scale: 1e5 вместо 1e6 —
        # это уменьшает величину Jacobi-поправки и делает fallback стабильнее.
        # Более высокий предел позволяет нормализовать матрицу на крупных моделях,
        # где диагональные элементы могут быть ~1e-12.  1e8 всё ещё безопасен для
        # float32 и не приводит к переполнению, но существенно улучшает кондиционирование.
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

        # ----- ROW SCALING (удалено) ----------------------------------------
        # Промышленные CPR-реализации после полноценной безразмеризации
        # не применяют дополнительный строковый масштаб.  Матрица уже
        # кондиционирована (scale ≤ 1e6), а Jacobi-диагональ вычисляется
        # напрямую из физической матрицы.

        self.row_scale = np.ones(N, dtype=np.float64)

        # Диагональ для Jacobi
        diag_inv = np.zeros(N, dtype=np.float64)
        for i in range(N):
            start = indptr[i]
            end   = indptr[i+1] if i < N-1 else pos
            for j in range(start, end):
                if indices[j] == i:
                    diag_inv[i] = 1.0 / max(abs(data[j]), 1e-12)
                    break

        self.diag_inv = diag_inv

        # После нормирования матрицы её масштаб равен factor 'scale';
        # сохраняем его, чтобы согласованно масштабировать RHS и восстановить решение.
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
        # --------------------------------------------------------------
        # Возможность автоматического восстановления AMG после временного
        # отключения (failed_amg=True) – актуально после фиксов GeoSolver.
        # Если backend геометрический и solver существует, пробуем снова.
        # --------------------------------------------------------------
        if self.failed_amg and self.backend in ("geo", "geo2") and self.solver is not None:
            print("    CPR: повторное включение Geo-AMG после предыдущего отключения")
            self.failed_amg = False
        # ---- Защита от нечисловых значений ---------------------------------
        if not torch.isfinite(vec).all():
            print("    CPR: RHS содержит NaN/Inf – возвращаем нулевой δ")
            return torch.zeros_like(vec, dtype=vec.dtype, device=vec.device)
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

        # ------------------------------------------------------------------
        # RHS давления
        #   • backend "geo2" уже оперирует в hat-пространстве ⇒ берём вектор
        #     напрямую, без обратного scale.
        #   • остальные backends ожидают физические Па ⇒ делаем unscale.
        # ------------------------------------------------------------------
        if self.backend == "geo2":
            rhs_hat_torch = vec[:n]
            rhs_p = rhs_hat_torch.detach().cpu().numpy()  # hat
        else:
            rhs_phys_torch = self.scaler.unscale_vec(vec)[:n]
            rhs_p = rhs_phys_torch.detach().cpu().numpy()  # Па

        # Row-scaling не применяем – матрица и правая часть уже
        # в согласованных физических единицах

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
        # Сдвигаем лимит одновременного масштабирования, чтобы не урезать rhs
        # при большом matrix_scale (крупные сетки >1e6 ячеек).
        MAX_COMBINED_SCALE = 1e9
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

        # --- Согласовываем RHS с масштабированной матрицей -------------
        # Для GeoSolverV2 ('geo2') матрица внутри решателя собирается в физическом масштабе,
        # поэтому *не* применяем дополнительный множитель matrix_scale к RHS.
        if self.backend == "geo2":
            rhs_scaled = rhs_p.copy()
        else:
            rhs_scaled = rhs_p * self.matrix_scale

        # --------------------------------------------------------------
        #  Дополнительное row-scale RHS → средний масштаб ~ O(1).
        #  Это резко снижает δp/RHS в первом вызове AMG.
        # --------------------------------------------------------------
        # --- Дополнительное row-scale: приводим RHS к среднему O(1) масштабу ---
        # ------------------------------------------------------
        # Вычисляем row_norm в hat-пространстве, чтобы избежать
        # переполнений при огромных RHS в физических Па.
        # Давление (первые N) делим на p_scale; насыщенности без изменений.
        # ------------------------------------------------------
        if self.backend != "geo2" and hasattr(self, 'scaler') and self.scaler is not None:
            n_cells_hat = self.scaler.n_cells
            rhs_hat_tmp = rhs_scaled.copy()
            rhs_hat_tmp[:n_cells_hat] *= self.scaler.inv_p_scale  # Pa → hat
            row_norm = max(np.linalg.norm(rhs_hat_tmp) / math.sqrt(len(rhs_hat_tmp)), 1e-12)
        else:
            row_norm = max(np.linalg.norm(rhs_scaled) / math.sqrt(len(rhs_scaled)), 1e-12)
        if os.environ.get("OIL_DEBUG", "0") == "1":
            print(f"[CPR-DBG] RHS before row_scale: min={rhs_scaled.min():.3e}, max={rhs_scaled.max():.3e}, row_norm={row_norm:.3e}")
        rhs_scaled /= row_norm
        local_row_scale = row_norm
        if os.environ.get("OIL_DEBUG", "0") == "1":
            print(f"[CPR-DBG] RHS after row_scale: min={rhs_scaled.min():.3e}, max={rhs_scaled.max():.3e}")

        # Решаем давление через AMG или Jacobi
        if self.solver is None or self.failed_amg:
            # Fallback: одна итерация Jacobi/L1GS
            print("    CPR: AMG недоступен – применяем одну итерацию Jacobi")
            delta_p_scaled = (self.diag_inv / max(self.matrix_scale, 1e-30)) * rhs_scaled
        else:
            try:
                print(f"    CPR: Используем AMG решение (RHS масштаб: {rhs_scale:.2e})")
                # Для геометрического AMG высокая точность (1e-8) не нужна:
                # - малые модели (≤500 клеток) – Jacobi справится
                # - средние (<5e5) – 1e-4 достаточно
                # - крупные (≥5e5) – 1e-5 удерживает надёжность без лишних циклов
                if n_cells < 500:
                    gmres_tol = 1e-6
                elif n_cells < 500_000:
                    gmres_tol = 1e-4
                else:
                    gmres_tol = 1e-5
                if self.backend == "geo":
                    gmres_tol = 1e-8 if n_cells < 20000 else 1e-6
                    if n_cells < 10000:
                        gmres_tol = 1e-5
                    delta_p_geom = self.solver.solve(rhs_scaled, tol=gmres_tol, max_iter=200)
                elif self.backend == "geo2":
                    # GeoSolverV2 принимает torch.Tensor
                    rhs_t = torch.from_numpy(rhs_scaled).to(dtype=torch.float64, device=self.solver.device)
                    delta_t = self.solver.solve(rhs_t, tol=1e-6, max_iter=10)
                    delta_p_geom = delta_t.astype(np.float64) if not isinstance(delta_t, np.ndarray) else delta_t
                else:
                    # AMG backend on GPU/CPU
                    try:
                        delta_p_geom = self.solver.solve(rhs_scaled, tol=gmres_tol, max_iter=200)
                    except Exception as e:
                        print(f"    CPR: Ошибка в AMG решателе: {e}, переключаемся на Jacobi")
                        self.failed_amg = True
                        delta_p_scaled = (self.diag_inv / max(self.matrix_scale, 1e-30)) * rhs_scaled
                        return out # Return the zero vector if AMG fails

                # Убираем константный null-space (среднее), характерный для Neumann BC
                delta_p_geom = delta_p_geom - delta_p_geom.mean()
                # --- ВОССТАНОВЛЕНИЕ РЕШЕНИЯ ---
                # Теперь матрица собирается без искусственного множителя, поэтому
                # решение AMG непосредственно связано с физическим через
                #   δ_geom = δ_true / rhs_scale  ⇒  δ_true = δ_geom · rhs_scale.
                # Пока откладываем умножение на rhs_scale до финального шага.
                delta_p_scaled = delta_p_geom
                
                # Проверяем результат на NaN/Inf
                if np.any(np.isnan(delta_p_scaled)) or np.any(np.isinf(delta_p_scaled)):
                    print("    CPR: AMG вернул NaN/Inf, переключаемся на Jacobi")
                    self.failed_amg = True
                    delta_p_scaled = (self.diag_inv / max(self.matrix_scale, 1e-30)) * rhs_scaled
                else:
                    delta_p_norm_scaled = np.linalg.norm(delta_p_scaled)
                    print(f"    CPR: AMG решение успешно, ||delta_p_scaled||={delta_p_norm_scaled:.3e}")

                    # --- ROBUST infinity-norm guard ---------------------------------------
                    ratio_inf = np.linalg.norm(delta_p_scaled, np.inf) / (rhs_norm + 1e-30)
                    # --- DEBUG: выводим guard ratio вне зависимости от порога ---
                    print(f"    CPR DEBUG: rhs_norm={rhs_norm:.3e}, delta_p_inf={np.linalg.norm(delta_p_scaled, np.inf):.3e}, ratio_inf={ratio_inf:.3e}")
                    if self.backend == "geo":
                        # Backend 'geo2' — та же геометрия, применяем те же пороги.
                        pass
                    if self.backend in ("geo", "geo2"):
                        # Для геометрического AMG допускаем ratio до 1e10.
                        # Если превышает – временно переходим на Jacobi в этом вызове,
                        # но НЕ отключаем AMG насовсем: повторные итерации Ньютона
                        # часто стабилизируются, когда невязка падает.
                        if ratio_inf > 1e10:
                            print(
                                f"    ⚠️  Geo-AMG нестабилен (ratio={ratio_inf:.2e}) – локальный fallback на Jacobi"
                            )
                            # Одноразовый Jacobi-доблинг без изменения self.failed_amg
                            delta_p_scaled = (self.diag_inv / max(self.matrix_scale, 1e-30)) * rhs_scaled
                        # для backend=='geo' дополнительно действий не требуется
                    else:
                        limit_ratio = 1e4  # для численных AMG
                        if ratio_inf > limit_ratio:
                            print(f"    ⚠️  CPR: ||δp||_inf слишком велик (ratio={ratio_inf:.2e}); fallback на Jacobi")
                            self.failed_amg = True
                            delta_p_scaled = (self.diag_inv / max(self.matrix_scale, 1e-30)) * rhs_scaled

                    if self.backend not in ("geo", "geo2"):
                        # --- ROBUST проверка для численных AMG ---
                        delta_p_phys_norm = delta_p_norm_scaled * self.matrix_scale
                        rel_ratio = delta_p_phys_norm / (rhs_norm + 1e-30)

                        # Если решение слишком велико (>1e8 раз RHS) – считаем AMG нестабильным
                        thr_rel = 1e10 if self.backend in ("geo", "geo2") else 1e8
                        if n_cells > 500 and rhs_norm > 1e-6 and rel_ratio > thr_rel:
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
                            if rel_ratio > thr_rel:
                                print("❌ CPR: Даже после смены сглаживателя решение остаётся нестабильным; окончательно отключаем AMG")
                                self.failed_amg = True
                                delta_p_scaled = (self.diag_inv / max(self.matrix_scale, 1e-30)) * rhs_scaled
                        elif n_cells <= 5000 and rel_ratio > 1e4:
                            print(f"    CPR: AMG слаб на малой модели (ratio={rel_ratio:.2e}) – переключаемся на Jacobi")
                            self.failed_amg = True
                            delta_p_scaled = (self.diag_inv / max(self.matrix_scale, 1e-30)) * rhs_scaled
                        elif n_cells > 500 and rhs_norm > 1e-6 and rel_ratio > 1e6:
                            print(f"    CPR: AMG решение выглядит подозрительно (||δp||/||rhs||={rel_ratio:.2e}), но продолжаем использовать")
                
            except Exception as e:
                print(f"    CPR: Ошибка в AMG решателе: {e}, переключаемся на Jacobi")
                self.failed_amg = True
                delta_p_scaled = (self.diag_inv / max(self.matrix_scale, 1e-30)) * rhs_scaled

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
        # переводим решение обратно, деля на тот же matrix_scale
        # После согласованного масштабирования RHS на matrix_scale дополнительное деление
        # приводит к искусственному занижению поправки давления. Используем решение напрямую.
        # Восстанавливаем физический масштаб: A_scaled = matrix_scale * A_phys,
        # поэтому δp_phys = δp_scaled / matrix_scale
        # Учитываем обратное row-scale
        # Для backend='geo2' обратно делить на matrix_scale не нужно –
        # решение уже в физическом масштабе.
        # Защита от переполнения: если local_row_scale аномально велик → обрезаем
        safe_row_scale = np.clip(local_row_scale, 0.0, 1e6)
        if self.backend == "geo2":
            # Решатель вернул Δp в hat-единицах.  Приводим к физическим Паскалям.
            delta_p_hat = delta_p_scaled * safe_row_scale           # hat
            delta_p_phys = delta_p_hat * self.p_scale               # Па
        else:
            delta_p_phys = (delta_p_scaled * safe_row_scale) / max(self.matrix_scale, 1e-30)  # Па

        # Заменяем нечисла на 0 (иначе попадают Inf и ломают GMRES)
        delta_p_phys = np.nan_to_num(delta_p_phys, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"    CPR: ||delta_p_phys||={np.linalg.norm(delta_p_phys):.3e}")

        # --- Адаптивное ограничение Δp (работаем уже с physical) ---------
        # 1) убираем экстремальные выбросы отдельных ячеек
        # --- Локальный кламп отдельных ячеек ---
        MAX_DP_HAT_LOCAL = 1e5  # 100 кМПа – более мягко
        np.clip(delta_p_phys, -MAX_DP_HAT_LOCAL, MAX_DP_HAT_LOCAL, out=delta_p_phys)

        # 2) ограничиваем энергию решения: ‖δp‖ ≤ 10 × ‖rhs‖
        # Сопоставляем единицы: delta_p_hat находится в том же масштабе, что и rhs_scaled
        # ⚠️  Удалён «энергетический» клип (limit_hi/lo).  Ограничиваем только локальные выбросы
        #     через MAX_DP_HAT и доверяем trust-region/line-search дальнейшую стабилизацию.

        # --------------------------------------------------------------
        # Безопасный кламп давления в hat-пространстве
        # --------------------------------------------------------------
        # MAX_DP_HAT = 1e3  # 1000 hat = 1 ГПа – баланс между стабильностью и эффективностью
        # delta_p_hat = np.clip(delta_p_hat, -MAX_DP_HAT, MAX_DP_HAT)
        # print(f"    CPR: ||delta_p_hat(clamped)||={np.linalg.norm(delta_p_hat):.3e}")

        # --- восстановление физических Па через Normalizer ------------
        # Вектор физических поправок (Па, безразмерные насыщенности)
        delta_phys_full = torch.zeros_like(vec, dtype=vec.dtype, device=vec.device)
        # Давление – физические Па
        delta_phys_full[:n] = torch.from_numpy(delta_p_phys).to(device=vec.device, dtype=vec.dtype)

        # Для возможной ψ-Relax работаем в hat-пространстве ----------------
        delta_hat_full = self.scaler.scale_vec(delta_phys_full)

        # ------------------------------------------------------------------
        # Stage-2: Saturation block (Jacobi)                                 
        # ------------------------------------------------------------------
        try:
            props = getattr(self.simulator, "_cell_props_cache", None)
            if props is not None:
                phi  = props["phi"]          # (N,)
                dt   = props["dt"]           # scalar tensor
                V    = props["V"]
                lam_w = props["lam_w"]
                lam_o = props["lam_o"]
                c_w   = props["c_w"]
                c_o   = props["c_o"]
                lam_g = props["lam_g"]
                c_g   = props["c_g"]

                # Диагональ Jacobi по насыщенности: φ ρ_w V / dt  (масса воды)
                rho_w = props["rho_w"]  # (N,)
                diag_SS = (phi * V * rho_w) / (dt + 1e-30)  # (N,)

                # rhs_s_phys нужен для лога, поэтому вычисляем его сразу
                rhs_s_phys = self.scaler.unscale_vec(vec)[n:]

                # Если данные нечисловые – пропускаем Stage-2
                if (not torch.isfinite(rhs_s_phys).all()) or (not torch.isfinite(diag_SS).all()):
                    raise ValueError("non-finite rhs_s or diag_SS")

                # --- DEBUG LOG -------------------------------------------------
                if not hasattr(self, "_dbg_diag_logged") or self._dbg_diag_logged < 10:
                    print(
                        f"    DEBUG Stage-2: rhs_s_norm={rhs_s_phys.norm():.3e}, "
                        f"diag_SS min={diag_SS.min():.3e}, mean={diag_SS.mean():.3e}, max={diag_SS.max():.3e}"
                    )
                    self._dbg_diag_logged = getattr(self, "_dbg_diag_logged", 0) + 1

                # Вклад давления в уравнение насыщенностей
                dFs_dp  = (lam_w * c_w + lam_o * c_o) * V / (dt + 1e-30)
                if lam_g is not None and c_g is not None:
                    dFs_dp = dFs_dp + lam_g * c_g * V / (dt + 1e-30)

                # delta_p (torch) уже физический; сразу ограничиваем, чтобы
                # Stage-2 не «взорвалось» из-за гигантского Δp.
                if self.scaler is not None:
                    P_CLIP_HAT = 20.0e6 / self.scaler.p_scale
                    delta_phys_full[:n] = delta_phys_full[:n].clamp(-P_CLIP_HAT, P_CLIP_HAT)
                else:
                    P_CLIP = 20.0e6
                    delta_phys_full[:n] = delta_phys_full[:n].clamp(-P_CLIP, P_CLIP)
 
                vars_per_cell = rhs_s_phys.numel() // n  # 1 (Sw) или 2 (Sw,Sg)
                delta_s_list = []
                for sat_idx in range(vars_per_cell):
                    start = sat_idx * n
                    end   = start + n
                    rhs_sat = rhs_s_phys[start:end]
                    delta_sat = (rhs_sat - dFs_dp * delta_p_phys) / (diag_SS + 1e-30)
                    # Кламп ±0.05 – обычный предел в промышленных решателях
                    delta_sat = torch.clamp(delta_sat, -0.05, 0.05)
                    delta_s_list.append(delta_sat)

                delta_s_full = torch.cat(delta_s_list, dim=0)

                # Проверяем на числовые аномалии; если есть Inf / NaN – обнуляем δS
                if not torch.isfinite(delta_s_full).all():
                    print("    CPR: non-finite δS detected – zeroing Stage-2 correction")
                    delta_s_full.zero_()

                # --- Диагностический вывод ---
                if not hasattr(self, "_dbg_stage2_logged") or self._dbg_stage2_logged < 20:
                    print(
                        f"    CPR: Stage-2 δS norm={delta_s_full.norm():.3e}, "
                        f"min={delta_s_full.min():.3e}, max={delta_s_full.max():.3e}, "
                        f"rhs_s_norm={rhs_s_phys.norm():.3e}, dFs_dp_norm={(dFs_dp*delta_p_phys).norm():.3e}"
                    )
                    self._dbg_stage2_logged = getattr(self, "_dbg_stage2_logged", 0) + 1

                delta_phys_full[n:n+rhs_s_phys.numel()] = delta_s_full
                # Обновляем hat-вектор
                delta_hat_full = self.scaler.scale_vec(delta_phys_full)
        except Exception as _e:
            # fallback: оставляем нули, но выводим предупреждение один раз
            if not hasattr(self, "_warn_stage2"):
                print(f"[CPR] Stage-2 saturation update failed: {_e}")
                self._warn_stage2 = True

        # --------------------------------------------------------------
        # ψ-Relax Chebyshev tail на полной системе
        # --------------------------------------------------------------
        try:
            from solver.csr_full import assemble_full_csr
            from solver.chebyshev import chebyshev_smooth
        except ImportError:
            assemble_full_csr = None

        if assemble_full_csr is not None:
            if not hasattr(self, "_full_A"):
                # Определяем число переменных на ячейку (2 – P+Sw, 3 – P+Sw+Sg)
                n_total = vec.shape[0]
                vars_per_cell_local = max(2, min(3, n_total // n))
                indptr_f, indices_f, data_f = assemble_full_csr(
                    self._indptr_p, self._indices_p, self._data_p,
                    vars_per_cell=vars_per_cell_local, diag_sat=1.0)
                indptr_t = torch.from_numpy(indptr_f)
                indices_t = torch.from_numpy(indices_f)
                data_t = torch.from_numpy(data_f).to(torch.float32)
                self._full_A = torch.sparse_csr_tensor(indptr_t, indices_t, data_t,
                                                      size=(vec.shape[0], vec.shape[0]))
            A_full = self._full_A

            n_blocks = 2 if self._n_cells > 1_000_000 else 1
            for _ in range(n_blocks):
                vec_cpu = vec.cpu()
                delta_hat_cpu = delta_hat_full.cpu()
                r_hat_cpu = vec_cpu - torch.sparse.mm(A_full, delta_hat_cpu.unsqueeze(1)).squeeze(1)
                delta_inc_cpu, _ = chebyshev_smooth(A_full, r_hat_cpu,
                                                     torch.zeros_like(r_hat_cpu), iters=2, omega=0.7)
                delta_inc = delta_inc_cpu.to(vec.device)
                delta_hat_full = delta_hat_full + delta_inc
                # синхронизируем физический вектор
                delta_phys_full = self.scaler.unscale_vec(delta_hat_full)

        # delta_phys_full уже в физических единицах
        pressure_result = delta_phys_full[:n]

        # --------------------------------------------------------------
        # ДИНАМИЧЕСКИЙ кламп давления: ограничиваем Δp относительно RHS.
        #   |Δp| ≤ c · ‖rhs_p‖₂ / √N
        # где c ≈ 20.  Для маленьких RHS оставляем базовый предел 50 МПа.
        # --------------------------------------------------------------
        # import math  # удалено как дублирующее
        if torch.isfinite(pressure_result).all():
            # Норма RHS (в Pa) = hat * p_scale
            rhs_norm_hat = vec[:n].norm().item()
            rhs_norm_phys = rhs_norm_hat * float(self.p_scale)
            n_cells_float = float(n)
            dynamic_lim = 10.0 * rhs_norm_phys / (math.sqrt(n_cells_float) + 1e-30)

            # Минимальный и максимальный пределы
            # Пределы клампа в Паскалях (динамические). Для dt≈0.02 сут 20 МПа —
            # физически разумный максимум; больший шаг часто ломает line-search.
            MIN_LIM = 1e7   # 10 МПа
            MAX_LIM = 2e7   # 20 МПа
            clamp_val = max(MIN_LIM, min(dynamic_lim, MAX_LIM))

            pressure_clamped = torch.clamp(pressure_result, -clamp_val, clamp_val)
            if not torch.allclose(pressure_clamped, pressure_result):
                print(
                    f"    CPR: Δp_phys клампирован динамически до ±{clamp_val/1e6:.1f} МПа"
                )
            pressure_result = pressure_clamped

        # В будущем, если появятся бекенды, где матрица не масштабируется,
        # достаточно выставлять self.matrix_scale = 1.0 во время сборки.
        
        # 🔧 ДОПОЛНИТЕЛЬНЫЙ DEBUG
        print(f"    CPR: ||delta_p_phys||={pressure_result.norm():.3e}")
        
        # ❌ УБРАНО: delta_p = delta_p / self.matrix_scale (двойное восстановление!)

        # --- DEBUG: выводим нормы результата CPR (первые 5 вызовов) ------
        if not hasattr(self, "_dbg_out_logged") or self._dbg_out_logged < 5:
            delta_s_phys = delta_phys_full[n:]
            print(
                f"[CPR out] δp_norm={pressure_result.norm():.3e}, "
                f"δS_norm={delta_s_phys.norm():.3e}"
            )
            self._dbg_out_logged = getattr(self, "_dbg_out_logged", 0) + 1

        # 🔧 ИСПРАВЛЕНО: правильная сборка результата
        # Собираем выходной вектор и конвертируем в hat-пространство
        out_phys = torch.zeros_like(vec, dtype=vec.dtype, device=vec.device, requires_grad=False)

        # Записываем компоненту давления (в Па) в выходной вектор
        out_phys[:n] = pressure_result

        # --------------------------------------------------------------
        # Saturation block — используем результат Stage-2, уже сохранённый
        # в `delta_phys_full`.  Никаких дополнительных Jacobi-заглушек не
        # требуется: рассчитанные ΔS коррелируют с давлением и RHS.
        # --------------------------------------------------------------

        delta_s_phys = delta_phys_full[n:]

        # --------------------------------------------------------------
        # Финальная защита: если даже после всех клампов поправка давления
        # остаётся огромной (||δp|| > 1e9 × ||rhs||) – обнуляем, чтобы не
        # испортить line-search.  JFNK при необходимости скорректирует шаг.
        # --------------------------------------------------------------

        # после корректной нормализации необходимость дополнительных клипов резко падает;
        # однако оставляем проверку NaN/Inf на всякий случай
        final_norm = pressure_result.norm().item()
        rhs_norm_torch = vec[:n].norm().item() + 1e-30
        if self.backend not in ("geo", "geo2") and n_cells > 500 and rhs_norm_torch > 1e-6 and final_norm > 1e9 * rhs_norm_torch:
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

        out_phys[:n] = pressure_result

        # Насыщенности (все фазы): простое Jacobi damping ω
        out_phys[n:] = delta_s_phys

        # Возвращаем физический вектор (давление – Па, насыщенности – безразмерные)
        return out_phys 