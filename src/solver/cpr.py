import torch, numpy as np
import math
from .amg import BoomerSolver, AmgXSolver
from .geom_amg import GeoSolver
from typing import Optional, Dict
import os

def _to_torch(x, ref_t: torch.Tensor):
    if isinstance(x, torch.Tensor):
        return x.to(ref_t.device, ref_t.dtype)
    return torch.as_tensor(x, device=ref_t.device, dtype=ref_t.dtype)

def _l2_inf(x):
    if isinstance(x, torch.Tensor):
        return x.norm().item(), x.abs().max().item()
    v = np.asarray(x)
    return float(np.linalg.norm(v)), float(np.max(np.abs(v)))


def _chk_tensor(tag, t):
    if isinstance(t, torch.Tensor):
        n2 = t.norm().item() if torch.isfinite(t).all() else float('nan')
        ni = t.abs().max().item() if torch.isfinite(t).all() else float('nan')
        print(f"[LOG {tag}] ‖·‖₂={n2:.3e}  ‖·‖∞={ni:.3e}  finite={torch.isfinite(t).all().item()}")
        if not torch.isfinite(t).all():
            raise ValueError(f"NaN/Inf in {tag}")
    elif isinstance(t, np.ndarray):
        n2 = np.linalg.norm(t) if np.isfinite(t).all() else float('nan')
        ni = np.max(np.abs(t)) if np.isfinite(t).all() else float('nan')
        print(f"[LOG {tag}] ‖·‖₂={n2:.3e}  ‖·‖∞={ni:.3e}  finite={np.isfinite(t).all()}")
        if not np.isfinite(t).all():
            raise ValueError(f"NaN/Inf in {tag}")


class CPRPreconditioner:
    def __init__(self, *args,
                 backend: str = "amgx",
                 omega: float = 0.3,
                 smoother: str = "chebyshev",
                 scaler=None,
                 geo_params: Optional[dict] = None,
                 # 🔽 новые параметры, читаемые из конфига/CLI
                 geo_tol: float = 1e-6,
                 geo_max_iter: int = 10,
                 gmres_tol: float = 1e-3,
                 gmres_max_iter: int = 60):
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
        self.geo_tol = geo_tol
        self.geo_max_iter = geo_max_iter
        self.gmres_tol = gmres_tol
        self.gmres_max_iter = gmres_max_iter        
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
            geo_params = geo_params or {}
            # добавим наши tol/iter в geo_params, если пользователь не переопределил
            geo_params.setdefault("default_tol", self.geo_tol)
            geo_params.setdefault("default_max_iter", self.geo_max_iter)

            allowed_geo2_keys = {
                "omega", "max_coarse_ratio", "device", "cycle_type",
                "cycles_per_call", "pre_smooth", "post_smooth",
                "omega_fine", "smoother_fine", "cheby_tail",
                "delta_clip_factor", "clip_kappa", "debug",
                "default_tol", "default_max_iter"
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
        N_cells = nx * ny * nz

        # 🔧 НОВОЕ: для микросеток (<100 ячеек) полностью отключаем scale,
        # чтобы избежать гигантских δp после восстановления.
        if self.backend == "geo2":
            scale = 1.0
        else:
            MAX_SCALE = 1e8
            if N_cells <= 100:
                scale = 1.0
            else:
                scale = min(scale_raw, MAX_SCALE)

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
        """
        CPR preconditioner application.
        ВОЗВРАЩАЕТ Δ в *global-hat* единицах (через self.scaler).

        Логика:
        - backend == "geo2": всё делаем в физических единицах и зовём GeoSolverV2.apply_prec_phys().
            GeoSolverV2 сам сделает переходы phys <-> geo2-hat, вернёт Δp в phys.
            Stage-2 по насыщенностям так же делаем в phys, затем один раз scale_vec -> hat.
        - другие backends: оставлена прежняя логика (AMGX/Boomer/Jacobi и т.п.),
            но в самом конце мы тоже переводим результат в global-hat.
        """
        import math
        import numpy as np
        import torch

        # ---- Общие проверки ----
        if not torch.isfinite(vec).all():
            _chk_tensor("A0 vec_in_hat", vec)
            print("    CPR: RHS содержит NaN/Inf – возвращаем нулевой δ")
            # Возвращаем нулевой вектор в тех же единицах (hat)
            return torch.zeros_like(vec, dtype=vec.dtype, device=vec.device)

        # Кол-во ячеек и переменных на ячейку
        if not hasattr(self, "_n_cells"):
            self._n_cells = self.diag_inv.shape[0]
        n = self._n_cells
        vars_per_cell = vec.shape[0] // n
        if vars_per_cell not in (2, 3):
            raise ValueError(f"CPRPreconditioner: unsupported vars_per_cell={vars_per_cell} (expected 2 or 3)")

        # -------------------------------------------------------------------------
        #                          backend == "geo2"
        # -------------------------------------------------------------------------
        if self.backend == "geo2":
            # 1) RHS в физических единицах
            rhs_phys_full = self.scaler.unscale_vec(vec)
            rhs_p_phys = rhs_phys_full[:n].to(self.solver.device, torch.float64)

            # 2) Давление: решаем в phys -> phys
            delta_p_phys = self.solver.apply_prec_phys(rhs_p_phys, cycles=1)
            if delta_p_phys is None:
                # Фоллбэк — нулевой шаг по давлению
                delta_p_phys = torch.zeros_like(rhs_p_phys)

            # 3) Собираем полный phys-вектор поправки
            delta_phys_full = torch.zeros_like(rhs_phys_full)
            delta_phys_full[:n] = delta_p_phys.to(rhs_phys_full.device, dtype=rhs_phys_full.dtype)

            # 4) Stage‑2 для насыщенностей (всё в phys)
            try:
                props = getattr(self.simulator, "_cell_props_cache", None)
                if props is not None:
                    phi, dt, V   = props["phi"], props["dt"], props["V"]
                    lam_w, lam_o = props["lam_w"], props["lam_o"]
                    c_w, c_o     = props["c_w"],  props["c_o"]
                    lam_g, c_g   = props.get("lam_g"), props.get("c_g")
                    rho_w        = props["rho_w"]

                    rhs_s_phys = rhs_phys_full[n:]
                    vp = rhs_s_phys.numel() // n

                    diag_SS = (phi * V * rho_w) / (dt + 1e-30)
                    dFs_dp  = (lam_w * c_w + lam_o * c_o) * V / (dt + 1e-30)
                    if lam_g is not None and c_g is not None:
                        dFs_dp = dFs_dp + lam_g * c_g * V / (dt + 1e-30)

                    # давление уже phys
                    delta_p_phys_local = delta_p_phys.to(dtype=rhs_phys_full.dtype, device=rhs_phys_full.device)

                    deltas = []
                    for s in range(vp):
                        s0, s1 = s * n, (s + 1) * n
                        rhs_sat = rhs_s_phys[s0:s1]
                        delta_sat = (rhs_sat - dFs_dp * delta_p_phys_local) / (diag_SS + 1e-30)
                        delta_sat = torch.clamp(delta_sat, -0.05, 0.05)
                        deltas.append(delta_sat)

                    if deltas:
                        delta_s_phys = torch.cat(deltas, dim=0)
                        delta_phys_full[n:n + delta_s_phys.numel()] = delta_s_phys.to(delta_phys_full.dtype)

            except Exception as e:
                if not hasattr(self, "_warn_stage2"):
                    print(f"[CPR geo2] Stage-2 saturation update failed: {e}")
                    self._warn_stage2 = True

            # 5) Возвращаемся в global-hat ровно один раз
            delta_hat_full = self.scaler.scale_vec(delta_phys_full).to(vec.device, vec.dtype)
            return delta_hat_full

        # -------------------------------------------------------------------------
        #             ДАЛЬШЕ — СТАРЫЕ БЭКЕНДЫ (geo/amgx/boomer/jacobi/…)
        # -------------------------------------------------------------------------

        # RHS в физических единицах (давление блок)
        rhs_phys_torch = self.scaler.unscale_vec(vec)[:n]
        _chk_tensor("A1 rhs_phys", rhs_phys_torch)

        rhs_p = rhs_phys_torch.detach().cpu().numpy()
        rhs_norm = float(np.linalg.norm(rhs_p))
        if rhs_norm < 1e-15:
            # Ничего делать не надо, вернём 0 в hat
            return torch.zeros_like(vec)

        # Подготовка масштабов (как у вас было)
        rhs_scale = 1.0
        MAX_COMBINED_SCALE = 1e9
        prod_scale = self.matrix_scale * rhs_scale
        if prod_scale > MAX_COMBINED_SCALE:
            rhs_scale = max(MAX_COMBINED_SCALE / max(self.matrix_scale, 1e-30), 1e-6)

        rhs_scaled = rhs_p * self.matrix_scale

        if hasattr(self, 'scaler') and self.scaler is not None:
            rhs_hat_tmp = rhs_scaled.copy()
            rhs_hat_tmp[:getattr(self.scaler, "n_cells", n)] *= getattr(self.scaler, "inv_p_scale", 1.0)
            row_norm = max(np.linalg.norm(rhs_hat_tmp) / math.sqrt(len(rhs_hat_tmp)), 1e-12)
        else:
            row_norm = max(np.linalg.norm(rhs_scaled) / math.sqrt(len(rhs_scaled)), 1e-12)

        rhs_scaled /= row_norm
        local_row_scale = row_norm
        _chk_tensor("A2 rhs_scaled", rhs_scaled)
        print(f"[LOG A2] row_norm={row_norm:.3e}, matrix_scale={self.matrix_scale:.3e}, rhs_scale={rhs_scale:.3e}")

        # Решаем давление
        if self.solver is None or self.failed_amg:
            print("    CPR: AMG недоступен – Jacobi fallback")
            delta_p_scaled = (self.diag_inv / max(self.matrix_scale, 1e-30)) * rhs_scaled
        else:
            try:
                print("    CPR: Используем AMG backend")
                tol = self.gmres_tol if self.gmres_tol is not None else (1e-6 if n < 500 else (1e-4 if n < 500_000 else 1e-5))
                iters = self.gmres_max_iter if self.gmres_max_iter is not None else 200

                delta_p_geom = self.solver.solve(rhs_scaled, tol=tol, max_iter=iters)
                _chk_tensor("A3 delta_p_geom", delta_p_geom)

                # центрируем, как у вас
                delta_p_geom = delta_p_geom - delta_p_geom.mean()
                _chk_tensor("A3b delta_p_geom_centered", delta_p_geom)
                delta_p_scaled = delta_p_geom

                if np.any(~np.isfinite(delta_p_scaled)):
                    print("    CPR: AMG дал NaN/Inf -> Jacobi fallback")
                    self.failed_amg = True
                    delta_p_scaled = (self.diag_inv / max(self.matrix_scale, 1e-30)) * rhs_scaled
                else:
                    ratio_inf = np.linalg.norm(delta_p_scaled, np.inf) / (rhs_norm + 1e-30)
                    if self.backend == "geo" and ratio_inf > 1e10:
                        print("    ⚠️ Geo-AMG нестабилен, локальный Jacobi")
                        delta_p_scaled = (self.diag_inv / max(self.matrix_scale, 1e-30)) * rhs_scaled
            except Exception as e:
                print(f"    CPR: Ошибка AMG: {e} -> Jacobi fallback")
                self.failed_amg = True
                delta_p_scaled = (self.diag_inv / max(self.matrix_scale, 1e-30)) * rhs_scaled

        # Восстанавливаем phys
        safe_row_scale = np.clip(local_row_scale, 0.0, 1e6)
        delta_p_phys_np = (delta_p_scaled * safe_row_scale) / max(self.matrix_scale, 1e-30)
        _chk_tensor("A4 delta_p_phys_preclip", delta_p_phys_np)
        delta_p_phys_np = np.nan_to_num(delta_p_phys_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Собираем phys-вектор
        delta_phys_full = torch.zeros_like(vec)
        delta_phys_full[:n] = torch.from_numpy(delta_p_phys_np).to(device=vec.device, dtype=vec.dtype)

        # -------- Stage‑2 (как у вас было) --------
        try:
            props = getattr(self.simulator, "_cell_props_cache", None)
            if props is not None:
                phi, dt, V = props["phi"], props["dt"], props["V"]
                lam_w, lam_o = props["lam_w"], props["lam_o"]
                c_w, c_o = props["c_w"], props["c_o"]
                lam_g, c_g = props.get("lam_g"), props.get("c_g")

                rho_w = props["rho_w"]
                diag_SS = (phi * V * rho_w) / (dt + 1e-30)
                rhs_s_phys = self.scaler.unscale_vec(vec)[n:]

                if (not torch.isfinite(rhs_s_phys).all()) or (not torch.isfinite(diag_SS).all()):
                    raise ValueError("non-finite rhs_s or diag_SS")

                dFs_dp = (lam_w * c_w + lam_o * c_o) * V / (dt + 1e-30)
                if lam_g is not None and c_g is not None:
                    dFs_dp = dFs_dp + lam_g * c_g * V / (dt + 1e-30)

                # кламп давления
                P_CLIP = 20.0e6
                delta_phys_full[:n] = delta_phys_full[:n].clamp(-P_CLIP, P_CLIP)

                vp = rhs_s_phys.numel() // n
                delta_s_list = []
                for s in range(vp):
                    s0, s1 = s * n, (s + 1) * n
                    rhs_sat = rhs_s_phys[s0:s1]
                    delta_sat = (rhs_sat - dFs_dp * delta_phys_full[:n].cpu().numpy()) / (diag_SS + 1e-30)
                    # преобразуем к torch и клампим
                    delta_sat = torch.as_tensor(delta_sat, device=vec.device, dtype=vec.dtype)
                    delta_sat = torch.clamp(delta_sat, -0.05, 0.05)
                    delta_s_list.append(delta_sat)

                if delta_s_list:
                    delta_s_full = torch.cat(delta_s_list, dim=0)
                    if not torch.isfinite(delta_s_full).all():
                        delta_s_full.zero_()
                    delta_phys_full[n:n + rhs_s_phys.numel()] = delta_s_full
        except Exception as _e:
            if not hasattr(self, "_warn_stage2"):
                print(f"[CPR] Stage-2 saturation update failed: {_e}")
                self._warn_stage2 = True

        # -------- ψ-tail (по желанию – оставляем как было, если нужно) --------
        try:
            from solver.csr_full import assemble_full_csr
            from solver.chebyshev import chebyshev_smooth
        except ImportError:
            assemble_full_csr = None
            chebyshev_smooth = None

        if assemble_full_csr is not None and chebyshev_smooth is not None:
            if not hasattr(self, "_full_A"):
                n_total = vec.shape[0]
                vars_pc = max(2, min(3, n_total // n))
                indptr_f, indices_f, data_f = assemble_full_csr(
                    self._indptr_p, self._indices_p, self._data_p,
                    vars_per_cell=vars_pc, diag_sat=1.0)
                self._full_A = torch.sparse_csr_tensor(
                    torch.from_numpy(indptr_f),
                    torch.from_numpy(indices_f),
                    torch.from_numpy(data_f).to(torch.float32),
                    size=(vec.shape[0], vec.shape[0])
                )

            A_full = self._full_A
            n_blocks = 2 if self._n_cells > 1_000_000 else 1
            delta_hat_tmp = self.scaler.scale_vec(delta_phys_full).to(vec.device, vec.dtype)
            for _ in range(n_blocks):
                r_hat_cpu = vec.cpu() - torch.sparse.mm(A_full, delta_hat_tmp.cpu().unsqueeze(1)).squeeze(1)
                delta_inc_cpu, _ = chebyshev_smooth(A_full, r_hat_cpu,
                                                    torch.zeros_like(r_hat_cpu), iters=2, omega=0.7)
                delta_hat_tmp = delta_hat_tmp + delta_inc_cpu.to(vec.device)
            # перенесём обратно в phys, чтобы клампнуть давление, затем снова в hat
            delta_phys_full = self.scaler.unscale_vec(delta_hat_tmp)

        # Финальные клампы и проверки
        pressure_result = delta_phys_full[:n]
        rhs_norm_hat = vec[:n].norm().item()
        rhs_norm_phys = rhs_norm_hat * float(getattr(self, "p_scale", 1.0))
        clamp_val = max(1e7, min(10.0 * rhs_norm_phys / (math.sqrt(float(n)) + 1e-30), 2e7))
        pressure_result = pressure_result.clamp(-clamp_val, clamp_val)
        delta_phys_full[:n] = pressure_result

        final_norm = pressure_result.norm().item()
        rhs_norm_torch = vec[:n].norm().item() + 1e-30
        if self.backend not in ("geo", "geo2") and n > 500 and rhs_norm_torch > 1e-6 and final_norm > 1e9 * rhs_norm_torch:
            print("    CPR: Δp экстремально велико – обнуляем")
            delta_phys_full[:n].zero_()

        # ---- ВАЖНО: возвращаем ВСЕГДА в global-hat ----
        delta_hat_full = self.scaler.scale_vec(delta_phys_full).to(vec.device, vec.dtype)
        return delta_hat_full
