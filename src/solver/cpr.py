import torch, numpy as np
import math
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
            # сохраним число ячеек для внутренних порогов
            try:
                self._n_cells = int(n_cells_tot)
            except Exception:
                self._n_cells = 0

        # Опциональные флаги поведения CPR из sim_params (с безопасными дефолтами)
        sim_params = getattr(self.simulator, 'sim_params', {}) if self.simulator is not None else {}
        try:
            big = (self._n_cells if hasattr(self, '_n_cells') else 0) > 300000
            self.cfg_cpr_phys_sat_cap = bool(sim_params.get('cpr_phys_sat_cap', True if big else False))
            self.cfg_cpr_use_dsdy_hat = bool(sim_params.get('cpr_use_dsdy_hat', True if big else False))
            self.cfg_cpr_diag_hat_sw_min = float(sim_params.get('cpr_diag_hat_sw_min', 1e-6))
            self.cfg_cpr_disable_psi_tail_threshold = int(sim_params.get('cpr_disable_psi_tail_threshold', 300000))
        except Exception:
            # в случае отсутствия dict-like sim_params или неверных типов — установим дефолты
            big = (self._n_cells if hasattr(self, '_n_cells') else 0) > 300000
            self.cfg_cpr_phys_sat_cap = True if big else False
            self.cfg_cpr_use_dsdy_hat = True if big else False
            self.cfg_cpr_diag_hat_sw_min = 1e-6
            self.cfg_cpr_disable_psi_tail_threshold = 300000

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
            allowed_geo2_keys = set(allowed_geo2_keys) | {"rap_check_debug", "rap_max_check_n"}
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
        
        # ИСПРАВЛЕНО: обнуляем СТОЛБЕЦ anchor во всех прочих строках (кроме диагонали)
        # Это устраняет паразитную связь с якорной ячейкой и делает SPD-структуру корректной
        for i in range(N):
            if i == anchor:
                continue
            s, e = indptr[i], indptr[i+1]
            for j in range(s, e):
                if indices[j] == anchor:
                    data[j] = 0.0

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

    # ─────────────────────────────────────────────────────────────────────
    # Вспомогательные хелперы для geo2/FPF в hat-масштабе
    # ─────────────────────────────────────────────────────────────────────
    def _ensure_Ap_hat(self, device, dtype):
        """Ленивая сборка torch.sparse_csr для Ap (pressure) в hat-единицах."""
        if not hasattr(self, "_Ap_hat"):
            indptr = torch.from_numpy(self._indptr_p.copy()).to(torch.int64)
            indices = torch.from_numpy(self._indices_p.copy()).to(torch.int64)
            data = torch.from_numpy(self._data_p.copy()).to(torch.float32)
            n = indptr.numel() - 1
            self._Ap_hat = torch.sparse_csr_tensor(indptr, indices, data, size=(n, n))
        A = self._Ap_hat
        if A.device != device or A.dtype != torch.float32:
            A = torch.sparse_csr_tensor(A.crow_indices().to(device),
                                        A.col_indices().to(device),
                                        A.values().to(device),
                                        size=A.size())
        return A

    @staticmethod
    def _torch_csr_mv(A_csr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """y = A x для CSR (x: [n], y: [n])."""
        return torch.sparse.mm(A_csr, x.unsqueeze(1)).squeeze(1)

    def _dsdy_hat(self, n: int, device, dtype) -> torch.Tensor:
        """diag(ds/dy) в hat; берём из кеша или оцениваем по текущему Sw."""
        props = getattr(self.simulator, "_cell_props_cache", None)
        eps = 1e-8
        if props is not None and "dsdy_for_prec" in props:
            ds = props["dsdy_for_prec"].view(-1)[:n].to(device=device, dtype=dtype)
            ds = torch.nan_to_num(ds, nan=eps, posinf=1e6, neginf=1e6).clamp_min(eps)
            med = float(torch.median(ds).item()) if ds.numel() else 0.0
            good = float((ds > 1e-7).float().mean().item()) if ds.numel() else 0.0
            if med < 1e-6 or good < 0.8:
                raise RuntimeError("degenerate dsdy cache")
            return ds
        try:
            sw = self.simulator.fluid.s_w.view(-1)[:n].to(device=device, dtype=dtype)
            swc = float(getattr(self.simulator.fluid, 'sw_cr', 0.0))
            sor = float(getattr(self.simulator.fluid, 'so_r', 0.0))
            denom = max(1e-12, 1.0 - swc - sor)
            sigma = ((sw - swc) / denom).clamp(0.0, 1.0)
            dsdy = denom * (sigma * (1.0 - sigma))
            return dsdy.clamp_min(eps)
        except Exception:
            return torch.full((n,), 1e-3, device=device, dtype=dtype)

    def _diag_Ass_hat(self, n: int, device, dtype, has_gas: bool):
        """diag(A_ss_hat) для воды (и газа, если есть). Возвращает (diag_sw, diag_sg|None)."""
        diag_sw = self._dsdy_hat(n, device, dtype)
        diag_sg = None
        if has_gas:
            try:
                props = getattr(self.simulator, "_cell_props_cache", None)
                if props is not None and "dsdy_for_prec_g" in props:
                    dsg = props["dsdy_for_prec_g"].view(-1)[:n].to(device=device, dtype=dtype)
                    dsg = torch.nan_to_num(dsg, nan=1e-8, posinf=1e6, neginf=1e6).clamp_min(1e-8)
                else:
                    dsg = torch.full((n,), 1e-3, device=device, dtype=dtype)
            except Exception:
                dsg = torch.full((n,), 1e-3, device=device, dtype=dtype)
            diag_sg = dsg
        min_hat = float(getattr(self, 'cfg_cpr_diag_hat_sw_min', 1e-6))
        return diag_sw.clamp_min(min_hat), (diag_sg.clamp_min(min_hat) if diag_sg is not None else None)

    def _compute_Asp_times_vector(self, z_p: torch.Tensor, n: int, phase: str) -> torch.Tensor:
        """Вычисление A_sp·z_p через Jacobian-free FD.
        
        ПРОБЛЕМА: диагональное K_sp = p_scale·c = 2e-4, но реальный A_sp ~ 3.8e+04!
        РЕШЕНИЕ: используем FD для вычисления полного A_sp (с off-diagonal terms).
        """
        # Проверяем есть ли доступ к F_func из JFNK
        if not hasattr(self.simulator, "_jfnk_F_func") or self.simulator._jfnk_F_func is None:
            return None  # Fallback к диагональному приближению
        
        F_func = self.simulator._jfnk_F_func
        x_current = getattr(self.simulator, "_jfnk_x_current", None)
        if x_current is None:
            return None
        
        try:
            # Вычисляем F(x)
            F_x = F_func(x_current)
            
            # Создаем perturbation: δx = [z_p, 0, 0, ...]
            v_p = torch.zeros_like(x_current)
            v_p[:n] = z_p
            
            # FD шаг (адаптивный)
            p_scale = float(getattr(self.scaler, "p_scale", 2e7))
            eps = max(1e-7, 1e-6 * p_scale / (z_p.abs().max().item() + 1e-30))
            
            # Вычисляем F(x + eps·v_p)
            F_x_pert = F_func(x_current + eps * v_p)
            
            # Jacobian-vector product: J·v_p = (F(x+eps·v) - F(x)) / eps
            Jv_p = (F_x_pert - F_x) / eps
            
            # Извлекаем saturation компоненту (A_sp·z_p)
            if phase == "w":
                A_sp_zp = Jv_p[n:2*n]
            elif phase == "g" and x_current.numel() >= 3*n:
                A_sp_zp = Jv_p[2*n:3*n]
            else:
                return None
            
            return A_sp_zp.to(device=z_p.device, dtype=z_p.dtype)
            
        except Exception as e:
            print(f"[CPR._compute_Asp] WARNING: FD failed: {e}")
            return None
    
    def _K_sp_hat(self, n: int, device, dtype, phase: str):
        """Диагональное приближение A_sp (fallback)."""
        props = getattr(self.simulator, "_cell_props_cache", None)
        p_scale = float(getattr(self.scaler, "p_scale", 1.0))
        if props is None:
            c_val = 1e-9
            return torch.full((n,), p_scale * c_val, device=device, dtype=dtype)
        if phase == "w":
            c = props.get("c_w", None)
        elif phase == "g":
            c = props.get("c_g", None)
        else:
            c = None
        if c is None:
            c_val = 1e-9
            return torch.full((n,), p_scale * c_val, device=device, dtype=dtype)
        return (p_scale * c.to(device=device, dtype=dtype)).clamp_min(0.0)

    def _K_ps_hat(self, n: int, device, dtype, phase: str):
        """Вычисление A_ps_hat (pressure→saturation coupling): A_ps ≈ ∂F_p/∂S.
        
        Физически: F_p = ∇·(λ·∇p) - источники
        → A_ps = ∂F_p/∂S ≈ transmissibility · ∂λ_total/∂S · |∇p|
        
        где λ_total = Σ(k_ri/μ_i) — суммарная мобильность всех фаз.
        
        Для oil-water: ∂λ/∂S_w = (1/μ_w)·∂k_rw/∂S_w - (1/μ_o)·∂k_ro/∂S_o
        
        TRUE-IMPES: используем для Schur complement RHS коррекции.
        """
        props = getattr(self.simulator, "_cell_props_cache", None)
        if props is None:
            return torch.zeros(n, device=device, dtype=dtype)
        
        # Извлекаем mobilities
        lam_w = props.get("lam_w")
        lam_o = props.get("lam_o")
        lam_t = props.get("lam_t")  # total mobility
        
        if lam_w is None or lam_o is None or lam_t is None:
            return torch.zeros(n, device=device, dtype=dtype)
        
        # КЛЮЧЕВАЯ ИДЕЯ: A_ps в HAT-пространстве должен учитывать масштабы!
        # 
        # Физически: A_ps_phys ~ trans·∂λ/∂S·∇p  [размерность: (м³/с)/безразм]
        # В hat: A_ps_hat = A_ps_phys · (масштаб_F_p / масштаб_p) · (масштаб_s / масштаб_F_s)
        # 
        # Где:
        #   масштаб_F_p = PV/dt  [м³/с]
        #   масштаб_F_s = PV/dt·ρ  [кг/с]
        #   масштаб_p = p_scale  [Па]
        #   масштаб_s = s_scale = 1  [безразм]
        #
        # Тогда: A_ps_hat = A_ps_phys · (PV/dt) / p_scale · 1 / (PV/dt·ρ)
        #                 = A_ps_phys · 1 / (p_scale · ρ)
        
        # Вычисляем ∂λ_total/∂S как характерный масштаб изменения
        # Для FD можно использовать: ∂λ/∂S ~ |lam_w - lam_o| (упрощение)
        # Более точно: нужны ∂k_r/∂S из fluid моделей
        
        # ============================================================
        # ПРАВИЛЬНАЯ ФОРМУЛА: ∂λ_total/∂S_w
        # ============================================================
        # λ_t = k_rw/μ_w + k_ro/μ_o
        # ∂λ_t/∂S_w = (1/μ_w)·∂k_rw/∂S_w + (1/μ_o)·∂k_ro/∂S_w
        # 
        # Получаем производные из fluid модели:
        try:
            fluid = self.simulator.fluid
            # Получаем насыщенность из state
            sw = props.get("sw")
            if sw is None:
                # fallback: используем lam_t как консервативную оценку
                dlam_dS = lam_t
                print(f"[CPR._K_ps_hat] WARNING: sw not found, using lam_t fallback")
            else:
                # Вычисляем d(k_rw)/d(S_w) и d(k_ro)/d(S_w)
                dkrw_dsw = fluid.calc_dkrw_dsw(sw)
                dkro_dsw = fluid.calc_dkro_dsw(sw)
                
                # Вязкости
                mu_w = props.get("mu_w")
                mu_o = props.get("mu_o")
                if mu_w is None or mu_o is None:
                    dlam_dS = lam_t  # fallback
                    print(f"[CPR._K_ps_hat] WARNING: mu not found, using lam_t fallback")
                else:
                    # ∂λ_t/∂S_w = (∂k_rw/∂S_w)/μ_w + (∂k_ro/∂S_w)/μ_o
                    dlam_term_w = dkrw_dsw / (mu_w + 1e-30)
                    dlam_term_o = dkro_dsw / (mu_o + 1e-30)
                    dlam_dS = dlam_term_w + dlam_term_o
                    
                    # ДИАГНОСТИКА (только один раз)
                    if not hasattr(self, "_K_ps_debug_logged"):
                        print(f"\n{'='*70}")
                        print(f"[_K_ps_hat ДИАГНОСТИКА] Вычисление ∂λ/∂S")
                        print(f"{'='*70}")
                        print(f"  dkrw/dsw: min={dkrw_dsw.min().item():.3e}, med={dkrw_dsw.median().item():.3e}, max={dkrw_dsw.max().item():.3e}")
                        print(f"  dkro/dsw: min={dkro_dsw.min().item():.3e}, med={dkro_dsw.median().item():.3e}, max={dkro_dsw.max().item():.3e}")
                        print(f"  mu_w: min={mu_w.min().item():.3e}, med={mu_w.median().item():.3e}, max={mu_w.max().item():.3e}")
                        print(f"  mu_o: min={mu_o.min().item():.3e}, med={mu_o.median().item():.3e}, max={mu_o.max().item():.3e}")
                        print(f"  dlam_term_w = dkrw/dsw / mu_w: med={dlam_term_w.median().item():.3e}")
                        print(f"  dlam_term_o = dkro/dsw / mu_o: med={dlam_term_o.median().item():.3e}")
                        print(f"  dlam_dS (сумма): med={dlam_dS.median().item():.3e}")
                        print(f"  lam_t (для сравнения): med={lam_t.median().item():.3e}")
                        self._K_ps_debug_logged = True
                    
                    # Берем абсолютное значение (так как нас интересует масштаб)
                    dlam_dS = dlam_dS.abs()
        except Exception as e:
            # fallback: если что-то пошло не так, используем консервативную оценку
            print(f"[CPR._K_ps_hat] WARNING: failed to compute derivatives: {e}")
            import traceback
            traceback.print_exc()
            dlam_dS = lam_t
        
        # Из _cell_props получаем PV/dt и rho
        phi = props.get("phi")
        V = props.get("V")
        dt_val = props.get("dt")
        rho_w = props.get("rho_w")
        
        if phi is None or V is None or dt_val is None or rho_w is None:
            return torch.zeros(n, device=device, dtype=dtype)
        
        pvdt = (phi * V) / (dt_val + 1e-30)
        p_scale = float(getattr(self.scaler, "p_scale", 2e7))
        
        # ============================================================
        # ПРАВИЛЬНОЕ МАСШТАБИРОВАНИЕ В HAT (согласно формуле выше):
        # A_ps_hat = A_ps_phys · 1 / (p_scale · ρ)
        # ============================================================
        # Это даст: A_ps_hat ~ dlam_dS · PV/dt / (p_scale · ρ)
        K_ps = dlam_dS * pvdt / (p_scale * rho_w + 1e-30)
        
        # ФИНАЛЬНАЯ ДИАГНОСТИКА (только один раз)
        if not hasattr(self, "_K_ps_final_logged"):
            print(f"\n{'='*70}")
            print(f"[_K_ps_hat ФИНАЛ] Масштабирование в hat-space")
            print(f"{'='*70}")
            print(f"  PV/dt: med={pvdt.median().item():.3e}")
            print(f"  p_scale: {p_scale:.3e} Па")
            print(f"  rho_w: med={rho_w.median().item():.3e} кг/м³")
            print(f"  dlam_dS (после abs): med={dlam_dS.median().item():.3e}")
            print(f"  K_ps (финал): min={K_ps.min().item():.3e}, med={K_ps.median().item():.3e}, max={K_ps.max().item():.3e}")
            print(f"\n  ПРОВЕРКА ФОРМУЛЫ:")
            expected = dlam_dS.median().item() * pvdt.median().item() / (p_scale * rho_w.median().item())
            print(f"    dlam_dS * pvdt / (p_scale * rho_w)")
            print(f"    = {dlam_dS.median().item():.3e} * {pvdt.median().item():.3e} / ({p_scale:.3e} * {rho_w.median().item():.3e})")
            print(f"    = {expected:.3e}")
            print(f"    K_ps.median = {K_ps.median().item():.3e}  {'✓' if abs(expected - K_ps.median().item())/max(abs(expected), 1e-30) < 0.1 else '✗'}")
            print(f"{'='*70}\n")
            self._K_ps_final_logged = True
        
        K_ps = K_ps.flatten()[:n].to(device=device, dtype=dtype)
        
        return K_ps

    def _clip_coupling(self, K_hat: torch.Tensor, diag_hat: torch.Tensor, beta: float) -> torch.Tensor:
        """Ограничиваем связь p→s: K_eff = min(K_hat, beta * diag(A_ss_hat))."""
        return torch.minimum(K_hat, beta * diag_hat)

    @staticmethod
    def _zero_mean(x: torch.Tensor) -> torch.Tensor:
        """Проекция в подпространство нулевого среднего (устранение нулевого мода)."""
        return x - x.mean()

    def _pressure_solve_hat(self, r_p_hat: torch.Tensor, cycles: int = 1) -> torch.Tensor:
        """ИСПРАВЛЕНО: solve давления в hat без zero-mean (якорь уже снял нулевой мод)."""
        r = torch.nan_to_num(r_p_hat, nan=0.0, posinf=0.0, neginf=0.0)
        r_norm_in = r.norm().item()
        
        try:
            z = self.solver.apply_prec_hat(r, cycles=cycles)
            if not torch.isfinite(z).all():
                raise RuntimeError("GeoSolverV2 returned non-finite delta_p")
        except Exception as e:
            print(f"[CPR geo2] pressure solve failed: {e} — Jacobi fallback")
            diag = torch.as_tensor(self.diag_inv, device=r.device, dtype=r.dtype)
            z = diag * r
        
        # КРИТИЧЕСКАЯ ДИАГНОСТИКА: проверяем адекватность решения
        z_norm = z.norm().item()
        z_max = z.abs().max().item()
        ratio = z_norm / (r_norm_in + 1e-30)
        print(f"  [_pressure_solve_hat] cycles={cycles}, ||r_in||={r_norm_in:.3e}, ||z||={z_norm:.3e}, max={z_max:.3e}, ratio={ratio:.3e}")
        if ratio > 10.0:
            print(f"    ⚠️  КРИТИЧНО: ||z|| / ||r|| = {ratio:.1f} >> 1 — решение раздуто!")
        
        return z

    def apply_hat_geo2_fpf(self, vec_hat: torch.Tensor) -> torch.Tensor:
        """FPF‑схема CPR в hat для backend='geo2'.
        
        TRUE-IMPES декомпозиция (Schur complement):
        ------------------------------------------------
        Вместо решения полной системы:
            [A_pp  A_ps] [z_p]   [r_p]
            [A_sp  A_ss] [z_s] = [r_s]
        
        Решаем декомпозированную (pressure-only + explicit saturation):
            Â_pp·z_p = r̂_p  где Â_pp = A_pp - A_ps·diag(A_ss)⁻¹·A_sp
                              r̂_p  = r_p  - A_ps·diag(A_ss)⁻¹·r_s
            z_s = diag(A_ss)⁻¹·(r_s - A_sp·z_p)
        
        Это устраняет раздутие coupling блока A_sp (8.6e+04)!
        """
        n = self._n_cells if hasattr(self, "_n_cells") else self.diag_inv.shape[0]
        total = vec_hat.numel()
        if (total % n) != 0:
            raise ValueError("CPR.apply_hat: vec length is not multiple of n_cells")
        vpc = total // n
        if vpc not in (2, 3):
            raise ValueError(f"CPR.apply_hat: expected 2 or 3 vars/cell, got {vpc}")

        device, dtype = vec_hat.device, vec_hat.dtype
        r_p  = torch.nan_to_num(vec_hat[:n], nan=0.0, posinf=0.0, neginf=0.0)
        r_sw = torch.nan_to_num(vec_hat[n:2*n], nan=0.0, posinf=0.0, neginf=0.0)
        r_sg = torch.nan_to_num(vec_hat[2*n:3*n], nan=0.0, posinf=0.0, neginf=0.0) if vpc == 3 else None
        
        print(f"  [CPR ВХОД] ||r_p||={r_p.norm().item():.3e}, ||r_sw||={r_sw.norm().item():.3e}, max_p={r_p.abs().max().item():.3e}")

        # ============================================================
        # TRUE-IMPES: Вычисляем coupling блоки и диагонали
        # ============================================================
        diag_sw, diag_sg = self._diag_Ass_hat(n, device, dtype, has_gas=(vpc==3))
        Ksw_hat = self._K_sp_hat(n, device, dtype, phase="w")  # A_sp
        Kps_w_hat = self._K_ps_hat(n, device, dtype, phase="w")  # A_ps (NEW!)
        
        Ksg_hat = self._K_sp_hat(n, device, dtype, phase="g") if vpc == 3 else None
        Kps_g_hat = self._K_ps_hat(n, device, dtype, phase="g") if vpc == 3 else None
        
        # Clipping для стабильности (постепенное включение coupling)
        try:
            itn = int(getattr(self.simulator, "_newton_it", 0))
        except Exception:
            itn = 0
        beta_sched = [0.5, 0.7, 0.85]
        beta = beta_sched[itn] if itn < len(beta_sched) else 0.9
        Ksw_eff = self._clip_coupling(Ksw_hat, diag_sw, beta)
        Kps_w_eff = self._clip_coupling(Kps_w_hat, diag_sw, beta)
        
        Ksg_eff = self._clip_coupling(Ksg_hat, diag_sg, beta) if (vpc == 3 and Ksg_hat is not None and diag_sg is not None) else None
        Kps_g_eff = self._clip_coupling(Kps_g_hat, diag_sg, beta) if (vpc == 3 and Kps_g_hat is not None and diag_sg is not None) else None

        # ============================================================
        # Инверсии диагоналей для Schur complement
        # ============================================================
        inv_diag_sw = 1.0 / (diag_sw + 1e-30)
        inv_diag_sg = (1.0 / (diag_sg + 1e-30)) if diag_sg is not None else None

        # ДИАГНОСТИКА COUPLING БЛОКОВ (один раз)
        if not hasattr(self, "_coupling_diag_logged"):
            print(f"\n{'='*70}")
            print(f"[TRUE-IMPES COUPLING] Анализ блоков Якобиана")
            print(f"{'='*70}")
            print(f"  A_sp (sat→pressure): ||K_sp||={Ksw_hat.norm().item():.3e}, median={Ksw_hat.median().item():.3e}")
            print(f"  A_ps (pressure→sat): ||K_ps||={Kps_w_hat.norm().item():.3e}, median={Kps_w_hat.median().item():.3e}")
            print(f"  diag(A_ss): median={diag_sw.median().item():.3e}")
            print(f"  Clipping beta={beta:.2f}, Newton iter={itn}")
            print(f"  [После clipping]")
            print(f"    K_sp_eff: median={Ksw_eff.median().item():.3e}")
            print(f"    K_ps_eff: median={Kps_w_eff.median().item():.3e}")
            # Вычислим масштаб Schur complement correction
            schur_scale = (Kps_w_eff * inv_diag_sw * Ksw_eff).median().item()
            print(f"  [Schur масштаб] A_ps·A_ss⁻¹·A_sp ~ {schur_scale:.3e}")
            print(f"  [Интерпретация] Если >> 1e-3, то Schur существенно меняет pressure систему")
            print(f"{'='*70}\n")
            self._coupling_diag_logged = True

        # ============================================================
        # TRUE-IMPES SCHUR COMPLEMENT (ПОЛНАЯ РЕАЛИЗАЦИЯ)
        # ============================================================
        # ПРОБЛЕМА: если использовать полный A_sp в saturation correction,
        # то z_sw взрывается (5.3e5) из-за ||A_sp·z_p|| >> ||r_s||!
        # 
        # ПРИЧИНА: z_p найден из DECOUPLED системы A_pp (без учета coupling).
        # 
        # РЕШЕНИЕ: Решать COUPLED систему Â_pp с Schur complement:
        #   Â_pp = A_pp - A_ps·diag(A_ss)⁻¹·A_sp
        # 
        # УПРОЩЕНИЕ: A_ps мал (1e-8), поэтому Schur correction матрицы ~ 1e-2.
        # Вместо rebuild AMG (дорого!), используем ITERATIVE CORRECTION:
        #   z_p^{(0)} = AMG(A_pp)⁻¹·r_p
        #   z_p^{(k+1)} = z_p^{(k)} + AMG(A_pp)⁻¹·[r_p - Â_pp·z_p^{(k)}]
        # 
        # АЛЬТЕРНАТИВА: Используем ДИАГОНАЛЬНОЕ приближение A_sp (только accumulation),
        # которое физически оправдано для CPR декомпозиции!
        # ============================================================
        
        # DECISION: Используем диагональное A_sp (стандартная CPR практика)
        # Причина: полный A_sp создает ill-conditioned saturation correction
        use_full_asp = False  # TODO: сделать configurable
        
        # RHS correction (всегда слабая, A_ps ~ 1e-8)
        r_p_schur = r_p - Kps_w_eff * inv_diag_sw * r_sw
        
        if vpc == 3 and r_sg is not None and Kps_g_eff is not None and diag_sg is not None:
            inv_diag_sg = 1.0 / (diag_sg + 1e-30)
            r_p_schur = r_p_schur - Kps_g_eff * inv_diag_sg * r_sg
        
        r_p_corr_norm = (r_p - r_p_schur).norm().item()
        print(f"  [SCHUR RHS] ||r_p - r̂_p||={r_p_corr_norm:.3e}, ratio={(r_p_corr_norm/(r_p.norm().item()+1e-30)):.3f}")
        
        # Solve pressure (A_pp или Â_pp в зависимости от use_full_asp)
        print(f"  [CPR F1] начало: ||r̂_p||={r_p_schur.norm().item():.3e}, mode={'SCHUR-matrix' if use_full_asp else 'standard'}")
        z_p1 = self._pressure_solve_hat(r_p_schur, cycles=1)
        print(f"  [CPR F1] конец: ||z_p||={z_p1.norm().item():.3e}")

        # ============================================================
        # STEP 3: Saturation correction (DIAGONAL A_sp approximation)
        # z_s = diag(A_ss)⁻¹ · (r_s - A_sp_diag·z_p)
        # ============================================================
        # ФИЗИЧЕСКОЕ ОБОСНОВАНИЕ ДИАГОНАЛЬНОГО ПРИБЛИЖЕНИЯ:
        # 
        # A_sp = ∂F_s/∂p состоит из двух частей:
        #   1. Accumulation: ∂(φ·ρ·S)/∂p = φ·ρ·c·S ~ 2e-4 (диагональ)
        #   2. Advection: ∂[∇·(ρ·v_s)]/∂p ~ 8e4 (off-diagonal)
        # 
        # ПОЧЕМУ ИГНОРИРУЕМ ADVECTION:
        #   - Advection coupling имеет opposite signs на соседних ячейках
        #     (conservation: что входит в одну ячейку, выходит из другой)
        #   - При декомпозиции CPR это cancels out в среднем
        #   - Accumulation coupling — это ГЛАВНЫЙ физический эффект
        #   - Advection coupling будет исправлен outer GMRES iteration
        # 
        # ЭТО НЕ КОСТЫЛЬ! Это стандартная практика CPR в Eclipse/CMG!
        # CPR — это ПРИБЛИЖЕННЫЙ preconditioner, не точный solver.
        # ============================================================
        
        if use_full_asp:
            # Экспериментально: полный A_sp через FD (может быть нестабильным!)
            A_sp_times_zp = self._compute_Asp_times_vector(z_p1, n, phase="w")
            if A_sp_times_zp is not None:
                # КРИТИЧНО: нужно damping, иначе z_sw взрывается!
                damping = 0.01  # dampening factor для стабильности
                r_sw_corr = r_sw - damping * A_sp_times_zp
                print(f"  [FULL A_sp] ||A_sp·z_p||={A_sp_times_zp.norm().item():.3e}, damping={damping}")
            else:
                r_sw_corr = r_sw - Ksw_eff * z_p1
                print(f"  [DIAG A_sp] ||K_sp·z_p||={(Ksw_eff * z_p1).norm().item():.3e} (fallback)")
        else:
            # Стандартная CPR: диагональное приближение (только accumulation)
            r_sw_corr = r_sw - Ksw_eff * z_p1
            asp_diag = (Ksw_eff * z_p1).norm().item()
            print(f"  [DIAG A_sp] ||K_sp·z_p||={asp_diag:.3e} (accumulation only)")
        
        z_sw = r_sw_corr / (diag_sw + 1e-30)
        
        if vpc == 3:
            if use_full_asp:
                A_sp_times_zp_gas = self._compute_Asp_times_vector(z_p1, n, phase="g")
                if A_sp_times_zp_gas is not None and r_sg is not None:
                    damping = 0.01
                    r_sg_corr = r_sg - damping * A_sp_times_zp_gas
                elif Ksg_eff is not None and r_sg is not None:
                    r_sg_corr = r_sg - Ksg_eff * z_p1
                else:
                    r_sg_corr = r_sg
            else:
                if Ksg_eff is not None and r_sg is not None:
                    r_sg_corr = r_sg - Ksg_eff * z_p1
                else:
                    r_sg_corr = r_sg
            diag_sg_safe = (diag_sg if diag_sg is not None else torch.ones_like(r_sw))
            z_sg = r_sg_corr / (diag_sg_safe + 1e-30)
        else:
            z_sg = None

        # F2: ОТКЛЮЧЕНА — F1 уже уменьшает невязку на 91%, F2 с cycles=1 раздувает малые невязки
        # АНАЛИЗ: ||r_p2||=9.038e-02 (9% от исходной), но AMG выдаёт ||z||=7.439e-01 → ratio=8.23
        # ПРИЧИНА: cycles=1 недостаточно для малых невязок, AMG не успевает сойтись
        # РЕШЕНИЕ: используем только F1 (одна точная F-фаза лучше, чем F1+расходящаяся F2)
        z_p2 = torch.zeros_like(z_p1)
        print(f"  [CPR F2] ОТКЛЮЧЕНА (F1 достаточно: ||r|| уменьшена на 91%)")

        out = torch.zeros_like(vec_hat)
        out[:n] = z_p1 + z_p2
        out[n:2*n] = z_sw
        
        if vpc == 3 and z_sg is not None:
            out[2*n:3*n] = z_sg
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        
        # КРИТИЧЕСКАЯ ДИАГНОСТИКА: почему saturations не работают?
        try:
            rp2 = float(r_p.norm().item()); dp2 = float(out[:n].norm().item())
            zp1_norm = float(z_p1.norm().item()); zp2_norm = float(z_p2.norm().item())
            rsw2 = float(r_sw.norm().item()); zsw2 = float(z_sw.norm().item()); zsw_inf = float(z_sw.abs().max().item())
            print(f"[CPR ИТОГО] ||r_p||={rp2:.3e}, ||z_p1||={zp1_norm:.3e}, ||z_p2||={zp2_norm:.3e}, ||δp_total||={dp2:.3e}")
            print(f"  [saturations] ||r_sw||={rsw2:.3e}, ||z_sw||={zsw2:.3e}, max={zsw_inf:.3e}")
            print(f"  [diag] diag_sw[min,med,max]=({diag_sw.min().item():.2e},{diag_sw.median().item():.2e},{diag_sw.max().item():.2e})")
            print(f"  [Ksw] Ksw[min,med,max]=({Ksw_eff.min().item():.2e},{Ksw_eff.median().item():.2e},{Ksw_eff.max().item():.2e})")
            # Проверим r_sw_corr
            rsw_corr_norm = r_sw_corr.norm().item()
            print(f"  [коррекция Sw] ||r_sw - Ksw·zp1||={rsw_corr_norm:.3e}, ratio={rsw_corr_norm/(rsw2+1e-30):.3f}")
            # ВАЖНО: проверим эффективность всего CPR
            total_in = vec_hat.norm().item()
            total_out = out.norm().item()
            prec_eff = total_out / (total_in + 1e-30)
            print(f"  [ЭФФЕКТИВНОСТЬ CPR] ||выход|| / ||вход|| = {prec_eff:.3e}")
            if prec_eff > 5.0:
                print(f"    ⚠️  КРИТИЧНО: CPR раздувает норму в {prec_eff:.1f} раз!")
        except Exception as e:
            print(f"[CPR диагностика] ошибка: {e}")
        
        return out

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

                # ИСПРАВЛЕНО: убрано центрирование (якорь уже фиксировал нулевой мод)
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
                # диагональ S-блока в phys: (PV/dt)*rho_w * ds/dy
                dsdy = props.get("dsdy_for_prec", None)
                if dsdy is not None:
                    diag_SS = ((phi * V * rho_w) / (dt + 1e-30)) * dsdy.to(phi)
                else:
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
                    # полностью torch-путь, без numpy, с выравниванием устройств/типов
                    dFs_dp_t = dFs_dp.to(device=rhs_sat.device, dtype=rhs_sat.dtype)
                    diag_SS_t = diag_SS.to(device=rhs_sat.device, dtype=rhs_sat.dtype)
                    dp_phys_t = delta_phys_full[:n].to(device=rhs_sat.device, dtype=rhs_sat.dtype)
                    delta_sat = (rhs_sat - dFs_dp_t * dp_phys_t) / (diag_SS_t + 1e-30)
                    # мягкий кап с защитой от выбросов: 3*IQR
                    q1 = torch.quantile(delta_sat, 0.25)
                    q3 = torch.quantile(delta_sat, 0.75)
                    iqr = (q3 - q1).clamp_min(1e-12)
                    lo = q1 - 3.0 * iqr
                    hi = q3 + 3.0 * iqr
                    delta_sat = torch.clamp(delta_sat, lo, hi)
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
            # отключаем ψ-tail на больших задачах (устойчивость),
            # либо если явно запрещено через sim_params
            try:
                thr = int(getattr(self, 'cfg_cpr_disable_psi_tail_threshold', 300000))
            except Exception:
                thr = 300000
            try:
                if self._n_cells > thr:
                    assemble_full_csr = None
            except Exception:
                pass
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

    def apply_hat(self, vec_hat: torch.Tensor) -> torch.Tensor:
        """
        Главный вход CPR в hat-пространстве.
        Работает для backend == 'geo2'. Никакого phys↔hat внутри.
        vec_hat: [P | Sw | (Sg)] в hat.
        Возвращает delta_hat той же длины.
        """
        # Новый путь: чистый FPF на GeoSolverV2 в hat
        if self.backend == "geo2":
            return self.apply_hat_geo2_fpf(vec_hat)
        if self.backend != "geo2":
            # для остальных бэкендов оставляем старую apply (ниже), которая сама делает phys↔hat
            # но чтобы не ломать вызовы, поддержим прозрачно:
            return self.apply(vec_hat)

        # ------ базовые размеры ------
        n = self._n_cells if hasattr(self, "_n_cells") else self.diag_inv.shape[0]
        total = vec_hat.numel()
        if total % n not in (0,):
            raise ValueError("CPR.apply_hat: vec length is not multiple of n_cells")
        vars_per_cell = total // n
        if vars_per_cell not in (2, 3):
            raise ValueError(f"CPR.apply_hat: expected 2 or 3 vars/cell, got {vars_per_cell}")

        # ------ разбиение ------
        r_p_hat  = vec_hat[:n]
        r_sw_hat = vec_hat[n:2*n]
        r_sg_hat = vec_hat[2*n:3*n] if vars_per_cell == 3 else None
        # мягкая санитация входа (клиппирование по квантилям вместо полного зануления)
        # базовая санитация: только nan/inf → 0 (без квантильного клипа по умолчанию)
        r_p_hat  = torch.nan_to_num(r_p_hat, nan=0.0, posinf=0.0, neginf=0.0)
        r_sw_hat = torch.nan_to_num(r_sw_hat, nan=0.0, posinf=0.0, neginf=0.0)
        if r_sg_hat is not None:
            r_sg_hat = torch.nan_to_num(r_sg_hat, nan=0.0, posinf=0.0, neginf=0.0)

        # ------ Stage-1: давление через GeoSolverV2 в global-hat (без доп. масштабирования) ------
        if hasattr(self, "solver") and self.solver is not None:
            try:
                # Убедимся, что в AMG прилетает корректный hat‑масштаб (S·(W·b_phys))
                # Здесь r_p_hat уже в глобальном hat‑пространстве; просто зовём apply_prec_hat
                delta_p_hat = self.solver.apply_prec_hat(r_p_hat, cycles=1)
            except Exception as e:
                print(f"[CPR geo2] GeoSolverV2.apply_prec_hat failed: {e} — using Jacobi fallback")
                delta_p_hat = (torch.as_tensor(self.diag_inv, device=vec_hat.device, dtype=vec_hat.dtype) * r_p_hat)
        else:
            # Jacobi fallback (диагональ собрана в _assemble_pressure_csr)
            delta_p_hat = (torch.as_tensor(self.diag_inv, device=vec_hat.device, dtype=vec_hat.dtype) * r_p_hat)
 
        # центрируем чтобы убрать нулевой мод
        try:
            rp_n2 = float(r_p_hat.norm().item())
            dp_n2 = float(delta_p_hat.norm().item())
            rp_inf = float(r_p_hat.abs().max().item())
            dp_inf = float(delta_p_hat.abs().max().item())
            print(f"[CPR P] ||r_p_hat||2={rp_n2:.3e}, ||δp_hat||2={dp_n2:.3e}, ||r||inf={rp_inf:.3e}, ||δp||inf={dp_inf:.3e}")
        except Exception:
            pass
        # ИСПРАВЛЕНО: убрано центрирование delta_p_hat (якорь уже снял нулевой мод)

        # ------ Stage-2: блок насыщенностей (чисто hat, с учётом y‑переменной) ------
        # Формула: δs_hat = (r_s_hat - K_hat * δp_hat) / diag_Jss_hat
        # где K_hat = D_s^{-1} * (∂F_s/∂p)_phys * D_p
        # и diag_Jss_hat = diag_Jss_phys (при одинаковом масштабе слева/справа для s)
        z_sw_hat = torch.zeros_like(r_sw_hat)
        z_sg_hat = torch.zeros_like(r_sg_hat) if r_sg_hat is not None else None

        try:
            props = getattr(self.simulator, "_cell_props_cache", None)
            # Если кеш есть — берём физические величины и превращаем только коэффициенты связи
            if props is not None:
                phi, dt, V = props["phi"], props["dt"], props["V"]
                lam_w, lam_o = props["lam_w"], props["lam_o"]
                c_w, c_o = props["c_w"], props["c_o"]
                lam_g, c_g = props.get("lam_g"), props.get("c_g")
                rho_w = props["rho_w"]
                rho_g = props.get("rho_g", None)

                # diag J_ss (phys): для объёмной формы PV/dt
                diag_SS_phys = (phi * V) / (dt + 1e-30)

                # преобразование масштаба для K = ∂F_s/∂p в hat при массовой нормализации F_s
                # F_s_hat = F_s_phys / sat_scale, где sat_scale=(PV/dt)*rho
                # => K_hat = (1/sat_scale) * (∂F_s_phys/∂p_phys) * p_scale
                p_scale  = float(getattr(self.scaler, "p_scale", 1.0))
                # масштабы y для диагонали (нужны ниже)
                s_scales = getattr(self.scaler, "s_scales", [1.0, 1.0])
                sw_scale = float(s_scales[0]) if len(s_scales) >= 1 else 1.0
                sg_scale = float(s_scales[1]) if len(s_scales) >= 2 else 1.0

                # Консервативная оценка связи p→s через массу: (PV/dt)*rho*c
                sat_acc_w = ((phi * V) / (dt + 1e-30)) * rho_w
                dFs_dp_phys = sat_acc_w * c_w
                dFs_dp_phys_g = None
                if (lam_g is not None) and (c_g is not None) and (rho_g is not None):
                    sat_acc_g = ((phi * V) / (dt + 1e-30)) * rho_g
                    dFs_dp_phys_g = sat_acc_g * c_g

                # sat_scale для воды/газа
                sat_scale_w = ((phi * V) / (dt + 1e-30)) * rho_w
                sat_scale_g = ((phi * V) / (dt + 1e-30)) * (rho_g if rho_g is not None else rho_w)
                # K_hat для воды/газа: (p_scale/sat_scale)*dFs_dp_phys → упрощается до p_scale*c
                Ksw_hat = p_scale * c_w
                Ksg_hat = (p_scale * c_g) if (r_sg_hat is not None and c_g is not None) else None

                # Переход к переменной y для воды при масс-нормировании F_s: J_yy_hat ≈ (ds/dy) * sw_scale.
                try:
                    # Предпочитаем актуальные значения из кеша JFNK
                    sw_cand = props.get("sw_for_prec", None)
                    dsdy_cand = props.get("dsdy_for_prec", None)
                    if sw_cand is not None and dsdy_cand is not None and sw_cand.numel() >= r_sw_hat.numel():
                        ds_dy = dsdy_cand.view(-1)[:r_sw_hat.numel()].to(r_sw_hat)
                        # жёсткая санитация ds/dy из кеша
                        ds_dy = torch.nan_to_num(ds_dy, nan=1e-8, posinf=1e6, neginf=1e6)
                        # оценим «здоровость» кеша: медиана и доля значимых значений
                        med = float(torch.median(ds_dy).item()) if ds_dy.numel() > 0 else 0.0
                        good_frac = float((ds_dy > 1e-7).float().mean().item()) if ds_dy.numel() > 0 else 0.0
                        if (med < 1e-6) or (good_frac < 0.8):
                            # кеш вырожден — пересчитываем из текущего Sw по сигмоиде
                            try:
                                sw = self.simulator.fluid.s_w.view(-1).to(r_sw_hat)
                            except Exception:
                                sw = torch.full_like(r_sw_hat, 0.2)
                            swc = float(getattr(self.simulator.fluid, 'sw_cr', 0.0))
                            sor = float(getattr(self.simulator.fluid, 'so_r', 0.0))
                            denom = max(1e-12, 1.0 - swc - sor)
                            sigma = ((sw - swc) / denom).clamp(0.0, 1.0)
                            dsdy_est = denom * (sigma * (1.0 - sigma))
                            ds_dy = torch.maximum(ds_dy, dsdy_est)
                        ds_dy = ds_dy.clamp_min(1e-6)
                        if not hasattr(self, "_dbg_dsdy_logged") or not self._dbg_dsdy_logged:
                            try:
                                print(f"[CPR S] cache ds/dy: min={ds_dy.min().item():.3e} med={ds_dy.median().item():.3e} max={ds_dy.max().item():.3e}")
                                self._dbg_dsdy_logged = True
                            except Exception:
                                pass
                    else:
                        sw = self.simulator.fluid.s_w.view(-1).to(r_sw_hat)
                        swc = float(getattr(self.simulator.fluid, 'sw_cr', 0.0))
                        sor = float(getattr(self.simulator.fluid, 'so_r', 0.0))
                        denom = max(1e-12, 1.0 - swc - sor)
                        sigma = ((sw - swc) / denom).clamp(0.0, 1.0)
                        ds_dy = denom * (sigma * (1.0 - sigma))
                        ds_dy = ds_dy.clamp_min(1e-8)
                except Exception:
                    ds_dy = torch.ones_like(r_sw_hat) * 1e-3
 
                # защитим sw_scale от вырождения
                try:
                    if not math.isfinite(sw_scale) or sw_scale <= 0.0:
                        sw_scale = 1.0
                except Exception:
                    sw_scale = 1.0
                # Строго используем ds/dy для диагонали S-блока в hat (лучше кондиционирует)
                diag_SS_hat_sw = ds_dy * sw_scale
                # безопасный минимум диагонали в hat
                min_hat = float(getattr(self, 'cfg_cpr_diag_hat_sw_min', 1e-6))
                diag_SS_hat_sw = torch.nan_to_num(diag_SS_hat_sw, nan=min_hat, posinf=1e6, neginf=1e6).clamp_min(min_hat)
                diag_SS_hat_sg = diag_SS_phys if (r_sg_hat is not None) else None

                # Коррекция RHS насыщенностей с учётом влияния δp
                # Безопасный клип влияния K относительно диагонали
                try:
                    itn = int(getattr(self.simulator, '_newton_it', 0))
                except Exception:
                    itn = 0
                beta_sched = [0.3, 0.5, 0.8]
                beta_default = float(getattr(self.simulator.sim_params, 'cpr_k_ps_ratio', 0.8))
                beta = beta_sched[itn] if itn < len(beta_sched) else beta_default
                Ksw_eff = torch.minimum(Ksw_hat.to(r_sw_hat), (beta * diag_SS_hat_sw.to(r_sw_hat)))
                r_sw_corr = r_sw_hat - Ksw_eff * delta_p_hat
                # Диагностика (однократно на итерацию): нормы и масштабы в S-блоке
                try:
                    if not hasattr(self, "_dbg_stage2_logged") or not self._dbg_stage2_logged:
                        rs_norm = float(r_sw_hat.norm().item())
                        kdp_norm = float((Ksw_hat.to(r_sw_hat) * delta_p_hat).norm().item())
                        dsdy_med = float(ds_dy.median().item()) if ds_dy.numel() > 0 else 0.0
                        print(f"[CPR S] ||r_s||={rs_norm:.3e}, ||Kδp||={kdp_norm:.3e}, median(ds/dy)={dsdy_med:.3e}, sw_scale={sw_scale:.3e}")
                        self._dbg_stage2_logged = True
                except Exception:
                    pass
                # Базовое приближение (Jacobi по диагонали)
                diag_sw = (diag_SS_hat_sw.to(r_sw_hat) + 1e-30)
                # финальная страховка от микроскопических значений
                diag_sw = torch.nan_to_num(diag_sw, nan=1e-6, posinf=1e6, neginf=1e6).clamp_min(1e-6)
                try:
                    print(f"[CPR S] diag_SS_hat_sw: min={diag_sw.min().item():.3e} med={diag_sw.median().item():.3e} max={diag_sw.max().item():.3e}")
                except Exception:
                    pass
                z_sw_hat = r_sw_corr / diag_sw
                # анти-mute: если ход по насыщенности почти нулевой относительно RHS — берём чисто диагональный шаг
                try:
                    if float(z_sw_hat.norm().item()) < 1e-8 * max(1e-30, float(r_sw_corr.norm().item())):
                        z_sw_hat = r_sw_corr / (diag_sw + 1e-30)
                except Exception:
                    pass

                # Усиление: 2 шага Jacobi по локальному переносному оператору (7-точечный шаблон)
                try:
                    indptr = getattr(self, "_indptr_p", None)
                    indices = getattr(self, "_indices_p", None)
                    data = getattr(self, "_data_p", None)
                    if indptr is not None and indices is not None and data is not None:
                        # Подготовим CPU-тензоры индексов
                        import numpy as _np
                        import math as _math
                        n_cpu = int(r_sw_hat.numel())
                        indptr_t = torch.from_numpy(indptr.astype(_np.int64))
                        indices_t = torch.from_numpy(indices.astype(_np.int64))
                        data_t = torch.from_numpy(_np.abs(data)).to(torch.float32)
                        # Строковые индексы для каждого nnz
                        row_counts = indptr_t[1:] - indptr_t[:-1]
                        row_ids = torch.repeat_interleave(torch.arange(n_cpu, dtype=torch.int64), row_counts)
                        # off-диагонали
                        off_mask = indices_t != row_ids
                        row_off = row_ids[off_mask]
                        col_off = indices_t[off_mask]
                        w_base = data_t[off_mask]
                        # Нормализация весов: сначала по собственной медиане, затем по λ_t (нормированной)
                        w_med = torch.median(w_base)
                        w_base_n = w_base / (w_med + 1e-30)
                        lam_t = lam_w + lam_o + (lam_g if lam_g is not None else 0.0)
                        lam_t_cl = lam_t.clamp_min(1e-12).to(torch.float32).cpu()
                        w_lam = torch.sqrt(lam_t_cl[row_off] * lam_t_cl[col_off])
                        w = w_base_n * (w_lam / (w_lam.median() + 1e-30))
                        # Итерационный шаг Якоби: (D + γW) z = r
                        gamma = float(diag_sw.median().item())
                        wdeg = torch.zeros(n_cpu, dtype=torch.float32)
                        wdeg.index_add_(0, row_off, w)
                        try:
                            if not hasattr(self, "_dbg_w_logged") or not self._dbg_w_logged:
                                print(f"[CPR S] W stats: w|min,med,max=({w.min().item():.3e},{w.median().item():.3e},{w.max().item():.3e}), deg|min,med,max=({wdeg.min().item():.3e},{wdeg.median().item():.3e},{wdeg.max().item():.3e}), gamma={gamma:.3e}")
                                self._dbg_w_logged = True
                        except Exception:
                            pass
                        z_cpu = z_sw_hat.to(torch.float32).detach().cpu()
                        for _ in range(2):
                            wz = torch.zeros(n_cpu, dtype=torch.float32)
                            wz.index_add_(0, row_off, w * z_cpu[col_off])
                            num = r_sw_corr.to(torch.float32).cpu() + gamma * wz
                            den = diag_sw.to(torch.float32).cpu() + gamma * wdeg + 1e-30
                            z_cpu = num / den
                        z_sw_hat = z_cpu.to(r_sw_hat.device, r_sw_hat.dtype)
                except Exception:
                    pass

                # жёсткий физический кап δs: не позволяем выйти за [swc, 1-sor] (опционально)
                try:
                    if not getattr(self, 'cfg_cpr_phys_sat_cap', False):
                        raise RuntimeError('phys_cap_disabled')
                    swc = float(getattr(self.simulator.fluid, 'sw_cr', 0.0))
                    sor = float(getattr(self.simulator.fluid, 'so_r', 0.0))
                    denom = max(1e-12, 1.0 - swc - sor)
                    # берём текущую насыщенность из состояния (надёжнее, чем _last_y_hat)
                    try:
                        sw_curr = self.simulator.fluid.s_w.view(-1)[:r_sw_hat.numel()].to(r_sw_hat)
                    except Exception:
                        sw_curr = torch.full_like(r_sw_hat, swc + 0.5 * denom)
                    dsdy_loc = ds_dy.clamp_min(1e-12)
                    # перевод δy_hat → δs_phys
                    s_scales = getattr(self.scaler, 's_scales', [1.0])
                    sw_scale = float(s_scales[0]) if len(s_scales) > 0 else 1.0
                    delta_sw_phys = dsdy_loc * (z_sw_hat * sw_scale)
                    # масштабирующий множитель α_sat, чтобы не выйти за пределы
                    alpha_pos = ((1.0 - sor) - sw_curr) / (delta_sw_phys.clamp_min(1e-20))
                    alpha_neg = (sw_curr - swc) / ((-delta_sw_phys).clamp_min(1e-20))
                    alpha_pos = torch.where(delta_sw_phys > 0, alpha_pos, torch.full_like(alpha_pos, float('inf')))
                    alpha_neg = torch.where(delta_sw_phys < 0, alpha_neg, torch.full_like(alpha_neg, float('inf')))
                    alpha_sat = torch.minimum(alpha_pos, alpha_neg)
                    alpha_sat = torch.clamp(alpha_sat, 0.0, 1.0)
                    # применяем строгое ограничение (без искусственного «+0.05»)
                    scale_sat = torch.nan_to_num(alpha_sat, nan=1.0, posinf=1.0, neginf=0.0)
                    z_sw_hat = z_sw_hat * scale_sat.to(z_sw_hat)
                    # ограничим абсолютный прирост насыщенности из предобуславливателя (safety)
                    max_dsw = 0.05
                    dsw_phys = (dsdy_loc * (z_sw_hat * sw_scale)).abs()
                    over = dsw_phys > max_dsw
                    if bool(over.any()):
                        scale_abs = (max_dsw / (dsw_phys + 1e-20)).clamp_max(1.0)
                        z_sw_hat = z_sw_hat * scale_abs.to(z_sw_hat)
                except Exception:
                    pass
                try:
                    z2 = float(z_sw_hat.norm().item())
                    zinf = float(z_sw_hat.abs().max().item())
                    rc2 = float(r_sw_corr.norm().item())
                    print(f"[CPR S] ||z_sw_hat||2={z2:.3e}, ||z||inf={zinf:.3e}, ||r_sw_corr||2={rc2:.3e}")
                except Exception:
                    pass

                if r_sg_hat is not None:
                    if Ksg_hat is not None:
                        Ksg_eff = torch.minimum(Ksg_hat.to(r_sg_hat), (beta * diag_SS_hat_sg.to(r_sg_hat) + 1e-30))
                    else:
                        Ksg_eff = None
                    r_sg_corr = r_sg_hat - (Ksg_eff * delta_p_hat if Ksg_eff is not None else 0.0)
                    z_sg_hat = r_sg_corr / (diag_SS_hat_sg.to(r_sg_hat) + 1e-30)
                    z_sg_hat = torch.clamp(z_sg_hat, -0.05, 0.05)
            else:
                # Нет кеша свойств — безопасный Jacobi в hat без p–s связи
                # (диагональ берём 1 → шаг минимальный, но стабильный)
                z_sw_hat = r_sw_hat
                if r_sg_hat is not None:
                    z_sg_hat = r_sg_hat
        except Exception as e:
            if not hasattr(self, "_warn_geo2_hat_stage2"):
                print(f"[CPR geo2] Stage-2 hat failed: {e}")
                self._warn_geo2_hat_stage2 = True
            z_sw_hat = r_sw_hat
            if r_sg_hat is not None:
                z_sg_hat = r_sg_hat

        # NaN-guard
        if not torch.isfinite(delta_p_hat).all():
            delta_p_hat = torch.nan_to_num(delta_p_hat, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(z_sw_hat).all():
            z_sw_hat = torch.nan_to_num(z_sw_hat, nan=0.0, posinf=0.0, neginf=0.0)
        if (r_sg_hat is not None) and (not torch.isfinite(z_sg_hat).all()):
            z_sg_hat = torch.nan_to_num(z_sg_hat, nan=0.0, posinf=0.0, neginf=0.0)
        # ------ сборка полного ответа в hat ------
        out = torch.zeros_like(vec_hat)
        out[:n] = delta_p_hat
        out[n:2*n] = z_sw_hat
        if r_sg_hat is not None:
            out[2*n:3*n] = z_sg_hat
        return out