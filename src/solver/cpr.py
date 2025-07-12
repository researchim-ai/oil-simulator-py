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
        
        # Выбираем масштаб с учетом сжимаемости
        if typical_T < 1e-12:
            matrix_scale = 1e12 * compressibility_factor  # Очень маленькие коэффициенты
        elif typical_T < 1e-6:
            matrix_scale = 1e6 * compressibility_factor   # Маленькие коэффициенты
        elif typical_T > 1e6:
            matrix_scale = 1e-6 / compressibility_factor  # Большие коэффициенты
        else:
            matrix_scale = 1.0 * compressibility_factor   # Нормальные коэффициенты
        
        print(f"🎯 CPR: Типичная transmissibility: {typical_T:.3e}")
        print(f"🎯 CPR: Максимальная сжимаемость: {max_compress:.3e}")
        print(f"🎯 CPR: Фактор сжимаемости: {compressibility_factor:.3e}")
        print(f"🎯 CPR: Масштаб матрицы: {matrix_scale:.3e}")
        
        # Сохраняем масштаб для восстановления решения
        self.matrix_scale = matrix_scale
        self.compressibility_factor = compressibility_factor

        # --- предварительное выделение памяти под CSR ---
        N = nx * ny * nz
        nnz_est = 7 * N
        indptr = np.zeros(N + 1, dtype=np.int64)
        indices = np.empty(nnz_est, dtype=np.int32)
        data = np.empty(nnz_est, dtype=np.float64)

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
                        t = Tx[i-1, j, k] * lam * self.matrix_scale
                        indices[pos] = center - 1
                        data[pos] = -t
                        pos += 1
                        diag += t
                    # X+
                    if i < nx - 1:
                        t = Tx[i, j, k] * lam * self.matrix_scale
                        indices[pos] = center + 1
                        data[pos] = -t
                        pos += 1
                        diag += t
                    # Y-
                    if j > 0:
                        t = Ty[i, j-1, k] * lam * self.matrix_scale
                        indices[pos] = center - nx
                        data[pos] = -t
                        pos += 1
                        diag += t
                    # Y+
                    if j < ny - 1:
                        t = Ty[i, j, k] * lam * self.matrix_scale
                        indices[pos] = center + nx
                        data[pos] = -t
                        pos += 1
                        diag += t
                    # Z-/Z+
                    if nz > 1:
                        if k > 0:
                            t = Tz[i, j, k-1] * lam * self.matrix_scale
                            indices[pos] = center - nx * ny
                            data[pos] = -t
                            pos += 1
                            diag += t
                        if k < nz - 1:
                            t = Tz[i, j, k] * lam * self.matrix_scale
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
                    data[pos] = diag + adaptive_shift * self.matrix_scale
                    pos += 1
                    idx += 1

        indptr[N] = pos
        
        # 🎯 Логирование масштабированной матрицы
        print(f"🎯 CPR: Масштабированная матрица - мин: {np.min(data[:pos]):.3e}, макс: {np.max(data[:pos]):.3e}")
        
        return indptr[:N+1], indices[:pos], data[:pos]

    def apply(self, vec: torch.Tensor) -> torch.Tensor:
        """🎯 ROBUST CPR предобуславливание с автоматическим масштабированием
        
        CPR применяется как:
        1. Решаем уравнение давления через AMG: A_p * delta_p = rhs_p  
        2. Насыщенность обрабатываем через простое Jacobi масштабирование
        3. Автоматическое масштабирование для робастности
        """
        n = vec.shape[0] // 2
        
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
                delta_p_scaled = self.solver.solve(rhs_scaled, tol=1e-8, max_iter=200)
                
                # Проверяем результат на NaN/Inf
                if np.any(np.isnan(delta_p_scaled)) or np.any(np.isinf(delta_p_scaled)):
                    print("    CPR: AMG вернул NaN/Inf, переключаемся на Jacobi")
                    self.failed_amg = True
                    delta_p_scaled = self.diag_inv * rhs_scaled
                else:
                    delta_p_norm = np.linalg.norm(delta_p_scaled)
                    print(f"    CPR: AMG решение успешно, ||delta_p||={delta_p_norm:.3e}")
                    
                    # 🎯 ROBUST проверка: решение должно быть разумного размера
                    if delta_p_norm > 1e8:  # слишком большое решение
                        print(f"    CPR: AMG дал огромное решение, переключаемся на Jacobi")
                        self.failed_amg = True
                        delta_p_scaled = self.diag_inv * rhs_scaled
                
            except Exception as e:
                print(f"    CPR: Ошибка в AMG решателе: {e}, переключаемся на Jacobi")
                self.failed_amg = True
                delta_p_scaled = self.diag_inv * rhs_scaled

        # 🎯 ПРАВИЛЬНОЕ ВОССТАНОВЛЕНИЕ МАСШТАБА
        # Нужно восстановить только rhs_scale, matrix_scale уже учтен в матрице
        delta_p = delta_p_scaled * rhs_scale
        
        # 🔧 ДОПОЛНИТЕЛЬНЫЙ DEBUG
        if hasattr(self, 'matrix_scale'):
            print(f"    CPR: Восстановление масштаба: rhs_scale={rhs_scale:.3e}, matrix_scale={self.matrix_scale:.3e}")
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
        
        # Ограничиваем решение разумным кратным от входного вектора
        if vec_norm > 1e-15:
            max_ratio = 1000.0  # Решение не должно быть более чем в 1000 раз больше входа
            if pressure_norm > max_ratio * vec_norm:
                scale_factor = (max_ratio * vec_norm) / pressure_norm
                pressure_result = pressure_result * scale_factor
                print(f"    CPR: Ограничили решение фактором {scale_factor:.3e}")
        
        out[:n] = pressure_result
        
        # Насыщенность: простое масштабирование
        out[n:] = self.omega * vec[n:]
        
        return out 