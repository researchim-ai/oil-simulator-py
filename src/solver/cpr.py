import torch, numpy as np
from .amg import BoomerSolver, AmgXSolver
from typing import Optional

class CPRPreconditioner:
    def __init__(self, reservoir, fluid, backend="amgx", omega=0.8):
        self.backend = backend
        self.omega = omega
        self.failed_amg = False  # Флаг провала AMG
        
        indptr, ind, data = self._assemble_pressure_csr(reservoir, fluid)
        
        # Сохраняем диагональ для Jacobi fallback
        self.diag_inv = self._extract_diagonal_inverse(indptr, ind, data)
        
        if backend == "amgx" and AmgXSolver is not None:
            try:
                self.solver = AmgXSolver(indptr, ind, data)
            except Exception as e:
                print(f"Ошибка инициализации AmgX: {e}")
                self.solver = None
                self.failed_amg = True
        elif backend in ("hypre", "boomer"):
            try:
                self.solver = BoomerSolver(indptr, ind, data)
            except Exception as e:
                print(f"Ошибка инициализации BoomerAMG: {e}")
                self.solver = None
                self.failed_amg = True
        else:
            # 'jacobi' или 'none' – не используем AMG
            self.solver = None

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

                    # Диагональный элемент
                    indices[pos] = center
                    data[pos] = diag + 1e-12  # стабилизационный сдвиг
                    pos += 1
                    idx += 1

        indptr[N] = pos
        return indptr[:N+1], indices[:pos], data[:pos]

    def apply(self, vec: torch.Tensor) -> torch.Tensor:
        n = vec.shape[0]//2
        rhs_p = vec[:n].cpu().numpy()

        if self.solver is None or self.failed_amg:
            # Fallback к диагональному предобуславливателю
            corr_p = self.diag_inv * rhs_p
        else:
            try:
                corr_p = self.solver.solve(rhs_p)
                
                # Проверяем результат на NaN/Inf
                if np.any(np.isnan(corr_p)) or np.any(np.isinf(corr_p)):
                    print("CPR: AMG вернул NaN/Inf, переключаемся на Jacobi")
                    self.failed_amg = True
                    corr_p = self.diag_inv * rhs_p
                    
            except Exception as e:
                print(f"CPR: Ошибка в AMG решателе: {e}, переключаемся на Jacobi")
                self.failed_amg = True
                corr_p = self.diag_inv * rhs_p

        out = torch.zeros_like(vec)
        out[:n] = torch.from_numpy(corr_p).to(vec.device)
        out[n:] = self.omega * vec[n:]
        return out 