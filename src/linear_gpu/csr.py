import torch

def dense_to_csr(mat: torch.Tensor) -> torch.Tensor:
    """Быстро переводит dense-тензор в `torch.sparse_csr_tensor`.
    Для работы на GPU требуется PyTorch ≥1.12.
    Если вход уже разреженный – возвращает как есть.
    """
    if mat.is_sparse_csr:
        return mat
    if mat.is_sparse:
        return mat.to_sparse_csr()
    # PyTorch позволяет to_sparse_csr начиная с 1.12
    try:
        return mat.to_sparse_csr()
    except RuntimeError:
        # На старых версиях fallback: через COO → CSR
        coo = mat.to_sparse()
        return coo.to_sparse_csr() 

# -----------------------------------------------------------------------------
# build_7pt_csr: формирует CSR-матрицу 7-точечного оператора div(T * grad) на
# прямоугольной сетке (nz, ny, nx).  Коэффициенты передаются в виде transmissibility
# тензоров (Tx, Ty, Tz) ИЛИ исходных проницаемостей kx,ky,kz + размеров ячейки.
# Здесь реализуем вариант с transmissibility Tx,Ty,Tz, чтобы вызывать после их
# рассчёта в GeoSolver.  Формат индексации: x – самая «быстрая» (nx), затем y.
# -----------------------------------------------------------------------------

import numpy as np


def build_7pt_csr(Tx: np.ndarray, Ty: np.ndarray, Tz: np.ndarray | None,
                  nx: int, ny: int, nz: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Собирает CSR для SPD-матрицы давления.

    Args:
        Tx: (nz, ny, nx-1) transmissibility между ячейками (i,j,k) и (i+1,j,k)
        Ty: (nz, ny-1, nx) между (i,j,k) и (i,j+1,k)
        Tz: (nz-1, ny, nx) между (i,j,k) и (i,j,k+1)  или None, если 2-D.
        nx,ny,nz: размеры сетки.
    Returns:
        indptr, indices, data – torch.int64, torch.int32, torch.float64 tensors.
    """
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
                    t = Tx[k, j, i-1]
                    indices[pos] = center - 1
                    data[pos] = -t
                    pos += 1
                    diag += t
                # X+
                if i < nx - 1:
                    t = Tx[k, j, i]
                    indices[pos] = center + 1
                    data[pos] = -t
                    pos += 1
                    diag += t

                # Y-
                if j > 0:
                    t = Ty[k, j-1, i]
                    indices[pos] = center - nx
                    data[pos] = -t
                    pos += 1
                    diag += t
                # Y+
                if j < ny - 1:
                    t = Ty[k, j, i]
                    indices[pos] = center + nx
                    data[pos] = -t
                    pos += 1
                    diag += t

                # Z faces
                if nz > 1:
                    if k > 0:
                        t = Tz[k-1, j, i]
                        indices[pos] = center - nx * ny
                        data[pos] = -t
                        pos += 1
                        diag += t
                    if k < nz - 1:
                        t = Tz[k, j, i]
                        indices[pos] = center + nx * ny
                        data[pos] = -t
                        pos += 1
                        diag += t

                # Центр
                indices[pos] = center
                data[pos] = diag
                pos += 1
                idx += 1

    indptr[N] = pos
    # Обрезаем массивы
    indices = indices[:pos]
    data = data[:pos]

    # Конвертируем в torch тензоры
    indptr_t = torch.from_numpy(indptr)
    indices_t = torch.from_numpy(indices.astype(np.int64))
    data_t = torch.from_numpy(data)
    return indptr_t, indices_t, data_t 