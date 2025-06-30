import torch
from .csr import dense_to_csr
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sp
    import cupyx.scipy.sparse.linalg as cpx_linalg
except ImportError:  # pragma: no cover
    cp = None
    cpx_sp = None
    cpx_linalg = None


def jacobi_precond(A, omega: float = 0.8):
    """Возвращает функцию-предобуславливатель Jacobi: z = M^{-1} r.

    M ≈ (1/ω)·diag(A), что эквивалентно damped Jacobi при ω∈(0,1].
    Для ω=1 даёт классический Jacobi.

    Работает как для dense, так и для sparse CSR.
    """
    if callable(A):
        raise ValueError("Jacobi precond требует матрицу, а не callable")

    if A.is_sparse_csr:
        # Элементы CSR сгруппированы по строкам; получаем диагональ
        rowptr = A.crow_indices()
        colidx = A.col_indices()
        values = A.values()
        diag = torch.zeros(A.size(0), device=A.device, dtype=A.dtype)
        for i in range(A.size(0)):
            start, end = rowptr[i].item(), rowptr[i+1].item()
            row_cols = colidx[start:end]
            row_vals = values[start:end]
            mask = row_cols == i
            if mask.any():
                diag[i] = row_vals[mask][0]
            else:
                diag[i] = 1.0  # если диагональный элемент отсутствует
    else:
        diag = torch.diagonal(A)

    scale = omega  # умножаем r, эквивалентно ω-Jacobi

    def apply(r):
        if r.dtype != diag.dtype:
            r = r.to(diag.dtype)
        return scale * r / (diag + 1e-12)

    return apply 

def fsai_precond(A: torch.Tensor, k: int = 1):
    """Строит простой FSAI предобуславливатель (без фильтрации)
    Для каждой строки i берём её ненулевые столбцы S_i (диаг+соседи) и
    решаем A_{S_i,S_i} m_i = e_i, где e_i индикатор в локальной нумерации.
    Возвращает функцию r ↦ M r, где M хранится как sparse CSR на том же устройстве,
    что и A (применяется на GPU через torch.sparse.mm).
    В текущей упрощённой реализации строим на CPU, т.к. размер N=2500.
    """
    if callable(A):
        raise ValueError("fsai_precond требует матрицу, а не callable")

    # Рабочую сборку делаем на CPU для упрощения
    A_cpu = A.to('cpu') if A.device.type != 'cpu' else A
    n = A_cpu.size(0)
    denseA = A_cpu.to_dense()

    rows = []
    cols = []
    vals = []
    for i in range(n):
        # ненулевые столбцы строки i
        nz_cols = A_cpu.col_indices()[A_cpu.crow_indices()[i]:A_cpu.crow_indices()[i+1]]
        S = nz_cols.tolist()
        if i not in S:
            S.append(i)
        S = sorted(set(S))
        # локальная подматрица
        S_idx = torch.tensor(S, dtype=torch.long)
        A_sub = denseA[S_idx][:, S_idx]
        # правый вектор e_i (в локальной нумерации)
        e = torch.zeros(len(S), dtype=A_sub.dtype)
        local_pos = S.index(i)
        e[local_pos] = 1.0
        # solve
        try:
            m = torch.linalg.solve(A_sub, e)
        except RuntimeError:
            m = torch.linalg.lstsq(A_sub, e).solution
        for j, col in enumerate(S):
            if m[j].abs() > 1e-12:
                rows.append(i)
                cols.append(col)
                vals.append(m[j])

    row_idx = torch.tensor(rows, dtype=torch.int64)
    col_idx = torch.tensor(cols, dtype=torch.int64)
    val_t   = torch.tensor(vals, dtype=A_cpu.dtype)
    M_coo = torch.sparse_coo_tensor(torch.stack([row_idx, col_idx]), val_t, (n, n))
    M_csr = M_coo.to_sparse_csr()
    M_csr = M_csr.to(A.device)

    def apply(r: torch.Tensor):
        if r.device != M_csr.device:
            r = r.to(M_csr.device)
        if r.dtype != M_csr.dtype:
            r = r.to(M_csr.dtype)
        return torch.sparse.mm(M_csr, r.unsqueeze(1)).squeeze(1)

    return apply 

def _torch_csr_to_scipy(A: torch.Tensor):
    """Convert torch sparse CSR to SciPy CSR (CPU)."""
    rowptr = A.crow_indices().cpu().numpy()
    colidx = A.col_indices().cpu().numpy()
    data = A.values().cpu().numpy()
    return sp.csr_matrix((data, colidx, rowptr), shape=A.shape)


def _torch_csr_to_cupy(A: torch.Tensor):
    """Convert torch sparse CSR on CUDA to CuPy CSR."""
    assert A.device.type == 'cuda'
    if cp is None:
        raise RuntimeError("cupy не установлен, но требуется для GPU ILU")
    rowptr = cp.asarray(A.crow_indices())
    colidx = cp.asarray(A.col_indices())
    data = cp.asarray(A.values())
    return cpx_sp.csr_matrix((data, colidx, rowptr), shape=A.shape)


def ilu_precond(A: torch.Tensor, drop_tol: float = 1e-4, fill_factor: int = 10):
    """ILU0/ILUT предобуславливатель.

    • Если A на CPU → SciPy spilu.
    • Если A на CUDA → CuPy spilu (cupyx).
    Возвращает функцию z = M^{-1} r.
    """
    if callable(A):
        raise ValueError("ILU требует матрицу, не callable")

    if A.device.type == 'cpu':
        A_csr = _torch_csr_to_scipy(A)
        ilu = spla.spilu(A_csr, drop_tol=drop_tol, fill_factor=fill_factor)

        def apply(r: torch.Tensor):
            r_np = r.detach().cpu().numpy()
            z_np = ilu.solve(r_np)
            return torch.from_numpy(z_np).to(r.device)

        return apply

    elif A.device.type == 'cuda':
        if cp is None:
            raise RuntimeError("cupy не доступен, установите cupy для GPU ILU")
        A_cp = _torch_csr_to_cupy(A)
        ilu = cpx_linalg.spilu(A_cp, drop_tol=drop_tol, fill_factor=fill_factor)

        def apply(r: torch.Tensor):
            r_cp = cp.asarray(r.detach())
            z_cp = ilu.solve(r_cp)
            return torch.utils.dlpack.from_dlpack(z_cp.toDlpack()).to(r.device)

        return apply

    else:
        raise RuntimeError(f"Неизвестный тип устройства: {A.device.type}") 