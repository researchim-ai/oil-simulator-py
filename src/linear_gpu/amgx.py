"""Wrapper вокруг NVIDIA AmgX, использующий пакет `pyamgx`.

Требования:
    pip install pyamgx
    и корректно настроенный AMGX (см. README).

Пример использования:
    from linear_gpu.amgx import solve_amgx
    x = solve_amgx(A_csr, b, tol=1e-8, max_iter=1000)
"""
from __future__ import annotations
import torch
import numpy as np

try:
    import pyamgx
except ImportError as e:  # pragma: no cover
    pyamgx = None

try:
    import amgx_ext
except ImportError:
    amgx_ext = None

__all__ = ["solve_amgx", "amgx_available"]


def amgx_available() -> bool:
    return pyamgx is not None


def _torch_csr_to_numpy(A: torch.Tensor):
    if not A.is_sparse_csr:
        raise ValueError("Матрица должна быть в формате torch.sparse_csr_tensor")
    row_ptr = A.crow_indices().cpu().numpy().astype(np.int32)
    col_ind = A.col_indices().cpu().numpy().astype(np.int32)
    data = A.values().cpu().numpy().astype(np.float64)
    return row_ptr, col_ind, data


def solve_amgx(A: torch.Tensor, b: torch.Tensor, tol: float = 1e-8, max_iter: int = 1000) -> torch.Tensor:
    """Решает A x = b используя NVIDIA AmgX (classic AMG + PCG).

    Parameters
    ----------
    A : torch.sparse_csr_tensor  (dtype=float64)
    b : torch.Tensor (1-D)
    tol, max_iter : параметры AMG-решателя
    """
    if pyamgx is None:
        raise RuntimeError("pyamgx не установлен. Установите пакет или выберите другой backend.")

    if A.dtype != torch.float64:
        A = A.double()
    if b.dtype != torch.float64:
        b = b.double()

    row_ptr, col_ind, data = _torch_csr_to_numpy(A)

    pyamgx.initialize()
    cfg_dict = {
        "config_version": 2,
        "solver": {
            "type": "PCG",
            "max_iters": max_iter,
            "monitor_res": True,
            "convergence": "RELATIVE_RESIDUAL",
            "tolerance": tol,
            "preconditioner": {
                "type": "AMG",
                "algorithm": "CLASSICAL",
                "max_iters": 1,
                "presweeps": 1,
                "postsweeps": 1,
                "cycle_type": "V",
                "print_grid": 0
            }
        }
    }
    cfg = pyamgx.Config().create_from_dict(cfg_dict)
    resources = pyamgx.Resources().create_simple(cfg)
    matrix = pyamgx.Matrix().create(resources, A.shape[0], A.shape[0],
                                     np.count_nonzero(data))
    matrix.upload_csr(row_ptr, col_ind, data)
    rhs = pyamgx.Vector().create(resources)
    sol = pyamgx.Vector().create(resources)
    rhs.upload(b.cpu().numpy())
    sol.upload(np.zeros_like(b.cpu().numpy()))
    solver = pyamgx.Solver().create(resources, cfg)
    solver.setup(matrix)
    solver.solve(rhs, sol)
    x_np = sol.download()

    # освобождение
    solver.destroy()
    rhs.destroy()
    sol.destroy()
    matrix.destroy()
    resources.destroy()
    cfg.destroy()
    pyamgx.finalize()

    return torch.from_numpy(x_np).to(b.device)


def solve_amgx_torch(A: torch.Tensor, b: torch.Tensor, tol: float = 1e-8, max_iter: int = 1000):
    """Использует собранное C++-расширение amgx_ext, либо fallback на torch.solve."""
    indptr = A.crow_indices()
    indices = A.col_indices()
    values = A.values()
    if amgx_ext is not None and amgx_available():
        return amgx_ext.solve(indptr, indices, values, b, tol, max_iter)
    else:
        # fallback: dense solve
        return torch.linalg.solve(A.to_dense(), b) 