import numpy as np

try:
    from linear_gpu.petsc_boomeramg import solve_boomeramg
except ImportError:
    solve_boomeramg = None

class BoomerSolver:
    """Обёртка над solve_boomeramg, хранящая CSR-матрицу"""
    def __init__(self, indptr, indices, data):
        self.indptr = indptr
        self.indices = indices
        self.data = data

    def solve(self, rhs, tol=1e-8, max_iter=400):
        if solve_boomeramg is None:
            raise RuntimeError("solve_boomeramg not available. Install linear_gpu.petsc_boomeramg")
        sol, its, res = solve_boomeramg(self.indptr, self.indices, self.data,
                                        rhs, tol=tol, max_iter=max_iter)
        return sol

try:
    import pyamgx
    _amgx_available = True
except ImportError:
    _amgx_available = False

if _amgx_available:
    class AmgXSolver:
        def __init__(self, indptr, indices, data, cfg="CLASSICAL"):
            from linear_gpu import amgx as _amgx
            self.solver = _amgx.AmgXSolver(indptr, indices, data, cfg=cfg)
        def solve(self, rhs, tol=1e-6, max_iter=200):
            return self.solver.solve(rhs, tol=tol, max_iter=max_iter)
else:
    AmgXSolver = None 