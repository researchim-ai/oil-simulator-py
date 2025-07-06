# Rewritten lightweight GMRES wrapper with iteration counting
# NOTE: This replaces the previous verbose implementation that has been
# accidentally corrupted.  The new version delegates the actual GMRES
# work to `solver.krylov.gmres` (which is already in the project) and
# simply counts the number of Krylov iterations performed.

from __future__ import annotations

from typing import Callable, Tuple
import torch

# Import the reference implementation (simple but robust)
from solver.krylov import gmres as _gmres_base


def gmres(
    A: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    M: Callable[[torch.Tensor], torch.Tensor] | None = None,
    *,
    tol: float = 1e-8,
    restart: int = 50,
    max_iter: int = 400,
) -> Tuple[torch.Tensor, int, int]:
    """GMRES with iteration counting.

    This is a thin wrapper around :pyfunc:`solver.krylov.gmres` that returns the
    total number of Arnoldi iterations actually performed in addition to the
    usual ``(x, info)`` pair.

    Returns
    -------
    x : torch.Tensor
        Solution vector (same shape as *b*).
    info : int
        ``0`` if converged, non-zero otherwise (same convention as the base
        implementation).
    iters : int
        Total number of Krylov iterations executed (Arnoldi steps).
    """
    iter_counter = {"k": 0}

    # Wrap the matvec to count how many times it is called.  Each call
    # corresponds to one Arnoldi vector generation (i.e. one Krylov iteration).
    def _A_count(v: torch.Tensor) -> torch.Tensor:  # noqa: D401
        iter_counter["k"] += 1
        return A(v)

    x, info = _gmres_base(
        _A_count,
        b,
        M=M,
        restart=restart,
        tol=tol,
        maxiter=max_iter,
    )

    return x, info, iter_counter["k"] 