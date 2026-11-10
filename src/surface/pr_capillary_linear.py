"""Checked sparse PyTorch Newton corrections on the physical capillary graph.

Only declared graph-neighbor blocks are assembled; large problems never
fall back to a dense Jacobian. The existing device GMRES and block Jacobi
are reused and the true, unpreconditioned linear residual is mandatory.
"""

import torch

from .liquid import HydraulicConvergenceError
from .torch_linear import BlockJacobi, ColoredJacobian, TorchCOO, gmres_device


class CapillaryLinearSystem:
    def __init__(self, blocks, edges, device, *, radius, mode="auto", dense_limit=192):
        if mode not in ("auto", "dense", "sparse"):
            raise ValueError("Capillary linear solver must be auto, dense or sparse")
        self.blocks = tuple(tuple(b) for b in blocks)
        self.size = sum(map(len, self.blocks))
        self.sparse = mode == "sparse" or (mode == "auto" and self.size > dense_limit)
        self.coloring = self.matrix = self.preconditioner = None
        self.solves = 0
        self.krylov_restarts = 0
        self.maximum_relative_residual = 0.0
        if not self.sparse:
            return
        neighbors = [{i} for i in range(len(blocks))]
        for i, j in edges:
            neighbors[i].add(j)
            neighbors[j].add(i)
        pattern = []
        for i in range(len(blocks)):
            support = {i}
            for _ in range(radius):
                support = set().union(*(neighbors[j] for j in support))
            pattern.extend((r, c) for j in sorted(support) for r in blocks[i] for c in blocks[j])
        self.rows = [r for r, _ in pattern]
        self.cols = [c for _, c in pattern]
        self.coloring = ColoredJacobian(self.size, self.rows, self.cols, device)
        self.preconditioner = BlockJacobi(self.rows, self.cols, self.blocks, self.size, device)

    def solve(self, function, point, residual):
        self.solves += 1
        if not self.sparse:
            jacobian = torch.func.jacfwd(function)(point)
            if not bool(torch.isfinite(jacobian).all().cpu()):
                raise HydraulicConvergenceError("Nonfinite capillary Jacobian")
            correction, info = torch.linalg.solve_ex(jacobian, -residual, check_errors=False)
            valid = info == 0
            defect = jacobian @ correction + residual
        else:
            # Bound transient forward-AD material tensors; all seeds, values,
            # factors and Krylov vectors remain on the requested device.
            compressed = torch.cat(
                [
                    torch.vmap(lambda seed: torch.func.jvp(function, (point,), (seed,))[1])(batch)
                    for batch in self.coloring.seeds.split(8)
                ]
            )
            values = compressed[self.coloring.color[self.coloring.c], self.coloring.r]
            if not bool(torch.isfinite(values).all().cpu()):
                raise HydraulicConvergenceError("Nonfinite sparse capillary Jacobian")
            if self.matrix is None:
                self.matrix = TorchCOO(self.rows, self.cols, values, self.size, point.device)
            else:
                self.matrix.values = values.index_select(-1, self.matrix.order)
            self.preconditioner.update(values)
            correction = torch.zeros_like(residual)
            target = 1e-9 * residual.abs().max() + 1e-14
            # Host status reads occur at restart boundaries, not per Arnoldi vector.
            # Small acoustic steps should not pay four full 32-vector cycles.
            for _ in range(6):
                correction = correction + gmres_device(
                    self.matrix.mv,
                    -residual - self.matrix.mv(correction),
                    self.matrix.diagonal(),
                    restart=12,
                    cycles=1,
                    preconditioner=self.preconditioner,
                )
                self.krylov_restarts += 1
                # Check by an independent JVP, not just by the assembled
                # matrix: an omitted graph dependency must fail explicitly.
                defect = torch.func.jvp(function, (point,), (correction,))[1] + residual
                assembled = self.matrix.mv(correction) + residual
                if bool(
                    ((defect - assembled).abs().max() > 2e-12 * residual.abs().max() + 1e-14).cpu()
                ):
                    raise RuntimeError("Capillary sparse pattern failed true-residual check")
                if bool((torch.isfinite(defect).all() & (defect.abs().max() <= target)).cpu()):
                    break
            valid = torch.isfinite(values).all()
        scale = residual.abs().max()
        relative = defect.abs().max() / scale.clamp_min(1e-30)
        self.maximum_relative_residual = max(self.maximum_relative_residual, float(relative.cpu()))
        if not bool(
            (
                valid
                & torch.isfinite(correction).all()
                & torch.isfinite(defect).all()
                & (defect.abs().max() <= 1e-9 * scale + 1e-14)
            ).cpu()
        ):
            raise HydraulicConvergenceError(
                "Capillary linear correction failed true-residual check"
            )
        return correction

    def report(self):
        return {
            "backend": "torch_colored_sparse_gmres" if self.sparse else "torch_dense",
            "unknowns": self.size,
            "stored_jacobian_entries": len(self.rows) if self.sparse else self.size**2,
            "colors": int(self.coloring.seeds.shape[0]) if self.sparse else self.size,
            "linear_solves": self.solves,
            "krylov_restarts": self.krylov_restarts,
            "maximum_relative_linear_residual": self.maximum_relative_residual,
            "large_dense_fallback": False,
        }
