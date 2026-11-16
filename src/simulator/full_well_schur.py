"""Full dense well-block elimination for arbitrary reservoir block size."""

import torch

try:
    from ..surface.liquid import HydraulicConvergenceError
except ImportError:
    from surface.liquid import HydraulicConvergenceError


class FullWellSchur:
    # Bound temporary dense RHS work without changing the well block or
    # assembling a dense reservoir matrix. Small systems retain one solve.
    maximum_rhs_entries = 1_048_576

    def factor_well_block(self, sys):
        """Queue the factorization/status on device; do not read status here."""
        operator = sys.get("_surface_operator")
        if operator is not None:
            sys["_surface_factor_bad"] = operator.factor_bad
            return
        if "_surface_lu" in sys:
            return
        D = sys["D_full"]
        scale = D.abs().amax(dim=1).clamp_min(1e-300).reciprocal()
        lu, piv, info = torch.linalg.lu_factor_ex(D * scale[:, None])
        sys["_surface_lu"], sys["_surface_piv"], sys["_surface_scale"] = lu, piv, scale
        sys["_surface_factor_bad"] = (info != 0).any() | ~torch.isfinite(lu).all()

    def _prepare_full(self, sys):
        if "_surface_local" in sys:
            return
        operator = sys.get("_surface_operator")
        if operator is not None:
            operator.prepare(sys.get("_surface_factor_validated"))
            sys["_surface_factor_validated"] = True
            sys["_surface_local"] = operator.local
            return
        self.factor_well_block(sys)
        valid = sys.get("_surface_factor_validated")
        if valid is None:
            valid = not bool(sys["_surface_factor_bad"])
            sys["_surface_factor_validated"] = valid
        if not valid:
            raise HydraulicConvergenceError("Singular coupled surface/well block")
        D = sys["D_full"]
        mapping = getattr(self, "_surface_cell_mapping", None)
        if mapping is None or mapping[0] is not self.conn_cell:
            cells, inverse = torch.unique(self.conn_cell, sorted=True, return_inverse=True)
            self._surface_cell_mapping = (self.conn_cell, cells, inverse)
        else:
            _, cells, inverse = mapping
        nb = sys["B_full"].shape[-1]
        budget = self.maximum_rhs_entries
        if type(budget) is not int or budget < self.n_w * nb:
            raise ValueError("Schur RHS budget must fit one cell block")
        sys["_surface_cells"], sys["_surface_inverse"], sys["_surface_Bcell"] = (
            cells,
            inverse,
            None,
        )
        if self.n_w * cells.numel() * nb <= budget:
            Bcell = D.new_zeros((self.n_w, cells.numel(), nb)).index_add_(1, inverse, sys["B_full"])
            solved = self._solve_D(sys, Bcell.reshape(self.n_w, -1)).reshape_as(Bcell)
            sys["_surface_Bcell"] = Bcell
            sys["_surface_local"] = solved[self.conn_well, inverse]
            return
        # Sort only immutable topology. One host prefix packet is cached;
        # no state-dependent GPU index inventory is read per Krylov matvec.
        layout = getattr(self, "_surface_chunk_layout", None)
        if layout is None or layout[0] is not self.conn_cell:
            order = torch.argsort(inverse, stable=True)
            counts = torch.bincount(inverse, minlength=cells.numel())
            prefix = [0, *counts.cumsum(0).detach().cpu().tolist()]
            layout = self._surface_chunk_layout = (self.conn_cell, order, prefix)
        _, order, prefix = layout
        chunk = budget // (self.n_w * nb)
        local = D.new_empty((self.conn_cell.numel(), nb))
        for start in range(0, cells.numel(), chunk):
            end = min(start + chunk, cells.numel())
            B = D.new_zeros((self.n_w, end - start, nb))
            # Several perforations may belong to one cell. Aggregate all of
            # them before selecting the local Schur term for each connection.
            for offset in range(prefix[start], prefix[end], chunk):
                selected = order[offset : min(offset + chunk, prefix[end])]
                B.index_add_(1, inverse[selected] - start, sys["B_full"][:, selected])
            solved = self._solve_D(sys, B.reshape(self.n_w, -1)).reshape_as(B)
            selected = order[prefix[start] : prefix[end]]
            local[selected] = solved[self.conn_well[selected], inverse[selected] - start]
        sys["_surface_local"] = local

    def _solve_D(self, sys, rhs):
        if "_surface_operator" in sys:
            return sys["_surface_operator"].solve(rhs)
        vector = rhs.ndim == 1
        if vector:
            rhs = rhs[:, None]
        answer = torch.linalg.lu_solve(
            sys["_surface_lu"], sys["_surface_piv"], rhs * sys["_surface_scale"][:, None]
        )
        return answer[:, 0] if vector else answer

    def add_local_schur(self, sys, J_vals):
        self._prepare_full(sys)
        local = sys["Cv"][:, :, None] * sys["_surface_local"][:, None, :]
        J_vals.index_add_(0, sys["diag_pos"], -local)

    def _Bx(self, sys, dx):
        self._prepare_full(sys)
        if "_surface_operator" in sys:
            return sys["_surface_operator"].Bx(dx)
        if sys["_surface_Bcell"] is not None:
            return torch.einsum("wub,ub->w", sys["_surface_Bcell"], dx[sys["_surface_cells"]])
        B = sys["B_full"]
        chunk = self.maximum_rhs_entries // (self.n_w * B.shape[-1])
        answer = B.new_zeros(self.n_w)
        for start in range(0, self.conn_cell.numel(), chunk):
            end = start + chunk
            answer = answer + torch.einsum(
                "wub,ub->w", B[:, start:end], dx[self.conn_cell[start:end]]
            )
        return answer

    def schur_correction(self, sys, x_blk, local_included=True):
        coeff = self._solve_D(sys, self._Bx(sys, x_blk))[self.conn_well]
        if local_included:
            coeff = coeff - (sys["_surface_local"] * x_blk[self.conn_cell]).sum(1)
        out = torch.zeros_like(x_blk)
        out.index_add_(0, self.conn_cell, -sys["Cv"] * coeff[:, None])
        return out

    def schur_rhs(self, sys, n_cells):
        self._prepare_full(sys)
        coeff = self._solve_D(sys, sys["r_w"])[self.conn_well]
        out = sys["Cv"].new_zeros((n_cells, sys["Cv"].shape[-1]))
        out.index_add_(0, self.conn_cell, sys["Cv"] * coeff[:, None])
        return out

    def recover_dpw(self, sys, dx_blk):
        return self._solve_D(sys, -sys["r_w"] - self._Bx(sys, dx_blk))
