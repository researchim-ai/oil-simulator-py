"""Full dense well-block elimination for arbitrary reservoir block size."""

import torch

try:
    from ..surface.liquid import HydraulicConvergenceError
except ImportError:
    from surface.liquid import HydraulicConvergenceError


class FullWellSchur:
    def factor_well_block(self, sys):
        """Queue the factorization/status on device; do not read status here."""
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
        Bcell = D.new_zeros((self.n_w, cells.numel(), sys["B_full"].shape[-1])).index_add_(
            1, inverse, sys["B_full"]
        )
        solved = self._solve_D(sys, Bcell.reshape(self.n_w, -1)).reshape_as(Bcell)
        sys["_surface_cells"], sys["_surface_inverse"], sys["_surface_Bcell"] = (
            cells,
            inverse,
            Bcell,
        )
        sys["_surface_local"] = solved[self.conn_well, inverse]

    def _solve_D(self, sys, rhs):
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
        return torch.einsum("wub,ub->w", sys["_surface_Bcell"], dx[sys["_surface_cells"]])

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
