"""Frozen structured B/D for producer wells attached to a directed tree.

Eliminate subtree component admittances instead of forming dense well matrices.
Bad local pivots require a weighted diagonal-dominance certificate before a
bounded matrix-free GMRES fallback. Every solve checks its actual residual.
"""

from collections import deque

import torch

try:
    from ..surface.liquid import HydraulicConvergenceError
    from ..surface.torch_linear import gmres_device
    from ..surface.tree_pressure_tangent import TreePressureTangent
except ImportError:
    from surface.liquid import HydraulicConvergenceError
    from surface.torch_linear import gmres_device
    from surface.tree_pressure_tangent import TreePressureTangent


class TreeWellSystem:
    def __init__(
        self,
        spec,
        well_nodes,
        conn_well,
        conn_cell,
        dq,
        Cv,
        control_weights,
        rate_mask,
        surface_mask,
        *,
        has_surface=None,
    ):
        self.nw = len(well_nodes)
        nc, _, self.nb = dq.shape
        if (
            not self.nw
            or dq.shape[1] != 3
            or Cv.shape != (nc, 3)
            or control_weights.shape != (nc, 3)
        ):
            raise ValueError("Exact structured connection dimensions required")
        if (
            conn_well.shape != (nc,)
            or conn_cell.shape != (nc,)
            or rate_mask.shape != (self.nw,)
            or surface_mask.shape != (self.nw,)
        ):
            raise ValueError("Exact structured index/control dimensions required")
        if any(
            x.dtype != torch.float64 or x.device != dq.device for x in (dq, Cv, control_weights)
        ):
            raise ValueError("Float64 connection data on one device required")
        if any(
            x.dtype != torch.long or x.device != dq.device for x in (conn_well, conn_cell)
        ) or any(x.dtype != torch.bool or x.device != dq.device for x in (rate_mask, surface_mask)):
            raise ValueError("Device-local long indices and boolean controls required")
        self.device = dq.device
        self.conn_well, self.conn_cell = conn_well, conn_cell
        self.dq, self.Cv, self.weights = dq.detach(), Cv.detach(), control_weights.detach()
        self.rate, self.surface = rate_mask, surface_mask
        self.has_surface = bool(surface_mask.any()) if has_surface is None else has_surface
        if type(self.has_surface) is not bool:
            raise ValueError("Explicit surface-row status must be boolean")
        self.U = Cv.new_zeros(self.nw, 3).index_add_(0, conn_well, self.Cv)
        self.B_rate = (self.weights[:, :, None] * self.dq).sum(1)
        self.rate_diagonal = Cv.new_zeros(self.nw).index_add_(
            0, conn_well, (self.weights * self.Cv).sum(1)
        )
        fixed = torch.where(self.rate, self.rate_diagonal, torch.ones_like(self.rate_diagonal))
        valid_fixed = ~self.rate | (fixed.abs() > 1e-30)
        self.fixed = torch.where(valid_fixed, fixed, torch.ones_like(fixed))
        self.factor_bad = (
            (~valid_fixed).any()
            | ~torch.isfinite(self.dq).all()
            | ~torch.isfinite(self.Cv).all()
            | ~torch.isfinite(self.weights).all()
        )
        self.factor_bad |= (surface_mask.any() != self.has_surface) | (
            rate_mask & surface_mask
        ).any()
        self.input_bad = self.factor_bad.clone() | ~torch.isfinite(self.U).all()
        self.pivot_bad = self.factor_bad.new_zeros(())
        self.factor_bad = self.input_bad | self.pivot_bad
        self.prepared = False
        self.iterative_fallback = False
        self.solve_count = 0
        self.maximum_relative_residual = Cv.new_zeros(())
        self.tangent = None
        if not self.has_surface:
            self.diagonal = fixed
            self.local = torch.where(
                self.rate[conn_well, None], self.B_rate / self.fixed[conn_well, None], 0.0
            )
            return
        self.tangent = TreePressureTangent(spec, device=self.device)
        names = self.tangent.node_names
        lookup = self.tangent.node_index
        if set(well_nodes) - set(self.tangent.source_names):
            raise ValueError("Every structured well must have a declared tree source")
        self.well_node = torch.tensor(
            [lookup[n] for n in well_nodes], dtype=torch.long, device=self.device
        )
        self.source_node = torch.tensor(
            [lookup[n] for n in self.tangent.source_names], dtype=torch.long, device=self.device
        )
        parent = [lookup[p] if p is not None else i for i, p in enumerate(spec["parent_names"])]
        self.parent = torch.tensor(parent, dtype=torch.long, device=self.device)
        root = spec["parent_names"].index(None)
        children = [[] for _ in names]
        for i, p in enumerate(parent):
            if i != root:
                children[p].append(i)
        self.levels = []
        queue = deque([root])
        while queue:
            group = list(queue)
            queue.clear()
            if group != [root]:
                self.levels.append(torch.tensor(group, dtype=torch.long, device=self.device))
            queue.extend(c for i in group for c in children[i])
        self.a = Cv.new_tensor(spec["pressure_coefficients"])
        self.b = Cv.new_tensor(spec["mass_coefficients"])
        self.H = Cv.new_zeros(len(names), 3).index_add_(
            0, self.well_node, self.U * self.surface[:, None]
        )
        self.inverse_pivot = Cv.new_ones(len(names))
        self.A = Cv.new_zeros(len(names))
        for ids in reversed(self.levels):
            product = self.b[ids] * self.H[ids]
            pivot = 1 - product.sum(1)
            scale = 1 + product.abs().sum(1)
            valid = torch.isfinite(pivot) & (
                pivot.abs() > 128 * torch.finfo(pivot.dtype).eps * scale
            )
            self.pivot_bad |= (~valid).any()
            inverse = torch.where(valid, pivot, torch.ones_like(pivot)).reciprocal()
            self.inverse_pivot[ids] = inverse
            self.A[ids] = self.a[ids] * inverse
            self.H.index_add_(0, self.parent[ids], self.H[ids] * self.A[ids, None])
        self.pivot_bad |= ~torch.isfinite(self.H).all() | ~torch.isfinite(self.A).all()
        self.factor_bad = self.input_bad | self.pivot_bad
        own_K = self.tangent.own_source_coefficients()[self.well_node]
        d_surface = 1 - (own_K * self.U).sum(1)
        self.diagonal = torch.where(self.surface, d_surface, fixed)
        local_B = torch.where(
            self.rate[conn_well, None],
            self.B_rate,
            torch.where(
                self.surface[conn_well, None], -(own_K[conn_well, :, None] * self.dq).sum(1), 0.0
            ),
        )
        threshold = 128 * torch.finfo(Cv.dtype).eps * (1 + (own_K * self.U).abs().sum(1))
        usable = torch.isfinite(self.diagonal) & torch.where(
            self.surface, self.diagonal.abs() > threshold, self.diagonal.abs() > 1e-30
        )
        safe = torch.where(usable, self.diagonal, torch.ones_like(self.diagonal))
        # Only a preconditioner approximation. The full Schur action restores
        # all omitted terms; unusable local diagonals contribute zero.
        self.local = torch.where(usable[conn_well, None], local_B / safe[conn_well, None], 0.0)

    def prepare(self, factor_validated=None):
        """One admission read, normally already included in the FIM packet."""
        if self.prepared:
            return
        if factor_validated is not True:
            if bool(self.input_bad):
                raise HydraulicConvergenceError(
                    "Invalid structured well coefficients or singular rate row"
                )
            self.iterative_fallback = bool(self.pivot_bad)
            if self.iterative_fallback:
                if not self._certify_fallback():
                    raise HydraulicConvergenceError(
                        "Local tree pivots require a certified nonsingular well block"
                    )
                # A local pivot may fail while global D is invertible. Only
                # a certified nonsingular block may use the Krylov fallback.
                self.factor_bad.zero_()
        self.prepared = True

    def _certify_fallback(self):
        """Sufficient weighted diagonal-dominance bound, without forming D."""
        spec = self.tangent.to_dict()
        spec["pressure_coefficients"] = [abs(v) for v in spec["pressure_coefficients"]]
        spec["mass_coefficients"] = [[abs(v) for v in row] for row in spec["mass_coefficients"]]
        absolute = TreePressureTangent(spec, device=self.device)
        own_bound = (absolute.own_source_coefficients()[self.well_node] * self.U.abs()).sum(1)
        diagonal = self.diagonal.abs()
        weights = torch.ones_like(diagonal)
        safe = diagonal.clamp_min(1e-300)
        for _ in range(12):
            own = self.Cv.new_zeros(len(absolute.node_names), 3).index_add_(
                0, self.well_node, self.U.abs() * weights[:, None]
            )
            total = absolute.matvec(own[self.source_node])[self.well_node]
            off = torch.where(self.surface, (total - own_bound * weights).clamp_min(0), 0.0)
            weights = 1 + off / safe
        own = self.Cv.new_zeros(len(absolute.node_names), 3).index_add_(
            0, self.well_node, self.U.abs() * weights[:, None]
        )
        total = absolute.matvec(own[self.source_node])[self.well_node]
        off = torch.where(self.surface, (total - own_bound * weights).clamp_min(0), 0.0)
        rounding = (
            512
            * torch.finfo(diagonal.dtype).eps
            * (diagonal * weights + total.abs() + own_bound * weights)
        )
        certified = (
            torch.isfinite(weights).all()
            & (weights > 0).all()
            & torch.isfinite(diagonal).all()
            & (diagonal * weights > off + rounding).all()
        )
        return bool(certified)

    def Bx(self, dx):
        dm = (self.dq * dx[self.conn_cell, None, :]).sum(2)
        rate = self.Cv.new_zeros(self.nw).index_add_(0, self.conn_well, (self.weights * dm).sum(1))
        if not self.has_surface:
            return torch.where(self.rate, rate, 0.0)
        own = self.Cv.new_zeros(len(self.tangent.node_names), 3).index_add_(
            0, self.well_node, self.Cv.new_zeros(self.nw, 3).index_add_(0, self.conn_well, dm)
        )
        pressure = self.tangent.matvec(own[self.source_node])[self.well_node]
        return torch.where(self.rate, rate, torch.where(self.surface, -pressure, 0.0))

    def apply_D(self, dp):
        vector = dp.ndim == 1
        values = dp[:, None] if vector else dp
        diagonal = self.fixed[:, None] * values
        if self.has_surface:
            own = self.Cv.new_zeros(len(self.tangent.node_names), 3, values.shape[1]).index_add_(
                0, self.well_node, self.U[:, :, None] * values[:, None, :]
            )
            pressure = self.tangent.matvec(own[self.source_node])[self.well_node]
            result = torch.where(self.surface[:, None], values - pressure, diagonal)
        else:
            result = diagonal
        return result[:, 0] if vector else result

    @torch.no_grad()
    def solve(self, rhs):
        if (
            rhs.ndim not in (1, 2)
            or rhs.shape[0] != self.nw
            or rhs.device != self.device
            or rhs.dtype != torch.float64
        ):
            raise ValueError("Structured solve requires device-local float64 well RHS")
        vector = rhs.ndim == 1
        rhs = rhs[:, None] if vector else rhs
        self.prepare()
        known = rhs / self.fixed[:, None]
        if self.iterative_fallback:
            answer = torch.stack(
                [
                    gmres_device(self.apply_D, rhs[:, j], self.diagonal, restart=24, cycles=2)
                    for j in range(rhs.shape[1])
                ],
                dim=1,
            )
        elif not self.has_surface:
            answer = known
        else:
            known = torch.where(self.surface[:, None], 0.0, known)
            offsets = known + torch.where(self.surface[:, None], rhs, 0.0)
            g = self.Cv.new_zeros(len(self.tangent.node_names), 3, rhs.shape[1]).index_add_(
                0, self.well_node, self.U[:, :, None] * offsets[:, None, :]
            )
            h = self.Cv.new_zeros(len(self.tangent.node_names), rhs.shape[1])
            for ids in reversed(self.levels):
                h[ids] = (self.b[ids, :, None] * g[ids]).sum(1) * self.inverse_pivot[ids, None]
                f = g[ids] + self.H[ids, :, None] * h[ids, None, :]
                g.index_add_(0, self.parent[ids], f)
            p = torch.zeros_like(h)
            for ids in self.levels:
                p[ids] = self.A[ids, None] * p[self.parent[ids]] + h[ids]
            answer = torch.where(self.surface[:, None], p[self.well_node] + rhs, known)
        actual = self.apply_D(answer)
        residual = (actual - rhs).abs()
        scale = rhs.abs() + actual.abs()
        # Rate and pressure rows have different physical units. The check is
        # per row, never a global norm dominated by pressure units.
        absolute = torch.where(self.rate, 1e-13, 1e-7)[:, None]
        accepted = (
            (residual <= absolute + 1e-10 * scale).all(0)
            & torch.isfinite(answer).all(0)
            & ~self.factor_bad
        )
        self.factor_bad.logical_or_((~accepted).any())
        self.maximum_relative_residual = torch.maximum(
            self.maximum_relative_residual, (residual / (absolute + scale.clamp_min(1e-300))).amax()
        )
        self.solve_count += 1
        answer = torch.where(accepted[None, :], answer, torch.full_like(answer, float("nan")))
        return answer[:, 0] if vector else answer
