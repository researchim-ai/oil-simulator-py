"""Roundoff-aware bounded Newton, separate from sparse linear algebra utilities.

The constructor and linear operators retain their legacy implementation. The
nonlinear stopping policy lives here so linear-operator edits cannot replace
its interface or independently invalidate network residual metadata.
"""

import math

import torch

from .torch_krylov import gmres_device
from .torch_linear import BlockJacobi, ColoredJacobian, TorchCOO
from .torch_linear import TorchNewton as _LinearNewton


class TorchNewton(_LinearNewton):
    _early_full_step = True
    _reuse_jacobian = True
    _jacobian_max_age = 3

    def solve(
        self,
        function,
        x,
        lower,
        upper,
        pattern=None,
        tolerance=2e-11,
        *,
        blocks=None,
        maximum_update=None,
        stagnation_tolerance=None,
        roundoff_tolerance=None,
    ):
        if stagnation_tolerance is not None and (
            not math.isfinite(stagnation_tolerance) or stagnation_tolerance < tolerance
        ):
            raise ValueError("Stagnation tolerance must be finite and no smaller than target")
        if roundoff_tolerance is not None and (
            stagnation_tolerance is None or not callable(roundoff_tolerance)
        ):
            raise ValueError("Row roundoff requires a callable and explicit stagnation budget")
        x = torch.maximum(lower, torch.minimum(upper, x)).clone()
        evaluations = 0
        best_acceptable = math.inf
        best_acceptable_x = None
        stagnant = 0
        self.jacobian_builds = 0
        self.jacobian_reuses = 0
        self.jacobian_retry_refreshes = 0
        # A reused solver object may now describe another phase pattern or
        # block partition. Compile topology anew for each solve as well.
        self.coloring = None
        self.matrix = None
        self.block_preconditioner = None
        # Numeric factors belong to this residual/branch/solve only. They
        # must never survive a new time step or a changed phase topology.
        have_jacobian = False
        age = 0
        previous_norm = math.inf
        force_refresh = False
        retrying = False
        iterations = 0
        reuse_allowed = self._reuse_jacobian and (
            self.device.type == "cpu" or self.host_convergence
        )
        self.stop_reason = "iteration_budget"
        while iterations < self.maximum_iterations:
            r = function(x)
            evaluations += 1
            current_norm = math.inf
            qualified = False
            # CPU checks require no device synchronization. CUDA keeps a fixed
            # iteration budget; accepted status is read once at the API boundary.
            if (
                self.device.type == "cpu"
                or self.host_convergence
                or stagnation_tolerance is not None
            ):
                eligible = False
                if roundoff_tolerance is not None:
                    bounds = roundoff_tolerance()
                    if not isinstance(bounds, torch.Tensor) or bounds.shape != r.shape:
                        raise ValueError("One roundoff scale per nonlinear residual row required")
                    good = torch.isfinite(bounds).all() & (bounds >= 0).all()
                    scale = torch.maximum(bounds, bounds.new_full((), stagnation_tolerance))
                    ratio = torch.where(good, (r.abs() / scale).max(), r.new_full((), math.inf))
                    metrics = torch.stack((r.abs().max(), ratio)).detach().cpu().tolist()
                    norm, eligible = metrics[0], metrics[1] <= 1
                    if self.device.type == "cuda":
                        self.status_reads += 1
                elif self.device.type == "cuda":
                    norm = float(r.abs().max().detach().cpu())
                    self.status_reads += 1
                else:
                    norm = float(r.abs().max().detach())
                current_norm = norm
                if norm <= tolerance:
                    self.stop_reason = "target_tolerance"
                    break
                qualified = stagnation_tolerance is not None and (
                    norm < stagnation_tolerance or eligible
                )
                if qualified and not retrying:
                    previous_best = best_acceptable
                    if norm < best_acceptable:
                        best_acceptable = norm
                        best_acceptable_x = x.clone()
                    stagnant = 0 if norm < 0.99 * previous_best else stagnant + 1
                    if stagnant >= 4:
                        # Keep the best infinity-norm candidate, not the last
                        # rounded iterate. This only shortens iterations already
                        # inside the explicit caller's strict acceptance budget.
                        x = best_acceptable_x
                        self.stop_reason = (
                            "stagnated_within_roundoff"
                            if norm >= stagnation_tolerance
                            else "stagnated_within_tolerance"
                        )
                        break
                elif not qualified:
                    best_acceptable = math.inf
                    best_acceptable_x = None
                    stagnant = 0
            retrying = False
            reused = (
                reuse_allowed
                and have_jacobian
                and not force_refresh
                and age < self._jacobian_max_age
                # Close to the requested arithmetic floor a stale tangent
                # costs more rejected trials than it saves assembly. Keep
                # exact Newton there; reuse only on a contracting trajectory.
                and not qualified
                and current_norm > 100 * tolerance
                and current_norm <= 0.2 * previous_norm
            )
            force_refresh = False
            if reused:
                self.jacobian_reuses += 1
            else:
                self.jacobian_builds += 1
                age = 0
            if x.numel() <= self.dense_limit:
                if not reused:
                    J = torch.func.jacfwd(function)(x)
                    factors, pivots, info = torch.linalg.lu_factor_ex(J, check_errors=False)
                    # Factor the explicit device regularization once per J,
                    # retaining the original singular-J fallback.
                    normal = J.T @ J + torch.eye(x.numel(), dtype=x.dtype, device=x.device) * 1e-12
                    lm_factors, lm_pivots, _ = torch.linalg.lu_factor_ex(normal, check_errors=False)
                step = torch.linalg.lu_solve(factors, pivots, -r[:, None]).squeeze(-1)
                lm = torch.linalg.lu_solve(lm_factors, lm_pivots, (-J.T @ r)[:, None]).squeeze(-1)
                step = torch.where((info == 0) & torch.isfinite(step).all(), step, lm)
            else:
                if pattern is None:
                    raise ValueError("Sparse pattern required above dense limit")
                if self.coloring is None:
                    self.coloring = ColoredJacobian(x.numel(), *pattern, self.device)
                if not reused:
                    values = self.coloring.values(function, x)
                    if self.matrix is None:
                        self.matrix = TorchCOO(*pattern, values, x.numel(), self.device)
                    else:
                        self.matrix.values = values.index_select(-1, self.matrix.order)
                    diagonal = self.matrix.diagonal()
                    preconditioner = torch.where(
                        diagonal.abs() > 1e-12, diagonal, self.matrix.row_scale()
                    )
                    if blocks is not None:
                        if self.block_preconditioner is None:
                            self.block_preconditioner = BlockJacobi(
                                *pattern, blocks, x.numel(), self.device
                            )
                        self.block_preconditioner.update(values)
                step = gmres_device(
                    self.matrix.mv,
                    -r,
                    preconditioner,
                    preconditioner=self.block_preconditioner if blocks is not None else None,
                )
            have_jacobian = True
            age += 1
            if maximum_update is not None:
                step = torch.maximum(-maximum_update, torch.minimum(maximum_update, step))
            best = x
            best_norm = torch.dot(r, r)
            scales = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
            remaining = scales
            if self._early_full_step and (self.device.type == "cpu" or self.host_convergence):
                candidate = torch.maximum(lower, torch.minimum(upper, x + step))
                value = function(candidate)
                evaluations += 1
                norm = torch.dot(value, value)
                take = torch.isfinite(norm) & (norm < best_norm)
                best = torch.where(take, candidate, best)
                previous_norm = best_norm
                best_norm = torch.where(take, norm, best_norm)
                if self.device.type == "cuda":
                    self.status_reads += 1
                # Strong actual contraction suffices to accept a full Newton
                # step. Otherwise retain the original best-of-six search.
                # This affects globalization, never the stopping tolerance.
                strong = take & (norm <= 0.25 * previous_norm)
                remaining = () if bool(strong.detach().cpu()) else scales[1:]
            if self.device.type == "cuda" and remaining:
                # Evaluate the SAME candidates together. This removes six
                # serial streams of tiny kernel launches without weakening
                # the line search or reading a scalar on the host.
                fractions = self.line_scales[-len(remaining) :]
                candidates = torch.maximum(
                    lower, torch.minimum(upper, x + fractions[:, None] * step)
                )
                values = torch.vmap(function)(candidates)
                norms = values.square().sum(-1)
                norms = torch.where(torch.isfinite(norms), norms, math.inf)
                index = norms.argmin()
                take = norms.index_select(0, index.reshape(1)).squeeze(0) < best_norm
                candidate = candidates.index_select(0, index.reshape(1)).squeeze(0)
                best = torch.where(take, candidate, best)
                best_norm = torch.where(
                    take, norms.index_select(0, index.reshape(1)).squeeze(0), best_norm
                )
                evaluations += len(remaining)
            else:
                for scale in remaining:
                    candidate = torch.maximum(lower, torch.minimum(upper, x + scale * step))
                    value = function(candidate)
                    evaluations += 1
                    norm = torch.dot(value, value)
                    take = torch.isfinite(norm) & (norm < best_norm)
                    best = torch.where(take, candidate, best)
                    best_norm = torch.where(take, norm, best_norm)
            if self.device.type == "cpu" or self.host_convergence:
                # Six trial lengths can all overshoot a positive log-stock
                # root. Refine only on stagnation, rather than spending the
                # remaining Newton budget at an unchanged iterate.
                if self.device.type == "cuda":
                    self.status_reads += 1
                if bool((best_norm >= torch.dot(r, r)).detach().cpu()):
                    if reused:
                        # Retry this SAME iterate with a fresh tangent before
                        # paying for tiny steps along an obsolete direction.
                        # A retry consumes no accepted-iteration credit and
                        # does not advance the roundoff plateau counter.
                        self.jacobian_retry_refreshes += 1
                        force_refresh = True
                        retrying = True
                        continue
                    for power in range(6, 25):
                        candidate = torch.maximum(
                            lower, torch.minimum(upper, x + 2.0 ** (-power) * step)
                        )
                        value = function(candidate)
                        evaluations += 1
                        norm = torch.dot(value, value)
                        take = torch.isfinite(norm) & (norm < best_norm)
                        best = torch.where(take, candidate, best)
                        best_norm = torch.where(take, norm, best_norm)
                    if self.device.type == "cuda":
                        self.status_reads += 1
                    if bool((best_norm >= torch.dot(r, r)).detach().cpu()):
                        self.stop_reason = "line_search_stalled"
                        break
            x = torch.where((r.abs().max() <= tolerance), x, best)
            previous_norm = current_norm
            iterations += 1
        self.iterations = iterations
        residual = function(x)
        evaluations += 1
        success = (
            torch.isfinite(x).all()
            & torch.isfinite(residual).all()
            & (residual.abs().max() <= 1e-8)
        )
        return x, residual, success, evaluations
