"""Roundoff-aware bounded Newton, separate from sparse linear algebra utilities.

The constructor and linear operators retain their legacy implementation. The
nonlinear stopping policy lives here so linear-operator edits cannot replace
its interface or independently invalidate network residual metadata.
"""

import math

import torch

from .cuda_execution import current_execution
from .torch_krylov import gmres_device
from .torch_linear import BlockJacobi, ColoredJacobian, TorchCOO
from .torch_linear import TorchNewton as _LinearNewton


class DenseNewtonHistory:
    """One bounded, caller-owned tangent; never a cache of physical solutions.

    Callers supply a phase/control topology key and the time coefficient.
    A candidate tangent still has to contract THIS residual before use. CUDA
    and differentiable solves deliberately retain their original policy.
    """

    def __init__(self):
        self.entry = None
        self.hits = 0
        self.refreshes = 0
        self.builds = 0

    def clear(self):
        self.entry = None

    def take(self, key, coefficient, x, lower, upper):
        saved = self.entry
        self.entry = None  # A failed/interrupted solve cannot publish this again.
        if saved is None:
            return None
        if (
            saved["key"] != key
            or saved["uses"] >= 3
            or not 0.8 <= coefficient / saved["coefficient"] <= 1.25
            or saved["anchor"].shape != x.shape
            or saved["anchor"].dtype != x.dtype
            or saved["anchor"].device != x.device
            or not torch.equal(saved["lower"], lower)
            or not bool(((x - saved["anchor"]).abs() <= 0.05 * (1 + x.abs())).all())
        ):
            return None
        if not torch.equal(saved["upper"], upper):
            # PR log-stock ceilings follow the current inventory. An inactive
            # ceiling is not a residual coefficient: exact equality would
            # forbid all reuse in a flowing finite vessel. Permit only small
            # drift, with BOTH linearization and current points strictly below
            # BOTH ceilings. Lower/physical bounds still match exactly.
            ceiling = torch.minimum(saved["upper"], upper)
            margin = 1e-8 * (1 + ceiling.abs())
            compatible = (
                ((saved["upper"] - upper).abs() <= 0.01 * (1 + upper.abs())).all()
                & (saved["anchor"] < ceiling - margin).all()
                & (x < ceiling - margin).all()
            )
            if not bool(compatible):
                return None
        return saved


class TorchNewton(_LinearNewton):
    _early_certified_roundoff = True
    _early_full_step = True
    _reuse_jacobian = True
    _jacobian_max_age = 3
    _reuse_identical_trials = True
    _reuse_accepted_residual = True
    _stagnation_patience = 2
    _roundoff_line_search = True
    _dense_jacobian_mode = "forward"

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
        linearization_history=None,
        linearization_key=None,
        time_coefficient=None,
    ):
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Newton tolerance must be finite and positive")
        if self._dense_jacobian_mode not in ("forward", "reverse"):
            raise ValueError("Unknown dense Jacobian AD mode")
        if stagnation_tolerance is not None and (
            not math.isfinite(stagnation_tolerance) or stagnation_tolerance < tolerance
        ):
            raise ValueError("Stagnation tolerance must be finite and no smaller than target")
        if roundoff_tolerance is not None and (
            stagnation_tolerance is None or not callable(roundoff_tolerance)
        ):
            raise ValueError("Row roundoff requires a callable and explicit stagnation budget")
        if linearization_history is not None and (
            not isinstance(linearization_history, DenseNewtonHistory)
            or linearization_key is None
            or time_coefficient is None
            or not math.isfinite(time_coefficient)
            or time_coefficient <= 0
        ):
            raise ValueError(
                "Tangent history requires an explicit key and positive time coefficient"
            )
        x = torch.maximum(lower, torch.minimum(upper, x)).clone()
        evaluations = 0
        best_acceptable = math.inf
        best_acceptable_x = None
        stagnant = 0
        self.jacobian_builds = 0
        self.jacobian_reuses = 0
        self.jacobian_retry_refreshes = 0
        self.identical_trial_skips = 0
        self.accepted_residual_reuses = 0
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
        history = (
            linearization_history
            if reuse_allowed
            and self._early_full_step
            and self.device.type == "cpu"
            and x.numel() <= self.dense_limit
            and not torch.is_grad_enabled()
            and not x.requires_grad
            else None
        )
        saved = (
            history.take(linearization_key, time_coefficient, x, lower, upper)
            if history is not None
            else None
        )
        imported = saved is not None
        self.cross_solve_reuses = 0
        self.cross_solve_refreshes = 0
        if imported:
            J, factors, pivots, info = (saved[k] for k in ("matrix", "factors", "pivots", "info"))
            anchor = saved["anchor"]
            lm_factors = None
            have_jacobian = True
        self.stop_reason = "iteration_budget"
        execution = current_execution()
        accelerated = execution is not None and execution.eligible(x)
        carry_residual = (
            self._reuse_accepted_residual
            and not torch.is_grad_enabled()
            and not x.requires_grad
            and not torch._C._functorch.is_functorch_wrapped_tensor(x)
            and (
                roundoff_tolerance is None
                or (
                    hasattr(function, "runtime_values")
                    and hasattr(function, "accept_runtime_bounds")
                )
            )
        )
        pending_residual = None

        def evaluate(z):
            if accelerated and hasattr(function, "runtime_values"):
                row, bounds = execution.residual_values(function.runtime_values, z)
                function.accept_runtime_bounds(bounds)
                return row
            if accelerated and hasattr(function, "runtime_residual"):
                return execution.residual_values(function.runtime_residual, z)
            return function(z)

        while iterations < self.maximum_iterations:
            if pending_residual is None:
                r = evaluate(x)
                evaluations += 1
            else:
                r, cached_bounds = pending_residual
                pending_residual = None
                if cached_bounds is not None:
                    function.accept_runtime_bounds(cached_bounds)
                self.accepted_residual_reuses += 1
            current_bounds = None
            objective_scale = None
            current_norm = math.inf
            qualified = False
            certified_floor = False
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
                    if carry_residual:
                        current_bounds = bounds.clone()
                    if not isinstance(bounds, torch.Tensor) or bounds.shape != r.shape:
                        raise ValueError("One roundoff scale per nonlinear residual row required")
                    good = torch.isfinite(bounds).all() & (bounds >= 0).all()
                    scale = torch.maximum(bounds, bounds.new_full((), stagnation_tolerance))
                    if self._roundoff_line_search:
                        # Freeze the caller-certified row scales for this
                        # entire line search. Cancellation noise in one row
                        # must not hide improvement in stricter rows. This
                        # changes globalization only, never the success budget.
                        objective_scale = scale.detach() / stagnation_tolerance
                    ratio = torch.where(good, (r.abs() / scale).max(), r.new_full((), math.inf))
                    # Stop only where EVERY non-target residual is covered by
                    # its own caller-supplied arithmetic certificate. Rows with
                    # no certificate still meet the original target tolerance.
                    floor = (
                        good
                        & (r.abs() <= torch.maximum(bounds, bounds.new_full((), tolerance))).all()
                    )
                    metrics = (
                        torch.stack((r.abs().max(), ratio, floor.to(r.dtype)))
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    norm, eligible, certified_floor = metrics[0], metrics[1] <= 1, bool(metrics[2])
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
                if self._early_certified_roundoff and certified_floor:
                    self.stop_reason = "certified_roundoff"
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
                    if stagnant >= self._stagnation_patience:
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
                and (imported or current_norm <= 0.2 * previous_norm)
            )
            force_refresh = False
            if reused:
                self.jacobian_reuses += 1
                if imported:
                    self.cross_solve_reuses += 1
                    history.hits += 1
            else:
                imported = False
                self.jacobian_builds += 1
                if history is not None:
                    history.builds += 1
                age = 0
            if x.numel() <= self.dense_limit:
                if not reused:
                    transform = (
                        torch.func.jacrev
                        if self._dense_jacobian_mode == "reverse"
                        else torch.func.jacfwd
                    )
                    J = execution.jacobian(function, x) if accelerated else transform(function)(x)
                    anchor = x.detach().clone()
                    factors, pivots, info = torch.linalg.lu_factor_ex(J, check_errors=False)
                    lm_factors = None
                step = torch.linalg.lu_solve(factors, pivots, -r[:, None]).squeeze(-1)
                direct_valid = (info == 0) & torch.isfinite(step).all()
                # CPU can inspect LU status without device synchronization:
                # do not form/factor J^T J or solve its RHS on a healthy J.
                # CUDA retains the device-selected fallback, without a new
                # host read. A needed regularized factor lasts only this J.
                if self.device.type != "cpu" or not bool(direct_valid):
                    if lm_factors is None:
                        normal = (
                            J.T @ J + torch.eye(x.numel(), dtype=x.dtype, device=x.device) * 1e-12
                        )
                        lm_factors, lm_pivots, _ = torch.linalg.lu_factor_ex(
                            normal, check_errors=False
                        )
                    lm = torch.linalg.lu_solve(lm_factors, lm_pivots, (-J.T @ r)[:, None]).squeeze(
                        -1
                    )
                    step = torch.where(direct_valid, step, lm)
            else:
                if pattern is None:
                    raise ValueError("Sparse pattern required above dense limit")
                if self.coloring is None:
                    self.coloring = ColoredJacobian(x.numel(), *pattern, self.device)
                if not reused:
                    values = (
                        execution.colored_values(self.coloring, function, x)
                        if accelerated
                        else self.coloring.values(function, x)
                    )
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
            best_residual = r
            best_bounds = current_bounds
            reusable_line = carry_residual

            def squared_norm(value, objective_scale=objective_scale):
                scaled = value if objective_scale is None else value / objective_scale
                return torch.dot(scaled, scaled)

            best_norm = squared_norm(r)
            scales = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
            remaining = scales
            previous_trial = None
            previous_value = None
            previous_bounds = None
            trial_bounds = None

            def retain(take, value, budget):
                nonlocal best_residual, best_bounds
                if carry_residual:
                    best_residual = torch.where(take, value, best_residual)
                    if best_bounds is not None:
                        best_bounds = torch.where(take, budget, best_bounds)

            def trial_value(candidate, x=x, r=r, current_bounds=current_bounds):
                nonlocal evaluations, previous_trial, previous_value, previous_bounds, trial_bounds
                # Near roundoff or a bound, different trial lengths can
                # produce the exact same vector. The residual is a pure
                # function: reuse its value, without changing candidates,
                # line-search ordering or any acceptance threshold. Keep
                # this CPU-only to avoid extra CUDA synchronization.
                if self._reuse_identical_trials and self.device.type == "cpu":
                    if torch.equal(candidate, x):
                        self.identical_trial_skips += 1
                        trial_bounds = current_bounds
                        return r
                    if previous_trial is not None and torch.equal(candidate, previous_trial):
                        self.identical_trial_skips += 1
                        trial_bounds = previous_bounds
                        return previous_value
                value = evaluate(candidate)
                evaluations += 1
                previous_trial, previous_value = candidate, value
                trial_bounds = (
                    roundoff_tolerance().clone()
                    if carry_residual and roundoff_tolerance is not None
                    else None
                )
                previous_bounds = trial_bounds
                return value

            if self._early_full_step and (self.device.type == "cpu" or self.host_convergence):
                candidate = torch.maximum(lower, torch.minimum(upper, x + step))
                value = trial_value(candidate)
                norm = squared_norm(value)
                take = torch.isfinite(norm) & (norm < best_norm)
                best = torch.where(take, candidate, best)
                retain(take, value, trial_bounds)
                previous_norm = best_norm
                best_norm = torch.where(take, norm, best_norm)
                if self.device.type == "cuda":
                    self.status_reads += 1
                # Strong actual contraction suffices to accept a full Newton
                # step. Otherwise retain the original best-of-six search.
                # This affects globalization, never the stopping tolerance.
                strong = take & (norm <= 0.25 * previous_norm)
                if imported and not bool(strong):
                    # An old tangent is only a cheap predictor. If it fails
                    # the ordinary full-step contraction test, retry this
                    # exact iterate with fresh Newton, before any backtracking.
                    self.cross_solve_refreshes += 1
                    history.refreshes += 1
                    imported = False
                    saved = None
                    force_refresh = True
                    retrying = True
                    if carry_residual:
                        pending_residual = r, current_bounds
                    continue
                imported = False
                remaining = () if bool(strong.detach().cpu()) else scales[1:]
            if self.device.type == "cuda" and remaining:
                # Vmap may use another numerical property kernel; keep the
                # original scalar recheck instead of carrying a batch result.
                reusable_line = False
                # Evaluate the SAME candidates together. This removes six
                # serial streams of tiny kernel launches without weakening
                # the line search or reading a scalar on the host.
                fractions = self.line_scales[-len(remaining) :]
                candidates = torch.maximum(
                    lower, torch.minimum(upper, x + fractions[:, None] * step)
                )
                values = torch.vmap(function)(candidates)
                scaled = values if objective_scale is None else values / objective_scale
                norms = scaled.square().sum(-1)
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
                    value = trial_value(candidate)
                    norm = squared_norm(value)
                    take = torch.isfinite(norm) & (norm < best_norm)
                    best = torch.where(take, candidate, best)
                    retain(take, value, trial_bounds)
                    best_norm = torch.where(take, norm, best_norm)
            if self.device.type == "cpu" or self.host_convergence:
                # Six trial lengths can all overshoot a positive log-stock
                # root. Refine only on stagnation, rather than spending the
                # remaining Newton budget at an unchanged iterate.
                if self.device.type == "cuda":
                    self.status_reads += 1
                if bool((best_norm >= squared_norm(r)).detach().cpu()):
                    if reused:
                        # Retry this SAME iterate with a fresh tangent before
                        # paying for tiny steps along an obsolete direction.
                        # A retry consumes no accepted-iteration credit and
                        # does not advance the roundoff plateau counter.
                        self.jacobian_retry_refreshes += 1
                        force_refresh = True
                        retrying = True
                        if carry_residual:
                            pending_residual = r, current_bounds
                        continue
                    for power in range(6, 25):
                        candidate = torch.maximum(
                            lower, torch.minimum(upper, x + 2.0 ** (-power) * step)
                        )
                        value = trial_value(candidate)
                        norm = squared_norm(value)
                        take = torch.isfinite(norm) & (norm < best_norm)
                        best = torch.where(take, candidate, best)
                        retain(take, value, trial_bounds)
                        best_norm = torch.where(take, norm, best_norm)
                    if self.device.type == "cuda":
                        self.status_reads += 1
                    if bool((best_norm >= squared_norm(r)).detach().cpu()):
                        self.stop_reason = "line_search_stalled"
                        break
            keep = r.abs().max() <= tolerance
            if reusable_line:
                next_residual = torch.where(keep, r, best_residual)
                next_bounds = (
                    None
                    if current_bounds is None
                    else torch.where(keep, current_bounds, best_bounds)
                )
                pending_residual = next_residual, next_bounds
            x = torch.where(keep, x, best)
            previous_norm = current_norm
            iterations += 1
        self.iterations = iterations
        # A native compiler may regroup floating-point arithmetic. Its kernel
        # accelerates iterations, never certifies its own final acceptance.
        residual = (
            function(x) if accelerated and getattr(execution, "eager_final", False) else evaluate(x)
        )
        evaluations += 1
        # Success is a statement about THIS caller's error budget. A generic
        # loose cutoff can otherwise turn a stalled or exhausted solve into a
        # successful time step even though its requested accuracy was missed.
        budget = residual.new_full(
            residual.shape, tolerance if stagnation_tolerance is None else stagnation_tolerance
        )
        valid_budget = residual.new_tensor(True, dtype=torch.bool)
        if roundoff_tolerance is not None:
            bounds = roundoff_tolerance()
            if not isinstance(bounds, torch.Tensor) or bounds.shape != residual.shape:
                raise ValueError("One roundoff scale per nonlinear residual row required")
            valid_budget = torch.isfinite(bounds).all() & (bounds >= 0).all()
            budget = torch.maximum(budget, bounds)
        success = (
            torch.isfinite(x).all()
            & torch.isfinite(residual).all()
            & valid_budget
            & (residual.abs() <= budget).all()
        )
        if history is not None and have_jacobian and bool(success) and bool(info == 0):
            history.entry = {
                "key": linearization_key,
                "coefficient": time_coefficient if self.jacobian_builds else saved["coefficient"],
                "uses": 0 if self.jacobian_builds else saved["uses"] + 1,
                "anchor": anchor,
                "lower": lower.detach().clone(),
                "upper": upper.detach().clone(),
                "matrix": J,
                "factors": factors,
                "pivots": pivots,
                "info": info,
            }
        return x, residual, success, evaluations
