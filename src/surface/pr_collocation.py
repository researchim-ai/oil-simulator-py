"""Conservative Gauss4/Radau3 for homogeneous-equilibrium PR graphs.

Reuse the existing PR chemical/volume constraints, HLLC flux and force
operators. Phase transitions use the existing backward-Euler branch search,
with the actual method explicitly recorded; no extrapolated phase states.
Radau3 also integrates mapped machines and shaft angular momentum using
combined fluid/rotor energy coordinates. Gravitational columns and feedback
controller states are not yet part of this adapter.
"""

import math
from dataclasses import replace

import torch

from .cuda_execution import current_execution, declare_runtime_residual
from .equilibrium_cache import cached_fluid_transaction
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .pr_equilibrium_transport import (
    PREquilibriumCellState,
    PREquilibriumTransportNetwork,
    PREquilibriumTransportState,
)
from .pr_pump_connection import PumpConnectionBundle
from .pr_pump_unit import serialize_pumps
from .time_interval import interval_step_s, split_step_s
from .torch_krylov import gmres_device
from .torch_linear import BlockJacobi, TorchCOO, sparse_jacobian_plan
from .torch_nonlinear import DenseNewtonHistory, TorchNewton


def _sdirk_guess(function, initial, first, diagonal, baseline, lower, upper, maximum_update):
    """A bounded initial guess, accepted only on actual residual contraction.

    This never accepts a time step or replaces its error/physical checks.
    The residual floor is 100 times the unchanged Newton tolerance.
    """
    with torch.no_grad():
        if not bool(baseline.abs().max() > 3e-10):
            return first, "unused"
        delta = ((1 / diagonal - 1) * (first - initial)).clamp(-maximum_update, maximum_update)
        candidate = torch.maximum(lower, torch.minimum(upper, first + delta))
        if torch.equal(candidate, first):
            return first, "unused"
        try:
            trial = function(candidate)
        except (PropertyModelError, HydraulicConvergenceError, torch.linalg.LinAlgError):
            return first, "inadmissible"
        if not bool(torch.isfinite(trial).all()):
            return first, "inadmissible"
        if bool(trial.square().sum() <= 0.5 * baseline.square().sum()):
            return candidate, "accepted"
        return first, "rejected"


class PRStageValveRetry(PropertyModelError):
    def __init__(self, updates):
        super().__init__("PR stage pump check-valve branch requires correction")
        self.updates = updates


class PRCollocation:
    _batch_stage_residual = True
    _block_preconditioning = True
    _reuse_step_jacobian = False
    # Dense CPU systems have costly forward-AD primitive dispatch. The same
    # bounded Jacobian is cheaper through VJPs; large systems retain coloring,
    # and native/CUDA execution retains its existing derivative kernels.
    _dense_cpu_jacobian_mode = "colored_reverse"
    _reuse_sdirk_points = True
    _reuse_sdirk_tangent = True
    # Experimental predictor is opt-in: rejected trials did not improve the
    # measured plant and must not add residual evaluations to its default path.
    _predict_sdirk_stage = False

    def __init__(
        self,
        physics,
        *,
        phase_fallback=True,
        scheme="gauss4",
        error_control="step_doubling",
        reuse_step_jacobian=None,
    ):
        if not isinstance(physics, PREquilibriumTransportNetwork):
            raise TypeError("Expected PREquilibriumTransportNetwork")
        if type(phase_fallback) is not bool:
            raise ValueError("phase_fallback must be boolean")
        if reuse_step_jacobian is not None:
            if type(reuse_step_jacobian) is not bool:
                raise ValueError("reuse_step_jacobian must be boolean")
            self._reuse_step_jacobian = reuse_step_jacobian
        if physics.drives and scheme not in ("radau3", "sdirk2"):
            raise ValueError("Dynamic drives require a stiffly accurate scheme")
        if any(
            unit.name in physics._drives and unit.signed_map is not None
            for unit in (*physics.pumps, *physics.pump_connections, *physics.compressors)
        ):
            raise ValueError("Signed drive zero-crossing work requires an event-aware adapter")
        if any(unit.equilibrium_liquid_mixture for unit in physics.pump_connections):
            raise ValueError("Liquid-mixture pump stage traces require a separate adapter")
        if physics.reservoir_ports != "riemann":
            raise ValueError("Stagnation traces require a separate collocation adapter")
        if any(
            n.hydrostatic_reconstruction
            or n.droplet_population
            or n.drift_flux
            or n.distribution_strength
            or n.buoyancy_relaxation_s
            for n in physics.nodes
        ):
            raise ValueError(
                "Only homogeneous PR cells without population/face inventories supported"
            )
        if scheme not in ("gauss4", "radau3", "sdirk2"):
            raise ValueError("Unknown PR collocation scheme")
        if error_control not in ("step_doubling", "embedded") or (
            error_control == "embedded" and scheme not in ("radau3", "sdirk2")
        ):
            raise ValueError("Embedded control requires Radau3")
        self.error_control = error_control
        self._solve_calls = 0
        self._endpoint_rate_cache = None
        self._stage_linearization = DenseNewtonHistory()
        self._linear_work = {
            key: 0
            for key in (
                "jacobian_builds",
                "jacobian_reuses",
                "cross_solve_reuses",
                "cross_solve_refreshes",
            )
        }
        self._stage_predictor_work = {
            k: 0 for k in ("accepted", "rejected", "inadmissible", "unused")
        }
        self._batch_stage_residual = getattr(
            physics, "_batch_collocation_stages", self._batch_stage_residual
        )
        if getattr(current_execution(), "eager_final", False):
            # Native forward AD avoids Dynamo's nested-vmap tracing bug.
            self._batch_stage_residual = False
        self._block_preconditioning = getattr(
            physics, "_collocation_block_preconditioning", self._block_preconditioning
        )
        self.scheme = scheme
        self.stiffly_accurate = scheme != "gauss4"
        self.order = {"gauss4": 4, "radau3": 3, "sdirk2": 2}[scheme]
        self.b = (0.5, 0.5) if scheme == "gauss4" else (0.75, 0.25)
        self.physics = physics
        self.model_id = physics.model_id
        self.phase_fallback = phase_fallback
        q = math.sqrt(3) / 6
        self.a = (
            ((0.25, 0.25 - q), (0.25 + q, 0.25))
            if scheme == "gauss4"
            else ((5 / 12, -1 / 12), (0.75, 0.25))
        )
        if scheme == "sdirk2":
            gamma = 1 - 1 / math.sqrt(2)
            self.a = ((gamma, 0.0), (1 - gamma, gamma))
            self.b = (1 - gamma, gamma)

    def _weighted(self, values):
        return sum(w * v for w, v in zip(self.b, values))

    @staticmethod
    def _physical_blocks(offsets, wall_indices, size, stages=1):
        blocks = [list(range(a, b)) for a, b in zip(offsets, offsets[1:])]
        for j, i in enumerate(wall_indices):
            blocks[i].append(offsets[-1] + j)
        coupled = [[stage * size + index for stage in range(stages) for index in b] for b in blocks]
        # Bound each local dense factor just as in the existing PR/BE path.
        return [b[j : j + 256] for b in coupled for j in range(0, len(b), 256)]

    def _stage_rows(self, z, size, dt, evaluate, rates, *, _return_bounds=False):
        def parts(v):
            row, physical = evaluate(v)
            bounds = physical.get("roundoff_bounds", row.new_zeros(0))
            return row, rates(physical), torch.cat((bounds, row.new_zeros(len(row) - len(bounds))))

        values = z.reshape(2, size)
        if self._batch_stage_residual:
            try:
                rows, rhs, bounds = torch.vmap(parts)(values)
            except RuntimeError:
                # Unsupported transform/backend: retain the original operator
                # and its failure diagnosis. Never substitute a valid residual.
                result = [parts(v) for v in values]
                rows, rhs, bounds = zip(*result)
        else:
            result = [parts(v) for v in values]
            rows, rhs, bounds = zip(*result)
        row = torch.cat(
            [rows[i] - dt * sum(self.a[i][j] * rhs[j] for j in range(2)) for i in range(2)]
        )
        return (row, torch.cat(tuple(bounds))) if _return_bounds else row

    def _problem(
        self,
        state,
        commands,
        heat,
        losses,
        _coupling=None,
        *,
        drive_torque_nm=None,
        restriction_openings=None,
        _boundary=None,
        _pump_valve_states=None,
    ):
        m = self.physics
        boundary = _coupling if _boundary is None else _boundary
        seeds = [m._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
        y, lo, hi, base_evaluate, offsets, problems, old = m._problem(
            state,
            seeds,
            0.0,
            commands,
            heat,
            losses,
            active_override=None if boundary is None else boundary.active,
            guesses_override=None if boundary is None else boundary.ordinary_guesses,
            external_ports=() if _boundary is None else _boundary.external_ports,
            _continuous=True,
            drive_torque_nm=drive_torque_nm,
            restriction_openings=restriction_openings,
            pump_valve_states=PumpConnectionBundle.previous_valves(m)
            if _pump_valve_states is None
            else _pump_valve_states,
        )
        body_size = len(y)
        # Conserved energy coordinates include each rotor in its receiving
        # fluid body. L=I*omega remains an independent differential stock.
        # This avoids the quadratic-energy defect of integrating omega and
        # fluid energy independently with a non-symplectic stiff RK method.
        units = {u.name: u for u in (*m.pumps, *m.compressors, *m.pump_connections)}
        rotor_hosts = {
            d.machine: m.names.index(
                units[d.machine].node
                if hasattr(units[d.machine], "node")
                else units[d.machine].target
            )
            for d in m.drives
        }
        old_rotors = dict(state.rotors)
        wall_indices = [i for i, node in enumerate(m.nodes) if node.wall is not None]
        f = m.fluid
        scale = {i: max(1.0, m.nodes[i].wall.heat_capacity_j_k * f.tref) for i in wall_indices}
        if wall_indices:
            y = torch.cat(
                (y, f.tensor([state.nodes[i][1].wall_temperature_k / f.tref for i in wall_indices]))
            )
            lo = torch.cat(
                (lo, f.tensor([m.nodes[i].wall.min_temperature_k / f.tref for i in wall_indices]))
            )
            hi = torch.cat(
                (hi, f.tensor([m.nodes[i].wall.max_temperature_k / f.tref for i in wall_indices]))
            )
        if (1 if self.scheme == "sdirk2" else 2) * len(y) > m.maximum_unknowns:
            raise ValueError("PR collocation exceeds configured unknown budget")
        old_wall_temperatures = f.tensor(
            [state.nodes[i][1].wall_temperature_k for i in wall_indices]
        )

        def evaluate(z, external_fluxes=None, runtime_controls=None):
            row, physical = base_evaluate(
                z[:body_size],
                external_fluxes=external_fluxes
                if external_fluxes is not None
                else (None if _coupling is None else _coupling.exchange),
                runtime_controls=runtime_controls,
            )
            energy_rows, energy_changes = [], []
            for drive in m.drives:
                i = rotor_hosts[drive.machine]
                energy_rows.append(offsets[i] + problems[i]["energy_row"])
                energy_changes.append(
                    (
                        physical["drives"][drive.machine]["kinetic_energy_j"]
                        - 0.5 * drive.drive.inertia_kg_m2 * old_rotors[drive.machine] ** 2
                    )
                    / problems[i]["energy_scale"]
                )
            if energy_rows:
                row = row.index_add(
                    0, torch.tensor(energy_rows, device=z.device), torch.stack(energy_changes)
                )
            walls, heat_rates, ambient = (
                list(physical["walls"]),
                list(physical["heat"]),
                list(physical["ambient"]),
            )
            wall_rows = []
            for j, i in enumerate(wall_indices):
                wall = m.nodes[i].wall
                tw = z[body_size + j] * f.tref
                walls[i] = tw
                heat_rates[i] = wall.fluid_ua_w_k * (tw - physical["data"][i]["t"])
                ambient[i] = wall.ambient_ua_w_k * (wall.ambient_temperature_k - tw)
                wall_rows.append(
                    (wall.heat_capacity_j_k * (tw - old_wall_temperatures[j]) / scale[i]).reshape(1)
                )
            return torch.cat((row, *wall_rows)), {
                **physical,
                "walls": walls,
                "heat": heat_rates,
                "ambient": ambient,
            }

        evaluate.maximum_update = torch.cat(
            (base_evaluate.maximum_update, f.tensor([math.inf] * len(wall_indices)))
        )
        evaluate.wall_indices = wall_indices
        evaluate.linear_blocks = [list(b) for b in base_evaluate.linear_blocks]
        for j, i in enumerate(wall_indices):
            evaluate.linear_blocks[i].append(body_size + j)
        pattern = (
            set(
                zip(
                    *m._sparsity(
                        offsets,
                        commands,
                        base_evaluate.reservoir_traces.entries,
                        base_evaluate.compressors.entries,
                        base_evaluate.hydrostatic_faces.entries,
                        base_evaluate.droplet_populations.entries,
                    )
                )
            )
            | base_evaluate.additional_sparsity
        )
        for drive, rotor_offset, *_ in base_evaluate.rotors.entries:
            i = rotor_hosts[drive.machine]
            erow = offsets[i] + problems[i]["energy_row"]
            pattern.add((erow, rotor_offset))
            pattern.update((erow, c) for r, c in tuple(pattern) if r == rotor_offset)
        for j, i in enumerate(wall_indices):
            index = body_size + j
            cell = range(offsets[i], offsets[i + 1])
            pattern.add((index, index))
            pattern.update((index, c) for c in cell)
            pattern.update((r, index) for r in cell)
        pattern = sorted(pattern)
        evaluate.pattern = ([r for r, c in pattern], [c for r, c in pattern])
        evaluate.rotors = base_evaluate.rotors
        evaluate.rotor_hosts = rotor_hosts
        evaluate.compressors = base_evaluate.compressors
        evaluate.pump_connections = base_evaluate.pump_connections
        evaluate.restrictions = base_evaluate.restrictions
        evaluate.reservoir_traces = base_evaluate.reservoir_traces
        evaluate.hydrostatic_faces = base_evaluate.hydrostatic_faces
        evaluate.droplet_populations = base_evaluate.droplet_populations
        evaluate.additional_sparsity = set(zip(*evaluate.pattern))

        def rates(physical):
            rows = []
            for i, (node, p, d) in enumerate(zip(m.nodes, problems, physical["data"])):
                # Same row ordering/scales as the original monolithic residual.
                nc = len(p["active"])
                erow = p["energy_row"]
                net = physical["net"][i]
                momentum = (
                    (net[-2] + physical["forces"][i]) / (old[i]["mass"] * m.maximum_velocity_m_s)
                    if node.kind == "pipe_cell"
                    else d["p"] * 0
                )
                rows.append(
                    torch.cat(
                        (
                            net[p["active"]] / p["nscale"],
                            y.new_zeros(erow - nc),
                            ((net[-1] + physical["heat"][i] + heat[i]) / p["energy_scale"]).reshape(
                                1
                            ),
                            momentum.reshape(1),
                        )
                    )
                )
            # Algebraic traces/maps have identically zero time derivatives.
            body_rates = torch.cat(rows)
            body_rates = torch.cat((body_rates, y.new_zeros(body_size - offsets[-1])))
            indices, values = [], []
            for drive, index, *_ in base_evaluate.rotors.entries:
                i = rotor_hosts[drive.machine]
                record = physical["drives"][drive.machine]
                indices.extend((offsets[i] + problems[i]["energy_row"], index))
                values.extend(
                    (
                        record["kinetic_energy_rate_w"] / problems[i]["energy_scale"],
                        record["angular_momentum_rate_nm"]
                        / (drive.drive.inertia_kg_m2 * drive.nominal_speed_rad_s),
                    )
                )
            if indices:
                body_rates = body_rates.index_add(
                    0, torch.tensor(indices, device=y.device), torch.stack(values)
                )
            rows = [body_rates]
            rows.extend(
                ((physical["ambient"][i] - physical["heat"][i]) / scale[i]).reshape(1)
                for i in wall_indices
            )
            return torch.cat(rows)

        return y, lo, hi, evaluate, rates, offsets, problems, old, seeds

    def _reverse_plan(self, initial, pattern):
        from .torch_linear import ReverseColoredJacobian

        # Only structural metadata is retained; never a tape, Jacobian,
        # physical property or numerical state. Exact support is the key:
        # phase, control and quadrature changes cannot alias another graph.
        key = (initial.numel(), tuple(pattern[0]), tuple(pattern[1]))
        cache = getattr(self, "_reverse_colorings", None)
        if cache is None:
            self._reverse_colorings = cache = {}
        coloring = cache.get(key)
        if coloring is None:
            coloring = ReverseColoredJacobian(
                initial.numel(), *pattern, initial.device, chunk_size=initial.numel()
            )
            if len(cache) >= 8:
                cache.pop(next(iter(cache)))
            cache[key] = coloring
        return coloring

    def _solve(self, fn, initial, lo, hi, pattern, update):
        self._solve_calls += 1
        m, f = self.physics, self.physics.fluid
        solver = TorchNewton(f.device, maximum_iterations=40, dense_limit=m.dense_limit)
        if f.device.type == "cpu":
            solver._dense_jacobian_mode = self._dense_cpu_jacobian_mode
            if self._dense_cpu_jacobian_mode == "colored_reverse":
                solver._sparse_jacobian_mode = "auto"
        execution = current_execution()
        coloring = (
            self._reverse_plan(initial, pattern)
            if f.device.type == "cpu"
            and self._dense_cpu_jacobian_mode == "colored_reverse"
            and initial.numel() <= m.dense_limit
            and pattern is not None
            and not (execution is not None and execution.eligible(initial))
            else None
        )
        solver.host_convergence = True
        with torch.no_grad():
            y, row, ok, evaluations = solver.solve(
                fn,
                initial,
                lo,
                hi,
                pattern=pattern,
                tolerance=3e-12,
                stagnation_tolerance=1e-11,
                roundoff_tolerance=getattr(fn, "roundoff_tolerance", None),
                maximum_update=update,
                blocks=getattr(fn, "linear_blocks", None) if self._block_preconditioning else None,
                linearization_history=getattr(fn, "linearization_history", None),
                linearization_key=getattr(fn, "linearization_key", None),
                time_coefficient=getattr(fn, "time_coefficient", None),
                reverse_coloring=coloring,
            )
            for key in self._linear_work:
                self._linear_work[key] += getattr(solver, key)
            # TorchNewton returns the final residual at exactly y, including
            # eager acceptance under native execution. Do not evaluate both
            # complete PR stages a second time just to obtain the same row.
        if not bool((ok & torch.isfinite(row).all() & (row.abs().max() < 1e-9)).cpu()):
            raise HydraulicConvergenceError("PR Gauss nonlinear/constraint solve rejected")
        return y, evaluations, float(row.abs().max().cpu())

    def _embedded_error(
        self, initial, endpoint, evaluate, rates, points, dt, initial_physical=None
    ):
        """Second-order companion, stabilized by an implicit Euler tangent.

        Companion weights at c=(0,1/3,1) are (1/6,1/2,1/3).
        Its defect from Radau3 is h*(f0-1.5*f1+0.5*f2)/6. Solve
        (dU/dy-h*df/dy)*delta=defect on the SAME algebraic manifold;
        no extra nonlinear integration or phase-state extrapolation is used.
        """
        m, f = self.physics, self.physics.fluid
        h = f.tensor(dt)
        if self.scheme == "sdirk2":
            # First-order companion b_hat=(1,0); no derivative at t_n is
            # needed for ideal rate-limited actuator constraints (index two).
            defect = h * self.b[1] * (rates(points[0]) - rates(points[1]))
        else:
            old = evaluate(initial)[1] if initial_physical is None else initial_physical
            defect = h * (rates(old) - 1.5 * rates(points[0]) + 0.5 * rates(points[1])) / 6

        def filtered(y):
            row, physical = evaluate(y)
            return row - h * rates(physical)

        declare_runtime_residual(filtered)
        execution = current_execution()
        accelerated = execution is not None and execution.eligible(endpoint)
        if endpoint.numel() <= m.dense_limit:
            if (
                not accelerated
                and f.device.type == "cpu"
                and self._dense_cpu_jacobian_mode == "colored_reverse"
            ):
                coloring = self._reverse_plan(endpoint, evaluate.pattern)
                if coloring.color_count < 0.75 * endpoint.numel():
                    matrix = coloring.dense(filtered, endpoint)
                else:
                    matrix = torch.func.jacrev(filtered)(endpoint)
            else:
                matrix = (
                    execution.jacobian(filtered, endpoint)
                    if accelerated
                    else torch.func.jacfwd(filtered)(endpoint)
                )
            delta, info = torch.linalg.solve_ex(matrix, defect, check_errors=False)
            valid = (info == 0) & torch.isfinite(delta).all()
            residual = matrix @ delta - defect
        else:
            coloring = sparse_jacobian_plan(
                endpoint.numel(),
                *evaluate.pattern,
                f.device,
                reverse=f.device.type == "cpu"
                and self._dense_cpu_jacobian_mode == "colored_reverse"
                and not accelerated,
            )
            values = (
                execution.colored_values(coloring, filtered, endpoint)
                if accelerated
                else coloring.values(filtered, endpoint)
            )
            matrix = TorchCOO(*evaluate.pattern, values, endpoint.numel(), f.device)
            diagonal = matrix.diagonal()
            preconditioner = torch.where(diagonal.abs() > 1e-12, diagonal, matrix.row_scale())
            blocks = None
            if self._block_preconditioning:
                blocks = BlockJacobi(
                    *evaluate.pattern, evaluate.linear_blocks, endpoint.numel(), f.device
                )
                blocks.update(values)
            delta = gmres_device(matrix.mv, defect, preconditioner, preconditioner=blocks)
            valid = torch.isfinite(delta).all()
            residual = matrix.mv(delta) - defect
        valid = (
            valid
            & torch.isfinite(residual).all()
            & (residual.abs().max() <= 1e-12 + 1e-8 * defect.abs().max())
        )
        if not bool(valid.cpu()):
            return math.inf
        _, low = evaluate(endpoint + delta)
        if not bool(low["valid"].cpu()) or not all(
            bool(d["q"]["valid"].all().cpu()) for d in low["data"]
        ):
            return math.inf
        errors = []
        for i, (node, a, b) in enumerate(zip(m.nodes, points[-1]["data"], low["data"])):
            errors.extend(
                [
                    (
                        (a["n"] - b["n"]).abs()
                        / (1e-10 + m.relative_tolerance * torch.maximum(a["n"].abs(), b["n"].abs()))
                    ).max(),
                    (a["energy"] - b["energy"]).abs()
                    / (
                        1e-6
                        + m.relative_tolerance
                        * torch.maximum(a["energy"].abs(), b["energy"].abs()).clamp_min(1)
                    ),
                    (a["p"] - b["p"]).abs()
                    / (1 + m.relative_tolerance * torch.maximum(a["p"], b["p"])),
                    (a["t"] - b["t"]).abs() / m.temperature_tolerance_k,
                    (a["u"] - b["u"]).abs() / m.velocity_tolerance_m_s,
                ]
            )
            if node.wall is not None:
                errors.append(
                    (points[-1]["walls"][i] - low["walls"][i]).abs() / m.temperature_tolerance_k
                )
        for drive in m.drives:
            a = points[-1]["drives"][drive.machine]["speed_rad_s"]
            b = low["drives"][drive.machine]["speed_rad_s"]
            errors.append(
                (a - b).abs()
                / (
                    1e-8
                    + drive.relative_speed_tolerance * torch.maximum(a.abs(), b.abs()).clamp_min(1)
                )
            )
        if hasattr(evaluate, "additional_error"):
            errors.append(evaluate.additional_error(points[-1], low))
        return float(torch.stack(errors).max().cpu())

    def _step(
        self,
        state,
        dt,
        commands,
        heat,
        utility,
        losses,
        *,
        _coupling=None,
        drive_torque_nm=None,
        restriction_openings=None,
        _prepared_problem=None,
        _initial_physical=None,
        _pump_valve_states=None,
        _valve_attempt=0,
    ):
        m, f = self.physics, self.physics.fluid
        solve_start = self._solve_calls
        prepared = (
            _prepared_problem
            if _prepared_problem is not None
            else self._problem(
                state,
                commands,
                heat,
                losses,
                _coupling,
                drive_torque_nm=drive_torque_nm,
                restriction_openings=restriction_openings,
                _pump_valve_states=_pump_valve_states,
            )
        )
        initial, lo, hi, evaluate, rates, _offsets, problems, old, seeds = prepared
        size = len(initial)
        initialization_solves = 0
        initial_physical = _initial_physical
        rate_key = (
            m.model_id,
            tuple(sorted(commands.items())),
            tuple(heat),
            tuple(sorted(losses.items())),
            tuple(sorted((drive_torque_nm or {}).items())),
            tuple(sorted((restriction_openings or {}).items())),
        )
        machine_graph = bool(m.pumps or m.compressors or m.pump_connections or m.restrictions)
        if (
            (machine_graph or getattr(evaluate, "requires_initial_projection", False))
            and self.error_control == "embedded"
            and self.scheme != "sdirk2"
            and initial_physical is None
        ):
            saved = self._endpoint_rate_cache
            self._endpoint_rate_cache = None
            if (
                _coupling is None
                and saved is not None
                and saved[0] == rate_key
                and saved[1] == state.nodes
                and saved[2] == state.rotors
            ):
                initial_physical = saved[3]
            else:
                # The embedded companion needs f(t_n), not the unprojected
                # PS/PH/map guesses used to initialize Newton. Solve only
                # the old-state algebraic manifold when necessary. Accepted
                # endpoints supply this operator on subsequent steps.
                with torch.no_grad():
                    initial_row, initial_physical = evaluate(initial)
                if not bool(
                    (torch.isfinite(initial_row).all() & (initial_row.abs().max() < 3e-12)).cpu()
                ):

                    def initial_rows(v):
                        return evaluate(v)[0]

                    declare_runtime_residual(initial_rows)
                    initial_rows.linear_blocks = evaluate.linear_blocks
                    initial, _, _ = self._solve(
                        initial_rows,
                        initial,
                        lo,
                        hi,
                        evaluate.pattern,
                        evaluate.maximum_update,
                    )
                    initialization_solves = 1
                    with torch.no_grad():
                        initial_physical = evaluate(initial)[1]
        interval_factor = f.tensor(dt)
        base = evaluate.pattern
        pattern = None
        if self.scheme != "sdirk2" and (
            2 * size > m.dense_limit or self._dense_cpu_jacobian_mode == "colored_reverse"
        ):
            entries = sorted(
                {
                    (i * size + r, j * size + c)
                    for r, c in zip(*base)
                    for i in range(2)
                    for j in range(2)
                }
            )
            pattern = ([r for r, c in entries], [c for r, c in entries])

        latest_bounds = [None]

        def stage_values(z):
            return self._stage_rows(z, size, interval_factor, evaluate, rates, _return_bounds=True)

        def stage_rows(z):
            row, bounds = stage_values(z)
            if not torch._C._functorch.is_functorch_wrapped_tensor(z):
                latest_bounds[0] = bounds
            return row

        declare_runtime_residual(stage_rows)
        stage_rows.runtime_values = stage_values
        stage_rows.accept_runtime_bounds = lambda bounds: latest_bounds.__setitem__(0, bounds)
        stage_rows.roundoff_tolerance = lambda: latest_bounds[0]
        stage_rows.linear_blocks = [
            block[j : j + 256]
            for b in evaluate.linear_blocks
            for block in [[stage * size + i for stage in range(2) for i in b]]
            for j in range(0, len(block), 256)
        ]
        # Only the already-adapted homogeneous graph has a complete branch
        # signature here. External material couplings retain fresh tangents.
        if self._reuse_step_jacobian and _coupling is None:
            stage_rows.linearization_history = self._stage_linearization
            stage_rows.linearization_key = (
                m.model_id,
                self.scheme,
                tuple(
                    (tuple(p["active"].tolist()), tuple(d["tags"])) for p, d in zip(problems, old)
                ),
                tuple(sorted(commands.items())),
                tuple(heat),
                tuple(t if q else None for q, t in zip(heat, utility)),
                tuple(sorted(losses.items())),
                tuple(sorted((drive_torque_nm or {}).items())),
                tuple(sorted((restriction_openings or {}).items())),
            )
            stage_rows.time_coefficient = dt
        else:
            self._stage_linearization.clear()

        if self.scheme == "sdirk2":
            # A is triangular: solve each full physical plant once per stage,
            # rather than factoring a doubled, mostly-zero stage matrix.
            # This is the same RK scheme, not operator/time splitting.
            solved, previous_rates, stage_points, physical_rows = [], [], [], []
            # Equal diagonal RK coefficients give the SAME derivative
            # operator in both stages: only the constant known RHS changes.
            # Keep one guarded tangent for this prepared problem only. A
            # quadrature/branch/command/time-step retry constructs a new one.
            stage_history = DenseNewtonHistory() if self._reuse_sdirk_tangent else None
            evaluations, residual = 0, 0.0
            for i in range(2):
                known = sum(
                    (interval_factor * self.a[i][j] * previous_rates[j] for j in range(i)),
                    initial.new_zeros(size),
                )

                def values(v, known=known, diagonal=self.a[i][i]):
                    row, point = evaluate(v)
                    bounds = point.get("roundoff_bounds", row.new_zeros(0))
                    return row - known - interval_factor * diagonal * rates(point), torch.cat(
                        (bounds, row.new_zeros(len(row) - len(bounds)))
                    )

                latest = [None]

                def single(v, values=values, latest=latest):
                    row, bounds = values(v)
                    if not torch._C._functorch.is_functorch_wrapped_tensor(v):
                        latest[0] = bounds
                    return row

                declare_runtime_residual(single)
                single.runtime_values = values
                single.accept_runtime_bounds = lambda bounds, latest=latest: latest.__setitem__(
                    0, bounds
                )
                single.roundoff_tolerance = lambda latest=latest: latest[0]
                single.linear_blocks = evaluate.linear_blocks
                if stage_history is not None:
                    single.linearization_history = stage_history
                    single.linearization_key = "sdirk-local-diagonal"
                    single.time_coefficient = dt * self.a[i][i]
                guess = solved[-1] if solved else initial
                if self._predict_sdirk_stage and i and f.device.type == "cpu":
                    # Extrapolate only a Newton GUESS, not the accepted RK
                    # state. Algebraic variables need not be linear in time:
                    # keep this trial only on strong ACTUAL residual decrease.
                    with torch.no_grad():
                        baseline = (
                            physical_rows[-1]
                            - known
                            - (interval_factor * self.a[i][i] * previous_rates[-1])
                        )
                        guess, status = _sdirk_guess(
                            single,
                            initial,
                            solved[0],
                            self.a[0][0],
                            baseline,
                            lo,
                            hi,
                            evaluate.maximum_update,
                        )
                        self._stage_predictor_work[status] += 1
                point, count, norm = self._solve(
                    single,
                    guess,
                    lo,
                    hi,
                    evaluate.pattern,
                    evaluate.maximum_update,
                )
                solved.append(point)
                with torch.no_grad():
                    physical_row, physical_point = evaluate(point)
                    physical_rows.append(physical_row)
                    previous_rates.append(rates(physical_point))
                    stage_points.append(physical_point)
                evaluations += count
                residual = max(residual, norm)
            stages = torch.cat(solved)
        else:
            stages, evaluations, residual = self._solve(
                stage_rows,
                initial.repeat(2),
                lo.repeat(2),
                hi.repeat(2),
                pattern,
                evaluate.maximum_update.repeat(2),
            )
        with torch.no_grad():
            points = (
                stage_points
                if self.scheme == "sdirk2" and self._reuse_sdirk_points
                else [evaluate(v)[1] for v in stages.reshape(2, size)]
            )
            updates = {}
            for z, point in zip(stages.reshape(2, size), points):
                updates.update(evaluate.pump_connections.check_valve_updates(z, point["data"]))
            if updates:
                self._stage_linearization.clear()
                if _prepared_problem is not None:
                    raise PRStageValveRetry(updates)
                if _valve_attempt >= 4:
                    raise PropertyModelError("PR stage check-valve active-set budget exhausted")
                new, report = self._step(
                    state,
                    dt,
                    commands,
                    heat,
                    utility,
                    losses,
                    _coupling=_coupling,
                    drive_torque_nm=drive_torque_nm,
                    restriction_openings=restriction_openings,
                    _pump_valve_states={**(_pump_valve_states or {}), **updates},
                    _valve_attempt=_valve_attempt + 1,
                )
                report["nonlinear_solves"] = self._solve_calls - solve_start
                return new, report
            integrated = dt * self._weighted([rates(point) for point in points])
            embedded_error = (
                self._embedded_error(
                    initial,
                    stages.reshape(2, size)[-1],
                    evaluate,
                    rates,
                    points,
                    dt,
                    initial_physical=initial_physical,
                )
                if self.error_control == "embedded"
                else None
            )
        if self.stiffly_accurate:
            final, count, final_residual = points[-1], 0, residual
        else:

            def endpoint_rows(v):
                return evaluate(v)[0] - integrated

            endpoint_rows.linear_blocks = evaluate.linear_blocks
            endpoint, count, final_residual = self._solve(
                endpoint_rows,
                stages.reshape(2, size)[-1],
                lo,
                hi,
                base,
                evaluate.maximum_update,
            )
            with torch.no_grad():
                final = evaluate(endpoint)[1]
        audited_points = points if self.stiffly_accurate else [*points, final]
        for point in audited_points:
            if not bool(point["valid"].cpu()) or not all(
                bool(d["q"]["valid"].all().cpu()) for d in point["data"]
            ):
                raise PropertyModelError("PR collocation stage outside physical flux/EOS domain")
            evaluate.compressors.validate(point["compressor_traces"])
            evaluate.pump_connections.validate(point["pump_connection_traces"])
        if any(
            post is not None
            for group in m._stable_all_many([point["data"] for point in audited_points])
            for post in group
        ):
            raise PropertyModelError("PR collocation phase topology changed")
        stage_couplings, coupling = [], None
        exchange_n = None
        exchange_e = exchange_s = 0.0
        if _coupling is not None:
            stage_couplings = [
                {**_coupling.accept(None, point), "closure": "stage_stirred_material_boundaries"}
                for point in points
            ]
            coupling = (
                {**stage_couplings[-1], "closure": "end_state_stirred_material_boundaries"}
                if self.stiffly_accurate
                else _coupling.accept(None, final)
            )
            exchange_n = (
                (
                    dt
                    * self._weighted(
                        [
                            sum(
                                (v[:-2] for v in point["exchange"]["net"]),
                                f.tensor([0.0] * len(f.names)),
                            )
                            for point in points
                        ]
                    )
                )
                .cpu()
                .tolist()
            )
            exchange_e = dt * self._weighted(
                [sum(float(v[-1].cpu()) for v in point["exchange"]["net"]) for point in points]
            )
            exchange_s = dt * self._weighted(
                [float(point["exchange"]["incoming_entropy"].cpu()) for point in points]
            )
        cells, local = [], []
        entropy = -exchange_s
        drive_records = {}
        drive_heat = 0.0
        work = 0.0
        for drive in m.drives:
            name = drive.machine
            powers = [point["drives"][name] for point in points]

            def integral(field, powers=powers):
                return dt * self._weighted([float(p[field].cpu()) for p in powers])

            electrical = integral("electrical_power_w")
            environmental = integral("heat_to_environment_w")
            work += electrical
            drive_heat -= environmental
            entropy += environmental / drive.ambient_temperature_k
            end_speed = float(final["drives"][name]["speed_rad_s"].cpu())
            kinetic_change = (
                0.5 * drive.drive.inertia_kg_m2 * (end_speed**2 - dict(state.rotors)[name] ** 2)
            )
            drive_records[name] = {
                "speed_rad_s": end_speed,
                "electrical_energy_j": electrical,
                "motor_work_j": integral("motor_power_w"),
                "shaft_work_j": integral("shaft_power_w"),
                "drag_heat_j": integral("drag_heat_w"),
                "motor_heat_j": integral("motor_heat_w"),
                "heat_to_environment_j": environmental,
                "heat_to_node_j": integral("heat_to_node_w"),
                "kinetic_energy_change_j": kinetic_change,
                # This difference belongs to the change of conserved energy
                # coordinates, not to an artificial fluid heat source.
                "kinetic_quadrature_difference_j": kinetic_change
                - integral("kinetic_energy_rate_w"),
            }
        for family in ("pumps", "pump_connections", "compressors"):
            for name in points[0][family]:
                if name not in m._drives:
                    work += dt * self._weighted(
                        [float(point[family][name]["shaft_power_w"].cpu()) for point in points]
                    )
        for i, (node, (_, previous), p, d) in enumerate(
            zip(m.nodes, state.nodes, problems, final["data"])
        ):
            rate = self._weighted([point["net"][i] for point in points])
            target_n = f.tensor(previous.component_moles) + dt * rate[:-2]
            fluid_wall_heat = dt * self._weighted([point["heat"][i] for point in points])
            ambient_heat = dt * self._weighted([point["ambient"][i] for point in points])
            target_energy = previous.total_energy_j + dt * (rate[-1] + heat[i]) + fluid_wall_heat
            for name, host in evaluate.rotor_hosts.items():
                if host == i:
                    target_energy = (
                        target_energy - drive_records[name]["kinetic_quadrature_difference_j"]
                    )
            tw = None
            wall_residual = 0.0
            if node.wall is not None:
                tw = float(final["walls"][i].cpu())
                for point in [*points, final]:
                    temperature = float(point["walls"][i].cpu())
                    if (
                        not node.wall.min_temperature_k
                        <= temperature
                        <= node.wall.max_temperature_k
                        or abs(temperature - previous.wall_temperature_k)
                        > node.wall.max_temperature_change_k
                    ):
                        raise PropertyModelError("PR Gauss wall temperature domain/change rejected")
                wall_change = node.wall.heat_capacity_j_k * (tw - previous.wall_temperature_k)
                wall_residual = wall_change - float((ambient_heat - fluid_wall_heat).cpu())
                entropy += (
                    node.wall.heat_capacity_j_k * math.log(tw / previous.wall_temperature_k)
                    - float(ambient_heat.cpu()) / node.wall.ambient_temperature_k
                )
            target_momentum = (
                previous.momentum_kg_m_s
                + dt * (rate[-2] + self._weighted([point["forces"][i] for point in points]))
                if node.kind == "pipe_cell"
                else d["p"] * 0
            )
            amounts = (
                d["amounts"]
                .new_zeros((len(d["amounts"]), len(f.names)))
                .index_copy(-1, p["active"], d["amounts"])
            )
            cell = PREquilibriumCellState(
                tuple(d["n"].cpu().tolist()),
                float(target_energy.cpu()),
                float(d["momentum"].cpu()) if node.kind == "pipe_cell" else 0.0,
                tuple(map(tuple, amounts.cpu().tolist())),
                tuple(m.cells[i]._tags(seeds[i])),
                float(d["p"].cpu()),
                float(d["t"].cpu()),
                tw,
            )
            cells.append((node.name, cell))
            local.append(
                {
                    "component_residuals_mol": (d["n"] - target_n).cpu().tolist(),
                    "energy_residual_j": float((d["energy"] - target_energy).cpu()),
                    "momentum_residual_kg_m_s": float((d["momentum"] - target_momentum).cpu()),
                    "wall_energy_residual_j": wall_residual,
                }
            )
            entropy += float((d["entropy"] - old[i]["entropy"]).cpu())
            if heat[i]:
                entropy -= dt * heat[i] / utility[i]
                if any(
                    heat[i] * (utility[i] - float(point["data"][i]["t"].cpu())) < -1e-10
                    for point in points
                ):
                    raise PropertyModelError("Heat source utility violates passive direction")
        if entropy < -1e-6 - 1e-10 * max(1.0, sum(abs(float(d["entropy"].cpu())) for d in old)):
            raise ConservationError("PR Gauss graph entropy rejected")
        new = PREquilibriumTransportState(
            tuple(cells),
            state.time_s + dt,
            state.next_step_s,
            m.model_id,
            tuple((name, record["speed_rad_s"]) for name, record in drive_records.items()),
        )
        m.inspect(new)
        ambient = dt * self._weighted(
            [sum(float(v.cpu()) for v in point["ambient"]) for point in points]
        )
        ambient += drive_heat
        ledger = m._ledger(
            state, new, dt * sum(heat) + ambient + exchange_e + work, incoming_n=exchange_n
        )
        PumpConnectionBundle.remember_valves(m, evaluate.pump_connections.valve_states)
        if machine_graph and self.error_control == "embedded" and _coupling is None:
            self._endpoint_rate_cache = (rate_key, new.nodes, new.rotors, final)
        return new, {
            "state": new.to_dict(),
            "method": self.scheme,
            "embedded_error_ratio": embedded_error,
            "nonlinear_solves": self._solve_calls - solve_start,
            "initialization_nonlinear_solves": initialization_solves,
            "pump_check_valve_retries": _valve_attempt,
            **(
                {
                    "stage_couplings": stage_couplings,
                    "coupling": coupling,
                    "exchange_component_moles": exchange_n,
                    "exchange_energy_j": exchange_e,
                }
                if _coupling is not None
                else {}
            ),
            "start_time_s": state.time_s,
            "end_time_s": new.time_s,
            "dt_s": dt,
            "conservation_residuals": ledger,
            "local_ledgers": local,
            "entropy_production_j_k": entropy,
            "external_heat_j": dt * sum(heat),
            "ambient_heat_j": ambient,
            "external_work_j": work,
            "drives": drive_records,
            "stage_rotors": [
                {
                    name: float(record["speed_rad_s"].cpu())
                    for name, record in point["drives"].items()
                }
                for point in points
            ],
            "stage_equipment": [
                {
                    "pumps": serialize_pumps(point["pumps"], commands),
                    "compressors": evaluate.compressors.serialize(point["compressors"]),
                    "pump_connections": evaluate.pump_connections.serialize(
                        point["pump_connections"]
                    ),
                    "restrictions": evaluate.restrictions.serialize(point["restrictions"]),
                }
                for point in points
            ],
            "max_scaled_residual": max(residual, final_residual),
            "nonlinear_evaluations": evaluations + count,
            "stage_states": [
                {
                    node.name: {
                        "component_moles": d["n"].cpu().tolist(),
                        "total_energy_j": float(d["energy"].cpu()),
                        "wall_temperature_k": float(point["walls"][i].cpu())
                        if node.wall is not None
                        else None,
                        "phase_component_moles": d["amounts"]
                        .new_zeros((len(d["amounts"]), len(f.names)))
                        .index_copy(-1, problems[i]["active"], d["amounts"])
                        .cpu()
                        .tolist(),
                        "phase_roots": list(d["tags"]),
                        "momentum_kg_m_s": float(d["momentum"].cpu()),
                        "velocity_m_s": float(d["u"].cpu()),
                        "characteristic_speeds_m_s": d["characteristic_speeds"].cpu().tolist(),
                        "pressure_pa": float(d["p"].cpu()),
                        "temperature_k": float(d["t"].cpu()),
                        "phase_count": len(d["phase_n"]),
                    }
                    for i, (node, d) in enumerate(zip(m.nodes, point["data"]))
                }
                for point in points
            ],
            "links": [
                {
                    "name": link.name,
                    "integrated_flux": (
                        dt * self._weighted([point["links"][j]["flux"] for point in points])
                    )
                    .cpu()
                    .tolist(),
                }
                for j, link in enumerate(m.links)
            ],
            "wall_exchanges": {
                node.name: {
                    "fluid_heat_j": dt
                    * self._weighted([float(point["heat"][i].cpu()) for point in points]),
                    "ambient_heat_j": dt
                    * self._weighted([float(point["ambient"][i].cpu()) for point in points]),
                    "wall_energy_change_j": node.wall.heat_capacity_j_k
                    * (float(final["walls"][i].cpu()) - state.nodes[i][1].wall_temperature_k),
                }
                for i, node in enumerate(m.nodes)
                if node.wall is not None
            },
            "heat_links": {
                link.name: dt
                * self._weighted(
                    [float(point["heat_links"][link.name]["heat_rate_w"].cpu()) for point in points]
                )
                for link in m.heat_links
            },
        }

    @cached_fluid_transaction(lambda self: (self.physics.fluid,))
    def advance(
        self,
        state,
        duration_s,
        *,
        openings=None,
        heat_w=None,
        utility_temperature_k=None,
        loss_coefficients=None,
        drive_torque_nm=None,
        restriction_openings=None,
        _coupling_factory=None,
    ):
        m = self.physics
        m.inspect(state)
        end = state.time_s + _number(duration_s, "duration_s", positive=True)
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Unrepresentable PR collocation interval")
        commands = dict(m.default_openings)
        if openings is not None:
            if set(openings) - set(commands) or any(type(v) is not bool for v in openings.values()):
                raise ValueError("Known binary openings required")
            commands.update(openings)
        heat_w, utility_temperature_k = dict(heat_w or {}), dict(utility_temperature_k or {})
        if set(heat_w) - set(m.names) or set(utility_temperature_k) - set(m.names):
            raise ValueError("Unknown thermal node")
        heat = [_number(heat_w.get(n, 0.0), "heat rate") for n in m.names]
        utility = [
            _number(
                utility_temperature_k.get(n, state.nodes[i][1].temperature_k),
                "utility temperature",
                positive=True,
            )
            for i, n in enumerate(m.names)
        ]
        if any(q and n not in utility_temperature_k for n, q in zip(m.names, heat)):
            raise ValueError("Nonzero heat requires explicit utility temperature")
        losses = {n.name: n.local_loss_coefficient for n in m.nodes if n.kind == "pipe_cell"}
        if loss_coefficients is not None:
            if set(loss_coefficients) - set(losses):
                raise ValueError("Unknown pipe loss command")
            losses.update(
                {
                    n: _number(v, "loss coefficient", nonnegative=True)
                    for n, v in loss_coefficients.items()
                }
            )
        solve_start = self._solve_calls
        drive_torque_nm = dict(drive_torque_nm or {})
        restriction_openings = dict(restriction_openings or {})
        if set(drive_torque_nm) - set(m._drives):
            raise ValueError("Unknown driven machine torque command")
        for name, value in drive_torque_nm.items():
            value = _number(value, "motor torque")
            drive = m._drives[name].drive
            if not -drive.max_braking_torque_nm <= value <= drive.max_motor_torque_nm:
                raise ValueError("Motor torque command outside limits")
        if set(restriction_openings) - {r.name for r in m.restrictions}:
            raise ValueError("Unknown restriction opening command")
        for value in restriction_openings.values():
            if not 0 <= _number(value, "restriction opening") <= 1:
                raise ValueError("Restriction opening outside [0,1]")
        linear_start = dict(self._linear_work)
        history_start = tuple(
            getattr(self._stage_linearization, key) for key in ("hits", "refreshes", "builds")
        )
        current, steps, attempts, rejected, fallbacks = state, [], 0, 0, 0
        dt = state.next_step_s
        while current.time_s < end:
            attempts += 1
            if attempts > m.maximum_attempts:
                raise HydraulicConvergenceError("PR collocation interval attempt budget exhausted")
            proposed = min(dt, m.maximum_step_s)
            step = interval_step_s(current.time_s, end, proposed, maximum_s=m.maximum_step_s)
            reached = current.time_s + step
            if reached < end:
                try:
                    split_step_s(reached, end - reached)
                except ValueError:
                    # At large epochs a capped step can leave one clock ULP.
                    # Borrow one ULP before solving, rather than skipping it or
                    # exceeding the numerical maximum step.
                    step = math.nextafter(reached, current.time_s) - current.time_s
            halves = split_step_s(current.time_s, step)
            method, fallback_reason = self.scheme, None

            def take_step(stepper, source, length):
                options = (
                    {}
                    if _coupling_factory is None
                    else {"_coupling": _coupling_factory(source, length, commands)}
                )
                if drive_torque_nm:
                    options["drive_torque_nm"] = drive_torque_nm
                if restriction_openings:
                    options["restriction_openings"] = restriction_openings
                return stepper(source, length, commands, heat, utility, losses, **options)

            try:
                try:
                    if self.error_control == "embedded":
                        fine, one = take_step(self._step, current, step)
                        error = one["embedded_error_ratio"]
                        accepted_records = (one,)
                        order = 1 if self.scheme == "sdirk2" else 2
                    else:
                        coarse, _ = take_step(self._step, current, step)
                        half, one = take_step(self._step, current, halves[0])
                        fine, two = take_step(self._step, half, halves[1])
                        error = m._error(coarse, fine)
                        order = self.order
                        accepted_records = (one, two)
                except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
                    self._stage_linearization.clear()
                    if not self.phase_fallback:
                        raise
                    fallback_reason = str(exc)
                    method = "backward_euler_fallback"
                    coarse, _ = take_step(m._step, current, step)
                    half, one = take_step(m._step, current, halves[0])
                    fine, two = take_step(m._step, half, halves[1])
                    error, order = m._error(coarse, fine), 1
                    accepted_records = (one, two)
                if not math.isfinite(error) or error > 1:
                    raise HydraulicConvergenceError("PR collocation temporal error exceeded")
            except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
                self._stage_linearization.clear()
                rejected += 1
                dt = step / 2
                if dt < m.minimum_step_s:
                    raise HydraulicConvergenceError(
                        "PR collocation minimum step exhausted"
                    ) from exc
                continue
            # An embedded estimate checks the actual candidate every time;
            # a tiny initial proposal need not force a long doubling ladder.
            # Keep the legacy bound for step doubling and phase fallback.
            growth_limit = 5.0 if self.error_control == "embedded" and method == "radau3" else 2.0
            factor = min(growth_limit, max(0.5, 0.9 * max(error, 1e-16) ** (-1 / (order + 1))))
            dt = min(m.maximum_step_s, max(m.minimum_step_s, step * factor))
            if step < proposed and (
                step < m.minimum_step_s or error <= (0.9 * step / proposed) ** (order + 1)
            ):
                # A report-only remainder below the configured minimum is
                # integrated and audited, but cannot provide a useful next
                # interval proposal. That interval still estimates its error
                # normally and rejects/shrinks the proposal when necessary.
                dt = max(dt, proposed)
            current = replace(
                fine, time_s=end if step == end - current.time_s else fine.time_s, next_step_s=dt
            )
            for record in accepted_records:
                record.update(method=method, error_ratio=error)
                if fallback_reason:
                    record["fallback_reason"] = fallback_reason
            fallbacks += method == "backward_euler_fallback"
            steps.extend(accepted_records)
        total_heat = math.fsum(s["external_heat_j"] + s["ambient_heat_j"] for s in steps)
        total_work = math.fsum(s.get("external_work_j", 0.0) for s in steps)
        incoming_n, incoming_e = None, 0.0
        if _coupling_factory is not None:
            incoming_n = [
                math.fsum(s["exchange_component_moles"][j] for s in steps)
                for j in range(len(m.fluid.names))
            ]
            incoming_e = math.fsum(s["exchange_energy_j"] for s in steps)
        return {
            "state": current.to_dict(),
            "steps": steps,
            "attempts": attempts,
            "rejected_attempts": rejected,
            "collocation_nonlinear_solves": self._solve_calls - solve_start,
            "cross_step_linearization": {
                key: getattr(self._stage_linearization, key) - old
                for key, old in zip(("hits", "refreshes", "builds"), history_start)
            },
            "linear_work": {
                key: value - linear_start[key] for key, value in self._linear_work.items()
            },
            "fallback_intervals": fallbacks,
            "error_control": self.error_control,
            "conservation_residuals": m._ledger(
                state, current, total_heat + total_work + incoming_e, incoming_n=incoming_n
            ),
            "total_heat_j": total_heat,
            "total_work_j": total_work,
            "openings": commands,
            "loss_coefficients": losses,
            "drive_torque_nm": drive_torque_nm,
            "restriction_openings": restriction_openings,
            "entropy_production_j_k": math.fsum(s["entropy_production_j_k"] for s in steps),
        }
