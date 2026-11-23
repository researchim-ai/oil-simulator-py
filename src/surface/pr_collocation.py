"""Conservative Gauss4/Radau3 for homogeneous-equilibrium PR graphs.

Reuse the existing PR chemical/volume constraints, HLLC flux and force
operators. Phase transitions use the existing backward-Euler branch search,
with the actual method explicitly recorded; no extrapolated phase states.
"""

import math
from dataclasses import replace

import torch

from .cuda_execution import current_execution
from .equilibrium_cache import cached_fluid_transaction
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .pr_equilibrium_transport import (
    PREquilibriumCellState,
    PREquilibriumTransportNetwork,
    PREquilibriumTransportState,
)
from .time_interval import interval_step_s, split_step_s
from .torch_krylov import gmres_device
from .torch_linear import BlockJacobi, ColoredJacobian, TorchCOO
from .torch_nonlinear import TorchNewton


class PRCollocation:
    _batch_stage_residual = True
    _block_preconditioning = True

    def __init__(
        self, physics, *, phase_fallback=True, scheme="gauss4", error_control="step_doubling"
    ):
        if not isinstance(physics, PREquilibriumTransportNetwork):
            raise TypeError("Expected PREquilibriumTransportNetwork")
        if type(phase_fallback) is not bool:
            raise ValueError("phase_fallback must be boolean")
        if (
            physics.pumps
            or physics.compressors
            or physics.pump_connections
            or physics.restrictions
            or physics.drives
        ):
            raise ValueError(
                "Machine/restriction inventories require a separate collocation adapter"
            )
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
        if scheme not in ("gauss4", "radau3"):
            raise ValueError("Unknown PR collocation scheme")
        if error_control not in ("step_doubling", "embedded") or (
            error_control == "embedded" and scheme != "radau3"
        ):
            raise ValueError("Embedded control requires Radau3")
        self.error_control = error_control
        self._solve_calls = 0
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
        self.order = 4 if scheme == "gauss4" else 3
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

    def _stage_rows(self, z, size, dt, evaluate, rates):
        def parts(v):
            row, physical = evaluate(v)
            return row, rates(physical)

        values = z.reshape(2, size)
        if self._batch_stage_residual:
            try:
                rows, rhs = torch.vmap(parts)(values)
            except RuntimeError:
                # Unsupported transform/backend: retain the original operator
                # and its failure diagnosis. Never substitute a valid residual.
                result = [parts(v) for v in values]
                rows, rhs = zip(*result)
        else:
            result = [parts(v) for v in values]
            rows, rhs = zip(*result)
        return torch.cat(
            [rows[i] - dt * sum(self.a[i][j] * rhs[j] for j in range(2)) for i in range(2)]
        )

    def _problem(self, state, commands, heat, losses, _coupling=None):
        m = self.physics
        seeds = [m._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
        y, lo, hi, base_evaluate, offsets, problems, old = m._problem(
            state,
            seeds,
            0.0,
            commands,
            heat,
            losses,
            active_override=None if _coupling is None else _coupling.active,
            guesses_override=None if _coupling is None else _coupling.ordinary_guesses,
        )
        if offsets[-1] != len(y):
            raise ValueError("Additional algebraic bundle is not adapted to PR collocation")
        body_size = len(y)
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
        if 2 * len(y) > m.maximum_unknowns:
            raise ValueError("PR collocation exceeds configured unknown budget")
        old_wall_temperatures = f.tensor(
            [state.nodes[i][1].wall_temperature_k for i in wall_indices]
        )

        def evaluate(z):
            row, physical = base_evaluate(
                z[:body_size], external_fluxes=None if _coupling is None else _coupling.exchange
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
        evaluate.linear_blocks = self._physical_blocks(offsets, wall_indices, len(y))
        pattern = set(zip(*m._sparsity(offsets, commands)))
        for j, i in enumerate(wall_indices):
            index = body_size + j
            cell = range(offsets[i], offsets[i + 1])
            pattern.add((index, index))
            pattern.update((index, c) for c in cell)
            pattern.update((r, index) for r in cell)
        pattern = sorted(pattern)
        evaluate.pattern = ([r for r, c in pattern], [c for r, c in pattern])

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
            rows.extend(
                ((physical["ambient"][i] - physical["heat"][i]) / scale[i]).reshape(1)
                for i in wall_indices
            )
            return torch.cat(rows)

        return y, lo, hi, evaluate, rates, offsets, problems, old, seeds

    def _solve(self, fn, initial, lo, hi, pattern, update):
        self._solve_calls += 1
        m, f = self.physics, self.physics.fluid
        solver = TorchNewton(f.device, maximum_iterations=40, dense_limit=m.dense_limit)
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
                maximum_update=update,
                blocks=getattr(fn, "linear_blocks", None) if self._block_preconditioning else None,
            )
            # TorchNewton returns the final residual at exactly y, including
            # eager acceptance under native execution. Do not evaluate both
            # complete PR stages a second time just to obtain the same row.
        if not bool((ok & torch.isfinite(row).all() & (row.abs().max() < 1e-9)).cpu()):
            raise HydraulicConvergenceError("PR Gauss nonlinear/constraint solve rejected")
        return y, evaluations, float(row.abs().max().cpu())

    def _embedded_error(self, initial, endpoint, evaluate, rates, points, dt):
        """Second-order companion, stabilized by an implicit Euler tangent.

        Companion weights at c=(0,1/3,1) are (1/6,1/2,1/3).
        Its defect from Radau3 is h*(f0-1.5*f1+0.5*f2)/6. Solve
        (dU/dy-h*df/dy)*delta=defect on the SAME algebraic manifold;
        no extra nonlinear integration or phase-state extrapolation is used.
        """
        m, f = self.physics, self.physics.fluid
        h = f.tensor(dt)
        _, old = evaluate(initial)
        defect = h * (rates(old) - 1.5 * rates(points[0]) + 0.5 * rates(points[1])) / 6

        def filtered(y):
            row, physical = evaluate(y)
            return row - h * rates(physical)

        filtered.runtime_residual = filtered
        execution = current_execution()
        accelerated = execution is not None and execution.eligible(endpoint)
        if endpoint.numel() <= m.dense_limit:
            matrix = (
                execution.jacobian(filtered, endpoint)
                if accelerated
                else torch.func.jacfwd(filtered)(endpoint)
            )
            delta, info = torch.linalg.solve_ex(matrix, defect, check_errors=False)
            valid = (info == 0) & torch.isfinite(delta).all()
            residual = matrix @ delta - defect
        else:
            coloring = ColoredJacobian(endpoint.numel(), *evaluate.pattern, f.device)
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
        return float(torch.stack(errors).max().cpu())

    def _step(self, state, dt, commands, heat, utility, losses, *, _coupling=None):
        m, f = self.physics, self.physics.fluid
        initial, lo, hi, evaluate, rates, offsets, problems, old, seeds = self._problem(
            state, commands, heat, losses, _coupling
        )
        size = len(initial)
        interval_factor = f.tensor(dt)
        base = evaluate.pattern
        pattern = None
        if 2 * size > m.dense_limit:
            entries = sorted(
                {
                    (i * size + r, j * size + c)
                    for r, c in zip(*base)
                    for i in range(2)
                    for j in range(2)
                }
            )
            pattern = ([r for r, c in entries], [c for r, c in entries])

        def stage_rows(z):
            return self._stage_rows(z, size, interval_factor, evaluate, rates)

        stage_rows.runtime_residual = stage_rows
        stage_rows.linear_blocks = self._physical_blocks(offsets, evaluate.wall_indices, size, 2)

        stages, evaluations, residual = self._solve(
            stage_rows,
            initial.repeat(2),
            lo.repeat(2),
            hi.repeat(2),
            pattern,
            evaluate.maximum_update.repeat(2),
        )
        with torch.no_grad():
            points = [evaluate(v)[1] for v in stages.reshape(2, size)]
            integrated = dt * self._weighted([rates(point) for point in points])
            embedded_error = (
                self._embedded_error(
                    initial, stages.reshape(2, size)[-1], evaluate, rates, points, dt
                )
                if self.error_control == "embedded"
                else None
            )
        if self.scheme == "radau3":
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
                base if size > m.dense_limit else None,
                evaluate.maximum_update,
            )
            with torch.no_grad():
                final = evaluate(endpoint)[1]
        audited_points = points if self.scheme == "radau3" else [*points, final]
        for point in audited_points:
            if not bool(point["valid"].cpu()) or not all(
                bool(d["q"]["valid"].all().cpu()) for d in point["data"]
            ):
                raise PropertyModelError("PR collocation stage outside physical flux/EOS domain")
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
                if self.scheme == "radau3"
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
        for i, (node, (_, previous), p, d) in enumerate(
            zip(m.nodes, state.nodes, problems, final["data"])
        ):
            rate = self._weighted([point["net"][i] for point in points])
            target_n = f.tensor(previous.component_moles) + dt * rate[:-2]
            fluid_wall_heat = dt * self._weighted([point["heat"][i] for point in points])
            ambient_heat = dt * self._weighted([point["ambient"][i] for point in points])
            target_energy = previous.total_energy_j + dt * (rate[-1] + heat[i]) + fluid_wall_heat
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
            tuple(cells), state.time_s + dt, state.next_step_s, m.model_id
        )
        m.inspect(new)
        ambient = dt * self._weighted(
            [sum(float(v.cpu()) for v in point["ambient"]) for point in points]
        )
        ledger = m._ledger(state, new, dt * sum(heat) + ambient + exchange_e, incoming_n=exchange_n)
        return new, {
            "state": new.to_dict(),
            "method": self.scheme,
            "embedded_error_ratio": embedded_error,
            "nonlinear_solves": 1 if self.scheme == "radau3" else 2,
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
                return stepper(source, length, commands, heat, utility, losses, **options)

            try:
                try:
                    if self.error_control == "embedded":
                        fine, one = take_step(self._step, current, step)
                        error = one["embedded_error_ratio"]
                        accepted_records = (one,)
                        order = 2
                    else:
                        coarse, _ = take_step(self._step, current, step)
                        half, one = take_step(self._step, current, halves[0])
                        fine, two = take_step(self._step, half, halves[1])
                        error = m._error(coarse, fine)
                        order = self.order
                        accepted_records = (one, two)
                except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
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
            "fallback_intervals": fallbacks,
            "error_control": self.error_control,
            "conservation_residuals": m._ledger(
                state, current, total_heat + incoming_e, incoming_n=incoming_n
            ),
            "total_heat_j": total_heat,
            "total_work_j": 0.0,
            "openings": commands,
            "loss_coefficients": losses,
            "entropy_production_j_k": math.fsum(s["entropy_production_j_k"] for s in steps),
        }
