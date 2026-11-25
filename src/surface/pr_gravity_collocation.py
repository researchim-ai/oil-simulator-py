"""Common stiffly accurate Radau stages for PR pipes, machines and columns.

The existing gravitational aperture operator supplies shared component,
energy and entropy fluxes. Spatial quadrature remains independently refined.
Controller states are not implicitly added by this plant-only adapter.
"""

import math
from dataclasses import replace

import torch

from .equilibrium_cache import cached_fluid_transaction
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .pr_collocation import PRCollocation, PRStageValveRetry
from .pr_gravity_transport import (
    PRGravityTransportNetwork,
    PRGravityTransportState,
    _ColumnCoupling,
)
from .pr_pump_connection import PumpConnectionBundle
from .stratified_pr import StratifiedPRState
from .time_interval import interval_step_s, split_step_s


class _StageColumns:
    def __init__(self, coupling):
        self.coupling = coupling
        self.final = None

    def accept(self, _y, point):
        c = self.coupling
        model, f = c.model, c.model.transport.fluid
        columns, reports = {}, {}
        for name, body in point["column_bodies"].items():
            column = model.columns[name]
            active = c.problems[name][4]
            n = body["component_moles"]
            energy = (
                body["internal_energy_j"]
                + body["relative_potential_energy_j"]
                + (n * f.molar_masses).sum() * column.gravity_m_s2 * column.elevation_m
            )
            seed = (
                [x[0] for x in body["compositions"]],
                c.seeds[name][1],
                body["interfaces"],
                float(body["pressures"][0][0].cpu()),
                float(body["temperature"].cpu()),
                torch.zeros_like(n).index_copy(0, active, body["constants"]),
            )
            state, report = column._accept(
                n, float(energy.cpu()), seed, c.order, point["column_variables"][name]
            )
            columns[name], reports[name] = state.to_dict(), report
        # Same local stability and 1-D transverse-head restriction as BE.
        checks = {name: ([], [], []) for name in model.columns}
        for port, _root, _height, weights, pp, xx, q in point["exchange"]["pieces"]:
            column = model.columns[port.column]
            active = weights > 0
            if bool(
                (
                    active
                    & (
                        q["density_kg_m3"] * column.gravity_m_s2 * port.diameter_m / pp
                        > model.maximum_transverse_head_fraction
                    )
                )
                .any()
                .cpu()
            ):
                raise PropertyModelError(
                    "Aperture requires resolved transverse pipe inlet hydrodynamics"
                )
            if bool(active.any().cpu()):
                pressures, compositions, properties = checks[port.column]
                pressures.append(pp[active])
                compositions.append(xx[active])
                properties.append({k: v[active] for k, v in q.items()})
        for name, (pressures, compositions, properties) in checks.items():
            model.columns[name]._check_locals(
                pressures, columns[name]["temperature_k"], compositions, properties
            )
        walls = {}
        for name, wall in model.walls.items():
            temperature = float(point["column_walls"][name].cpu())
            if (
                not wall.min_temperature_k <= temperature <= wall.max_temperature_k
                or abs(temperature - dict(c.state.walls)[name]) > wall.max_temperature_change_k
            ):
                raise PropertyModelError("Radau column wall domain/change rejected")
            walls[name] = temperature
        self.final = point
        return {
            "columns": columns,
            "reports": reports,
            "walls": walls,
            "column_entropy_rate": {n: float(v.cpu()) for n, v in point["exchange"]["cs"].items()},
            "column_ambient_w": {n: float(v.cpu()) for n, v in point["column_ambient"].items()},
            "heat_link_entropy_rate": sum(
                float(row["entropy_production_w_k"].cpu())
                for row in point["exchange"]["heat_links"].values()
            ),
            **(
                {"controls": {k: v.detach().cpu().tolist() for k, v in point["controls"].items()}}
                if "controls" in point
                else {}
            ),
            "ports": {
                name: {
                    "flux": p["flux"].cpu().tolist(),
                    "entropy_flux": float(p["entropy_flux"].cpu()),
                    "area_fractions": [float(v.cpu()) for v in p["area_fractions"]],
                }
                for name, p in point["exchange"]["ports"].items()
            },
        }


class PRGravityCollocation:
    def __init__(self, physics, *, phase_fallback=True, scheme="radau3"):
        if not isinstance(physics, PRGravityTransportNetwork):
            raise TypeError("Expected PRGravityTransportNetwork")
        if physics.darcy_connections:
            raise ValueError("Darcy material traces require a continuous-stage adapter")
        if type(phase_fallback) is not bool:
            raise ValueError("phase_fallback must be boolean")
        if scheme not in ("radau3", "sdirk2"):
            raise ValueError("A stiffly accurate gravitational scheme is required")
        self.physics, self.phase_fallback = physics, phase_fallback
        self.engine = PRCollocation(
            physics.transport,
            scheme=scheme,
            error_control="embedded",
            phase_fallback=False,
        )
        self._endpoints = {}

    def _problem(
        self,
        state,
        order,
        openings,
        heat,
        utility,
        losses,
        *,
        drive_torque_nm=None,
        restriction_openings=None,
        _controls=None,
        _pump_valve_states=None,
    ):
        m, t, engine = self.physics, self.physics.transport, self.engine
        f = t.fluid
        coupling = _ColumnCoupling(m, state, 0.0, order, openings, heat, utility)
        commands = {c.name: openings[c.name] for c in t.switches}
        base = engine._problem(
            state.transport,
            commands,
            [heat.get(n, 0.0) for n in t.names],
            losses,
            drive_torque_nm=drive_torque_nm,
            restriction_openings=restriction_openings,
            _boundary=coupling,
            _pump_valve_states=_pump_valve_states,
        )
        y, lo, hi, ordinary, ordinary_rates, offsets, problems, old, seeds = base
        y, lo, hi, combined, pattern = coupling.prepare(
            y,
            lo,
            hi,
            ordinary,
            offsets,
            0.0,
            ordinary.pattern,
        )
        body_size = len(y)
        wall_names = tuple(m.walls)
        wall_scales = {
            name: max(1.0, m.walls[name].heat_capacity_j_k * f.tref) for name in wall_names
        }
        if wall_names:
            old_walls = dict(state.walls)
            y = torch.cat((y, f.tensor([old_walls[n] / f.tref for n in wall_names])))
            lo = torch.cat(
                (lo, f.tensor([m.walls[n].min_temperature_k / f.tref for n in wall_names]))
            )
            hi = torch.cat(
                (hi, f.tensor([m.walls[n].max_temperature_k / f.tref for n in wall_names]))
            )
        if (1 if engine.scheme == "sdirk2" else 2) * len(y) > t.maximum_unknowns:
            raise ValueError("Common gravitational Radau exceeds configured unknown budget")

        def evaluate(z, runtime_controls=None):
            row, point = combined(z[:body_size], runtime_controls=runtime_controls)
            walls, heats, ambient = (
                dict(point[k]) for k in ("column_walls", "column_heat", "column_ambient")
            )
            extra = []
            for j, name in enumerate(wall_names):
                wall = m.walls[name]
                tw = z[body_size + j] * f.tref
                walls[name] = tw
                heats[name] = wall.fluid_ua_w_k * (tw - point["column_bodies"][name]["temperature"])
                ambient[name] = wall.ambient_ua_w_k * (wall.ambient_temperature_k - tw)
                extra.append(
                    (wall.heat_capacity_j_k * (tw - old_walls[name]) / wall_scales[name]).reshape(1)
                )
            return torch.cat((row, *extra)), {
                **point,
                "column_walls": walls,
                "column_heat": heats,
                "column_ambient": ambient,
                "column_variables": {name: z[a:b] for name, (a, b) in coupling.blocks.items()},
            }

        def rates(point):
            parts = [ordinary_rates(point)]
            for name in coupling.blocks:
                column = m.columns[name]
                a, b = coupling.blocks[name]
                active = coupling.problems[name][4]
                count = len(active)
                nscale, escale = coupling.scales[name]
                net = point["exchange"]["cn"][name]
                energy = (
                    point["exchange"]["ce"][name]
                    - (net * f.molar_masses).sum() * column.gravity_m_s2 * column.elevation_m
                )
                parts.append(
                    torch.cat(
                        (
                            y.new_zeros(b - a - count - 1),
                            net[active] / nscale,
                            (
                                (energy + point["column_heat"][name] + heat.get(name, 0.0)) / escale
                            ).reshape(1),
                        )
                    )
                )
            port_end = sum(len(p) for p in parts)
            parts.append(y.new_zeros(body_size - port_end))
            parts.extend(
                ((point["column_ambient"][n] - point["column_heat"][n]) / wall_scales[n]).reshape(1)
                for n in wall_names
            )
            return torch.cat(parts)

        # prepare() returns None below the dense threshold; for the embedded
        # tangent retain exact support rather than constructing a dense mask.
        if pattern is None:
            # Rebuild support through the existing sparse planner, independent
            # of numeric solves. Its policy is local to this call.
            pattern = getattr(combined, "continuous_pattern", None)
        if pattern is None:
            raise ValueError("Gravitational stage operator did not publish its sparsity")
        support = set(zip(*pattern))
        for j, name in enumerate(wall_names):
            index = body_size + j
            a, b = coupling.blocks[name]
            support.add((index, index))
            support.update((index, c) for c in range(a, b))
            support.add((b - 1, index))
        support = sorted(support)
        evaluate.pattern = ([r for r, c in support], [c for r, c in support])
        evaluate.maximum_update = torch.cat(
            (combined.maximum_update, f.tensor([math.inf] * len(wall_names)))
        )
        evaluate.linear_blocks = [list(b) for b in combined.linear_blocks]
        evaluate.linear_blocks.extend([[body_size + j] for j in range(len(wall_names))])
        for key in ("rotors", "rotor_hosts", "compressors", "pump_connections", "restrictions"):
            setattr(evaluate, key, getattr(ordinary, key))
        evaluate.requires_initial_projection = True

        def additional_error(a, b):
            errors = []
            for name, column in m.columns.items():
                x, v = a["column_bodies"][name], b["column_bodies"][name]
                errors.extend(
                    (
                        (
                            (x["component_moles"] - v["component_moles"]).abs()
                            / (
                                1e-10
                                + t.relative_tolerance
                                * torch.maximum(
                                    x["component_moles"].abs(), v["component_moles"].abs()
                                )
                            )
                        ).max(),
                        (x["temperature"] - v["temperature"]).abs() / t.temperature_tolerance_k,
                        (x["interfaces"] - v["interfaces"]).abs().max()
                        / (1e-8 + column.relative_tolerance * column.geometry.height_m)
                        if len(x["interfaces"])
                        else y.new_zeros(()),
                    )
                )
            errors.extend(
                (a["column_walls"][n] - b["column_walls"][n]).abs() / t.temperature_tolerance_k
                for n in wall_names
            )
            return torch.stack(errors).max() if errors else y.new_zeros(())

        evaluate.additional_error = additional_error
        prepared = (y, lo, hi, evaluate, rates, offsets, problems, old, seeds)
        if _controls is not None:
            prepared = _controls.decorate(prepared, coupling)
        return prepared, coupling

    def _step_order(self, state, dt, order, openings, heat, utility, losses, **commands):
        m, t, engine = self.physics, self.physics.transport, self.engine
        key = (
            order,
            tuple(sorted(openings.items())),
            tuple(sorted(heat.items())),
            tuple(sorted(utility.items())),
            tuple(sorted(losses.items())),
            tuple(sorted((commands.get("drive_torque_nm") or {}).items())),
            tuple(sorted((commands.get("restriction_openings") or {}).items())),
        )
        initial = None
        saved = self._endpoints.pop(order, None)
        if (
            saved is not None
            and saved[1] == key
            and saved[0].transport.nodes == state.transport.nodes
            and saved[0].transport.rotors == state.transport.rotors
            and saved[0].columns == state.columns
            and saved[0].walls == state.walls
        ):
            initial = saved[2]
        valve_states = PumpConnectionBundle.previous_valves(t) or {}
        for retry in range(5):
            prepared, coupling = self._problem(
                state,
                order,
                openings,
                heat,
                utility,
                losses,
                _pump_valve_states=valve_states,
                **commands,
            )
            accepted = _StageColumns(coupling)
            try:
                transport, report = engine._step(
                    state.transport,
                    dt,
                    {c.name: openings[c.name] for c in t.switches},
                    [heat.get(n, 0.0) for n in t.names],
                    [
                        utility.get(n, s.temperature_k)
                        for n, (_, s) in zip(t.names, state.transport.nodes)
                    ],
                    losses,
                    _coupling=accepted,
                    _prepared_problem=prepared,
                    _initial_physical=initial if retry == 0 else None,
                    **{k: v for k, v in commands.items() if k != "_controls"},
                )
                report["pump_check_valve_retries"] = retry
                break
            except PRStageValveRetry as exc:
                valve_states.update(exc.updates)
        else:
            raise PropertyModelError("Common stage check-valve active-set budget exhausted")
        cr = report["coupling"]
        new = PRGravityTransportState(
            transport,
            tuple((n, StratifiedPRState.from_dict(s)) for n, s in sorted(cr["columns"].items())),
            tuple(sorted(cr["walls"].items())),
            m.model_id,
        )
        stages = report["stage_couplings"]
        # Stage snapshots retain the flux/entropy powers needed for an audit.
        # The final snapshot is also the exact starting operator for the next
        # interval when all physical states and commands match.
        point = accepted.final
        ports = {}
        for name in cr["ports"]:
            ports[name] = {
                "flux": [
                    engine._weighted([s["ports"][name]["flux"][j] for s in stages])
                    for j in range(len(cr["ports"][name]["flux"]))
                ],
                "integrated_flux": [
                    dt * engine._weighted([s["ports"][name]["flux"][j] for s in stages])
                    for j in range(len(cr["ports"][name]["flux"]))
                ],
                "area_fractions": cr["ports"][name]["area_fractions"],
            }
        # Column entropy fluxes and wall/utility powers are serialized in the
        # stage snapshots, without re-evaluating a separate time integration.
        heat_columns = entropy_columns = 0.0
        for name, previous in state.columns:
            column_report = cr["reports"][name]
            entropy_columns += column_report["entropy_j_k"] - coupling.old_entropy[name]
            entropy_columns -= dt * engine._weighted(
                [s["column_entropy_rate"][name] for s in stages]
            )
            q = heat.get(name, 0.0)
            heat_columns += dt * q
            if q:
                if any(
                    q * (utility[name] - s["columns"][name]["temperature_k"]) < -1e-10
                    for s in stages
                ):
                    raise PropertyModelError("Column heat violates passive utility direction")
                entropy_columns -= dt * q / utility[name]
            if name in m.walls:
                w = m.walls[name]
                qa = dt * engine._weighted([s["column_ambient_w"][name] for s in stages])
                heat_columns += qa
                entropy_columns += (
                    w.heat_capacity_j_k * math.log(dict(new.walls)[name] / dict(state.walls)[name])
                    - qa / w.ambient_temperature_k
                )
        entropy_columns += dt * engine._weighted([s["heat_link_entropy_rate"] for s in stages])
        total_heat = report["external_heat_j"] + report["ambient_heat_j"] + heat_columns
        work = report["external_work_j"]
        entropy = report["entropy_production_j_k"] + entropy_columns
        if entropy < -1e-6:
            raise ConservationError("Common gravitational Radau entropy rejected")
        m.inspect(new)
        residual = m._ledger(state, new, total_heat + work)
        self._endpoints[order] = (new, key, point)
        return new, {
            "dt_s": dt,
            "state": new.to_dict(),
            "method": engine.scheme,
            "transport": report,
            "columns": {**cr, "ports": ports},
            "quadrature_order": order,
            "embedded_error_ratio": report["embedded_error_ratio"],
            "total_heat_j": total_heat,
            "total_work_j": work,
            "entropy_production_j_k": entropy,
            "conservation_residuals": residual,
            **({"controls": cr["controls"]} if "controls" in cr else {}),
        }

    def _step(self, state, dt, openings, heat, utility, losses, **commands):
        m = self.physics
        previous = None
        order = m.start_order
        while order <= m.maximum_order:
            new, report = self._step_order(
                state, dt, order, openings, heat, utility, losses, **commands
            )
            if previous is not None:
                error = m._error(previous[0], new, openings=openings)
                if commands.get("_controls") is not None:
                    error = max(error, commands["_controls"].report_error(previous[1], report))
                for name, port in report["columns"]["ports"].items():
                    for a, b in zip(port["flux"], previous[1]["columns"]["ports"][name]["flux"]):
                        error = max(
                            error,
                            abs(a - b)
                            / (1e-9 + m.transport.relative_tolerance * max(abs(a), abs(b))),
                        )
                if error <= 1:
                    report["quadrature_error"] = error
                    return new, report
            previous = (new, report)
            order *= 2
        raise HydraulicConvergenceError("Common Radau aperture quadrature budget exhausted")

    @cached_fluid_transaction(lambda self: (self.physics.transport.fluid,))
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
    ):
        m, t = self.physics, self.physics.transport
        m.inspect(state)
        end = state.time_s + _number(duration_s, "duration", positive=True)
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Unrepresentable gravitational Radau interval")
        commands = {p.name: True for p in (*t.switches, *m.ports, *m.heat_links)}
        commands.update(t.default_openings)
        changes = dict(openings or {})
        if set(changes) - set(commands) or any(type(v) is not bool for v in changes.values()):
            raise ValueError("Known binary isolation commands required")
        commands.update(changes)
        heat, utility, losses = (
            dict(heat_w or {}),
            dict(utility_temperature_k or {}),
            dict(loss_coefficients or {}),
        )
        names = set(t.names) | set(m.columns)
        if (set(heat) | set(utility)) - names:
            raise ValueError("Unknown thermal node")
        for name, q in heat.items():
            _number(q, "heat rate")
            if q and name not in utility:
                raise ValueError("Heat requires utility temperature")
        for value in utility.values():
            _number(value, "utility temperature", positive=True)
        if set(losses) - {n.name for n in t.nodes if n.kind == "pipe_cell"}:
            raise ValueError("Unknown resistance cell")
        for value in losses.values():
            _number(value, "loss coefficient", nonnegative=True)
        equipment = {
            "drive_torque_nm": drive_torque_nm,
            "restriction_openings": restriction_openings,
        }
        current, records, attempts, rejected, fallbacks = state, [], 0, 0, 0
        dt = state.transport.next_step_s
        start_solves = self.engine._solve_calls
        while current.time_s < end:
            attempts += 1
            if attempts > t.maximum_attempts:
                raise HydraulicConvergenceError(
                    "Common gravitational Radau attempt budget exhausted"
                )
            step = interval_step_s(
                current.time_s, end, min(dt, t.maximum_step_s), maximum_s=t.maximum_step_s
            )
            fallback = None
            try:
                try:
                    new, report = self._step(
                        current, step, commands, heat, utility, losses, **equipment
                    )
                    error, accepted, order = report["embedded_error_ratio"], (report,), 2
                except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
                    self._endpoints.clear()
                    if not self.phase_fallback:
                        raise
                    fallback = str(exc)
                    coarse, _ = m._step(current, step, commands, heat, utility, losses, **equipment)
                    halves = split_step_s(current.time_s, step)
                    half, one = m._step(
                        current, halves[0], commands, heat, utility, losses, **equipment
                    )
                    new, two = m._step(
                        half, halves[1], commands, heat, utility, losses, **equipment
                    )
                    error, accepted, order = m._error(coarse, new, openings=commands), (one, two), 1
                if not math.isfinite(error) or error > 1:
                    raise HydraulicConvergenceError(
                        "Common gravitational Radau temporal error exceeded"
                    )
            except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
                self._endpoints.clear()
                rejected += 1
                dt = step / 2
                if dt < t.minimum_step_s:
                    raise HydraulicConvergenceError(
                        "Common gravitational Radau minimum step exhausted"
                    ) from exc
                continue
            dt = min(
                t.maximum_step_s,
                max(
                    t.minimum_step_s,
                    step
                    * min(
                        2.0 if fallback else 5.0,
                        max(0.5, 0.9 * max(error, 1e-16) ** (-1 / (order + 1))),
                    ),
                ),
            )
            current = replace(
                new,
                transport=replace(
                    new.transport,
                    time_s=end if step == end - current.time_s else new.time_s,
                    next_step_s=dt,
                ),
            )
            for row in accepted:
                row.update(
                    method="backward_euler_fallback" if fallback else "radau3", error_ratio=error
                )
                if fallback:
                    row["fallback_reason"] = fallback
            fallbacks += int(fallback is not None)
            records.extend(accepted)
        heat_total = math.fsum(row["total_heat_j"] for row in records)
        work_total = math.fsum(row["total_work_j"] for row in records)
        return {
            "state": current.to_dict(),
            "steps": records,
            "attempts": attempts,
            "rejected_attempts": rejected,
            "fallback_intervals": fallbacks,
            "collocation_nonlinear_solves": self.engine._solve_calls - start_solves,
            "total_heat_j": heat_total,
            "total_work_j": work_total,
            "entropy_production_j_k": math.fsum(r["entropy_production_j_k"] for r in records),
            "conservation_residuals": m._ledger(state, current, heat_total + work_total),
        }
