"""Conservative Gauss4 for closed homogeneous-equilibrium PR graphs.

Reuse the existing PR chemical/volume constraints, HLLC flux and force
operators. Phase transitions use the existing backward-Euler branch search,
with the actual method explicitly recorded; no extrapolated phase states.
"""

import math
from dataclasses import replace

import torch

from .equilibrium_cache import cached_fluid_transaction
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .pr_equilibrium_transport import (
    PREquilibriumCellState,
    PREquilibriumTransportNetwork,
    PREquilibriumTransportState,
)
from .torch_nonlinear import TorchNewton


class PRCollocation:
    def __init__(self, physics, *, phase_fallback=True):
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
            n.wall
            or n.hydrostatic_reconstruction
            or n.droplet_population
            or n.drift_flux
            or n.distribution_strength
            or n.buoyancy_relaxation_s
            for n in physics.nodes
        ):
            raise ValueError(
                "Only homogeneous PR cells without wall/population/face inventories supported"
            )
        self.physics = physics
        self.model_id = physics.model_id
        self.phase_fallback = phase_fallback
        q = math.sqrt(3) / 6
        self.a = ((0.25, 0.25 - q), (0.25 + q, 0.25))

    def _problem(self, state, commands, heat, losses):
        m = self.physics
        seeds = [m._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
        y, lo, hi, evaluate, offsets, problems, old = m._problem(
            state, seeds, 0.0, commands, heat, losses
        )
        if offsets[-1] != len(y):
            raise ValueError("Additional algebraic bundle is not adapted to PR collocation")
        if 2 * len(y) > m.maximum_unknowns:
            raise ValueError("PR collocation exceeds configured unknown budget")

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
            return torch.cat(rows)

        return y, lo, hi, evaluate, rates, offsets, problems, old, seeds

    def _solve(self, fn, initial, lo, hi, pattern, update):
        m, f = self.physics, self.physics.fluid
        solver = TorchNewton(f.device, maximum_iterations=40, dense_limit=m.dense_limit)
        solver.host_convergence = True
        with torch.no_grad():
            y, _, ok, evaluations = solver.solve(
                fn,
                initial,
                lo,
                hi,
                pattern=pattern,
                tolerance=3e-12,
                stagnation_tolerance=1e-11,
                maximum_update=update,
            )
            row = fn(y)
        if not bool((ok & torch.isfinite(row).all() & (row.abs().max() < 1e-9)).cpu()):
            raise HydraulicConvergenceError("PR Gauss nonlinear/constraint solve rejected")
        return y, evaluations, float(row.abs().max().cpu())

    def _step(self, state, dt, commands, heat, utility, losses):
        m, f = self.physics, self.physics.fluid
        initial, lo, hi, evaluate, rates, offsets, problems, old, seeds = self._problem(
            state, commands, heat, losses
        )
        size = len(initial)
        base = m._sparsity(offsets, commands, (), (), (), ())
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
            result = [evaluate(v) for v in z.reshape(2, size)]
            rhs = [rates(p) for _, p in result]
            return torch.cat(
                [
                    r - dt * sum(self.a[i][j] * rhs[j] for j in range(2))
                    for i, (r, _) in enumerate(result)
                ]
            )

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
            integrated = dt * (rates(points[0]) + rates(points[1])) / 2
        endpoint, count, final_residual = self._solve(
            lambda v: evaluate(v)[0] - integrated,
            stages.reshape(2, size)[-1],
            lo,
            hi,
            base if size > m.dense_limit else None,
            evaluate.maximum_update,
        )
        with torch.no_grad():
            final = evaluate(endpoint)[1]
        for point in [*points, final]:
            if not bool(point["valid"].cpu()) or not all(
                bool(d["q"]["valid"].all().cpu()) for d in point["data"]
            ):
                raise PropertyModelError("PR Gauss stage outside physical flux/EOS domain")
            if any(post is not None for post in m._stable_all(point["data"])):
                raise PropertyModelError("PR Gauss phase topology changed")
        cells, local = [], []
        entropy = 0.0
        for i, (node, (_, previous), p, d) in enumerate(
            zip(m.nodes, state.nodes, problems, final["data"])
        ):
            rate = (points[0]["net"][i] + points[1]["net"][i]) / 2
            target_n = f.tensor(previous.component_moles) + dt * rate[:-2]
            target_energy = previous.total_energy_j + dt * (rate[-1] + heat[i])
            target_momentum = (
                previous.momentum_kg_m_s
                + dt * (rate[-2] + (points[0]["forces"][i] + points[1]["forces"][i]) / 2)
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
                None,
            )
            cells.append((node.name, cell))
            local.append(
                {
                    "component_residuals_mol": (d["n"] - target_n).cpu().tolist(),
                    "energy_residual_j": float((d["energy"] - target_energy).cpu()),
                    "momentum_residual_kg_m_s": float((d["momentum"] - target_momentum).cpu()),
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
        ledger = m._ledger(state, new, dt * sum(heat))
        return new, {
            "state": new.to_dict(),
            "method": "gauss4",
            "start_time_s": state.time_s,
            "end_time_s": new.time_s,
            "dt_s": dt,
            "conservation_residuals": ledger,
            "local_ledgers": local,
            "entropy_production_j_k": entropy,
            "external_heat_j": dt * sum(heat),
            "max_scaled_residual": max(residual, final_residual),
            "nonlinear_evaluations": evaluations + count,
            "stage_states": [
                {
                    node.name: {
                        "component_moles": d["n"].cpu().tolist(),
                        "total_energy_j": float(d["energy"].cpu()),
                        "wall_temperature_k": None,
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
                        dt * (points[0]["links"][j]["flux"] + points[1]["links"][j]["flux"]) / 2
                    )
                    .cpu()
                    .tolist(),
                }
                for j, link in enumerate(m.links)
            ],
            "heat_links": {
                link.name: dt
                * sum(
                    float(point["heat_links"][link.name]["heat_rate_w"].cpu()) for point in points
                )
                / 2
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
        current, steps, attempts, rejected, fallbacks = state, [], 0, 0, 0
        dt = state.next_step_s
        while current.time_s < end:
            attempts += 1
            if attempts > m.maximum_attempts:
                raise HydraulicConvergenceError("PR collocation interval attempt budget exhausted")
            proposed = min(dt, m.maximum_step_s)
            step = min(proposed, end - current.time_s)
            midpoint = current.time_s + step / 2
            if not current.time_s < midpoint < current.time_s + step:
                raise ValueError("PR collocation substep below time resolution")
            halves = (midpoint - current.time_s, (current.time_s + step) - midpoint)
            method, fallback_reason = "gauss4", None
            try:
                try:
                    coarse, _ = self._step(current, step, commands, heat, utility, losses)
                    half, one = self._step(current, halves[0], commands, heat, utility, losses)
                    fine, two = self._step(half, halves[1], commands, heat, utility, losses)
                    error = m._error(coarse, fine)
                    order = 4
                except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
                    if not self.phase_fallback:
                        raise
                    fallback_reason = str(exc)
                    method = "backward_euler_fallback"
                    coarse, _ = m._step(current, step, commands, heat, utility, losses)
                    half, one = m._step(current, halves[0], commands, heat, utility, losses)
                    fine, two = m._step(half, halves[1], commands, heat, utility, losses)
                    error, order = m._error(coarse, fine), 1
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
            factor = min(2.0, max(0.5, 0.9 * max(error, 1e-16) ** (-1 / (order + 1))))
            dt = min(m.maximum_step_s, max(m.minimum_step_s, step * factor))
            if step < proposed and error <= (0.9 * step / proposed) ** (order + 1):
                dt = max(dt, proposed)
            current = replace(
                fine, time_s=end if step == end - current.time_s else fine.time_s, next_step_s=dt
            )
            for record in (one, two):
                record.update(method=method, error_ratio=error)
                if fallback_reason:
                    record["fallback_reason"] = fallback_reason
            fallbacks += method != "gauss4"
            steps.extend((one, two))
        total_heat = math.fsum(s["external_heat_j"] for s in steps)
        return {
            "state": current.to_dict(),
            "steps": steps,
            "attempts": attempts,
            "rejected_attempts": rejected,
            "fallback_intervals": fallbacks,
            "conservation_residuals": m._ledger(state, current, total_heat),
            "total_heat_j": total_heat,
            "total_work_j": 0.0,
            "openings": commands,
            "loss_coefficients": losses,
            "entropy_production_j_k": math.fsum(s["entropy_production_j_k"] for s in steps),
        }
