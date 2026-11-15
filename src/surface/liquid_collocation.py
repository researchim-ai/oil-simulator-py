"""Opt-in fourth-order Torch collocation for fixed-control liquid DAEs.

Two-stage Gauss, index-one algebraic projection, embedded third-order error
estimate and independent quadratic mass/mechanical-energy ledgers. This is
not a multiphase PR backend or a substitute for event-aware station schedules.
"""

import math

import torch

from .liquid import EquipmentOperatingError, HydraulicConvergenceError, _number
from .liquid_transient import LiquidTransientNetwork, LiquidTransientState
from .torch_liquid import TorchLiquidEquations
from .torch_nonlinear import TorchNewton


class LiquidCollocation:
    def __init__(
        self,
        physics,
        *,
        relative_tolerance=None,
        pressure_tolerance_pa=None,
        flow_tolerance_m3_s=None,
        speed_tolerance_rad_s=None,
    ):
        if not isinstance(physics, LiquidTransientNetwork):
            raise TypeError("Expected LiquidTransientNetwork")
        if physics.backend != "torch":
            raise ValueError("Collocation requires the explicit Torch backend")
        if physics.actuated_valves:
            raise ValueError("Moving actuators require an event-aware collocation schedule")
        self.physics = physics
        self.model_id = physics.model_id
        self.e = e = TorchLiquidEquations(physics, physics.device)
        self.rtol = _number(
            physics.relative_tolerance if relative_tolerance is None else relative_tolerance,
            "relative_tolerance",
            positive=True,
        )

        def tolerance(value, name):
            return _number(getattr(physics, name) if value is None else value, name, positive=True)

        pa = tolerance(pressure_tolerance_pa, "pressure_tolerance_pa")
        qa = tolerance(flow_tolerance_m3_s, "flow_tolerance_m3_s")
        wa = tolerance(speed_tolerance_rad_s, "speed_tolerance_rad_s")
        self.atol = torch.cat(
            (
                e.tensor([pa / e.ps] * e.nf),
                e.tensor([qa / e.qs] * e.nb),
                e.tensor([wa] * e.npump) / e.nominal,
            )
        )
        self.a = e.tensor([[0.25, 0.25 - math.sqrt(3) / 6], [0.25 + math.sqrt(3) / 6, 0.25]])
        self.c = e.tensor([0.5 - math.sqrt(3) / 6, 0.5 + math.sqrt(3) / 6])
        self.lower = torch.cat(
            (
                e.tensor([e.pmin / e.ps] * e.nf),
                e.tensor([-math.inf] * e.nb),
                torch.zeros_like(e.nominal),
            )
        )
        self.upper = torch.cat(
            (e.tensor([e.pmax / e.ps] * e.nf), e.tensor([math.inf] * e.nb), e.maxw / e.nominal)
        )
        self.lower[e.nf + e.npipe : e.nf + e.npipe + e.npump] = 0.0
        self.boundary_mask = torch.ones(e.nn, dtype=e.dtype, device=e.device).scatter(
            0, e.free, 0.0
        )
        algebraic = set(range(e.nf + e.npipe, e.nf + e.nb))
        entries = {
            (i * e.size + r, j * e.size + c)
            for r, c in zip(*e.pattern)
            for i in range(2)
            for j in ([i] if r in algebraic else range(2))
        }
        entries.update((i, i) for i in range(2 * e.size))
        self.pattern = tuple(zip(*sorted(entries)))
        self.newton = TorchNewton(e.device, maximum_iterations=24)
        self.newton.host_convergence = True

    def _consistent(self, y, fixed, openings):
        e = self.e
        p = fixed.scatter(0, e.free, y[: e.nf] * e.ps)
        qp, qv, _ = e.algebraic_seed(p, y[e.nf + e.nb :] * e.nominal, openings)
        return torch.cat((y[: e.nf + e.npipe], qp / e.qs, qv / e.qs, y[e.nf + e.nb :]))

    def _equations(self, y, commands, fixed, openings):
        return self.e.instantaneous(y, commands, fixed, openings)

    def _stage_residual(self, stage, old, h, commands, fixed, openings):
        e = self.e
        ys = stage.reshape(2, e.size)
        values = [self._equations(y, commands, fixed, openings) for y in ys]
        rates = torch.stack([v[0] for v in values])
        delta = ys - old - h * (self.a @ rates)
        return torch.cat(
            [
                torch.cat((delta[i, : e.nf + e.npipe], values[i][1], delta[i, e.nf + e.nb :]))
                for i in range(2)
            ]
        )

    def _energy_change(self, old, new):
        e = self.e
        op, np = old[: e.nf] * e.ps, new[: e.nf] * e.ps
        oq, nq = old[e.nf : e.nf + e.npipe] * e.qs, new[e.nf : e.nf + e.npipe] * e.qs
        ow, nw = old[e.nf + e.nb :] * e.nominal, new[e.nf + e.nb :] * e.nominal
        return (
            0.5 * (e.C * (np - op) * (np + op - 2 * self.physics.reference_pressure_pa)).sum()
            + 0.5 * (e.L * (nq - oq) * (nq + oq)).sum()
            + 0.5 * (e.J * (nw - ow) * (nw + ow)).sum()
        )

    def _powers(self, data):
        e = self.e
        _, _, w, p, q, _, torque, power, losses, _, _ = data
        work = torque * w
        electric = torch.where(work >= 0, work / e.motor_eta, work * e.regen_eta)
        dp = p[e.start] - p[e.end]
        heats = torch.stack(
            (
                (electric - work).sum(),
                (e.drag * w.square()).sum(),
                (power + dp[e.npipe : e.npipe + e.npump] * q[e.npipe : e.npipe + e.npump]).sum(),
                losses.sum(),
            )
        )
        outward = e.incidence(q)
        boundary = ((p - self.physics.reference_pressure_pa) * outward * self.boundary_mask).sum()
        return torch.cat((electric.sum().reshape(1), boundary.reshape(1), heats)), outward

    def _valid(self, y, constraints, data):
        e = self.e
        basic = (
            torch.isfinite(y).all()
            & (y >= self.lower).all()
            & (y <= self.upper).all()
            & torch.isfinite(constraints).all()
        )
        if constraints.numel():
            basic = basic & (constraints.abs().max() <= 2e-11)
        # Identical reverse-flow, map, actual-flow NPSH and total shaft-power
        # audits to the existing bridge, rather than weaker sensor estimates.
        return basic & ~(e.equipment_flags(data)[:4] != 0).any()

    def _embedded(self, old, h, commands, fixed, openings, initial_rate=None):
        """Independent Kutta3 estimate; endpoint/stage combinations of Gauss
        alone can have identically zero defect on affine vector fields.
        The two extra evaluations do not solve extra implicit time steps.
        """
        k0 = (
            self._equations(old, commands, fixed, openings)[0]
            if initial_rate is None
            else initial_rate
        )
        midpoint = self._consistent(old + 0.5 * h * k0, fixed, openings)
        middle = self._equations(midpoint, commands, fixed, openings)
        endpoint = self._consistent(old + h * (-k0 + 2 * middle[0]), fixed, openings)
        last = self._equations(endpoint, commands, fixed, openings)
        low = self._consistent(old + h / 6 * (k0 + 4 * middle[0] + last[0]), fixed, openings)
        return low, (midpoint, endpoint), (middle, last)

    def _trial(self, old, h, commands, fixed, openings):
        e = self.e
        initial_rate, _, _ = self._equations(old, commands, fixed, openings)
        guesses = [
            self._consistent(
                torch.maximum(self.lower, torch.minimum(self.upper, old + h * c * initial_rate)),
                fixed,
                openings,
            )
            for c in self.c
        ]
        fn = lambda z: self._stage_residual(z, old, h, commands, fixed, openings)
        stages, residual, ok, evaluations = self.newton.solve(
            fn,
            torch.stack(guesses).flatten(),
            self.lower.repeat(2),
            self.upper.repeat(2),
            self.pattern,
        )
        self._last_solve_stats = (evaluations, self.newton.jacobian_builds)
        if not bool(ok.detach().cpu()):
            raise HydraulicConvergenceError("Gauss stage solve did not meet its residual target")
        stages = stages.reshape(2, e.size)
        values = [self._equations(y, commands, fixed, openings) for y in stages]
        new = self._consistent(old + h * 0.5 * (values[0][0] + values[1][0]), fixed, openings)
        _, end_constraint, end_data = self._equations(new, commands, fixed, openings)
        low, auxiliary, auxiliary_values = self._embedded(
            old, h, commands, fixed, openings, initial_rate
        )
        error = (
            (new - low).abs() / (self.atol + self.rtol * torch.maximum(old.abs(), new.abs()))
        ).max()
        powers, outwards = zip(*(self._powers(v[2]) for v in values))
        integrals = h * 0.5 * (powers[0] + powers[1])
        net = h * 0.5 * (outwards[0] + outwards[1])
        energy_change = self._energy_change(old, new)
        energy_error = integrals[:2].sum() - integrals[2:].sum() - energy_change
        local_mass = e.rho * (e.C * e.ps * (new[: e.nf] - old[: e.nf]) + net[e.free])
        mass_error = (
            e.rho * (e.C * e.ps * (new[: e.nf] - old[: e.nf])).sum()
            - (e.rho * net * self.boundary_mask).sum()
        )
        energy_scale = integrals.abs().sum() + energy_change.abs()
        ow, nw = old[e.nf + e.nb :] * e.nominal, new[e.nf + e.nb :] * e.nominal
        shaft_work = [v[2][6] * v[2][2] - v[2][7] - e.drag * v[2][2].square() for v in values]
        shaft_error = h * 0.5 * (shaft_work[0] + shaft_work[1]) - 0.5 * e.J * (nw - ow) * (nw + ow)
        max_shaft_error = shaft_error.abs().max() if e.npump else energy_error.new_zeros(())
        # The collocation polynomial is quadratic. Check every coordinate's
        # interior extremum too, so stages/endpoints cannot hide an overshoot.
        k0, k1 = values[0][0], values[1][0]
        slope = (k1 - k0) / (self.c[1] - self.c[0])
        intercept = (k0 * self.c[1] - k1 * self.c[0]) / (self.c[1] - self.c[0])
        position = (-intercept / torch.where(slope != 0, slope, torch.ones_like(slope))).clamp(0, 1)
        extrema = old + h * (intercept * position + 0.5 * slope * position.square())
        excursion = torch.maximum((extrema - old).abs(), (new - old).abs())
        valid = self._valid(new, end_constraint, end_data)
        for stage, value in zip(stages, values):
            valid = valid & self._valid(stage, value[1], value[2])
        for stage, value in zip(auxiliary, auxiliary_values):
            valid = valid & self._valid(stage, value[1], value[2])
        valid = (
            valid
            & torch.isfinite(error)
            & torch.isfinite(integrals).all()
            & (energy_error.abs() <= 1e-6 + 1e-7 * energy_scale)
            & (max_shaft_error <= 1e-6 + 1e-7 * energy_scale)
            & (integrals[2:] >= -1e-7).all()
            & (mass_error.abs() <= 1e-8)
            & (local_mass.abs().max() <= 1e-8)
            & (extrema >= self.lower).all()
            & (extrema <= self.upper).all()
            & (excursion[: e.nf] * e.ps <= self.physics.max_pressure_change_pa).all()
            & (excursion[e.nf + e.nb :] * e.nominal <= self.physics.max_speed_change_rad_s).all()
        )
        # One packed trial status read. Accepted tensors stay on their device.
        metrics = torch.stack((error, valid.to(e.dtype))).detach().cpu().tolist()
        record = torch.cat(
            (
                new,
                integrals,
                energy_change.reshape(1),
                energy_error.reshape(1),
                mass_error.reshape(1),
                local_mass.abs().max().reshape(1),
                residual.abs().max().reshape(1),
                max_shaft_error.reshape(1),
                stages.flatten(),
            )
        )
        return new, metrics[0], bool(metrics[1]), record, evaluations

    @torch.no_grad()
    def advance(
        self,
        state,
        duration_s,
        *,
        motor_torques_nm=None,
        boundary_pressures_pa=None,
        valve_openings=None,
        initial_step_s=None,
        max_attempts=10000,
        _coupling=None,
    ):
        m, e = self.physics, self.e
        m._check(state)
        if m.model_id != self.model_id:
            raise ValueError("Collocation model changed after kernel construction")
        if state.tripped or state.unavailable:
            raise ValueError(
                "Trip/availability transitions require the event-aware midpoint schedule"
            )
        duration = _number(duration_s, "duration_s", positive=True)
        maximum_dt = min(m.maximum_dt_s, _coupling.maximum_dt_s) if _coupling else m.maximum_dt_s
        minimum_dt = max(m.minimum_dt_s, _coupling.minimum_dt_s) if _coupling else m.minimum_dt_s
        if minimum_dt > maximum_dt:
            raise ValueError("Coupled timestep ranges do not overlap")
        target = state.time_s + duration
        if not math.isfinite(target) or target <= state.time_s:
            raise ValueError("Collocation cannot advance representable time")
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("Positive integer attempt budget required")
        commands = dict(motor_torques_nm or {})
        if set(commands) - set(m.names):
            raise ValueError("Unknown motor command")
        for n, v in commands.items():
            _number(v, "motor torque")
            if not -m.drives[n].max_braking_torque_nm <= v <= m.drives[n].max_motor_torque_nm:
                raise EquipmentOperatingError("Motor torque outside declared limits")
        command = e.tensor([commands.get(n, 0.0) for n in m.names])
        boundary, valves = m._controls(boundary_pressures_pa, valve_openings)
        fixed = e.tensor([boundary.get(n, m.reference_pressure_pa) for n in m._nodes])
        openings = e.tensor([v.opening for v in valves])
        y = torch.cat(
            (
                e.tensor([dict(state.pressures_pa)[n] / e.ps for n in m.free]),
                e.tensor(
                    [dict(state.pipe_volume_rates_m3_s)[p.name] / e.qs for p in m.network.pipes]
                ),
                e.tensor([0.0] * (e.nb - e.npipe)),
                e.tensor([dict(state.speeds_rad_s)[n] for n in m.names]) / e.nominal,
            )
        )
        y = self._consistent(y, fixed, openings)
        _, g, data = self._equations(y, command, fixed, openings)
        if not bool(self._valid(y, g, data).detach().cpu()):
            raise EquipmentOperatingError("Initial collocation state outside physical envelope")
        h = min(
            maximum_dt,
            _number(
                maximum_dt / 8 if initial_step_s is None else initial_step_s,
                "initial_step_s",
                positive=True,
            ),
        )
        if h < minimum_dt and duration >= minimum_dt:
            raise ValueError("Initial step below model minimum")
        time, accepted, rejected, evaluations, jacobians = state.time_s, [], 0, 0, 0
        for attempt in range(1, max_attempts + 1):
            remaining = target - time
            proposed_step = h
            step = min(h, remaining)
            if time + step <= time:
                raise HydraulicConvergenceError(
                    "Collocation step cannot advance representable time"
                )
            try:
                self._last_solve_stats = (0, 0)
                new, error, valid, record, _count = self._trial(y, step, command, fixed, openings)
                candidate = None
                if _coupling is not None and valid and error <= 1:
                    candidate, coupled_error, coupled_valid = _coupling.trial(
                        self, y, new, step, record, command, fixed, openings
                    )
                    error = max(error, coupled_error)
                    valid = valid and coupled_valid
            except HydraulicConvergenceError:
                error, valid = math.inf, False
            evaluations += self._last_solve_stats[0]
            jacobians += self._last_solve_stats[1]
            if not valid or error > 1:
                rejected += 1
                if step <= minimum_dt or remaining < minimum_dt:
                    raise HydraulicConvergenceError("Collocation exhausted its minimum timestep")
                factor = (
                    0.5
                    if not math.isfinite(error) or not valid
                    else max(0.1, min(0.5, 0.8 * error ** (-0.25)))
                )
                h = max(minimum_dt, step * factor)
                continue
            end = target if step == remaining else time + step
            if _coupling is not None:
                _coupling.commit(candidate, time, end, step)
            accepted.append((time, end, step, error, record))
            y, time = new, end
            h = min(
                maximum_dt,
                max(minimum_dt, step * min(2.0, max(0.5, 0.9 * max(error, 1e-16) ** (-0.25)))),
            )
            # Reporting/control boundaries do not provide evidence that the
            # underlying dynamics require the clipped tail as the next step.
            # Keep the already proposed scale; every following trial is still
            # subject to the same error, domain and conservation checks.
            if step < proposed_step and error <= (0.9 * step / proposed_step) ** 4:
                h = max(h, proposed_step)
            if time == target:
                break
        else:
            raise HydraulicConvergenceError("Collocation interval attempt budget exhausted")
        # Materialize the accepted journal once, after the interval commits.
        rows = torch.stack([s[4] for s in accepted]).detach().cpu().tolist()
        endpoint = rows[-1][: e.size]
        final = LiquidTransientState(
            tuple(zip(m.free, [v * e.ps for v in endpoint[: e.nf]])),
            tuple(
                (p.name, v * e.qs) for p, v in zip(m.network.pipes, endpoint[e.nf : e.nf + e.npipe])
            ),
            tuple(
                (n, v * m.nominal_speeds_rad_s[n]) for n, v in zip(m.names, endpoint[e.nf + e.nb :])
            ),
            state.unavailable,
            target,
            m.model_id,
            state.tripped,
            state.actuator_positions,
        )
        m._check(final)
        steps = []
        for (start, end, dt, error, _), row in zip(accepted, rows):
            info = dict(
                zip(
                    (
                        "electrical_energy_j",
                        "boundary_work_in_j",
                        "motor_heat_j",
                        "shaft_drag_heat_j",
                        "pump_heat_j",
                        "friction_heat_j",
                        "mechanical_energy_change_j",
                        "energy_residual_j",
                        "mass_residual_kg",
                        "max_node_mass_residual_kg",
                        "max_scaled_equation_residual",
                        "max_unit_shaft_energy_residual_j",
                    ),
                    row[e.size : e.size + 12],
                )
            )
            steps.append(
                dict(
                    time_s=start,
                    end_time_s=end,
                    dt_s=dt,
                    error_ratio=error,
                    scaled_state=row[: e.size],
                    stage_scaled_states=[
                        row[e.size + 12 : e.size * 2 + 12],
                        row[e.size * 2 + 12 :],
                    ],
                    **info,
                )
            )
        return {
            "integrator": "torch_gauss4_fixed_controls_v1",
            "state": final.to_dict(),
            "next_step_s": h,
            "steps": steps,
            "attempts": attempt,
            "rejected_steps": rejected,
            "nonlinear_evaluations": evaluations,
            "jacobian_builds": jacobians,
            "duration_s": duration,
            "model": m.to_dict(),
            "initial": state.to_dict(),
            "numerical_policy": {
                "relative_tolerance": self.rtol,
                "absolute_scaled_tolerances": self.atol.detach().cpu().tolist(),
                "embedded_order": 3,
                "error_estimator": "independent_kutta3",
                "accepted_order": 4,
                "nonlinear_tolerance": 2e-11,
            },
            "controls": {
                "motor_torques_nm": commands,
                "boundary_pressures_pa": boundary,
                "valve_openings": {v.name: v.opening for v in valves},
            },
        }
