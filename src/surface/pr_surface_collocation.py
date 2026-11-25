"""Implicit PR plant/PI/rate-limited actuator stages on one physical graph.

Uses L-stable SDIRK2. Its embedded companion needs no left-end derivative,
which is important for ideal (tau=0) rate-limited tracking constraints.
Unsupported protection/feedback regimes are explicitly rejected; the existing
controlled solver remains the event/phase fallback, with its own error test.
"""

import math
from dataclasses import replace

import torch

from .actuator import ConnectionActuator
from .equilibrium_cache import cached_fluid_transaction
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .peng_robinson import R
from .pr_control_channels import PRMotorControl
from .pr_gravity_collocation import PRGravityCollocation
from .pr_surface_control import PRControlledSurface, PRControlValve
from .stratified_pr import StratifiedPRState
from .time_interval import interval_step_s, split_step_s


class _ControlStages:
    def __init__(self, model, state, manual, modes, commands):
        self.model, self.state = model, state
        self.manual, self.modes, self.commands = manual, modes, commands
        self.positions, self.integrals = dict(state.positions), dict(state.integrals)
        self.controllers = {c.valve: c for c in model.controllers}
        self.channels = model.channels

    def decorate(self, prepared, coupling):
        m, t, f = self.model, self.model.plant.transport, self.model.plant.transport.fluid
        y, lo, hi, base, base_rates, offsets, problems, old, seeds = prepared
        size = len(y)
        initial, lower, upper = [y], [lo], [hi]
        positions, velocities, integral_indices, traces = {}, {}, {}, {}
        cursor = size
        blocks = [list(b) for b in base.linear_blocks]
        for channel in self.channels:
            positions[channel.name], velocities[channel.name] = cursor, cursor + 1
            value = self.positions[channel.name]
            if isinstance(channel, PRControlValve) and value <= 0:
                raise PropertyModelError("Closed valve opening requires the event-aware fallback")
            initial.append(f.tensor([value, 0.0]))
            a = channel.actuator
            lower.append(f.tensor([max(0.0, value - a.max_position_change), -a.closing_rate_per_s]))
            upper.append(f.tensor([min(1.0, value + a.max_position_change), a.opening_rate_per_s]))
            blocks.append([cursor, cursor + 1])
            cursor += 2
        for controller in m.controllers:
            integral_indices[controller.valve] = cursor
            initial.append(f.tensor([self.integrals[controller.valve]]))
            lower.append(f.tensor([-controller.maximum_integral_abs]))
            upper.append(f.tensor([controller.maximum_integral_abs]))
            blocks.append([cursor])
            cursor += 1
            if controller.measurement == "pressure":
                port = m._ports[m._valves[controller.valve].port]
                if port.column is not None:
                    state = dict(self.state.plant.columns)[port.column]
                    active = coupling.problems[port.column][4]
                    phase = sum(port.height_m > h for h in state.interface_heights_m)
                    count = len(active)
                    x = f.tensor(state.phase_seeds[phase])[active].clamp_min(1e-30)
                    initial.append(
                        torch.cat(
                            (
                                f.tensor(
                                    [
                                        state.solution_variables[
                                            phase * state.quadrature_order * count
                                        ]
                                    ]
                                ),
                                (x[:-1] / x[-1]).log(),
                            )
                        )
                    )
                    lower.append(f.tensor([math.log(f.min_pressure_pa)] + [-100.0] * (count - 1)))
                    upper.append(f.tensor([math.log(f.max_pressure_pa)] + [100.0] * (count - 1)))
                    traces[controller.valve] = (
                        cursor,
                        cursor + count,
                        port,
                        phase,
                        state.phase_roots[phase],
                    )
                    blocks.append(list(range(cursor, cursor + count)))
                    cursor += count
        # Integral and trace unknowns were appended per controller. Residuals
        # below use their actual indices rather than assuming contiguous I's.
        y, lo, hi = torch.cat(initial), torch.cat(lower), torch.cat(upper)
        if len(y) > t.maximum_unknowns:
            raise ValueError("Common controlled stages exceed configured unknown budget")

        def runtime(z):
            losses = {
                n.name: f.tensor(
                    self.commands.get("loss_coefficients", {}).get(n.name, n.local_loss_coefficient)
                )
                for n in t.nodes
                if n.kind == "pipe_cell"
            }
            torques = {
                name: f.tensor(value)
                for name, value in self.commands.get("drive_torque_nm", {}).items()
            }
            for channel in self.channels:
                x = z[positions[channel.name]]
                if isinstance(channel, PRMotorControl):
                    torques[channel.machine] = channel.minimum_torque_nm + x * (
                        channel.maximum_torque_nm - channel.minimum_torque_nm
                    )
                elif coupling.openings[channel.port]:
                    fraction = (
                        x
                        if channel.characteristic == "linear"
                        else torch.expm1(x * math.log(channel.rangeability))
                        / (channel.rangeability - 1)
                    )
                    node = m._ports[channel.port].node
                    losses[node] = (
                        losses[node]
                        + channel.full_open_loss_coefficient / fraction.clamp_min(1e-15).square()
                    )
            return {"loss_coefficients": losses, "drive_torque_nm": torques}

        def measurement(controller, point, z, trace_values):
            kind = controller.measurement
            if kind == "speed":
                return point["drives"][controller.sensor or m._valves[controller.valve].port][
                    "speed_rad_s"
                ]
            port = m._ports[m._valves[controller.valve].port]
            if kind == "temperature":
                return point["data"][t.names.index(controller.sensor or port.sensor_node)]["t"]
            if kind == "pressure":
                return (
                    trace_values[controller.valve]
                    if controller.valve in traces
                    else point["data"][t.names.index(controller.sensor or port.sensor_node)]["p"]
                )
            name = (
                controller.sensor
                if kind == "level" and controller.sensor is not None
                else port.column
            )
            index = f.names.index(controller.component)
            if name is not None:
                body = point["column_bodies"][name]
                phase = body["phase_component_moles"]
                fraction = phase[:, index] / phase.sum(-1)
                peak = fraction.argmax()
                levels = torch.cat(
                    (body["interfaces"], z.new_tensor([m.plant.columns[name].geometry.height_m]))
                )
                return torch.where(
                    fraction.max() >= controller.minimum_mole_fraction,
                    levels.gather(0, peak.reshape(1))[0],
                    z.new_zeros(()),
                )
            path = m._inventory_path(controller)
            volume = z.new_zeros(())
            for node in path:
                d = point["data"][t.names.index(node.name)]
                volume = (
                    volume
                    + (
                        d["phase_n"]
                        * d["q"]["molar_volume_m3_mol"]
                        * (d["x"][:, index] >= controller.minimum_mole_fraction)
                    ).sum()
                )
            return volume / (math.pi * path[0].diameter_m ** 2 / 4)

        def evaluate(z):
            row, point = base(z[:size], runtime_controls=runtime(z))
            trace_rows, trace_values = {}, {}
            valid = point["valid"]
            for name, (a, b, port, phase, root) in traces.items():
                column = m.plant.columns[port.column]
                body = point["column_bodies"][port.column]
                active = coupling.problems[port.column][4]
                pp, xx, q = column._decode_nodes(
                    z[a:b].reshape(1, b - a), active, body["temperature"], root
                )
                mu = (
                    xx[0, active].clamp_min(1e-300).log()
                    + q["log_fugacity_coefficients"][0, active]
                    + (pp[0] / f.pref).log()
                )
                trace_rows[name] = (
                    mu
                    + f.molar_masses[active]
                    * column.gravity_m_s2
                    * port.height_m
                    / (R * body["temperature"])
                    - body["constants"]
                )
                trace_values[name] = pp[0]
                selected = (body["interfaces"] < port.height_m).sum()
                valid = valid & q["valid"].all() & (selected == phase)
            rows, rate_values = {}, {}
            measured, targets, integrals = [], [], []
            for channel in self.channels:
                name, actuator = channel.name, channel.actuator
                x, velocity = z[positions[name]], z[velocities[name]]
                if isinstance(channel, PRControlValve) and coupling.openings[channel.port]:
                    valid = valid & (x > 0)
                controller = self.controllers.get(name)
                value = (
                    z.new_zeros(())
                    if controller is None
                    else measurement(controller, point, z, trace_values)
                )
                if controller is not None:
                    integral = z[integral_indices[name]]
                    proportional = (
                        controller.proportional_gain
                        * controller.action
                        * (value - controller.setpoint)
                    )
                    if name in self.manual:
                        target = z.new_tensor(self.manual[name])
                        desired = (target - controller.opening_bias - proportional).clamp(
                            -controller.maximum_integral_abs, controller.maximum_integral_abs
                        )
                        rows[integral_indices[name]] = integral - desired
                        rate_values[integral_indices[name]] = z.new_zeros(())
                    else:
                        target = (controller.opening_bias + proportional + integral).clamp(0, 1)
                        derivative = (
                            controller.integral_gain_per_s
                            * controller.action
                            * (value - controller.setpoint)
                            + (x - controller.opening_bias - proportional - integral)
                            / controller.tracking_time_s
                        )
                        derivative = torch.where(
                            ((integral >= controller.maximum_integral_abs) & (derivative > 0))
                            | ((integral <= -controller.maximum_integral_abs) & (derivative < 0)),
                            z.new_zeros(()),
                            derivative,
                        )
                        rows[integral_indices[name]] = integral - self.integrals[name]
                        rate_values[integral_indices[name]] = derivative
                    integrals.append(integral)
                else:
                    target = z.new_tensor(self.manual.get(name, self.positions[name]))
                mode = self.modes.get(name, "normal")
                if mode == "fail_safe":
                    target = z.new_tensor(actuator.fail_position)
                if mode == "stuck":
                    velocity_row = velocity
                elif actuator.time_constant_s > 0:
                    velocity_row = velocity - ((target - x) / actuator.time_constant_s).clamp(
                        -actuator.closing_rate_per_s, actuator.opening_rate_per_s
                    )
                else:
                    # Exact ideal rate limiter as a complementarity equation.
                    # At target, velocity is determined by the stage storage
                    # equations, without inserting an artificial relaxation.
                    velocity_row = velocity - (velocity + target - x).clamp(
                        -actuator.closing_rate_per_s, actuator.opening_rate_per_s
                    )
                rows[positions[name]], rows[velocities[name]] = (
                    x - self.positions[name],
                    velocity_row,
                )
                rate_values[positions[name]], rate_values[velocities[name]] = (
                    velocity,
                    z.new_zeros(()),
                )
                measured.append(value)
                targets.append(target)
            for name, (a, b, *_rest) in traces.items():
                for j in range(b - a):
                    rows[a + j] = trace_rows[name][j]
                    rate_values[a + j] = z.new_zeros(())
            return torch.cat((row, torch.stack([rows[i] for i in range(size, len(z))]))), {
                **point,
                "valid": valid,
                "control_rates": torch.stack([rate_values[i] for i in range(size, len(z))]),
                "controls": {
                    "positions": torch.stack([z[positions[c.name]] for c in self.channels]),
                    "velocities": torch.stack([z[velocities[c.name]] for c in self.channels]),
                    "integrals": torch.stack([z[integral_indices[c.valve]] for c in m.controllers])
                    if m.controllers
                    else z.new_zeros(0),
                    "measurements": torch.stack(measured),
                    "targets": torch.stack(targets),
                },
            }

        def rates(point):
            return torch.cat((base_rates(point), point["control_rates"]))

        support = set(zip(*base.pattern))
        rotor_offsets = {d.machine: offset for d, offset, *_ in base.rotors.entries}
        for channel in self.channels:
            name = channel.name
            x, v = positions[name], velocities[name]
            dependencies = {x, v}
            controller = self.controllers.get(name)
            if isinstance(channel, PRMotorControl):
                rotor = rotor_offsets[channel.machine]
                support.update((r, x) for r, c in zip(*base.pattern) if c == rotor)
            else:
                node = t.names.index(m._ports[channel.port].node)
                support.add((offsets[node + 1] - 1, x))
            if controller is not None:
                dependencies.add(integral_indices[name])
                if controller.measurement == "speed":
                    dependencies.add(rotor_offsets[controller.sensor or channel.port])
                elif name in traces:
                    a, b, *_ = traces[name]
                    dependencies.update(range(a, b))
                else:
                    port = m._ports[channel.port]
                    column_name = (
                        controller.sensor
                        if controller.measurement == "level" and controller.sensor is not None
                        else port.column
                    )
                    if (
                        controller.measurement in ("level", "inventory_level")
                        and column_name is not None
                    ):
                        a, b = coupling.blocks[column_name]
                        dependencies.update(range(a, b))
                    else:
                        sensors = (
                            [controller.sensor or port.sensor_node]
                            if controller.measurement in ("pressure", "temperature")
                            else [n.name for n in m._inventory_path(controller)]
                        )
                        for sensor in sensors:
                            i = t.names.index(sensor)
                            dependencies.update(range(offsets[i], offsets[i + 1]))
                support.update((integral_indices[name], j) for j in dependencies)
            support.update((r, c) for r in (x, v) for c in dependencies)
        for name, (a, b, port, *_rest) in traces.items():
            ca, cb = coupling.blocks[port.column]
            support.update((r, c) for r in range(a, b) for c in (*range(a, b), *range(ca, cb)))
        support = sorted(support)
        evaluate.pattern = ([r for r, c in support], [c for r, c in support])
        evaluate.maximum_update = torch.cat(
            (base.maximum_update, f.tensor([math.inf] * (len(y) - size)))
        )
        evaluate.linear_blocks = blocks
        for key in ("rotors", "rotor_hosts", "compressors", "pump_connections", "restrictions"):
            setattr(evaluate, key, getattr(base, key))

        def extra_error(a, b):
            values = [
                base.additional_error(a, b),
                (a["controls"]["positions"] - b["controls"]["positions"]).abs().max()
                / m.position_tolerance,
            ]
            if m.controllers:
                values.append(
                    (a["controls"]["integrals"] - b["controls"]["integrals"]).abs().max()
                    / m.integral_tolerance
                )
            return torch.stack(values).max()

        evaluate.additional_error = extra_error
        return y, lo, hi, evaluate, rates, offsets, problems, old, seeds

    def report_error(self, a, b):
        x, y = a["controls"], b["controls"]
        error = (
            max(abs(u - v) for u, v in zip(x["positions"], y["positions"]))
            / self.model.position_tolerance
        )
        if x["integrals"]:
            error = max(
                error,
                max(abs(u - v) for u, v in zip(x["integrals"], y["integrals"]))
                / self.model.integral_tolerance,
            )
        return error


class PRSurfaceCollocation:
    _restart_on_command_change = True

    def __init__(self, physics, *, phase_fallback=True):
        if not isinstance(physics, PRControlledSurface):
            raise TypeError("Expected PRControlledSurface")
        if type(phase_fallback) is not bool:
            raise ValueError("phase_fallback must be boolean")
        self.source, self.physics = physics, getattr(physics, "reference_surface", physics)
        m = self.physics
        if not m.channels:
            raise ValueError("Use the plant integrator when there are no control channels")
        if m.trips:
            raise ValueError("Implicit surface protections require a stage event adapter")
        if any(not isinstance(c, (PRMotorControl, PRControlValve)) for c in m.channels):
            raise ValueError("Implicit throat actuators require a live area adapter")
        if any(
            c.measurement not in ("speed", "pressure", "temperature", "level", "inventory_level")
            for c in m.controllers
        ):
            raise ValueError(
                "Implicit machine-flow/margin feedback requires its own sensor adapter"
            )
        self.plant = PRGravityCollocation(m.plant, phase_fallback=False, scheme="sdirk2")
        self.phase_fallback = phase_fallback
        self._last_advance_commands = None
        self._last_advance_time = None

    def _step(self, state, dt, manual, modes, commands):
        m = self.physics
        physical = m.plant
        controller = _ControlStages(m, state, manual, modes, commands)
        openings = {
            p.name: True
            for p in (*physical.transport.switches, *physical.ports, *physical.heat_links)
        }
        openings.update(physical.transport.default_openings)
        changes = dict(commands.get("openings") or {})
        if set(changes) - set(openings) or any(type(v) is not bool for v in changes.values()):
            raise ValueError("Known binary isolation commands required")
        openings.update(changes)
        heat, utility, losses = (
            dict(commands.get(k) or {})
            for k in ("heat_w", "utility_temperature_k", "loss_coefficients")
        )
        torques = dict(commands.get("drive_torque_nm") or {})
        if set(torques) & {c.machine for c in m.motors}:
            raise ValueError("Controlled torque requires an actuator command")
        for c in m.motors:
            torques[c.machine] = c.torque(dict(state.positions)[c.name])
        new_plant, report = self.plant._step(
            state.plant,
            dt,
            openings,
            heat,
            utility,
            losses,
            drive_torque_nm=torques,
            restriction_openings=commands.get("restriction_openings"),
            _controls=controller,
        )
        control = report["controls"]
        new = replace(
            state,
            plant=new_plant,
            positions=tuple((c.name, x) for c, x in zip(m.channels, control["positions"])),
            integrals=tuple((c.valve, x) for c, x in zip(m.controllers, control["integrals"])),
        )
        m.inspect(new)
        # Endpoint and all stage bounds are checked on the actual solved
        # primitives; a valve cannot gain speed by leaving its HEM envelope.
        for snapshot, column in zip(
            report["transport"]["stage_states"], report["transport"]["stage_couplings"]
        ):
            for c, x in zip(m.channels, column["controls"]["positions"]):
                if isinstance(c, PRMotorControl) or not openings[c.port]:
                    continue
                port = m._ports[c.port]
                body = snapshot[port.node]
                node = physical.transport.nodes[physical.transport.names.index(port.node)]
                mass = sum(
                    a * b
                    for a, b in zip(
                        body["component_moles"],
                        physical.transport.fluid.molar_masses.cpu().tolist(),
                    )
                )
                rho = mass / node.volume_m3
                k = c.coefficient(x)
                pressure = body["pressure_pa"]
                if port.column is not None:
                    column_pressure = physical.columns[port.column].sample(
                        StratifiedPRState.from_dict(column["columns"][port.column]),
                        port.height_m,
                        side="below",
                    )["pressure_pa"]
                    pressure = min(pressure, column_pressure)
                else:
                    pressure = min(pressure, snapshot[port.sensor_node]["pressure_pa"])
                drop = 0.5 * rho * body["velocity_m_s"] ** 2 * k
                if drop / pressure > c.maximum_pressure_drop_fraction:
                    raise PropertyModelError("Implicit valve exceeds low-drop envelope")
                sound = max(
                    abs(v - body["velocity_m_s"]) for v in body["characteristic_speeds_m_s"]
                )
                if sound <= 0 or abs(body["velocity_m_s"]) / sound > c.maximum_mach_number:
                    raise PropertyModelError("Implicit valve exceeds subsonic envelope")
        return new, {
            "state": new.to_dict(),
            "dt_s": dt,
            "method": "sdirk2-common",
            "plant": report,
            "error_ratio": report["embedded_error_ratio"],
            "total_heat_j": report["total_heat_j"],
            "total_work_j": report["total_work_j"],
        }

    @cached_fluid_transaction(lambda self: (self.physics.plant.transport.fluid,))
    def advance(
        self, state, duration_s, *, targets=None, actuator_modes=None, progress=None, **commands
    ):
        self.source.inspect(state)
        m = self.physics
        state = replace(state, model_id=m.model_id)
        end = state.time_s + _number(duration_s, "duration", positive=True)
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Unrepresentable implicit controlled interval")
        manual, modes = dict(targets or {}), dict(actuator_modes or {})
        if (set(manual) | set(modes)) - {c.name for c in m.channels}:
            raise ValueError("Unknown actuator command")
        for value in manual.values():
            ConnectionActuator.position(value)
        if any(v not in ("normal", "stuck", "fail_safe") for v in modes.values()):
            raise ValueError("Unknown actuator mode")
        if set(commands) - {
            "openings",
            "heat_w",
            "utility_temperature_k",
            "loss_coefficients",
            "drive_torque_nm",
            "restriction_openings",
        }:
            raise ValueError("Unknown physical command")
        if progress is not None and not callable(progress):
            raise TypeError("Progress callback must be callable")
        commands = {key: dict(value or {}) for key, value in commands.items()}
        thermal = set(m.plant.transport.names) | set(m.plant.columns)
        if (
            set(commands.get("heat_w", {})) | set(commands.get("utility_temperature_k", {}))
        ) - thermal:
            raise ValueError("Unknown thermal node")
        for name, value in commands.get("heat_w", {}).items():
            _number(value, "heat rate")
            if value and name not in commands.get("utility_temperature_k", {}):
                raise ValueError("Nonzero heat requires a utility temperature")
        for value in commands.get("utility_temperature_k", {}).values():
            _number(value, "utility temperature", positive=True)
        pipes = {n.name for n in m.plant.transport.nodes if n.kind == "pipe_cell"}
        if set(commands.get("loss_coefficients", {})) - pipes:
            raise ValueError("Unknown resistance cell")
        for value in commands.get("loss_coefficients", {}).values():
            _number(value, "loss coefficient", nonnegative=True)
        drives = m.plant.transport._drives
        if set(commands.get("drive_torque_nm", {})) - set(drives):
            raise ValueError("Unknown driven machine")
        for name, value in commands.get("drive_torque_nm", {}).items():
            value = _number(value, "motor torque")
            spec = drives[name].drive
            if not -spec.max_braking_torque_nm <= value <= spec.max_motor_torque_nm:
                raise ValueError("Motor torque outside drive limits")
        if set(commands.get("restriction_openings", {})) - {
            r.name for r in m.plant.transport.restrictions
        }:
            raise ValueError("Unknown restriction opening")
        for value in commands.get("restriction_openings", {}).values():
            if not 0 <= _number(value, "restriction opening") <= 1:
                raise ValueError("Restriction opening outside [0,1]")
        if any(
            c.name not in manual
            and c.name not in {p.valve for p in m.controllers}
            and not isinstance(c, PRMotorControl)
            and modes.get(c.name) not in ("stuck", "fail_safe")
            for c in m.channels
        ):
            raise ValueError("Every valve requires feedback or a manual target")
        current, steps, attempts, rejected, fallbacks = state, [], 0, 0, 0
        dt = state.next_step_s
        command_key = _digest(
            {
                "targets": {k: float(v) + 0.0 for k, v in manual.items()},
                "modes": {k: v for k, v in modes.items() if v != "normal"},
                "physical": {
                    k: {
                        name: value if k == "openings" else float(value) + 0.0
                        for name, value in values.items()
                    }
                    for k, values in commands.items()
                    if values
                },
            }
        )
        command_restart = (
            self._restart_on_command_change
            and self._last_advance_time == state.time_s
            and self._last_advance_commands != command_key
        )
        if command_restart:
            # An accepted step from the old regime predicts neither the new
            # forcing nor the local error at a known discontinuity. Restart
            # conservatively, then use the unchanged adaptive error controller.
            dt = min(dt, m.initial_step_s)
        initial_step_hint = dt
        start_solves = self.plant.engine._solve_calls
        linear_start = dict(self.plant.engine._linear_work)
        predictor_start = dict(self.plant.engine._stage_predictor_work)
        while current.time_s < end:
            attempts += 1
            if attempts > m.maximum_attempts:
                raise HydraulicConvergenceError("Implicit controlled attempt budget exhausted")
            step = interval_step_s(
                current.time_s, end, min(dt, m.maximum_step_s), maximum_s=m.maximum_step_s
            )
            fallback = None
            try:
                try:
                    new, report = self._step(current, step, manual, modes, commands)
                    error, records, order = report["error_ratio"], (report,), 1
                except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
                    self.plant._endpoints.clear()
                    if not self.phase_fallback:
                        raise
                    fallback = str(exc)
                    coarse, _ = m._step(current, step, manual, modes, commands)
                    halves = split_step_s(current.time_s, step)
                    half, one = m._step(current, halves[0], manual, modes, commands)
                    new, two = m._step(half, halves[1], manual, modes, commands)
                    error, records, order = m._error(coarse, new), (one, two), 1
                if not math.isfinite(error) or error > 1:
                    raise HydraulicConvergenceError("Implicit controlled temporal error exceeded")
            except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
                rejected += 1
                self.plant._endpoints.clear()
                dt = step / 2
                if dt < m.minimum_step_s:
                    raise HydraulicConvergenceError(
                        "Implicit controlled minimum step exhausted"
                    ) from exc
                continue
            dt = min(
                m.maximum_step_s,
                max(
                    m.minimum_step_s,
                    step
                    * min(
                        5.0 if not fallback else 2.0,
                        max(0.5, 0.9 * max(error, 1e-16) ** (-1 / (order + 1))),
                    ),
                ),
            )
            current = replace(new, next_step_s=dt)
            for row in records:
                row.update(
                    method="controlled_backward_euler_fallback" if fallback else "sdirk2-common",
                    error_ratio=error,
                )
                if fallback:
                    row["fallback_reason"] = fallback
            steps.extend(records)
            fallbacks += int(fallback is not None)
            if progress is not None:
                progress(
                    {
                        "event": "accepted",
                        "time_s": current.time_s,
                        "dt_s": step,
                        "error_ratio": error,
                    }
                )
        heat = math.fsum(r["total_heat_j"] for r in steps)
        work = math.fsum(r["total_work_j"] for r in steps)
        current = replace(current, model_id=self.source.model_id)
        self.source.inspect(current)
        self._last_advance_commands = command_key
        self._last_advance_time = current.time_s
        return {
            "state": current.to_dict(),
            "steps": steps,
            "attempts": attempts,
            "rejected_attempts": rejected,
            "fallback_intervals": fallbacks,
            "integrator": "sdirk2-common",
            "command_step_restart": command_restart,
            "initial_step_hint_s": initial_step_hint,
            "collocation_nonlinear_solves": self.plant.engine._solve_calls - start_solves,
            "collocation_linear_work": {
                key: value - linear_start[key]
                for key, value in self.plant.engine._linear_work.items()
            },
            "collocation_predictor_work": {
                key: value - predictor_start[key]
                for key, value in self.plant.engine._stage_predictor_work.items()
            },
            "total_heat_j": heat,
            "total_work_j": work,
            "conservation_residuals": m.plant._ledger(state.plant, current.plant, heat + work),
        }
