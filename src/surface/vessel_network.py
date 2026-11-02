"""Monolithic conservative dynamics of equilibrium PR control volumes.

Connections use calibrated linear conductance or explicit single-phase PR
sonic flow. No line inventory/inertia, water, slip, or GPU implementation.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from .dynamic_vessel import VesselState
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _name, _number
from .peng_robinson import R
from .separator_geometry import LevelWithdrawal


def _port(q, n, port):
    if port == "bulk":
        return np.array(n) / sum(n), q["molar_enthalpy_j_mol"], q["molar_entropy_j_mol_k"]
    if port not in ("phase_0", "phase_1"):
        raise ValueError("Unknown dynamic network phase port")
    index = int(port[-1])
    if index >= q["phase_count"]:
        raise PropertyModelError("Dynamic network phase port is absent")
    phase = q["phases"][index]
    return (
        np.array(phase["mole_fractions"]),
        phase["molar_enthalpy_j_mol"],
        phase["molar_entropy_j_mol_k"],
    )


@dataclass(frozen=True)
class VesselConnection:
    name: str
    source: str
    target: str
    molar_conductance_mol_s_pa: float
    max_pressure_difference_pa: float
    max_molar_rate_mol_s: float
    source_port: str = "bulk"
    target_port: str = "bulk"
    allow_reverse: bool = False

    def __post_init__(self):
        for name in ("name", "source", "target"):
            _name(getattr(self, name))
        if self.source == self.target:
            raise ValueError("Vessel connection cannot be a self loop")
        for key in (
            "molar_conductance_mol_s_pa",
            "max_pressure_difference_pa",
            "max_molar_rate_mol_s",
        ):
            _number(getattr(self, key), key, positive=True)
        if not isinstance(self.allow_reverse, bool):
            raise TypeError("allow_reverse must be boolean")
        if any(
            x not in ("bulk", "phase_0", "phase_1") for x in (self.source_port, self.target_port)
        ):
            raise ValueError("Unknown connection phase port")

    def flow(self, pressures, validate=True, factor=1.0, *, values=None, fluid=None, cache=None):
        factor = _number(factor, "conductance_factor", nonnegative=True)
        if factor > 1:
            raise ValueError("Conductance factor must be in [0,1]")
        drop = pressures[self.source] - pressures[self.target]
        rate = (
            factor
            * self.molar_conductance_mol_s_pa
            * (drop if self.allow_reverse else max(0.0, drop))
        )
        if (
            validate
            and rate
            and (
                abs(drop) > self.max_pressure_difference_pa or abs(rate) > self.max_molar_rate_mol_s
            )
        ):
            raise PropertyModelError("Connection outside calibrated pressure/flow domain")
        return rate


@dataclass(frozen=True)
class DynamicVesselNetwork:
    vessels: object
    connections: tuple = ()
    walls: object = ()
    actuators: object = ()
    geometries: object = ()

    def __post_init__(self):
        items = tuple(sorted(dict(self.vessels).items()))
        if not items or any(name != v.name for name, v in items):
            raise ValueError("Vessel keys must match unique vessel names")
        connections = tuple(self.connections)
        if len({c.name for c in connections}) != len(connections):
            raise ValueError("Connection names must be unique")
        names = {name for name, _ in items}
        if any({c.source, c.target} - names for c in connections):
            raise ValueError("Connection references an unknown vessel")
        object.__setattr__(self, "vessels", items)
        object.__setattr__(self, "connections", connections)
        from .vessel_wall import VesselWall

        walls = tuple(sorted(dict(self.walls).items()))
        if {name for name, _ in walls} - names or any(
            not isinstance(w, VesselWall) for _, w in walls
        ):
            raise ValueError("Wall models must match network vessels")
        object.__setattr__(self, "walls", walls)
        from .actuator import ConnectionActuator

        actuators = tuple(sorted(dict(self.actuators).items()))
        if {name for name, _ in actuators} - {c.name for c in connections} or any(
            not isinstance(a, ConnectionActuator) for _, a in actuators
        ):
            raise ValueError("Actuators must reference existing connections")
        object.__setattr__(self, "actuators", actuators)
        from .separator_geometry import SeparatorGeometry

        geometries = tuple(sorted(dict(self.geometries).items()))
        for name, geometry in geometries:
            if name not in names or not isinstance(geometry, SeparatorGeometry):
                raise ValueError("Separator geometry must reference a vessel")
            if not math.isclose(geometry.volume_m3, dict(items)[name].volume_m3, rel_tol=1e-10):
                raise ValueError("Separator geometry and vessel volume differ")
        object.__setattr__(self, "geometries", geometries)

    def inspect_actuators(self, positions, modes=None):
        models = dict(self.actuators)
        positions = dict(positions or {})
        modes = dict(modes or {})
        if set(positions) != set(models) or set(modes) - set(models):
            raise ValueError("Actuator positions must exactly match configured actuators")
        for name, model in models.items():
            model.position(positions[name])
            if modes.get(name, "normal") not in ("normal", "fail_safe", "stuck"):
                raise ValueError("Unknown actuator mode")
        return positions, modes

    def initialize(self, fluid, specifications):
        if set(specifications) != {name for name, _ in self.vessels}:
            raise ValueError("Initial specifications must match vessel names")
        walls = dict(self.walls)
        states = {}
        for name, v in self.vessels:
            spec = dict(specifications[name])
            temperature = spec.pop("wall_temperature_k", None)
            state = v.initialize(fluid, **spec)
            if name in walls:
                if temperature is None:
                    raise ValueError("Wall-bearing vessel needs explicit wall_temperature_k")
                state = walls[name].attach(state, temperature)
            elif temperature is not None:
                raise ValueError("Wall temperature supplied without a wall model")
            states[name] = state
        return states

    def inspect_walls(self, states):
        walls = dict(self.walls)
        for name, state in states.items():
            if name in walls:
                walls[name].inspect(state)
            elif state.wall_temperature_k is not None:
                raise PropertyModelError("Wall-bearing state supplied to an unwalled vessel")

    def _step(
        self,
        states,
        fluid,
        dt,
        feeds,
        withdrawals,
        heat,
        factors,
        ambient_temperatures=None,
        actuator_positions=None,
        actuator_modes=None,
    ):
        positions, modes = self.inspect_actuators(actuator_positions, actuator_modes)
        actuator_records = {
            name: model.step(positions[name], factors.get(name, 1.0), dt, modes.get(name, "normal"))
            for name, model in self.actuators
        }
        factors = {**factors, **{name: r["final_position"] for name, r in actuator_records.items()}}
        vessels = dict(self.vessels)
        names = tuple(vessels)
        nc = len(fluid.components)
        walls = dict(self.walls)
        coupled_wall = bool(walls)
        block = nc + 2 + int(coupled_wall)
        ambient_temperatures = ambient_temperatures or {}
        self.inspect_walls(states)
        initial = {name: np.array(states[name].component_moles) for name in names}
        old_info = {name: vessels[name].inspect(states[name], fluid) for name in names}
        fin = {}
        hin = {}
        sin = {}
        guess = []
        lower = []
        upper = []
        nscale = {}
        escale = {}
        for name, v in self.vessels:
            incoming = feeds.get(name, ())
            fin[name] = (
                np.sum([s.component_molar_rates_mol_s for s in incoming], axis=0)
                if incoming
                else np.zeros(nc)
            )
            hin[name] = math.fsum(s.enthalpy_rate_w for s in incoming)
            sin[name] = math.fsum(
                s.molar_rate_mol_s * s.state(fluid)["molar_entropy_j_mol_k"]
                for s in incoming
                if s.molar_rate_mol_s
            )
            upper_p = min(
                [v.max_pressure_pa] + [s.pressure_pa for s in incoming if s.molar_rate_mol_s]
            )
            if upper_p < states[name].pressure_pa or upper_p <= v.min_pressure_pa:
                raise PropertyModelError(
                    "Passive feed supply pressure below network vessel pressure"
                )
            connection_capacity = math.fsum(
                c.max_molar_rate_mol_s
                for c in self.connections
                if c.target == name or c.allow_reverse and c.source == name
            )
            upper_n = initial[name] + dt * (fin[name] + connection_capacity)
            guess.extend(
                [
                    *np.log(initial[name]),
                    math.log(states[name].pressure_pa),
                    states[name].temperature_k,
                ]
            )
            lower.extend(
                [
                    *[math.log(v.minimum_component_moles)] * nc,
                    math.log(v.min_pressure_pa),
                    max(c.min_temperature_k for c in fluid.components),
                ]
            )
            upper.extend(
                [
                    *np.log(upper_n * (1 + 1e-10)),
                    math.log(upper_p),
                    min(c.max_temperature_k for c in fluid.components),
                ]
            )
            if coupled_wall:
                wall = walls.get(name)
                guess.append(
                    states[name].wall_temperature_k
                    if wall is not None
                    else states[name].temperature_k
                )
                lower.append(
                    wall.min_temperature_k
                    if wall
                    else max(c.min_temperature_k for c in fluid.components)
                )
                upper.append(
                    wall.max_temperature_k
                    if wall
                    else min(c.max_temperature_k for c in fluid.components)
                )
            nscale[name] = np.maximum(initial[name], dt * fin[name])
            escale[name] = max(
                1.0,
                abs(states[name].internal_energy_j),
                sum(initial[name]) * R * states[name].temperature_k,
                abs(dt * (hin[name] + heat.get(name, {}).get("heat_w", 0.0))),
            )
        local_cache = {name: {} for name in names}
        global_cache = {}
        connection_cache = {}
        evaluations = 0

        def evaluate(x, linearizations=None):
            nonlocal evaluations
            key = tuple(x)
            if linearizations is None and key in global_cache:
                return global_cache[key]
            values = {}
            for i, name in enumerate(names):
                part = x[i * block : (i + 1) * block]
                part_key = tuple(part[: nc + 2])
                if part_key not in local_cache[name]:
                    n = np.exp(part[:nc])
                    p = math.exp(part[nc])
                    t = part[nc + 1]
                    q, volume, energy, entropy = vessels[name]._values(fluid, n, p, t)
                    local_cache[name][part_key] = (n, p, t, q, volume, energy, entropy)
                    evaluations += 1
                values[name] = local_cache[name][part_key]
            net = {name: np.array(fin[name], copy=True) for name in names}
            energy_net = dict(hin)
            entropy_net = dict(sin)
            ext_out = {}
            ext_hout = {}
            link_reports = []
            for name in names:
                n, _, _, q, _, _, _ = values[name]
                ext_out[name] = np.zeros(nc)
                ext_hout[name] = 0.0
                for port, rate in withdrawals.get(name, {}).items():
                    if isinstance(rate, LevelWithdrawal):
                        geometry = dict(self.geometries)[name]
                        rate = rate.rate(geometry.from_equilibrium(n, q))
                    if not rate:
                        continue
                    z, h, s = _port(q, n, port)
                    ext_out[name] += rate * z
                    ext_hout[name] += rate * h
                    net[name] -= rate * z
                    energy_net[name] -= rate * h
                    entropy_net[name] -= rate * s
            pressures = {name: values[name][1] for name in names}
            for c in self.connections:
                if linearizations is not None and c.name in linearizations:
                    signed = linearizations[c.name](pressures, values)
                else:
                    signed = c.flow(
                        pressures,
                        validate=False,
                        factor=factors.get(c.name, 1.0),
                        values=values,
                        fluid=fluid,
                        cache=connection_cache,
                    )
                if signed == 0:
                    link_reports.append(
                        {
                            "name": c.name,
                            "flow_model": getattr(c, "flow_model", "linear_molar_conductance"),
                            **connection_cache.get(("diagnostic", c.name), {}),
                            "molar_rate_mol_s": 0.0,
                            "component_rates_mol_s": [0.0] * nc,
                            "enthalpy_rate_w": 0.0,
                        }
                    )
                    continue
                source, target, port = (
                    (c.source, c.target, c.source_port)
                    if signed > 0
                    else (c.target, c.source, c.target_port)
                )
                n, _, _, q, _, _, _ = values[source]
                z, h, s = _port(q, n, port)
                rate = abs(signed)
                flux = rate * z
                power = rate * h
                entropy_flux = rate * s
                net[source] -= flux
                net[target] += flux
                energy_net[source] -= power
                energy_net[target] += power
                entropy_net[source] -= entropy_flux
                entropy_net[target] += entropy_flux
                link_reports.append(
                    {
                        "name": c.name,
                        "flow_model": getattr(c, "flow_model", "linear_molar_conductance"),
                        **connection_cache.get(("diagnostic", c.name), {}),
                        "molar_rate_mol_s": signed,
                        "source": source,
                        "target": target,
                        "component_rates_mol_s": flux.tolist(),
                        "enthalpy_rate_w": power,
                    }
                )
            residual = []
            detail = {}
            for node_index, (name, v) in enumerate(self.vessels):
                n, p, t, q, volume, energy, entropy = values[name]
                command = heat.get(name, {})
                qw = command.get("heat_w", 0.0)
                wall = walls.get(name)
                tw = x[node_index * block + nc + 2] if coupled_wall else None
                qf = wall.fluid_ua_w_k * (tw - t) if wall else 0.0
                tamb = ambient_temperatures.get(name, wall.ambient_temperature_k) if wall else None
                qa = wall.ambient_ua_w_k * (tamb - tw) if wall else 0.0
                wres = (
                    (
                        wall.heat_capacity_j_k * (tw - states[name].wall_temperature_k)
                        - dt * (qa - qf)
                    )
                    if wall
                    else 0.0
                )
                wsgen = (
                    (
                        wall.entropy(tw)
                        - wall.entropy(states[name].wall_temperature_k)
                        - dt * (qa / tamb - qf / tw)
                    )
                    if wall
                    else 0.0
                )
                mass = n - initial[name] - dt * net[name]
                eres = energy - states[name].internal_energy_j - dt * (energy_net[name] + qw + qf)
                sgen = (
                    entropy
                    - old_info[name]["entropy_j_k"]
                    - dt
                    * (
                        entropy_net[name]
                        + (qw / command["utility_temperature_k"] if qw else 0.0)
                        + (qf / tw if wall else 0.0)
                    )
                )
                residual.extend(
                    [
                        *(mass / nscale[name]),
                        eres / escale[name],
                        (volume - v.volume_m3) / v.volume_m3,
                    ]
                )
                if coupled_wall:
                    residual.append(
                        wres / max(1.0, wall.heat_capacity_j_k * states[name].wall_temperature_k)
                        if wall
                        else (tw - t) / max(1.0, t)
                    )
                detail[name] = {
                    "mass": mass,
                    "energy": eres,
                    "entropy": sgen,
                    "wall_temperature": tw,
                    "wall_residual": wres,
                    "wall_entropy": wsgen,
                    "wall_to_fluid": qf,
                    "ambient_heat": qa,
                }
            result = (np.array(residual), values, detail, link_reports, ext_out, ext_hout)
            if linearizations is None:
                global_cache[key] = result
            return result

        required_difference = max(
            [0.0] + [getattr(c, "relative_flow_difference_step", 0.0) for c in self.connections]
        )

        def jacobian(x):
            base = evaluate(x)
            values = base[1]
            pressures = {name: values[name][1] for name in names}
            # Restore each full physical connection at the base before asking
            # for a tangent; a preceding Jacobian probe may have changed its
            # private diagnostic cache, never the physical residual cache.
            tangents = {}
            for connection in self.connections:
                if hasattr(connection, "linearized_flow"):
                    connection.flow(
                        pressures,
                        validate=False,
                        factor=factors.get(connection.name, 1.0),
                        values=values,
                        fluid=fluid,
                        cache=connection_cache,
                    )
                    tangent = connection.linearized_flow(
                        pressures,
                        values,
                        fluid,
                        factors.get(connection.name, 1.0),
                        connection_cache,
                    )
                    if tangent is not None:
                        tangents[connection.name] = tangent
            matrix = np.empty((len(x), len(x)))
            for j in range(len(x)):
                h = (
                    required_difference
                    if j % block < nc + 1
                    else required_difference * max(1.0, abs(x[j]))
                )
                probe = np.array(x, copy=True)
                if x[j] + h >= upper[j]:
                    h = -h
                if x[j] + h <= lower[j]:
                    h = min(required_difference, max(1e-12, (upper[j] - x[j]) * 0.5))
                probe[j] += h
                matrix[:, j] = (evaluate(probe, tangents)[0] - base[0]) / h
            return matrix

        fit = least_squares(
            lambda x: evaluate(x)[0],
            guess,
            bounds=(lower, upper),
            jac=jacobian if required_difference else "2-point",
            xtol=1e-11,
            ftol=1e-11,
            gtol=None,
            max_nfev=100,
        )
        residual, values, detail, links, ext_out, ext_hout = evaluate(fit.x)
        if not fit.success or max(abs(residual)) > 1e-9:
            raise HydraulicConvergenceError("Monolithic dynamic network solve failed")
        pressures = {name: values[name][1] for name in names}
        for c in self.connections:
            c.flow(
                pressures,
                validate=True,
                factor=factors.get(c.name, 1.0),
                values=values,
                fluid=fluid,
                cache=connection_cache,
            )
        result = {}
        reports = {}
        for name, v in self.vessels:
            n, p, t, _q, _volume, energy, entropy = values[name]
            old = states[name]
            command = heat.get(name, {})
            if (
                max(abs(n - initial[name]) / initial[name]) > v.max_fractional_inventory_change
                or abs(t - old.temperature_k) > v.max_temperature_change_k
            ):
                raise HydraulicConvergenceError("Network step state change exceeds accuracy guard")
            qw = command.get("heat_w", 0.0)
            if qw and (
                qw > 0
                and command["utility_temperature_k"] < max(t, old.temperature_k)
                or qw < 0
                and command["utility_temperature_k"] > min(t, old.temperature_k)
            ):
                raise PropertyModelError("Network heat reservoir crosses vessel temperature")
            if detail[name]["entropy"] < -1e-6 - 1e-9 * max(1.0, abs(entropy)):
                raise ConservationError("Network vessel discrete entropy balance is negative")
            result[name] = VesselState(tuple(n), energy, p, t, old.time_s + dt, old.model_id)
            if name in walls:
                wall = walls[name]
                tw = detail[name]["wall_temperature"]
                if abs(tw - old.wall_temperature_k) > wall.max_temperature_change_k:
                    raise HydraulicConvergenceError(
                        "Wall step temperature change exceeds accuracy guard"
                    )
                if abs(detail[name]["wall_residual"]) > 1e-6 + 1e-8 * max(
                    1.0,
                    dt * (abs(detail[name]["ambient_heat"]) + abs(detail[name]["wall_to_fluid"])),
                ):
                    raise ConservationError("Wall energy step balance failed")
                if detail[name]["wall_entropy"] < -1e-6:
                    raise ConservationError("Wall discrete entropy balance is negative")
                result[name] = wall.attach(result[name], tw)
            v.inspect(result[name], fluid)
            reports[name] = {
                "component_residuals_mol": detail[name]["mass"].tolist(),
                "energy_residual_j": detail[name]["energy"],
                "entropy_production_j_k": detail[name]["entropy"] + detail[name]["wall_entropy"],
                "wall_energy_residual_j": detail[name]["wall_residual"],
                "wall_to_fluid_heat_j": dt * detail[name]["wall_to_fluid"],
                "ambient_heat_j": dt * detail[name]["ambient_heat"],
            }
            if name in dict(self.geometries):
                reports[name]["liquid_level_m"] = dict(self.geometries)[name].from_equilibrium(
                    n, _q
                )
        global_mass = (
            np.sum([result[name].component_moles for name in names], axis=0)
            - np.sum([initial[name] for name in names], axis=0)
            - dt * np.sum([fin[name] - ext_out[name] for name in names], axis=0)
        )
        global_energy = math.fsum(
            result[name].internal_energy_j
            + (result[name].wall_energy_j or 0.0)
            - states[name].internal_energy_j
            - (states[name].wall_energy_j or 0.0)
            for name in names
        ) - dt * math.fsum(
            hin[name]
            - ext_hout[name]
            + heat.get(name, {}).get("heat_w", 0.0)
            + detail[name]["ambient_heat"]
            for name in names
        )
        if max(abs(global_mass)) > 1e-8 * max(1.0, sum(sum(n) for n in initial.values())) or abs(
            global_energy
        ) > 1e-5 + 1e-8 * sum(escale.values()):
            raise ConservationError("Dynamic network boundary balance failed")
        return result, {
            "dt_s": dt,
            "vessels": reports,
            "connections": links,
            "component_residuals_mol": global_mass.tolist(),
            "energy_residual_j": global_energy,
            "inlet_component_moles": (dt * np.sum(list(fin.values()), axis=0)).tolist(),
            "outlet_component_moles": (dt * np.sum(list(ext_out.values()), axis=0)).tolist(),
            "inlet_enthalpy_j": dt * math.fsum(hin.values()),
            "outlet_enthalpy_j": dt * math.fsum(ext_hout.values()),
            "heat_j": dt
            * math.fsum(
                [
                    *(c.get("heat_w", 0.0) for c in heat.values()),
                    *(detail[name]["ambient_heat"] for name in names),
                ]
            ),
            "direct_heat_j": dt * math.fsum(c.get("heat_w", 0.0) for c in heat.values()),
            "ambient_heat_j": dt * math.fsum(detail[name]["ambient_heat"] for name in names),
            "eos_evaluations": evaluations,
            "actuators": actuator_records,
        }

    def advance(
        self,
        states,
        fluid,
        duration_s,
        feeds=None,
        withdrawals=None,
        heat=None,
        initial_dt_s=1.0,
        maximum_dt_s=None,
        minimum_dt_s=1e-5,
        max_substeps=10000,
        connection_factors=None,
        stop_conditions=None,
        event_time_tolerance_s=1e-7,
        ambient_temperatures=None,
        actuator_positions=None,
        actuator_modes=None,
    ):
        positions, modes = self.inspect_actuators(actuator_positions, actuator_modes)
        vessels = dict(self.vessels)
        names = set(vessels)
        self.inspect_walls(states)
        ambient_temperatures = dict(ambient_temperatures or {})
        if set(ambient_temperatures) - set(dict(self.walls)):
            raise ValueError("Ambient temperatures require a configured wall")
        for t in ambient_temperatures.values():
            _number(t, "ambient_temperature_k", positive=True)
        factors = dict(connection_factors or {})
        if set(factors) - {c.name for c in self.connections}:
            raise ValueError("Unknown connection factor command")
        for factor in factors.values():
            if not 0 <= _number(factor, "conductance_factor") <= 1:
                raise ValueError("Conductance factor must be in [0,1]")
        if set(states) != names:
            raise ValueError("Network requires one state per vessel")
        feeds = {name: tuple(s) for name, s in (feeds or {}).items()}
        withdrawals = withdrawals or {}
        heat = heat or {}
        if (set(feeds) | set(withdrawals) | set(heat)) - names:
            raise ValueError("Unknown network vessel command")
        times = {s.time_s for s in states.values()}
        if len(times) != 1:
            raise ValueError("Network vessel clocks must agree")
        start = next(iter(times))
        duration = _number(duration_s, "duration_s", positive=True)
        dt = _number(initial_dt_s, "initial_dt_s", positive=True)
        maximum = _number(
            maximum_dt_s if maximum_dt_s is not None else dt, "maximum_dt_s", positive=True
        )
        minimum = _number(minimum_dt_s, "minimum_dt_s", positive=True)
        dt = min(dt, maximum)
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError("max_substeps must be positive integer")
        if not math.isfinite(start + duration) or start + duration <= start:
            raise ValueError("Network interval cannot advance representable time")
        for name, v in self.vessels:
            info = v.inspect(states[name], fluid)
            for s in feeds.get(name, ()):
                s._validate_fluid(fluid)
                if s.molar_rate_mol_s and s.pressure_pa < states[name].pressure_pa:
                    raise PropertyModelError("Passive network feed pressure is too low")
            for port, rate in withdrawals.get(name, {}).items():
                if isinstance(rate, LevelWithdrawal):
                    geometry = dict(self.geometries).get(name)
                    if geometry is None or port != "phase_0" or rate.setpoint_m > geometry.height_m:
                        raise ValueError(
                            "Level withdrawal needs geometry and a liquid phase_0 port"
                        )
                    rate = rate.rate(
                        geometry.from_equilibrium(states[name].component_moles, info["equilibrium"])
                    )
                _number(rate, "withdrawal_rate", nonnegative=True)
                if rate:
                    _port(info["equilibrium"], states[name].component_moles, port)
                elif port not in ("bulk", "phase_0", "phase_1"):
                    raise ValueError("Unknown network outlet port")
            command = heat.get(name, {})
            if set(command) - {"heat_w", "utility_temperature_k"}:
                raise ValueError("Unknown network heat command")
            qw = _number(command.get("heat_w", 0.0), "heat_w")
            if qw:
                _number(
                    command.get("utility_temperature_k"), "utility_temperature_k", positive=True
                )
        pressures = {name: s.pressure_pa for name, s in states.items()}
        initial_values = {
            name: (
                np.array(s.component_moles),
                s.pressure_pa,
                s.temperature_k,
                vessels[name].inspect(s, fluid)["equilibrium"],
                None,
                None,
                None,
            )
            for name, s in states.items()
        }
        for c in self.connections:
            signed = c.flow(
                pressures,
                factor=positions.get(c.name, factors.get(c.name, 1.0)),
                values=initial_values,
                fluid=fluid,
            )
            if signed:
                source, port = (
                    (c.source, c.source_port) if signed > 0 else (c.target, c.target_port)
                )
                q = vessels[source].inspect(states[source], fluid)["equilibrium"]
                _port(q, states[source].component_moles, port)
        conditions = tuple(stop_conditions or ())
        if conditions:
            _number(event_time_tolerance_s, "event_time_tolerance_s", positive=True)
            if len({r.name for r in conditions}) != len(conditions):
                raise ValueError("Duplicate event condition name")
            for rule in conditions:
                rule.validate(self, fluid)
            initial_events = [
                r.record(states, self, fluid, initial=True)
                for r in conditions
                if r.gap(states, self, fluid) <= 0
            ]
            if initial_events:
                return {
                    "states": {name: s.to_dict() for name, s in states.items()},
                    "steps": [],
                    "failed_steps": 0,
                    "duration_s": 0.0,
                    "requested_duration_s": duration,
                    "events": sorted(initial_events, key=lambda e: e["name"]),
                    "actuator_positions": positions,
                }
        current = dict(states)
        elapsed = 0.0
        records = []
        failures = 0
        detected = []
        while elapsed < duration:
            if len(records) >= max_substeps:
                raise HydraulicConvergenceError("Network substep budget exhausted")
            step = min(dt, duration - elapsed)
            try:
                new, record = self._step(
                    current,
                    fluid,
                    step,
                    feeds,
                    withdrawals,
                    heat,
                    factors,
                    ambient_temperatures=ambient_temperatures,
                    actuator_positions=positions,
                    actuator_modes=modes,
                )
                if conditions:
                    from .network_events import locate_network_event

                    event = locate_network_event(
                        self,
                        fluid,
                        current,
                        new,
                        record,
                        step,
                        feeds,
                        withdrawals,
                        heat,
                        factors,
                        conditions,
                        event_time_tolerance_s,
                        ambient_temperatures=ambient_temperatures,
                        actuator_positions=positions,
                        actuator_modes=modes,
                    )
                    if event is not None:
                        step, new, record, triggered = event
                        if not step:
                            detected = [r.record(current, self, fluid) for r in triggered]
                            break
                        detected = [r.record(new, self, fluid) for r in triggered]
            except (HydraulicConvergenceError, PropertyModelError) as error:
                failures += 1
                if step / 2 < minimum:
                    raise HydraulicConvergenceError(
                        f"Network interval rejected at minimum dt: {error}"
                    ) from error
                dt = step / 2
                continue
            next_elapsed = math.fsum([elapsed, step])
            if (
                next_elapsed <= elapsed
                or start + next_elapsed <= next(iter(current.values())).time_s
            ):
                raise ValueError("Network substep cannot advance representable time")
            elapsed = next_elapsed
            current = {name: replace(state, time_s=start + elapsed) for name, state in new.items()}
            record["states"] = {name: s.to_dict() for name, s in current.items()}
            records.append(record)
            positions = {name: r["final_position"] for name, r in record["actuators"].items()}
            if detected:
                break
            dt = min(maximum, step * 1.5)
        result = {
            "states": {name: s.to_dict() for name, s in current.items()},
            "steps": records,
            "failed_steps": failures,
            "duration_s": elapsed if conditions else duration,
            "actuator_positions": positions,
        }
        if conditions:
            result.update(requested_duration_s=duration, events=detected)
        return result
