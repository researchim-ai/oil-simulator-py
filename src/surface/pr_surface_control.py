"""Conservative low-drop HEM valves, actuators and PI surface feedback.

A valve is a resistance in a resolved pipe control volume, with binary port
isolation at zero position. Coarse/fine control integration includes actuator,
integrator and physical plant state. Not a cavitating/choked valve model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .actuator import ConnectionActuator
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _name, _number
from .operation import _digest
from .pr_gravity_transport import PRGravityTransportNetwork, PRGravityTransportState


@dataclass(frozen=True)
class PRControlValve:
    name: str
    port: str
    full_open_loss_coefficient: float
    actuator: ConnectionActuator
    characteristic: str = "linear"
    rangeability: float = 50.0
    maximum_pressure_drop_fraction: float = 0.05
    maximum_mach_number: float = 0.3

    def __post_init__(self):
        _name(self.name)
        _name(self.port)
        _number(self.full_open_loss_coefficient, "valve full-open K", positive=True)
        if not isinstance(self.actuator, ConnectionActuator):
            raise TypeError("Expected ConnectionActuator")
        if self.characteristic not in ("linear", "modified_equal_percentage"):
            raise ValueError("Unknown effective-flow characteristic")
        if _number(self.rangeability, "rangeability", positive=True) <= 1:
            raise ValueError("Rangeability must exceed one")
        if (
            not 0 < self.maximum_pressure_drop_fraction <= 0.1
            or not 0 < self.maximum_mach_number <= 0.5
        ):
            raise ValueError("Low-drop subsonic valve envelope required")

    def coefficient(self, position):
        x = ConnectionActuator.position(position)
        if x == 0:
            return None
        fraction = (
            x
            if self.characteristic == "linear"
            else math.expm1(x * math.log(self.rangeability)) / (self.rangeability - 1)
        )
        k = self.full_open_loss_coefficient / fraction**2
        if not math.isfinite(k):
            raise PropertyModelError("Valve coefficient exceeds representable domain")
        return k


@dataclass(frozen=True)
class PRSurfacePI:
    valve: str
    setpoint: float
    proportional_gain: float
    integral_gain_per_s: float = 0.0
    measurement: str = "pressure"
    component: str | None = None
    minimum_mole_fraction: float = 0.5
    opening_bias: float = 0.5
    action: int = 1
    tracking_time_s: float = 1.0
    maximum_integral_abs: float = 10.0

    def __post_init__(self):
        _name(self.valve)
        _number(self.setpoint, "PI setpoint", nonnegative=True)
        _number(self.proportional_gain, "PI proportional gain", nonnegative=True)
        _number(self.integral_gain_per_s, "PI integral gain", nonnegative=True)
        _number(self.tracking_time_s, "PI tracking time", positive=True)
        _number(self.maximum_integral_abs, "PI integral limit", positive=True)
        ConnectionActuator.position(self.opening_bias)
        if (
            self.measurement not in ("pressure", "level")
            or type(self.action) is not int
            or self.action not in (-1, 1)
        ):
            raise ValueError("Explicit pressure/level measurement and action +/-1 required")
        if (self.measurement == "level") != (self.component is not None):
            raise ValueError("Level measurement requires exactly one component")
        if self.component is not None:
            _name(self.component)
        if not 0 < self.minimum_mole_fraction <= 1:
            raise ValueError("Invalid layer identification threshold")

    def update(self, measurement, integral, position, dt, manual=None):
        error = self.action * (measurement - self.setpoint)
        proportional = self.proportional_gain * error
        if manual is not None:
            target = ConnectionActuator.position(manual)
            updated = target - self.opening_bias - proportional
        else:
            # External-reset feedback tracks the actual actuator. For held
            # error/position this linear anti-windup ODE has an exact solution.
            equilibrium = (
                self.tracking_time_s * self.integral_gain_per_s * error
                + position
                - self.opening_bias
                - proportional
            )
            updated = equilibrium + (integral - equilibrium) * math.exp(-dt / self.tracking_time_s)
            target = min(1.0, max(0.0, self.opening_bias + proportional + updated))
        updated = min(self.maximum_integral_abs, max(-self.maximum_integral_abs, updated))
        if manual is None:
            target = min(1.0, max(0.0, self.opening_bias + proportional + updated))
        return target, updated, error


@dataclass(frozen=True)
class PRControlledSurfaceState:
    plant: PRGravityTransportState
    positions: tuple
    integrals: tuple
    next_step_s: float
    model_id: str

    def __post_init__(self):
        object.__setattr__(self, "positions", tuple((n, float(v)) for n, v in self.positions))
        object.__setattr__(self, "integrals", tuple((n, float(v)) for n, v in self.integrals))

    @property
    def time_s(self):
        return self.plant.time_s

    def to_dict(self):
        return {
            "plant": self.plant.to_dict(),
            "positions": dict(self.positions),
            "integrals": dict(self.integrals),
            "next_step_s": self.next_step_s,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"plant", "positions", "integrals", "next_step_s", "model_id"}:
            raise ValueError("Unknown controlled checkpoint fields")
        return cls(
            PRGravityTransportState.from_dict(data["plant"]),
            tuple(sorted(data["positions"].items())),
            tuple(sorted(data["integrals"].items())),
            data["next_step_s"],
            data["model_id"],
        )


class PRControlledSurface:
    def __init__(
        self,
        plant,
        valves,
        controllers=(),
        *,
        initial_step_s=0.001,
        minimum_step_s=1e-8,
        maximum_step_s=0.1,
        position_tolerance=1e-4,
        integral_tolerance=1e-4,
        maximum_attempts=1000,
    ):
        if not isinstance(plant, PRGravityTransportNetwork):
            raise TypeError("Expected gravitational transport plant")
        self.plant = plant
        self.valves = tuple(sorted(valves, key=lambda v: v.name))
        self.controllers = tuple(sorted(controllers, key=lambda c: c.valve))
        if (
            not self.valves
            or any(not isinstance(v, PRControlValve) for v in self.valves)
            or len({v.name for v in self.valves}) != len(self.valves)
        ):
            raise ValueError("Unique named control valves required")
        ports = {p.name: p for p in plant.ports}
        if len({v.port for v in self.valves}) != len(self.valves) or any(
            v.port not in ports for v in self.valves
        ):
            raise ValueError("Each valve requires its own gravitational port")
        self._ports = ports
        self._valves = {v.name: v for v in self.valves}
        if any(not isinstance(c, PRSurfacePI) for c in self.controllers) or len(
            {c.valve for c in self.controllers}
        ) != len(self.controllers):
            raise ValueError("At most one PI controller per valve")
        for v in self.valves:
            node = plant.transport.nodes[plant.transport.names.index(ports[v.port].node)]
            if (
                node.drift_flux is not None
                or node.distribution_strength
                or node.buoyancy_relaxation_s
            ):
                raise ValueError(
                    "Valve control volume requires HEM; slip remains allowed elsewhere"
                )
        for c in self.controllers:
            if c.valve not in self._valves:
                raise ValueError("Unknown controlled valve")
            column = plant.columns[ports[self._valves[c.valve].port].column]
            if c.component is not None and c.component not in column.fluid.names:
                raise ValueError("Unknown controlled level component")
            if c.measurement == "level" and c.setpoint > column.geometry.height_m:
                raise ValueError("Level setpoint above column")
        for key, value in [
            ("initial_step_s", initial_step_s),
            ("minimum_step_s", minimum_step_s),
            ("maximum_step_s", maximum_step_s),
            ("position_tolerance", position_tolerance),
            ("integral_tolerance", integral_tolerance),
        ]:
            setattr(self, key, _number(value, key, positive=True))
        if not minimum_step_s <= initial_step_s <= maximum_step_s:
            raise ValueError("Invalid controller integration bounds")
        if type(maximum_attempts) is not int or maximum_attempts < 1:
            raise ValueError("Positive attempt budget required")
        self.maximum_attempts = maximum_attempts
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_controlled_surface_v1",
            "plant": self.plant.to_dict(),
            "valves": [asdict(v) for v in self.valves],
            "controllers": [asdict(c) for c in self.controllers],
            **{
                k: getattr(self, k)
                for k in (
                    "initial_step_s",
                    "minimum_step_s",
                    "maximum_step_s",
                    "position_tolerance",
                    "integral_tolerance",
                    "maximum_attempts",
                )
            },
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        data = dict(data)
        if data.pop("model", None) != "pr_controlled_surface_v1":
            raise ValueError("Unknown controlled surface model")
        data["plant"] = PRGravityTransportNetwork.from_dict(data["plant"], device=device)
        data["valves"] = [
            PRControlValve(**{**v, "actuator": ConnectionActuator(**v["actuator"])})
            for v in data["valves"]
        ]
        data["controllers"] = [PRSurfacePI(**c) for c in data["controllers"]]
        return cls(**data)

    def initialize(self, plant_state, positions, integrals=None):
        values = {c.valve: 0.0 for c in self.controllers}
        values.update(integrals or {})
        s = PRControlledSurfaceState(
            plant_state,
            tuple(sorted(positions.items())),
            tuple(sorted(values.items())),
            self.initial_step_s,
            self.model_id,
        )
        self.inspect(s)
        return s

    def inspect(self, state):
        if (
            not isinstance(state, PRControlledSurfaceState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
        ):
            raise PropertyModelError("Controlled model/checkpoint identity differs")
        self.plant.inspect(state.plant)
        if tuple(n for n, _ in state.positions) != tuple(v.name for v in self.valves) or tuple(
            n for n, _ in state.integrals
        ) != tuple(c.valve for c in self.controllers):
            raise ValueError("Controller state registry differs")
        for _, value in state.positions:
            ConnectionActuator.position(value)
        for c, (_, value) in zip(self.controllers, state.integrals):
            if not math.isfinite(value) or abs(value) > c.maximum_integral_abs:
                raise PropertyModelError("Controller integral outside domain")
        if not self.minimum_step_s <= state.next_step_s <= self.maximum_step_s:
            raise ValueError("Controller restart step outside domain")

    def measure(self, state, controller):
        port = self._ports[self._valves[controller.valve].port]
        column = self.plant.columns[port.column]
        s = dict(state.plant.columns)[port.column]
        if controller.measurement == "pressure":
            return column.sample(s, port.height_m, side="below")["pressure_pa"]
        index = column.fluid.names.index(controller.component)
        f = column.fluid
        seed = (
            [f.tensor(x) for x in s.phase_seeds],
            list(s.phase_roots),
            f.tensor(s.interface_heights_m),
            math.exp(s.solution_variables[0]),
            s.temperature_k,
            f.tensor(s.chemical_constants),
        )
        data = column._problem(
            f.tensor(s.component_moles), s.total_energy_j, seed, s.quadrature_order
        )[3](f.tensor(s.solution_variables))[1]
        phase = data["phase_component_moles"]
        fraction = phase[:, index] / phase.sum(-1)
        peak = int(fraction.argmax().cpu())
        if float(fraction[peak].cpu()) < controller.minimum_mole_fraction:
            return 0.0
        return (
            s.interface_heights_m[peak]
            if peak < len(s.interface_heights_m)
            else column.geometry.height_m
        )

    def _step(self, state, dt, manual, modes, commands):
        positions = dict(state.positions)
        integrals = dict(state.integrals)
        targets = dict(manual)
        measurements = {}
        for c in self.controllers:
            value = self.measure(state, c)
            measurements[c.valve] = value
            targets[c.valve], integrals[c.valve], _ = c.update(
                value, integrals[c.valve], positions[c.valve], dt, manual.get(c.valve)
            )
        for v in self.valves:
            if v.name not in targets and modes.get(v.name) in ("fail_safe", "stuck"):
                targets[v.name] = positions[v.name]
        if set(targets) != {v.name for v in self.valves}:
            raise ValueError("Every valve requires feedback or an explicit manual target")
        openings = dict(commands.get("openings") or {})
        losses = dict(commands.get("loss_coefficients") or {})
        known_links = {p.name for p in (*self.plant.transport.links, *self.plant.ports)}
        if set(openings) - known_links:
            raise ValueError("Unknown hydraulic isolation command")
        physical = {
            n.name: n.local_loss_coefficient
            for n in self.plant.transport.nodes
            if n.kind == "pipe_cell"
        }
        if set(losses) - set(physical):
            raise ValueError("Unknown external resistance command")
        for n, k in losses.items():
            physical[n] = _number(k, "external resistance", nonnegative=True)
        records = {}
        coefficients = {}
        for v in self.valves:
            record = v.actuator.step(
                positions[v.name], targets[v.name], dt, modes.get(v.name, "normal")
            )
            x = record["final_position"]
            positions[v.name] = x
            records[v.name] = record
            actual = openings.get(v.port, True) and x > 0
            openings[v.port] = actual
            k = v.coefficient(x) if actual else None
            coefficients[v.name] = k
            if k is not None:
                physical[self._ports[v.port].node] += k
        plant_report = self.plant.advance(
            state.plant, dt, **{**commands, "openings": openings, "loss_coefficients": physical}
        )
        plant_state = PRGravityTransportState.from_dict(plant_report["state"])
        envelope = {
            v.name: {"maximum_pressure_drop_fraction": 0.0, "maximum_mach_number": 0.0}
            for v in self.valves
        }
        # Check each accepted hydraulic substep, not only the report endpoint.
        for step in plant_report["steps"]:
            snapshot = PRGravityTransportState.from_dict(step["state"])
            for v in self.valves:
                k = coefficients[v.name]
                if k is None:
                    continue
                port = self._ports[v.port]
                i = self.plant.transport.names.index(port.node)
                cell = dict(snapshot.transport.nodes)[port.node]
                d = self.plant.transport._state_data(i, cell)
                drop = 0.5 * float((d["rho"] * d["u"].square()).cpu()) * k
                cs = dict(snapshot.columns)[port.column]
                p = self.plant.columns[port.column].sample(cs, port.height_m, side="below")[
                    "pressure_pa"
                ]
                fraction = drop / min(p, cell.pressure_pa)
                envelope[v.name]["maximum_pressure_drop_fraction"] = max(
                    envelope[v.name]["maximum_pressure_drop_fraction"], fraction
                )
                if fraction > v.maximum_pressure_drop_fraction:
                    raise PropertyModelError("Valve exceeds low-pressure-drop envelope")
                speeds = step["transport"]["nodes"][port.node]["characteristic_speeds_m_s"]
                u = float(d["u"].cpu())
                sound = max(abs(a - u) for a in speeds)
                mach = abs(u) / sound if sound > 0 else math.inf
                envelope[v.name]["maximum_mach_number"] = max(
                    envelope[v.name]["maximum_mach_number"], mach
                )
                if mach > v.maximum_mach_number:
                    raise PropertyModelError("Valve exceeds declared subsonic envelope")
        result = PRControlledSurfaceState(
            plant_state,
            tuple(sorted(positions.items())),
            tuple(sorted(integrals.items())),
            state.next_step_s,
            self.model_id,
        )
        self.inspect(result)
        return result, {
            "dt_s": dt,
            "state": result.to_dict(),
            "plant": plant_report,
            "actuators": records,
            "measurements": measurements,
            "effective_valve_K": coefficients,
            "valve_envelope": envelope,
            "total_heat_j": plant_report["total_heat_j"],
        }

    def _error(self, a, b):
        e = self.plant._error(a.plant, b.plant)
        for (_, x), (_, y) in zip(a.positions, b.positions):
            e = max(e, abs(x - y) / self.position_tolerance)
        for (_, x), (_, y) in zip(a.integrals, b.integrals):
            e = max(e, abs(x - y) / self.integral_tolerance)
        return e

    def advance(self, state, duration_s, *, targets=None, actuator_modes=None, **plant_commands):
        from dataclasses import replace

        self.inspect(state)
        end = state.time_s + _number(duration_s, "controlled duration", positive=True)
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Unrepresentable controlled interval")
        manual = dict(targets or {})
        modes = dict(actuator_modes or {})
        if (set(manual) | set(modes)) - set(self._valves):
            raise ValueError("Unknown actuator command")
        for value in manual.values():
            ConnectionActuator.position(value)
        if any(v not in ("normal", "fail_safe", "stuck") for v in modes.values()):
            raise ValueError("Unknown actuator mode")
        if set(plant_commands) - {
            "openings",
            "heat_w",
            "utility_temperature_k",
            "loss_coefficients",
        }:
            raise ValueError("Unknown controlled hydraulic command")
        current = state
        dt = min(state.next_step_s, duration_s)
        steps = []
        attempts = 0
        while current.time_s < end:
            attempts += 1
            if attempts > self.maximum_attempts:
                raise HydraulicConvergenceError("Controlled interval budget exhausted")
            dt = min(dt, end - current.time_s)
            if current.time_s + dt / 2 <= current.time_s:
                raise ValueError("Controlled substep below time resolution")
            try:
                coarse, _ = self._step(current, dt, manual, modes, plant_commands)
                half, one = self._step(current, dt / 2, manual, modes, plant_commands)
                fine, two = self._step(half, dt / 2, manual, modes, plant_commands)
                error = self._error(coarse, fine)
                if error > 1:
                    raise HydraulicConvergenceError(
                        "Combined plant/control temporal error exceeded"
                    )
            except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
                dt /= 2
                if dt < self.minimum_step_s:
                    raise HydraulicConvergenceError("Controlled minimum step exhausted") from exc
                continue
            nextdt = min(
                self.maximum_step_s, max(self.minimum_step_s, dt * (1.5 if error < 0.25 else 1.0))
            )
            reached_end = dt >= end - current.time_s
            current = replace(fine, next_step_s=nextdt)
            if reached_end:
                current = replace(
                    current,
                    plant=replace(
                        current.plant, transport=replace(current.plant.transport, time_s=end)
                    ),
                )
            steps.extend((one, two))
            dt = nextdt
        heat = math.fsum(s["total_heat_j"] for s in steps)
        return {
            "state": current.to_dict(),
            "steps": steps,
            "attempts": attempts,
            "total_heat_j": heat,
            "conservation_residuals": self.plant._ledger(state.plant, current.plant, heat),
        }
