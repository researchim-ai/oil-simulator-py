"""Conservative low-drop HEM valves, actuators and PI surface feedback.

A valve is a resistance in a resolved pipe control volume, with binary port
isolation at zero position. Coarse/fine control integration includes actuator,
integrator and physical plant state. Not a cavitating/choked valve model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace

from .actuator import ConnectionActuator
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _name, _number
from .operation import _digest
from .pr_control_channels import PRMotorControl, PRSurfaceTrip, compressor_margin
from .pr_gravity_transport import PRGravityTransportNetwork, PRGravityTransportState
from .pr_plant_registry import (
    align_physical_clock,
    physical_plant_from_dict,
    physical_state_from_dict,
)
from .pr_profile_graph import ProfileGraphState, PRProfileGraphNetwork
from .time_interval import interval_step_s, split_step_s


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
    resistance_node: str | None = None

    def __post_init__(self):
        _name(self.name)
        _name(self.port)
        if self.resistance_node is not None:
            _name(self.resistance_node)
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
class PRThroatControlValve:
    """Actuator-to-effective-area law of a PR homogeneous equilibrium throat."""

    name: str
    port: str
    actuator: ConnectionActuator
    characteristic: str = "linear"
    rangeability: float = 50.0
    model: str = "hem_throat_area"

    def __post_init__(self):
        _name(self.name)
        _name(self.port)
        if not isinstance(self.actuator, ConnectionActuator):
            raise TypeError("Expected ConnectionActuator")
        if self.model != "hem_throat_area" or self.characteristic not in (
            "linear",
            "modified_equal_percentage",
        ):
            raise ValueError("Unknown critical valve area law")
        if _number(self.rangeability, "rangeability", positive=True) <= 1:
            raise ValueError("Rangeability must exceed one")

    def fraction(self, position):
        x = ConnectionActuator.position(position)
        return (
            x
            if self.characteristic == "linear"
            else math.expm1(x * math.log(self.rangeability)) / (self.rangeability - 1)
        )


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
    sensor: str | None = None

    def __post_init__(self):
        _name(self.valve)
        _number(
            self.setpoint, "PI setpoint", nonnegative=self.measurement not in ("speed", "pump_flow")
        )
        _number(self.proportional_gain, "PI proportional gain", nonnegative=True)
        _number(self.integral_gain_per_s, "PI integral gain", nonnegative=True)
        _number(self.tracking_time_s, "PI tracking time", positive=True)
        _number(self.maximum_integral_abs, "PI integral limit", positive=True)
        ConnectionActuator.position(self.opening_bias)
        if (
            self.measurement
            not in (
                "pressure",
                "temperature",
                "speed",
                "level",
                "inventory_level",
                "compressor_margin",
                "pump_flow",
            )
            or type(self.action) is not int
            or self.action not in (-1, 1)
        ):
            raise ValueError("Explicit pressure/level measurement and action +/-1 required")
        if (self.measurement in ("level", "inventory_level")) != (self.component is not None):
            raise ValueError("Level measurement requires exactly one component")
        if self.component is not None:
            _name(self.component)
        if self.sensor is not None:
            _name(self.sensor)
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
    plant: PRGravityTransportState | ProfileGraphState
    positions: tuple
    integrals: tuple
    next_step_s: float
    model_id: str
    trips: tuple = ()
    inactive_feedback: tuple = ()
    machine_feedback: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "positions", tuple((n, float(v)) for n, v in self.positions))
        object.__setattr__(self, "integrals", tuple((n, float(v)) for n, v in self.integrals))
        object.__setattr__(
            self, "trips", tuple((n, latched, float(elapsed)) for n, latched, elapsed in self.trips)
        )
        object.__setattr__(
            self, "machine_feedback", tuple((n, float(v)) for n, v in self.machine_feedback)
        )
        if len(dict(self.machine_feedback)) != len(self.machine_feedback):
            raise ValueError("Duplicate machine feedback sample")
        for name, value in self.machine_feedback:
            _name(name)
            _number(value, "machine feedback")
        object.__setattr__(self, "inactive_feedback", tuple(self.inactive_feedback))
        if len(set(self.inactive_feedback)) != len(self.inactive_feedback):
            raise ValueError("Duplicate inactive controller")
        for name in self.inactive_feedback:
            _name(name)
        for name, latched, elapsed in self.trips:
            _name(name)
            if type(latched) is not bool:
                raise TypeError("Protection latch must be boolean")
            _number(elapsed, "protection elapsed time", nonnegative=True)

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
            **({"machine_feedback": dict(self.machine_feedback)} if self.machine_feedback else {}),
            **(
                {"inactive_feedback": list(self.inactive_feedback)}
                if self.inactive_feedback
                else {}
            ),
            **(
                {
                    "trips": {
                        n: {"latched": latched, "elapsed_s": elapsed}
                        for n, latched, elapsed in self.trips
                    }
                }
                if self.trips
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) - {
            "plant",
            "positions",
            "integrals",
            "next_step_s",
            "model_id",
            "trips",
            "inactive_feedback",
            "machine_feedback",
        } or not {"plant", "positions", "integrals", "next_step_s", "model_id"} <= set(data):
            raise ValueError("Unknown controlled checkpoint fields")
        if not isinstance(data.get("machine_feedback", {}), dict):
            raise TypeError("Invalid machine feedback checkpoint")
        if not isinstance(data.get("trips", {}), dict) or any(
            set(v) != {"latched", "elapsed_s"} for v in data.get("trips", {}).values()
        ):
            raise ValueError("Invalid protection checkpoint fields")
        return cls(
            physical_state_from_dict(data["plant"]),
            tuple(sorted(data["positions"].items())),
            tuple(sorted(data["integrals"].items())),
            data["next_step_s"],
            data["model_id"],
            tuple(
                (n, v["latched"], v["elapsed_s"]) for n, v in sorted(data.get("trips", {}).items())
            ),
            tuple(data.get("inactive_feedback", ())),
            tuple(sorted(data.get("machine_feedback", {}).items())),
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
        trips=(),
        motors=(),
    ):
        if not isinstance(plant, (PRGravityTransportNetwork, PRProfileGraphNetwork)):
            raise TypeError("Expected gravitational or profile-graph transport plant")
        self.plant = plant
        self.valves = tuple(sorted(valves, key=lambda v: v.name))
        if any(not isinstance(v, (PRControlValve, PRThroatControlValve)) for v in self.valves):
            raise TypeError("Put motor channels in motors, not valves")
        self.motors = tuple(sorted(motors, key=lambda v: v.name))
        if any(not isinstance(v, PRMotorControl) for v in self.motors):
            raise TypeError("Expected motor control channels")
        self.channels = tuple(sorted((*self.valves, *self.motors), key=lambda v: v.name))
        self.controllers = tuple(sorted(controllers, key=lambda c: c.valve))
        self.trips = tuple(sorted(trips, key=lambda t: t.name))
        if any(not isinstance(t, PRSurfaceTrip) for t in self.trips) or len(
            {t.name for t in self.trips}
        ) != len(self.trips):
            raise ValueError("Unique named physical protections required")
        if (
            not self.channels
            or any(
                not isinstance(v, (PRControlValve, PRThroatControlValve, PRMotorControl))
                for v in self.channels
            )
            or len({v.name for v in self.channels}) != len(self.channels)
        ):
            raise ValueError("Unique named control valves required")
        ports = {p.name: p for p in plant.ports}
        by_name = {n.name: n for n in plant.transport.nodes}
        links = {c.name: c for c in plant.transport.links}
        restrictions = {u.name: u for u in plant.transport.restrictions}
        self._signed_machines = {
            u.name
            for u in (
                *plant.transport.pumps,
                *plant.transport.pump_connections,
                *plant.transport.compressors,
            )
            if getattr(u, "signed_map", None) is not None
        }
        for valve in self.channels:
            if isinstance(valve, PRMotorControl):
                if valve.port not in plant.transport._drives:
                    raise ValueError("Motor controller requires a coupled PR rotor")
                physics = plant.transport._drives[valve.port].drive
                if (
                    valve.minimum_torque_nm < -physics.max_braking_torque_nm
                    or valve.maximum_torque_nm > physics.max_motor_torque_nm
                ):
                    raise ValueError("Controller torque limits exceed the physical drive")
                peak_positive = valve.maximum_torque_nm
                peak_negative = -valve.minimum_torque_nm
                if valve.machine in self._signed_machines:
                    peak_positive = peak_negative = max(peak_positive, peak_negative)
                if (
                    physics.max_electrical_draw_w is not None
                    and peak_positive * physics.max_speed_rad_s / physics.motor_efficiency
                    > physics.max_electrical_draw_w
                ):
                    raise ValueError(
                        "Controller torque must respect electrical power over the declared speed range"
                    )
                if (
                    physics.max_electrical_export_w is not None
                    and peak_negative * physics.max_speed_rad_s * physics.regeneration_efficiency
                    > physics.max_electrical_export_w
                ):
                    raise ValueError("Controller braking torque exceeds the declared export power")
                drive = next(
                    u
                    for u in (
                        *plant.transport.pumps,
                        *plant.transport.compressors,
                        *plant.transport.pump_connections,
                    )
                    if u.name == valve.port
                )
                node = drive.node if hasattr(drive, "node") else drive.target
                ports[valve.port] = SimpleNamespace(
                    name=valve.port, node=node, sensor_node=node, column=None, motor=True
                )
                continue
            if isinstance(valve, PRThroatControlValve):
                if valve.port not in restrictions:
                    raise ValueError("Unknown controlled HEM restriction")
                unit = restrictions[valve.port]
                ports[valve.port] = SimpleNamespace(
                    name=valve.port,
                    node=unit.target,
                    column=None,
                    sensor_node=unit.target,
                    restriction=True,
                )
                continue
            if valve.port in ports:
                if valve.resistance_node not in (None, ports[valve.port].node):
                    raise ValueError("Column valve resistance must occupy its physical port body")
                continue
            if valve.port not in links:
                raise ValueError("Unknown controlled material connection")
            link = links[valve.port]
            candidates = [
                name
                for side, name in (("source", link.source), ("target", link.target))
                if by_name[name].kind == "pipe_cell"
                and getattr(link, side + "_port") in ("left", "right")
                and by_name[name].drift_flux is None
                and not by_name[name].distribution_strength
                and not by_name[name].buoyancy_relaxation_s
            ]
            body = valve.resistance_node
            if body is None and len(candidates) == 1:
                body = candidates[0]
            if body not in candidates:
                raise ValueError(
                    "Material valve needs an unambiguous HEM resistance_node on its link"
                )
            sensor_node = link.target if body == link.source else link.source
            ports[valve.port] = SimpleNamespace(
                name=valve.port, node=body, column=None, sensor_node=sensor_node
            )
        if len({v.port for v in self.channels}) != len(self.channels) or any(
            v.port not in ports for v in self.channels
        ):
            raise ValueError("Each valve requires its own material port")
        self._ports = ports
        self._valves = {v.name: v for v in self.channels}
        if any(not isinstance(c, PRSurfacePI) for c in self.controllers) or len(
            {c.valve for c in self.controllers}
        ) != len(self.controllers):
            raise ValueError("At most one PI controller per valve")
        for v in self.channels:
            if isinstance(v, PRMotorControl):
                continue
            node = plant.transport.nodes[plant.transport.names.index(ports[v.port].node)]
            if (
                node.drift_flux is not None
                or node.distribution_strength
                or node.buoyancy_relaxation_s
            ):
                raise ValueError(
                    "Valve control volume requires HEM; slip remains allowed elsewhere"
                )
        self._compressors = {u.name: u for u in plant.transport.compressors}
        self._pump_connections = {u.name: u for u in plant.transport.pump_connections}
        self._feedback_units = set()
        for c in self.controllers:
            if c.valve not in self._valves:
                raise ValueError("Unknown controlled valve")
            port = ports[self._valves[c.valve].port]
            if c.measurement == "pump_flow":
                if c.sensor not in self._pump_connections:
                    raise ValueError(
                        "Pump-flow feedback requires an actual-flow PR pump connection"
                    )
                if c.setpoint < 0 and c.sensor not in self._signed_machines:
                    raise ValueError("Negative flow setpoint requires an explicit signed pump map")
                self._feedback_units.add(c.sensor)
                continue
            if c.measurement == "compressor_margin":
                if c.sensor not in self._compressors or c.sensor in self._signed_machines:
                    raise ValueError(
                        "Anti-surge control requires an explicit forward compressor surge map"
                    )
                continue
            if c.measurement == "speed":
                sensor = c.sensor or self._valves[c.valve].port
                if (
                    sensor not in plant.transport._drives
                    or abs(c.setpoint) > plant.transport._drives[sensor].drive.max_speed_rad_s
                    or (c.setpoint < 0 and sensor not in self._signed_machines)
                ):
                    raise ValueError(
                        "Speed feedback needs a represented rotor and admissible setpoint"
                    )
                continue
            if c.measurement == "temperature":
                if (c.sensor or port.sensor_node) not in by_name:
                    raise ValueError("Temperature feedback requires a finite transport node")
                continue
            if c.measurement == "level" and c.sensor is not None:
                if c.sensor not in plant.columns:
                    raise ValueError(
                        "Level sensor must reference a represented gravitational column"
                    )
                column = plant.columns[c.sensor]
                if c.component not in column.fluid.names or c.setpoint > column.geometry.height_m:
                    raise ValueError("Remote column level sensor outside component/geometry domain")
                continue
            if port.column is None:
                if c.component is not None and c.component not in plant.transport.fluid.names:
                    raise ValueError("Unknown inventory-level component")
                if c.measurement == "level":
                    raise ValueError(
                        "Resolved vessels use inventory_level, not an equilibrium-column interface"
                    )
                if c.measurement == "pressure":
                    if (c.sensor or port.sensor_node) not in by_name:
                        raise ValueError("Unknown pressure sensor node")
                else:
                    self._inventory_path(c)
                continue
            if c.sensor is not None or c.measurement == "inventory_level":
                raise ValueError(
                    "Column controllers use their own pressure or interface-level sensor"
                )
            column = plant.columns[port.column]
            if c.component is not None and c.component not in column.fluid.names:
                raise ValueError("Unknown controlled level component")
            if c.measurement == "level" and c.setpoint > column.geometry.height_m:
                raise ValueError("Level setpoint above column")
        switches = {
            s.name
            for s in (
                *plant.transport.switches,
                *plant.ports,
                *plant.heat_links,
                *getattr(plant, "isolation_switches", ()),
            )
        }
        for trip in self.trips:
            if trip.target not in self._valves or set(trip.isolate_switches) - switches:
                raise ValueError("Protection target/isolation registry differs from plant")
            sensors = plant.transport._drives if trip.measurement == "speed" else by_name
            if trip.sensor not in sensors:
                raise ValueError("Unknown physical protection sensor")
            if (
                trip.measurement == "speed"
                and min(trip.trip_threshold, trip.reset_threshold) < 0
                and trip.sensor not in self._signed_machines
            ):
                raise ValueError(
                    "Negative speed protection requires an explicit signed machine map"
                )
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
            "valves": [
                {k: x for k, x in asdict(v).items() if k != "resistance_node" or x is not None}
                for v in self.valves
            ],
            **({"motors": [asdict(m) for m in self.motors]} if self.motors else {}),
            **({"trips": [asdict(t) for t in self.trips]} if self.trips else {}),
            "controllers": [
                {k: x for k, x in asdict(c).items() if k != "sensor" or x is not None}
                for c in self.controllers
            ],
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
        data["plant"] = physical_plant_from_dict(data["plant"], device=device)
        data["valves"] = [
            (
                {"hem_throat_area": PRThroatControlValve, "motor_torque": PRMotorControl}.get(
                    v.get("model"), PRControlValve
                )
            )(**{**v, "actuator": ConnectionActuator(**v["actuator"])})
            for v in data["valves"]
        ]
        data["motors"] = [
            PRMotorControl(**{**v, "actuator": ConnectionActuator(**v["actuator"])})
            for v in data.get("motors", ())
        ]
        data["controllers"] = [PRSurfacePI(**c) for c in data["controllers"]]
        data["trips"] = [PRSurfaceTrip(**t) for t in data.get("trips", ())]
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
            tuple((t.name, False, 0.0) for t in self.trips),
            tuple(c.valve for c in self.controllers if c.measurement == "pump_flow"),
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
        if tuple(n for n, _ in state.positions) != tuple(v.name for v in self.channels) or tuple(
            n for n, _ in state.integrals
        ) != tuple(c.valve for c in self.controllers):
            raise ValueError("Controller state registry differs")
        if tuple(n for n, _, _ in state.trips) != tuple(t.name for t in self.trips):
            raise ValueError("Protection checkpoint registry differs")
        for trip, (_, latched, elapsed) in zip(self.trips, state.trips):
            if elapsed > trip.delay_s or (
                trip.delay_s > 0 and elapsed == trip.delay_s and not latched
            ):
                raise ValueError("Protection timer exceeds configured delay")
        samples = dict(state.machine_feedback)
        if set(samples) - self._feedback_units or tuple(
            n for n, _ in state.machine_feedback
        ) != tuple(sorted(samples)):
            raise ValueError("Machine feedback checkpoint registry differs")
        for c in self.controllers:
            if (
                c.measurement == "pump_flow"
                and c.sensor not in samples
                and c.valve not in state.inactive_feedback
            ):
                raise ValueError("Active pump-flow controller lost its feedback sample")
        allowed = {
            c.valve for c in self.controllers if c.measurement in ("compressor_margin", "pump_flow")
        }
        if (
            set(state.inactive_feedback) - allowed
            or tuple(sorted(state.inactive_feedback)) != state.inactive_feedback
        ):
            raise ValueError("Inactive feedback registry differs from controllers")
        for _, value in state.positions:
            ConnectionActuator.position(value)
        for c, (_, value) in zip(self.controllers, state.integrals):
            if not math.isfinite(value) or abs(value) > c.maximum_integral_abs:
                raise PropertyModelError("Controller integral outside domain")
        if not self.minimum_step_s <= state.next_step_s <= self.maximum_step_s:
            raise ValueError("Controller restart step outside domain")

    def _inventory_path(self, controller):
        t = self.plant.transport
        if controller.sensor not in t.pipe_paths:
            raise ValueError("inventory_level needs an explicit vertical grouped sensor path")
        path = [t.nodes[t.names.index(name)] for name in t.pipe_paths[controller.sensor]]
        if any(
            abs(n.elevation_change_m - n.length_m) > 1e-12
            or abs(n.diameter_m - path[0].diameter_m) > 1e-12
            for n in path
        ):
            raise ValueError("Inventory-level sensor requires a constant-area vertical vessel")
        if any(
            abs(a.elevation_m + a.length_m / 2 - b.elevation_m + b.length_m / 2) > 1e-8
            for a, b in zip(path, path[1:])
        ):
            raise ValueError("Inventory-level sensor cells must be contiguous bottom-to-top")
        if controller.setpoint > sum(n.length_m for n in path):
            raise ValueError("Inventory-level setpoint above resolved vessel")
        return path

    def measure(self, state, controller):
        if controller.measurement == "pump_flow":
            return dict(state.machine_feedback).get(controller.sensor)
        if controller.measurement == "compressor_margin":
            return compressor_margin(self._compressors[controller.sensor], state.plant.transport)
        if controller.measurement == "speed":
            return dict(state.plant.transport.rotors)[
                controller.sensor or self._valves[controller.valve].port
            ]
        if controller.measurement == "temperature":
            port = self._ports[self._valves[controller.valve].port]
            return dict(state.plant.transport.nodes)[
                controller.sensor or port.sensor_node
            ].temperature_k
        port = self._ports[self._valves[controller.valve].port]
        column_name = (
            controller.sensor
            if controller.measurement == "level" and controller.sensor is not None
            else port.column
        )
        if column_name is None:
            t = self.plant.transport
            cells = dict(state.plant.transport.nodes)
            if controller.measurement == "pressure":
                return cells[controller.sensor or port.sensor_node].pressure_pa
            index = t.fluid.names.index(controller.component)
            volume = 0.0
            path = self._inventory_path(controller)
            for node in path:
                d = t._state_data(t.names.index(node.name), cells[node.name])
                selected = d["x"][:, index] >= controller.minimum_mole_fraction
                volume = volume + (d["phase_n"] * d["q"]["molar_volume_m3_mol"] * selected).sum()
            return float((volume / (math.pi * path[0].diameter_m ** 2 / 4)).cpu())
        column = self.plant.columns[column_name]
        s = dict(state.plant.columns)[column_name]
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
        modes = dict(modes)
        old_trips = {n: (latched, elapsed) for n, latched, elapsed in state.trips}
        trip_states, trip_records, isolation = [], {}, set()
        for trip in self.trips:
            latched, elapsed = old_trips[trip.name]
            value = trip.read(state.plant)
            final_latch, timer, active = trip.step(latched, elapsed, value, dt)
            trip_states.append((trip.name, final_latch, timer))
            if active:
                modes[trip.target] = "stuck" if modes.get(trip.target) == "stuck" else "fail_safe"
                isolation.update(trip.isolate_switches)
            trip_records[trip.name] = {
                "measurement": value,
                "latched_before": latched,
                "elapsed_before_s": elapsed,
                "latched": final_latch,
                "elapsed_s": timer,
                "active": active,
                "actuator_blocked": active and modes.get(trip.target) == "stuck",
            }
        measurements = {}
        inactive = []
        for c in self.controllers:
            if c.measurement == "pump_flow" and (
                c.sensor not in dict(state.machine_feedback)
                or c.sensor in isolation
                or not (commands.get("openings") or {}).get(
                    c.sensor, self._pump_connections[c.sensor].card.open
                )
            ):
                measurements[c.valve] = None
                inactive.append(c.valve)
                targets[c.valve] = manual.get(c.valve, positions[c.valve])
                continue
            if c.measurement == "compressor_margin" and (
                c.sensor in isolation
                or not (commands.get("openings") or {}).get(
                    c.sensor, self._compressors[c.sensor].open
                )
            ):
                measurements[c.valve] = None
                inactive.append(c.valve)
                targets[c.valve] = manual.get(c.valve, positions[c.valve])
                continue
            value = self.measure(state, c)
            measurements[c.valve] = value
            if c.valve in state.inactive_feedback:
                integrals[c.valve] = min(
                    c.maximum_integral_abs,
                    max(
                        -c.maximum_integral_abs,
                        positions[c.valve]
                        - c.opening_bias
                        - c.proportional_gain * c.action * (value - c.setpoint),
                    ),
                )
            targets[c.valve], integrals[c.valve], _ = c.update(
                value, integrals[c.valve], positions[c.valve], dt, manual.get(c.valve)
            )
        for v in self.channels:
            if v.name not in targets and (
                isinstance(v, PRMotorControl) or modes.get(v.name) in ("fail_safe", "stuck")
            ):
                targets[v.name] = positions[v.name]
        if set(targets) != {v.name for v in self.channels}:
            raise ValueError("Every valve requires feedback or an explicit manual target")
        openings = dict(commands.get("openings") or {})
        openings.update({name: False for name in isolation})
        losses = dict(commands.get("loss_coefficients") or {})
        known_links = {
            p.name
            for p in (
                *self.plant.transport.switches,
                *self.plant.ports,
                *self.plant.heat_links,
                *getattr(self.plant, "isolation_switches", ()),
            )
        }
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
        areas = dict(commands.get("restriction_openings") or {})
        controlled_areas = {v.port for v in self.channels if isinstance(v, PRThroatControlValve)}
        if controlled_areas & areas.keys():
            raise ValueError(
                "Controlled throat area requires an actuator target, not a second area command"
            )
        restrictions = {u.name: u for u in self.plant.transport.restrictions}
        records = {}
        coefficients = {}
        throat_areas = {}
        torques = dict(commands.get("drive_torque_nm") or {})
        if set(torques) & {m.machine for m in self.motors}:
            raise ValueError(
                "Controlled motor torque requires an actuator target, not a second torque command"
            )
        effective_torques = {}
        for v in self.channels:
            record = v.actuator.step(
                positions[v.name], targets[v.name], dt, modes.get(v.name, "normal")
            )
            x = record["final_position"]
            positions[v.name] = x
            records[v.name] = record
            if isinstance(v, PRMotorControl):
                torque = v.torque(x)
                torques[v.machine] = torque
                effective_torques[v.name] = torque
                coefficients[v.name] = None
                continue
            default = restrictions[v.port].open if isinstance(v, PRThroatControlValve) else True
            actual = openings.get(v.port, default) and x > 0
            openings[v.port] = actual
            if isinstance(v, PRThroatControlValve):
                area = restrictions[v.port].opening * v.fraction(x) if actual else 0.0
                areas[v.port] = area
                throat_areas[v.name] = area
                coefficients[v.name] = None
                continue
            k = v.coefficient(x) if actual else None
            coefficients[v.name] = k
            if k is not None:
                physical[self._ports[v.port].node] += k
        forced = {t.target for t in self.trips if trip_records[t.name]["active"]}
        for c in self.controllers:
            if c.valve in forced and measurements[c.valve] is not None:
                proportional = c.proportional_gain * c.action * (measurements[c.valve] - c.setpoint)
                integrals[c.valve] = min(
                    c.maximum_integral_abs,
                    max(
                        -c.maximum_integral_abs, positions[c.valve] - c.opening_bias - proportional
                    ),
                )
        # Safety isolation takes precedence over a simultaneous manual branch
        # opening. The physical graph still rejects ordinary conflicting commands.
        if isolation and hasattr(self.plant, "endpoint_switches"):
            profile_commands = {
                n: list(v) for n, v in (commands.get("profile_openings") or {}).items()
            }
            for switch in isolation:
                if switch in self.plant.endpoint_switches:
                    branch, side = self.plant.endpoint_switches[switch]
                    if branch in profile_commands:
                        profile_commands[branch][side] = False
            if profile_commands:
                commands = {**commands, "profile_openings": profile_commands}
        plant_report = self.plant.advance(
            state.plant,
            dt,
            **{
                **commands,
                "openings": openings,
                "loss_coefficients": physical,
                **({"restriction_openings": areas} if areas else {}),
                **({"drive_torque_nm": torques} if torques else {}),
            },
        )
        plant_state = physical_state_from_dict(plant_report["state"])
        envelope = {
            v.name: (
                {
                    "flow_model": "homogeneous_equilibrium_throat",
                    "maximum_throat_velocity_m_s": 0.0,
                    "choked_substeps": 0,
                }
                if isinstance(v, PRThroatControlValve)
                else {
                    "maximum_speed_rad_s": abs(dict(state.plant.transport.rotors)[v.machine]),
                    "motor_torque_nm": effective_torques[v.name],
                }
                if isinstance(v, PRMotorControl)
                else {"maximum_pressure_drop_fraction": 0.0, "maximum_mach_number": 0.0}
            )
            for v in self.channels
        }
        # Check each accepted hydraulic substep, not only the report endpoint.
        for step in plant_report["steps"]:
            snapshot = physical_state_from_dict(step["state"])
            for v in self.channels:
                if isinstance(v, PRMotorControl):
                    envelope[v.name]["maximum_speed_rad_s"] = max(
                        envelope[v.name]["maximum_speed_rad_s"],
                        abs(step["transport"]["drives"][v.machine]["speed_rad_s"]),
                    )
                    continue
                if isinstance(v, PRThroatControlValve):
                    point = step["transport"]["restrictions"][v.port]
                    envelope[v.name]["maximum_throat_velocity_m_s"] = max(
                        envelope[v.name]["maximum_throat_velocity_m_s"],
                        point["throat_velocity_m_s"],
                    )
                    envelope[v.name]["choked_substeps"] += int(point["choked"])
                    continue
                k = coefficients[v.name]
                if k is None:
                    continue
                port = self._ports[v.port]
                i = self.plant.transport.names.index(port.node)
                cell = dict(snapshot.transport.nodes)[port.node]
                d = self.plant.transport._state_data(i, cell)
                drop = 0.5 * float((d["rho"] * d["u"].square()).cpu()) * k
                if port.column is None:
                    p = dict(snapshot.transport.nodes)[port.sensor_node].pressure_pa
                else:
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
            tuple(trip_states),
            tuple(sorted(inactive)),
            tuple(
                (
                    name,
                    plant_report["steps"][-1]["transport"]["pump_connections"][name]["flow_m3_s"],
                )
                for name in sorted(self._feedback_units)
            ),
        )
        self.inspect(result)
        return result, {
            "dt_s": dt,
            "state": result.to_dict(),
            "plant": plant_report,
            "actuators": records,
            "measurements": measurements,
            "effective_valve_K": {v.name: coefficients[v.name] for v in self.valves},
            **(
                {
                    "effective_motor_torque_nm": effective_torques,
                    "motor_envelope": {v.name: envelope[v.name] for v in self.motors},
                }
                if self.motors
                else {}
            ),
            **({"protections": trip_records} if self.trips else {}),
            **({"effective_throat_openings": throat_areas} if throat_areas else {}),
            "valve_envelope": {v.name: envelope[v.name] for v in self.valves},
            "total_heat_j": plant_report["total_heat_j"],
            "total_work_j": plant_report["total_work_j"],
        }

    def _error(self, a, b):
        e = self.plant._error(a.plant, b.plant)
        # Flow samples belong to the last accepted hydraulic substep. Coarse
        # and fine samples have different averaging windows, so comparing them
        # as simultaneous conserved coordinates would force artificial cuts.
        # Their physical residuals are checked by the pump solver; temporal
        # error is controlled in the plant, actuator and active PI states.
        for trip, (_, al, at), (_, bl, bt) in zip(self.trips, a.trips, b.trips):
            e = max(
                e,
                2.0 if al != bl else 0.0,
                abs(at - bt) / max(self.minimum_step_s, 1e-6 * max(trip.delay_s, 1e-6)),
            )
        for (_, x), (_, y) in zip(a.positions, b.positions):
            e = max(e, abs(x - y) / self.position_tolerance)
        unavailable = set(a.inactive_feedback) | set(b.inactive_feedback)
        for (name, x), (_, y) in zip(a.integrals, b.integrals):
            # Before a first valid sample the PI integral is unobservable. Its
            # bumpless initialization is checked through the physical actuator
            # error, rather than rejecting an O(1) change of internal bias.
            if name not in unavailable:
                e = max(e, abs(x - y) / self.integral_tolerance)
        return e

    def advance(
        self,
        state,
        duration_s,
        *,
        targets=None,
        actuator_modes=None,
        reset_trips=(),
        **plant_commands,
    ):
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
            "drive_torque_nm",
            "restriction_openings",
            *(
                {"profile_heat_w", "profile_utility_temperature_k", "profile_openings"}
                if hasattr(self.plant, "profiles") or hasattr(self.plant, "bridges")
                else set()
            ),
        }:
            raise ValueError("Unknown controlled hydraulic command")
        if (
            not isinstance(reset_trips, (list, tuple))
            or len(set(reset_trips)) != len(reset_trips)
            or set(reset_trips) - {t.name for t in self.trips}
        ):
            raise ValueError("Unknown or duplicate protection reset")
        trips = {n: (latched, elapsed) for n, latched, elapsed in state.trips}
        for trip in self.trips:
            if trip.name in reset_trips:
                if not trip.can_reset(trip.read(state.plant)):
                    raise ValueError(
                        "Protection reset requires the sensor inside its hysteresis boundary"
                    )
                trips[trip.name] = (False, 0.0)
        current = replace(state, trips=tuple((n, *trips[n]) for n in sorted(trips)))
        dt = min(state.next_step_s, duration_s)
        steps = []
        attempts = 0
        while current.time_s < end:
            attempts += 1
            if attempts > self.maximum_attempts:
                raise HydraulicConvergenceError("Controlled interval budget exhausted")
            dt = interval_step_s(current.time_s, end, dt)
            pending = {n: (latched, elapsed) for n, latched, elapsed in current.trips}
            for trip in self.trips:
                latched, elapsed = pending[trip.name]
                if not latched and trip.delay_s > elapsed and trip.unsafe(trip.read(current.plant)):
                    remaining = trip.delay_s - elapsed
                    event = current.time_s + remaining
                    # Never fire before the held unsafe measurement has lasted
                    # the requested delay. A rounded-down event leaves an
                    # unsolvable sub-ULP timer remainder on a long-running clock.
                    if event - current.time_s < remaining:
                        event = math.nextafter(event, math.inf)
                    until_event = event - current.time_s
                    rounding = 8 * max(math.ulp(current.time_s), math.ulp(event))
                    if 0 <= until_event - dt <= rounding:
                        dt = min(until_event, end - current.time_s)
                    else:
                        dt = min(dt, until_event)
            dt = interval_step_s(current.time_s, end, dt)
            if current.time_s + dt / 2 <= current.time_s:
                raise ValueError("Controlled substep below time resolution")
            first_dt, second_dt = split_step_s(current.time_s, dt)
            try:
                coarse, _ = self._step(current, dt, manual, modes, plant_commands)
                half, one = self._step(current, first_dt, manual, modes, plant_commands)
                fine, two = self._step(half, second_dt, manual, modes, plant_commands)
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
                    plant=align_physical_clock(current.plant, end),
                )
            steps.extend((one, two))
            dt = nextdt
        heat = math.fsum(s["total_heat_j"] for s in steps)
        work = math.fsum(s["total_work_j"] for s in steps)
        return {
            "state": current.to_dict(),
            "steps": steps,
            "attempts": attempts,
            "total_heat_j": heat,
            "total_work_j": work,
            "conservation_residuals": self.plant._ledger(state.plant, current.plant, heat + work),
        }
