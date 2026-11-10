"""Conservative IMEX active-phase pipes, finite gravitational apertures and equipment.

All face transfers are frozen on the explicit CFL substep. Finite columns are
recovered at their new N/(U+PE), while equipment uses its existing implicit solve.
No fictitious plenum and no equilibrium projection of the pipe inventories.
"""

import math
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .phase_geometry import validate_pickup
from .pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
from .pr_equilibrium_euler_plant import _KnownFaceExchange
from .pr_equilibrium_transport import PREquilibriumTransportNetwork, PREquilibriumTransportState
from .pr_interphase_relaxation import PRRelaxationPhase
from .pr_phase_equilibration import PRPhaseEquilibriumState
from .stratified_pr import StratifiedPRColumn, StratifiedPRState
from .stratified_pr_flow import StratifiedPRFlow
from .stratified_pr_thermal import StratifiedPRThermal
from .vessel_wall import VesselWall


@dataclass(frozen=True)
class PRActiveColumnPort:
    name: str
    column: str
    height_m: float

    def __post_init__(self):
        if any(not isinstance(v, str) or not v for v in (self.name, self.column)):
            raise ValueError("Named finite column port required")
        _number(self.height_m, "aperture height", nonnegative=True)


@dataclass(frozen=True)
class PRActiveReservoir:
    """Maintained physical inlet/outlet state; no finite network inventory.

    Phase volumes define boundary holdup only. Actual transfers are integrated
    from the same Riemann face flux used by the pipe, including energy work.
    """

    phases: tuple
    phase_roots: tuple
    elevation_m: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "phase_roots", tuple(self.phase_roots))
        if (
            not self.phases
            or len(self.phases) != len(self.phase_roots)
            or any(not isinstance(p, PRRelaxationPhase) for p in self.phases)
            or any(r not in ("liquid", "vapor") for r in self.phase_roots)
        ):
            raise ValueError("Explicit PR reservoir phases and roots required")
        _number(self.elevation_m, "reservoir elevation")

    def to_dict(self):
        return {
            "phases": [asdict(p) for p in self.phases],
            "phase_roots": list(self.phase_roots),
            "elevation_m": self.elevation_m,
        }

    @classmethod
    def from_dict(cls, raw):
        if set(raw) != {"phases", "phase_roots", "elevation_m"}:
            raise ValueError("Unknown reservoir fields")
        return cls(
            tuple(PRRelaxationPhase.from_dict(p) for p in raw["phases"]),
            tuple(raw["phase_roots"]),
            raw["elevation_m"],
        )


@dataclass(frozen=True)
class PRActivePipeContinuation:
    """A shared grid face: left pipe outlet to right pipe inlet, no inventory."""

    left_pipe: str
    right_pipe: str

    def __post_init__(self):
        if any(not isinstance(n, str) or not n for n in (self.left_pipe, self.right_pipe)):
            raise ValueError("Named continuation pipes required")


@dataclass(frozen=True)
class PRActiveSurfaceState:
    equipment: object
    columns: tuple
    pipes: tuple
    time_s: float
    next_step_s: float
    model_id: str
    wall_temperatures: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "pipes", tuple(self.pipes))
        object.__setattr__(self, "wall_temperatures", tuple(self.wall_temperatures))
        for _, temperature in self.wall_temperatures:
            _number(temperature, "column wall temperature", positive=True)
        _number(self.time_s, "surface clock", nonnegative=True)
        _number(self.next_step_s, "surface step", positive=True)

    def to_dict(self):
        result = {
            "equipment": None if self.equipment is None else self.equipment.to_dict(),
            "columns": {n: s.to_dict() for n, s in self.columns},
            "pipes": {n: s.to_dict() for n, s in self.pipes},
            "time_s": self.time_s,
            "next_step_s": self.next_step_s,
            "model_id": self.model_id,
        }

        if self.wall_temperatures:
            result["wall_temperatures_k"] = dict(self.wall_temperatures)
        return result

    @classmethod
    def from_dict(cls, data):
        if set(data) - {"wall_temperatures_k"} != {
            "equipment",
            "columns",
            "pipes",
            "time_s",
            "next_step_s",
            "model_id",
        }:
            raise ValueError("Unknown active surface checkpoint")
        return cls(
            None
            if data["equipment"] is None
            else PREquilibriumTransportState.from_dict(data["equipment"]),
            tuple((n, StratifiedPRState.from_dict(s)) for n, s in sorted(data["columns"].items())),
            tuple(
                (n, PRActiveMultifluidState.from_dict(s)) for n, s in sorted(data["pipes"].items())
            ),
            data["time_s"],
            data["next_step_s"],
            data["model_id"],
            tuple(sorted(data.get("wall_temperatures_k", {}).items())),
        )


class PRActiveSurfaceNetwork:
    def __init__(
        self,
        pipes,
        columns,
        ports,
        *,
        equipment=None,
        maximum_transverse_head_fraction=1e-3,
        aperture_relative_tolerance=1e-6,
        column_walls=None,
        reservoirs=None,
        continuations=None,
    ):
        if not isinstance(columns, dict) or any(
            not isinstance(n, str) or not n or not isinstance(c, StratifiedPRColumn)
            for n, c in columns.items()
        ):
            raise TypeError("Named gravitational columns required")
        ports = tuple(ports)
        if any(not isinstance(p, PRActiveColumnPort) for p in ports):
            raise TypeError("Finite active column ports required")
        self.columns = dict(sorted(columns.items()))
        self.ports = {p.name: p for p in sorted(ports, key=lambda p: p.name)}
        self.pipes = dict(sorted(pipes.items()))
        self.equipment = equipment
        self.reservoirs = dict(sorted((reservoirs or {}).items()))
        if any(
            not isinstance(n, str) or not n or not isinstance(r, PRActiveReservoir)
            for n, r in self.reservoirs.items()
        ):
            raise ValueError("Named physical reservoirs required")
        self.continuations = dict(sorted((continuations or {}).items()))
        if any(
            not isinstance(n, str) or not n or not isinstance(c, PRActivePipeContinuation)
            for n, c in self.continuations.items()
        ):
            raise ValueError("Named direct pipe continuations required")
        self.walls = dict(sorted((column_walls or {}).items()))
        if set(self.walls) - set(self.columns) or any(
            not isinstance(w, VesselWall) for w in self.walls.values()
        ):
            raise ValueError("Named physical column walls required")
        self.thermal = {
            n: StratifiedPRThermal(self.columns[n], w, minimum_step_s=1e-12)
            for n, w in self.walls.items()
        }
        self.head_limit = _number(
            maximum_transverse_head_fraction, "transverse head limit", positive=True
        )
        self.aperture_tolerance = _number(
            aperture_relative_tolerance, "aperture tolerance", positive=True
        )
        if (
            not self.pipes
            or not (self.columns or self.reservoirs or self.continuations or equipment is not None)
            or len(self.ports) != len(ports)
            or self.head_limit > 0.01
        ):
            raise ValueError(
                "Pipes with physical endpoints, unique apertures and small transverse head required"
            )
        first_pipe = next(iter(self.pipes.values()))["pipe"]
        if not isinstance(first_pipe, PRActiveMultifluidPipe):
            raise TypeError("Active PR pipes required")
        self.fluid = first_pipe.fluid
        self.gravity = first_pipe.gravity
        if equipment is not None and (
            not isinstance(equipment, PREquilibriumTransportNetwork)
            or any(n.kind != "vessel" or n.geometry is not None for n in equipment.nodes)
        ):
            raise ValueError("Equipment endpoints must be real mixed finite vessels")
        bodies = [*self.columns.values(), *([equipment] if equipment is not None else [])]
        if any(
            b.fluid.model_id != self.fluid.model_id
            or b.fluid.device != self.fluid.device
            or b.gravity_m_s2 != self.gravity
            for b in bodies
        ):
            raise ValueError("Shared EOS, device and gravity required")
        names = set(equipment.names if equipment is not None else ())
        if (
            names & set(self.ports)
            or set(self.columns) & names
            or set(self.reservoirs) & (names | set(self.ports) | set(self.columns))
            or set(self.continuations)
            & (names | set(self.ports) | set(self.columns) | set(self.reservoirs))
        ):
            raise ValueError("Physical endpoint names must be distinct")
        used = []
        joined = []
        for name, entry in self.pipes.items():
            if not isinstance(name, str) or not name or set(entry) != {"pipe", "boundary_nodes"}:
                raise ValueError("Named active pipe and two endpoint bindings required")
            p = entry["pipe"]
            ends = entry["boundary_nodes"]
            if (
                not isinstance(p, PRActiveMultifluidPipe)
                or p.boundary_mode != "walls"
                or p.elevations is None
                or p.gravity != self.gravity
                or p.fluid.model_id != self.fluid.model_id
                or p.fluid.device != self.fluid.device
            ):
                raise ValueError("Surveyed active pipes with shared gravity/EOS/device required")
            if len(ends) != 2 or not any(n is not None for n in ends):
                raise ValueError("At least one physical endpoint required")
            for side, node in enumerate(ends):
                if node is None:
                    continue
                z = p.elevations[0 if side == 0 else -1]
                if node in self.ports:
                    port = self.ports[node]
                    if port.column not in self.columns:
                        raise ValueError("Unknown aperture column")
                    c = self.columns[port.column]
                    validate_pickup(c.geometry, port.height_m, p.diameter)
                    if abs(z - c.elevation_m - port.height_m) > 1e-9:
                        raise ValueError("Pipe face and physical aperture elevations differ")
                    used.append(node)
                elif node in self.continuations:
                    link = self.continuations[node]
                    if name != (link.left_pipe if side == 1 else link.right_pipe):
                        raise ValueError("Continuation must bind left outlet to right inlet")
                    joined.append((node, name, side))
                elif node in self.reservoirs:
                    reservoir = self.reservoirs[node]
                    if abs(z - reservoir.elevation_m) > 1e-9:
                        raise ValueError("Pipe face and reservoir elevations differ")
                    mapping = self._assign(p, (), reservoir.phases, reservoir.phase_roots)
                    p._boundary_packet(mapping)
                elif (
                    node not in names
                    or abs(z - equipment.nodes[equipment.names.index(node)].elevation_m) > 1e-9
                ):
                    raise ValueError("Unknown endpoint or incompatible physical elevation")
                elif equipment.nodes[equipment.names.index(node)].droplet_population is not None:
                    raise ValueError(
                        "Connect droplet populations through an explicit capture volume"
                    )
        if len(used) != len(set(used)) or set(used) != set(self.ports):
            raise ValueError("Each finite aperture must bind exactly one pipe face")
        expected = []
        for name, link in self.continuations.items():
            if link.left_pipe not in self.pipes or link.right_pipe not in self.pipes:
                raise ValueError("Unknown continuation pipe")
            left = self.pipes[link.left_pipe]["pipe"]
            right = self.pipes[link.right_pipe]["pipe"]
            if (
                left.diameter != right.diameter
                or left.slots != right.slots
                or left.wave_speed != right.wave_speed
                or left.elevations[-1] != right.elevations[0]
            ):
                raise ValueError(
                    "Continuation requires matching diameter, slots, wave bound and elevation"
                )
            expected.extend(((name, link.left_pipe, 1), (name, link.right_pipe, 0)))
        if sorted(expected) != sorted(joined):
            raise ValueError("Every continuation requires exactly two physical pipe faces")
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        result = {
            "model": "pr_active_surface_v1",
            "columns": {n: c.to_dict() for n, c in self.columns.items()},
            "ports": [asdict(p) for p in self.ports.values()],
            "pipes": {
                n: {"pipe": e["pipe"].to_dict(), "boundary_nodes": list(e["boundary_nodes"])}
                for n, e in self.pipes.items()
            },
            "equipment": None if self.equipment is None else self.equipment.to_dict(),
            "maximum_transverse_head_fraction": self.head_limit,
            "aperture_relative_tolerance": self.aperture_tolerance,
        }

        if self.walls:
            result["model"] = "pr_active_surface_v2"
            result["column_walls"] = {n: asdict(w) for n, w in self.walls.items()}
        if self.reservoirs:
            result["model"] = "pr_active_surface_v3"
            result["reservoirs"] = {n: r.to_dict() for n, r in self.reservoirs.items()}
        if self.continuations:
            result["model"] = "pr_active_surface_v4"
            result["continuations"] = {n: asdict(c) for n, c in self.continuations.items()}
        return result

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        d = dict(data)
        if d.pop("model", None) not in (
            "pr_active_surface_v1",
            "pr_active_surface_v2",
            "pr_active_surface_v3",
            "pr_active_surface_v4",
        ):
            raise ValueError("Unknown active surface model")
        d["columns"] = {
            n: StratifiedPRColumn.from_dict(c, device=device) for n, c in d["columns"].items()
        }
        d["pipes"] = {
            n: {**e, "pipe": PRActiveMultifluidPipe.from_dict(e["pipe"], device=device)}
            for n, e in d["pipes"].items()
        }
        d["ports"] = [PRActiveColumnPort(**p) for p in d["ports"]]
        d["equipment"] = (
            None
            if d["equipment"] is None
            else PREquilibriumTransportNetwork.from_dict(d["equipment"], device=device)
        )
        d["column_walls"] = {n: VesselWall(**w) for n, w in d.get("column_walls", {}).items()}
        d["reservoirs"] = {
            n: PRActiveReservoir.from_dict(r) for n, r in d.get("reservoirs", {}).items()
        }
        d["continuations"] = {
            n: PRActivePipeContinuation(**c) for n, c in d.get("continuations", {}).items()
        }
        return cls(**d)

    @property
    def maximum_step_s(self):
        return min(
            *[e["pipe"].maximum_step_s for e in self.pipes.values()],
            *[self.equipment.maximum_step_s] if self.equipment is not None else [math.inf],
        )

    def initialize(
        self,
        column_states,
        pipe_states,
        equipment_state=None,
        *,
        time_s=0.0,
        wall_temperatures_k=None,
    ):
        walls = dict(wall_temperatures_k or {})
        if set(walls) - set(self.walls):
            raise ValueError("Initial wall without physical column wall")
        state = PRActiveSurfaceState(
            equipment_state,
            tuple(sorted(column_states.items())),
            tuple(sorted(pipe_states.items())),
            time_s,
            self.maximum_step_s,
            self.model_id,
            tuple((n, walls.get(n, column_states[n].temperature_k)) for n in self.walls),
        )
        self.inspect(state)
        return state

    def inspect(self, state):
        if (
            not isinstance(state, PRActiveSurfaceState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
            or tuple(n for n, _ in state.columns) != tuple(self.columns)
            or tuple(n for n, _ in state.pipes) != tuple(self.pipes)
        ):
            raise PropertyModelError("Active surface identity or registry differs")
        if self.equipment is None:
            if state.equipment is not None:
                raise ValueError("Unexpected equipment inventory")
        else:
            self.equipment.inspect(state.equipment)
            if state.equipment.time_s != state.time_s:
                raise ValueError("Equipment clock differs")
        if tuple(n for n, _ in state.wall_temperatures) != tuple(self.walls):
            raise PropertyModelError("Column wall registry differs")
        for n, t in state.wall_temperatures:
            if not self.walls[n].min_temperature_k <= t <= self.walls[n].max_temperature_k:
                raise PropertyModelError("Column wall temperature outside domain")
        for n, s in state.columns:
            self.columns[n].inspect(s)
        for n, s in state.pipes:
            self.pipes[n]["pipe"].inspect(s)
            if s.time_s != state.time_s:
                raise ValueError("Pipe clock differs")

    def _assign(self, pipe, cell, phases, roots):
        target = PRPhaseEquilibriumState(tuple(phases), tuple(roots), "boundary", 0.0)
        return dict(
            pipe._assign_source(
                cell, target, {"phase_pairs": [(None, j, 1.0) for j in range(len(phases))]}
            )
        )

    def _samples(self, pipe, cell, port, state, order, position=1.0):
        c = self.columns[port.column]
        bounds = self.fluid.tensor([0.0, *state.interface_heights_m, c.geometry.height_m])
        rows = []
        for index, root in enumerate(state.phase_roots):
            heights, weights = StratifiedPRFlow._aperture(
                SimpleNamespace(column=c),
                bounds,
                index,
                SimpleNamespace(height_m=port.height_m, diameter_m=pipe.diameter),
                order,
            )
            for height, weight in zip(heights.cpu().tolist(), weights.cpu().tolist()):
                if weight <= 0:
                    continue
                q = c.sample(state, height, side="below")
                amount = 1.0 / q["molar_volume_m3_mol"]
                phase = PRRelaxationPhase(
                    tuple(amount * x for x in q["mole_fractions"]),
                    1.0,
                    q["pressure_pa"],
                    state.temperature_k,
                    0.0,
                    amount * q["molar_enthalpy_j_mol"] - q["pressure_pa"],
                )
                rho = q["density_kg_m3"]
                if (
                    self.gravity
                    * rho
                    * pipe.diameter
                    / max(q["pressure_pa"], self.fluid.min_pressure_pa)
                    > self.head_limit
                ):
                    raise PropertyModelError(
                        "Aperture exceeds cross-section averaged transverse head envelope"
                    )
                rows.append(
                    {
                        "cell": self._assign(pipe, cell, [phase], [root]),
                        "weight": position * weight,
                        "elevation_m": c.elevation_m + height,
                    }
                )
        if position < 1:
            rows.append(
                {
                    "cell": {n: replace(p, momentum_kg_m_s=-p.momentum_kg_m_s) for n, p in cell},
                    "weight": 1 - position,
                    "elevation_m": c.elevation_m + port.height_m,
                    "kind": "wall",
                }
            )
        return rows

    def _pipe_step(
        self,
        pipe,
        old,
        ends,
        state,
        dt,
        positions=None,
        heat_rates_w=None,
        heat_temperatures_k=None,
    ):
        boundary = [None, None]
        columns = dict(state.columns)
        for side, node in enumerate(ends):
            if node is None or node in self.ports:
                continue
            if node in self.continuations:
                link = self.continuations[node]
                neighbour = dict(state.pipes)[link.right_pipe if side == 1 else link.left_pipe]
                boundary[side] = dict(neighbour.cells[0 if side == 1 else -1])
                continue
            if node in self.reservoirs:
                reservoir = self.reservoirs[node]
                boundary[side] = self._assign(
                    pipe, old.cells[0 if side == 0 else -1], reservoir.phases, reservoir.phase_roots
                )
                continue
            s = dict(state.equipment.nodes)[node]
            phases = []
            for amounts, root in zip(s.phase_component_moles, s.phase_roots):
                n = math.fsum(amounts)
                q = self.fluid.homogeneous(
                    s.pressure_pa, s.temperature_k, [v / n for v in amounts], root
                )
                phases.append(
                    PRRelaxationPhase(
                        tuple(amounts),
                        n * float(q["molar_volume_m3_mol"].cpu()),
                        s.pressure_pa,
                        s.temperature_k,
                        0.0,
                        n * float(q["molar_internal_energy_j_mol"].cpu()),
                    )
                )
            boundary[side] = self._assign(
                pipe, old.cells[0 if side == 0 else -1], phases, s.phase_roots
            )
        connected = [self.columns[self.ports[n].column] for n in ends if n in self.ports]
        maximum = min(c.maximum_quadrature_order for c in connected) if connected else 2
        order = min(c.quadrature_order for c in connected) if connected else 2
        previous = None
        while order <= maximum:
            samples = [
                self._samples(
                    pipe,
                    old.cells[0 if side == 0 else -1],
                    self.ports[n],
                    columns[self.ports[n].column],
                    order,
                    (positions or {}).get(n, 1.0),
                )
                if n in self.ports
                else None
                for side, n in enumerate(ends)
            ]
            new, r = pipe.step(
                old,
                dt,
                boundary_cells=boundary,
                boundary_samples=samples,
                heat_rates_w=heat_rates_w,
                heat_temperatures_k=heat_temperatures_k,
            )
            vector = self.fluid.tensor(
                [
                    v
                    for key in ("face_fluxes", "phase_path")
                    for side in (0, -1)
                    for row in r[key][side]
                    for v in row
                ]
                + [v for side in (0, -1) for v in r["alpha_path"][side]]
                + list(r["boundary_entropy_fluxes"])
            )
            # Lax-Friedrichs flux subtracts large phasic energy densities.
            # Quadrature cannot resolve below their floating-point forward error.
            packet = pipe.inspect(old)
            qscale = packet["conserved"].abs().amax(0)
            fscale = packet["flux"].abs().amax(0)
            eps = torch.finfo(qscale.dtype).eps
            flux_roundoff = 64 * eps * (fscale + pipe.wave_speed * qscale)
            path_roundoff = torch.zeros_like(flux_roundoff)
            pressure_scale = packet["pressure"].abs().amax(0)
            velocity_scale = packet["velocity"].abs().amax(0)
            path_roundoff[..., -2] = 64 * eps * pressure_scale
            path_roundoff[..., -1] = 64 * eps * pressure_scale * velocity_scale
            alpha_roundoff = [64 * eps * pipe.wave_speed] * len(pipe.names)
            entropy_roundoff = (
                64
                * eps
                * pipe.wave_speed
                * max(1.0, float(packet["entropy_density"].abs().max().cpu()))
            )
            floor = self.fluid.tensor(
                [
                    *flux_roundoff.flatten().cpu().tolist(),
                    *flux_roundoff.flatten().cpu().tolist(),
                    *path_roundoff.flatten().cpu().tolist(),
                    *path_roundoff.flatten().cpu().tolist(),
                    *alpha_roundoff,
                    *alpha_roundoff,
                    entropy_roundoff,
                    entropy_roundoff,
                ]
            )
            if previous is not None:
                error = float(
                    (
                        (vector - previous).abs()
                        / (
                            1e-7
                            + floor
                            + self.aperture_tolerance * torch.maximum(vector.abs(), previous.abs())
                        )
                    )
                    .max()
                    .cpu()
                )
                if error <= 1:
                    return new, {
                        **r,
                        "aperture_quadrature_order": order,
                        "aperture_quadrature_error": error,
                    }
            elif not connected:
                return new, r
            previous = vector
            order *= 2
        raise HydraulicConvergenceError("Active aperture quadrature budget exhausted")

    def totals(self, state):
        nc = len(self.fluid.names)
        values = [self.pipes[n]["pipe"]._totals(s) for n, s in state.pipes]
        for n, s in state.columns:
            values.append([*s.component_moles, s.total_energy_j, 0.0])
        if state.equipment is not None:
            t = self.equipment
            for node, (_, s) in zip(t.nodes, state.equipment.nodes):
                mass = math.fsum(
                    v * c.molar_mass_kg_mol
                    for v, c in zip(s.component_moles, self.fluid.components)
                )
                values.append(
                    [
                        *s.component_moles,
                        s.total_energy_j
                        + mass * self.gravity * t.reference_elevation_m
                        + (
                            node.wall.heat_capacity_j_k * s.wall_temperature_k
                            if node.wall is not None
                            else 0.0
                        ),
                        0.0,
                    ]
                )
            values.append([*([0.0] * nc), t.rotor_energy(state.equipment), 0.0])
        values.extend(
            [[*([0.0] * nc), self.walls[n].energy(t), 0.0] for n, t in state.wall_temperatures]
        )
        return [math.fsum(v[j] for v in values) for j in range(nc + 1)]

    def capillary_totals(self, state, sources):
        """Augmented stocks for registered NORMAL sources, not axial transport.

        The geometry is recovered from the actual pipe phase volumes; no
        independent film inventory, thermostat or projection is introduced.
        """
        from .pr_capillary_pressure import PRCapillaryPressure

        self.inspect(state)
        if not isinstance(sources, dict) or set(sources) - set(self.pipes):
            raise ValueError("Capillary sources require named graph pipes")
        for name, source in sources.items():
            pipe = self.pipes[name]["pipe"]
            if (
                not isinstance(source, PRCapillaryPressure)
                or source.pipe.model_id != pipe.model_id
                or source.pipe.fluid.device != pipe.fluid.device
            ):
                raise ValueError("Capillary source and active graph pipe differ")
        surface = math.fsum(
            source.surface_energy(dict(state.pipes)[name]) for name, source in sources.items()
        )
        stocks = self.totals(state)
        stocks[-1] += surface
        return {"stocks": stocks, "surface_energy_j": surface}

    def relax_capillary(self, state, dt_s, sources):
        """Atomic fixed-clock normal/thermal/chemical source transaction.

        Does not call axial transport. Passing its output to ordinary step()
        does NOT yet close the surface-energy flux at pipe faces/junctions.
        """
        dt = _number(dt_s, "graph capillary source interval", positive=True)
        before = self.capillary_totals(state, sources)
        pipes = dict(state.pipes)
        records = {}
        for name, source in sorted(sources.items()):
            pipes[name], records[name] = source.apply(pipes[name], dt)
        new = replace(state, pipes=tuple(sorted(pipes.items())))
        after = self.capillary_totals(new, sources)
        residual = [b - a for a, b in zip(before["stocks"], after["stocks"], strict=True)]
        for j, value in enumerate(residual):
            allowance = 256 * (math.ulp(before["stocks"][j]) + math.ulp(after["stocks"][j]))
            if abs(value) > allowance + (1e-22 if j < len(residual) - 1 else 1e-20):
                raise ConservationError("Augmented capillary graph source balance failed")
        return new, {
            "state": new.to_dict(),
            "dt_s": dt,
            "clock_advanced": False,
            "sources": records,
            "surface_energy_before_j": before["surface_energy_j"],
            "surface_energy_after_j": after["surface_energy_j"],
            "conservation_residuals": residual,
            "entropy_production_j_k": math.fsum(
                r["entropy_production_j_k"] for r in records.values()
            ),
            "qualification": "Normal source on live graph stocks; axial surface-energy transport not implemented",
        }

    def capillary_chain_totals(self, state, pipe_order, parameters, *, port_openings=None):
        from .pr_capillary_chain import assemble

        _, view, source = assemble(self, state, pipe_order, parameters, port_openings=port_openings)
        surface = source.surface_energy(view)
        values = self.totals(state)
        values[-1] += surface
        return {"stocks": values, "surface_energy_j": surface}

    def relax_capillary_chain(self, state, dt_s, pipe_order, parameters, *, port_openings=None):
        """Joint normal/chemical solve, including shared continuation edges.

        Source transaction only; the old axial step is not a capillary step.
        """
        from .pr_capillary_chain import relax

        return relax(self, state, dt_s, pipe_order, parameters, port_openings=port_openings)

    def capillary_chain_transport_work(
        self,
        before,
        after,
        pipe_order,
        parameters,
        integrated_liquid_flux_m3,
        *,
        volume_source_m3=None,
        port_openings=None,
    ):
        """Return constitutive axial work/ports; does not advance graph stocks."""
        from .pr_capillary_chain import transport_work

        return transport_work(
            self,
            before,
            after,
            pipe_order,
            parameters,
            integrated_liquid_flux_m3,
            volume_source_m3=volume_source_m3,
            port_openings=port_openings,
        )

    def advance_capillary_open_line(self, state, duration_s, pipe_order, parameters, **options):
        """Coupled line with maintained PR face states and explicit capillary work."""
        from .pr_capillary_chain import advance_line

        return advance_line(
            self, state, duration_s, pipe_order, parameters, maintained_faces=True, **options
        )

    def advance_capillary_line(self, state, duration_s, pipe_order, parameters, **options):
        """Coupled PR/capillary dynamics for a complete isolated line or loop."""
        from .pr_capillary_chain import advance_line

        return advance_line(self, state, duration_s, pipe_order, parameters, **options)

    def advance_capillary_loop(self, state, duration_s, pipe_order, parameters, **options):
        """Actual coupled transport/normal dynamics for a complete isolated loop."""
        from .pr_capillary_chain import advance_loop

        return advance_loop(self, state, duration_s, pipe_order, parameters, **options)

    def _column_entropy(self, column, state):
        f = self.fluid
        seed = (
            [f.tensor(x) for x in state.phase_seeds],
            list(state.phase_roots),
            f.tensor(state.interface_heights_m),
            math.exp(state.solution_variables[0]),
            state.temperature_k,
            f.tensor(state.chemical_constants),
        )
        problem = column._problem(
            f.tensor(state.component_moles), state.total_energy_j, seed, state.quadrature_order
        )
        return float(problem[3](f.tensor(state.solution_variables))[1]["entropy_j_k"].cpu())

    def ledger(self, old, new, external_energy=0.0, external_components=None):
        a, b = self.totals(old), self.totals(new)
        r = [y - x for x, y in zip(a, b)]
        r[-1] -= external_energy
        components = (
            [0.0] * len(self.fluid.names)
            if external_components is None
            else list(external_components)
        )
        if len(components) != len(self.fluid.names) or not all(
            math.isfinite(v) for v in components
        ):
            raise ValueError("Finite external component ledger required")
        for j, value in enumerate(components):
            r[j] -= value
        if any(
            abs(v)
            > max(1e-7 if j < len(r) - 1 else 1e-5, 128 * math.ulp(max(abs(a[j]), abs(b[j]), 1.0)))
            for j, v in enumerate(r)
        ):
            raise ConservationError("Active surface component/total-energy ledger rejected")
        return r

    def step(
        self,
        state,
        dt_s,
        *,
        openings=None,
        drive_torque_nm=None,
        restriction_openings=None,
        port_openings=None,
        port_positions=None,
        pipe_heat_w=None,
        pipe_heat_temperatures_k=None,
    ):
        self.inspect(state)
        dt = _number(dt_s, "surface step", positive=True)
        end = state.time_s + dt
        dt = end - state.time_s
        if not math.isfinite(end) or dt <= 0 or dt > self.maximum_step_s * (1 + 1e-12):
            raise ValueError("Surface clock/CFL violated")
        pipe_heat_w = pipe_heat_w or {}
        pipe_heat_temperatures_k = pipe_heat_temperatures_k or {}
        if (set(pipe_heat_w) | set(pipe_heat_temperatures_k)) - set(self.pipes):
            raise ValueError("Unknown heated pipe")
        port_commands = {n: True for n in (*self.ports, *self.reservoirs, *self.continuations)}
        changes = dict(port_openings or {})
        if set(changes) - set(port_commands) or any(type(v) is not bool for v in changes.values()):
            raise ValueError("Known boolean aperture isolation commands required")
        port_commands.update(changes)
        positions = {n: 1.0 for n in self.ports}
        specified = dict(port_positions or {})
        if set(specified) - set(positions):
            raise ValueError("Unknown controlled aperture")
        for n, value in specified.items():
            value = _number(value, "aperture open area fraction", nonnegative=True)
            if value > 1:
                raise ValueError("Aperture open area fraction must lie in [0,1]")
            positions[n] = value
        for n, opened in port_commands.items():
            if not opened:
                positions[n] = 0.0

        nc = len(self.fluid.names)
        cn = {n: [0.0] * (nc + 1) for n in self.columns}
        cs = {n: 0.0 for n in self.columns}
        graph_net = (
            [self.fluid.tensor([0.0] * (nc + 2)) for _ in self.equipment.nodes]
            if self.equipment is not None
            else []
        )
        graph_entropy = 0.0
        reservoir_exchange = {n: [0.0] * (nc + 1) for n in self.reservoirs}
        reports = {}
        new_pipes = []
        for name, old in state.pipes:
            entry = self.pipes[name]
            pipe = entry["pipe"]
            ends = tuple(
                None
                if (not port_commands.get(n, True) or (n in positions and positions[n] == 0))
                else n
                for n in entry["boundary_nodes"]
            )
            new, r = self._pipe_step(
                pipe,
                old,
                ends,
                state,
                dt,
                positions,
                pipe_heat_w.get(name),
                pipe_heat_temperatures_k.get(name),
            )
            new_pipes.append((name, new))
            reports[name] = r
            for side, node in enumerate(ends):
                if node is None:
                    continue
                sign = -1.0 if side == 0 else 1.0
                index = 0 if side == 0 else -1
                fan = self.fluid.tensor(r["face_fluxes"][index]).sum(0).cpu().tolist()
                massflux = math.fsum(
                    v * c.molar_mass_kg_mol for v, c in zip(fan[:-2], self.fluid.components)
                )
                energy = fan[-1] + self.gravity * pipe.elevations[index] * massflux
                rate = [sign * pipe.area * v for v in [*fan[:-2], energy]]
                entropy = sign * pipe.area * r["boundary_entropy_fluxes"][side]
                if node in self.ports:
                    column = self.ports[node].column
                    cn[column] = [a + b for a, b in zip(cn[column], rate)]
                    cs[column] += entropy
                elif node in self.continuations:
                    # Same physical face is audited after both pipes advance;
                    # it is not a source/sink in the external network ledger.
                    continue
                elif node in self.reservoirs:
                    reservoir_exchange[node] = [
                        a - dt * b for a, b in zip(reservoir_exchange[node], rate)
                    ]
                else:
                    i = self.equipment.names.index(node)
                    graph_net[i] += self.fluid.tensor(
                        [
                            *rate[:-1],
                            0.0,
                            rate[-1]
                            - sign
                            * pipe.area
                            * massflux
                            * self.gravity
                            * self.equipment.reference_elevation_m,
                        ]
                    )
                    graph_entropy += entropy
        continuation_fluxes = {}
        for name, link in self.continuations.items():
            if not port_commands[name]:
                continuation_fluxes[name] = {"closed": True}
                continue
            left_pipe = self.pipes[link.left_pipe]["pipe"]
            right_pipe = self.pipes[link.right_pipe]["pipe"]
            lp = left_pipe.inspect(dict(state.pipes)[link.left_pipe])
            rp = right_pipe.inspect(dict(state.pipes)[link.right_pipe])
            qscale = torch.maximum(lp["conserved"][-1].abs(), rp["conserved"][0].abs())
            eps = torch.finfo(qscale.dtype).eps
            for field in ("face_fluxes", "phase_path", "alpha_path"):
                a = self.fluid.tensor(reports[link.left_pipe][field][-1])
                b = self.fluid.tensor(reports[link.right_pipe][field][0])
                scale = (
                    left_pipe.wave_speed * qscale
                    if field != "alpha_path"
                    else self.fluid.tensor(left_pipe.wave_speed)
                )
                allowance = 256 * eps * (a.abs() + b.abs() + scale)
                if not bool(
                    (torch.isfinite(a) & torch.isfinite(b) & ((a - b).abs() <= allowance))
                    .all()
                    .cpu()
                ):
                    raise ConservationError("Continuation face flux/path mismatch")
            continuation_fluxes[name] = {
                "closed": False,
                "left_pipe": link.left_pipe,
                "right_pipe": link.right_pipe,
                "face_fluxes": reports[link.left_pipe]["face_fluxes"][-1],
            }
        new_columns = []
        new_walls = []
        column_external = 0.0
        column_reports = {}
        entropy = math.fsum(r["entropy_production_j_k"] for r in reports.values())
        for name, old in state.columns:
            c = self.columns[name]
            target = [
                a + dt * b for a, b in zip([*old.component_moles, old.total_energy_j], cn[name])
            ]
            if min(target[:-1]) < 0 or math.fsum(target[:-1]) <= 0:
                raise PropertyModelError("Aperture exhausts a finite column component")
            if not any(cn[name]):
                new = old
                cr = {"state": old.to_dict()}
            else:
                cr = c.equilibrate(
                    target[:-1],
                    target[-1],
                    pressure_guess_pa=math.exp(old.solution_variables[0]),
                    temperature_guess_k=old.temperature_k,
                )
                new = StratifiedPRState.from_dict(cr["state"])
            si = self._column_entropy(c, new) - self._column_entropy(c, old) - dt * cs[name]
            if si < -1e-6 - 1e-10 * abs(self._column_entropy(c, old)):
                raise ConservationError("Finite aperture column entropy rejected")
            flow_state = new
            thermal_report = None
            if name in self.walls:
                thermal = self.thermal[name]
                thermal_old = replace(
                    thermal.initialize(new, dict(state.wall_temperatures)[name]),
                    time_s=state.time_s,
                )
                thermal_new, thermal_report = thermal._step(thermal_old, dt)
                new = thermal_new.column
                new_walls.append((name, thermal_new.wall_temperature_k))
                column_external += thermal_report["ambient_heat_j"]
                si += thermal_report["entropy_production_j_k"]
            entropy += si
            column_reports[name] = {
                **cr,
                "state": new.to_dict(),
                "flow_state": flow_state.to_dict(),
                "thermal": thermal_report,
                "entropy_production_j_k": si,
                "incoming_rates": cn[name],
                "incoming_entropy_w_k": cs[name],
            }
            new_columns.append((name, new))
        graph = None
        gr = None
        external = column_external + math.fsum(
            r.get("external_heat_j", 0.0) for r in reports.values()
        )
        external += math.fsum(v[-1] for v in reservoir_exchange.values())
        external_components = [
            math.fsum(v[j] for v in reservoir_exchange.values()) for j in range(nc)
        ]
        if self.equipment is not None:
            t = self.equipment
            commands = dict(t.default_openings)
            changes = dict(openings or {})
            if set(changes) - set(commands) or any(type(v) is not bool for v in changes.values()):
                raise ValueError("Unknown equipment isolation command")
            commands.update(changes)
            coupling = _KnownFaceExchange(
                t, state.equipment, dt, commands, graph_net, graph_entropy
            )
            graph, gr = t._step(
                state.equipment,
                dt,
                commands,
                [0.0] * len(t.nodes),
                [s.temperature_k for _, s in state.equipment.nodes],
                _coupling=coupling,
                drive_torque_nm=drive_torque_nm,
                restriction_openings=restriction_openings,
            )
            entropy += gr["entropy_production_j_k"]
            external += gr["ambient_heat_j"] + gr["external_heat_j"] + gr["external_work_j"]
        elif openings or drive_torque_nm or restriction_openings:
            raise ValueError("Equipment command without equipment graph")
        new = PRActiveSurfaceState(
            graph,
            tuple(new_columns),
            tuple(new_pipes),
            end,
            state.next_step_s,
            self.model_id,
            tuple(new_walls),
        )
        self.inspect(new)
        return new, {
            "state": new.to_dict(),
            "dt_s": dt,
            "port_openings": port_commands,
            "port_positions": positions,
            "pipes": reports,
            "columns": column_reports,
            "equipment": gr,
            "external_energy_j": external,
            "external_component_moles": external_components,
            "reservoir_exchange": reservoir_exchange,
            "continuation_fluxes": continuation_fluxes,
            "entropy_production_j_k": entropy,
            "conservation_residuals": self.ledger(state, new, external, external_components),
        }

    def advance(
        self, state, duration_s, *, maximum_attempts=10000, minimum_step_s=1e-12, **commands
    ):
        self.inspect(state)
        duration = _number(duration_s, "surface duration", positive=True)
        minimum = _number(minimum_step_s, "surface minimum step", positive=True)
        if type(maximum_attempts) is not int or maximum_attempts < 1:
            raise ValueError("Invalid integration budget")
        end = state.time_s + duration
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Unrepresentable surface interval")
        first = state
        dt = min(state.next_step_s, self.maximum_step_s)
        steps = []
        for attempt in range(maximum_attempts):
            target = min(end, state.time_s + min(dt, end - state.time_s))
            effective = target - state.time_s
            if effective > min(dt, end - state.time_s):
                target = math.nextafter(target, state.time_s)
            effective = target - state.time_s
            if effective <= 0:
                raise ValueError("Surface CFL below clock resolution")
            try:
                new, r = self.step(state, effective, **commands)
            except (PropertyModelError, HydraulicConvergenceError, ConservationError):
                dt = effective / 2
                if dt < minimum:
                    raise
                continue
            dt = min(self.maximum_step_s, effective * 1.25)
            state = replace(new, next_step_s=dt)
            steps.append(r)
            if state.time_s >= end:
                energy = math.fsum(r["external_energy_j"] for r in steps)
                return {
                    "state": state.to_dict(),
                    "steps": steps,
                    "attempts": attempt + 1,
                    "conservation_residuals": self.ledger(
                        first,
                        state,
                        energy,
                        [
                            math.fsum(r["external_component_moles"][j] for r in steps)
                            for j in range(len(self.fluid.names))
                        ],
                    ),
                }
        raise HydraulicConvergenceError("Active surface integration budget exhausted")
