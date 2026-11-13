"""Monolithic gravitational PR columns and arbitrary equilibrium transport graphs.

Apertures integrate local gravitational chemical equilibrium on each phase.
Pipe endpoints retain their one-dimensional cross-section approximation, with
an explicit transverse-head guard. No claim of resolved 2-D pipe inlet flow.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _name, _number
from .operation import _digest
from .peng_robinson import R
from .pr_equilibrium_transport import (
    PREquilibriumLink,
    PREquilibriumTransportNetwork,
    PREquilibriumTransportState,
)
from .pr_heat_link import PRHeatLink, thermal_exchange, thermal_records
from .pr_material_predictor import carrier_inventory_guess
from .pr_momentum_vessels import hllc_port
from .pr_phase_batch import PhaseEOSGroups
from .stratified_pr import StratifiedPRColumn, StratifiedPRState
from .stratified_pr_flow import StratifiedPRFlow
from .time_interval import interval_step_s, split_step_s
from .vessel_wall import VesselWall


@dataclass(frozen=True)
class PRColumnPort:
    name: str
    column: str
    node: str
    height_m: float
    diameter_m: float
    pipe_port: str = "left"

    def __post_init__(self):
        for key in ("name", "column", "node"):
            _name(getattr(self, key))
        _number(self.height_m, "port height", nonnegative=True)
        _number(self.diameter_m, "port diameter", positive=True)
        if self.pipe_port not in ("left", "right"):
            raise ValueError("Unknown pipe port")


@dataclass(frozen=True)
class PRGravityTransportState:
    transport: PREquilibriumTransportState
    columns: tuple
    walls: tuple
    model_id: str

    @property
    def time_s(self):
        return self.transport.time_s

    def to_dict(self):
        return {
            "transport": self.transport.to_dict(),
            "columns": {n: s.to_dict() for n, s in self.columns},
            "walls": dict(self.walls),
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"transport", "columns", "walls", "model_id"}:
            raise ValueError("Unknown coupled checkpoint fields")
        return cls(
            PREquilibriumTransportState.from_dict(data["transport"]),
            tuple((n, StratifiedPRState.from_dict(s)) for n, s in sorted(data["columns"].items())),
            tuple(sorted(data["walls"].items())),
            data["model_id"],
        )


class PRGravityTransportNetwork:
    def __init__(
        self,
        transport,
        columns,
        ports,
        *,
        walls=None,
        maximum_transverse_head_fraction=1e-3,
        heat_links=(),
    ):
        if not isinstance(transport, PREquilibriumTransportNetwork):
            raise TypeError("Expected equilibrium transport graph")
        if not isinstance(columns, dict):
            raise TypeError("Named gravitational columns dictionary required")
        self.transport = transport
        self.columns = dict(sorted(columns.items()))
        self.ports = tuple(sorted(ports, key=lambda p: p.name))
        self.heat_links = tuple(sorted(heat_links, key=lambda p: p.name))
        if any(not isinstance(link, PRHeatLink) for link in self.heat_links):
            raise TypeError("Expected PRHeatLink")
        self.walls = dict(sorted((walls or {}).items()))
        self.maximum_transverse_head_fraction = _number(
            maximum_transverse_head_fraction, "transverse head fraction", positive=True
        )
        if self.maximum_transverse_head_fraction > 0.01:
            raise ValueError("One-dimensional inlet approximation requires transverse head <= 1%")
        if set(self.walls) - set(self.columns) or any(
            not isinstance(w, VesselWall) for w in self.walls.values()
        ):
            raise ValueError("Unknown column wall")
        occupied = set(transport._used)
        names = {c.name for c in transport.switches}
        for name, col in self.columns.items():
            _name(name)
            if name in transport.names or not isinstance(col, StratifiedPRColumn):
                raise ValueError("Distinct named StratifiedPRColumns required")
            if (
                col.fluid.model_id != transport.fluid.model_id
                or col.fluid.device != transport.fluid.device
                or col.gravity_m_s2 != transport.gravity_m_s2
            ):
                raise ValueError("Column/graph EOS, device or gravity differs")
        for port in self.ports:
            if not isinstance(port, PRColumnPort) or port.name in names:
                raise ValueError("Unique column ports required")
            names.add(port.name)
            if port.column not in self.columns or port.node not in transport.names:
                raise ValueError("Unknown column port endpoint")
            i = transport.names.index(port.node)
            node = transport.nodes[i]
            if node.droplet_population is not None:
                raise ValueError(
                    "Route droplets through an explicit capture volume before a gravitational column"
                )
            col = self.columns[port.column]
            if (
                node.kind != "pipe_cell"
                or abs(node.face_diameter(port.pipe_port) - port.diameter_m) > 1e-12
                or (i, port.pipe_port) in occupied
            ):
                raise ValueError("Column port must occupy a free matching pipe end")
            occupied.add((i, port.pipe_port))
            endpoint = node.elevation_m + node.elevation_change_m * (
                0.5 if port.pipe_port == "right" else -0.5
            )
            if abs(endpoint - col.elevation_m - port.height_m) > 1e-8:
                raise ValueError("Physical port elevations differ")
            if (
                port.height_m < port.diameter_m / 2
                or port.height_m + port.diameter_m / 2 > col.geometry.height_m
            ):
                raise ValueError("Aperture lies outside column")
        body_names = set(self.columns) | set(transport.names)
        for link in self.heat_links:
            if link.name in names or {link.source, link.target} - body_names:
                raise ValueError("Unique thermal connection and known endpoints required")
            names.add(link.name)
        self.start_order = max((c.quadrature_order for c in self.columns.values()), default=2)
        self.maximum_order = min(
            (c.maximum_quadrature_order for c in self.columns.values()), default=4
        )
        if self.start_order >= self.maximum_order:
            raise ValueError("Common quadrature refinement range required")
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_gravity_transport_graph_v1",
            "transport": self.transport.to_dict(),
            "columns": {n: c.to_dict() for n, c in self.columns.items()},
            "ports": [asdict(p) for p in self.ports],
            "heat_links": [asdict(p) for p in self.heat_links],
            "walls": {n: asdict(w) for n, w in self.walls.items()},
            "maximum_transverse_head_fraction": self.maximum_transverse_head_fraction,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        data = dict(data)
        if data.pop("model", None) != "pr_gravity_transport_graph_v1":
            raise ValueError("Unknown gravitational transport model")
        data["transport"] = PREquilibriumTransportNetwork.from_dict(
            data["transport"], device=device
        )
        data["columns"] = {
            n: StratifiedPRColumn.from_dict(c, device=device) for n, c in data["columns"].items()
        }
        data["ports"] = [PRColumnPort(**p) for p in data["ports"]]
        data["heat_links"] = [PRHeatLink(**p) for p in data.get("heat_links", [])]
        data["walls"] = {n: VesselWall(**w) for n, w in data["walls"].items()}
        return cls(**data)

    def initialize(self, transport_state, column_states, wall_temperatures=None):
        if set(column_states) != set(self.columns):
            raise ValueError("Initial columns differ")
        tw = dict(wall_temperatures or {})
        if set(tw) - set(self.walls):
            raise ValueError("Initial temperature without wall")
        state = PRGravityTransportState(
            transport_state,
            tuple(sorted(column_states.items())),
            tuple((n, tw.get(n, column_states[n].temperature_k)) for n in self.walls),
            self.model_id,
        )
        self.inspect(state)
        return state

    def inspect(self, state):
        if (
            not isinstance(state, PRGravityTransportState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
        ):
            raise PropertyModelError("Coupled graph/checkpoint identity differs")
        self.transport.inspect(state.transport)
        if tuple(n for n, _ in state.columns) != tuple(self.columns) or tuple(
            n for n, _ in state.walls
        ) != tuple(self.walls):
            raise ValueError("Column or wall registry differs")
        for n, s in state.columns:
            self.columns[n].inspect(s)
        for n, t in state.walls:
            w = self.walls[n]
            if not math.isfinite(t) or not w.min_temperature_k <= t <= w.max_temperature_k:
                raise PropertyModelError("Column wall outside domain")

    def _ledger(self, before, after, heat, *, incoming_n=None):
        f = self.transport.fluid
        nc = len(f.names)

        def stocks(s):
            n = [
                math.fsum(v.component_moles[i] for _, v in s.transport.nodes)
                + math.fsum(v.component_moles[i] for _, v in s.columns)
                for i in range(nc)
            ]
            e = math.fsum(
                v.total_energy_j for _, v in s.transport.nodes
            ) + self.transport.rotor_energy(s.transport)
            e += math.fsum(
                v.total_energy_j
                - self.transport.gravity_m_s2
                * self.transport.reference_elevation_m
                * sum(a * b for a, b in zip(v.component_moles, f.molar_masses.cpu().tolist()))
                for _, v in s.columns
            )
            e += math.fsum(
                node.wall.heat_capacity_j_k * v.wall_temperature_k
                for node, (_, v) in zip(self.transport.nodes, s.transport.nodes)
                if node.wall
            )
            e += math.fsum(self.walls[name].heat_capacity_j_k * t for name, t in s.walls)
            return n, e

        a, ea = stocks(before)
        b, eb = stocks(after)
        r = [
            y - x - (incoming_n[i] if incoming_n is not None else 0.0)
            for i, (x, y) in enumerate(zip(a, b))
        ] + [eb - ea - heat]
        if any(abs(x) > 1e-8 + 1e-9 * max(1, abs(a[i])) for i, x in enumerate(r[:-1])) or abs(
            r[-1]
        ) > 1e-5 + 1e-9 * max(1, abs(heat)):
            raise ConservationError("Coupled gravitational graph N/E balance rejected")
        return r

    def _step(
        self,
        state,
        dt,
        openings,
        heat_w,
        utility,
        losses,
        *,
        drive_torque_nm=None,
        restriction_openings=None,
        _material=None,
        _column_work_j=None,
        _batch_aperture_rays=True,
    ):
        column_work = dict(_column_work_j or {})
        if set(column_work) - set(self.columns) or any(
            not math.isfinite(v) for v in column_work.values()
        ):
            raise ValueError("Known columns and finite mechanical work required")
        if not self.columns and not self.heat_links:
            # A resolved vessel already belongs to the transport graph. Do
            # not manufacture a column or repeat an empty aperture quadrature.
            t = self.transport
            result, report = t._step(
                state.transport,
                dt,
                {c.name: openings[c.name] for c in t.switches},
                [heat_w.get(n, 0.0) for n in t.names],
                [
                    utility.get(n, s.temperature_k)
                    for n, (_, s) in zip(t.names, state.transport.nodes)
                ],
                losses,
                _coupling=_material,
                drive_torque_nm=drive_torque_nm,
                restriction_openings=restriction_openings,
            )
            material = {} if _material is None else _material.ledger(report["coupling"], dt)
            new = PRGravityTransportState(result, (), (), self.model_id)
            self.inspect(new)
            heat = report["ambient_heat_j"] + report["external_heat_j"]
            work = report["external_work_j"]
            return new, {
                "dt_s": dt,
                "state": new.to_dict(),
                "transport": report,
                "columns": {
                    "columns": {},
                    "walls": {},
                    "reports": {},
                    "heat_j": 0.0,
                    "entropy_production_j_k": 0.0,
                    "heat_links": {},
                    "ports": {},
                },
                "quadrature_order": 0,
                "quadrature_error": 0.0,
                "total_heat_j": heat,
                "total_work_j": work,
                "conservation_residuals": self._ledger(
                    state,
                    new,
                    heat + work + material.get("material_energy_j", 0.0),
                    incoming_n=material.get("material_component_moles"),
                ),
                **material,
                "entropy_production_j_k": report["entropy_production_j_k"],
            }
        previous = None
        order = self.start_order
        while order <= self.maximum_order:
            overrides = {}
            base_open = {c.name: openings[c.name] for c in self.transport.switches}
            initial = state.transport
            for _ in range(8):
                coupling = _ColumnCoupling(
                    self,
                    state,
                    dt,
                    order,
                    openings,
                    heat_w,
                    utility,
                    seed_overrides=overrides,
                    warm_state=None if previous is None else previous[0],
                    material=_material,
                    column_work_j=column_work,
                    _batch_aperture_rays=_batch_aperture_rays,
                )
                try:
                    result, report = self.transport._step(
                        initial,
                        dt,
                        base_open,
                        [heat_w.get(n, 0.0) for n in self.transport.names],
                        [
                            utility.get(n, s.temperature_k)
                            for n, (_, s) in zip(self.transport.names, initial.nodes)
                        ],
                        losses,
                        _coupling=coupling,
                        drive_torque_nm=drive_torque_nm,
                        restriction_openings=restriction_openings,
                    )
                    break
                except _TopologyRetry:
                    overrides.update(coupling.retry_seeds)
            else:
                raise HydraulicConvergenceError("Gravitational phase topology budget exhausted")
            cr = report["coupling"]
            new = PRGravityTransportState(
                result,
                tuple(
                    (n, StratifiedPRState.from_dict(v)) for n, v in sorted(cr["columns"].items())
                ),
                tuple(sorted(cr["walls"].items())),
                self.model_id,
            )
            self.inspect(new)
            heat = report["ambient_heat_j"] + report["external_heat_j"] + cr["heat_j"]
            work = report["external_work_j"] + math.fsum(column_work.values())
            material = {} if _material is None else _material.ledger(cr["material"], dt)
            residual = self._ledger(
                state,
                new,
                heat + work + material.get("material_energy_j", 0.0),
                incoming_n=material.get("material_component_moles"),
            )
            entropy = report["entropy_production_j_k"] + cr["entropy_production_j_k"]
            if entropy < -1e-6:
                raise ConservationError("Coupled graph entropy rejected")
            record = {
                "dt_s": dt,
                "state": new.to_dict(),
                "transport": report,
                "columns": cr,
                "quadrature_order": order,
                "total_heat_j": heat,
                "total_work_j": work,
                **({"column_mechanical_work_j": column_work} if column_work else {}),
                "conservation_residuals": residual,
                **material,
                "entropy_production_j_k": entropy,
            }
            if previous is not None:
                err = self._error(previous[0], new)
                oldports = previous[1]["columns"]["ports"]
                for name, p in cr["ports"].items():
                    for x, y in zip(p["flux"], oldports[name]["flux"]):
                        err = max(
                            err,
                            abs(x - y)
                            / (1e-9 + self.transport.relative_tolerance * max(abs(x), abs(y))),
                        )
                if err <= 1:
                    record["quadrature_error"] = err
                    return new, record
            previous = (new, record)
            order *= 2
        raise HydraulicConvergenceError("Coupled aperture quadrature budget exhausted")

    def _error(self, a, b):
        error = self.transport._error(a.transport, b.transport)
        for (n, x), (other, y) in zip(a.columns, b.columns, strict=True):
            if n != other:
                raise ValueError("Temporal comparison requires the same column registry")
            # Empty zip(interface heights) must not hide a disappeared layer.
            # Cubic-root names alone do not identify a physical phase.
            if len(x.phase_roots) != len(y.phase_roots):
                error = max(error, 2.0)
            error = max(
                error,
                abs(x.temperature_k - y.temperature_k) / self.transport.temperature_tolerance_k,
                max(
                    abs(u - v) / (1e-10 + self.transport.relative_tolerance * max(abs(u), abs(v)))
                    for u, v in zip(x.component_moles, y.component_moles)
                ),
                max(
                    [0.0]
                    + [
                        abs(u - v)
                        / (
                            1e-8
                            + self.columns[n].relative_tolerance * self.columns[n].geometry.height_m
                        )
                        for u, v in zip(x.interface_heights_m, y.interface_heights_m)
                    ]
                ),
            )
        for (_, x), (_, y) in zip(a.walls, b.walls):
            error = max(error, abs(x - y) / self.transport.temperature_tolerance_k)
        return error

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
        _material_factory=None,
        _controlled_step=False,
    ):
        self.inspect(state)
        end = state.time_s + _number(duration_s, "duration", positive=True)
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Unrepresentable coupled interval")
        commands = {p.name: True for p in (*self.transport.switches, *self.ports, *self.heat_links)}
        commands.update(self.transport.default_openings)
        changes = dict(openings or {})
        if set(changes) - set(commands) or any(type(v) is not bool for v in changes.values()):
            raise ValueError("Known binary isolation commands required")
        commands.update(changes)
        heat = dict(heat_w or {})
        utility = dict(utility_temperature_k or {})
        names = set(self.transport.names) | set(self.columns)
        if (set(heat) | set(utility)) - names:
            raise ValueError("Unknown thermal node")
        for n, q in heat.items():
            _number(q, "heat rate")
            if q and n not in utility:
                raise ValueError("Heat requires utility temperature")
        for v in utility.values():
            _number(v, "utility temperature", positive=True)
        losses = dict(loss_coefficients or {})
        pipes = {n.name for n in self.transport.nodes if n.kind == "pipe_cell"}
        if set(losses) - pipes:
            raise ValueError("Unknown resistance cell")
        for v in losses.values():
            _number(v, "loss coefficient", nonnegative=True)
        if _controlled_step:
            # The enclosing controller compares the complete coarse/fine
            # plant state, including N/E, velocities, interfaces and rotors.
            # A second adaptive coarse/fine loop here repeats that same
            # physical error estimate three times per controller attempt.
            # Keep aperture quadrature and all physical/domain audits; let
            # failures and temporal error reach the enclosing adaptive loop.
            new, record = self._step(
                state,
                duration_s,
                commands,
                heat,
                utility,
                losses,
                _material=None
                if _material_factory is None
                else _material_factory(state, duration_s, commands),
                drive_torque_nm=drive_torque_nm,
                restriction_openings=restriction_openings,
            )
            new = replace(new, transport=replace(new.transport, time_s=end))
            record = {**record, "state": new.to_dict()}
            material = {
                key: record[key]
                for key in ("material_component_moles", "material_energy_j")
                if key in record
            }
            return {
                "state": new.to_dict(),
                "steps": [record],
                "attempts": 1,
                "total_heat_j": record["total_heat_j"],
                "total_work_j": record["total_work_j"],
                "conservation_residuals": record["conservation_residuals"],
                "openings": commands,
                "loss_coefficients": losses,
                **material,
            }
        current = state
        dt = min(state.transport.next_step_s, duration_s)
        steps = []
        attempts = 0
        while current.time_s < end:
            attempts += 1
            if attempts > self.transport.maximum_attempts:
                raise HydraulicConvergenceError("Coupled interval attempt budget exhausted")
            dt = interval_step_s(current.time_s, end, dt)
            if current.time_s + dt / 2 <= current.time_s:
                raise ValueError("Coupled substep below time resolution")
            first_dt, second_dt = split_step_s(current.time_s, dt)
            try:
                coarse, _ = self._step(
                    current,
                    dt,
                    commands,
                    heat,
                    utility,
                    losses,
                    _material=None
                    if _material_factory is None
                    else _material_factory(current, dt, commands),
                    drive_torque_nm=drive_torque_nm,
                    restriction_openings=restriction_openings,
                )
                half, one = self._step(
                    current,
                    first_dt,
                    commands,
                    heat,
                    utility,
                    losses,
                    _material=None
                    if _material_factory is None
                    else _material_factory(current, first_dt, commands),
                    drive_torque_nm=drive_torque_nm,
                    restriction_openings=restriction_openings,
                )
                fine, two = self._step(
                    half,
                    second_dt,
                    commands,
                    heat,
                    utility,
                    losses,
                    _material=None
                    if _material_factory is None
                    else _material_factory(half, second_dt, commands),
                    drive_torque_nm=drive_torque_nm,
                    restriction_openings=restriction_openings,
                )
                error = self._error(coarse, fine)
                if error > 1:
                    raise HydraulicConvergenceError("Coupled temporal error exceeded")
            except (HydraulicConvergenceError, PropertyModelError, ConservationError) as exc:
                dt /= 2
                if dt < self.transport.minimum_step_s:
                    raise HydraulicConvergenceError("Coupled minimum step exhausted") from exc
                continue
            nextdt = min(
                self.transport.maximum_step_s,
                max(self.transport.minimum_step_s, dt * (1.5 if error < 0.25 else 1)),
            )
            current = replace(
                fine,
                transport=replace(
                    fine.transport,
                    time_s=end if dt >= end - current.time_s else fine.time_s,
                    next_step_s=nextdt,
                ),
            )
            steps.extend((one, two))
            dt = nextdt
        heat = math.fsum(s["total_heat_j"] for s in steps)
        work = math.fsum(s["total_work_j"] for s in steps)
        material = {}
        if _material_factory is not None:
            material = {
                "material_component_moles": [
                    math.fsum(r["material_component_moles"][j] for r in steps)
                    for j in range(len(self.transport.fluid.names))
                ],
                "material_energy_j": math.fsum(r["material_energy_j"] for r in steps),
            }
        return {
            "state": current.to_dict(),
            "steps": steps,
            "attempts": attempts,
            "total_heat_j": heat,
            "total_work_j": work,
            "conservation_residuals": self._ledger(
                state,
                current,
                heat + work + material.get("material_energy_j", 0.0),
                incoming_n=material.get("material_component_moles"),
            ),
            **material,
            "openings": commands,
            "loss_coefficients": losses,
        }


class _TopologyRetry(HydraulicConvergenceError):
    """A stable column topology discovered outside the Newton residual."""


class _ColumnCoupling:
    def __init__(
        self,
        model,
        state,
        dt,
        order,
        openings,
        heat,
        utility,
        *,
        seed_overrides=None,
        warm_state=None,
        material=None,
        column_work_j=None,
        _batch_aperture_rays=True,
    ):
        self.column_work_j = dict(column_work_j or {})
        self.batch_aperture_rays = _batch_aperture_rays
        self.material = material
        self.model = model
        self.state = state
        self.dt = dt
        self.order = order
        self.seed_overrides = dict(seed_overrides or {})
        self.warm_state = warm_state
        self.retry_seeds = {}
        self.openings = openings
        self.heat = heat
        self.utility = utility
        self.external_ports = tuple(
            (model.transport.names.index(p.node), p.pipe_port) for p in model.ports
        )
        self._component_guesses()
        self._thermal_guesses()

    def _thermal_guesses(self):
        """Bounded implicit-diagonal predictor for branch discovery only.

        A raw dt*UA*deltaT predictor can exceed the entire EOS domain for a
        stiff contact. Frozen sensible capacities give a convex temperature
        estimate. Neither old conserved stocks nor the accepted energy target
        are changed; the full PR latent heat remains in the joint residual.
        """
        m, state = self.model, self.state
        self.thermal_guesses = {}
        if not m.heat_links:
            return
        bodies = {**dict(state.transport.nodes), **dict(state.columns)}
        cv = [c.ideal_cp_j_mol_k - R for c in m.transport.fluid.components]
        capacities = {
            name: max(1e-12, sum(v * c for v, c in zip(body.component_moles, cv)))
            for name, body in bodies.items()
        }
        numerator = {name: capacities[name] * body.temperature_k for name, body in bodies.items()}
        denominator = dict(capacities)
        connected = set()
        for link in m.heat_links:
            if self.openings[link.name]:
                for a, b in [(link.source, link.target), (link.target, link.source)]:
                    numerator[a] += self.dt * link.conductance_w_k * bodies[b].temperature_k
                    denominator[a] += self.dt * link.conductance_w_k
                    connected.add(a)
        for name in set(m.columns) & connected:
            predicted_t = numerator[name] / denominator[name]
            geometry = m.columns[name].geometry
            boundaries = [0.0, *bodies[name].interface_heights_m, geometry.height_m]
            volumes = [geometry.liquid_volume_m3(height) for height in boundaries]
            thin = (
                len(boundaries) > 2
                and min(b - a for a, b in zip(volumes[:-1], volumes[1:]))
                < 1e-4 * geometry.volume_m3
            )
            # Ordinary small thermal steps retain the previous seed, avoiding
            # repeated extra flashes. Near layer extinction, probe even a small
            # temperature change to prevent endless subdivision at the boundary.
            if abs(predicted_t - bodies[name].temperature_k) >= 0.25 or thin:
                self.thermal_guesses[name] = predicted_t

    def _component_guesses(self):
        """Reachable supports and explicit predictor; physical old stocks stay exact."""
        m, state = self.model, self.state
        t, f = m.transport, m.transport.fluid
        ordinary = dict(state.transport.nodes)
        columns = dict(state.columns)
        names = list(t.names) + list(m.columns)
        supports = {
            n: {i for i, v in enumerate(s.component_moles) if v > 0}
            for n, s in {**ordinary, **columns}.items()
        }
        if self.material is not None:
            for i, name in enumerate(t.names):
                supports[name].update(self.material.active[i].cpu().tolist())
        edges = [(c.source, c.target) for c in t.links if self.openings[c.name]]
        edges += [(p.column, p.node) for p in m.ports if self.openings[p.name]]
        edges += [(u.source, u.target) for u in t.restrictions if self.openings[u.name]]
        directed = [
            (unit.source, unit.target)
            for unit in (*t.compressors, *t.pump_connections)
            if self.openings[unit.name] and getattr(unit, "signed_map", None) is None
        ]
        edges += [
            (unit.source, unit.target)
            for unit in (*t.compressors, *t.pump_connections)
            if self.openings[unit.name] and getattr(unit, "signed_map", None) is not None
        ]
        changed = True
        while changed:
            changed = False
            for a, b in edges:
                union = supports[a] | supports[b]
                if supports[a] != union or supports[b] != union:
                    supports[a] = set(union)
                    supports[b] = set(union)
                    changed = True
            for a, b in directed:
                if not supports[a] <= supports[b]:
                    supports[b] |= supports[a]
                    changed = True
        index = lambda n: torch.tensor(sorted(supports[n]), dtype=torch.long, device=f.device)
        self.active = [index(n) for n in t.names]
        self.column_active = {n: index(n) for n in m.columns}
        self.ordinary_guesses = [f.tensor(ordinary[n].component_moles) for n in t.names]
        self.column_guesses = {n: f.tensor(columns[n].component_moles) for n in m.columns}
        self.column_energy_guesses = {n: s.total_energy_j for n, s in columns.items()}
        missing = {
            n
            for n in names
            if len(supports[n]) != sum(v > 0 for v in ({**ordinary, **columns})[n].component_moles)
        }
        self.new_support = missing
        if not missing:
            return
        data = [t._state_data(i, s) for i, (_, s) in enumerate(state.transport.nodes)]
        net, records, _ = t._fluxes(
            data, {c.name: self.openings[c.name] for c in t.links}, self.external_ports
        )
        lookup = {n: i for i, n in enumerate(names)}
        carrier_edges = [
            (r["source_index"], r["target_index"], r["flux"][:-2].sum()) for r in records
        ]
        if self.material is not None:
            external = self.material.exchange(data)
            net = [a + b for a, b in zip(net, external["net"])]
        cn = {n: f.tensor([0.0] * len(f.names)) for n in m.columns}
        ce = {n: f.tensor(0.0) for n in m.columns}
        for port in m.ports:
            if not self.openings[port.name]:
                continue
            c = m.columns[port.column]
            s = columns[port.column]
            i = t.names.index(port.node)
            bounds = f.tensor([0.0, *s.interface_heights_m, c.geometry.height_m])
            link = PREquilibriumLink(
                port.name, port.column, port.node, port.diameter_m, target_port=port.pipe_port
            )
            pipe, _, normal, extra = t._endpoint(i, data[i], "target", link)
            total = f.tensor([0.0] * (len(f.names) + 2))
            for phase in range(len(s.phase_roots)):
                heights, weights = StratifiedPRFlow._aperture(
                    SimpleNamespace(column=c), bounds, phase, port, 2
                )
                for h, w in zip(heights, weights):
                    if float(w.cpu()) <= 0:
                        continue
                    point = c.sample(s, float(h.cpu()), side="below")
                    x = f.tensor(point["mole_fractions"])
                    pp = f.tensor(point["pressure_pa"])
                    vm = point["molar_volume_m3_mol"]
                    rho = point["density_kg_m3"]
                    phi = c.gravity_m_s2 * (c.elevation_m + h - t.reference_elevation_m)
                    uc = torch.cat(
                        (
                            x / vm,
                            pp.new_zeros(1),
                            (point["molar_enthalpy_j_mol"] / vm - pp + rho * phi).reshape(1),
                        )
                    )
                    fc = torch.cat((torch.zeros_like(x), pp.reshape(1), pp.new_zeros(1)))
                    column = (
                        uc,
                        fc,
                        pp,
                        pp * 0,
                        pp.new_tensor(rho),
                        pp.new_tensor(point["molar_entropy_j_mol_k"] / vm),
                    )
                    pu, pf, pr, pv, pd, ps = pipe
                    du = c.gravity_m_s2 * (h - port.height_m)
                    pu = torch.cat((pu[:-1], (pu[-1] + pd * du).reshape(1)))
                    pf = torch.cat((pf[:-1], (pf[-1] + pd * du * pv).reshape(1)))
                    flux, _, _, _, _ = hllc_port(
                        column, (pu, pf, pr, pv, pd, ps), t.wave_speed_bound_m_s
                    )
                    correction = torch.where(
                        extra["actual_velocity"][:, None] < 0,
                        extra["actual"],
                        torch.zeros_like(extra["actual"]),
                    ).sum(0) - torch.where(
                        extra["bulk_velocity"] < 0, extra["bulk"], torch.zeros_like(extra["bulk"])
                    ).sum(0)
                    flux = flux + correction
                    total = total + w * flux
            total = total * (math.pi * port.diameter_m**2 / 4)
            carrier_edges.append((lookup[port.column], i, total[:-2].sum()))
            net[i] = net[i] + torch.cat(
                (total[:-2], (-normal * total[-2]).reshape(1), total[-1].reshape(1))
            )
            cn[port.column] = cn[port.column] - total[:-2]
            ce[port.column] = (
                ce[port.column]
                - total[-1]
                - (total[:-2] * f.molar_masses).sum() * c.gravity_m_s2 * t.reference_elevation_m
            )
        for i, n in enumerate(t.names):
            guess = (f.tensor(ordinary[n].component_moles) + self.dt * net[i][:-2]).clamp_min(0)
            self.ordinary_guesses[i] = guess.index_copy(
                0, self.active[i], guess[self.active[i]].clamp_min(guess.sum() * 1e-8)
            )
        for n in m.columns:
            guess = (f.tensor(columns[n].component_moles) + self.dt * cn[n]).clamp_min(0)
            self.column_guesses[n] = guess.index_copy(
                0, self.column_active[n], guess[self.column_active[n]].clamp_min(guess.sum() * 1e-8)
            )
            self.column_energy_guesses[n] += self.dt * float(ce[n].cpu())
        # Newly supplied species can be many orders of magnitude below the
        # old carrier inventory. A conservative implicit carrier predictor
        # warms the WHOLE connected chain, rather than seeding every dry
        # component at the same arbitrary mole fraction. Multiphase/selective
        # equipment keeps its original branch-aware predictor.
        bodies = {**ordinary, **columns}
        if (
            self.material is not None
            and not (t.pumps or t.compressors or t.pump_connections or t.restrictions)
            and all(len(v.phase_roots) == 1 for v in bodies.values())
        ):
            stocks = f.tensor([bodies[n].component_moles for n in names])
            incoming = torch.zeros_like(stocks)
            outgoing = stocks.new_zeros(len(names))
            for i, _feed, amount, _energy, _entropy in self.material.prepared.feeds:
                incoming[i] = incoming[i] + amount
            for name, orientation, amount, _energy, _entropy, _volume in external[
                "boundary_records"
            ]:
                if orientation == "feed":
                    continue
                port = next(p for p in self.material.model.pressure_ports if p.name == name)
                i = lookup[port.node]
                incoming[i] = incoming[i] + (-amount).clamp_min(0)
                outgoing[i] = outgoing[i] + amount.clamp_min(0).sum()
            predicted = carrier_inventory_guess(stocks, carrier_edges, incoming, outgoing, self.dt)
            if predicted is not None:
                for i, name in enumerate(names):
                    active = self.active[i] if i < len(t.names) else self.column_active[name]
                    guess = predicted[i].index_copy(
                        0, active, predicted[i, active].clamp_min(predicted[i].sum() * 1e-30)
                    )
                    if i < len(t.names):
                        self.ordinary_guesses[i] = guess
                    else:
                        self.column_guesses[name] = guess

    def prepare(self, y, lo, hi, base, offsets, dt, pattern):
        m = self.model
        t = m.transport
        f = t.fluid
        order = self.order
        if self.warm_state is not None:
            pieces = []
            for i, (_, w) in enumerate(self.warm_state.transport.nodes):
                active = self.active[i]
                amounts = f.tensor(w.phase_component_moles)[:, active]
                guess = torch.cat(
                    (
                        amounts.clamp_min(1e-300).log().flatten(),
                        f.tensor([math.log(w.pressure_pa), w.temperature_k / f.tref]),
                        (
                            f.tensor(w.momentum_kg_m_s)
                            / (f.tensor(w.component_moles) * f.molar_masses).sum()
                            / t.maximum_velocity_m_s
                        ).reshape(1),
                    )
                )
                pieces.append(
                    guess
                    if len(guess) == offsets[i + 1] - offsets[i]
                    else y[offsets[i] : offsets[i + 1]]
                )
            y = torch.cat((*pieces, y[offsets[-1] :]))
        self.size = len(y)
        self.blocks = {}
        self.portblocks = []
        initial = [y]
        lower = [lo]
        upper = [hi]
        offset = len(y)
        self.seeds = {}
        self.problems = {}
        self.scales = {}
        for name, s in self.state.columns:
            c = m.columns[name]
            n = f.tensor(s.component_moles)
            seed = (
                [f.tensor(x) for x in s.phase_seeds],
                list(s.phase_roots),
                f.tensor(s.interface_heights_m),
                math.exp(s.solution_variables[0]),
                s.temperature_k,
                f.tensor(s.chemical_constants),
            )
            if self.warm_state is not None:
                warm = dict(self.warm_state.columns)[name]
                seed = (
                    [f.tensor(x) for x in warm.phase_seeds],
                    list(warm.phase_roots),
                    f.tensor(warm.interface_heights_m),
                    math.exp(warm.solution_variables[0]),
                    warm.temperature_k,
                    f.tensor(warm.chemical_constants),
                )
            if name in self.new_support:
                seed = c._seed(
                    self.column_guesses[name], self.column_energy_guesses[name], seed[3], seed[4]
                )
            if name in self.seed_overrides:
                seed = self.seed_overrides[name]
            elif self.warm_state is None and name in self.thermal_guesses:
                predicted_t = self.thermal_guesses[name]
                guess_n = self.column_guesses[name]
                pt = f.flash_pt(seed[3], predicted_t, guess_n / guess_n.sum())
                # Choose the phase topology at a bounded temperature rather
                # than extrapolating sensible Cv through a latent-heat plateau.
                # Normalize guessed layer volumes only; _problem below still
                # receives the exact old N/E and solves the actual common P/T.
                seed = c._seed_from_flash(pt, c.geometry.volume_m3 / pt["molar_volume_m3_mol"])
                # PT at the old pressure may select a dense liquid even when
                # the fixed inventory/volume requires coexistence on cooling.
                # Retain a UV-discovered NEW phase count in that case. Both
                # alternatives are guesses, never accepted replacement stocks.
                cv = sum(
                    value * (component.ideal_cp_j_mol_k - R)
                    for value, component in zip(s.component_moles, f.components)
                )
                guessed_energy = (
                    self.column_energy_guesses[name]
                    + cv * (predicted_t - s.temperature_k)
                    + dt * self.heat.get(name, 0.0)
                )
                try:
                    uvseed = c._seed(guess_n, guessed_energy, seed[3], predicted_t)
                    if len(uvseed[1]) != len(s.phase_roots):
                        seed = uvseed
                except (PropertyModelError, HydraulicConvergenceError):
                    pass
            elif self.warm_state is None and self.heat.get(name, 0.0):
                seed = c._seed(n, s.total_energy_j + dt * self.heat[name], seed[3], seed[4])
            q = c._problem(
                n,
                s.total_energy_j,
                seed,
                order,
                _network_active=self.column_active[name],
                _network_guess=self.column_guesses[name],
            )
            self.seeds[name] = seed
            self.problems[name] = q
            self.blocks[name] = (offset, offset + len(q[0]))
            offset += len(q[0])
            initial.append(
                f.tensor(s.solution_variables)
                if self.warm_state is None
                and s.quadrature_order == order
                and tuple(seed[1]) == s.phase_roots
                and len(s.solution_variables) == len(q[0])
                else q[0]
            )
            lower.append(q[1])
            upper.append(q[2])
            rel = (
                s.total_energy_j
                - float((n * f.molar_masses).sum().cpu()) * c.gravity_m_s2 * c.elevation_m
            )
            self.scales[name] = (
                self.column_guesses[name][q[4]].clamp_min(self.column_guesses[name].sum() * 1e-12),
                max(1.0, abs(rel), float(n.sum().cpu()) * R * s.temperature_k),
            )
        for port in m.ports:
            name = port.column
            seed = self.seeds[name]
            active = self.problems[name][4]
            count = len(active)
            for phase, root in enumerate(seed[1]):
                x = seed[0][phase][active].clamp_min(1e-30)
                row = torch.cat((f.tensor([math.log(seed[3])]), (x[:-1] / x[-1]).log()))
                self.portblocks.append((port, phase, root, offset, offset + order * count))
                initial.append(row.repeat(order))
                offset += order * count
                lower.append(
                    f.tensor([math.log(f.min_pressure_pa)] + [-100.0] * (count - 1)).repeat(order)
                )
                upper.append(
                    f.tensor([math.log(f.max_pressure_pa)] + [100.0] * (count - 1)).repeat(order)
                )
        self.port_eos_plans = {}
        for name, column in m.columns.items():
            blocks = [
                (index, a, b, root)
                for index, (port, _phase, root, a, b) in enumerate(self.portblocks)
                if port.column == name
            ]
            if blocks and column._batch_column_eos:
                plan = PhaseEOSGroups(
                    column.fluid, [root for _index, _a, _b, root in blocks for _ in range(order)]
                )
                self.port_eos_plans[name] = (plan, blocks)
        if offset > t.maximum_unknowns:
            raise ValueError("Coupled graph unknown budget exceeded")
        # Row blocks have the same offsets as unknown blocks. A column/pipe
        # flux couples only its two bodies and that aperture's trace variables.
        block_dependencies = {}
        base_open = {link.name: self.openings[link.name] for link in t.switches}
        neighbours = [{i} for i in range(len(t.nodes))]
        for link, (a, b) in zip(t.links, t.ends):
            if base_open[link.name]:
                neighbours[a].add(b)
                neighbours[b].add(a)
        for i, connected in enumerate(neighbours):
            block_dependencies[(offsets[i], offsets[i + 1])] = {
                (offsets[j], offsets[j + 1]) for j in connected
            }
        for block in self.blocks.values():
            block_dependencies[block] = {block}
        for port, phase, root, a, b in self.portblocks:
            i = t.names.index(port.node)
            pipe_block = (offsets[i], offsets[i + 1])
            col_block = self.blocks[port.column]
            block_dependencies[(a, b)] = {(a, b), col_block}
            block_dependencies[pipe_block].update({col_block, (a, b)})
            block_dependencies[col_block].update({pipe_block, (a, b)})
            hydrostatic = getattr(base, "hydrostatic_faces", None)
            for entry in () if hydrostatic is None else hydrostatic.entries:
                if entry.cell == i:
                    block_dependencies[col_block].add((entry.offset, entry.offset + entry.count))
        body_blocks = {
            **{name: (offsets[i], offsets[i + 1]) for i, name in enumerate(t.names)},
            **self.blocks,
        }
        for link in m.heat_links:
            if self.openings[link.name]:
                a, b = body_blocks[link.source], body_blocks[link.target]
                block_dependencies[a].add(b)
                block_dependencies[b].add(a)
        rows, cols = [], []
        for (a, b), dependencies in sorted(block_dependencies.items()):
            for row in range(a, b):
                for c, d in sorted(dependencies):
                    for col in range(c, d):
                        rows.append(row)
                        cols.append(col)
        traces = getattr(base, "reservoir_traces", None)
        compressors = getattr(base, "compressors", None)
        hydrostatic = getattr(base, "hydrostatic_faces", None)
        populations = getattr(base, "droplet_populations", None)
        base_rows, base_cols = t._sparsity(
            offsets,
            base_open,
            () if traces is None else traces.entries,
            () if compressors is None else compressors.entries,
            () if hydrostatic is None else hydrostatic.entries,
            () if populations is None else populations.entries,
        )
        support = sorted(
            set(zip(rows, cols)) | set(zip(base_rows, base_cols)) | base.additional_sparsity
        )
        pattern = (
            ([p[0] for p in support], [p[1] for p in support]) if offset > t.dense_limit else None
        )
        self.old_entropy = {}
        for name, s in self.state.columns:
            c = m.columns[name]
            old_problem = c._problem(
                f.tensor(s.component_moles),
                s.total_energy_j,
                (
                    [f.tensor(x) for x in s.phase_seeds],
                    list(s.phase_roots),
                    f.tensor(s.interface_heights_m),
                    math.exp(s.solution_variables[0]),
                    s.temperature_k,
                    f.tensor(s.chemical_constants),
                ),
                s.quadrature_order,
            )
            self.old_entropy[name] = float(
                old_problem[3](f.tensor(s.solution_variables))[1]["entropy_j_k"].cpu()
            )

        def evaluate(z):
            bodies = {}
            raw = {}
            walls = {}
            heats = {}
            ambient = {}
            for name, s in self.state.columns:
                a, b = self.blocks[name]
                raw[name], bodies[name] = self.problems[name][3](z[a:b])
                temp = bodies[name]["temperature"]
                w = m.walls.get(name)
                if w is None:
                    walls[name] = temp
                    heats[name] = temp * 0
                    ambient[name] = temp * 0
                else:
                    oldtw = dict(self.state.walls)[name]
                    walls[name] = (
                        w.heat_capacity_j_k * oldtw
                        + dt * (w.fluid_ua_w_k * temp + w.ambient_ua_w_k * w.ambient_temperature_k)
                    ) / (w.heat_capacity_j_k + dt * (w.fluid_ua_w_k + w.ambient_ua_w_k))
                    heats[name] = w.fluid_ua_w_k * (walls[name] - temp)
                    ambient[name] = w.ambient_ua_w_k * (w.ambient_temperature_k - walls[name])

            def exchange(data):
                net = [z.new_zeros(len(f.names) + 2) for _ in t.nodes]
                cn = {name: z.new_zeros(len(f.names)) for name in m.columns}
                ce = {
                    name: z.new_tensor(self.column_work_j.get(name, 0.0) / dt) for name in m.columns
                }
                cs = {name: z.new_zeros(()) for name in m.columns}
                chemical = []
                details = {}
                valid = z.new_ones((), dtype=torch.bool)
                pieces = []
                decoded_ports = {}
                for name, (plan, blocks) in self.port_eos_plans.items():
                    column = m.columns[name]
                    active = self.problems[name][4]
                    local = torch.cat(
                        [z[a:b].reshape(order, len(active)) for _index, a, b, _root in blocks]
                    )
                    pp, xx, qq = column._decode_grouped_nodes(
                        local, active, bodies[name]["temperature"], plan
                    )
                    for j, (index, _a, _b, _root) in enumerate(blocks):
                        region = slice(j * order, (j + 1) * order)
                        decoded_ports[index] = (
                            pp[region],
                            xx[region],
                            {key: value[region] for key, value in qq.items()},
                        )
                for port_index, (port, phase, root, a, b) in enumerate(self.portblocks):
                    name = port.column
                    c = m.columns[name]
                    d = bodies[name]
                    active = self.problems[name][4]
                    bounds = torch.cat(
                        (z.new_zeros(1), d["interfaces"], z.new_tensor([c.geometry.height_m]))
                    )
                    helper = SimpleNamespace(column=c)
                    heights, weights = StratifiedPRFlow._aperture(
                        helper, bounds, phase, port, order
                    )
                    if port_index in decoded_ports:
                        pp, xx, q = decoded_ports[port_index]
                    else:
                        pp, xx, q = c._decode_nodes(
                            z[a:b].reshape(order, len(active)), active, d["temperature"], root
                        )
                    mu = (
                        xx[:, active].clamp_min(1e-300).log()
                        + q["log_fugacity_coefficients"][:, active]
                        + (pp / f.pref).log()[:, None]
                    )
                    chemical.append(
                        (
                            mu
                            + c.gravity_m_s2
                            * heights[:, None]
                            * f.molar_masses[active]
                            / (R * d["temperature"])
                            - d["constants"]
                        ).flatten()
                    )
                    i = t.names.index(port.node)
                    link = PREquilibriumLink(
                        port.name,
                        port.column,
                        port.node,
                        port.diameter_m,
                        target_port=port.pipe_port,
                    )
                    endpoint, _, normal, extra = t._endpoint(i, data[i], "target", link)
                    fluxsum = z.new_zeros(len(f.names) + 2)
                    entropy = z.new_zeros(())

                    def ray(
                        height,
                        point_pressure,
                        composition,
                        properties,
                        c=c,
                        port=port,
                        endpoint=endpoint,
                        extra=extra,
                        i=i,
                        normal=normal,
                    ):
                        phi = c.gravity_m_s2 * (c.elevation_m + height - t.reference_elevation_m)
                        rho = properties["density_kg_m3"]
                        vm = properties["molar_volume_m3_mol"]
                        energy = properties["molar_internal_energy_j_mol"] / vm + rho * phi
                        uc = torch.cat((composition / vm, z.new_zeros(1), energy.reshape(1)))
                        fc = torch.cat(
                            (z.new_zeros(len(f.names)), point_pressure.reshape(1), z.new_zeros(1))
                        )
                        colend = (
                            uc,
                            fc,
                            point_pressure,
                            z.new_zeros(()),
                            rho,
                            properties["molar_entropy_j_mol_k"] / vm,
                        )
                        # The pipe is cross-section averaged; retain the same
                        # datum at both ends of each aperture quadrature ray.
                        du = phi - c.gravity_m_s2 * (
                            c.elevation_m + port.height_m - t.reference_elevation_m
                        )
                        pu, pf, pr, pv, pd, ps = endpoint
                        pu = torch.cat((pu[:-1], (pu[-1] + pd * du).reshape(1)))
                        pf = torch.cat((pf[:-1], (pf[-1] + pd * du * pv).reshape(1)))
                        if self.openings[port.name]:
                            flux, sflux, _, _, ok = hllc_port(
                                colend, (pu, pf, pr, pv, pd, ps), t.wave_speed_bound_m_s
                            )
                            # Actual phase donor correction on the pipe side.
                            actual = extra["actual"]
                            bulk = extra["bulk"]
                            actual = torch.cat(
                                (
                                    actual[:, :-1],
                                    (
                                        actual[:, -1]
                                        + (actual[:, :-2] * f.molar_masses).sum(-1) * du
                                    )[:, None],
                                ),
                                -1,
                            )
                            bulk = torch.cat(
                                (
                                    bulk[:, :-1],
                                    (bulk[:, -1] + (bulk[:, :-2] * f.molar_masses).sum(-1) * du)[
                                        :, None
                                    ],
                                ),
                                -1,
                            )
                            flux = (
                                flux
                                + torch.where(
                                    extra["actual_velocity"][:, None] < 0,
                                    actual,
                                    torch.zeros_like(actual),
                                ).sum(0)
                                - torch.where(
                                    extra["bulk_velocity"] < 0, bulk, torch.zeros_like(bulk)
                                ).sum(0)
                            )
                            sflux = (
                                sflux
                                + torch.where(
                                    extra["actual_velocity"] < 0,
                                    extra["actual_entropy"],
                                    torch.zeros_like(extra["actual_entropy"]),
                                ).sum()
                                - torch.where(
                                    extra["bulk_velocity"] < 0,
                                    extra["bulk_entropy"],
                                    torch.zeros_like(extra["bulk_entropy"]),
                                ).sum()
                            )
                        else:
                            pressure = (
                                data[i]["p"]
                                + data[i]["rho"] * data[i]["u"].square()
                                + data[i]["drift_stress"]
                                + t.wave_speed_bound_m_s * data[i]["rho"] * data[i]["u"] * normal
                            )
                            flux = torch.cat(
                                (z.new_zeros(len(f.names)), pressure.reshape(1), z.new_zeros(1))
                            )
                            sflux = z.new_zeros(())
                            ok = z.new_ones((), dtype=torch.bool)
                        return flux, sflux, ok

                    if self.batch_aperture_rays:
                        ray_flux, ray_entropy, ray_valid = torch.vmap(ray)(heights, pp, xx, q)
                    else:
                        rays = [
                            ray(heights[j], pp[j], xx[j], {k: v[j] for k, v in q.items()})
                            for j in range(order)
                        ]
                        ray_flux = torch.stack([r[0] for r in rays])
                        ray_entropy = torch.stack([r[1] for r in rays])
                        ray_valid = torch.stack([r[2] for r in rays])
                    valid = valid & ((weights <= 0) | ray_valid).all()
                    # Keep the reference accumulation order; batching changes
                    # ray execution, not quadrature weights or summation.
                    for j in range(order):
                        fluxsum = fluxsum + weights[j] * ray_flux[j]
                        entropy = entropy + weights[j] * ray_entropy[j]
                    area = math.pi * port.diameter_m**2 / 4
                    physical = area * fluxsum
                    # Ray-resolved PE belongs to the column. The 1-D pipe
                    # receives flux referred to its centreline potential.
                    net[i] = net[i] + torch.cat(
                        (
                            physical[:-2],
                            (-normal * physical[-2]).reshape(1),
                            physical[-1].reshape(1),
                        )
                    )
                    if self.openings[port.name]:
                        cn[name] = cn[name] - physical[:-2]
                        ce[name] = (
                            ce[name]
                            - physical[-1]
                            - (physical[:-2] * f.molar_masses).sum()
                            * c.gravity_m_s2
                            * t.reference_elevation_m
                        )
                        cs[name] = cs[name] - area * entropy
                    if port.name not in details:
                        details[port.name] = {
                            "flux": z.new_zeros(len(f.names) + 2),
                            "entropy_flux": z.new_zeros(()),
                            "area_fractions": [],
                        }
                    details[port.name]["flux"] = details[port.name]["flux"] + physical
                    details[port.name]["entropy_flux"] = (
                        details[port.name]["entropy_flux"] + area * entropy
                    )
                    details[port.name]["area_fractions"].append(weights.sum())
                    pieces.append((port, root, heights, weights, pp, xx, q))
                incoming_entropy = -sum(cs.values())
                temperatures = {
                    **{name: d["t"] for name, d in zip(t.names, data)},
                    **{name: body["temperature"] for name, body in bodies.items()},
                }
                thermal_rates, heat_records = thermal_exchange(
                    m.heat_links, temperatures, self.openings
                )
                for name, rate in thermal_rates.items():
                    if name in ce:
                        ce[name] = ce[name] + rate
                        cs[name] = cs[name] + rate / temperatures[name]
                    else:
                        i = t.names.index(name)
                        net[i] = net[i] + torch.cat(
                            (z.new_zeros(len(f.names) + 1), rate.reshape(1))
                        )
                        incoming_entropy = incoming_entropy + rate / temperatures[name]
                material = None
                if self.material is not None:
                    material = self.material.exchange(data)
                    net = [a + b for a, b in zip(net, material["net"])]
                    incoming_entropy = incoming_entropy + material["incoming_entropy"]
                    valid = valid & material["valid"]
                return {
                    "material": material,
                    "net": net,
                    "valid": valid,
                    "incoming_entropy": incoming_entropy,
                    "heat_links": heat_records,
                    "cn": cn,
                    "ce": ce,
                    "cs": cs,
                    "chemical": chemical,
                    "ports": details,
                    "pieces": pieces,
                }

            br, result = base(z[: self.size], exchange)
            ex = result["exchange"]
            res = []
            for name, s in self.state.columns:
                c = m.columns[name]
                active = self.problems[name][4]
                count = len(active)
                nscale, escale = self.scales[name]
                rr = raw[name]
                mass = rr[-count - 1 : -1] - dt * ex["cn"][name][active] / nscale
                relative = (
                    ex["ce"][name]
                    - (ex["cn"][name] * f.molar_masses).sum() * c.gravity_m_s2 * c.elevation_m
                )
                energy = rr[-1] - dt * (relative + heats[name] + self.heat.get(name, 0.0)) / escale
                res.append(torch.cat((rr[: -count - 1], mass, energy.reshape(1))))
            result["column_bodies"] = bodies
            result["column_walls"] = walls
            result["column_heat"] = heats
            result["column_ambient"] = ambient
            return torch.cat((br, *res, *ex["chemical"])), result

        blocks = [list(block) for block in base.linear_blocks]
        for name, (a, b) in self.blocks.items():
            block = list(range(a, b))
            for port, _phase, _root, left, right in self.portblocks:
                if port.column == name:
                    block.extend(range(left, right))
            blocks.extend(block[j : j + 256] for j in range(0, len(block), 256))
        evaluate.linear_blocks = blocks
        evaluate.maximum_update = torch.cat(
            (
                base.maximum_update,
                f.tensor([math.inf] * (offset - len(base.maximum_update))),
            )
        )
        self.evaluate = evaluate
        return torch.cat(initial), torch.cat(lower), torch.cat(upper), evaluate, pattern

    def accept(self, y, result):
        m = self.model
        f = m.transport.fluid
        dt = self.dt
        ex = result["exchange"]
        columns = {}
        walls = {}
        reports = {}
        entropy = 0.0
        heat = 0.0
        for name, old in self.state.columns:
            c = m.columns[name]
            d = result["column_bodies"][name]
            active = self.problems[name][4]
            target_n = f.tensor(old.component_moles) + dt * ex["cn"][name]
            n = d["component_moles"]
            if float((n - target_n).abs().max().cpu()) > 1e-8 + 1e-9 * float(n.sum().cpu()):
                raise ConservationError("Column graph component balance rejected")
            energy = old.total_energy_j + dt * (
                ex["ce"][name] + result["column_heat"][name] + self.heat.get(name, 0.0)
            )
            if not bool((n[active] > 0).all().cpu()):
                raise HydraulicConvergenceError("Column component exhausted")
            roots = self.seeds[name][1]
            constants = torch.zeros_like(n).index_copy(0, active, d["constants"])
            seed = (
                [x[0] for x in d["compositions"][: len(roots)]],
                roots,
                d["interfaces"],
                float(d["pressures"][0][0].cpu()),
                float(d["temperature"].cpu()),
                constants,
            )
            a, b = self.blocks[name]
            try:
                s, report = c._accept(n, float(energy.cpu()), seed, self.order, y[a:b])
            except (PropertyModelError, HydraulicConvergenceError):
                replacement = c._seed(n, float(energy.cpu()), seed[3], seed[4])
                if tuple(replacement[1]) == tuple(seed[1]) and all(
                    float((a - b).abs().max().cpu()) < 1e-7 for a, b in zip(replacement[0], seed[0])
                ):
                    raise
                self.retry_seeds[name] = replacement
                raise _TopologyRetry("Column phase topology changed") from None
            columns[name] = s.to_dict()
            reports[name] = {
                **report,
                "transport_component_residuals_mol": (n - target_n).cpu().tolist(),
            }
            entropy += (
                report["entropy_j_k"] - self.old_entropy[name] - dt * float(ex["cs"][name].cpu())
            )
            w = m.walls.get(name)
            if w:
                tw = float(result["column_walls"][name].cpu())
                oldtw = dict(self.state.walls)[name]
                if (
                    not w.min_temperature_k <= tw <= w.max_temperature_k
                    or abs(tw - oldtw) > w.max_temperature_change_k
                ):
                    raise HydraulicConvergenceError("Column wall step/domain rejected")
                walls[name] = tw
                entropy += (
                    w.heat_capacity_j_k * math.log(tw / oldtw)
                    - dt * float(result["column_ambient"][name].cpu()) / w.ambient_temperature_k
                )
            q = self.heat.get(name, 0.0)
            if q:
                utility = self.utility[name]
                if q * (utility - s.temperature_k) < -1e-10:
                    raise PropertyModelError("Column heat violates passive utility direction")
                entropy -= dt * q / utility
            heat += dt * (float(result["column_ambient"][name].cpu()) + q)
        aperture_checks = {name: ([], [], []) for name in m.columns}
        for port, root, heights, weights, pp, xx, q in ex["pieces"]:
            c = m.columns[port.column]
            active = weights > 0
            head = q["density_kg_m3"] * c.gravity_m_s2 * port.diameter_m / pp
            if bool((active & (head > m.maximum_transverse_head_fraction)).any().cpu()):
                raise PropertyModelError(
                    "Aperture requires resolved transverse pipe inlet hydrodynamics"
                )
            if bool(active.any().cpu()):
                pressures, compositions, properties = aperture_checks[port.column]
                pressures.append(pp[active])
                compositions.append(xx[active])
                properties.append({k: v[active] for k, v in q.items()})
        for name, (pressures, compositions, properties) in aperture_checks.items():
            m.columns[name]._check_locals(
                pressures, columns[name]["temperature_k"], compositions, properties
            )
        material = (
            {}
            if self.material is None
            else {"material": self.material.accept(y, {**result, "exchange": ex["material"]})}
        )
        return {
            **material,
            "columns": columns,
            "walls": walls,
            "reports": reports,
            "heat_j": heat,
            "entropy_production_j_k": entropy
            + dt
            * sum(float(row["entropy_production_w_k"].cpu()) for row in ex["heat_links"].values()),
            "heat_links": thermal_records(ex["heat_links"]),
            "ports": {
                n: {
                    "flux": p["flux"].cpu().tolist(),
                    "entropy_flux": float(p["entropy_flux"].cpu()),
                    "area_fractions": [float(v.cpu()) for v in p["area_fractions"]],
                }
                for n, p in ex["ports"].items()
            },
        }
