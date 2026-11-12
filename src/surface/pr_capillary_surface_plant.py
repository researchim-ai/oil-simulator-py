"""Conservative first-order composition of capillary and ordinary surface graphs.

One public copy of each finite column. Capillary transport acts first; the
ordinary implicit graph follows with a self-consistent mechanical debit for
changed capillary endpoint surface energy. This is Lie splitting, not a
monolithic all-physics Newton; full/two-half control bounds splitting error.
"""

import math
from contextlib import ExitStack
from dataclasses import dataclass, replace

from .equilibrium_cache import cached_flashes
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .pr_active_multifluid_pipe import PRActiveMultifluidState
from .pr_capillary_facility import PRCapillaryFacility, PRCapillaryFacilityState
from .pr_gravity_transport import PRGravityTransportState
from .pr_open_gravity_transport import PROpenGravityTransportNetwork
from .pr_open_transport import _BoundaryProperties, _OpenCoupling
from .time_interval import interval_step_s, split_step_s


@dataclass(frozen=True)
class PRCapillarySurfaceState:
    surface: PRGravityTransportState
    capillary_pipes: tuple
    layer_slots: tuple
    next_step_s: float
    model_id: str

    def __post_init__(self):
        for field in ("capillary_pipes", "layer_slots"):
            rows = tuple(getattr(self, field))
            if len({n for n, _ in rows}) != len(rows):
                raise ValueError("Duplicate plant checkpoint names")
            if field == "layer_slots":
                rows = tuple((n, tuple(v)) for n, v in rows)
            object.__setattr__(self, field, tuple(sorted(rows)))
        _number(self.next_step_s, "next plant step", positive=True)

    @property
    def time_s(self):
        return self.surface.time_s

    @property
    def transport(self):
        return self.surface.transport

    @property
    def columns(self):
        return self.surface.columns

    @property
    def walls(self):
        return self.surface.walls

    def to_dict(self):
        return {
            "surface": self.surface.to_dict(),
            "capillary_pipes": {n: s.to_dict() for n, s in self.capillary_pipes},
            "layer_slots": dict(self.layer_slots),
            "next_step_s": self.next_step_s,
            "time_s": self.time_s,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, raw):
        if set(raw) != {
            "surface",
            "capillary_pipes",
            "layer_slots",
            "next_step_s",
            "time_s",
            "model_id",
        }:
            raise ValueError("Explicit complete capillary plant checkpoint required")
        state = cls(
            PRGravityTransportState.from_dict(raw["surface"]),
            tuple(
                (n, PRActiveMultifluidState.from_dict(s))
                for n, s in sorted(raw["capillary_pipes"].items())
            ),
            tuple((n, tuple(v)) for n, v in sorted(raw["layer_slots"].items())),
            raw["next_step_s"],
            raw["model_id"],
        )
        if state.time_s != raw["time_s"]:
            raise ValueError("Capillary plant clocks differ")
        return state


class PRCapillarySurfacePlant:
    def __init__(
        self,
        surface,
        capillary,
        *,
        maximum_geometry_iterations=16,
        relative_tolerance=1e-4,
        temperature_tolerance_k=0.02,
        velocity_tolerance_m_s=0.005,
    ):
        if (
            type(surface) is not PROpenGravityTransportNetwork
            or type(capillary) is not PRCapillaryFacility
        ):
            raise TypeError("Explicit open gravitational graph and capillary facility required")
        self.surface, self.capillary = surface, capillary
        self.graph, self.transport, self.fluid = surface.graph, surface.transport, surface.fluid
        if not set(capillary.columns) <= set(self.graph.columns):
            raise ValueError("Capillary columns must be actual shared surface vertices")
        for name, column in capillary.columns.items():
            host = self.graph.columns[name]
            if column.model_id != host.model_id or column.fluid.device != host.fluid.device:
                raise ValueError("Shared column EOS, geometry or device differs")
        if set(capillary.lines) & (set(self.transport.names) | set(self.graph.columns)):
            raise ValueError("Distinct physical capillary line and surface body names required")
        if type(maximum_geometry_iterations) is not int or maximum_geometry_iterations < 1:
            raise ValueError("Positive geometry closure budget required")
        self.maximum_geometry_iterations = maximum_geometry_iterations
        for name, value in [
            ("relative_tolerance", relative_tolerance),
            ("temperature_tolerance_k", temperature_tolerance_k),
            ("velocity_tolerance_m_s", velocity_tolerance_m_s),
        ]:
            setattr(self, name, _number(value, name, positive=True))
        self.model_id = _digest(self.to_dict())

    @property
    def feeds(self):
        return self.surface.feeds

    @property
    def pressure_ports(self):
        return self.surface.pressure_ports

    def to_dict(self):
        return {
            "model": "pr_capillary_surface_plant_v1",
            "surface": self.surface.to_dict(),
            "capillary": self.capillary.to_dict(),
            **{
                k: getattr(self, k)
                for k in (
                    "maximum_geometry_iterations",
                    "relative_tolerance",
                    "temperature_tolerance_k",
                    "velocity_tolerance_m_s",
                )
            },
        }

    @classmethod
    def from_dict(cls, raw, *, device="cpu"):
        raw = dict(raw)
        if raw.pop("model", None) != "pr_capillary_surface_plant_v1":
            raise ValueError("Unknown capillary surface plant")
        raw["surface"] = PROpenGravityTransportNetwork.from_dict(raw["surface"], device=device)
        raw["capillary"] = PRCapillaryFacility.from_dict(raw["capillary"], device=device)
        return cls(**raw)

    def fluids(self):
        result = self.surface.fluids()
        for f in [
            *[c.fluid for c in self.capillary.columns.values()],
            *[r["pipe"].fluid for r in self.capillary.lines.values()],
        ]:
            if all(f is not q for q in result):
                result.append(f)
        return result

    def capillary_state(self, state, *, columns=None, slots=None):
        values = dict(state.columns) if columns is None else dict(columns)
        return PRCapillaryFacilityState(
            state.capillary_pipes,
            tuple((n, values[n]) for n in self.capillary.columns),
            state.layer_slots if slots is None else tuple(slots),
            state.next_step_s,
            self.capillary.model_id,
        )

    def initialize(self, surface_state, capillary_state, *, next_step_s=None):
        self.surface.inspect(surface_state)
        self.capillary.inspect(capillary_state)
        if surface_state.time_s != capillary_state.time_s:
            raise ValueError("Shared plant clocks differ")
        for n, c in capillary_state.columns:
            if c != dict(surface_state.columns)[n]:
                raise ValueError("One authoritative initial column profile required")
        state = PRCapillarySurfaceState(
            surface_state,
            capillary_state.pipes,
            capillary_state.layer_slots,
            capillary_state.next_step_s if next_step_s is None else next_step_s,
            self.model_id,
        )
        self.inspect(state)
        return state

    def inspect(self, state):
        if (
            not isinstance(state, PRCapillarySurfaceState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
        ):
            raise PropertyModelError("Capillary plant state/model differs")
        _number(state.next_step_s, "next plant step", positive=True)
        self.surface.inspect(state.surface)
        self.capillary.inspect(self.capillary_state(state))
        if any(p.time_s != state.time_s for _, p in state.capillary_pipes):
            raise ValueError("Capillary and surface clocks differ")

    def _bind(self, previous, columns):
        return self.capillary.rebind_columns(
            previous, {n: columns[n] for n in self.capillary.columns}
        ).layer_slots

    def capillary_stocks(self, state):
        terms = self.capillary.conserved_terms(
            self.capillary_state(state),
            include_columns=False,
            reference_elevation_m=self.transport.reference_elevation_m,
        )
        return [math.fsum(v) for v in terms[:-1]], math.fsum(terms[-1])

    def totals(self, state):
        nc = len(self.fluid.names)
        n, e = self.capillary_stocks(state)
        n = [
            math.fsum(
                (
                    n[j],
                    *(s.component_moles[j] for _, s in state.transport.nodes),
                    *(s.component_moles[j] for _, s in state.columns),
                )
            )
            for j in range(nc)
        ]
        e = math.fsum(
            (
                e,
                *(s.total_energy_j for _, s in state.transport.nodes),
                *(s.total_energy_j for _, s in state.columns),
                self.transport.rotor_energy(state.transport),
                *(
                    node.wall.heat_capacity_j_k * s.wall_temperature_k
                    for node, (_, s) in zip(self.transport.nodes, state.transport.nodes)
                    if node.wall
                ),
                *(self.graph.walls[name].heat_capacity_j_k * t for name, t in state.walls),
            )
        )
        # Columns use absolute potential energy; ordinary graph uses its datum.
        masses = self.fluid.molar_masses.detach().cpu().tolist()
        e -= (
            self.transport.gravity_m_s2
            * self.transport.reference_elevation_m
            * math.fsum(n * m for _, s in state.columns for n, m in zip(s.component_moles, masses))
        )
        return n, e

    def step(
        self,
        state,
        dt_s,
        *,
        openings=None,
        port_openings=None,
        heat_w=None,
        utility_temperature_k=None,
        loss_coefficients=None,
        drive_torque_nm=None,
        restriction_openings=None,
        normal_relaxation=True,
    ):
        self.inspect(state)
        dt = _number(dt_s, "plant step", positive=True)
        dt = (state.time_s + dt) - state.time_s
        if dt <= 0:
            raise ValueError("Plant step below clock resolution")
        # All commands are validated before any trial, with immutable stocks.
        commands = {
            **self.transport.default_openings,
            **{p.name: True for p in (*self.graph.ports, *self.graph.heat_links)},
        }
        if set(openings or {}) - set(commands) or any(
            type(v) is not bool for v in (openings or {}).values()
        ):
            raise ValueError("Known binary surface commands required")
        commands.update(openings or {})
        external = self.surface._commands(port_openings)
        heat = dict(heat_w or {})
        utility = dict(utility_temperature_k or {})
        losses = dict(loss_coefficients or {})
        if (set(heat) | set(utility)) - (set(self.transport.names) | set(self.graph.columns)):
            raise ValueError("Unknown thermal body")
        for n, q in heat.items():
            _number(q, "heat power")
            if q and n not in utility:
                raise ValueError("Heat requires utility temperature")
        for q in utility.values():
            _number(q, "utility temperature", positive=True)
        pipe_names = {n.name for n in self.transport.nodes if n.kind == "pipe_cell"}
        if set(losses) - pipe_names:
            raise ValueError("Unknown resistance cell")
        for q in losses.values():
            _number(q, "loss coefficient", nonnegative=True)
        with ExitStack() as stack:
            stack.enter_context(self.capillary.cached_inspections())
            for f in self.fluids():
                stack.enter_context(cached_flashes(f))
            cap_old = self.capillary_state(state)
            cap_end, cap_report = self.capillary.step(
                cap_old, dt, normal_relaxation=normal_relaxation
            )
            midpoint = replace(
                state.surface,
                columns=tuple((n, dict(cap_end.columns).get(n, c)) for n, c in state.columns),
            )
            prepared = _BoundaryProperties(self.surface, external)
            old_models = self.capillary.inspect(cap_end)
            work = {n: 0.0 for n in self.capillary.columns}
            for iteration in range(self.maximum_geometry_iterations):
                material = _OpenCoupling(self.surface, prepared, midpoint.transport, dt, commands)
                end, surface_report = self.graph._step(
                    midpoint,
                    dt,
                    commands,
                    heat,
                    utility,
                    losses,
                    drive_torque_nm=drive_torque_nm,
                    restriction_openings=restriction_openings,
                    _material=material,
                    _column_work_j={n: -v for n, v in work.items()},
                )
                slots = self._bind(cap_end, dict(end.columns))
                candidate = PRCapillarySurfaceState(
                    end, cap_end.pipes, slots, state.next_step_s, self.model_id
                )
                cap_final = self.capillary_state(candidate)
                models = self.capillary.inspect(cap_final)
                actual = {n: 0.0 for n in self.capillary.columns}
                for line, row in self.capillary.lines.items():
                    ps = dict(cap_end.pipes)[line]
                    for side, port in enumerate(row["ports"]):
                        if port is not None:
                            actual[port["column"]] += self.capillary.boundary_energy(
                                models[line], ps, side
                            ) - self.capillary.boundary_energy(old_models[line], ps, side)
                if all(
                    abs(actual[n] - work[n])
                    <= 8 * math.ulp(dict(end.columns)[n].total_energy_j) + 1e-20
                    for n in work
                ):
                    break
                work = actual
            else:
                raise HydraulicConvergenceError("Capillary/surface geometry work closure failed")
            self.inspect(candidate)
            # Full final augmented energy is checked, not just the frozen work estimate.
            n0, e0 = self.totals(state)
            n1, e1 = self.totals(candidate)
            cap_n, cap_e = self._external_capillary(cap_report)
            incoming = [a + b for a, b in zip(surface_report["material_component_moles"], cap_n)]
            energy = (
                surface_report["material_energy_j"]
                + cap_e
                + surface_report["total_heat_j"]
                + surface_report["total_work_j"]
                + math.fsum(work.values())
            )
            residual = [b - a - v for a, b, v in zip(n0, n1, incoming)] + [e1 - e0 - energy]
            limits = [
                1e-20 + 256 * (math.ulp(a) + math.ulp(b) + math.ulp(v))
                for a, b, v in zip(n0, n1, incoming)
            ]
            limits.append(1e-18 + 256 * (math.ulp(e0) + math.ulp(e1) + math.ulp(energy)))
            if any(not math.isfinite(r) or abs(r) > t for r, t in zip(residual, limits)):
                raise ConservationError("Shared plant component/augmented energy balance rejected")
            # Each checked suboperator includes its own physical boundary entropy;
            # columns cancel once in the sum and surface energy carries no entropy.
            entropy = (
                cap_report["entropy_production_j_k"] + surface_report["entropy_production_j_k"]
            )
            if not math.isfinite(entropy) or entropy < -1e-10:
                raise ConservationError("Shared plant entropy rejected")
            return candidate, {
                "state": candidate.to_dict(),
                "dt_s": dt,
                "midpoint": midpoint.to_dict(),
                "capillary_initial": cap_old.to_dict(),
                "capillary_end": cap_end.to_dict(),
                "capillary_final": cap_final.to_dict(),
                "capillary": cap_report,
                "surface": surface_report,
                "geometry_work_j": work,
                "actual_geometry_work_j": actual,
                "geometry_iterations": iteration + 1,
                "conservation_residuals": residual,
                "roundoff_allowances": limits,
                "material_component_moles": incoming,
                "material_energy_j": surface_report["material_energy_j"] + cap_e,
                "total_heat_j": surface_report["total_heat_j"],
                "total_work_j": surface_report["total_work_j"] + math.fsum(work.values()),
                "entropy_production_j_k": entropy,
                "openings": commands,
                "port_openings": external,
                "qualification": "Conservative first-order Lie composition with one shared column state and endpoint surface-work closure. Not a monolithic all-physics Newton or a field calibration.",
            }

    def _external_capillary(self, report):
        n, energy, _ = self.capillary.external_exchange(
            report, reference_elevation_m=self.transport.reference_elevation_m
        )
        return n, energy

    def error(self, coarse, fine):
        error = self.graph._error(coarse.surface, fine.surface)
        for (_, a), (_, b) in zip(coarse.capillary_pipes, fine.capillary_pipes):
            for x, y in zip(a.cells, b.cells):
                if tuple(n for n, _ in x) != tuple(n for n, _ in y):
                    return max(error, 2.0)
                for (_, p), (_, q) in zip(x, y):
                    for u, v in zip(p.component_moles, q.component_moles):
                        error = max(
                            error,
                            abs(u - v) / (1e-22 + self.relative_tolerance * max(abs(u), abs(v))),
                        )
                    error = max(
                        error,
                        abs(p.volume_m3 - q.volume_m3)
                        / (1e-24 + self.relative_tolerance * max(p.volume_m3, q.volume_m3)),
                        abs(p.temperature_k - q.temperature_k) / self.temperature_tolerance_k,
                        abs(p.pressure_pa - q.pressure_pa)
                        / (1.0 + self.relative_tolerance * max(p.pressure_pa, q.pressure_pa)),
                        abs(p.total_energy_j - q.total_energy_j)
                        / (
                            1e-20
                            + self.relative_tolerance
                            * max(1e-12, abs(p.total_energy_j), abs(q.total_energy_j))
                        ),
                    )
                    mp = math.fsum(
                        n * c.molar_mass_kg_mol
                        for n, c in zip(p.component_moles, self.fluid.components)
                    )
                    mq = math.fsum(
                        n * c.molar_mass_kg_mol
                        for n, c in zip(q.component_moles, self.fluid.components)
                    )
                    error = max(
                        error,
                        abs(p.momentum_kg_m_s / mp - q.momentum_kg_m_s / mq)
                        / self.velocity_tolerance_m_s,
                    )
        return error

    def advance(self, state, duration_s, *, maximum_attempts=1000, **commands):
        self.inspect(state)
        target = state.time_s + _number(duration_s, "plant duration", positive=True)
        if not math.isfinite(target) or target <= state.time_s:
            raise ValueError("Plant interval below clock resolution")
        if type(maximum_attempts) is not int or maximum_attempts < 1:
            raise ValueError("Positive plant attempt budget required")
        current = state
        dt = min(state.next_step_s, duration_s, self.transport.maximum_step_s)
        steps = []
        rejected = []
        for _ in range(maximum_attempts):
            if current.time_s == target:
                return current, {
                    "state": current.to_dict(),
                    "steps": steps,
                    "rejected": rejected,
                    "scheme": "first_order_Lie_full_two_half_error_control",
                }
            dt = interval_step_s(
                current.time_s, target, dt, maximum_s=self.transport.maximum_step_s
            )
            a, b = split_step_s(current.time_s, dt)
            try:
                coarse, _ = self.step(current, dt, **commands)
                half, one = self.step(current, a, **commands)
                fine, two = self.step(half, b, **commands)
                error = self.error(coarse, fine)
                if not math.isfinite(error) or error > 1:
                    raise HydraulicConvergenceError(
                        "Capillary plant temporal/splitting error exceeded"
                    )
            except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
                rejected.append({"time_s": current.time_s, "dt_s": dt, "reason": str(exc)})
                dt /= 2
                if dt < self.transport.minimum_step_s or current.time_s + dt == current.time_s:
                    raise HydraulicConvergenceError("Plant clock resolution exhausted") from exc
                continue
            nextdt = min(self.transport.maximum_step_s, dt * (1.5 if error < 0.25 else 1.0))
            current = replace(fine, next_step_s=nextdt)
            one["temporal_error_ratio"] = error
            two["temporal_error_ratio"] = error
            steps.extend((one, two))
            dt = nextdt
        raise HydraulicConvergenceError("Capillary plant interval budget exhausted")
