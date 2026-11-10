"""Synchronous conservative capillary lines sharing finite gravity columns.

Columns are actual graph vertices, lines are edges; parallel edges and
cycles do not clone inventories. Pipe solves use beginning-of-step column
profiles. Their accepted per-port transfers are aggregated before any
column is updated, including gravitational enthalpy and end-geometry work.
This is first-order partitioned IMEX, not a monolithic facility Newton.
"""

import math
from dataclasses import dataclass, replace

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
from .pr_capillary_axial import PRCapillaryAxialPipe
from .pr_capillary_column import PRCapillaryColumnNetwork, PRCapillaryColumnState
from .pr_capillary_pressure import PRCapillaryPressure
from .stratified_pr import StratifiedPRColumn, StratifiedPRState
from .time_interval import interval_step_s


@dataclass(frozen=True)
class PRCapillaryFacilityState:
    pipes: tuple
    columns: tuple
    layer_slots: tuple
    next_step_s: float
    model_id: str

    def __post_init__(self):
        for key in ("pipes", "columns", "layer_slots"):
            rows = tuple(getattr(self, key))
            if len({n for n, _ in rows}) != len(rows):
                raise ValueError("Duplicate facility checkpoint name")
            object.__setattr__(self, key, tuple(sorted(rows)))
        object.__setattr__(self, "layer_slots", tuple((n, tuple(s)) for n, s in self.layer_slots))
        _number(self.next_step_s, "next facility step", positive=True)
        if not self.pipes or len({s.time_s for _, s in self.pipes}) != 1:
            raise ValueError("One common clock and at least one pipe required")

    @property
    def time_s(self):
        return self.pipes[0][1].time_s

    def to_dict(self):
        return {
            "pipes": {n: s.to_dict() for n, s in self.pipes},
            "columns": {n: s.to_dict() for n, s in self.columns},
            "layer_slots": dict(self.layer_slots),
            "next_step_s": self.next_step_s,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, raw):
        if set(raw) != {"pipes", "columns", "layer_slots", "next_step_s", "model_id"}:
            raise ValueError("Explicit facility checkpoint required")
        return cls(
            tuple((n, PRActiveMultifluidState.from_dict(s)) for n, s in raw["pipes"].items()),
            tuple((n, StratifiedPRState.from_dict(s)) for n, s in raw["columns"].items()),
            tuple(raw["layer_slots"].items()),
            raw["next_step_s"],
            raw["model_id"],
        )


class PRCapillaryFacility:
    def __init__(self, lines, columns):
        if not lines or not columns:
            raise ValueError("Nonempty named lines and finite columns required")
        if any(not isinstance(n, str) or not n for n in (*lines, *columns)):
            raise ValueError("Nonempty facility names required")
        self.columns = dict(sorted(columns.items()))
        if any(not isinstance(c, StratifiedPRColumn) for c in self.columns.values()):
            raise TypeError("Actual finite gravitational PR columns required")
        self.lines, self.adapters, self.sources, self.cache = {}, {}, {}, {}
        self.column_ports = {n: [] for n in self.columns}
        common = None
        for name, raw in sorted(lines.items()):
            required = {
                "pipe",
                "parameters",
                "ports",
                "maximum_unknowns",
                "linear_solver",
                "volume_diffusion",
            }
            if set(raw) != required or not isinstance(raw["pipe"], PRActiveMultifluidPipe):
                raise ValueError("Explicit pipe, material, two ports and solver settings required")
            p = raw["pipe"]
            if (
                p.boundary_mode != "walls"
                or p.elevations is None
                or any(z != p.elevations[0] for z in p.elevations)
            ):
                raise ValueError("Facility lines require horizontal surveyed nonperiodic geometry")
            signature = (p.fluid.model_id, str(p.fluid.device), p.names)
            if common is not None and signature != common:
                raise ValueError("All lines share one EOS, device and phase registry")
            common = signature
            if len(raw["ports"]) != 2:
                raise ValueError("Exactly two endpoints per facility line required")
            row = {**raw, "parameters": dict(raw["parameters"]), "ports": tuple(raw["ports"])}
            if any(
                row["parameters"].get(k) is not None
                for k in ("boundary_heights_m", "boundary_distances_m")
            ):
                raise ValueError("Facility columns own endpoint geometry")
            source = PRCapillaryPressure(p, **row["parameters"])
            for side, port in enumerate(row["ports"]):
                if port is None:
                    continue
                if set(port) != {"column", "height_m"} or port["column"] not in self.columns:
                    raise ValueError("Endpoint binds a known column and physical aperture height")
                adapter = PRCapillaryColumnNetwork(
                    p,
                    self.columns[port["column"]],
                    row["parameters"],
                    port_side=side,
                    port_height_m=port["height_m"],
                    maximum_unknowns=row["maximum_unknowns"],
                    linear_solver=row["linear_solver"],
                    volume_diffusion=row["volume_diffusion"],
                )
                self.adapters[name, side] = adapter
                self.column_ports[port["column"]].append((name, side))
            if (
                len({self.columns[q["column"]].gravity_m_s2 for q in row["ports"] if q is not None})
                > 1
            ):
                raise ValueError("Connected columns must use the same gravitational acceleration")
            PRCapillaryAxialPipe(
                source,
                maximum_unknowns=row["maximum_unknowns"],
                linear_solver=row["linear_solver"],
                volume_diffusion=row["volume_diffusion"],
            )
            self.lines[name], self.sources[name] = row, source
        if any(not ports for ports in self.column_ports.values()):
            raise ValueError("Every declared column must own at least one graph endpoint")
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_capillary_facility_imex_v1",
            "lines": {
                n: {
                    **r,
                    "pipe": r["pipe"].to_dict(),
                    "parameters": dict(r["parameters"]),
                    "ports": list(r["ports"]),
                }
                for n, r in self.lines.items()
            },
            "columns": {n: c.to_dict() for n, c in self.columns.items()},
        }

    @classmethod
    def from_dict(cls, raw, *, device="cpu"):
        if (
            set(raw) != {"model", "lines", "columns"}
            or raw["model"] != "pr_capillary_facility_imex_v1"
        ):
            raise ValueError("Explicit capillary facility model required")
        return cls(
            {
                n: {**r, "pipe": PRActiveMultifluidPipe.from_dict(r["pipe"], device=device)}
                for n, r in raw["lines"].items()
            },
            {n: StratifiedPRColumn.from_dict(c, device=device) for n, c in raw["columns"].items()},
        )

    def initialize(self, pipes, columns, *, layer_slots=None, next_step_s=1e-7):
        if set(pipes) != set(self.lines) or set(columns) != set(self.columns):
            raise ValueError("Initial facility inventory names differ")
        slots = {} if layer_slots is None else dict(layer_slots)
        if set(slots) - set(self.columns):
            raise ValueError("Unknown column phase binding")
        for name, col in columns.items():
            if name not in slots and len(col.phase_roots) == 2:
                line, side = self.column_ports[name][0]
                a = self.adapters[line, side]
                slots[name] = (a._base_source.liquid, a._base_source.gas)
        state = PRCapillaryFacilityState(
            tuple(pipes.items()),
            tuple(columns.items()),
            tuple(slots.items()),
            next_step_s,
            self.model_id,
        )
        self.inspect(state)
        return state

    def _models(self, state):
        columns, slots = dict(state.columns), dict(state.layer_slots)
        models = {}
        for name, row in self.lines.items():
            boundaries = [None, None]
            for side, port in enumerate(row["ports"]):
                if port is not None:
                    boundaries[side], _ = self.adapters[name, side]._port(
                        columns[port["column"]], slots[port["column"]]
                    )
            m = PRCapillaryAxialPipe(
                self.sources[name],
                boundaries=boundaries,
                maximum_unknowns=row["maximum_unknowns"],
                linear_solver=row["linear_solver"],
                volume_diffusion=row["volume_diffusion"],
                boundary_geometry="reservoir_cell",
            )
            if name in self.cache:
                m._linear, m.source._normal_linear = self.cache[name]
            models[name] = m
        return models

    def inspect(self, state):
        if state.model_id != self.model_id or _digest(self.to_dict()) != self.model_id:
            raise ValueError("Facility model or checkpoint identity differs")
        if (
            set(dict(state.pipes)) != set(self.lines)
            or set(dict(state.columns)) != set(self.columns)
            or set(dict(state.layer_slots)) != set(self.columns)
        ):
            raise ValueError("Facility checkpoint inventory names differ")
        for name, cs in state.columns:
            line, side = self.column_ports[name][0]
            a = self.adapters[line, side]
            binding = dict(state.layer_slots)[name]
            if (
                len(binding) != len(cs.phase_roots)
                or len(set(binding)) != len(binding)
                or (
                    len(binding) == 2
                    and tuple(binding) != (a._base_source.liquid, a._base_source.gas)
                )
            ):
                raise ValueError("Explicit physical column layer binding required")
        models = self._models(state)
        for name, ps in state.pipes:
            models[name].pipe.inspect(ps)
            models[name].source.surface_energy(ps)
        return models

    @staticmethod
    def boundary_energy(model, state, side):
        s = model.source
        hf = s.boundary_heights[side]
        if hf is None:
            return 0.0
        ci = 0 if side == 0 else -1
        h = dict(state.cells[ci])[s.liquid].volume_m3 / s.area[ci].item()
        return s.sigma * s.width / (2 * s.boundary_distances[side]) * (h - hf) ** 2

    def ledger(self, before, after, *, external_heat_j=0.0):
        old_models, new_models = self.inspect(before), self.inspect(after)
        nc = len(next(iter(self.lines.values()))["pipe"].fluid.names)
        oldp, newp = dict(before.pipes), dict(after.pipes)
        oldc, newc = dict(before.columns), dict(after.columns)
        terms = [[] for _ in range(nc + 1)]
        for name, row in self.lines.items():
            p = row["pipe"]
            elevation = p.elevations[0]
            # All attached columns agree on gravity; pipe equations store U+KE.
            attached = [r for r in row["ports"] if r is not None]
            gravity = self.columns[attached[0]["column"]].gravity_m_s2 if attached else 0.0
            for sign, state, model in (
                (-1, oldp[name], old_models[name]),
                (1, newp[name], new_models[name]),
            ):
                terms[-1].append(sign * model.source.surface_energy(state))
                for cell in state.cells:
                    for _, phase in cell:
                        for j, amount in enumerate(phase.component_moles):
                            terms[j].append(sign * amount)
                        mass = math.fsum(
                            n * c.molar_mass_kg_mol
                            for n, c in zip(phase.component_moles, p.fluid.components)
                        )
                        terms[-1].append(sign * (phase.total_energy_j + mass * gravity * elevation))
        for name in self.columns:
            for j in range(nc):
                terms[j].extend((-oldc[name].component_moles[j], newc[name].component_moles[j]))
            terms[-1].extend((-oldc[name].total_energy_j, newc[name].total_energy_j))
        terms[-1].append(-external_heat_j)
        return [math.fsum(t) for t in terms], [
            128 * math.fsum(math.ulp(v) for v in t) + (1e-18 if j == nc else 1e-20)
            for j, t in enumerate(terms)
        ]

    def step(self, state, dt_s, *, normal_relaxation=True, heaters=None):
        if type(normal_relaxation) is not bool:
            raise ValueError("Explicit normal relaxation switch required")
        models = self.inspect(state)
        dt = _number(dt_s, "facility interval", positive=True)
        dt = (state.time_s + dt) - state.time_s
        if dt <= 0 or not math.isfinite(dt):
            raise ValueError("Facility interval below clock resolution")
        heaters = {} if heaters is None else dict(heaters)
        if set(heaters) - set(self.columns):
            raise ValueError("Unknown facility heater column")
        heat, bath = {}, {}
        for name in self.columns:
            command = heaters.get(name, {"power_w": 0.0, "temperature_k": None})
            if set(command) != {"power_w", "temperature_k"}:
                raise ValueError("Physical heater power and reservoir temperature required")
            heat[name] = dt * _number(command["power_w"], "column heater power")
            bath[name] = (
                None
                if command["temperature_k"] is None
                else _number(command["temperature_k"], "heat reservoir temperature", positive=True)
            )
            if heat[name] and bath[name] is None:
                raise ValueError("Nonzero heat needs a physical reservoir temperature")
        oldp, oldc, oldslots = dict(state.pipes), dict(state.columns), dict(state.layer_slots)
        pipes, pipe_reports, transfers = {}, {}, {}
        for name, model in models.items():
            final, report = (
                model.step_coupled(oldp[name], dt)
                if normal_relaxation
                else model.step(oldp[name], dt)
            )
            axial = report["axial"] if normal_relaxation else report
            axial_state = PRActiveMultifluidState.from_dict(axial["state"])
            ports = model.port_transfers(oldp[name], axial_state)
            self.cache[name] = model._linear, model.source._normal_linear
            pipes[name], pipe_reports[name], transfers[name] = final, report, ports
        columns, slots, column_reports = {}, {}, {}
        for name, c in self.columns.items():
            nc = len(c.fluid.names)
            owned = self.column_ports[name]
            ns = [
                oldc[name].component_moles[j]
                - math.fsum(transfers[line][side]["transfer_n_m_e"][j] for line, side in owned)
                for j in range(nc)
            ]
            if min(ns) < 0:
                raise PropertyModelError("Aggregate graph transfers exceed the finite column stock")
            energy_ports = []
            for line, side in owned:
                tr = transfers[line][side]
                mass = math.fsum(
                    n * comp.molar_mass_kg_mol
                    for n, comp in zip(tr["transfer_n_m_e"][:nc], c.fluid.components)
                )
                energy_ports.append(
                    tr["transfer_n_m_e"][-1]
                    + tr["capillary_work_j"]
                    + mass * c.gravity_m_s2 * self.adapters[line, side].elevation
                )
            target = oldc[name].total_energy_j - math.fsum(energy_ports) + heat[name]
            energy = target
            line0, side0 = owned[0]
            adapter = self.adapters[line0, side0]
            previous = PRCapillaryColumnState(
                oldp[line0], oldc[name], state.next_step_s, adapter.model_id, oldslots[name]
            )
            for iteration in range(16):
                result = c.equilibrate(
                    ns,
                    energy,
                    pressure_guess_pa=math.exp(oldc[name].solution_variables[0]),
                    temperature_guess_k=oldc[name].temperature_k,
                )
                candidate = StratifiedPRState.from_dict(result["state"])
                binding = adapter._match_layers(previous, candidate)
                work = []
                for line, side in owned:
                    a = self.adapters[line, side]
                    phases, _ = a._port(candidate, binding)
                    src = models[line].source
                    ci = 0 if side == 0 else -1
                    height = (
                        phases[src.liquid].volume_m3 / (a.pipe.lengths[ci] * src.width)
                        if src.liquid in phases
                        else 0.0
                    )
                    body = dict(pipes[line].cells[ci])[src.liquid].volume_m3 / (
                        a.pipe.lengths[ci] * src.width
                    )
                    value = src.sigma * src.width / (2 * a.pipe.lengths[ci]) * (body - height) ** 2
                    work.append(value - self.boundary_energy(models[line], pipes[line], side))
                shape = math.fsum(work)
                required = target - shape
                if abs(energy - required) <= 8 * math.ulp(energy) + 1e-20:
                    break
                energy = required
            else:
                raise HydraulicConvergenceError("Facility column/geometry energy iteration failed")
            columns[name], slots[name] = candidate, binding
            column_reports[name] = {
                "equilibrium": result,
                "port_energy_to_pipes_j": energy_ports,
                "boundary_shape_work_j": work,
                "heat_j": heat[name],
                "reservoir_temperature_k": bath[name],
                "energy_iterations": iteration + 1,
            }
        end = PRCapillaryFacilityState(
            tuple(pipes.items()), tuple(columns.items()), tuple(slots.items()), dt, self.model_id
        )
        residual, tolerance = self.ledger(state, end, external_heat_j=math.fsum(heat.values()))
        if any(not math.isfinite(v) or abs(v) > t for v, t in zip(residual, tolerance)):
            raise ConservationError("Facility shared component/augmented energy balance failed")
        entropy_terms = []
        for name, row in self.lines.items():
            entropy_terms.extend(
                (
                    -row["pipe"].inspect(oldp[name])["entropy"],
                    row["pipe"].inspect(pipes[name])["entropy"],
                )
            )
        for name, owned in self.column_ports.items():
            a = self.adapters[owned[0]]
            entropy_terms.extend(
                (-a._entropy(oldc[name]), column_reports[name]["equilibrium"]["entropy_j_k"])
            )
            if heat[name]:
                entropy_terms.append(-heat[name] / bath[name])
        production = math.fsum(entropy_terms)
        if production < -256 * math.fsum(math.ulp(v) for v in entropy_terms) - 1e-10:
            raise ConservationError("Facility entropy decreased")
        return end, {
            "state": end.to_dict(),
            "dt_s": dt,
            "normal_relaxation": normal_relaxation,
            "pipe_models": {n: m.to_dict() for n, m in models.items()},
            "pipes": pipe_reports,
            "port_transfers": transfers,
            "columns": column_reports,
            "conservation_residuals": residual,
            "roundoff_allowances": tolerance,
            "entropy_production_j_k": production,
            "qualification": "Shared finite columns and horizontal capillary lines, first-order synchronous IMEX; fixed graph, positive axial phase support; not monolithic facility Newton",
        }

    def advance(self, state, duration_s, *, maximum_attempts=1000, **commands):
        self.inspect(state)
        duration = _number(duration_s, "facility reporting interval", positive=True)
        target = state.time_s + duration
        if not math.isfinite(target) or target <= state.time_s:
            raise ValueError("Facility report interval below clock resolution")
        if type(maximum_attempts) is not int or maximum_attempts < 1:
            raise ValueError("Positive facility attempt budget required")
        current, proposed, steps, rejected = state, state.next_step_s, [], []
        for _ in range(maximum_attempts):
            if current.time_s == target:
                return current, {"state": current.to_dict(), "steps": steps, "rejected": rejected}
            dt = interval_step_s(current.time_s, target, proposed)
            try:
                end, report = self.step(current, dt, **commands)
            except (HydraulicConvergenceError, PropertyModelError, ConservationError) as error:
                rejected.append({"time_s": current.time_s, "dt_s": dt, "reason": str(error)})
                proposed = dt / 2
                if current.time_s + proposed == current.time_s:
                    raise HydraulicConvergenceError(
                        "Facility interval reached clock resolution"
                    ) from error
                continue
            proposed = min(duration, 1.5 * dt)
            current = replace(end, next_step_s=proposed)
            report["state"] = current.to_dict()
            steps.append(report)
            if current.time_s == target:
                return current, {"state": current.to_dict(), "steps": steps, "rejected": rejected}
        raise HydraulicConvergenceError("Facility interval attempt budget exhausted")
