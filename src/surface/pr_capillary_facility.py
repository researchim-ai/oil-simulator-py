"""Conservative capillary lines sharing finite gravity columns.

Columns are actual graph vertices, lines are edges; parallel edges and
cycles do not clone inventories. Pipe solves use beginning-of-step column
profiles. Their accepted per-port transfers are aggregated before any
column is updated, including gravitational enthalpy and end-geometry work.
Optional nonlinear iteration closes flows against actual END profiles,
with a consistent change of the surface-work ordering. Both schemes are
first-order split operators, not a monolithic facility Newton matrix.
"""

import math
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, replace

from .equilibrium_cache import _signature
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


class _InspectionCache:
    def __init__(self, maximum_entries):
        self.maximum_entries = maximum_entries
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def statistics(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "entries": len(self.entries),
            "maximum_entries": self.maximum_entries,
        }


class PRCapillaryFacility:
    def __init__(
        self,
        lines,
        columns,
        *,
        coupling="imex",
        maximum_coupling_iterations=12,
        maximum_schur_unknowns=96,
    ):
        if coupling not in ("imex", "implicit", "newton"):
            raise ValueError("Facility coupling must be imex, implicit or newton")
        if type(maximum_schur_unknowns) is not int or maximum_schur_unknowns < 1:
            raise ValueError("Positive global Schur unknown budget required")
        self.maximum_schur_unknowns = maximum_schur_unknowns
        if type(maximum_coupling_iterations) is not int or maximum_coupling_iterations < 1:
            raise ValueError("Positive nonlinear coupling iteration budget required")
        self.coupling = coupling
        self.maximum_coupling_iterations = maximum_coupling_iterations
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
        result = {
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
        if self.coupling != "imex":
            result.update(
                coupling=self.coupling, maximum_coupling_iterations=self.maximum_coupling_iterations
            )
        if self.coupling == "newton":
            result["maximum_schur_unknowns"] = self.maximum_schur_unknowns
        return result

    @classmethod
    def from_dict(cls, raw, *, device="cpu"):
        if (
            set(raw) - {"coupling", "maximum_coupling_iterations", "maximum_schur_unknowns"}
            != {"model", "lines", "columns"}
            or raw["model"] != "pr_capillary_facility_imex_v1"
        ):
            raise ValueError("Explicit capillary facility model required")
        return cls(
            {
                n: {**r, "pipe": PRActiveMultifluidPipe.from_dict(r["pipe"], device=device)}
                for n, r in raw["lines"].items()
            },
            {n: StratifiedPRColumn.from_dict(c, device=device) for n, c in raw["columns"].items()},
            coupling=raw.get("coupling", "imex"),
            maximum_coupling_iterations=raw.get("maximum_coupling_iterations", 12),
            maximum_schur_unknowns=raw.get("maximum_schur_unknowns", 96),
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

    @contextmanager
    def cached_inspections(self, *, enabled=True, maximum_entries=64):
        """Exact bounded scratch for one accepted/rejected outer transaction.

        Physics/model mutations and differentiable EOS parameters invalidate
        reuse. No cache survives the transaction or enters a checkpoint.
        An explicit outer disabled scope also disables nested default scopes.
        """
        if type(enabled) is not bool or type(maximum_entries) is not int or maximum_entries < 1:
            raise ValueError("Explicit inspection-cache switch and positive budget required")
        marker = object()
        previous = getattr(self, "_active_inspection_cache", marker)
        if previous is not marker and enabled:
            yield previous
            return
        cache = _InspectionCache(maximum_entries) if enabled else None
        self._active_inspection_cache = cache
        try:
            yield cache
        finally:
            if previous is marker:
                del self._active_inspection_cache
            else:
                self._active_inspection_cache = previous
            if cache is not None:
                cache.entries.clear()

    def _inspection_key(self, state):
        fluids = [r["pipe"].fluid for r in self.lines.values()]
        fluids.extend(c.fluid for c in self.columns.values())
        return (state.pipes, state.columns, state.layer_slots, tuple(_signature(f) for f in fluids))

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
        cache = getattr(self, "_active_inspection_cache", None)
        key = None
        if cache is not None:
            try:
                key = self._inspection_key(state)
            except (TypeError, RuntimeError):
                pass  # Parameter sensitivities retain the complete checked path.
            if key is not None and key in cache.entries:
                models = cache.entries[key]
                if set(models) == set(self.lines) and all(
                    _digest(m.to_dict()) == m.model_id for m in models.values()
                ):
                    cache.hits += 1
                    cache.entries.move_to_end(key)
                    for name, model in models.items():
                        if name in self.cache:
                            model._linear, model.source._normal_linear = self.cache[name]
                    return dict(models)
                del cache.entries[key]
            cache.misses += 1
        models = self._models(state)
        for name, ps in state.pipes:
            models[name].pipe.inspect(ps)
            models[name].source.surface_energy(ps)
        if cache is not None and key is not None:
            cache.entries[key] = dict(models)
            if len(cache.entries) > cache.maximum_entries:
                cache.entries.popitem(last=False)
                cache.evictions += 1
        return models

    @staticmethod
    def boundary_energy(model, state, side):
        s = model.source
        hf = s.boundary_heights[side]
        if hf is None:
            return 0.0
        ci = 0 if side == 0 else -1
        phase = dict(state.cells[ci]).get(s.liquid)
        h = (phase.volume_m3 if phase is not None else 0.0) / (s.width * model.pipe.lengths[ci])
        return s.sigma * s.width / (2 * s.boundary_distances[side]) * (h - hf) ** 2

    def rebind_columns(self, state, columns):
        """Bind shared physical profiles through the existing phase registry.

        No material is moved here. A caller replacing shared columns owns
        their conservative update; this method only follows physical layer
        appearance/disappearance and validates the resulting pipe boundaries.
        """
        self.inspect(state)
        columns = dict(columns)
        if set(columns) != set(self.columns):
            raise ValueError("Exactly the existing facility columns are required")
        pipes, previous, slots = dict(state.pipes), dict(state.columns), dict(state.layer_slots)
        bindings = []
        for name in self.columns:
            line, side = self.column_ports[name][0]
            adapter = self.adapters[line, side]
            before = PRCapillaryColumnState(
                pipes[line], previous[name], state.next_step_s, adapter.model_id, slots[name]
            )
            bindings.append((name, adapter._match_layers(before, columns[name])))
        rebound = replace(state, columns=tuple(columns.items()), layer_slots=tuple(bindings))
        self.inspect(rebound)
        return rebound

    def conserved_terms(self, state, *, include_columns=True, reference_elevation_m=0.0):
        """Component and augmented-energy operands for shared graph accounting.

        Pipe energy includes kinetic, gravitational, wetting and interfacial
        contributions from the existing checked source model. Excluding columns
        lets a host graph count shared vessels exactly once. Operands are kept
        separate so ledgers retain cancellation-aware floating-point bounds.
        """
        models = self.inspect(state)
        if type(include_columns) is not bool:
            raise ValueError("Explicit column inclusion switch required")
        datum = _number(reference_elevation_m, "energy reference elevation")
        nc = len(next(iter(self.lines.values()))["pipe"].fluid.names)
        terms = [[] for _ in range(nc + 1)]
        for name, ps in state.pipes:
            row = self.lines[name]
            pipe = row["pipe"]
            attached = [r for r in row["ports"] if r is not None]
            gravity = self.columns[attached[0]["column"]].gravity_m_s2 if attached else 0.0
            terms[-1].append(models[name].source.surface_energy(ps))
            for cell in ps.cells:
                for _, phase in cell:
                    for j, amount in enumerate(phase.component_moles):
                        terms[j].append(amount)
                    mass = math.fsum(
                        n * c.molar_mass_kg_mol
                        for n, c in zip(phase.component_moles, pipe.fluid.components)
                    )
                    terms[-1].append(
                        phase.total_energy_j + mass * gravity * (pipe.elevations[0] - datum)
                    )
        if include_columns:
            for name, cs in state.columns:
                for j, amount in enumerate(cs.component_moles):
                    terms[j].append(amount)
                column = self.columns[name]
                mass = math.fsum(
                    n * c.molar_mass_kg_mol
                    for n, c in zip(cs.component_moles, column.fluid.components)
                )
                terms[-1].append(cs.total_energy_j - mass * column.gravity_m_s2 * datum)
        return terms

    def external_exchange(self, report, *, reference_elevation_m=0.0):
        """Accepted side-port N/E/S, using the same datum as conserved_terms."""
        datum = _number(reference_elevation_m, "energy reference elevation")
        nc = len(next(iter(self.lines.values()))["pipe"].fluid.names)
        components = [[] for _ in range(nc)]
        energies, entropies = [], []
        for name, row in self.lines.items():
            ax = (
                report["pipes"][name]["axial"]
                if report["normal_relaxation"]
                else report["pipes"][name]
            )
            side = ax.get("sideport_transfer_n_m_e", [0.0] * (nc + 2))
            for values, value in zip(components, side[:-2]):
                values.append(value)
            attached = [p for p in row["ports"] if p is not None]
            gravity = self.columns[attached[0]["column"]].gravity_m_s2 if attached else 0.0
            mass = math.fsum(
                n * c.molar_mass_kg_mol for n, c in zip(side[:-2], row["pipe"].fluid.components)
            )
            energies.append(
                side[-1]
                + ax.get("sideport_actuator_work_j", 0.0)
                + mass * gravity * (row["pipe"].elevations[0] - datum)
            )
            entropies.append(ax.get("sideport_incoming_entropy_j_k", 0.0))
        return [math.fsum(v) for v in components], math.fsum(energies), math.fsum(entropies)

    def ledger(
        self, before, after, *, external_heat_j=0.0, external_components=None, external_energy_j=0.0
    ):
        old, new = self.conserved_terms(before), self.conserved_terms(after)
        nc = len(old) - 1
        terms = [[*(-v for v in a), *b] for a, b in zip(old, new)]
        if external_components is not None:
            for j, v in enumerate(external_components):
                terms[j].append(-v)
        terms[-1].extend((-external_heat_j, -external_energy_j))
        return [math.fsum(t) for t in terms], [
            128 * math.fsum(math.ulp(v) for v in t) + (1e-18 if j == nc else 1e-20)
            for j, t in enumerate(terms)
        ]

    def step(self, state, dt_s, *, normal_relaxation=True, heaters=None, phase_controls=None):
        if self.coupling == "newton":
            from .pr_facility_schur import solve_schur

            return solve_schur(
                self,
                state,
                dt_s,
                normal_relaxation=normal_relaxation,
                heaters=heaters,
                phase_controls=phase_controls,
            )
        if self.coupling == "imex":
            return self._trial(
                state,
                dt_s,
                normal_relaxation=normal_relaxation,
                heaters=heaters,
                phase_controls=phase_controls,
            )
        boundary_state = state
        for iteration in range(self.maximum_coupling_iterations):
            end, report = self._trial(
                state,
                dt_s,
                normal_relaxation=normal_relaxation,
                heaters=heaters,
                phase_controls=phase_controls,
                boundary_state=boundary_state,
            )
            if report["coupling_converged"]:
                report["nonlinear_coupling_iterations"] = iteration + 1
                return end, report
            # Only exterior profiles are iterated. Every pipe trial starts
            # from the ORIGINAL pipe stocks and original report clock.
            boundary_state = replace(state, columns=end.columns, layer_slots=end.layer_slots)
        raise HydraulicConvergenceError("Implicit facility boundary coupling failed to converge")

    def _trial(
        self,
        state,
        dt_s,
        *,
        normal_relaxation=True,
        heaters=None,
        boundary_state=None,
        phase_controls=None,
    ):
        if type(normal_relaxation) is not bool:
            raise ValueError("Explicit normal relaxation switch required")
        initial_models = self.inspect(state)
        models = initial_models if boundary_state is None else self._models(boundary_state)
        dt = _number(dt_s, "facility interval", positive=True)
        dt = (state.time_s + dt) - state.time_s
        if dt <= 0 or not math.isfinite(dt):
            raise ValueError("Facility interval below clock resolution")
        heaters = {} if heaters is None else dict(heaters)
        if set(heaters) - set(self.columns):
            raise ValueError("Unknown facility heater column")
        phase_controls = {} if phase_controls is None else dict(phase_controls)
        if set(phase_controls) - set(self.lines):
            raise ValueError("Unknown facility phase-control line")
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
                model.step_coupled(oldp[name], dt, phase_controls=phase_controls.get(name))
                if normal_relaxation
                else model.step(oldp[name], dt, phase_controls=phase_controls.get(name))
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
                    shape_state = pipes[line] if boundary_state is None else oldp[line]
                    phase = dict(shape_state.cells[ci]).get(src.liquid)
                    body = (phase.volume_m3 if phase is not None else 0.0) / (
                        a.pipe.lengths[ci] * src.width
                    )
                    value = src.sigma * src.width / (2 * a.pipe.lengths[ci]) * (body - height) ** 2
                    work.append(
                        value - self.boundary_energy(initial_models[line], shape_state, side)
                    )
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
        converged = boundary_state is None or (
            end.columns == boundary_state.columns and end.layer_slots == boundary_state.layer_slots
        )
        if not converged:
            # This is a nonlinear trial, never a partially accepted time step.
            return end, {"coupling_converged": False}
        external_components, external_energy, _ = self.external_exchange(
            {"pipes": pipe_reports, "normal_relaxation": normal_relaxation}
        )
        residual, tolerance = self.ledger(
            state,
            end,
            external_heat_j=math.fsum(heat.values()),
            external_components=external_components,
            external_energy_j=external_energy,
        )
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
        # Retain individual operand ULPs in the entropy acceptance bound.
        entropy_terms.extend(
            -(body["axial"] if normal_relaxation else body).get(
                "sideport_incoming_entropy_j_k", 0.0
            )
            for body in pipe_reports.values()
        )
        production = math.fsum(entropy_terms)
        if production < -256 * math.fsum(math.ulp(v) for v in entropy_terms) - 1e-10:
            raise ConservationError("Facility entropy decreased")
        return end, {
            "state": end.to_dict(),
            "dt_s": dt,
            "normal_relaxation": normal_relaxation,
            "coupling": self.coupling,
            "coupling_converged": True,
            "nonlinear_coupling_iterations": 1,
            "pipe_models": {n: m.to_dict() for n, m in models.items()},
            "pipes": pipe_reports,
            "port_transfers": transfers,
            "phase_controls": {
                name: (
                    pipe_reports[name]["axial"] if normal_relaxation else pipe_reports[name]
                ).get("phase_controls", {})
                for name in phase_controls
            },
            "columns": column_reports,
            "conservation_residuals": residual,
            "roundoff_allowances": tolerance,
            "entropy_production_j_k": production,
            "qualification": "Shared finite columns and horizontal capillary lines; first-order split pipe operator with initial or converged end-profile coupling; fixed graph, positive or active axial support with explicit material/actuator side-port ledgers; global reduced Newton is reported separately",
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
