"""Conservative IMEX coupling of a capillary pipe to a finite PR gravity column.

The aperture has exact circular phase areas and phasewise midpoint samples
of the actual gravitational profile, with an explicit small transverse-head
guard. Pipe transport is implicit with the beginning-of-step column state;
the same accepted transfers update the finite column. End geometry work is
included in its conserved energy constraint. This is first-order partitioned
coupling, not a monolithic aperture Newton or resolved inlet CFD.
"""

import math
from dataclasses import dataclass, replace

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
from .pr_capillary_axial import PRCapillaryAxialPipe
from .pr_capillary_pressure import PRCapillaryPressure
from .pr_interphase_relaxation import PRRelaxationPhase
from .stratified_pr import StratifiedPRColumn, StratifiedPRState
from .three_phase_properties import disk_band_area
from .time_interval import interval_step_s


@dataclass(frozen=True)
class PRCapillaryColumnState:
    pipe: PRActiveMultifluidState
    column: StratifiedPRState
    next_step_s: float
    model_id: str
    layer_slots: tuple = ()

    def __post_init__(self):
        slots = tuple(self.layer_slots)
        if not slots and len(self.column.phase_roots) == 2:
            slots = tuple(n for n, _ in self.pipe.cells[0])
        if len(slots) != len(self.column.phase_roots) or len(set(slots)) != len(slots):
            raise ValueError("Explicit distinct layer_slots required for the column phase registry")
        object.__setattr__(self, "layer_slots", slots)

    @property
    def time_s(self):
        return self.pipe.time_s

    def to_dict(self):
        return {
            "pipe": self.pipe.to_dict(),
            "column": self.column.to_dict(),
            "next_step_s": self.next_step_s,
            "model_id": self.model_id,
            "layer_slots": self.layer_slots,
        }

    @classmethod
    def from_dict(cls, raw):
        if set(raw) - {"layer_slots"} != {"pipe", "column", "next_step_s", "model_id"}:
            raise ValueError("Explicit capillary/column checkpoint required")
        return cls(
            PRActiveMultifluidState.from_dict(raw["pipe"]),
            StratifiedPRState.from_dict(raw["column"]),
            raw["next_step_s"],
            raw["model_id"],
            tuple(raw.get("layer_slots", ())),
        )


class PRCapillaryColumnNetwork:
    def __init__(
        self,
        pipe,
        column,
        parameters,
        *,
        port_side,
        port_height_m,
        maximum_unknowns=4096,
        volume_diffusion="acoustic",
        linear_solver="auto",
        maximum_transverse_head_fraction=1e-3,
    ):
        if not isinstance(pipe, PRActiveMultifluidPipe) or not isinstance(
            column, StratifiedPRColumn
        ):
            raise TypeError("Actual PR pipe and gravitational column required")
        if pipe.fluid.model_id != column.fluid.model_id or pipe.fluid.device != column.fluid.device:
            raise ValueError("Pipe/column EOS, component registry and device must agree")
        if type(port_side) is not int or port_side not in (0, 1):
            raise ValueError("Port side must be zero or one")
        self.pipe, self.column = pipe, column
        self.parameters = dict(parameters)
        if any(
            self.parameters.get(key) is not None
            for key in ("boundary_heights_m", "boundary_distances_m")
        ):
            raise ValueError("The finite column owns the coupled face geometry")
        self.port_side = port_side
        self.height = _number(port_height_m, "column port height", positive=True)
        self.maximum_unknowns, self.volume_diffusion, self.linear_solver = (
            maximum_unknowns,
            volume_diffusion,
            linear_solver,
        )
        self.maximum_head = _number(
            maximum_transverse_head_fraction, "transverse head guard", positive=True
        )
        if self.maximum_head > 0.01:
            raise ValueError("One-dimensional aperture approximation requires transverse head <=1%")
        if not pipe.diameter / 2 <= self.height <= column.geometry.height_m - pipe.diameter / 2:
            raise ValueError("Finite aperture lies outside the column")
        self.elevation = column.elevation_m + self.height
        if pipe.elevations is None or any(abs(z - self.elevation) > 1e-10 for z in pipe.elevations):
            raise ValueError("Horizontal pipe and physical port elevations must agree")
        # Validate the underlying numerical model before any operation.
        self._base_source = PRCapillaryPressure(pipe, **parameters)
        self._axial_linear = self._normal_linear = None
        PRCapillaryAxialPipe(
            self._base_source,
            maximum_unknowns=maximum_unknowns,
            volume_diffusion=volume_diffusion,
            linear_solver=linear_solver,
        )
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_capillary_column_imex_v1",
            "pipe": self.pipe.to_dict(),
            "column": self.column.to_dict(),
            "parameters": self.parameters,
            "port_side": self.port_side,
            "port_height_m": self.height,
            "maximum_unknowns": self.maximum_unknowns,
            "volume_diffusion": self.volume_diffusion,
            "linear_solver": self.linear_solver,
            "maximum_transverse_head_fraction": self.maximum_head,
            "boundary_geometry": "reservoir_cell",
        }

    @classmethod
    def from_dict(cls, raw, *, device="cpu"):
        data = dict(raw)
        if data.pop("model", None) != "pr_capillary_column_imex_v1":
            raise ValueError("Unknown capillary/column coupling model")
        if data.pop("boundary_geometry", "reservoir_cell") != "reservoir_cell":
            raise ValueError("Finite column coupling requires reservoir_cell geometry")
        data["pipe"] = PRActiveMultifluidPipe.from_dict(data["pipe"], device=device)
        data["column"] = StratifiedPRColumn.from_dict(data["column"], device=device)
        return cls(**data)

    def _port(self, state, layer_slots=None):
        c, p, f = self.column, self.pipe, self.pipe.fluid
        c.inspect(state)
        if layer_slots is None:
            if len(state.phase_roots) != 2:
                raise ValueError("A single-phase column needs an explicit layer_slots binding")
            layer_slots = (self._base_source.liquid, self._base_source.gas)
        if len(layer_slots) != len(state.phase_roots) or any(n not in p.names for n in layer_slots):
            raise ValueError("Column layers must bind the declared pipe slots")
        bounds = (0.0, *state.interface_heights_m, c.geometry.height_m)
        radius = p.diameter / 2
        phases, samples = {}, []
        for i, column_root in enumerate(state.phase_roots):
            name = layer_slots[i]
            root = p.slots[p.names.index(name)]["root"]
            low = max(bounds[i], self.height - radius)
            high = min(bounds[i + 1], self.height + radius)
            if high <= low:
                samples.append(
                    {
                        "slot": name,
                        "root": root,
                        "column_root": column_root,
                        "area_fraction": 0.0,
                        "sample": None,
                        "transverse_head_fraction": 0.0,
                    }
                )
                continue
            band_low, band_high = (
                (self.height + radius - high, self.height + radius - low)
                if low >= self.height
                else (low - self.height + radius, high - self.height + radius)
            )
            area = min(
                p.area,
                float(disk_band_area(f.tensor(band_low), f.tensor(band_high), p.diameter).cpu()),
            )
            if area <= 0:
                samples.append(
                    {
                        "slot": name,
                        "root": root,
                        "column_root": column_root,
                        "area_fraction": 0.0,
                        "sample": None,
                        "transverse_head_fraction": 0.0,
                    }
                )
                continue
            q = c.sample(
                state,
                (low + high) / 2,
                side=("below" if i == 0 else "above") if len(layer_slots) == 2 else None,
            )
            # A unique cubic root may be tagged "liquid" by the column even
            # in its gas layer. Verify the actual molar volume against the
            # requested pipe root instead of inferring phase identity by tag.
            eos = f.homogeneous(q["pressure_pa"], q["temperature_k"], q["mole_fractions"], root)
            vm = float(eos["molar_volume_m3_mol"].cpu())
            if abs(vm - q["molar_volume_m3_mol"]) > 1e-10 * q["molar_volume_m3_mol"]:
                raise PropertyModelError("Column layer does not match the declared pipe phase root")
            head = q["density_kg_m3"] * c.gravity_m_s2 * p.diameter / q["pressure_pa"]
            if head > self.maximum_head:
                raise PropertyModelError(
                    "Capillary aperture exceeds its transverse-head approximation guard"
                )
            volume = area * p.lengths[0 if self.port_side == 0 else -1]
            amount = volume / q["molar_volume_m3_mol"]
            numbers = tuple(amount * x for x in q["mole_fractions"])
            # A phasewise midpoint state, at rest in the stationary vessel.
            energy = amount * q["molar_enthalpy_j_mol"] - q["pressure_pa"] * volume
            phases[name] = PRRelaxationPhase(
                numbers, volume, q["pressure_pa"], q["temperature_k"], 0.0, energy
            )
            samples.append(
                {
                    "root": root,
                    "slot": name,
                    "column_root": column_root,
                    "area_fraction": area / p.area,
                    "sample": q,
                    "transverse_head_fraction": head,
                }
            )
        if set(phases) != set(p.names) and self.volume_diffusion != "acoustic":
            raise PropertyModelError("A phase-free aperture requires acoustic volume transport")
        return phases, samples

    def _axial(self, column_state, layer_slots=None):
        phases, samples = self._port(column_state, layer_slots)
        boundaries = [None, None]
        boundaries[self.port_side] = phases
        model = PRCapillaryAxialPipe(
            self._base_source,
            boundaries=boundaries,
            maximum_unknowns=self.maximum_unknowns,
            volume_diffusion=self.volume_diffusion,
            linear_solver=self.linear_solver,
            boundary_geometry="reservoir_cell",
        )
        model._linear = self._axial_linear
        model.source._normal_linear = self._normal_linear
        return model, samples

    def initialize(self, pipe_state, column_state, *, next_step_s=1e-6, layer_slots=None):
        _number(next_step_s, "next coupled step", positive=True)
        state = PRCapillaryColumnState(
            pipe_state,
            column_state,
            next_step_s,
            self.model_id,
            () if layer_slots is None else tuple(layer_slots),
        )
        self.inspect(state)
        return state

    def inspect(self, state):
        if state.model_id != self.model_id or _digest(self.to_dict()) != self.model_id:
            raise ValueError("Capillary/column configuration or checkpoint differs")
        _number(state.next_step_s, "next coupled step", positive=True)
        if len(state.layer_slots) == 2 and state.layer_slots != (
            self._base_source.liquid,
            self._base_source.gas,
        ):
            raise ValueError("Two column layers use the declared lower-liquid/upper-gas binding")
        self.pipe.inspect(state.pipe)
        model, _ = self._axial(state.column, state.layer_slots)
        model.source.surface_energy(state.pipe)
        return model

    def _match_layers(self, before, new_column):
        """Continue physical slots, rather than interpreting a cubic-root label.

        Two layers retain the declared lower-liquid/upper-gas binding. When
        one remains, match actual composition and molar volume to the old
        layers, as in the existing active multifluid phase registry.
        """
        if len(new_column.phase_roots) == 2:
            return (self._base_source.liquid, self._base_source.gas)
        if len(new_column.phase_roots) != 1:
            raise PropertyModelError("Coupled column supports at most two layers")
        if len(before.layer_slots) == 1:
            return before.layer_slots
        c = self.column
        new = c.sample(new_column, c.geometry.height_m / 2)
        bounds = (0.0, *before.column.interface_heights_m, c.geometry.height_m)
        choices = []
        for i, slot in enumerate(before.layer_slots):
            old = c.sample(
                before.column, (bounds[i] + bounds[i + 1]) / 2, side="below" if i == 0 else "above"
            )
            cost = math.fsum(
                abs(a - b) for a, b in zip(old["mole_fractions"], new["mole_fractions"])
            )
            cost += abs(math.log(old["molar_volume_m3_mol"] / new["molar_volume_m3_mol"]))
            choices.append((cost, i, slot))
        return (min(choices)[2],)

    def _entropy(self, state):
        c, f = self.column, self.pipe.fluid
        seed = (
            [f.tensor(x) for x in state.phase_seeds],
            list(state.phase_roots),
            f.tensor(state.interface_heights_m),
            math.exp(state.solution_variables[0]),
            state.temperature_k,
            f.tensor(state.chemical_constants),
        )
        _, _, _, evaluate, _, _ = c._problem(
            f.tensor(state.component_moles), state.total_energy_j, seed, state.quadrature_order
        )
        with torch.no_grad():
            _, data = evaluate(f.tensor(state.solution_variables))
        return float(data["entropy_j_k"].cpu())

    def ledger(self, before, after, *, external_heat_j=0.0):
        a, b = self.pipe._totals(before.pipe), self.pipe._totals(after.pipe)
        old = self.inspect(before)
        new = self.inspect(after)
        nc = len(self.pipe.fluid.names)
        components = [
            math.fsum(
                (b[j], after.column.component_moles[j], -a[j], -before.column.component_moles[j])
            )
            for j in range(nc)
        ]
        mw = [v.molar_mass_kg_mol for v in self.pipe.fluid.components]
        mass_change = math.fsum((b[j] - a[j]) * mw[j] for j in range(nc))
        energy = math.fsum(
            (
                b[-2],
                after.column.total_energy_j,
                new.source.surface_energy(after.pipe),
                -a[-2],
                -before.column.total_energy_j,
                -old.source.surface_energy(before.pipe),
                self.column.gravity_m_s2 * self.elevation * mass_change,
            )
        )
        return components + [energy - external_heat_j]

    def step(
        self, state, dt_s, *, normal_relaxation=True, column_heat_w=0.0, heat_temperature_k=None
    ):
        if type(normal_relaxation) is not bool:
            raise ValueError("Normal-relaxation switch must be boolean")
        model = self.inspect(state)
        dt = _number(dt_s, "coupled interval", positive=True)
        heating = _number(column_heat_w, "column heater power")
        if heating and heat_temperature_k is None:
            raise ValueError("Nonzero heat requires its physical reservoir temperature")
        bath = (
            None
            if heat_temperature_k is None
            else _number(heat_temperature_k, "heat reservoir temperature", positive=True)
        )
        final, pipe_report = (
            model.step_coupled(state.pipe, dt) if normal_relaxation else model.step(state.pipe, dt)
        )
        self._axial_linear = model._linear
        self._normal_linear = model.source._normal_linear
        axial = pipe_report["axial"] if normal_relaxation else pipe_report
        heat = heating * axial["dt_s"]
        transfer = axial["external_transfer_n_m_e"]
        nc = len(self.pipe.fluid.names)
        numbers = [n - dn for n, dn in zip(state.column.component_moles, transfer[:nc])]
        if min(numbers) < 0:
            raise PropertyModelError("Pipe transfer exceeds the actual finite column inventory")
        mass = math.fsum(
            transfer[j] * self.pipe.fluid.components[j].molar_mass_kg_mol for j in range(nc)
        )
        port_energy = (
            transfer[-1]
            + axial["external_capillary_work_j"]
            + mass * self.column.gravity_m_s2 * self.elevation
        )
        target = state.column.total_energy_j - port_energy + heat
        old_geometry_energy = model.source.surface_energy(final)
        candidate_energy = target
        # Reconfiguration work belongs to the moving column/pipe boundary.
        # Solve its energy constraint, rather than discarding that work when
        # rebuilding the boundary after the column interface has moved.
        for iterations in range(16):
            column_report = self.column.equilibrate(
                numbers,
                candidate_energy,
                pressure_guess_pa=math.exp(state.column.solution_variables[0]),
                temperature_guess_k=state.column.temperature_k,
            )
            column = StratifiedPRState.from_dict(column_report["state"])
            layer_slots = self._match_layers(state, column)
            updated, samples = self._axial(column, layer_slots)
            shape_work = updated.source.surface_energy(final) - old_geometry_energy
            required = target - shape_work
            if abs(required - candidate_energy) <= 8 * math.ulp(candidate_energy) + 1e-20:
                break
            candidate_energy = required
        else:
            raise HydraulicConvergenceError(
                "Coupled column/surface geometry energy iteration failed"
            )
        end = PRCapillaryColumnState(final, column, dt, self.model_id, layer_slots)
        residual = self.ledger(state, end, external_heat_j=heat)
        energy_allowance = (
            128
            * math.fsum(
                math.ulp(v)
                for v in (
                    state.column.total_energy_j,
                    column.total_energy_j,
                    *(p.total_energy_j for s in (state.pipe, final) for c in s.cells for _, p in c),
                )
            )
            + 1e-18
        )
        component_allowance = [
            128
            * math.fsum(
                (
                    math.ulp(a),
                    math.ulp(b),
                    *(
                        math.ulp(p.component_moles[j])
                        for s in (state.pipe, final)
                        for c in s.cells
                        for _, p in c
                    ),
                )
            )
            + 1e-20
            for j, (a, b) in enumerate(zip(state.column.component_moles, numbers))
        ]
        if any(abs(v) > tol for v, tol in zip(residual, component_allowance + [energy_allowance])):
            raise ConservationError(
                "Finite column/pipe component or augmented energy balance failed"
            )
        entropy_before = self.pipe.inspect(state.pipe)["entropy"] + self._entropy(state.column)
        entropy_after = self.pipe.inspect(final)["entropy"] + column_report["entropy_j_k"]
        entropy = entropy_after - entropy_before - (heat / bath if bath is not None else 0.0)
        if entropy < -128 * (math.ulp(entropy_before) + math.ulp(entropy_after)) - 1e-11:
            raise ConservationError("Coupled column/pipe entropy decreased")
        return end, {
            "state": end.to_dict(),
            "dt_s": final.time_s - state.time_s,
            "pipe_model": model.to_dict(),
            "end_pipe_model": updated.to_dict(),
            "pipe": pipe_report,
            "column": column_report,
            "end_aperture": samples,
            "port_energy_from_column_j": port_energy,
            "boundary_shape_work_from_column_j": shape_work,
            "column_energy_iterations": iterations + 1,
            "column_heat_j": heat,
            "heat_reservoir_temperature_k": bath,
            "column_phase_count_before": len(state.column.phase_roots),
            "column_phase_count_after": len(column.phase_roots),
            "column_layer_slots_before": state.layer_slots,
            "column_layer_slots_after": layer_slots,
            "component_roundoff_allowances_mol": component_allowance,
            "conservation_residuals": residual,
            "energy_roundoff_allowance_j": energy_allowance,
            "entropy_production_j_k": entropy,
            "stationary_column_support_impulse_to_fluid_kg_m_s": transfer[-2]
            + axial["external_capillary_impulse_kg_m_s"],
            "qualification": "Conservative first-order IMEX; exact aperture areas, phasewise midpoint profile with head guard; active column/aperture phases, positive pipe phase support; no arbitrary monolithic network claim",
        }

    def advance(
        self,
        state,
        duration_s,
        *,
        normal_relaxation=True,
        maximum_attempts=256,
        column_heat_w=0.0,
        heat_temperature_k=None,
    ):
        if type(maximum_attempts) is not int or maximum_attempts <= 0:
            raise ValueError("Positive coupled interval attempt budget required")
        self.inspect(state)
        if type(maximum_attempts) is not int or maximum_attempts < 1:
            raise ValueError("Positive coupled interval attempt budget required")
        duration = _number(duration_s, "coupled report interval", positive=True)
        target = state.time_s + duration
        if not math.isfinite(target) or target <= state.time_s:
            raise ValueError("Coupled report interval is below clock resolution")
        current, proposed = state, state.next_step_s
        steps, rejected = [], []
        for _ in range(maximum_attempts):
            if current.time_s == target:
                return current, {"state": current.to_dict(), "steps": steps, "rejected": rejected}
            dt = interval_step_s(current.time_s, target, proposed)
            try:
                end, report = self.step(
                    current,
                    dt,
                    normal_relaxation=normal_relaxation,
                    column_heat_w=column_heat_w,
                    heat_temperature_k=heat_temperature_k,
                )
            except (HydraulicConvergenceError, PropertyModelError, ConservationError) as error:
                rejected.append({"time_s": current.time_s, "dt_s": dt, "reason": str(error)})
                proposed = dt / 2
                if current.time_s + proposed == current.time_s:
                    raise HydraulicConvergenceError(
                        "Coupled step reached clock resolution"
                    ) from error
                continue
            proposed = min(duration, 1.5 * dt)
            current = replace(end, next_step_s=proposed)
            report["state"] = current.to_dict()
            steps.append(report)
            if current.time_s == target:
                return current, {"state": current.to_dict(), "steps": steps, "rejected": rejected}
        raise HydraulicConvergenceError("Coupled column/pipe interval attempt budget exhausted")
