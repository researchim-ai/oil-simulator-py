"""Explicit active-phase PR multifluid transport and thermochemical sources.

Surveyed constant-area cells. Each present phase owns N/V/M/E and P/T/u.
Absent slots have exactly zero inventory and no EOS evaluation. Pairwise
interface closure is explicit in pr_active_multifluid_flux, with CFL, domain
and entropy rejection. This is not the previous fixed-reference BN scheme.
"""

import itertools
import math
from dataclasses import asdict, dataclass, replace

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _name, _number
from .operation import _digest
from .peng_robinson import R
from .pr_active_multifluid_flux import fluctuations
from .pr_active_pipe_friction import PRActivePipeFriction
from .pr_batched_homogeneous import homogeneous_batch
from .pr_homogeneous_uv import recover_homogeneous_phases
from .pr_interphase_relaxation import PRRelaxationPhase
from .pr_phase_equilibration import PRPhaseEquilibration
from .pr_stock_balances import balance_limits, entropy_allowance
from .pr_thermochemical_relaxation import PRThermochemicalRelaxation, phase_families


@dataclass(frozen=True)
class PRActiveMultifluidState:
    cells: tuple
    time_s: float
    model_id: str

    def __post_init__(self):
        object.__setattr__(
            self, "cells", tuple(tuple((name, p) for name, p in cell) for cell in self.cells)
        )
        _number(self.time_s, "active pipe clock", nonnegative=True)

    @property
    def bulk_stocks(self):
        return tuple(
            (
                *(
                    math.fsum(p.component_moles[j] for _, p in cell)
                    for j in range(len(cell[0][1].component_moles))
                ),
                math.fsum(p.momentum_kg_m_s for _, p in cell),
                math.fsum(p.total_energy_j for _, p in cell),
            )
            for cell in self.cells
        )

    def to_dict(self):
        from dataclasses import asdict

        return {
            "cells": [{name: asdict(p) for name, p in cell} for cell in self.cells],
            "time_s": self.time_s,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"cells", "time_s", "model_id"}:
            raise ValueError("Unknown active pipe checkpoint fields")
        return cls(
            tuple(
                tuple((name, PRRelaxationPhase.from_dict(p)) for name, p in cell.items())
                for cell in data["cells"]
            ),
            data["time_s"],
            data["model_id"],
        )


class PRActiveMultifluidPipe:
    def __init__(
        self,
        fluid,
        phase_slots,
        lengths_m,
        diameter_m,
        *,
        boundary_mode="walls",
        wave_speed_bound_m_s=4000.0,
        maximum_courant=0.2,
        thermochemical_relaxation_time_s=None,
        face_elevations_m=None,
        gravity_m_s2=0.0,
        hydrostatic_reference=None,
        wall_friction=None,
    ):
        self.fluid = fluid
        self.slots = tuple(dict(row) for row in phase_slots)
        if not 1 <= len(self.slots) <= 6:
            raise ValueError("One to six explicit phase slots required")
        for row in self.slots:
            if set(row) - {"name", "root", "component_marker", "minimum_marker_fraction"} or not {
                "name",
                "root",
            } <= set(row):
                raise ValueError("Unknown phase slot fields")
            _name(row["name"])
            if row["root"] not in ("liquid", "vapor"):
                raise ValueError("Explicit homogeneous slot root required")
            marker = row.get("component_marker")
            if marker is not None and (
                type(marker) is not int or not 0 <= marker < len(fluid.names)
            ):
                raise ValueError("Unknown phase composition marker")
            threshold = row.get("minimum_marker_fraction", 0.5)
            if not math.isfinite(threshold) or not 0 < threshold <= 1:
                raise ValueError("Positive bounded marker fraction required")
        self.names = tuple(row["name"] for row in self.slots)
        if len(set(self.names)) != len(self.names):
            raise ValueError("Unique phase slot names required")
        self.lengths = tuple(_number(v, "cell length", positive=True) for v in lengths_m)
        if not self.lengths:
            raise ValueError("Nonempty active pipe mesh required")
        self.diameter = _number(diameter_m, "pipe diameter", positive=True)
        self.area = math.pi * self.diameter**2 / 4
        self.volumes = tuple(self.area * v for v in self.lengths)
        if boundary_mode not in ("walls", "periodic"):
            raise ValueError("Walls or periodic ends required")
        self.boundary_mode = boundary_mode
        self.wave_speed = _number(wave_speed_bound_m_s, "wave envelope", positive=True)
        self.courant = _number(maximum_courant, "Courant limit", positive=True)
        if self.courant > 0.25:
            raise ValueError("Use a conservative Courant bound <=0.25")
        self.gravity = _number(gravity_m_s2, "gravity", nonnegative=True)
        self.elevations = (
            None
            if face_elevations_m is None
            else tuple(_number(z, "face elevation") for z in face_elevations_m)
        )
        if self.elevations is not None and (
            len(self.elevations) != len(self.lengths) + 1
            or any(
                abs(b - a) > l * (1 + 1e-12)
                for a, b, l in zip(self.elevations, self.elevations[1:], self.lengths)
            )
        ):
            raise ValueError("Face survey must match physical segment lengths")
        if self.gravity and self.elevations is None:
            raise ValueError("Gravity requires an explicit face survey")
        if (
            self.boundary_mode == "periodic"
            and self.elevations is not None
            and self.elevations[0] != self.elevations[-1]
        ):
            raise ValueError("Periodic survey must close in elevation")
        self.centers = (
            tuple((a + b) / 2 for a, b in zip(self.elevations, self.elevations[1:]))
            if self.elevations is not None
            else (0.0,) * len(self.lengths)
        )
        self.reference_specs = hydrostatic_reference
        self._reference_operator = None
        self.thermochemical_time = thermochemical_relaxation_time_s
        self.chemical = (
            None
            if self.thermochemical_time is None
            else PRThermochemicalRelaxation(
                fluid, self.thermochemical_time, maximum_active_phases=len(self.names)
            )
        )
        self.friction = (
            None if wall_friction is None else PRActivePipeFriction.from_dict(wall_friction)
        )
        if self.friction is not None and (
            set(self.friction.viscosity) != set(self.names)
            or self.friction.roughness / self.diameter > 0.05
        ):
            raise ValueError(
                "Wall friction requires every named slot viscosity and relative roughness <=.05"
            )
        if self.friction is not None and hasattr(self.friction, "validate_pipe"):
            self.friction.validate_pipe(self)
        self.equilibrium = PRPhaseEquilibration(fluid)
        self.model_id = _digest(self.to_dict())
        self._cache = None
        if hydrostatic_reference is not None:
            self._prepare_reference()

    def to_dict(self):
        result = {
            "model": "pr_active_multifluid_pipe_v2"
            if self.elevations is not None
            else "pr_active_multifluid_pipe_v1",
            "fluid": self.fluid.to_dict(),
            "phase_slots": self.slots,
            "lengths_m": self.lengths,
            "diameter_m": self.diameter,
            "boundary_mode": self.boundary_mode,
            "wave_speed_bound_m_s": self.wave_speed,
            "maximum_courant": self.courant,
            "thermochemical_relaxation_time_s": self.thermochemical_time,
        }
        if self.friction is not None:
            result["model"] = "pr_active_multifluid_pipe_v3"
            result["wall_friction"] = self.friction.to_dict()
        if self.elevations is not None:
            result.update(
                face_elevations_m=self.elevations,
                gravity_m_s2=self.gravity,
                hydrostatic_reference=self.reference_specs,
            )
        return result

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        from .equilibrium_properties import EquilibriumFluid

        raw = dict(data)
        if raw.pop("model", None) not in (
            "pr_active_multifluid_pipe_v1",
            "pr_active_multifluid_pipe_v2",
            "pr_active_multifluid_pipe_v3",
        ):
            raise ValueError("Unknown active multifluid model")
        raw["fluid"] = EquilibriumFluid.from_dict(raw["fluid"], device=device)
        return cls(**raw)

    @property
    def maximum_step_s(self):
        return self.courant * min(self.lengths) / self.wave_speed

    def initialize(self, cells, *, time_s=0.0):
        if len(cells) != len(self.volumes):
            raise ValueError("Active initial mesh differs")
        state = PRActiveMultifluidState(
            tuple(
                tuple(
                    (
                        name,
                        cell[name]
                        if isinstance(cell[name], PRRelaxationPhase)
                        else PRRelaxationPhase.from_dict(cell[name]),
                    )
                    for name in self.names
                    if name in cell
                )
                for cell in cells
            ),
            time_s,
            self.model_id,
        )
        if any(set(cell) - set(self.names) for cell in cells):
            raise ValueError("Unknown initial active phase")
        self.inspect(state)
        return state

    def inspect(self, state):
        if (
            not isinstance(state, PRActiveMultifluidState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
            or len(state.cells) != len(self.volumes)
        ):
            raise PropertyModelError("Active multifluid model/state differs")
        if self._cache is not None and self._cache[0] is state:
            return self._cache[1]
        nc = len(self.fluid.names)
        f = self.fluid
        flat = []
        for i, (cell, volume) in enumerate(zip(state.cells, self.volumes)):
            names = tuple(name for name, _ in cell)
            if (
                not names
                or len(set(names)) != len(names)
                or names != tuple(n for n in self.names if n in names)
            ):
                raise PropertyModelError("Canonical nonempty active registry required")
            if any(
                not isinstance(p, PRRelaxationPhase) or len(p.component_moles) != nc
                for _, p in cell
            ):
                raise PropertyModelError("Typed phase stocks with the declared components required")
            if abs(math.fsum(p.volume_m3 for _, p in cell) - volume) > 64 * math.ulp(volume):
                raise ConservationError("Active phase volumes do not fill the cell")
            for name, phase in cell:
                j = self.names.index(name)
                amount = math.fsum(phase.component_moles)
                mass = math.fsum(
                    n * c.molar_mass_kg_mol for n, c in zip(phase.component_moles, f.components)
                )
                velocity = phase.momentum_kg_m_s / mass
                x = [v / amount for v in phase.component_moles]
                marker = self.slots[j].get("component_marker")
                if marker is not None and x[marker] < self.slots[j].get(
                    "minimum_marker_fraction", 0.5
                ):
                    raise PropertyModelError("Phase composition outside its marker domain")
                flat.append((i, j, phase, amount, mass, velocity, x))
        eos = homogeneous_batch(
            f,
            [r[2].pressure_pa for r in flat],
            [r[2].temperature_k for r in flat],
            [r[6] for r in flat],
            [self.slots[r[1]]["root"] for r in flat],
        )
        values = (
            torch.stack(
                [
                    eos[k]
                    for k in (
                        "molar_volume_m3_mol",
                        "molar_internal_energy_j_mol",
                        "molar_entropy_j_mol_k",
                        "frozen_sound_speed_m_s",
                        "homogeneous_molar_cv_j_mol_k",
                    )
                ],
                -1,
            )
            .detach()
            .cpu()
            .tolist()
        )
        slots = len(self.names)
        rows = [[[0.0] * (nc + 2) for _ in self.names] for _ in state.cells]
        alpha = [[0.0] * slots for _ in state.cells]
        pressures = [[0.0] * slots for _ in state.cells]
        temperatures = [[0.0] * slots for _ in state.cells]
        velocities = [[0.0] * slots for _ in state.cells]
        sounds = [[0.0] * slots for _ in state.cells]
        precisions = [[0.0] * slots for _ in state.cells]
        entropies = [[] for _ in state.cells]
        entropy_fluxes = [[] for _ in state.cells]
        entropy_terms = []
        phase_entropy_density = [[0.0] * slots for _ in state.cells]
        for (i, j, phase, amount, mass, velocity, x), (vm, internal, entropy, acoustic, cv) in zip(
            flat, values
        ):
            volume = self.volumes[i]
            expected_volume = amount * vm
            expected_energy = amount * internal + 0.5 * phase.momentum_kg_m_s * velocity
            if abs(expected_volume - phase.volume_m3) > 1e-12 + 1e-9 * phase.volume_m3 or abs(
                expected_energy - phase.total_energy_j
            ) > 1e-6 + 1e-9 * abs(expected_energy):
                raise ConservationError(
                    "Active phase checkpoint disagrees with its homogeneous EOS"
                )
            phase_entropy = amount * entropy
            phase_entropy_density[i][j] = phase_entropy / volume
            entropy_terms.append(phase_entropy)
            entropies[i].append(phase_entropy / volume)
            entropy_fluxes[i].append(phase_entropy * velocity / volume)
            sound = abs(velocity) + acoustic
            if sound > self.wave_speed * (1 + 1e-12):
                raise PropertyModelError("Active phase exceeds acoustic envelope")
            ideal_cp = math.fsum(v * c.ideal_cp_j_mol_k for v, c in zip(x, f.components))
            bulk_modulus = mass / phase.volume_m3 * acoustic**2
            energy_scale = abs(phase.total_energy_j) / amount + (ideal_cp + R) * (
                phase.temperature_k + f.tref
            )
            precisions[i][j] = (
                64
                * torch.finfo(torch.float64).eps
                * (abs(phase.pressure_pa) + bulk_modulus + cv * phase.temperature_k / vm)
                * (1 + energy_scale / (cv * phase.temperature_k))
            )
            rows[i][j] = [
                *(v / volume for v in phase.component_moles),
                phase.momentum_kg_m_s / volume,
                phase.total_energy_j / volume,
            ]
            alpha[i][j] = phase.volume_m3 / volume
            pressures[i][j] = phase.pressure_pa
            temperatures[i][j] = phase.temperature_k
            velocities[i][j] = velocity
            sounds[i][j] = sound
        entropy_density = [math.fsum(v) for v in entropies]
        entropy_flux = [math.fsum(v) for v in entropy_fluxes]
        conserved = f.tensor(rows)
        a = f.tensor(alpha)
        p = f.tensor(pressures)
        u = f.tensor(velocities)
        flux = torch.cat(
            (
                conserved[..., :nc] * u[..., None],
                (conserved[..., -2] * u + a * p)[..., None],
                ((conserved[..., -1] + a * p) * u)[..., None],
            ),
            -1,
        )
        data = {
            "conserved": conserved,
            "alpha": a,
            "pressure": p,
            "temperature": f.tensor(temperatures),
            "velocity": u,
            "sound": f.tensor(sounds),
            "pressure_precision": f.tensor(precisions),
            "flux": flux,
            "entropy": math.fsum(entropy_terms),
            "entropy_density": f.tensor(entropy_density),
            "phase_entropy_density": f.tensor(phase_entropy_density),
            "entropy_flux": f.tensor(entropy_flux),
        }
        self._cache = (state, data)
        return data

    def _totals(self, state):
        nc = len(self.fluid.names)
        return [
            *(
                math.fsum(p.component_moles[j] for cell in state.cells for _, p in cell)
                for j in range(nc)
            ),
            math.fsum(
                p.total_energy_j
                + self.gravity
                * z
                * math.fsum(
                    n * c.molar_mass_kg_mol
                    for n, c in zip(p.component_moles, self.fluid.components)
                )
                for cell, z in zip(state.cells, self.centers)
                for _, p in cell
            ),
            math.fsum(p.momentum_kg_m_s for cell in state.cells for _, p in cell),
        ]

    def ledger(self, old, new, wall_impulse=0.0, external_heat_j=0.0, external_transfer=None):
        r = [b - a for a, b in zip(self._totals(old), self._totals(new))]
        if external_transfer is not None:
            r = [v - t for v, t in zip(r, external_transfer)]
        r[-2] -= external_heat_j
        r[-1] -= wall_impulse
        limits = balance_limits(
            [p for c in old.cells for _, p in c],
            [p for c in new.cells for _, p in c],
            len(self.fluid.names),
            include_volume=False,
        )
        if any(abs(v) > limit for v, limit in zip(r, limits)):
            raise ConservationError("Active pipe N/E/M ledger failed")
        return r

    def _assign_source(self, before, after, report):
        old_names = [n for n, _ in before]
        families = phase_families(self.fluid, after)
        possibilities = []
        for i, (phase, family) in enumerate(zip(after.phases, families)):
            row = []
            amount = math.fsum(phase.component_moles)
            x = [v / amount for v in phase.component_moles]
            for j, slot in enumerate(self.slots):
                if family != "stable" and slot["root"] != family:
                    continue
                marker = slot.get("component_marker")
                if marker is not None and x[marker] < slot.get("minimum_marker_fraction", 0.5):
                    continue
                q = self.fluid.homogeneous(phase.pressure_pa, phase.temperature_k, x, slot["root"])
                if (
                    abs(float(q["molar_volume_m3_mol"].cpu()) * amount - phase.volume_m3)
                    > 1e-12 + 1e-9 * phase.volume_m3
                ):
                    continue
                old_index = report["phase_pairs"][i][0]
                if old_index is not None:
                    cost = 0.0 if old_names[old_index] == slot["name"] else 1.0
                elif slot["name"] in dict(before):
                    previous = dict(before)[slot["name"]]
                    previous_amount = math.fsum(previous.component_moles)
                    cost = (
                        1.0
                        + math.fsum(
                            abs(a / previous_amount - b)
                            for a, b in zip(previous.component_moles, x)
                        )
                        + abs(
                            math.log(
                                (previous.volume_m3 / previous_amount) / (phase.volume_m3 / amount)
                            )
                        )
                    )
                else:
                    cost = 10.0
                row.append((cost, j))
            possibilities.append(row)
        best = None
        for choices in itertools.product(*possibilities):
            indices = [j for _, j in choices]
            if len(set(indices)) != len(indices):
                continue
            key = (sum(c for c, _ in choices), indices)
            if best is None or key < best[0]:
                best = (key, indices)
        if best is None:
            raise ValueError("Chemical phases cannot fit the declared active slot/marker registry")
        assigned = {self.names[j]: p for j, p in zip(best[1], after.phases)}
        return tuple((name, assigned[name]) for name in self.names if name in assigned)

    def _boundary_packet(self, cell):
        """Validated physical phase densities; ghost volumes are never ledger stocks."""
        if not isinstance(cell, dict) or not cell or set(cell) - set(self.names):
            raise ValueError("Boundary requires a nonempty named phase mapping")
        phases = {
            n: p if isinstance(p, PRRelaxationPhase) else PRRelaxationPhase.from_dict(p)
            for n, p in cell.items()
        }
        volume = math.fsum(p.volume_m3 for p in phases.values())
        if not math.isfinite(volume) or volume <= 0:
            raise ValueError("Boundary phase volume must be positive")
        ghost = type(self)(
            self.fluid,
            self.slots,
            [self.lengths[0]],
            self.diameter,
            wave_speed_bound_m_s=self.wave_speed,
            maximum_courant=self.courant,
        )
        scale = ghost.volumes[0] / volume
        scaled = {
            n: replace(
                p,
                component_moles=tuple(v * scale for v in p.component_moles),
                volume_m3=p.volume_m3 * scale,
                momentum_kg_m_s=p.momentum_kg_m_s * scale,
                total_energy_j=p.total_energy_j * scale,
            )
            for n, p in phases.items()
        }
        return ghost.inspect(ghost.initialize([scaled])), phases

    def _operator(self, old, ghosts, fields):
        if self.boundary_mode == "periodic":
            left = {key: old[key] for key in fields}
            right = {key: torch.roll(old[key], -1, 0) for key in fields}
            minus, plus, aminus, aplus, faces = fluctuations(left, right, self.wave_speed)
            delta = minus + torch.roll(plus, 1, 0)
            adelta = aminus + torch.roll(aplus, 1, 0)
            wall = 0.0
        else:
            left = {}
            right = {}
            for key in fields:
                gl = old[key][0:1].clone()
                gr = old[key][-1:].clone()
                if key == "velocity":
                    gl = -gl
                    gr = -gr
                if key == "conserved":
                    gl[..., -2] *= -1
                    gr[..., -2] *= -1
                if key == "flux":
                    gl[..., :-2] *= -1
                    gr[..., :-2] *= -1
                    gl[..., -1] *= -1
                    gr[..., -1] *= -1
                if ghosts[0] is not None:
                    gl = ghosts[0][0][key]
                if ghosts[1] is not None:
                    gr = ghosts[1][0][key]
                left[key] = torch.cat((gl, old[key]), 0)
                right[key] = torch.cat((old[key], gr), 0)
            minus, plus, aminus, aplus, faces = fluctuations(left, right, self.wave_speed)
            delta = minus[1:] + plus[:-1]
            adelta = aminus[1:] + aplus[:-1]
            wall = float(
                (self.area * (faces["flux"][0, :, -2].sum() - faces["flux"][-1, :, -2].sum())).cpu()
            )
        return delta, adelta, faces, wall

    def _gravity_operator(self, old, delta, faces):
        if not self.gravity:
            return delta, 0.0
        nc = len(self.fluid.names)
        mass = (old["conserved"][..., :nc] * self.fluid.molar_masses).sum(-1)
        slope = self.fluid.tensor(
            [(b - a) / l for a, b, l in zip(self.elevations, self.elevations[1:], self.lengths)]
        )
        force_density = -self.gravity * mass * slope[:, None]
        out = delta.clone()
        out[..., -2] -= force_density * self.fluid.tensor(self.lengths)[:, None]
        mass_flux = (faces["flux"][..., :nc] * self.fluid.molar_masses).sum(-1)
        if self.boundary_mode == "periodic":
            fr = mass_flux
            fl = torch.roll(fr, 1, 0)
        else:
            fl, fr = mass_flux[:-1], mass_flux[1:]
        zl = self.fluid.tensor([a - z for a, z in zip(self.elevations, self.centers)])
        zr = self.fluid.tensor([b - z for b, z in zip(self.elevations[1:], self.centers)])
        out[..., -1] += self.gravity * (zr[:, None] * fr - zl[:, None] * fl)
        load = float((force_density * self.fluid.tensor(self.volumes)[:, None]).sum().cpu())
        return out, load

    def _prepare_reference(self):
        if self.elevations is None or self.boundary_mode != "walls" or self.chemical is not None:
            raise ValueError("Hydrostatic reference needs surveyed walls without chemical source")
        ref = self.initialize(self.reference_specs)
        packet = self.inspect(ref)
        first = dict(ref.cells[0])
        for cell, z, volume in zip(ref.cells, self.centers, self.volumes):
            if tuple(n for n, _ in cell) != tuple(first):
                raise ValueError("Reference requires the same active branches in each cell")
            for name, p in cell:
                a = first[name]
                amount = math.fsum(p.component_moles)
                x = [v / amount for v in p.component_moles]
                ax = [v / math.fsum(a.component_moles) for v in a.component_moles]
                if (
                    p.momentum_kg_m_s != 0
                    or abs(p.temperature_k - a.temperature_k) > 1e-8
                    or max(abs(v - w) for v, w in zip(x, ax)) > 1e-10
                    or abs(p.volume_m3 / volume - a.volume_m3 / self.volumes[0]) > 1e-10
                ):
                    raise ValueError(
                        "Reference must be isothermal constant-composition/fraction hydrostatics"
                    )
                root = self.slots[self.names.index(name)]["root"]

                def gibbs(phase, x=x, root=root):
                    q = self.fluid.homogeneous(phase.pressure_pa, phase.temperature_k, x, root)
                    return float(
                        (
                            q["molar_enthalpy_j_mol"]
                            - phase.temperature_k * q["molar_entropy_j_mol_k"]
                        ).cpu()
                    ) / math.fsum(v * c.molar_mass_kg_mol for v, c in zip(x, self.fluid.components))

                if (
                    abs(gibbs(p) + self.gravity * z - gibbs(a) - self.gravity * self.centers[0])
                    > 1e-5
                ):
                    raise ValueError("Reference violates PR hydrostatic Gibbs balance")
        fields = (
            "conserved",
            "alpha",
            "pressure",
            "temperature",
            "velocity",
            "pressure_precision",
            "flux",
        )
        d, ad, faces, wall = self._operator(packet, [None, None], fields)
        d, load = self._gravity_operator(packet, d, faces)
        self._reference_operator = (d, ad, faces, wall + load)

    def step(
        self,
        state,
        dt_s,
        *,
        boundary_cells=None,
        boundary_samples=None,
        heat_rates_w=None,
        heat_temperatures_k=None,
    ):
        old = self.inspect(state)
        dt = _number(dt_s, "active pipe step", positive=True)
        end = state.time_s + dt
        dt = end - state.time_s
        if not math.isfinite(end) or dt <= 0:
            raise ValueError("Active pipe step below clock resolution")
        if dt > self.maximum_step_s * (1 + 1e-12):
            raise ValueError("Active pipe CFL exceeded")
        fields = (
            "conserved",
            "alpha",
            "pressure",
            "temperature",
            "velocity",
            "pressure_precision",
            "flux",
        )
        boundaries = (None, None) if boundary_cells is None else tuple(boundary_cells)
        if len(boundaries) != 2 or (
            self.boundary_mode == "periodic" and any(b is not None for b in boundaries)
        ):
            raise ValueError("Two endpoint phase mappings require wall-mode geometry")
        ghosts = [self._boundary_packet(b) if b is not None else None for b in boundaries]
        sample_rows = [[], []]
        if boundary_samples is not None:
            if self.boundary_mode == "periodic" or len(boundary_samples) != 2:
                raise ValueError("Two sampled endpoints require nonperiodic geometry")
            for side, rows in enumerate(boundary_samples):
                if rows is None:
                    continue
                if boundaries[side] is not None or not rows:
                    raise ValueError(
                        "Use either a physical cell or aperture samples at an endpoint"
                    )
                for row in rows:
                    if set(row) - {"kind"} != {"cell", "weight", "elevation_m"} or row.get(
                        "kind", "physical"
                    ) not in ("physical", "wall"):
                        raise ValueError("Aperture sample requires cell, weight and elevation")
                    weight = _number(row["weight"], "area weight", positive=True)
                    z = _number(row["elevation_m"], "sample elevation")
                    sample_rows[side].append(
                        (
                            *self._boundary_packet(row["cell"]),
                            weight,
                            z,
                            row.get("kind", "physical"),
                        )
                    )
                if abs(math.fsum(r[2] for r in sample_rows[side]) - 1) > 1e-12:
                    raise ValueError("Aperture weights must cover exactly one pipe face")
                ghosts[side] = sample_rows[side][0][:2]

        external = [0.0] * (len(self.fluid.names) + 2)
        entropy_in = 0.0
        boundary_entropy_fluxes = [0.0, 0.0]
        delta, adelta, faces, wall_rate = self._operator(old, ghosts, fields)
        sample_records = [[], []]
        sample_entropy = [None, None]
        for side, rows in enumerate(sample_rows):
            if not rows:
                continue
            ci = 0 if side == 0 else -1
            cell = {k: old[k][ci : ci + 1] if ci == 0 else old[k][-1:] for k in fields}
            original_l = {k: ghosts[side][0][k] if side == 0 else cell[k] for k in fields}
            original_r = {k: cell[k] if side == 0 else ghosts[side][0][k] for k in fields}
            dm, dp, am, ap, _ = fluctuations(original_l, original_r, self.wave_speed)
            original_delta = dp[0] if side == 0 else dm[0]
            original_adelta = ap[0] if side == 0 else am[0]
            aggregate = torch.zeros_like(original_delta)
            aaggregate = torch.zeros_like(original_adelta)
            faggregate = torch.zeros_like(faces["flux"][ci])
            path = torch.zeros_like(faces["phase_path"][ci])
            apath = torch.zeros_like(faces["alpha_path"][ci])
            entropy = 0.0
            contacts = True
            face_z = self.elevations[ci] if self.elevations is not None else 0.0
            for packet, phases, weight, z, kind in rows:
                l, r = (packet, cell) if side == 0 else (cell, packet)
                dm, dp, am, ap, ff = fluctuations(l, r, self.wave_speed)
                flux = ff["flux"][0].clone()
                correction = (
                    self.gravity
                    * (z - face_z)
                    * (flux[..., : len(self.fluid.names)] * self.fluid.molar_masses).sum(-1)
                )
                flux[..., -1] += correction
                update = (dp[0] if side == 0 else dm[0]).clone()
                update[..., -1] += (-1 if side == 0 else 1) * correction
                aggregate += weight * update
                aaggregate += weight * (ap[0] if side == 0 else am[0])
                faggregate += weight * flux
                path += weight * ff["phase_path"][0]
                apath += weight * ff["alpha_path"][0]
                contacts = contacts and bool(ff["contact"][0].cpu())
                sg = float(packet["entropy_density"][0].cpu())
                fg = float(packet["entropy_flux"][0].cpu())
                sc = float(old["entropy_density"][ci].cpu())
                fc = float(old["entropy_flux"][ci].cpu())
                sl, sr = (sg, sc) if side == 0 else (sc, sg)
                sf = 0.5 * (fg + fc) - 0.5 * self.wave_speed * (sr - sl)
                entropy += weight * sf
                sample_records[side].append(
                    {
                        "cell": {n: asdict(p) for n, p in phases.items()},
                        "kind": kind,
                        "weight": weight,
                        "elevation_m": z,
                        "flux": flux.cpu().tolist(),
                        "entropy_flux_w_m2_k": sf,
                    }
                )
            delta[ci] += aggregate - original_delta
            adelta[ci] += aaggregate - original_adelta
            wall_rate += (
                self.area
                * (1 if side == 0 else -1)
                * float((faggregate[..., -2] - faces["flux"][ci, ..., -2]).sum().cpu())
            )
            faces["flux"][ci] = faggregate
            faces["phase_path"][ci] = path
            faces["alpha_path"][ci] = apath
            faces["contact"][ci] = contacts
            sample_entropy[side] = entropy

        delta, gravity_rate = self._gravity_operator(old, delta, faces)
        gravity_impulse = dt * gravity_rate
        wall = dt * wall_rate
        if self._reference_operator is not None:
            if any(g is not None for g in ghosts):
                raise ValueError("Hydrostatic reference currently requires reflecting endpoints")
            rd, ra, rf, ri = self._reference_operator
            delta, adelta = delta - rd, adelta - ra
            faces = {**faces, "flux": faces["flux"] - rf["flux"]}
            wall -= dt * ri
        if self.boundary_mode != "periodic":
            for side, ghost in enumerate(ghosts):
                if ghost is None:
                    continue
                sign = 1.0 if side == 0 else -1.0
                index = 0 if side == 0 else -1
                flow = (sign * dt * self.area * faces["flux"][index].sum(0)).cpu().tolist()
                if sample_records[side]:
                    flow[-2] = (
                        sign
                        * dt
                        * self.area
                        * math.fsum(
                            row["weight"] * math.fsum(p[-2] for p in row["flux"])
                            for row in sample_records[side]
                            if row["kind"] != "wall"
                        )
                    )

                potential = (
                    self.gravity * self.elevations[index] if self.elevations is not None else 0.0
                )
                energy = flow[-1] + potential * math.fsum(
                    n * c.molar_mass_kg_mol for n, c in zip(flow[:-2], self.fluid.components)
                )
                transfer = flow[:-2] + [energy, flow[-2]]
                external = [a + b for a, b in zip(external, transfer)]
                wall -= transfer[-1]
                packet = ghost[0]
                sg = float(packet["entropy_density"][0].cpu())
                fg = float(packet["entropy_flux"][0].cpu())
                sc = float(old["entropy_density"][index].cpu())
                fc = float(old["entropy_flux"][index].cpu())
                sl, sr = (sg, sc) if side == 0 else (sc, sg)
                boundary_entropy_fluxes[side] = 0.5 * (fg + fc) - 0.5 * self.wave_speed * (sr - sl)
                if sample_entropy[side] is not None:
                    boundary_entropy_fluxes[side] = sample_entropy[side]
                entropy_in += sign * dt * self.area * boundary_entropy_fluxes[side]
        factor = self.fluid.tensor([dt / l for l in self.lengths])
        qnew = old["conserved"] - factor[:, None, None] * delta
        anew = old["alpha"] - factor[:, None] * adelta
        if not bool(
            (
                torch.isfinite(qnew).all()
                & torch.isfinite(anew).all()
                & (qnew[..., :-2] >= 0).all()
                & (anew >= 0).all()
            ).cpu()
        ):
            raise PropertyModelError("Active phase transport lost positivity")
        unchanged = ((delta == 0).all(dim=(-2, -1)) & (adelta == 0).all(dim=-1)).cpu().tolist()
        qq = qnew.cpu().tolist()
        aa = anew.cpu().tolist()
        rows = []
        locations = []
        temperatures = []
        for i, (values, alphas, volume) in enumerate(zip(qq, aa, self.volumes)):
            if unchanged[i]:
                continue
            owner = max(range(len(alphas)), key=alphas.__getitem__)
            alphas[owner] = 1.0 - math.fsum(a for j, a in enumerate(alphas) if j != owner)
            if alphas[owner] < 0:
                raise ConservationError("Active phase volume sum failed")
            for j, (row, alpha) in enumerate(zip(values, alphas)):
                n = [v * volume for v in row[:-2]]
                v = alpha * volume
                m = row[-2] * volume
                e = row[-1] * volume
                if not any(n) and v == 0.0 and m == 0.0 and e == 0.0:
                    continue
                if not math.fsum(n) > 0 or v <= 0:
                    raise PropertyModelError("An active transport phase has no mass or volume")
                previous = dict(state.cells[i]).get(self.names[j])
                if previous is not None:
                    guess = previous.temperature_k
                else:
                    donors = [
                        dict(cell)[self.names[j]]
                        for cell in state.cells
                        if self.names[j] in dict(cell)
                    ]
                    donors.extend(
                        g[1][self.names[j]]
                        for g in [*ghosts, *(r for side_rows in sample_rows for r in side_rows)]
                        if g is not None and self.names[j] in g[1]
                    )
                    if not donors:
                        raise PropertyModelError("Transport created inventory without a donor")
                    guess = math.fsum(p.temperature_k for p in donors) / len(donors)
                rows.append((n, v, m, e))
                locations.append((i, j))
                temperatures.append(guess)
        recovered = (
            recover_homogeneous_phases(
                self.fluid,
                [r[0] for r in rows],
                [r[1] for r in rows],
                [r[2] for r in rows],
                [r[3] for r in rows],
                [self.slots[j]["root"] for _, j in locations],
                temperatures,
            )
            if rows
            else ()
        )
        cells = [list(state.cells[i]) if unchanged[i] else [] for i in range(len(self.volumes))]
        for (i, j), phase in zip(locations, recovered):
            cells[i].append((self.names[j], phase))
        transport = PRActiveMultifluidState(tuple(tuple(c) for c in cells), end, self.model_id)
        transported = self.inspect(transport)
        if transported["entropy"] - old["entropy"] - entropy_in < -entropy_allowance(
            transported["entropy"], old["entropy"]
        ):
            raise ConservationError("Active transport entropy decreased before chemical relaxation")
        thermal = None
        heated = transport
        heat = heat_entropy = 0.0
        if heat_rates_w is not None or heat_temperatures_k is not None:
            from .pr_active_pipe_heat import apply_heat

            heated, thermal = apply_heat(self, transport, dt, heat_rates_w, heat_temperatures_k)
            heat = thermal["external_heat_j"]
            heat_entropy = thermal["incoming_entropy_j_k"]
        source_reports = []
        if self.chemical is not None:
            output = []
            for cell in heated.cells:
                s = self.chemical.initialize(
                    [p for _, p in cell],
                    [self.slots[self.names.index(n)]["root"] for n, _ in cell],
                    time_s=state.time_s,
                )
                after, report = self.chemical.step(s, dt)
                output.append(self._assign_source(cell, after, report))
                source_reports.append(report)
            new = PRActiveMultifluidState(tuple(output), end, self.model_id)
        else:
            new = heated
        chemical_state = new
        friction_report = None
        if self.friction is not None:
            new, friction_report = self.friction.apply(self, new, dt)
            wall += friction_report["wall_impulse_kg_m_s"]
        current = self.inspect(new)
        production = current["entropy"] - old["entropy"] - entropy_in - heat_entropy
        if production < -entropy_allowance(current["entropy"], old["entropy"]):
            raise ConservationError("Active pipe total entropy decreased")
        return new, {
            "state": new.to_dict(),
            "transport_state": transport.to_dict(),
            **(
                {"thermal": thermal, "heated_state": heated.to_dict(), "external_heat_j": heat}
                if thermal is not None
                else {}
            ),
            "chemical_state": chemical_state.to_dict(),
            "friction": friction_report,
            "dt_s": dt,
            "path_quadrature_order": 8,
            "face_fluxes": faces["flux"].cpu().tolist(),
            "phase_path": faces["phase_path"].cpu().tolist(),
            "alpha_path": faces["alpha_path"].cpu().tolist(),
            "contact_faces": faces["contact"].cpu().tolist(),
            "wall_impulse_kg_m_s": wall,
            "gravity_impulse_kg_m_s": gravity_impulse,
            "external_transfer": external,
            "incoming_entropy_j_k": entropy_in,
            "boundary_entropy_fluxes": boundary_entropy_fluxes,
            "boundary_sample_fluxes": sample_records,
            "boundary_cells": [
                None if b is None else {n: asdict(p) for n, p in g[1].items()}
                for b, g in zip(boundaries, ghosts)
            ],
            "entropy_production_j_k": production,
            "transport_entropy_production_j_k": transported["entropy"]
            - old["entropy"]
            - entropy_in,
            "sources": source_reports,
            "conservation_residuals": self.ledger(
                state, new, wall + gravity_impulse, external_heat_j=heat, external_transfer=external
            ),
            "active_phase_counts": [len(c) for c in new.cells],
        }

    def advance(
        self,
        state,
        duration_s,
        *,
        maximum_step_s=None,
        minimum_step_s=1e-12,
        maximum_attempts=10000,
        boundary_cells=None,
        boundary_samples=None,
        heat_rates_w=None,
        heat_temperatures_k=None,
    ):
        self.inspect(state)
        duration = _number(duration_s, "active pipe interval", positive=True)
        maximum = (
            self.maximum_step_s
            if maximum_step_s is None
            else min(self.maximum_step_s, _number(maximum_step_s, "maximum step", positive=True))
        )
        minimum = _number(minimum_step_s, "minimum step", positive=True)
        if type(maximum_attempts) is not int or maximum_attempts < 1 or minimum > maximum:
            raise ValueError("Invalid active pipe budget")
        end = state.time_s + duration
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Active pipe interval below clock resolution")
        first = state
        steps = []
        dt = min(maximum, duration)
        for attempt in range(maximum_attempts):
            proposed = min(dt, end - state.time_s)
            target = min(end, state.time_s + proposed)
            if target - state.time_s > proposed:
                target = math.nextafter(target, state.time_s)
            dt = target - state.time_s
            if dt <= 0:
                raise ValueError("Active CFL below physical clock resolution")
            try:
                new, report = self.step(
                    state,
                    dt,
                    boundary_cells=boundary_cells,
                    boundary_samples=boundary_samples,
                    heat_rates_w=heat_rates_w,
                    heat_temperatures_k=heat_temperatures_k,
                )
            except (PropertyModelError, HydraulicConvergenceError, ConservationError):
                dt *= 0.5
                if dt < minimum:
                    raise
                continue
            steps.append(report)
            state = new
            dt = min(maximum, dt * 1.5)
            if state.time_s >= end:
                impulse = math.fsum(
                    r["wall_impulse_kg_m_s"] + r["gravity_impulse_kg_m_s"] for r in steps
                )
                return {
                    "state": state.to_dict(),
                    "steps": steps,
                    "attempts": attempt + 1,
                    "conservation_residuals": self.ledger(
                        first,
                        state,
                        impulse,
                        external_heat_j=math.fsum(r.get("external_heat_j", 0.0) for r in steps),
                        external_transfer=[
                            math.fsum(r["external_transfer"][j] for r in steps)
                            for j in range(len(self.fluid.names) + 2)
                        ],
                    ),
                }
        raise HydraulicConvergenceError("Active multifluid interval budget exhausted")
