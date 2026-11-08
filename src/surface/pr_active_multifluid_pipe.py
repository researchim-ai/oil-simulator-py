"""Explicit active-phase PR multifluid transport and thermochemical sources.

Horizontal constant-area cells. Each present phase owns N/V/M/E and P/T/u.
Absent slots have exactly zero inventory and no EOS evaluation. Pairwise
interface closure is explicit in pr_active_multifluid_flux, with CFL, domain
and entropy rejection. This is not the previous fixed-reference BN scheme.
"""

import itertools
import math
from dataclasses import dataclass

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _name, _number
from .operation import _digest
from .peng_robinson import R
from .pr_active_multifluid_flux import fluctuations
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
        self.thermochemical_time = thermochemical_relaxation_time_s
        self.chemical = (
            None
            if self.thermochemical_time is None
            else PRThermochemicalRelaxation(
                fluid, self.thermochemical_time, maximum_active_phases=len(self.names)
            )
        )
        self.equilibrium = PRPhaseEquilibration(fluid)
        self.model_id = _digest(self.to_dict())
        self._cache = None

    def to_dict(self):
        return {
            "model": "pr_active_multifluid_pipe_v1",
            "fluid": self.fluid.to_dict(),
            "phase_slots": self.slots,
            "lengths_m": self.lengths,
            "diameter_m": self.diameter,
            "boundary_mode": self.boundary_mode,
            "wave_speed_bound_m_s": self.wave_speed,
            "maximum_courant": self.courant,
            "thermochemical_relaxation_time_s": self.thermochemical_time,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        from .equilibrium_properties import EquilibriumFluid

        raw = dict(data)
        if raw.pop("model", None) != "pr_active_multifluid_pipe_v1":
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
        rows = []
        alpha = []
        pressures = []
        temperatures = []
        velocities = []
        sounds = []
        entropy_terms = []
        precisions = []
        for cell, volume in zip(state.cells, self.volumes):
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
            stocks = dict(cell)
            qq = []
            aa = []
            pp = []
            tt = []
            uu = []
            cc = []
            precision = []
            for j, name in enumerate(self.names):
                if name not in stocks:
                    qq.append([0.0] * (nc + 2))
                    aa.append(0.0)
                    pp.append(0.0)
                    tt.append(0.0)
                    uu.append(0.0)
                    cc.append(0.0)
                    precision.append(0.0)
                    continue
                phase = stocks[name]
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
                eos = f.homogeneous(
                    phase.pressure_pa, phase.temperature_k, x, self.slots[j]["root"]
                )
                expected_volume = amount * float(eos["molar_volume_m3_mol"].cpu())
                expected_energy = (
                    amount * float(eos["molar_internal_energy_j_mol"].cpu())
                    + 0.5 * phase.momentum_kg_m_s * velocity
                )
                if abs(expected_volume - phase.volume_m3) > 1e-12 + 1e-9 * phase.volume_m3 or abs(
                    expected_energy - phase.total_energy_j
                ) > 1e-6 + 1e-9 * abs(expected_energy):
                    raise ConservationError(
                        "Active phase checkpoint disagrees with its homogeneous EOS"
                    )
                entropy_terms.append(amount * float(eos["molar_entropy_j_mol_k"].cpu()))
                acoustic = float(eos["frozen_sound_speed_m_s"].cpu())
                sound = abs(velocity) + acoustic
                cv = float(eos["homogeneous_molar_cv_j_mol_k"].cpu())
                ideal_cp = math.fsum(v * c.ideal_cp_j_mol_k for v, c in zip(x, f.components))
                molar_v = expected_volume / amount
                bulk_modulus = mass / phase.volume_m3 * acoustic**2
                energy_scale = abs(phase.total_energy_j) / amount + (ideal_cp + R) * (
                    phase.temperature_k + f.tref
                )
                precision.append(
                    64
                    * torch.finfo(torch.float64).eps
                    * (abs(phase.pressure_pa) + bulk_modulus + cv * phase.temperature_k / molar_v)
                    * (1 + energy_scale / (cv * phase.temperature_k))
                )
                if sound > self.wave_speed * (1 + 1e-12):
                    raise PropertyModelError("Active phase exceeds acoustic envelope")
                qq.append(
                    [
                        *(v / volume for v in phase.component_moles),
                        phase.momentum_kg_m_s / volume,
                        phase.total_energy_j / volume,
                    ]
                )
                aa.append(phase.volume_m3 / volume)
                pp.append(phase.pressure_pa)
                tt.append(phase.temperature_k)
                uu.append(velocity)
                cc.append(sound)
            rows.append(qq)
            alpha.append(aa)
            pressures.append(pp)
            temperatures.append(tt)
            velocities.append(uu)
            sounds.append(cc)
            precisions.append(precision)
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
            math.fsum(p.total_energy_j for cell in state.cells for _, p in cell),
            math.fsum(p.momentum_kg_m_s for cell in state.cells for _, p in cell),
        ]

    def ledger(self, old, new, wall_impulse=0.0, external_heat_j=0.0):
        r = [b - a for a, b in zip(self._totals(old), self._totals(new))]
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

    def step(self, state, dt_s):
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
                left[key] = torch.cat((gl, old[key]), 0)
                right[key] = torch.cat((old[key], gr), 0)
            minus, plus, aminus, aplus, faces = fluctuations(left, right, self.wave_speed)
            delta = minus[1:] + plus[:-1]
            adelta = aminus[1:] + aplus[:-1]
            wall = float(
                (
                    dt
                    * self.area
                    * (faces["flux"][0, :, -2].sum() - faces["flux"][-1, :, -2].sum())
                ).cpu()
            )
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
        if transported["entropy"] - old["entropy"] < -entropy_allowance(
            transported["entropy"], old["entropy"]
        ):
            raise ConservationError("Active transport entropy decreased before chemical relaxation")
        source_reports = []
        if self.chemical is not None:
            output = []
            for cell in transport.cells:
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
            new = transport
        current = self.inspect(new)
        production = current["entropy"] - old["entropy"]
        if production < -entropy_allowance(current["entropy"], old["entropy"]):
            raise ConservationError("Active pipe total entropy decreased")
        return new, {
            "state": new.to_dict(),
            "transport_state": transport.to_dict(),
            "dt_s": dt,
            "path_quadrature_order": 8,
            "face_fluxes": faces["flux"].cpu().tolist(),
            "phase_path": faces["phase_path"].cpu().tolist(),
            "alpha_path": faces["alpha_path"].cpu().tolist(),
            "contact_faces": faces["contact"].cpu().tolist(),
            "wall_impulse_kg_m_s": wall,
            "entropy_production_j_k": production,
            "transport_entropy_production_j_k": transported["entropy"] - old["entropy"],
            "sources": source_reports,
            "conservation_residuals": self.ledger(state, new, wall),
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
                new, report = self.step(state, dt)
            except (PropertyModelError, HydraulicConvergenceError, ConservationError):
                dt *= 0.5
                if dt < minimum:
                    raise
                continue
            steps.append(report)
            state = new
            dt = min(maximum, dt * 1.5)
            if state.time_s >= end:
                impulse = math.fsum(r["wall_impulse_kg_m_s"] for r in steps)
                return {
                    "state": state.to_dict(),
                    "steps": steps,
                    "attempts": attempt + 1,
                    "conservation_residuals": self.ledger(first, state, impulse),
                }
        raise HydraulicConvergenceError("Active multifluid interval budget exhausted")
