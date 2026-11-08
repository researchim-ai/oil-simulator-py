"""Conservative HEM Euler transport with a variable number of active PR phases.

Explicit first-order Rusanov/CFL transport of bulk components, momentum and
U+K, followed by an equilibrium UV recovery. All phases share P/T/velocity.
This is not the finite-slip multifluid closure, and is not a fast implicit
replacement for it. Horizontal, constant-area, periodic or rigid closed ends.
"""

import math
from dataclasses import dataclass, replace

import torch

from .equilibrium_properties import EquilibriumFluid
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .pr_phase_equilibration import PRPhaseEquilibration, PRPhaseEquilibriumState


@dataclass(frozen=True)
class PREquilibriumEulerPipeState:
    cells: tuple
    bulk_stocks: tuple
    time_s: float
    next_step_s: float
    model_id: str

    def __post_init__(self):
        object.__setattr__(self, "cells", tuple(self.cells))
        object.__setattr__(
            self,
            "bulk_stocks",
            tuple(tuple(_number(v, "bulk stock") for v in row) for row in self.bulk_stocks),
        )
        if len(self.cells) != len(self.bulk_stocks):
            raise ValueError("Bulk/phase grid differs")
        _number(self.time_s, "Euler pipe clock", nonnegative=True)
        _number(self.next_step_s, "next Euler step", positive=True)

    def to_dict(self):
        return {
            "cells": [c.to_dict() for c in self.cells],
            "bulk_stocks": self.bulk_stocks,
            "time_s": self.time_s,
            "next_step_s": self.next_step_s,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"cells", "bulk_stocks", "time_s", "next_step_s", "model_id"}:
            raise ValueError("Unknown active Euler checkpoint fields")
        return cls(
            tuple(PRPhaseEquilibriumState.from_dict(c) for c in data["cells"]),
            data["bulk_stocks"],
            data["time_s"],
            data["next_step_s"],
            data["model_id"],
        )


class PREquilibriumEulerPipe:
    def __init__(
        self,
        fluid,
        lengths_m,
        diameter_m,
        *,
        boundary_mode="periodic",
        wave_speed_bound_m_s=4000.0,
        maximum_courant=0.4,
    ):
        if not isinstance(fluid, EquilibriumFluid):
            raise TypeError("Caloric PR fluid required")
        self.fluid = fluid
        self.lengths = tuple(_number(l, "cell length", positive=True) for l in lengths_m)
        if not self.lengths:
            raise ValueError("Nonempty pipe grid required")
        self.diameter = _number(diameter_m, "diameter", positive=True)
        self.area = math.pi * self.diameter**2 / 4
        self.volumes = tuple(self.area * l for l in self.lengths)
        if boundary_mode not in ("periodic", "walls"):
            raise ValueError("Periodic or closed ends required")
        self.boundary_mode = boundary_mode
        self.wave_speed = _number(wave_speed_bound_m_s, "wave envelope", positive=True)
        self.courant = _number(maximum_courant, "maximum Courant number", positive=True)
        if self.courant > 0.5:
            raise ValueError("Use a conservative explicit Courant limit <= 0.5")
        self.equilibration = PRPhaseEquilibration(fluid)
        self.model_id = _digest(self.to_dict())
        self._volumes = fluid.tensor(self.volumes)
        self._cache = None
        self._verified = None

    def to_dict(self):
        return {
            "model": "pr_equilibrium_euler_pipe_v1",
            "fluid": self.fluid.to_dict(),
            "lengths_m": self.lengths,
            "diameter_m": self.diameter,
            "boundary_mode": self.boundary_mode,
            "wave_speed_bound_m_s": self.wave_speed,
            "maximum_courant": self.courant,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        raw = dict(data)
        if raw.pop("model", None) != "pr_equilibrium_euler_pipe_v1":
            raise ValueError("Unknown active Euler model")
        raw["fluid"] = EquilibriumFluid.from_dict(raw["fluid"], device=device)
        return cls(**raw)

    @property
    def maximum_step_s(self):
        return self.courant * min(self.lengths) / self.wave_speed

    def initialize(self, cells, *, time_s=0.0):
        if len(cells) != len(self.volumes):
            raise ValueError("Initial active Euler grid differs")
        normalized = []
        bulks = []
        for spec, volume in zip(cells, self.volumes):
            if set(spec) != {
                "component_moles",
                "total_energy_j",
                "momentum_kg_m_s",
                "pressure_guess_pa",
                "temperature_guess_k",
            }:
                raise ValueError("Explicit conservative initialization fields required")
            bulks.append(
                (*spec["component_moles"], spec["momentum_kg_m_s"], spec["total_energy_j"])
            )
            normalized.append(
                self.equilibration.recover(
                    spec["component_moles"],
                    spec["total_energy_j"],
                    spec["momentum_kg_m_s"],
                    volume,
                    pressure_guess_pa=spec["pressure_guess_pa"],
                    temperature_guess_k=spec["temperature_guess_k"],
                    time_s=time_s,
                )
            )
        state = PREquilibriumEulerPipeState(
            tuple(normalized), tuple(bulks), time_s, self.maximum_step_s, self.model_id
        )
        self._validate(state, recovered=True)
        return state

    def _packets(self, state):
        if self._cache is not None and self._cache[0] is state:
            return self._cache[1]
        nc = len(self.fluid.names)
        stocks = []
        pressures = []
        sounds = []
        entropies = []
        for cell in state.cells:
            entropy = self.equilibration.inspect(cell)
            pressure = cell.phases[0].pressure_pa
            temperature = cell.phases[0].temperature_k
            mass = math.fsum(self.equilibration._mass(p) for p in cell.phases)
            momentum = math.fsum(p.momentum_kg_m_s for p in cell.phases)
            velocity = momentum / mass
            if any(
                abs(p.pressure_pa - pressure) > 1e-6
                or abs(p.temperature_k - temperature) > 1e-8
                or abs(p.momentum_kg_m_s / self.equilibration._mass(p) - velocity) > 1e-9
                for p in cell.phases
            ):
                raise PropertyModelError("HEM requires common phase pressure/temperature/velocity")
            bound = 0.0
            for p, root in zip(cell.phases, cell.phase_roots):
                total = math.fsum(p.component_moles)
                q = self.fluid.homogeneous(
                    pressure, temperature, [n / total for n in p.component_moles], root
                )
                bound = max(bound, abs(velocity) + float(q["frozen_sound_speed_m_s"].cpu()))
            if bound > self.wave_speed:
                raise PropertyModelError("HEM acoustic envelope violated")
            stocks.append(
                [
                    *[math.fsum(p.component_moles[j] for p in cell.phases) for j in range(nc)],
                    momentum,
                    math.fsum(p.total_energy_j for p in cell.phases),
                ]
            )
            pressures.append(pressure)
            sounds.append(bound)
            entropies.append(entropy)
        phase_stocks = self.fluid.tensor(stocks)
        stock = self.fluid.tensor(state.bulk_stocks)
        errors = (stock - phase_stocks).abs()
        tolerance = stock.new_ones(stock.shape) * 1e-8
        tolerance[:, -1] = 1e-6 + 1e-9 * stock[:, -1].abs()
        if not bool((errors <= tolerance).all().cpu()):
            raise ConservationError("Conservative stocks disagree with recovered phase views")
        q = stock / self._volumes[:, None]
        mass_density = (q[:, :nc] * self.fluid.molar_masses).sum(-1)
        velocity = q[:, -2] / mass_density
        pressure = self.fluid.tensor(pressures)
        flux = torch.cat(
            (
                q[:, :nc] * velocity[:, None],
                (q[:, -2] * velocity + pressure)[:, None],
                ((q[:, -1] + pressure) * velocity)[:, None],
            ),
            dim=-1,
        )
        result = {
            "stock": stock,
            "conserved": q,
            "flux": flux,
            "sound": self.fluid.tensor(sounds),
            "entropy": math.fsum(entropies),
        }
        self._cache = (state, result)
        return result

    def inspect(self, state):
        return self._validate(state, recovered=state is self._verified)

    def _validate(self, state, *, recovered=False):
        if (
            not isinstance(state, PREquilibriumEulerPipeState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
            or len(state.cells) != len(self.volumes)
        ):
            raise PropertyModelError("Active Euler model/state differs")
        for cell, bulk, volume in zip(state.cells, state.bulk_stocks, self.volumes):
            if (
                len(bulk) != len(self.fluid.names) + 2
                or any(n < 0 for n in bulk[:-2])
                or not math.fsum(bulk[:-2]) > 0
            ):
                raise PropertyModelError("Invalid conservative component stocks")
            if (
                not isinstance(cell, PRPhaseEquilibriumState)
                or cell.model_id != self.equilibration.model_id
                or cell.time_s != state.time_s
            ):
                raise PropertyModelError("Active Euler cell identity/clock differs")
            if abs(math.fsum(p.volume_m3 for p in cell.phases) - volume) > 1e-12 + 1e-9 * volume:
                raise ConservationError("Active Euler cell volume differs")
        # A new accepted state is already a checked UV recovery. For a restart,
        # verify stability/phase allocation rather than trusting one PT root.
        if not recovered:
            for cell, bulk, volume in zip(state.cells, state.bulk_stocks, self.volumes):
                equilibrium = self.equilibration.recover(
                    bulk[:-2],
                    bulk[-1],
                    bulk[-2],
                    volume,
                    pressure_guess_pa=cell.phases[0].pressure_pa,
                    temperature_guess_k=cell.phases[0].temperature_k,
                    time_s=state.time_s,
                )
                if len(equilibrium.phases) != len(cell.phases):
                    raise PropertyModelError("Restart has a non-equilibrium phase registry")
                for a, b in zip(cell.phases, equilibrium.phases):
                    if (
                        abs(a.pressure_pa - b.pressure_pa) > 1e-6 + 1e-8 * a.pressure_pa
                        or max(abs(x - y) for x, y in zip(a.component_moles, b.component_moles))
                        > 1e-8
                    ):
                        raise PropertyModelError(
                            "Restart is not in the declared equilibrium closure"
                        )
        result = self._packets(state)
        self._verified = state
        return result

    def _sum_impulses(self, values):
        return math.fsum(values)

    def ledger(
        self,
        old,
        new,
        wall_impulse=0.0,
        external_heat_j=0.0,
        *,
        incoming_n=None,
        boundary_energy_j=0.0,
    ):
        nc = len(self.fluid.names)
        a = self._packets(old)["stock"]
        b = self._packets(new)["stock"]
        residual = (b - a).sum(0).detach().cpu().tolist()
        if incoming_n is not None:
            residual[:nc] = [v - incoming_n[j] for j, v in enumerate(residual[:nc])]
        residual[-1] -= external_heat_j + boundary_energy_j
        residual[-2] -= wall_impulse
        if (
            max(map(abs, residual[:nc])) > 1e-8
            or abs(residual[-2]) > 1e-8
            or abs(residual[-1]) > 1e-6
        ):
            raise ConservationError("Active Euler component/momentum/energy ledger failed")
        return residual

    def step(
        self,
        state,
        dt_s,
        *,
        heat_rates_w=None,
        heat_temperatures_k=None,
        _boundary_fluxes=None,
        _boundary_entropy_fluxes=None,
    ):
        dt = _number(dt_s, "Euler step", positive=True)
        end = state.time_s + dt
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Step below physical clock resolution")
        dt = end - state.time_s
        if dt > self.maximum_step_s * (1 + 1e-12):
            raise ValueError("Explicit Euler CFL limit exceeded")
        old = self.inspect(state)
        count = len(self.volumes)
        heat = (
            [0.0] * count
            if heat_rates_w is None
            else [_number(v, "heat rate") for v in heat_rates_w]
        )
        baths = [None] * count if heat_temperatures_k is None else list(heat_temperatures_k)
        if len(heat) != count or len(baths) != count:
            raise ValueError("Heat boundary registry differs")
        for power, bath in zip(heat, baths):
            if power and bath is None:
                raise ValueError("Heat input requires reservoir temperature")
            if bath is not None:
                _number(bath, "reservoir temperature", positive=True)
        if _boundary_fluxes is not None and (
            self.boundary_mode != "walls"
            or len(_boundary_fluxes) != 2
            or _boundary_entropy_fluxes is None
            or len(_boundary_entropy_fluxes) != 2
        ):
            raise ValueError("External fluxes require two explicit closed/open endpoint records")
        q, ff = old["conserved"], old["flux"]
        incoming_n = [0.0] * len(self.fluid.names)
        boundary_energy = 0.0
        boundary_entropy = 0.0
        if self.boundary_mode == "periodic":
            flux = 0.5 * (ff + torch.roll(ff, -1, 0)) - 0.5 * self.wave_speed * (
                torch.roll(q, -1, 0) - q
            )
            divergence = flux - torch.roll(flux, 1, 0)
            wall = 0.0
        else:
            interior = 0.5 * (ff[:-1] + ff[1:]) - 0.5 * self.wave_speed * (q[1:] - q[:-1])
            left = ff[0] * 0
            right = ff[-1] * 0
            left[-2] = ff[0, -2] - old["sound"][0] * q[0, -2]
            right[-2] = ff[-1, -2] + old["sound"][-1] * q[-1, -2]
            if _boundary_fluxes is not None:
                if _boundary_fluxes[0] is not None:
                    left = self.fluid.tensor(_boundary_fluxes[0])
                if _boundary_fluxes[1] is not None:
                    right = self.fluid.tensor(_boundary_fluxes[1])
                if (
                    left.shape != (len(self.fluid.names) + 2,)
                    or right.shape != left.shape
                    or not bool((torch.isfinite(left).all() & torch.isfinite(right).all()).cpu())
                ):
                    raise ValueError("Invalid external Euler flux vector")
                incoming = dt * self.area * (left - right)
                incoming_n = incoming[:-2].cpu().tolist()
                boundary_energy = float(incoming[-1].cpu())
                boundary_entropy = (
                    dt * self.area * (_boundary_entropy_fluxes[0] - _boundary_entropy_fluxes[1])
                )
            if float(torch.minimum(left[-2], right[-2]).cpu()) <= 0:
                raise PropertyModelError("Closed-end traction leaves the supported domain")
            flux = torch.cat((left[None, :], interior, right[None, :]), dim=0)
            divergence = flux[1:] - flux[:-1]
            wall = float((dt * self.area * (left[-2] - right[-2])).cpu())
        delta = -dt * self.area * divergence
        delta = delta + torch.nn.functional.pad(
            self.fluid.tensor(heat)[:, None] * dt, (q.shape[1] - 1, 0)
        )
        target = (old["stock"] + delta).detach().cpu().tolist()
        changed = (delta != 0).any(-1).detach().cpu().tolist()
        cells = []
        for i, (values, cell) in enumerate(zip(target, state.cells)):
            if (
                not all(math.isfinite(v) for v in values)
                or any(v < 0 for v in values[:-2])
                or not math.fsum(values[:-2]) > 0
            ):
                raise PropertyModelError("Explicit transport left positive conservative domain")
            if not changed[i]:
                cells.append(replace(cell, time_s=end))
                continue
            cells.append(
                self.equilibration.recover(
                    values[:-2],
                    values[-1],
                    values[-2],
                    self.volumes[i],
                    pressure_guess_pa=cell.phases[0].pressure_pa,
                    temperature_guess_k=cell.phases[0].temperature_k,
                    time_s=end,
                )
            )
        new = PREquilibriumEulerPipeState(
            tuple(cells), tuple(tuple(v) for v in target), end, state.next_step_s, self.model_id
        )
        current = self._validate(new, recovered=True)
        entropy_supply = dt * math.fsum(
            power / bath for power, bath in zip(heat, baths) if bath is not None
        )
        production = current["entropy"] - old["entropy"] - entropy_supply - boundary_entropy
        if production < -1e-7:
            raise ConservationError("Active Euler entropy decreased below its heat supply")
        external_heat = dt * math.fsum(heat)
        return new, {
            "dt_s": dt,
            "state": new.to_dict(),
            "face_fluxes": flux.detach().cpu().tolist(),
            "external_heat_j": external_heat,
            "incoming_component_moles": incoming_n,
            "boundary_energy_j": boundary_energy,
            "boundary_entropy_j_k": boundary_entropy,
            "entropy_supply_j_k": entropy_supply,
            "entropy_production_j_k": production,
            "wall_impulse_kg_m_s": wall,
            "phase_counts": [len(c.phases) for c in new.cells],
            "conservation_residuals": self.ledger(
                state,
                new,
                wall,
                external_heat,
                incoming_n=incoming_n,
                boundary_energy_j=boundary_energy,
            ),
        }

    def advance(
        self,
        state,
        duration_s,
        *,
        heat_rates_w=None,
        heat_temperatures_k=None,
        maximum_step_s=None,
        minimum_step_s=1e-12,
        maximum_attempts=10000,
    ):
        self.inspect(state)
        duration = _number(duration_s, "Euler interval", positive=True)
        minimum = _number(minimum_step_s, "minimum step", positive=True)
        maximum = (
            self.maximum_step_s
            if maximum_step_s is None
            else min(self.maximum_step_s, _number(maximum_step_s, "maximum step", positive=True))
        )
        if type(maximum_attempts) is not int or maximum_attempts < 1 or minimum > maximum:
            raise ValueError("Invalid active Euler integration budget")
        start = state
        end = state.time_s + duration
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Interval below clock resolution")
        dt = min(state.next_step_s, maximum)
        steps = []
        for attempt in range(maximum_attempts):
            proposed = min(dt, maximum, end - state.time_s)
            target = min(end, state.time_s + proposed)
            if target - state.time_s > proposed:
                target = math.nextafter(target, state.time_s)
            dt = target - state.time_s
            if dt <= 0:
                raise ValueError("Explicit CFL step below clock resolution")
            try:
                new, report = self.step(
                    state, dt, heat_rates_w=heat_rates_w, heat_temperatures_k=heat_temperatures_k
                )
            except (HydraulicConvergenceError, PropertyModelError, ConservationError):
                dt *= 0.5
                if dt < minimum:
                    raise
                continue
            steps.append(report)
            nextdt = min(maximum, dt * 1.25)
            cached = self._cache
            state = replace(new, next_step_s=nextdt)
            self._verified = state
            if cached is not None and cached[0] is new:
                self._cache = (state, cached[1])
            dt = nextdt
            if state.time_s >= end:
                heat = math.fsum(r["external_heat_j"] for r in steps)
                wall = math.fsum(r["wall_impulse_kg_m_s"] for r in steps)
                return {
                    "state": state.to_dict(),
                    "steps": steps,
                    "attempts": attempt + 1,
                    "conservation_residuals": self.ledger(start, state, wall, heat),
                }
        raise HydraulicConvergenceError("Active Euler interval budget exhausted")
