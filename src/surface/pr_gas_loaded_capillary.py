"""PR liquid film under separate isothermal closed gas chambers.

Gas pressure/volume feedback is implicit in the film Newton solve. This is a
confined benchmark geometry, not an axially transported gas core: chamber
partitions carry horizontal reaction and thermostats hold gas/liquid T fixed.
"""

import math
from dataclasses import dataclass

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .peng_robinson import R
from .pr_homogeneous_uv import caloric_tv, recover_homogeneous_phases
from .pr_interphase_relaxation import PRRelaxationPhase
from .pr_isothermal_capillary_film import PRIsothermalCapillaryFilm, PRIsothermalCapillaryState


@dataclass(frozen=True)
class PRGasLoadedCapillaryState:
    film: PRIsothermalCapillaryState
    gas: tuple
    gas_thermostat_energy_j: float
    model_id: str

    def __post_init__(self):
        object.__setattr__(self, "gas", tuple(self.gas))
        _number(self.gas_thermostat_energy_j, "gas thermostat energy")

    @property
    def time_s(self):
        return self.film.time_s

    def to_dict(self):
        from dataclasses import asdict

        return {
            "film": self.film.to_dict(),
            "gas": [asdict(p) for p in self.gas],
            "gas_thermostat_energy_j": self.gas_thermostat_energy_j,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, raw):
        if set(raw) != {"film", "gas", "gas_thermostat_energy_j", "model_id"}:
            raise ValueError("Unknown gas-loaded film checkpoint fields")
        return cls(
            PRIsothermalCapillaryState.from_dict(raw["film"]),
            tuple(PRRelaxationPhase.from_dict(p) for p in raw["gas"]),
            raw["gas_thermostat_energy_j"],
            raw["model_id"],
        )


class _GasPotential:
    def __init__(self, model, gas):
        self.model = model
        f = model.film.fluid
        n = f.tensor([p.component_moles for p in gas])
        self.amount = n.sum(-1)
        self.x = n / self.amount[:, None]
        self.temperature = f.tensor([p.temperature_k for p in gas])
        q = caloric_tv(
            f, self.temperature, f.tensor([p.volume_m3 for p in gas]) / self.amount, self.x
        )
        self.a, self.b = q["a"], q["b"]

    @staticmethod
    def log_ratio(x):
        # log1p(x)/x with an analytic continuation at zero. Both branches
        # remain finite for jacfwd at an exactly unchanged volume.
        small = x.abs() < 1e-4
        safe = torch.where(small, torch.ones_like(x), x)
        polynomial = 1 + x * (-0.5 + x * (1 / 3 + x * (-0.25 + x / 5)))
        return torch.where(small, polynomial, torch.log1p(x) / safe)

    def pressure_and_energy(self, old_h, dh):
        m = self.model
        area = m.film.mechanics.surface.cell_area
        v0 = area * (m.ceiling - old_h) / self.amount
        dv = -area * dh / self.amount
        c0 = v0 - self.b
        c1 = v0 + (1 - math.sqrt(2)) * self.b
        c2 = v0 + (1 + math.sqrt(2)) * self.b
        pressure = R * self.temperature / c0 * self.log_ratio(dv / c0) - self.a / (
            2 * math.sqrt(2) * self.b
        ) * (self.log_ratio(dv / c1) / c1 - self.log_ratio(dv / c2) / c2)
        return pressure, area * dh * pressure


class PRGasLoadedCapillaryFilm:
    def __init__(self, film, ceiling_height_m, gas_temperature_k):
        if not isinstance(film, PRIsothermalCapillaryFilm):
            raise TypeError("Isothermal PR film required")
        self.film = film
        self.heights = tuple(
            _number(x, "gas chamber ceiling", positive=True) for x in ceiling_height_m
        )
        if len(self.heights) != film.mechanics.surface.shape[0]:
            raise ValueError("One chamber ceiling per material cell required")
        self.ceiling = film.fluid.tensor(self.heights)
        self.temperature = _number(gas_temperature_k, "gas thermostat temperature", positive=True)
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_closed_gas_loaded_capillary_v1",
            "film": self.film.to_dict(),
            "ceiling_height_m": self.heights,
            "gas_temperature_k": self.temperature,
        }

    @classmethod
    def from_dict(cls, raw, *, device="cpu"):
        data = dict(raw)
        if data.pop("model", None) != "pr_closed_gas_loaded_capillary_v1":
            raise ValueError("Unknown confined gas/film model")
        data["film"] = PRIsothermalCapillaryFilm.from_dict(data["film"], device=device)
        return cls(**data)

    def initialize(
        self, thickness_m, gas_mole_fractions, *, face_velocity_m_s=None, gas_pressures_pa=None
    ):
        liquid = self.film.initialize(thickness_m, face_velocity_m_s)
        f, s = self.film.fluid, self.film.mechanics.surface
        volumes = (self.ceiling - f.tensor(thickness_m)) * s.cell_area
        if not bool((volumes > 0).all().cpu()):
            raise ValueError("Positive initial gas chamber clearance required")
        pressures = (
            [self.film.pressure] * len(self.heights)
            if gas_pressures_pa is None
            else list(gas_pressures_pa)
        )
        if len(pressures) != len(self.heights):
            raise ValueError("One initial gas pressure per chamber required")
        phases = []
        for volume, pressure in zip(volumes.cpu().tolist(), pressures, strict=True):
            q = f.homogeneous(pressure, self.temperature, gas_mole_fractions, "vapor")
            amount = volume / float(q["molar_volume_m3_mol"].cpu())
            phases.append(
                PRRelaxationPhase(
                    tuple(amount * x for x in gas_mole_fractions),
                    volume,
                    pressure,
                    self.temperature,
                    0.0,
                    amount * float(q["molar_internal_energy_j_mol"].cpu()),
                )
            )
        state = PRGasLoadedCapillaryState(liquid, tuple(phases), 0.0, self.model_id)
        self.inspect(state)
        return state

    def inspect(self, state):
        if state.model_id != self.model_id or _digest(self.to_dict()) != self.model_id:
            raise ValueError("Gas-loaded film checkpoint identity differs")
        liquid = self.film.inspect(state.film)
        if len(state.gas) != len(self.heights):
            raise ValueError("Gas chamber mesh differs")
        f, s = self.film.fluid, self.film.mechanics.surface
        volume = (self.ceiling - f.tensor(state.film.mechanics.thickness_m)) * s.cell_area
        recovered = recover_homogeneous_phases(
            f,
            [p.component_moles for p in state.gas],
            volume,
            [0.0] * len(state.gas),
            [p.total_energy_j for p in state.gas],
            ["vapor"] * len(state.gas),
            [self.temperature] * len(state.gas),
        )
        for i, (actual, target) in enumerate(zip(state.gas, recovered, strict=True)):
            if (
                actual.momentum_kg_m_s != 0.0
                or actual.temperature_k != self.temperature
                or abs(actual.volume_m3 - float(volume[i].cpu())) > 1e-25 + 1e-11 * actual.volume_m3
                or abs(target.temperature_k - self.temperature) > 1e-7
                or abs(actual.pressure_pa - target.pressure_pa) > 1e-4 + 1e-9 * target.pressure_pa
            ):
                raise PropertyModelError("Gas chamber geometry or PR calorics disagree")
        response = (
            max(abs(p.pressure_pa - self.film.pressure) for p in state.gas) / self.film.bulk_modulus
        )
        if response + liquid["estimated_density_response"] > self.film.density_response:
            raise PropertyModelError("Gas pressure exceeds the liquid incompressibility bound")
        _number(state.gas_thermostat_energy_j, "gas thermostat energy")
        return {
            "total_energy_j": math.fsum(
                [
                    liquid["total_energy_j"],
                    state.gas_thermostat_energy_j,
                    *(p.total_energy_j for p in state.gas),
                ]
            ),
            "liquid": liquid,
            "gas_energy_j": math.fsum(p.total_energy_j for p in state.gas),
        }

    def step(self, state, dt_s):
        before = self.inspect(state)
        m = self.film
        f, s = m.fluid, m.mechanics.surface
        load = _GasPotential(self, state.gas)
        mechanical, report = m.mechanics.step(state.film.mechanics, dt_s, normal_load=load)
        liquid = PRIsothermalCapillaryState(mechanical, m._material(mechanical), m.model_id)
        volumes = s.cell_area * (self.ceiling - f.tensor(mechanical.thickness_m))
        n = f.tensor([p.component_moles for p in state.gas])
        amount = n.sum(-1)
        q = caloric_tv(
            f, f.tensor([self.temperature] * len(state.gas)), volumes / amount, n / amount[:, None]
        )
        gas = recover_homogeneous_phases(
            f,
            n,
            volumes,
            [0.0] * len(state.gas),
            amount * q["u"],
            ["vapor"] * len(state.gas),
            [self.temperature] * len(state.gas),
        )
        # Force exact reference T in the checkpoint, after homogeneous recovery
        # has independently checked branch admissibility and caloric stocks.
        gas = tuple(
            PRRelaxationPhase(
                p.component_moles,
                p.volume_m3,
                p.pressure_pa,
                self.temperature,
                0.0,
                p.total_energy_j,
            )
            for p in gas
        )
        potential_change = math.fsum(report["cell_normal_load_potential_change_j"])
        internal_change = math.fsum(
            b.total_energy_j - a.total_energy_j for a, b in zip(state.gas, gas, strict=True)
        )
        bath = potential_change - internal_change
        end = PRGasLoadedCapillaryState(
            liquid, gas, state.gas_thermostat_energy_j + bath, self.model_id
        )
        after = self.inspect(end)
        residual = after["total_energy_j"] - before["total_energy_j"]
        allowance = (
            128
            * math.fsum(
                math.ulp(p.total_energy_j)
                for st in (state, end)
                for p in (*st.gas, *st.film.phases)
            )
            + 1e-21
        )
        if not math.isfinite(residual) or abs(residual) > allowance:
            raise ConservationError("Gas pressure/film/thermostat total energy balance failed")
        return end, {
            "mechanics": report,
            "state": end.to_dict(),
            "gas_helmholtz_change_j": potential_change,
            "gas_internal_energy_change_j": internal_change,
            "gas_thermostat_heat_j": bath,
            "energy_residual_j": residual,
            "energy_roundoff_allowance_j": allowance,
            "support_impulse_kg_m_s": -report["normal_load_impulse_kg_m_s"],
            "qualification": "Separate closed isothermal gas chambers; no axial gas transport",
        }

    def advance(
        self, state, duration_s, *, initial_step_s, minimum_step_s=1e-12, maximum_attempts=10000
    ):
        self.inspect(state)
        duration = _number(duration_s, "gas-loaded report interval", positive=True)
        maximum = _number(initial_step_s, "gas-loaded maximum step", positive=True)
        minimum = _number(minimum_step_s, "gas-loaded minimum step", positive=True)
        if minimum > maximum or type(maximum_attempts) is not int or maximum_attempts < 1:
            raise ValueError("Ordered steps and bounded attempts required")
        target = state.time_s + duration
        if not math.isfinite(target) or target <= state.time_s:
            raise ValueError("Gas-loaded clock cannot represent this interval")
        dt, rejected, steps = maximum, 0, []
        while state.time_s < target:
            if len(steps) + rejected >= maximum_attempts:
                raise HydraulicConvergenceError("Gas-loaded attempt limit")
            trial = min(dt, target - state.time_s)
            try:
                new, r = self.step(state, trial)
            except (ConservationError, PropertyModelError, HydraulicConvergenceError):
                dt = trial / 2
                rejected += 1
                if dt < minimum:
                    raise
                continue
            state = new
            steps.append(r)
            dt = min(maximum, trial * 1.5)
        return state, {"steps": steps, "rejected_steps": rejected}
