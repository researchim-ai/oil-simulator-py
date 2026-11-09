"""PR material inventories coupled to an isothermal incompressible film.

The mechanical free surface has a passive exterior. PR supplies uniform
reference density/calorics, composition and liquid root; it is not the
compressible thermodynamic pressure solver of the mechanical subsystem.
Neglected density response is explicitly bounded by pressure/bulk modulus.
A derived nonnegative velocity-variance energy prevents kinetic double count.
"""

import math
from dataclasses import dataclass

import torch

from .equilibrium_properties import EquilibriumFluid
from .inertial_capillary_film import InertialCapillaryFilm, InertialCapillaryState
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .pr_homogeneous_uv import caloric_tv
from .pr_interphase_relaxation import PRRelaxationPhase


@dataclass(frozen=True)
class PRIsothermalCapillaryState:
    mechanics: InertialCapillaryState
    phases: tuple
    model_id: str

    def __post_init__(self):
        object.__setattr__(self, "phases", tuple(self.phases))

    @property
    def time_s(self):
        return self.mechanics.time_s

    def to_dict(self):
        from dataclasses import asdict

        return {
            "mechanics": self.mechanics.to_dict(),
            "phases": [asdict(p) for p in self.phases],
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"mechanics", "phases", "model_id"}:
            raise ValueError("Unknown PR film checkpoint")
        return cls(
            InertialCapillaryState.from_dict(data["mechanics"]),
            tuple(PRRelaxationPhase.from_dict(p) for p in data["phases"]),
            data["model_id"],
        )


class PRIsothermalCapillaryFilm:
    def __init__(
        self,
        fluid,
        mechanics,
        reference_pressure_pa,
        temperature_k,
        mole_fractions,
        *,
        maximum_density_response=1e-4,
    ):
        if not isinstance(fluid, EquilibriumFluid) or not isinstance(
            mechanics, InertialCapillaryFilm
        ):
            raise TypeError("Explicit PR fluid and joint mechanical model required")
        if fluid.device != mechanics.surface.device:
            raise ValueError("Shared material/mechanics device required")
        self.fluid = fluid
        self.mechanics = mechanics
        self.pressure = _number(reference_pressure_pa, "PR reference pressure", positive=True)
        self.temperature = _number(temperature_k, "thermostat temperature", positive=True)
        self.x = tuple(
            _number(v, "uniform mole fraction", nonnegative=True) for v in mole_fractions
        )
        if len(self.x) != len(fluid.names) or abs(math.fsum(self.x) - 1) > 1e-12:
            raise ValueError("Normalized uniform reference composition required")
        self.density_response = _number(
            maximum_density_response, "density-response bound", positive=True
        )
        if self.density_response > 0.01:
            raise ValueError("Incompressible response bound must not exceed .01")
        q = fluid.homogeneous(self.pressure, self.temperature, self.x, "liquid")
        self.vm = float(q["molar_volume_m3_mol"].cpu())
        self.um = float(q["molar_internal_energy_j_mol"].cpu())
        self.sm = float(q["molar_entropy_j_mol_k"].cpu())
        self.rho = float((q["molar_mass_kg_mol"] / q["molar_volume_m3_mol"]).cpu())
        self.bulk_modulus = float(
            (
                -self.vm
                * caloric_tv(
                    fluid,
                    fluid.tensor(self.temperature),
                    fluid.tensor(self.vm),
                    fluid.tensor(self.x),
                )["dpdv"]
            ).cpu()
        )
        _number(self.bulk_modulus, "reference bulk modulus", positive=True)
        if abs(self.rho - mechanics.surface.rho) > 1e-10 * self.rho:
            raise ValueError("Mechanical density must be derived from the PR reference state")
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_isothermal_capillary_film_v1",
            "fluid": self.fluid.to_dict(),
            "mechanics": self.mechanics.to_dict(),
            "reference_pressure_pa": self.pressure,
            "temperature_k": self.temperature,
            "mole_fractions": self.x,
            "maximum_density_response": self.density_response,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        raw = dict(data)
        if raw.pop("model", None) != "pr_isothermal_capillary_film_v1":
            raise ValueError("Unknown PR capillary model")
        raw["fluid"] = EquilibriumFluid.from_dict(raw["fluid"], device=device)
        raw["mechanics"] = InertialCapillaryFilm.from_dict(raw["mechanics"], device=device)
        return cls(**raw)

    def _material(self, state):
        f = self.fluid
        s = self.mechanics.surface
        h = f.tensor(state.thickness_m)
        u = f.tensor(state.face_velocity_m_s)
        v = h * s.cell_area
        n = (v / self.vm)[:, None] * f.tensor(self.x)
        uc = self.mechanics.center(u)
        momentum = self.rho * v * uc
        energy = v / self.vm * self.um + 0.5 * self.rho * v * uc.square()
        values = torch.stack((v, momentum, energy), -1).cpu().tolist()
        numbers = n.cpu().tolist()
        return tuple(
            PRRelaxationPhase(tuple(nn), vv, self.pressure, self.temperature, mm, ee)
            for nn, (vv, mm, ee) in zip(numbers, values, strict=True)
        )

    def initialize(self, thickness_m, face_velocity_m_s=None):
        m = self.mechanics.initialize(thickness_m, face_velocity_m_s)
        state = PRIsothermalCapillaryState(m, self._material(m), self.model_id)
        self.inspect(state)
        return state

    def inspect(self, state):
        if state.model_id != self.model_id or _digest(self.to_dict()) != self.model_id:
            raise ValueError("PR mechanical checkpoint identity differs")
        mechanical = self.mechanics.inspect(state.mechanics)
        expected = self._material(state.mechanics)
        if len(expected) != len(state.phases):
            raise ValueError("PR mechanical material mesh differs")
        for actual, target in zip(state.phases, expected, strict=True):
            if actual.pressure_pa != self.pressure or actual.temperature_k != self.temperature:
                raise ConservationError("PR film left its declared isothermal reference")
            for aa, bb in zip(
                (
                    *actual.component_moles,
                    actual.volume_m3,
                    actual.momentum_kg_m_s,
                ),
                (
                    *target.component_moles,
                    target.volume_m3,
                    target.momentum_kg_m_s,
                ),
                strict=True,
            ):
                if abs(aa - bb) > 1e-24 + 1e-11 * max(abs(aa), abs(bb)):
                    raise ConservationError(
                        "PR film material inventory disagrees with mechanical field"
                    )
            energy_scale = abs(
                math.fsum(target.component_moles) * self.um
            ) + 0.5 * target.momentum_kg_m_s**2 / (self.rho * target.volume_m3)
            if (
                abs(actual.total_energy_j - target.total_energy_j)
                > 128 * torch.finfo(torch.float64).eps * energy_scale + 1e-24
            ):
                raise ConservationError(
                    "PR film caloric or kinetic energy disagrees with mechanical field"
                )
        f = self.fluid
        s = self.mechanics.surface
        h = f.tensor(state.mechanics.thickness_m)
        u = f.tensor(state.mechanics.face_velocity_m_s)
        pressure = -s.sigma * s.laplacian(h) + s.rho * s.normal_g * h
        bound = float(((pressure.abs() + s.rho * u.square()) / self.bulk_modulus).max().cpu())
        if not math.isfinite(bound) or bound > self.density_response:
            raise PropertyModelError("Film exceeds the declared incompressible PR response bound")
        variance = float(
            (0.125 * s.rho * s.cell_area * (h * (u - torch.roll(u, 1, 0)).square()).sum()).cpu()
        )
        volume = math.fsum(p.volume_m3 for p in state.phases)
        amount = [
            math.fsum(p.component_moles[j] for p in state.phases) for j in range(len(f.names))
        ]
        bulk = math.fsum(p.total_energy_j for p in state.phases)
        bath = state.mechanics.physical_heat_j + state.mechanics.numerical_heat_j
        return {
            "component_moles": amount,
            "liquid_volume_m3": volume,
            "momentum_kg_m_s": math.fsum(p.momentum_kg_m_s for p in state.phases),
            "pr_bulk_energy_j": bulk,
            "velocity_variance_energy_j": variance,
            "potential_energy_j": mechanical["potential_energy_j"],
            "thermostat_energy_j": bath,
            "total_energy_j": bulk + variance + mechanical["potential_energy_j"] + bath,
            "estimated_density_response": bound,
            "physical_entropy_production_j_k": state.mechanics.physical_heat_j / self.temperature,
            "numerical_entropy_j_k": state.mechanics.numerical_heat_j / self.temperature,
        }

    def step(self, state, dt_s):
        before = self.inspect(state)
        m, r = self.mechanics.step(state.mechanics, dt_s)
        end = PRIsothermalCapillaryState(m, self._material(m), self.model_id)
        after = self.inspect(end)
        dE = after["total_energy_j"] - before["total_energy_j"]
        roundoff = 64 * math.fsum(
            math.ulp(p.total_energy_j) for st in (state, end) for p in st.phases
        )
        if abs(dE) > roundoff + 1e-21:
            raise ConservationError("PR plus capillary plus thermostat energy balance failed")
        for a, b in zip(before["component_moles"], after["component_moles"], strict=True):
            if abs(a - b) > 1e-22 + 1e-11 * max(abs(a), abs(b)):
                raise ConservationError("PR capillary component balance failed")
        dM = after["momentum_kg_m_s"] - before["momentum_kg_m_s"] - r["wall_impulse_kg_m_s"]
        if abs(dM) > 1e-22 + 1e-10 * (
            abs(before["momentum_kg_m_s"]) + abs(after["momentum_kg_m_s"])
        ):
            raise ConservationError("PR capillary impulse balance failed")
        q = self.fluid.tensor(r["face_volume_flux_m2_s"]) * self.mechanics.surface.width
        component_flux = q[:, None] * self.fluid.tensor(self.x) / self.vm
        f = self.fluid
        s = self.mechanics.surface
        energy_flux = f.tensor(r["face_mechanical_energy_flux_w"]) + self.um / self.vm * q

        def local_energy(st):
            h = f.tensor(st.mechanics.thickness_m)
            u = f.tensor(st.mechanics.face_velocity_m_s)
            g = s.gradient(h, 0)
            variance = 0.125 * self.rho * s.cell_area * h * (u - torch.roll(u, 1, 0)).square()
            potential = s.cell_area * (
                0.25 * s.sigma * (g.square() + torch.roll(g, 1, 0).square())
                + 0.5 * self.rho * s.normal_g * h.square()
            )
            return f.tensor([p.total_energy_j for p in st.phases]) + variance + potential

        e0, e1 = local_energy(state), local_energy(end)
        heat = f.tensor(r["cell_physical_heat_j"]) + f.tensor(r["cell_numerical_heat_j"])
        local_residual = e1 - e0 + r["dt_s"] * (energy_flux - torch.roll(energy_flux, 1, 0)) + heat
        local_budget = (
            e0.abs()
            + e1.abs()
            + r["dt_s"] * (energy_flux.abs() + torch.roll(energy_flux, 1, 0).abs())
            + heat
        )
        if not bool(
            (local_residual.abs() <= 1e-23 + 256 * torch.finfo(torch.float64).eps * local_budget)
            .all()
            .cpu()
        ):
            raise ConservationError("Local PR capillary energy flux balance failed")
        return end, {
            "mechanics": r,
            "state": end.to_dict(),
            "component_flux_mol_s": component_flux.cpu().tolist(),
            "face_total_energy_flux_w": energy_flux.cpu().tolist(),
            "entropy_flux_w_k": (self.sm / self.vm * q).cpu().tolist(),
            "local_energy_residual_j": local_residual.cpu().tolist(),
            "pressure_work_convention": "uniform_exterior_pressure_enthalpy_and_surface_work_cancelled",
            "thermostat_physical_heat_j": r["physical_heat_j"],
            "thermostat_numerical_heat_j": r["numerical_heat_j"],
            "joint_energy_residual_j": dE,
            "momentum_residual_kg_m_s": dM,
        }

    def advance(
        self, state, duration_s, *, initial_step_s, minimum_step_s=1e-12, maximum_attempts=10000
    ):
        self.inspect(state)
        duration = _number(duration_s, "PR film duration", positive=True)
        maximum = _number(initial_step_s, "PR film step", positive=True)
        minimum = _number(minimum_step_s, "PR film minimum step", positive=True)
        if minimum > maximum or type(maximum_attempts) is not int or maximum_attempts < 1:
            raise ValueError("Ordered PR film steps and attempt bound required")
        target = state.time_s + duration
        if not math.isfinite(target) or target <= state.time_s:
            raise ValueError("PR capillary clock cannot represent this report interval")
        dt = maximum
        steps = []
        rejected = 0
        while state.time_s < target:
            if len(steps) + rejected >= maximum_attempts:
                raise PropertyModelError("PR film interval attempt limit")
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
