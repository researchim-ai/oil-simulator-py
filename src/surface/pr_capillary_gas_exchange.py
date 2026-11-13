"""Two-way local shear between PR gas stocks and staggered PR film faces.

A frozen nonnegative conductance is supplied explicitly (kg/s per dual cell),
not fitted or inferred from an unqualified entrainment correlation. The liquid
plug velocity is the interface velocity in the legacy plug mode. Explicit
stokes_profile mode reconstructs the surface speed and stationary-wall
reaction using the shared local viscous film response. Fixed-volume source: no gas
transport, normal pressure work, phase transfer or report-clock advancement.
"""

import math
from dataclasses import replace

import torch

from .liquid import ConservationError, PropertyModelError, _number
from .operation import _digest
from .planar_film import resistance_response
from .pr_homogeneous_uv import recover_homogeneous_phases
from .pr_isothermal_capillary_film import PRIsothermalCapillaryFilm


class PRCapillaryGasExchange:
    def __init__(self, film, *, interface_model="plug"):
        if not isinstance(film, PRIsothermalCapillaryFilm):
            raise TypeError("Explicit isothermal PR capillary film required")
        self.film = film
        if interface_model not in ("plug", "stokes_profile"):
            raise ValueError("Explicit plug or stokes_profile interface model required")
        if interface_model == "stokes_profile" and not film.mechanics.drag:
            raise ValueError("Stokes-profile exchange requires the viscous wall-film model")
        self.interface_model = interface_model
        self.model_id = self._identity()

    def _identity(self):
        return _digest(
            {
                "model": "pr_capillary_gas_exchange_v1",
                "film_model_id": self.film.model_id,
                "interface_model": self.interface_model,
            }
        )

    def to_dict(self):
        return {
            "model": "pr_capillary_gas_exchange_v1",
            "film": self.film.to_dict(),
            "interface_model": self.interface_model,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        if (
            set(data) != {"model", "film", "interface_model"}
            or data["model"] != "pr_capillary_gas_exchange_v1"
        ):
            raise ValueError("Known complete gas/film exchange configuration required")
        return cls(
            PRIsothermalCapillaryFilm.from_dict(data["film"], device=device),
            interface_model=data["interface_model"],
        )

    def apply(self, film_state, gas_phases, dt_s, conductance_kg_s):
        """Return film, face-aligned gas phases and conservative source ledger.

        Must be composed once per source substep by the transport integrator.
        Gas phases own physical dual-cell inventories; they are not interpolated
        copies of primal inventories. Inputs are immutable, including on failure.
        """
        m = self.film
        if self.model_id != self._identity():
            raise ValueError("Gas/film exchange configuration identity changed")
        old_film = m.inspect(film_state)
        dt = _number(dt_s, "interface source interval", positive=True)
        f, s = m.fluid, m.mechanics.surface
        gas = tuple(gas_phases)
        count = s.shape[0]
        if len(gas) != count:
            raise ValueError("One physical gas inventory per film dual cell required")
        k = f.tensor(conductance_kg_s)
        if k.shape != (count,) or not bool((torch.isfinite(k) & (k >= 0)).all().cpu()):
            raise ValueError("Finite nonnegative conductance per dual cell required")
        n = f.tensor([p.component_moles for p in gas])
        v = f.tensor([p.volume_m3 for p in gas])
        pg = f.tensor([p.momentum_kg_m_s for p in gas])
        eg = f.tensor([p.total_energy_j for p in gas])
        mg = (n * f.molar_masses).sum(-1)
        # Validate actual N/V/M/E and declared P/T before performing any exchange.
        checked = recover_homogeneous_phases(
            f, n, v, pg, eg, ["vapor"] * count, [p.temperature_k for p in gas]
        )
        for actual, recovered in zip(gas, checked, strict=True):
            if (
                not math.isfinite(actual.pressure_pa)
                or not math.isfinite(actual.temperature_k)
                or abs(actual.pressure_pa - recovered.pressure_pa)
                > 1e-4 + 1e-9 * abs(recovered.pressure_pa)
                or abs(actual.temperature_k - recovered.temperature_k) > 1e-7
            ):
                raise PropertyModelError("Gas caloric checkpoint disagrees with its stocks")
        ul = f.tensor(film_state.mechanics.face_velocity_m_s)
        h = f.tensor(film_state.mechanics.thickness_m)
        ml = s.rho * s.cell_area * m.mechanics.face(h)
        ug = pg / mg
        if self.interface_model == "stokes_profile":
            eta, beta, _ = resistance_response(m.mechanics.face(h), s.mu, s.slip)
        else:
            eta, beta = torch.ones_like(ml), torch.zeros_like(ml)
        # ui=eta*ul+beta*tau, tau=J/(A*dt). Eliminate profile compliance
        # before the two-mass solve. The reciprocal branch retains finite
        # effective conductance even when k*beta/A overflows.
        compliance = beta / s.cell_area
        product = k * compliance
        effective = torch.where(product <= 1, k / (1 + product), 1 / (1 / k + compliance))
        attenuation = torch.where(product <= 1, 1 / (1 + product), (1 / k) / (1 / k + compliance))
        denominator = ml + eta.square() * mg
        liquid_share, gas_share = ml / denominator, mg / denominator
        reduced = mg * liquid_share
        if not bool((torch.isfinite(reduced) & (reduced > 0)).all().cpu()):
            raise PropertyModelError("Interface reduced mass is outside floating-point domain")
        stiffness = dt * effective / reduced
        # Overflow of a positive stiffness is its resolved equilibrium limit,
        # not inf/inf. Report only bounded weights, never a nonfinite stiffness.
        inverse = 1 / stiffness
        fraction = torch.where(stiffness <= 1, stiffness / (1 + stiffness), 1 / (1 + inverse))
        retention = torch.where(stiffness <= 1, 1 / (1 + stiffness), inverse / (1 + inverse))
        drive = ug - eta * ul
        impulse = reduced * fraction * drive
        liquid_impulse = eta * impulse
        wall_impulse = (eta - 1) * impulse
        # Positive weighted sums avoid cancellation of nearly equal gas
        # velocities when a light gas relaxes toward a massive film.
        uln = (retention + fraction * liquid_share) * ul + fraction * eta * gas_share * ug
        ugn = (
            retention + fraction * eta.square() * gas_share
        ) * ug + fraction * eta * liquid_share * ul
        physical = impulse * drive * retention * attenuation
        numerical_gas = 0.5 * impulse.square() / mg
        numerical_liquid = 0.5 * liquid_impulse.square() / ml
        shear = impulse / dt / s.cell_area
        surface_velocity = eta * uln + beta * shear
        liquid_heat = impulse * beta * shear
        work = impulse * surface_velocity
        if not bool(
            (
                torch.isfinite(work)
                & torch.isfinite(physical)
                & (physical >= -1e-24)
                & torch.isfinite(numerical_liquid)
                & torch.isfinite(liquid_heat)
                & (liquid_heat >= 0)
            )
            .all()
            .cpu()
        ):
            raise ConservationError("Nonfinite or nonpassive interface work")
        # Gas loses interface work. Its remaining kinetic-energy loss heats
        # the gas (physical slip heat plus BE temporal heat); no clipping.
        new_gas = recover_homogeneous_phases(
            f,
            n,
            v,
            mg * ugn,
            eg - work,
            ["vapor"] * count,
            [p.temperature_k for p in gas],
        )
        mechanical = replace(
            film_state.mechanics,
            face_velocity_m_s=tuple(uln.cpu().tolist()),
            numerical_heat_j=film_state.mechanics.numerical_heat_j
            + float(numerical_liquid.sum().cpu()),
            physical_heat_j=film_state.mechanics.physical_heat_j + float(liquid_heat.sum().cpu()),
        )
        # This fixed-volume source must not rebuild N/V from a fresh EOS or
        # geometric multiplication (not even by a CPU/CUDA rounding unit).
        # Preserve each original caloric stock and update actual kinetic
        # energy by its momentum increment instead of subtracting bulk totals.
        material_mass = f.tensor([p.component_moles for p in film_state.phases]) @ f.molar_masses
        material_momentum = f.tensor([p.momentum_kg_m_s for p in film_state.phases])
        increment = material_mass * (m.mechanics.center(uln) - m.mechanics.center(ul))
        new_material_momentum = material_momentum + increment
        kinetic_increment = (
            (new_material_momentum - material_momentum)
            * (new_material_momentum + material_momentum)
            / (2 * material_mass)
        )
        material_updates = (
            torch.stack((new_material_momentum, kinetic_increment), -1).cpu().tolist()
        )
        phases = tuple(
            replace(p, momentum_kg_m_s=momentum, total_energy_j=p.total_energy_j + kinetic)
            for p, (momentum, kinetic) in zip(film_state.phases, material_updates, strict=True)
        )
        new_film = replace(film_state, mechanics=mechanical, phases=phases)
        after = m.inspect(new_film)
        # Check local kinetic work directly, before summing caloric energies.
        liquid_error = 0.5 * ml * (uln - ul) * (uln + ul) + numerical_liquid + liquid_heat - work
        gas_error = -work - 0.5 * mg * (ugn - ug) * (ugn + ug) - physical - numerical_gas
        scale = work.abs() + physical.abs() + numerical_gas + numerical_liquid + liquid_heat
        if not bool(((liquid_error.abs() + gas_error.abs()) <= 1e-24 + 1e-10 * scale).all().cpu()):
            raise ConservationError("Local gas/film kinetic-work identity failed")
        energy_residual = math.fsum(
            [
                after["total_energy_j"],
                -old_film["total_energy_j"],
                *(p.total_energy_j for p in new_gas),
                *(-p.total_energy_j for p in gas),
            ]
        )
        allowance = (
            128
            * math.fsum(
                math.ulp(p.total_energy_j)
                for phases in (gas, new_gas, film_state.phases, new_film.phases)
                for p in phases
            )
            + 1e-22
        )
        momentum_residual = math.fsum(
            [
                after["momentum_kg_m_s"],
                -old_film["momentum_kg_m_s"],
                *(p.momentum_kg_m_s for p in new_gas),
                *(-p.momentum_kg_m_s for p in gas),
                -float(wall_impulse.sum().cpu()),
            ]
        )
        momentum_scale = float((pg.abs() + (ml * ul).abs() + impulse.abs()).sum().cpu())
        if (
            abs(energy_residual) > allowance
            or abs(momentum_residual) > 1e-24 + 1e-11 * momentum_scale
        ):
            raise ConservationError("Coupled gas/film source ledger failed")
        return (
            new_film,
            new_gas,
            {
                "dt_s": dt,
                "clock_advanced": False,
                "interface_model": self.interface_model,
                "model_id": self.model_id,
                "gas_to_liquid_impulse_kg_m_s": impulse.cpu().tolist(),
                "liquid_net_impulse_kg_m_s": liquid_impulse.cpu().tolist(),
                "stationary_wall_impulse_kg_m_s": wall_impulse.cpu().tolist(),
                "interface_velocity_m_s": surface_velocity.cpu().tolist(),
                "interface_shear_pa": shear.cpu().tolist(),
                "relaxation_weight": fraction.cpu().tolist(),
                "gas_to_liquid_work_j": work.cpu().tolist(),
                "gas_physical_heat_j": physical.cpu().tolist(),
                "gas_numerical_heat_j": numerical_gas.cpu().tolist(),
                "liquid_numerical_heat_j": numerical_liquid.cpu().tolist(),
                "liquid_profile_heat_j": liquid_heat.cpu().tolist(),
                "local_liquid_work_residual_j": liquid_error.cpu().tolist(),
                "local_gas_heat_residual_j": gas_error.cpu().tolist(),
                "energy_residual_j": energy_residual,
                "energy_roundoff_allowance_j": allowance,
                "momentum_residual_kg_m_s": momentum_residual,
                "qualification": "Local fixed-volume interface shear; supplied conductance; declared plug or Stokes-profile response; no normal-pressure coupling or empirical qualification",
            },
        )
