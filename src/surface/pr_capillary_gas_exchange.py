"""Two-way local shear between PR gas stocks and staggered PR film faces.

A frozen nonnegative conductance is supplied explicitly (kg/s per dual cell),
not fitted or inferred from an unqualified entrainment correlation. The liquid
plug velocity is the interface velocity. Fixed-volume source only: no gas
transport, normal pressure work, phase transfer or report-clock advancement.
"""

import math
from dataclasses import replace

import torch

from .liquid import ConservationError, PropertyModelError, _number
from .pr_homogeneous_uv import recover_homogeneous_phases
from .pr_isothermal_capillary_film import PRIsothermalCapillaryFilm


class PRCapillaryGasExchange:
    def __init__(self, film):
        if not isinstance(film, PRIsothermalCapillaryFilm):
            raise TypeError("Explicit isothermal PR capillary film required")
        self.film = film

    def apply(self, film_state, gas_phases, dt_s, conductance_kg_s):
        """Return film, face-aligned gas phases and conservative source ledger.

        Must be composed once per source substep by the transport integrator.
        Gas phases own physical dual-cell inventories; they are not interpolated
        copies of primal inventories. Inputs are immutable, including on failure.
        """
        m = self.film
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
        # J = dt*k*(ug_new-ul_new). Stable reduced-mass form avoids
        # subtracting two nearly equal momenta in the stiff limit.
        reduced = mg * (ml / (mg + ml))
        stiffness = dt * k / reduced
        if not bool(torch.isfinite(stiffness).all().cpu()):
            raise PropertyModelError("Interface stiffness exceeds floating-point domain")
        fraction = stiffness / (1 + stiffness)
        impulse = reduced * fraction * (ug - ul)
        uln = ul + impulse / ml
        ugn = ug - impulse / mg
        physical = impulse * ((ug - ul) / (1 + stiffness))
        numerical_gas = 0.5 * impulse.square() / mg
        numerical_liquid = 0.5 * impulse.square() / ml
        work = impulse * uln
        if not bool(
            (
                torch.isfinite(work)
                & torch.isfinite(physical)
                & (physical >= -1e-24)
                & torch.isfinite(numerical_liquid)
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
        )
        new_film = replace(film_state, mechanics=mechanical, phases=m._material(mechanical))
        after = m.inspect(new_film)
        # Check local kinetic work directly, before summing caloric energies.
        liquid_error = 0.5 * ml * (uln - ul) * (uln + ul) + numerical_liquid - work
        gas_error = -work - 0.5 * mg * (ugn - ug) * (ugn + ug) - physical - numerical_gas
        scale = work.abs() + physical.abs() + numerical_gas + numerical_liquid
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
                "gas_to_liquid_impulse_kg_m_s": impulse.cpu().tolist(),
                "gas_to_liquid_work_j": work.cpu().tolist(),
                "gas_physical_heat_j": physical.cpu().tolist(),
                "gas_numerical_heat_j": numerical_gas.cpu().tolist(),
                "liquid_numerical_heat_j": numerical_liquid.cpu().tolist(),
                "local_liquid_work_residual_j": liquid_error.cpu().tolist(),
                "local_gas_heat_residual_j": gas_error.cpu().tolist(),
                "energy_residual_j": energy_residual,
                "energy_roundoff_allowance_j": allowance,
                "momentum_residual_kg_m_s": momentum_residual,
                "qualification": "Local fixed-volume plug-interface shear; supplied conductance; no normal-pressure coupling",
            },
        )
