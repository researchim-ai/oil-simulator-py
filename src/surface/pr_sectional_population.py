"""Fixed-pivot spectrum with an untruncated upper-tail number/volume pair.

Reference droplet volumes dilate uniformly with the live dispersed EOS volume.
Aggregation preserves reference M1 and removes one particle per collision;
breakup conserves M1 and adds the declared daughter count minus one per event. The tail is a two-moment
closure, not a resolved distribution beyond the last pivot.
"""

import math
from dataclasses import dataclass

import torch

from .liquid import _number
from .pr_droplet_population import PRDropletPopulation, PRPoweredDropletPopulation


@dataclass(frozen=True, kw_only=True)
class PRSectionalDropletPopulation(PRPoweredDropletPopulation):
    reference_volumes_m3: tuple
    initial_volume_fractions: tuple
    section_breakup_rates_s: tuple
    daughter_volume_fractions: tuple = (0.5, 0.5)
    coalescence_model: str = "constant"
    collision_efficiency: float = 1.0
    tail_initial_volume_ratio: float = 2.0
    maximum_tail_volume_ratio: float = 1e6
    breakup_rate_s: float = 0.0
    model: str = "sectional_droplet_population"

    def __post_init__(self):
        PRDropletPopulation.__post_init__(self)
        for key in (
            "reference_volumes_m3",
            "initial_volume_fractions",
            "section_breakup_rates_s",
            "daughter_volume_fractions",
        ):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        v, weights, rates = (
            self.reference_volumes_m3,
            self.initial_volume_fractions,
            self.section_breakup_rates_s,
        )
        if self.model != "sectional_droplet_population" or not 3 <= len(v) <= 16:
            raise ValueError("Sectional population requires 3..16 pivots including the upper tail")
        if len(weights) != len(v) or len(rates) != len(v):
            raise ValueError("Spectrum, initial distribution and breakup rates differ")
        for value in v:
            _number(value, "reference volume", positive=True)
        if any(a >= b for a, b in zip(v, v[1:])):
            raise ValueError("Strictly increasing sectional pivots required")
        for value in weights:
            _number(value, "initial volume fraction", positive=True)
        if not math.isclose(math.fsum(weights), 1.0, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Initial spectrum volume fractions must sum to one")
        daughters = self.daughter_volume_fractions
        if not 2 <= len(daughters) <= 16 or not math.isclose(
            math.fsum(daughters), 1.0, abs_tol=1e-12, rel_tol=0
        ):
            raise ValueError("Conservative daughter volume fractions required")
        for fraction in daughters:
            _number(fraction, "daughter volume fraction", positive=True)
        if self.coalescence_model not in ("constant", "brownian_gravity"):
            raise ValueError("Unknown sectional collision kernel")
        _number(self.collision_efficiency, "collision efficiency", nonnegative=True)
        if self.collision_efficiency > 1:
            raise ValueError("Collision efficiency must not exceed one")
        if self.coalescence_model == "constant" and self.collision_efficiency != 1:
            raise ValueError("A prescribed effective kernel already includes collision efficiency")
        if self.coalescence_model != "constant" and self.coalescence_kernel_m3_s:
            raise ValueError("Select a prescribed kernel or a physical kernel, not both")
        for volume, rate in zip(v, rates):
            _number(rate, "section breakup rate", nonnegative=True)
            if rate and volume * min(daughters) < v[0] * (1 - 1e-12):
                raise ValueError("Daughters must remain above the smallest pivot")
        if self.breakup_rate_s != 0:
            raise ValueError("Use section_breakup_rates_s, not a second scalar breakup law")
        _number(self.agitation_power_w, "mechanical mixer power", nonnegative=True)
        if any(rates) and self.agitation_power_w <= 0:
            raise ValueError("Breakup requires explicit positive mechanical power")
        if not 1 <= self.tail_initial_volume_ratio <= self.maximum_tail_volume_ratio:
            raise ValueError("Invalid declared tail volume range")
        _number(self.maximum_tail_volume_ratio, "tail volume limit", positive=True)

    @property
    def width(self):
        return len(self.reference_volumes_m3) + 1

    def properties(self, law, phase_volumes, x, number, q=None):
        base = PRDropletPopulation.properties(self, law, phase_volumes, x, None, q)
        volume, present = base["drop_volume_m3"], base["present"]
        pivots = volume.new_tensor(self.reference_volumes_m3)
        if number is None:
            fractions = volume.new_tensor(self.initial_volume_fractions)
            representatives = torch.cat((pivots[:-1], pivots[-1:] * self.tail_initial_volume_ratio))
            counts = volume * fractions / representatives
            number = torch.cat((counts, (volume * fractions[-1]).reshape(1)))
        counts, tail_volume = number[:-1], number[-1]
        tail_mean = torch.where(
            counts[-1] > 1e-30, tail_volume / counts[-1].clamp_min(1e-30), pivots[-1]
        )
        representatives = torch.cat((pivots[:-1], tail_mean.reshape(1)))
        reference_parts = torch.cat((counts[:-1] * pivots[:-1], tail_volume.reshape(1)))
        reference_total = reference_parts.sum()
        fraction = reference_parts / reference_total.clamp_min(1e-100)
        particle_volumes = volume * representatives / reference_total.clamp_min(1e-100)
        diameters = torch.where(
            present,
            (6 * particle_volumes / math.pi).clamp_min(1e-300).pow(1 / 3),
            volume.new_full(counts.shape, law.droplet_diameter_m),
        )
        surfaces = law.surface_tension_n_m * math.pi * counts * diameters.square()
        surfaces = torch.where(present, surfaces, torch.zeros_like(surfaces))
        effective = torch.where(
            present,
            (fraction * diameters.square()).sum().clamp_min(1e-300).sqrt(),
            volume.new_tensor(law.droplet_diameter_m),
        )
        valid = (
            base["valid"]
            & (counts >= 0).all()
            & (tail_volume >= 0)
            & torch.where(
                present,
                (counts.sum() > 0) & (reference_total > 0),
                (counts == 0).all() & (tail_volume == 0),
            )
            & (tail_volume >= counts[-1] * pivots[-1] * (1 - 1e-10))
            & ((counts[-1] > 0) | (tail_volume == 0))
            & (tail_mean <= pivots[-1] * self.maximum_tail_volume_ratio)
            & torch.isfinite(diameters).all()
        )
        return {
            **base,
            "number": counts.sum(),
            "diameter_m": effective,
            "surface_energy_j": surfaces.sum(),
            "valid": valid,
            "coordinates": number,
            "section_numbers": counts,
            "section_volume_fractions": fraction,
            "section_diameters_m": diameters,
            "section_surface_energy_j": surfaces,
            "reference_representatives_m3": representatives,
            "reference_volume_m3": reference_total,
            "tail_reference_volume_m3": tail_volume,
        }

    def weights(self, volumes):
        pivots = volumes.new_tensor(self.reference_volumes_m3)
        weights = []
        for i in range(len(pivots) - 1):
            right = (pivots[i + 1] - volumes) / (pivots[i + 1] - pivots[i])
            value = (
                right
                if i == 0
                else torch.minimum(right, (volumes - pivots[i - 1]) / (pivots[i] - pivots[i - 1]))
            )
            weights.append(value.clamp(0, 1))
        regular = torch.stack(weights, -1)
        tail = (1 - regular.sum(-1)).clamp_min(0)
        tail_volume = volumes - (regular * pivots[:-1]).sum(-1)
        return torch.cat((regular, tail[..., None]), -1), tail_volume

    def sources(
        self,
        coordinates,
        cell_volume,
        *,
        population=None,
        temperature_k=None,
        carrier_viscosity_pa_s=None,
        brownian_mobility_factor=1.0,
    ):
        counts, tail = coordinates[:-1], coordinates[-1]
        pivots = counts.new_tensor(self.reference_volumes_m3)
        representatives = torch.cat(
            (
                pivots[:-1],
                torch.where(
                    counts[-1] > 1e-30, tail / counts[-1].clamp_min(1e-30), pivots[-1]
                ).reshape(1),
            )
        )
        product = representatives[:, None] + representatives[None, :]
        weights, tail_gain = self.weights(product)
        if self.coalescence_model == "constant":
            kernel = counts.new_full((len(counts), len(counts)), self.coalescence_kernel_m3_s)
        else:
            if population is None or temperature_k is None or carrier_viscosity_pa_s is None:
                raise ValueError(
                    "Physical collision kernel requires live sizes, velocities and temperature"
                )
            diameter = population["section_diameters_m"]
            velocity = population["section_velocities_m_s"]
            pair = diameter[:, None] + diameter[None, :]
            brownian = (
                2
                * 1.380649e-23
                * temperature_k
                * brownian_mobility_factor
                / (3 * carrier_viscosity_pa_s)
                * pair.square()
                / (diameter[:, None] * diameter[None, :]).clamp_min(1e-100)
            )
            gravity = math.pi / 4 * pair.square() * (velocity[:, None] - velocity[None, :]).abs()
            kernel = self.collision_efficiency * (brownian + gravity)
        events = kernel * counts[:, None] * counts[None, :] / (2 * cell_volume)
        loss = kernel @ counts / cell_volume
        coal_n = (events[..., None] * weights).sum((0, 1)) - counts * loss
        coal_v = (events * tail_gain).sum() - tail * loss[-1]
        rates = counts.new_tensor(self.section_breakup_rates_s) * counts
        child = (
            representatives[:, None] * counts.new_tensor(self.daughter_volume_fractions)[None, :]
        ).clamp_min(self.reference_volumes_m3[0])
        weights, tail_gain = self.weights(child)
        break_n = (rates[:, None, None] * weights).sum((0, 1)) - rates
        break_v = (rates[:, None] * tail_gain).sum() - self.section_breakup_rates_s[-1] * tail
        return torch.cat((coal_n, coal_v.reshape(1))), torch.cat((break_n, break_v.reshape(1)))

    def surface_at(self, coordinates, live_volume, sigma):
        counts, tail = coordinates[:-1], coordinates[-1]
        pivots = counts.new_tensor(self.reference_volumes_m3)
        total = (counts[:-1] * pivots[:-1]).sum() + tail
        scale = live_volume / total.clamp_min(1e-100)
        regular = (counts[:-1] * (scale * pivots[:-1]).clamp_min(1e-300).pow(2 / 3)).sum()
        upper = counts[-1].clamp_min(0).pow(1 / 3) * (scale * tail).clamp_min(0).pow(2 / 3)
        return sigma * (36 * math.pi) ** (1 / 3) * (regular + upper)


def endpoint_data(data):
    """Expand the dispersed phase into size-resolved Riemann packets."""
    pop = data.get("population")
    if pop is None or "section_velocities_m_s" not in pop:
        return data
    q, x = data["q"], data["x"]
    n = len(x)
    count = len(pop["section_numbers"])
    drop_index = pop["drop_mask"].argmax()
    index = torch.cat((torch.arange(n, device=x.device), drop_index.expand(count)))
    alpha_drop = (data["alpha"] * pop["drop_mask"]).sum()
    alpha = torch.cat(
        (data["alpha"] * (1 - pop["drop_mask"]), alpha_drop * pop["section_volume_fractions"])
    )
    velocity = torch.cat((data["phase_velocity"], pop["section_velocities_m_s"]))
    return {
        **data,
        "q": {k: v.index_select(0, index) for k, v in q.items()},
        "x": x.index_select(0, index),
        "alpha": alpha,
        "phase_velocity": velocity,
        "drift": velocity - data["u"],
        "population": {
            **pop,
            "drop_mask": torch.cat((pop["drop_mask"] * 0, pop["drop_mask"].new_ones(count))),
            "surface_density_phase": torch.cat(
                (
                    data["alpha"] * 0,
                    alpha_drop
                    * pop["section_surface_energy_j"]
                    / pop["drop_volume_m3"].clamp_min(1e-100),
                )
            ),
        },
    }


def spectral_flux(base_flux, contact, left, right, law, left_native, right_native, spec):
    """Per-section count and upper-tail reference-volume flux, with common fan."""

    def concentration(packet, native):
        if native is None:
            roles, _ = spec.roles(
                law, packet["phase_x"], packet["phase_z"], packet["phase_densities"]
            )
            mask = roles[1] if law.dispersed_liquid == "oil" else roles[2]
            weight = (packet["phase_weights"] * mask).sum()
            fractions = weight.new_tensor(spec.initial_volume_fractions)
            pivots = weight.new_tensor(spec.reference_volumes_m3)
            representatives = torch.cat((pivots[:-1], pivots[-1:] * spec.tail_initial_volume_ratio))
            concentration = weight * torch.cat((fractions / representatives, fractions[-1:]))
            velocity = (packet["actual_velocity"] * mask).sum()
            return concentration, velocity.expand_as(pivots)
        mask = packet["phase_weights"][-len(native["section_numbers"]) :]
        # Sum over packets is the local dispersed fraction; each section gets
        # its own particle concentration, not the phase-average number flux.
        factor = mask.sum() / native["drop_volume_m3"].clamp_min(1e-100)
        return factor * native["coordinates"], packet["actual_velocity"][
            -len(native["section_numbers"]) :
        ]

    a, va = concentration(left, left_native)
    b, vb = concentration(right, right_native)
    va, vb = torch.cat((va, va[-1:])), torch.cat((vb, vb[-1:]))
    donor = torch.where(contact >= 0, a, b)
    total = torch.where(
        contact >= 0, left["phase_concentration"].sum(), right["phase_concentration"].sum()
    )
    return (
        base_flux[:-2].sum() / total.clamp_min(1e-100) * donor
        + a * (va.clamp_min(0) - left["bulk_velocity"].clamp_min(0))
        + b * (vb.clamp_max(0) - right["bulk_velocity"].clamp_max(0))
    )


def characteristics(model, index, data, active):
    """Constrained local spectrum/EOS flux tangent, including size velocities."""
    from .liquid import HydraulicConvergenceError, PropertyModelError

    f, node = model.fluid, model.nodes[index]
    phases, components = data["amounts"].shape
    pop = data["population"]
    carrying = bool(pop["present"].cpu())
    scales = pop["coordinates"].abs().clamp_min(1e-20)
    prefix = phases * components
    y0 = torch.cat(
        (
            data["amounts"].log().flatten(),
            data["p"].log().reshape(1),
            (data["t"] / f.tref).reshape(1),
            (data["u"] / model.maximum_velocity_m_s).reshape(1),
            pop["coordinates"] / scales if carrying else pop["coordinates"].new_zeros(0),
        )
    )

    def quantities(y):
        coordinates = y[prefix + 3 :] * scales if carrying else pop["coordinates"]
        d = model._decode(
            index,
            y[:prefix].exp().reshape(phases, components),
            y[prefix].exp(),
            y[prefix + 1] * f.tref,
            y[prefix + 2] * model.maximum_velocity_m_s,
            active,
            data["tags"],
            coordinates,
        )
        ep = endpoint_data(d)
        q, x, alpha, velocity = ep["q"], ep["x"], ep["alpha"], ep["phase_velocity"]
        mu = (
            d["x"][:, active].clamp_min(1e-300).log()
            + d["q"]["log_fugacity_coefficients"][:, active]
        )
        conserved = torch.cat(
            (
                d["n"][active] / node.volume_m3,
                (d["momentum"] / node.volume_m3).reshape(1),
                (d["energy"] / node.volume_m3).reshape(1),
                coordinates / scales / node.volume_m3 if carrying else coordinates.new_zeros(0),
                (mu[:-1] - mu[-1]).flatten(),
                (d["volume"] / node.volume_m3 - 1).reshape(1),
            )
        )
        component_flux = (
            alpha[:, None] * x / q["molar_volume_m3_mol"][:, None] * velocity[:, None]
        ).sum(0)
        momentum_flux = (alpha * q["density_kg_m3"] * velocity.square()).sum() + d["p"]
        mass = (x * f.molar_masses).sum(-1)
        phi = model.gravity_m_s2 * (node.elevation_m - model.reference_elevation_m)
        energy_flux = (
            alpha
            / q["molar_volume_m3_mol"]
            * (q["molar_enthalpy_j_mol"] + mass * (0.5 * velocity.square() + phi))
            * velocity
        ).sum()
        energy_flux = energy_flux + (ep["population"]["surface_density_phase"] * velocity).sum()
        section_velocity = d["population"]["section_velocities_m_s"]
        particle_flux = (
            coordinates
            / scales
            / node.volume_m3
            * torch.cat((section_velocity, section_velocity[-1:]))
        )
        flux = torch.cat(
            (
                component_flux[active],
                momentum_flux.reshape(1),
                energy_flux.reshape(1),
                particle_flux if carrying else particle_flux.new_zeros(0),
            )
        )
        return conserved, flux

    a = torch.func.jacfwd(lambda y: quantities(y)[0])(y0)
    b = torch.func.jacfwd(lambda y: quantities(y)[1])(y0)
    count = components + 2 + (len(scales) if carrying else 0)
    rhs = y0.new_zeros((len(y0), count))
    rhs[:count] = torch.eye(count, dtype=y0.dtype, device=y0.device)
    try:
        speeds = torch.linalg.eigvals(b @ torch.linalg.solve(a, rhs))
    except RuntimeError as exc:
        raise HydraulicConvergenceError("Singular sectional transport tangent") from exc
    if not bool(torch.isfinite(speeds).all().cpu()) or float(
        speeds.imag.abs().max().cpu()
    ) > 1e-7 * max(1.0, float(speeds.real.abs().max().cpu())):
        raise PropertyModelError("Sectional transport loses local hyperbolicity")
    if float(speeds.real.abs().max().cpu()) > model.wave_speed_bound_m_s:
        raise PropertyModelError("Sectional wave speed exceeds configured bound")
    return speeds.real.detach()


def source_context(law, population, temperature):
    carrier = law.water_viscosity_pa_s if law.dispersed_liquid == "oil" else law.oil_viscosity_pa_s
    dispersed = (
        law.oil_viscosity_pa_s if law.dispersed_liquid == "oil" else law.water_viscosity_pa_s
    )
    mobility = (
        3 * (carrier + dispersed) / (2 * carrier + 3 * dispersed)
        if law.interface == "clean"
        else 1.0
    )
    return {
        "population": population,
        "temperature_k": temperature,
        "carrier_viscosity_pa_s": carrier,
        "brownian_mobility_factor": mobility,
    }
