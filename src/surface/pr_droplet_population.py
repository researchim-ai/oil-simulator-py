"""Conservative M0/M1 monodisperse closure with Smoluchowski coalescence.

M1 is the live equilibrium volume of the dispersed liquid; M0 is an additional
Newton unknown. Surface energy is carried by particles and retained during
capture/coalescence. Drop size changes the Stokes/HR velocity, not just output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .liquid import EquipmentOperatingError, _name, _number


@dataclass(frozen=True)
class PRDropletPopulation:
    family: str
    coalescence_kernel_m3_s: float
    maximum_number_density_m3: float = 1e24
    gas_compressibility_threshold: float = 0.5

    def __post_init__(self):
        _name(self.family)
        _number(self.coalescence_kernel_m3_s, "coalescence kernel", nonnegative=True)
        _number(self.maximum_number_density_m3, "maximum droplet density", positive=True)
        if not 0 < self.gas_compressibility_threshold <= 1:
            raise ValueError("Explicit gas compressibility classification threshold required")

    def roles(self, law, x, z, densities):
        light = torch.nn.functional.one_hot(densities.argmin(), len(densities)).to(x.dtype)
        gas = light * ((z * light).sum() >= self.gas_compressibility_threshold).to(x.dtype)
        if len(densities) > 1:
            gas = gas * (
                densities.min() / densities.max() <= law.maximum_gas_liquid_density_ratio
            ).to(x.dtype)
        scores = x[:, [law.oil_component_index, law.water_component_index]]
        liquids = torch.nn.functional.one_hot(scores.argmax(-1), 2).to(x.dtype) * (1 - gas)[:, None]
        oil, water = liquids.unbind(-1)
        valid = (
            ((scores.max(-1).values >= law.minimum_marker_fraction) | (gas > 0)).all()
            & (oil.sum() <= 1)
            & (water.sum() <= 1)
        )
        return (gas, oil, water), valid

    @staticmethod
    def masks(law, x):
        roles, valid = law.roles(x)
        _gas, oil, water = roles
        drop, carrier = (oil, water) if law.dispersed_liquid == "oil" else (water, oil)
        present = (drop.sum() > 0) & (carrier.sum() > 0)
        return drop, present, valid

    def properties(self, law, phase_volumes, x, number, q=None):
        roles = (
            law.roles(x)
            if q is None
            else self.roles(law, x, q["compressibility_factor"], q["density_kg_m3"])
        )
        _gas, oil, water = roles[0]
        drop, carrier = (oil, water) if law.dispersed_liquid == "oil" else (water, oil)
        present = (drop.sum() > 0) & (carrier.sum() > 0)
        valid = roles[1]
        volume = torch.where(present, (phase_volumes * drop).sum(), phase_volumes.sum() * 0)
        initial_volume = math.pi * law.droplet_diameter_m**3 / 6
        if number is None:
            number = torch.where(present, volume / initial_volume, volume * 0)
        diameter = torch.where(
            present,
            (6 * volume / (math.pi * number.clamp_min(1e-30))).clamp_min(1e-300).pow(1 / 3),
            volume.new_tensor(law.droplet_diameter_m),
        )
        surface = torch.where(
            present, law.surface_tension_n_m * math.pi * number * diameter.square(), volume * 0
        )
        valid = (
            (number == 0) if len(x) == 1 else valid & torch.where(present, number > 0, number == 0)
        )
        valid = valid & torch.isfinite(surface)
        return {
            "number": number,
            "diameter_m": diameter,
            "surface_energy_j": surface,
            "drop_volume_m3": volume,
            "drop_mask": drop,
            "present": present,
            "valid": valid,
            "roles": roles,
        }


@dataclass(frozen=True, kw_only=True)
class PRPoweredDropletPopulation(PRDropletPopulation):
    """Monomoment fragmentation with an explicit mechanical-power budget.

    dN/dt=rate*N represents count increase; M1 stays the live liquid inventory.
    It is not a resolved daughter-size distribution. All new interface energy
    is taken from the declared mixer work in the same total-energy equation.
    """

    breakup_rate_s: float
    agitation_power_w: float
    model: str = "powered_monomoment_breakup"

    def __post_init__(self):
        super().__post_init__()
        _number(self.breakup_rate_s, "breakup rate", nonnegative=True)
        _number(self.agitation_power_w, "mechanical agitation power", positive=True)
        if self.model != "powered_monomoment_breakup":
            raise ValueError("Unknown powered population model")


def population_from_dict(data):
    return (
        PRPoweredDropletPopulation(**data)
        if data.get("model") == "powered_monomoment_breakup"
        else PRDropletPopulation(**data)
    )


def marker_index(law):
    return law.oil_component_index if law.dispersed_liquid == "oil" else law.water_component_index


def packet_density(packet, law, native, spec):
    """Count concentration at an opening, with explicit incoming drop size."""
    phase = packet["phase_x"]
    index = marker_index(law)
    roles, _valid = spec.roles(law, phase, packet["phase_z"], packet["phase_densities"])
    mask = (roles[1] if law.dispersed_liquid == "oil" else roles[2]).to(torch.bool)
    mask = mask & (phase[:, index] >= law.minimum_marker_fraction)
    weight = (packet["phase_weights"] * mask).sum()
    if native is None:
        density = weight / (math.pi * law.droplet_diameter_m**3 / 6)
    else:
        density = weight * native["number"] / native["drop_volume_m3"].clamp_min(1e-30)
    speed = (packet["actual_velocity"] * mask).sum()
    return density, speed


def number_flux(base_flux, contact, left, right, law, left_native, right_native, spec):
    a, va = packet_density(left, law, left_native, spec)
    b, vb = packet_density(right, law, right_native, spec)
    concentration_left = left["phase_concentration"].sum()
    concentration_right = right["phase_concentration"].sum()
    n_up = torch.where(contact >= 0, a, b)
    concentration = torch.where(contact >= 0, concentration_left, concentration_right)
    advective = base_flux[:-2].sum() / concentration.clamp_min(1e-30) * n_up

    def directed(v, positive):
        return v.clamp_min(0) if positive else v.clamp_max(0)

    return (
        advective
        + a * (directed(va, True) - directed(left["bulk_velocity"], True))
        + b * (directed(vb, False) - directed(right["bulk_velocity"], False))
    )


class DropletPopulations:
    def __init__(self, model, state, seeds, offset, previous_data=None):
        self.model = model
        self.entries = []
        self.initial = []
        self.lower = []
        self.upper = []
        f = model.fluid
        self.by_cell = {}
        for i, (node, (_, old), seed) in enumerate(zip(model.nodes, state.nodes, seeds)):
            if node.droplet_population is None:
                continue
            number = old.droplet_number
            if number is None:
                raise ValueError("Droplet checkpoint has no particle inventory")
            law = node.drift_flux
            spec = node.droplet_population
            if isinstance(spec, PRPoweredDropletPopulation) and previous_data is not None:
                required = (
                    spec.breakup_rate_s
                    * float(previous_data[i]["population"]["surface_energy_j"].cpu())
                    / 3
                )
                if spec.agitation_power_w < required * (1 - 1e-12):
                    raise EquipmentOperatingError(
                        "Prescribed breakup rate exceeds the available mechanical surface-creation power"
                    )
            guess = number
            if seed["phase_count"] > 1 and not number:
                volumes = f.tensor(
                    [float(p["molar_fraction"] * p["molar_volume_m3_mol"]) for p in seed["phases"]]
                )
                compositions = torch.stack([p["mole_fractions"] for p in seed["phases"]])
                pop = node.droplet_population.properties(
                    law, volumes / volumes.sum() * node.volume_m3, compositions, None
                )
                guess = float(pop["number"].cpu())
            scale = max(1.0, number, guess)
            entry = (i, offset, scale, number)
            self.entries.append(entry)
            self.by_cell[i] = entry
            self.initial.append(f.tensor([guess / scale]))
            self.lower.append(f.tensor([0.0]))
            self.upper.append(
                f.tensor(
                    [node.droplet_population.maximum_number_density_m3 * node.volume_m3 / scale]
                )
            )
            offset += 1

    def numbers(self, y):
        return {i: y[offset] * scale for i, offset, scale, _ in self.entries}

    def balances(self, data, records, dt):
        net = {i: data[i]["p"] * 0 for i in self.by_cell}
        for record in records:
            for i, sign in ((record["source_index"], -1.0), (record["target_index"], 1.0)):
                if i in net:
                    family = self.model.nodes[i].droplet_population.family
                    net[i] = net[i] + sign * record["droplet_number_fluxes"][family]
        rows = []
        details = {}
        for i, _offset, scale, old in self.entries:
            node = self.model.nodes[i]
            pop = data[i]["population"]
            number = pop["number"]
            loss = (
                0.5
                * node.droplet_population.coalescence_kernel_m3_s
                * number.square()
                / node.volume_m3
            )
            spec = node.droplet_population
            breakup = (
                spec.breakup_rate_s * number
                if isinstance(spec, PRPoweredDropletPopulation)
                else number * 0
            )
            standard = (number - old - dt * (net[i] - loss + breakup)) / scale
            initial_volume = math.pi * node.drift_flux.droplet_diameter_m**3 / 6
            nucleation = (number - pop["drop_volume_m3"] / initial_volume) / scale
            active = torch.where((net[i] > 0) | (old > 0), standard, nucleation)
            row = torch.where(pop["present"], active, number / scale)
            rows.append(row.reshape(1))
            details[node.name] = {
                **pop,
                "net_number_rate_s": net[i],
                "coalescence_loss_rate_s": loss,
                "balance_residual": row,
            }
            if isinstance(spec, PRPoweredDropletPopulation):
                fraction = spec.breakup_rate_s * dt
                # Counterfactual area before this implicit fragmentation source,
                # at the same live dispersed volume. log1p/expm1 avoids loss at
                # short steps; fraction>=1 has no positive backward-Euler stock.
                area_factor = -math.expm1(math.log1p(-fraction) / 3) if fraction < 1 else math.inf
                required = torch.where(
                    pop["present"],
                    pop["surface_energy_j"] * area_factor / dt,
                    pop["surface_energy_j"] * 0,
                )
                budget = required <= spec.agitation_power_w * (1 + 1e-10)
                active = pop["present"] & ((net[i] > 0) | (old > 0))
                details[node.name].update(
                    breakup_gain_rate_s=breakup,
                    agitation_power_w=number.new_tensor(spec.agitation_power_w),
                    breakup_surface_power_w=required,
                    power_budget_valid=torch.where(active, budget, True),
                )
        return rows, details

    @staticmethod
    def pattern(model, offsets, entries, openings):
        support = set()
        aux = {i: offset for i, offset, *_ in entries}
        for i, offset, *_ in entries:
            neighbours = {i}
            for link, (a, b) in zip(model.links, model.ends):
                if openings[link.name] and i in (a, b):
                    neighbours.update((a, b))
            columns = [offset]
            for j in neighbours:
                columns.extend(range(offsets[j], offsets[j + 1]))
                if j in aux:
                    columns.append(aux[j])
            support.update((offset, col) for col in columns)
            for j in neighbours:
                support.update((row, offset) for row in range(offsets[j], offsets[j + 1]))
        return support

    @staticmethod
    def serialize(details):
        result = {}
        for name, row in details.items():
            keys = (
                "number",
                "diameter_m",
                "surface_energy_j",
                "drop_volume_m3",
                "net_number_rate_s",
                "coalescence_loss_rate_s",
                "balance_residual",
            )
            if "breakup_gain_rate_s" in row:
                keys = (
                    *keys,
                    "breakup_gain_rate_s",
                    "agitation_power_w",
                    "breakup_surface_power_w",
                )
            values = torch.stack([row[k] for k in keys]).detach().cpu().tolist()
            result[name] = dict(zip(keys, values))
        return result
