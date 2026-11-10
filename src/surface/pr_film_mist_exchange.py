"""Local PR film/drop exchange with explicitly financed surface creation.

Prescribed gross volume rates, deposition followed by entrainment. This is
an exchange operator, not an empirical onset/rate law or a network solver.
Monodisperse fixed diameter is defined for this local fixed-volume source.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import torch

from .film_wetting import LongWaveWettingPotential
from .liquid import ConservationError, PropertyModelError, _name, _number
from .pr_homogeneous_uv import recover_homogeneous_phases
from .pr_stock_balances import entropy_allowance


@dataclass(frozen=True)
class PRFilmMistExchange:
    film_slot: str
    mist_slot: str
    gas_slot: str
    droplet_diameter_m: float
    surface_tension_n_m: float
    atomization_efficiency: float
    heat_to_mist_fraction: float
    film_surface_model: str
    model: str = "pr_film_mist_exchange_v1"
    wetting: dict | None = None

    def __post_init__(self):
        for n in (self.film_slot, self.mist_slot, self.gas_slot):
            _name(n)
        if len({self.film_slot, self.mist_slot, self.gas_slot}) != 3:
            raise ValueError("Three distinct morphology slots required")
        _number(self.droplet_diameter_m, "drop diameter", positive=True)
        _number(self.surface_tension_n_m, "surface tension", positive=True)
        _number(self.atomization_efficiency, "atomization efficiency", positive=True)
        _number(self.heat_to_mist_fraction, "heat partition", nonnegative=True)
        if self.film_surface_model not in ("planar_complete_wetting", "planar_long_wave_wetting"):
            raise ValueError("Explicit planar complete or long-wave wetting model required")
        if self.film_surface_model == "planar_complete_wetting":
            if self.wetting is not None:
                raise ValueError(
                    "Complete-wetting reservoir cannot carry a material wall potential"
                )
        else:
            if not isinstance(self.wetting, dict) or set(self.wetting) - {"dry_extension"} != {
                "film_width_m",
                "adsorption_thickness_m",
                "equilibrium_contact_angle_deg",
            }:
                raise ValueError(
                    "Long-wave wetting requires width, adsorption length and contact angle"
                )
            _number(self.wetting["film_width_m"], "film width", positive=True)
            self._potential()
        volume = (
            math.pi
            / 6
            * self.droplet_diameter_m
            * self.droplet_diameter_m
            * self.droplet_diameter_m
        )
        if (
            not math.isfinite(volume)
            or volume <= 0
            or not math.isfinite(6 * self.surface_tension_n_m / self.droplet_diameter_m)
        ):
            raise ValueError("Unrepresentable droplet volume or surface energy density")
        if not 0 < self.atomization_efficiency <= 1 or not 0 <= self.heat_to_mist_fraction <= 1:
            raise ValueError("Explicit efficiency in (0,1] and heat partition in [0,1] required")
        if self.model != "pr_film_mist_exchange_v1":
            raise ValueError("Unknown film/mist exchange model")

    def to_dict(self):
        result = asdict(self)
        if self.wetting is None:
            result.pop("wetting")
        return result

    def _potential(self):
        return LongWaveWettingPotential(
            self.surface_tension_n_m,
            self.wetting["adsorption_thickness_m"],
            self.wetting["equilibrium_contact_angle_deg"],
            self.wetting.get("dry_extension"),
        )

    def wall_energy(self, pipe, film_volume):
        if self.wetting is None:
            return torch.zeros_like(film_volume)
        potential = self._potential()
        area = pipe.fluid.tensor(pipe.lengths) * self.wetting["film_width_m"]
        if (
            potential.scale > 0
            and potential.dry_extension is None
            and bool((film_volume <= 0).any().cpu())
        ):
            raise PropertyModelError(
                "Singular adsorption potential requires real positive film; no dry-state floor"
            )
        energy = area * potential.potential(film_volume / area)
        if not bool(torch.isfinite(energy).all().cpu()):
            raise PropertyModelError("Nonfinite material wall energy")
        return energy

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def apply(self, pipe, state, dt_s, *, entrainment_m3_s, deposition_m3_s):
        dt = _number(dt_s, "exchange interval", positive=True)
        before = pipe.inspect(state)
        names = (self.gas_slot, self.film_slot, self.mist_slot)
        roots = {s["name"]: s["root"] for s in pipe.slots}
        expected = {self.gas_slot: "vapor", self.film_slot: "liquid", self.mist_slot: "liquid"}
        if any(roots.get(n) != r for n, r in expected.items()):
            raise ValueError("Exchange requires named gas vapor and two liquid morphology slots")
        count = len(state.cells)
        nc = len(pipe.fluid.names)
        f = pipe.fluid
        rates = []
        for field in (entrainment_m3_s, deposition_m3_s):
            if len(field) != count:
                raise ValueError("One gross exchange rate per cell required")
            rates.append(
                f.tensor([_number(v, "gross volume rate", nonnegative=True) for v in field])
            )
        ve, vd = (dt * r for r in rates)
        old = f.tensor(
            [
                [
                    [
                        *dict(c)[n].component_moles,
                        dict(c)[n].volume_m3,
                        dict(c)[n].momentum_kg_m_s,
                        dict(c)[n].total_energy_j,
                    ]
                    if n in dict(c)
                    else [0.0] * (nc + 3)
                    for n in names
                ]
                for c in state.cells
            ]
        )
        new = old.clone()
        area_energy = 6 * self.surface_tension_n_m / self.droplet_diameter_m
        wall_before = self.wall_energy(pipe, old[:, 1, nc])
        if bool((vd > old[:, 2, nc]).any().cpu()):
            raise PropertyModelError("Deposition exceeds actual droplet volume")
        safe_v = torch.where(old[:, 2, nc] > 0, old[:, 2, nc], torch.ones_like(vd))
        fd = vd / safe_v
        parcel_d = old[:, 2, :] * fd[:, None]
        new[:, 2, :] = old[:, 2, :] * (1 - fd)[:, None]
        new[:, 1, :] += parcel_d
        wall_deposited = self.wall_energy(pipe, new[:, 1, nc])
        deposition_heat = area_energy * vd - (wall_deposited - wall_before)
        new[:, 1, -1] += deposition_heat
        if bool((ve > new[:, 1, nc]).any().cpu()):
            raise PropertyModelError("Entrainment exceeds actual post-deposition film volume")
        safe_v = torch.where(new[:, 1, nc] > 0, new[:, 1, nc], torch.ones_like(ve))
        fe = ve / safe_v
        parcel_e = new[:, 1, :] * fe[:, None]
        new[:, 1, :] = new[:, 1, :] * (1 - fe)[:, None]
        new[:, 2, :] += parcel_e
        wall_after = self.wall_energy(pipe, new[:, 1, nc])
        mass = new[:, :, :nc] @ f.molar_masses
        mg, md = mass[:, 0], mass[:, 2]
        needs = ve > 0
        if bool((needs & ((mg <= 0) | (md <= 0))).any().cpu()):
            raise PropertyModelError("Atomization needs a real gas/drop kinetic reservoir")
        sg = torch.where(mg > 0, mg, torch.ones_like(mg))
        sd = torch.where(md > 0, md, torch.ones_like(md))
        reduced = torch.where(mg <= md, mg / (1 + mg / sd), md / (1 + md / sg))
        ug, ud = new[:, 0, -2] / sg, new[:, 2, -2] / sd
        relative = ug - ud
        available = 0.5 * reduced * relative.square()
        surface_creation = area_energy * ve + (wall_after - wall_deposited)
        # Releasing wall energy heats the real droplets. Positive creation
        # alone consumes the declared gas/drop relative kinetic reservoir.
        kinetic_loss = torch.clamp_min(surface_creation, 0) / self.atomization_efficiency
        if bool((kinetic_loss > available).any().cpu()):
            raise PropertyModelError(
                "Insufficient relative kinetic energy for declared atomization"
            )
        ratio = kinetic_loss / torch.where(available > 0, available, torch.ones_like(available))
        fraction = ratio / (1 + torch.sqrt(1 - ratio))
        impulse = reduced * relative * fraction
        pg, pd = new[:, 0, -2].clone(), new[:, 2, -2].clone()
        new[:, 0, -2] = pg - impulse
        new[:, 2, -2] = pd + impulse
        dkg = -impulse * (2 * pg - impulse) / (2 * sg)
        dkd = impulse * (2 * pd + impulse) / (2 * sd)
        extra_heat = kinetic_loss - surface_creation
        new[:, 0, -1] += dkg + (1 - self.heat_to_mist_fraction) * extra_heat
        new[:, 2, -1] += dkd + self.heat_to_mist_fraction * extra_heat
        data = new.cpu().tolist()
        locations = []
        rows = []
        guesses = []
        phase_roots = []
        active = ((ve + vd) > 0).cpu().tolist()
        for i, cell in enumerate(data):
            if not active[i]:
                continue
            old_cell = dict(state.cells[i])
            for name in pipe.names:
                if name not in names:
                    continue
                j = names.index(name)
                r = cell[j]
                if all(v == 0 for v in r):
                    continue
                if min(r[:nc]) < 0 or r[nc] <= 0 or not math.fsum(r[:nc]) > 0:
                    raise PropertyModelError("Exchange produced invalid active phase stocks")
                previous = old_cell.get(name)
                if previous is None:
                    previous = old_cell.get(self.film_slot) or old_cell.get(self.mist_slot)
                locations.append((i, name))
                rows.append(r)
                guesses.append(previous.temperature_k)
                phase_roots.append(roots[name])
        phases = (
            recover_homogeneous_phases(
                f,
                [r[:nc] for r in rows],
                [r[nc] for r in rows],
                [r[-2] for r in rows],
                [r[-1] for r in rows],
                phase_roots,
                guesses,
            )
            if rows
            else ()
        )
        cells = [
            {n: p for n, p in c if not active[i] or n not in names}
            for i, c in enumerate(state.cells)
        ]
        for (i, name), p in zip(locations, phases, strict=True):
            cells[i][name] = p
        end = replace(
            state, cells=tuple(tuple((n, c[n]) for n in pipe.names if n in c) for c in cells)
        )
        after = pipe.inspect(end)
        production = after["entropy"] - before["entropy"]
        if production < -entropy_allowance(after["entropy"], before["entropy"]):
            raise ConservationError("Film/mist exchange decreased fluid entropy")
        delta_surface = area_energy * (ve - vd) + wall_after - wall_before
        residual = pipe.ledger(state, end, external_heat_j=-float(delta_surface.sum().cpu()))
        roundoff = 64 * math.fsum(
            math.ulp(p.total_energy_j) for s in (state, end) for c in s.cells for _, p in c
        )
        if abs(residual[-2]) > roundoff + 1e-18 + 1e-10 * float(
            (surface_creation.abs() + deposition_heat.abs()).sum().cpu()
        ):
            raise ConservationError("Film/mist surface plus bulk energy balance failed")
        return end, {
            "model": self.to_dict(),
            "dt_s": dt,
            "state": end.to_dict(),
            "entrainment_m3_s": list(entrainment_m3_s),
            "deposition_m3_s": list(deposition_m3_s),
            "surface_energy_before_j": (area_energy * old[:, 2, nc] + wall_before).cpu().tolist(),
            "surface_energy_after_j": (area_energy * new[:, 2, nc] + wall_after).cpu().tolist(),
            "surface_creation_j": surface_creation.cpu().tolist(),
            "deposition_heat_j": deposition_heat.cpu().tolist(),
            "kinetic_energy_removed_j": kinetic_loss.cpu().tolist(),
            "irreversible_heat_j": extra_heat.cpu().tolist(),
            "gas_to_mist_impulse_kg_m_s": impulse.cpu().tolist(),
            "droplet_number": (
                new[:, 2, nc]
                / (
                    math.pi
                    / 6
                    * self.droplet_diameter_m
                    * self.droplet_diameter_m
                    * self.droplet_diameter_m
                )
            )
            .cpu()
            .tolist(),
            "entropy_production_j_k": production,
            "conservation_residuals": residual,
            "joint_energy_residual_j": residual[-2],
        }
