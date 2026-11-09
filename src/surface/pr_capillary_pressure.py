"""Energy-consistent PR pressure/volume relaxation with resolved film geometry.

Temperatures, phasic volumes and optional component/heat exchange evolve
jointly. Positive mobilities model damped normal and chemical relaxation.
Both phases and transferring components must already have positive support;
axial interface transport and phase-topology changes are not implemented.
No post-hoc bulk-energy correction and no forced equal-pressure projection.
"""

import math
from dataclasses import replace

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .peng_robinson import R
from .pr_homogeneous_uv import caloric_tv, recover_homogeneous_phases


def chemical_potentials_tv(fluid, temperature, molar_volume, composition):
    """Molar Gibbs potentials with the same caloric reference as PR N/V/E.

    TV evaluation avoids differentiating cubic branch selection. A zero
    component is allowed only when its chemical mobility is zero.
    """
    t, v, x = temperature, molar_volume, composition
    ai = fluid.a0 * (1 + fluid.kappa * (1 - torch.sqrt(t[..., None] / fluid.tc))).square()
    aij = torch.sqrt(ai[..., :, None] * ai[..., None, :]) * (1 - fluid.kij)
    am = (x[..., :, None] * x[..., None, :] * aij).sum((-2, -1))
    b = (x * fluid.b).sum(-1)
    q = caloric_tv(fluid, t, v, x)
    z = q["p"] * v / (R * t)
    ratio = torch.log1p(2 * math.sqrt(2) * b / (v + (1 - math.sqrt(2)) * b))
    attraction = 2 * (aij * x[..., None, :]).sum(-1) / am[..., None] - fluid.b / b[..., None]
    ideal = fluid.cp * (
        t[..., None] - fluid.tref - t[..., None] * torch.log(t[..., None] / fluid.tref)
    )
    return (
        ideal
        + R
        * t[..., None]
        * (
            x.clamp_min(1e-300).log()
            + torch.log(R * t / (fluid.pref * (v - b)))[..., None]
            + fluid.b / b[..., None] * (z - 1)[..., None]
        )
        - (am / (2 * math.sqrt(2) * b) * ratio)[..., None] * attraction
    )


class PRCapillaryPressure:
    def __init__(
        self,
        pipe,
        liquid_slot,
        gas_slot,
        width_m,
        surface_tension_n_m,
        volume_mobility_m3_pa_s,
        *,
        liquid_heat_fraction=0.5,
        maximum_unknowns=192,
        component_mobility_mol2_k_j_s=None,
        transfer_energy_j_mol=None,
        heat_conductance_w_k=0.0,
        maximum_slope=0.2,
    ):
        self.pipe = pipe
        roots = {s["name"]: s["root"] for s in pipe.slots}
        if (
            liquid_slot == gas_slot
            or roots.get(liquid_slot) != "liquid"
            or roots.get(gas_slot) != "vapor"
        ):
            raise ValueError("Declared distinct liquid/vapor slots required")
        self.liquid, self.gas = liquid_slot, gas_slot
        self.width = _number(width_m, "resolved interface width", positive=True)
        self.sigma = _number(surface_tension_n_m, "surface tension", positive=True)
        self.mobility = _number(volume_mobility_m3_pa_s, "normal volume mobility", positive=True)
        self.maximum_slope = _number(maximum_slope, "capillary slope envelope", positive=True)
        if self.maximum_slope > 0.2:
            raise ValueError("Small-slope source requires a slope envelope <= .2")
        self.share = _number(liquid_heat_fraction, "liquid heat fraction", nonnegative=True)
        if (
            self.share > 1
            or type(maximum_unknowns) is not int
            or maximum_unknowns < 3 * len(pipe.lengths)
        ):
            raise ValueError("Heat share in [0,1] and bounded dense solve required")
        count = len(pipe.fluid.names)
        self.component_mobility = (
            tuple(0.0 for _ in range(count))
            if component_mobility_mol2_k_j_s is None
            else tuple(
                _number(v, "component mobility", nonnegative=True)
                for v in component_mobility_mol2_k_j_s
            )
        )
        if len(self.component_mobility) != count:
            raise ValueError("One mobility per component required")
        self.active = tuple(i for i, v in enumerate(self.component_mobility) if v > 0)
        self.transfer_energy = (
            tuple(0.0 for _ in range(count))
            if transfer_energy_j_mol is None
            else tuple(_number(v, "molar transfer energy") for v in transfer_energy_j_mol)
        )
        if len(self.transfer_energy) != count or (self.active and transfer_energy_j_mol is None):
            raise ValueError("Explicit component transfer energies required for mass exchange")
        self.conductance = _number(
            heat_conductance_w_k, "interface heat conductance", nonnegative=True
        )
        if maximum_unknowns < (3 + len(self.active)) * len(pipe.lengths):
            raise ValueError("Capillary chemical solve exceeds configured unknown budget")
        self.maximum_unknowns = maximum_unknowns
        self.area = pipe.fluid.tensor(pipe.lengths) * self.width
        edges = [(i, i + 1) for i in range(len(pipe.lengths) - 1)]
        if pipe.boundary_mode == "periodic":
            if len(pipe.lengths) < 3:
                raise ValueError("Periodic film requires at least three cells")
            edges.append((len(pipe.lengths) - 1, 0))
        self.incidence = pipe.fluid.tensor(
            [[float(k == j) - float(k == i) for k in range(len(pipe.lengths))] for i, j in edges]
        )
        if not edges:
            raise ValueError("Resolved film needs connected cells")
        self.distances = pipe.fluid.tensor(
            [0.5 * (pipe.lengths[i] + pipe.lengths[j]) for i, j in edges]
        )
        self.weights = pipe.fluid.tensor(
            [
                self.sigma * self.width / (0.5 * (pipe.lengths[i] + pipe.lengths[j]))
                for i, j in edges
            ]
        )
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_capillary_pressure_v1",
            "pipe_model_id": self.pipe.model_id,
            "liquid_slot": self.liquid,
            "gas_slot": self.gas,
            "width_m": self.width,
            "surface_tension_n_m": self.sigma,
            "volume_mobility_m3_pa_s": self.mobility,
            "liquid_heat_fraction": self.share,
            "maximum_unknowns": self.maximum_unknowns,
            "component_mobility_mol2_k_j_s": self.component_mobility,
            "transfer_energy_j_mol": self.transfer_energy,
            "heat_conductance_w_k": self.conductance,
            "maximum_slope": self.maximum_slope,
        }

    @classmethod
    def from_dict(cls, pipe, raw):
        data = dict(raw)
        if (
            data.pop("model", None) != "pr_capillary_pressure_v1"
            or data.pop("pipe_model_id", None) != pipe.model_id
        ):
            raise ValueError("Capillary normal-source pipe identity differs")
        return cls(pipe, **data)

    def surface_energy(self, state):
        self.pipe.inspect(state)
        if any(self.liquid not in dict(c) or self.gas not in dict(c) for c in state.cells):
            raise PropertyModelError("Both film phases required for surface geometry")
        h = (
            self.pipe.fluid.tensor([dict(c)[self.liquid].volume_m3 for c in state.cells])
            / self.area
        )
        if not bool(
            ((self.incidence @ h).abs() / self.distances <= self.maximum_slope).all().cpu()
        ):
            raise PropertyModelError("Capillary geometry exceeds the small-slope envelope")
        return float((0.5 * self.weights * (self.incidence @ h).square()).sum().cpu())

    def apply(self, state, dt_s):
        pipe = self.pipe
        f = pipe.fluid
        pipe.inspect(state)
        if _digest(self.to_dict()) != self.model_id:
            raise ValueError("Changed capillary normal-source model")
        dt = _number(dt_s, "capillary volume source interval", positive=True)
        if any(set(dict(c)) != {self.liquid, self.gas} for c in state.cells):
            raise PropertyModelError(
                "Capillary normal source requires both declared phases in each cell"
            )
        self.surface_energy(state)
        old = [dict(c)[name] for c in state.cells for name in (self.liquid, self.gas)]
        nc = len(state.cells)
        n = f.tensor([p.component_moles for p in old]).reshape(nc, 2, -1)
        amount = n.sum(-1)
        x = n / amount[..., None]
        v0 = f.tensor([p.volume_m3 for p in old]).reshape(nc, 2)
        t0 = f.tensor([p.temperature_k for p in old]).reshape(nc, 2)
        total = v0.sum(-1)
        momentum = f.tensor([p.momentum_kg_m_s for p in old]).reshape(nc, 2)
        mass = (n * f.molar_masses).sum(-1)
        e0 = f.tensor([p.total_energy_j for p in old]).reshape(nc, 2)
        thermal = amount * caloric_tv(f, t0, v0 / amount, x)["cv"] * t0
        pscale = f.tensor(
            [max(1.0, a.pressure_pa, b.pressure_pa) for a, b in zip(old[::2], old[1::2])]
        )
        h0 = v0[:, 0] / self.area

        active = torch.tensor(self.active, dtype=torch.long, device=f.device)
        if self.active and not bool((n[:, :, active] > 0).all().cpu()):
            raise PropertyModelError(
                "Mass-exchanging components require positive support in both phases"
            )
        scales = torch.minimum(n[:, 0, active], n[:, 1, active])
        mobility = f.tensor(self.component_mobility)[active]
        epsilon = f.tensor(self.transfer_energy)
        interface_velocity = 0.5 * (momentum / mass).sum(-1)
        transfer_momentum = interface_velocity[:, None] * f.molar_masses

        def evaluate(z):
            dv = z[:nc] * total
            volumes = v0 + torch.stack((dv, -dv), -1)
            temperature = t0 * (1 + z[nc : 3 * nc].reshape(nc, 2))
            dn = n[:, 0, :] * 0
            dn = dn.index_copy(1, active, z[3 * nc :].reshape(nc, len(self.active)) * scales)
            numbers = n + torch.stack((dn, -dn), 1)
            amounts = numbers.sum(-1)
            compositions = numbers / amounts[..., None]
            dm = (dn * transfer_momentum).sum(-1)
            momenta = momentum + torch.stack((dm, -dm), -1)
            masses = (numbers * f.molar_masses).sum(-1)
            ke = 0.5 * momenta.square() / masses
            q = caloric_tv(f, temperature, volumes / amounts, compositions)
            midheight = h0 + dv / (2 * self.area)
            chemical = (
                self.incidence.T @ (self.weights * (self.incidence @ midheight))
            ) / self.area
            force = q["p"][:, 0] - q["p"][:, 1] - chemical
            heat = dv.square() / (dt * self.mobility)
            heat_transfer = dt * self.conductance * (temperature[:, 1] - temperature[:, 0])
            energy_transfer = heat_transfer + (dn * epsilon).sum(-1)
            work = torch.stack((-q["p"][:, 0] * dv, q["p"][:, 1] * dv), -1)
            energy = (
                amounts * q["u"]
                + ke
                - e0
                - work
                - heat[:, None] * f.tensor([self.share, 1 - self.share])
                - torch.stack((energy_transfer, -energy_transfer), -1)
            )
            affinity = dn * 0
            if self.active:
                mu = chemical_potentials_tv(f, temperature, volumes / amounts, compositions)
                velocity = momenta / masses
                term = (
                    epsilon
                    - velocity[..., None] * transfer_momentum[:, None, :]
                    + 0.5 * velocity[..., None].square() * f.molar_masses
                    - mu
                )
                affinity = (
                    term[:, 0, :] / temperature[:, 0, None]
                    - term[:, 1, :] / temperature[:, 1, None]
                )
            rows = [((dv / (dt * self.mobility) - force) / pscale), (energy / thermal).flatten()]
            if self.active:
                rows.append(((dn[:, active] / (dt * mobility) - affinity[:, active]) / R).flatten())
            residual = torch.cat(rows)
            valid = (
                (volumes > 0).all()
                & (numbers >= 0).all()
                & (amounts > 0).all()
                & (numbers[:, :, active] > 0).all()
                & (volumes / amounts > q["b"]).all()
                & (temperature >= f.min_temperature_k).all()
                & (temperature <= f.max_temperature_k).all()
                & (q["dpdv"] < 0).all()
                & (q["cv"] > 0).all()
                & (q["p"] >= f.min_pressure_pa).all()
                & (q["p"] <= f.max_pressure_pa).all()
                & (
                    (self.incidence @ (volumes[:, 0] / self.area)).abs() / self.distances
                    <= self.maximum_slope
                ).all()
            )
            return (
                residual,
                valid,
                volumes,
                temperature,
                q,
                chemical,
                heat,
                dv,
                numbers,
                momenta,
                ke,
                dn,
                heat_transfer,
            )

        z = f.tensor([0.0] * ((3 + len(self.active)) * nc))
        for iteration in range(31):
            r, valid, *_ = evaluate(z)
            norm = float(r.abs().max().cpu())
            if not bool(valid.cpu()) or not math.isfinite(norm):
                raise PropertyModelError("Inadmissible capillary PR iterate")
            if norm <= 2e-12:
                break
            if iteration == 30:
                raise HydraulicConvergenceError("Capillary normal Newton exhausted")
            try:
                change = torch.linalg.solve(torch.func.jacfwd(lambda y: evaluate(y)[0])(z), -r)
            except RuntimeError as exc:
                raise HydraulicConvergenceError("Capillary normal linear solve failed") from exc
            for power in range(25):
                trial = z + change * 2.0 ** (-power)
                rr, ok, *_ = evaluate(trial)
                if bool(ok.cpu()) and float(rr.abs().max().cpu()) < norm:
                    z = trial
                    break
            else:
                raise HydraulicConvergenceError("Capillary normal line search failed")
        (
            _,
            _,
            volumes,
            temperature,
            q,
            chemical,
            heat,
            dv,
            numbers,
            momenta,
            ke,
            dn,
            heat_transfer,
        ) = evaluate(z)
        if not bool((torch.isfinite(heat) & (heat >= -1e-20)).all().cpu()):
            raise ConservationError("Nonpassive capillary normal motion")
        recovered = recover_homogeneous_phases(
            f,
            numbers.reshape(2 * nc, -1),
            volumes.flatten(),
            momenta.flatten(),
            (numbers.sum(-1) * q["u"] + ke).flatten(),
            [r for _ in state.cells for r in ("liquid", "vapor")],
            temperature.flatten(),
        )
        cells = []
        for i, c in enumerate(state.cells):
            updates = {self.liquid: recovered[2 * i], self.gas: recovered[2 * i + 1]}
            cells.append(tuple((name, updates[name]) for name, _ in c))
        end = replace(state, cells=tuple(cells))
        before = pipe.inspect(state)
        after = pipe.inspect(end)
        delta_surface = float((chemical * dv).sum().cpu())
        surface_before = self.surface_energy(state)
        surface_after = self.surface_energy(end)
        bulk_change = math.fsum(
            b.total_energy_j - a.total_energy_j for a, b in zip(old, recovered, strict=True)
        )
        allowance = 64 * math.fsum(math.ulp(p.total_energy_j) for p in (*old, *recovered)) + 1e-22
        defect = bulk_change + delta_surface
        if abs(defect) > allowance or abs(
            surface_after - surface_before - delta_surface
        ) > 1e-20 + 1e-10 * max(abs(surface_before), abs(surface_after)):
            raise ConservationError("Capillary pressure/volume/surface energy balance failed")
        production = after["entropy"] - before["entropy"]
        lower = float(
            (heat * (self.share / temperature[:, 0] + (1 - self.share) / temperature[:, 1]))
            .sum()
            .cpu()
        )
        lower += float(
            (heat_transfer * (1 / temperature[:, 0] - 1 / temperature[:, 1])).sum().cpu()
        )
        if self.active:
            lower += float((dn[:, active].square() / (dt * mobility)).sum().cpu())
        if production < lower - 1e-22 - 128 * math.ulp(
            max(abs(before["entropy"]), abs(after["entropy"]))
        ):
            raise ConservationError("Capillary normal source entropy rejected")
        return end, {
            "state": end.to_dict(),
            "source_model_id": self.model_id,
            "source_model": self.to_dict(),
            "surface_energy_definition": "small_slope_excess_over_fixed_flat_interface",
            "dt_s": dt,
            "clock_advanced": False,
            "newton_iterations": iteration,
            "volume_transfer_m3": dv.cpu().tolist(),
            "component_transfer_to_liquid_mol": dn.cpu().tolist(),
            "heat_transfer_to_liquid_j": heat_transfer.cpu().tolist(),
            "capillary_pressure_pa": chemical.cpu().tolist(),
            "normal_dissipation_j": heat.cpu().tolist(),
            "surface_energy_before_j": surface_before,
            "surface_energy_after_j": surface_after,
            "bulk_energy_change_j": bulk_change,
            "energy_residual_j": defect,
            "energy_roundoff_allowance_j": allowance,
            "entropy_production_j_k": production,
            "entropy_production_lower_bound_j_k": lower,
            "qualification": "Nonisothermal capillary normal/chemical/heat source on positive two-phase support; axial surface transport and phase births not included",
        }
