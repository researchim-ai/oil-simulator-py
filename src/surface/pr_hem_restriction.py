"""Homogeneous equilibrium critical restrictions in the common PR Newton.

The throat keeps inlet entropy and stagnation enthalpy. Phase allocations,
pressure, temperature and velocity are algebraic unknowns of the same graph.
A complementarity condition chooses receiving pressure or a smooth interior
maximum of mass flux. No frozen-quality or ideal-gas replacement is made.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .liquid import HydraulicConvergenceError, PropertyModelError, _name, _number
from .peng_robinson import R


@dataclass(frozen=True)
class PRHEMRestriction:
    name: str
    source: str
    target: str
    throat_area_m2: float
    maximum_molar_rate_mol_s: float
    minimum_pressure_pa: float
    discharge_coefficient: float = 1.0
    opening: float = 1.0
    maximum_throat_velocity_m_s: float = 2000.0
    allow_reverse: bool = True
    open: bool = True

    def __post_init__(self):
        for key in ("name", "source", "target"):
            _name(getattr(self, key))
        if self.source == self.target:
            raise ValueError("Restriction needs distinct finite plenums")
        for key in (
            "throat_area_m2",
            "maximum_molar_rate_mol_s",
            "minimum_pressure_pa",
            "maximum_throat_velocity_m_s",
            "discharge_coefficient",
        ):
            _number(getattr(self, key), key, positive=True)
        _number(self.opening, "opening", nonnegative=True)
        if self.opening > 1 or self.discharge_coefficient > 1:
            raise ValueError("Opening and discharge coefficient must not exceed one")
        if type(self.allow_reverse) is not bool or type(self.open) is not bool:
            raise ValueError("Explicit boolean restriction modes required")


class EquilibriumThroat:
    """Exact component allocation parameterization of a fixed phase branch."""

    def __init__(self, fluid, seed, active, tags):
        self.f, self.active, self.tags = fluid, active, tags
        self.count = seed["phase_count"]
        self.nalloc = (self.count - 1) * len(active)
        self.initial = fluid.tensor([float(seed["temperature_k"]) / fluid.tref])
        if self.nalloc:
            allocations = torch.stack(
                [p["molar_fraction"] * p["mole_fractions"][active] for p in seed["phases"]]
            )
            logs = allocations.clamp_min(1e-100).log()
            self.initial = torch.cat((self.initial, (logs[:-1] - logs[-1]).reshape(-1)))

    def values(self, w, pressure, composition):
        temperature = w[0] * self.f.tref
        z = composition[self.active]
        if self.count == 1:
            beta = w.new_ones(1)
            xa = z[None, :]
        else:
            logits = torch.cat(
                (
                    w[1:].reshape(self.count - 1, len(self.active)),
                    w.new_zeros((1, len(self.active))),
                )
            )
            allocation = torch.softmax(logits, 0) * z
            beta = allocation.sum(-1)
            xa = allocation / beta[:, None].clamp_min(1e-100)
        x = w.new_zeros((self.count, len(composition))).index_copy(-1, self.active, xa)
        qs = [
            self.f.tensor_homogeneous(pressure, temperature, x[j], tag, derivative_order=2)
            for j, tag in enumerate(self.tags)
        ]
        h, s, v = [
            sum(beta[j] * q[key] for j, q in enumerate(qs))
            for key in ("molar_enthalpy_j_mol", "molar_entropy_j_mol_k", "molar_volume_m3_mol")
        ]
        mu = torch.stack(
            [
                xa[j].clamp_min(1e-100).log() + q["log_fugacity_coefficients"][self.active]
                for j, q in enumerate(qs)
            ]
        )
        chemical = (mu[:-1] - mu[-1]).reshape(-1)
        if self.nalloc:
            chemical = torch.where((z > 0).repeat(self.count - 1), chemical, w[1:])
        valid = torch.stack([q["valid"] for q in qs]).all() & (beta > 1e-13).all()
        return h, s, v, chemical, beta, x, valid

    def constraints(self, w, pressure, composition, entropy):
        q = self.values(w, pressure, composition)
        return torch.cat((q[3], ((q[1] - entropy) / R).reshape(1)))

    def isentropic_volume_derivative(
        self, w, pressure, composition, entropy, *, return_state=False
    ):
        fw, fp = torch.func.jacfwd(self.constraints, argnums=(0, 1))(
            w, pressure, composition, entropy
        )
        # The small equilibrium thermodynamic tangent is factored on device.
        _, info = torch.linalg.solve_ex(fw.detach(), -fp.detach(), check_errors=False)
        regular = (info == 0) & torch.isfinite(fw).all() & torch.isfinite(fp).all()
        identity = torch.eye(len(w), dtype=w.dtype, device=w.device)
        tangent = torch.linalg.solve(
            torch.where(regular, fw, identity), torch.where(regular, -fp, torch.zeros_like(fp))
        )
        vw, vp = torch.func.jacfwd(lambda a, b: self.values(a, b, composition)[2], argnums=(0, 1))(
            w, pressure
        )
        value = vp + vw @ tangent
        value = torch.where(regular & torch.isfinite(value), value, pressure.new_tensor(1.0))
        return (value, tangent) if return_state else value


class IncipientEquilibriumThroat(EquilibriumThroat):
    """Linear phase fractions keep the thermodynamic tangent finite at birth.

    Composition variables and the lever rule are explicit. In contrast to
    log allocations, a zero phase fraction is representable exactly.
    """

    def __init__(self, fluid, seed, active, tags, boundary_phase):
        self.f, self.active, self.tags = fluid, active, tags
        self.count = seed["phase_count"]
        self.boundary_phase = boundary_phase
        self.boundary_birth = seed.get("_boundary_birth", True)
        beta = torch.stack([p["molar_fraction"] for p in seed["phases"]])
        x = torch.stack([p["mole_fractions"][active] for p in seed["phases"]]).clamp_min(1e-100)
        logits = (x[:, :-1].log() - x[:, -1:].log()).reshape(-1)
        self.initial = torch.cat(
            (fluid.tensor([float(seed["temperature_k"]) / fluid.tref]), beta[:-1], logits)
        )
        self.nalloc = len(self.initial) - 1
        self.lower = torch.cat(
            (
                fluid.tensor([fluid.min_temperature_k / fluid.tref]),
                fluid.tensor([0.0] * (self.count - 1)),
                fluid.tensor([-80.0] * len(logits)),
            )
        )
        self.upper = torch.cat(
            (
                fluid.tensor([fluid.max_temperature_k / fluid.tref]),
                fluid.tensor([1.0] * (self.count - 1)),
                fluid.tensor([80.0] * len(logits)),
            )
        )
        self.parent_indices = [j for j in range(self.count) if j != boundary_phase]
        parent_phases = [seed["phases"][j] for j in self.parent_indices]
        self.parent = EquilibriumThroat(
            fluid,
            {**seed, "phase_count": self.count - 1, "phases": parent_phases},
            active,
            tuple(tags[j] for j in self.parent_indices),
        )

    def parent_coordinates(self, w, beta, x):
        if len(self.parent_indices) == 1:
            return w[:1]
        allocation = beta[self.parent_indices, None] * x[self.parent_indices][:, self.active]
        logs = allocation.clamp_min(1e-100).log()
        return torch.cat((w[:1], (logs[:-1] - logs[-1]).reshape(-1)))

    def values(self, w, pressure, composition):
        temperature = w[0] * self.f.tref
        beta = torch.cat((w[1 : self.count], (1 - w[1 : self.count].sum()).reshape(1)))
        logits = w[self.count :].reshape(self.count, len(self.active) - 1)
        xa = torch.softmax(torch.cat((logits, w.new_zeros((self.count, 1))), -1), -1)
        x = w.new_zeros((self.count, len(composition))).index_copy(-1, self.active, xa)
        qs = [
            self.f.tensor_homogeneous(pressure, temperature, x[j], tag, derivative_order=2)
            for j, tag in enumerate(self.tags)
        ]
        h, s, v = [
            sum(beta[j] * q[k] for j, q in enumerate(qs))
            for k in ("molar_enthalpy_j_mol", "molar_entropy_j_mol_k", "molar_volume_m3_mol")
        ]
        mu = torch.stack(
            [xa[j].log() + q["log_fugacity_coefficients"][self.active] for j, q in enumerate(qs)]
        )
        material = (beta[:, None] * xa).sum(0) - composition[self.active]
        constraints = torch.cat((material[:-1], (mu[:-1] - mu[-1]).reshape(-1)))
        valid = (
            torch.stack([q["valid"] for q in qs]).all()
            & (beta >= -1e-12).all()
            & ((composition[self.active].sum() - 1).abs() <= 1e-12)
        )
        return h, s, v, constraints, beta, x, valid


class HEMRestrictionBundle:
    def __init__(
        self,
        model,
        state,
        olddata,
        openings,
        offset,
        area_commands=None,
        seeds=None,
        active_support=None,
    ):
        self.model = model
        self.entries = []
        self.initial, self.lower, self.upper = [], [], []
        commands = area_commands or {}
        if set(commands) - {u.name for u in model.restrictions}:
            raise ValueError("Unknown restriction area command")
        self.seeds = seeds or {}
        self.seed_states = {}
        self.olddata = olddata
        f = model.fluid
        for unit in model.restrictions:
            a, b = model.names.index(unit.source), model.names.index(unit.target)
            opening = _number(
                commands.get(unit.name, unit.opening), "restriction opening", nonnegative=True
            )
            if opening > 1:
                raise ValueError("Restriction opening must not exceed one")
            enabled = openings[unit.name] and opening > 0
            donor, receiver = (
                (a, b) if state.nodes[a][1].pressure_pa >= state.nodes[b][1].pressure_pa else (b, a)
            )
            d = olddata[donor]
            z = d["n"] / d["n"].sum()
            active = (
                torch.nonzero((olddata[a]["n"] + olddata[b]["n"]) > 0).flatten()
                if active_support is None
                else torch.cat((active_support[a], active_support[b])).unique()
            )
            entropy = float((d["entropy"] / d["n"].sum()).cpu())
            h = float(((d["phase_n"] * d["q"]["molar_enthalpy_j_mol"]).sum() / d["n"].sum()).cpu())
            p0 = float(d["p"].cpu())
            pb = float(olddata[receiver]["p"].cpu())
            seed = self.seeds.get(unit.name)
            if seed is None and len(d["phase_n"]) > 1:
                p = max(unit.minimum_pressure_pa, pb, 0.55 * p0) if enabled else p0
                beta = d["phase_n"] / d["phase_n"].sum()
                light = int(d["q"]["density_kg_m3"].argmin().cpu())
                gas = (
                    float(d["q"]["compressibility_factor"][light].cpu()) > 0.5
                    and float((d["q"]["density_kg_m3"].min() / d["q"]["density_kg_m3"].max()).cpu())
                    < 0.5
                )
                gas_fraction = float(beta[light].cpu()) if gas else 0.0
                cp = sum(float(z[j].cpu()) * c.ideal_cp_j_mol_k for j, c in enumerate(f.components))
                temp = min(
                    f.max_temperature_k,
                    max(
                        f.min_temperature_k,
                        float(d["t"].cpu()) * (p / p0) ** (R * gas_fraction / cp),
                    ),
                )
                pp, tt = f.tensor(p), f.tensor(temp)
                phases = []
                for j in range(len(d["phase_n"])):
                    liquid = f.tensor_homogeneous(pp, tt, d["x"][j], "liquid")
                    vapor = f.tensor_homogeneous(pp, tt, d["x"][j], "vapor")
                    reference = d["q"]["compressibility_factor"][j]
                    q = (
                        liquid
                        if float((liquid["compressibility_factor"] - reference).abs().cpu())
                        < float((vapor["compressibility_factor"] - reference).abs().cpu())
                        else vapor
                    )
                    phases.append({**q, "mole_fractions": d["x"][j], "molar_fraction": beta[j]})
                if all(bool(q["valid"].cpu()) for q in phases):
                    seed = {
                        "pressure_pa": pp,
                        "temperature_k": tt,
                        "mole_fractions": z,
                        "phase_count": len(phases),
                        "phases": phases,
                        "molar_enthalpy_j_mol": sum(
                            q["molar_enthalpy_j_mol"] * q["molar_fraction"] for q in phases
                        ),
                    }
            if seed is None:
                for ratio in (0.55, 0.8, 0.95, 1.0):
                    p = max(unit.minimum_pressure_pa, pb, ratio * p0) if enabled else p0
                    try:
                        seed = f.flash_ps(p, entropy, z, temperature_guess_k=float(d["t"].cpu()))
                        break
                    except (PropertyModelError, HydraulicConvergenceError):
                        continue
                else:
                    raise PropertyModelError("No covered PR isentropic throat seed")
            tags = model.cells[0]._tags(seed)
            boundary_phase = seed.get("_boundary_phase")
            if boundary_phase is not None:
                active = torch.nonzero(seed["mole_fractions"] > 0).flatten()
            trace = (
                EquilibriumThroat(f, seed, active, tags)
                if boundary_phase is None
                else IncipientEquilibriumThroat(f, seed, active, tags, boundary_phase)
            )
            self.seed_states[unit.name] = seed
            speed = (
                math.sqrt(
                    max(
                        0.0,
                        2
                        * (h - float(seed["molar_enthalpy_j_mol"]))
                        / float((z * f.molar_masses).sum()),
                    )
                )
                if enabled
                else 0.0
            )
            self.initial.append(
                torch.cat(
                    (
                        f.tensor([math.log(float(seed["pressure_pa"]))]),
                        trace.initial,
                        f.tensor([min(1.0, speed / unit.maximum_throat_velocity_m_s)]),
                    )
                )
            )
            lower = (
                trace.lower
                if boundary_phase is not None
                else torch.cat(
                    (f.tensor([f.min_temperature_k / f.tref]), f.tensor([-80.0] * trace.nalloc))
                )
            )
            upper = (
                trace.upper
                if boundary_phase is not None
                else torch.cat(
                    (f.tensor([f.max_temperature_k / f.tref]), f.tensor([80.0] * trace.nalloc))
                )
            )
            self.lower.append(
                torch.cat((f.tensor([math.log(unit.minimum_pressure_pa)]), lower, f.tensor([0.0])))
            )
            self.upper.append(
                torch.cat((f.tensor([math.log(f.max_pressure_pa)]), upper, f.tensor([1.0])))
            )
            self.entries.append((unit, a, b, offset, trace, enabled, opening, self.initial[-1]))
            offset += 3 + trace.nalloc

    def evaluate(self, y, data):
        f = self.model.fluid
        rows, records, traces = [], {}, []
        net = {}
        valid = y.new_ones((), dtype=torch.bool)
        for unit, a, b, offset, trace, enabled, opening, initial in self.entries:
            local = y[offset : offset + len(initial)]
            pressure = local[0].exp()
            w = local[1:-1]
            velocity = local[-1] * unit.maximum_throat_velocity_m_s
            left, right = data[a], data[b]
            forward = left["p"] >= right["p"]
            choose = lambda x, z, forward=forward: torch.where(forward, x, z)
            p0, pb = choose(left["p"], right["p"]), choose(right["p"], left["p"])
            z = choose(left["n"] / left["n"].sum(), right["n"] / right["n"].sum())
            h0 = choose(
                (left["phase_n"] * left["q"]["molar_enthalpy_j_mol"]).sum() / left["n"].sum(),
                (right["phase_n"] * right["q"]["molar_enthalpy_j_mol"]).sum() / right["n"].sum(),
            )
            s0 = choose(left["entropy"] / left["n"].sum(), right["entropy"] / right["n"].sum())
            mw = (z * f.molar_masses).sum()
            if not enabled:
                rows.append(local - initial)
                zero = local[-1] * 0
                records[unit.name] = {
                    k: zero
                    for k in (
                        "molar_rate_mol_s",
                        "mass_rate_kg_s",
                        "throat_velocity_m_s",
                        "throat_pressure_pa",
                        "throat_temperature_k",
                        "throat_molar_volume_m3_mol",
                        "throat_molar_enthalpy_j_mol",
                        "throat_molar_entropy_j_mol_k",
                        "stagnation_enthalpy_j_mol",
                        "isentropic_volume_derivative",
                        "sonic_criterion",
                        "energy_residual_j_mol",
                    )
                }
                records[unit.name].update(
                    choked=zero.to(torch.bool), forward=forward, opening=zero, enabled=False
                )
                continue
            h, s, v, mu, beta, x, admissible = trace.values(w, pressure, z)
            boundary = isinstance(trace, IncipientEquilibriumThroat)
            if boundary:
                vp, state_tangent = trace.isentropic_volume_derivative(
                    w, pressure, z, s0, return_state=True
                )
                phase_slope = (
                    state_tangent[1 + trace.boundary_phase]
                    if trace.boundary_phase < trace.count - 1
                    else -state_tangent[1 : trace.count].sum()
                )
                birth = phase_slope < 0
            else:
                vp = trace.isentropic_volume_derivative(w, pressure, z, s0)
                birth = pressure.new_zeros((), dtype=torch.bool)
            criterion = 1 + 2 * (h0 - h) * vp / v.square()
            energy = h + 0.5 * mw * velocity.square() - h0
            parent_criterion = criterion
            if boundary:
                parent_vp = trace.parent.isentropic_volume_derivative(
                    trace.parent_coordinates(w, beta, x), pressure, z, s0
                )
                parent_criterion = 1 + 2 * (h0 - h) * parent_vp / v.square()
            critical_row = (
                beta[trace.boundary_phase]
                if boundary
                else torch.minimum((pressure - pb) / p0, criterion)
            )
            row = torch.cat(
                (
                    mu,
                    ((s - s0) / R).reshape(1),
                    (energy / (R * f.tref)).reshape(1),
                    critical_row.reshape(1),
                )
            )
            # Algebraic rows can be reordered independently of local variables.
            rows.append(row)
            molar = unit.discharge_coefficient * unit.throat_area_m2 * opening * velocity / v
            signed = torch.where(forward, molar, -molar)
            phi = self.model.gravity_m_s2 * (
                self.model.nodes[a].elevation_m - self.model.reference_elevation_m
            )
            flux = torch.cat((signed * z, y.new_zeros(1), (signed * (h0 + mw * phi)).reshape(1)))
            net[a] = net.get(a, 0) - flux
            net[b] = net.get(b, 0) + flux
            valid = (
                valid
                & admissible
                & (vp < 0)
                & (pressure <= p0 * (1 + 1e-9))
                & (pressure >= pb * (1 - 1e-9))
                & (molar <= unit.maximum_molar_rate_mol_s)
                & torch.isfinite(criterion)
            )
            if boundary:
                valid = valid & (parent_vp < 0) & (phase_slope.abs() * pressure > 1e-10)
                valid = valid & torch.where(
                    birth,
                    (parent_criterion >= -1e-7) & (criterion <= 1e-7),
                    (criterion >= -1e-7) & (parent_criterion <= 1e-7),
                )
            if not unit.allow_reverse:
                valid = valid & forward
            records[unit.name] = {
                "molar_rate_mol_s": signed,
                "mass_rate_kg_s": signed * mw,
                "throat_velocity_m_s": velocity,
                "throat_pressure_pa": pressure,
                "throat_temperature_k": w[0] * f.tref,
                "throat_molar_volume_m3_mol": v,
                "throat_molar_enthalpy_j_mol": h,
                "throat_molar_entropy_j_mol_k": s,
                "stagnation_enthalpy_j_mol": h0,
                "isentropic_volume_derivative": vp,
                "sonic_criterion": criterion,
                "energy_residual_j_mol": energy,
                "choked": pressure > pb * (1 + 1e-8),
                "forward": forward,
                "opening": pressure.new_tensor(opening),
                "enabled": True,
                "critical_phase_boundary": pressure.new_tensor(boundary, dtype=torch.bool),
                "phase_boundary_birth": birth,
                "parent_sonic_criterion": parent_criterion,
            }
            records[unit.name].update(
                phase_fractions=beta, phase_mole_fractions=x, phase_roots=trace.tags
            )
            traces.append((unit, pressure, w[0] * f.tref, z, trace.count, beta, x, s0, pb))
        return net, rows, records, traces, valid

    def _seed_parts(self, p, t, z, beta, x, tags):
        phases = []
        for j, tag in enumerate(tags):
            q = self.model.fluid.tensor_homogeneous(p, t, x[j], tag)
            phases.append({**q, "mole_fractions": x[j], "molar_fraction": beta[j]})
        return {
            "pressure_pa": p,
            "temperature_k": t,
            "mole_fractions": z,
            "phase_count": len(beta),
            "phases": phases,
            "molar_enthalpy_j_mol": sum(
                b * q["molar_enthalpy_j_mol"] for b, q in zip(beta, phases)
            ),
        }

    def replacements(self, traces):
        """TPD test of the solved assemblage; no inverse PS-flash is needed.

        Tiny incipient phases are initialized for the common Newton, rather
        than asking a cold, unseeded allocation optimizer to resolve them.
        """
        changes = {}
        f = self.model.fluid
        for unit, p, t, z, count, beta, x, entropy, pb in traces:
            entry = next(e for e in self.entries if e[0].name == unit.name)
            trace = entry[4]
            boundary = isinstance(trace, IncipientEquilibriumThroat)
            active = torch.nonzero(z > 0).flatten()
            qs = [f.tensor_homogeneous(p, t, x[j], tag) for j, tag in enumerate(trace.tags)]
            if not boundary and count > 1 and float(beta.min().cpu()) < 1e-12:
                phase = int(beta.argmin().cpu())
                seed = self._seed_parts(p, t, z, beta, x, trace.tags)
                if float(p.cpu()) > float(pb.cpu()) * (1 + 1e-8):
                    seed["_boundary_phase"] = phase
                else:
                    keep = [j for j in range(count) if j != phase]
                    seed = self._seed_parts(
                        p,
                        t,
                        z,
                        beta[keep] / beta[keep].sum(),
                        x[keep],
                        tuple(trace.tags[j] for j in keep),
                    )
                changes[unit.name] = seed
                continue
            # Coincident compositions on the same cubic root are duplicate
            # phases, unlike pure-fluid liquid/vapour coexistence.
            duplicate = None
            for j in range(count):
                for k in range(j):
                    if (
                        float((x[j] - x[k]).abs().max().cpu()) < 1e-8
                        and float(
                            (qs[j]["compressibility_factor"] - qs[k]["compressibility_factor"])
                            .abs()
                            .cpu()
                        )
                        < 1e-8
                    ):
                        duplicate = (j, k)
            if duplicate is not None:
                j, k = duplicate
                combined = beta.clone()
                combined[k] += combined[j]
                keep = [i for i in range(count) if i != j]
                changes[unit.name] = self._seed_parts(
                    p, t, z, combined[keep], x[keep], tuple(trace.tags[i] for i in keep)
                )
                continue
            reference = x[-1].clamp_min(1e-100).log() + qs[-1]["log_fugacity_coefficients"]
            mismatch = max(
                float(
                    (
                        x[j, active].clamp_min(1e-100).log()
                        + qs[j]["log_fugacity_coefficients"][active]
                        - reference[active]
                    )
                    .abs()
                    .max()
                    .cpu()
                )
                for j in range(count)
            )
            if mismatch > 1e-5:
                continue
            minimum, trial, _props = f._stability(p, t, z, reference, active)
            if float(minimum.cpu()) >= -1e-8:
                continue
            if count >= f.max_phases:
                raise PropertyModelError(
                    "Restriction needs more equilibrium phases than configured"
                )
            fraction = min(
                1e-4, 0.05 * float((z[active] / trial[active].clamp_min(1e-100)).min().cpu())
            )
            take = f.tensor(fraction) * trial
            allocation = beta[:, None] * x
            allocation = allocation * (1 - take / z.clamp_min(1e-100))[None, :]
            allocation = torch.cat((allocation, take[None, :]), 0)
            weights = allocation.sum(-1)
            composition = allocation / weights[:, None]
            seed = self._seed_parts(p, t, z, weights, composition, (*trace.tags, "stable"))
            if float(p.cpu()) > float(pb.cpu()) * (1 + 1e-8):
                seed["_boundary_phase"] = count
            changes[unit.name] = seed
        return changes

    def regular_seeds(self, records, traces):
        """Release a phase-onset candidate when the smooth maximum is elsewhere."""
        seeds = {}
        for unit, p, t, z, _count, beta, x, _entropy, pb in traces:
            entry = next(e for e in self.entries if e[0].name == unit.name)
            trace = entry[4]
            if not isinstance(trace, IncipientEquilibriumThroat):
                continue
            row = records[unit.name]
            birth = bool(row["phase_boundary_birth"].cpu())
            full = float(row["sonic_criterion"].cpu())
            parent = float(row["parent_sonic_criterion"].cpu())
            before, after = (parent, full) if birth else (full, parent)
            if before >= -1e-7 and after <= 1e-7:
                continue
            if before < 0 and after > 0:
                raise PropertyModelError(
                    "Multiple critical maxima require explicit branch analysis"
                )
            revive = (full > 0) if birth else (full < 0)
            if revive:
                i = trace.boundary_phase
                keep = trace.parent_indices
                fraction = min(
                    1e-4, 0.05 * float((z[z > 0] / x[i, z > 0].clamp_min(1e-100)).min().cpu())
                )
                take = p.new_tensor(fraction) * x[i]
                allocation = beta[:, None] * x
                rest = allocation[keep].sum(0)
                allocation[keep] = allocation[keep] * (1 - take / rest.clamp_min(1e-100))[None, :]
                allocation[i] = take
                weights = allocation.sum(-1)
                xx = allocation / weights[:, None]
                seed = self._seed_parts(p, t, z, weights, xx, trace.tags)
            else:
                keep = trace.parent_indices
                weights = beta[keep] / beta[keep].sum()
                seed = self._seed_parts(
                    p, t, z, weights, x[keep], tuple(trace.tags[j] for j in keep)
                )
            seeds[unit.name] = seed
        return seeds

    def boundary_seeds(self):
        """Locate a phase-onset maximum after an interior branch fails.

        This is an outer initialization search, never part of Newton/JVP.
        Subsequent PS, fugacity and zero-fraction equations are monolithic.
        """
        result = {}
        f = self.model.fluid
        for unit, a, b, _offset, trace, enabled, _opening, _initial in self.entries:
            if not enabled or isinstance(trace, IncipientEquilibriumThroat):
                continue
            d = max((self.olddata[a], self.olddata[b]), key=lambda q: float(q["p"].cpu()))
            z = d["n"] / d["n"].sum()
            entropy = float((d["entropy"] / d["n"].sum()).cpu())
            active = torch.nonzero(z > 0).flatten()
            if len(active) == 1 and len(d["phase_n"]) == 1:
                index = int(active[0].cpu())
                temperature = float(d["t"].cpu())
                component = f.components[index]
                # Wilson K=1 is only an initial pressure estimate. The final
                # coexistence and entropy equations are solved by Torch Newton.
                estimate = component.critical_pressure_pa * math.exp(
                    5.373
                    * (1 + component.acentric_factor)
                    * (1 - component.critical_temperature_k / temperature)
                )
                floor = max(
                    unit.minimum_pressure_pa,
                    min(float(self.olddata[a]["p"].cpu()), float(self.olddata[b]["p"].cpu())),
                )
                pressure = min(0.95 * float(d["p"].cpu()), max(floor, estimate))
                if pressure <= unit.minimum_pressure_pa or pressure >= float(d["p"].cpu()):
                    continue
                pp, tt = f.tensor(pressure), f.tensor(temperature)
                liquid = f.tensor_homogeneous(pp, tt, z, "liquid")
                vapor = f.tensor_homogeneous(pp, tt, z, "vapor")
                # Only a seed: saturation and PS are solved again in common Newton.
                gas = float(d["q"]["compressibility_factor"][0].cpu()) > 0.5
                newborn = 0 if gas else 1
                fractions = (1e-6, 1 - 1e-6) if gas else (1 - 1e-6, 1e-6)
                phases = [
                    {**q, "mole_fractions": z, "molar_fraction": f.tensor(frac)}
                    for q, frac in zip((liquid, vapor), fractions)
                ]
                result[unit.name] = {
                    "pressure_pa": pp,
                    "temperature_k": tt,
                    "mole_fractions": z,
                    "phase_count": 2,
                    "phases": phases,
                    "molar_enthalpy_j_mol": sum(
                        q["molar_enthalpy_j_mol"] * q["molar_fraction"] for q in phases
                    ),
                    "_boundary_phase": newborn,
                }
                continue
            high = float(d["p"].cpu())
            floor = max(
                unit.minimum_pressure_pa,
                min(float(self.olddata[a]["p"].cpu()), float(self.olddata[b]["p"].cpu())),
            )
            if floor >= high:
                continue
            prior = {
                "pressure_pa": d["p"],
                "temperature_k": d["t"],
                "phase_count": len(d["phase_n"]),
                "phases": [
                    {
                        **{k: v[j] for k, v in d["q"].items()},
                        "mole_fractions": d["x"][j],
                        "molar_fraction": d["phase_n"][j] / d["phase_n"].sum(),
                    }
                    for j in range(len(d["phase_n"]))
                ],
            }
            prior_pressure = high
            h0 = float(
                ((d["phase_n"] * d["q"]["molar_enthalpy_j_mol"]).sum() / d["phase_n"].sum()).cpu()
            )
            pressures = [high * (floor / high) ** (k / 12) for k in range(1, 13)]
            attempts = 0
            while pressures and attempts < 80:
                attempts += 1
                pressure = pressures.pop(0)
                try:
                    current = f.flash_ps(
                        pressure,
                        entropy,
                        z,
                        temperature_guess_k=float(prior["temperature_k"].cpu()),
                    )
                except (PropertyModelError, HydraulicConvergenceError):
                    continue
                upper_count, lower_count = prior["phase_count"], current["phase_count"]
                if abs(upper_count - lower_count) > 1:
                    pressures[:0] = [(pressure * prior_pressure) ** 0.5, pressure]
                    continue
                if upper_count != lower_count:
                    upper, lower = prior_pressure, pressure
                    full = max(upper_count, lower_count)
                    candidate = prior if upper_count == full else current
                    for _ in range(40):
                        fractions = [float(q["molar_fraction"].cpu()) for q in candidate["phases"]]
                        if min(fractions) < 1e-5 and upper - lower < 1e-3 * upper:
                            break
                        mid = (upper * lower) ** 0.5
                        try:
                            q = f.flash_ps(
                                mid,
                                entropy,
                                z,
                                temperature_guess_k=float(candidate["temperature_k"].cpu()),
                            )
                        except (PropertyModelError, HydraulicConvergenceError):
                            break
                        if q["phase_count"] == upper_count:
                            upper = mid
                        elif q["phase_count"] == lower_count:
                            lower = mid
                        else:
                            break
                        if q["phase_count"] == full:
                            candidate = q
                    fractions = [float(q["molar_fraction"].cpu()) for q in candidate["phases"]]
                    phase = min(range(full), key=lambda j: fractions[j])
                    # This is only a starting point. The common Newton still
                    # enforces exactly zero incipient fraction and both one-sided
                    # critical inequalities; flash TPD thresholds can suppress
                    # a phase before its fraction reaches the seed target.
                    if fractions[phase] < 0.01:
                        seed = {
                            **candidate,
                            "_boundary_phase": phase,
                            "_boundary_birth": lower_count == full,
                            "mole_fractions": z,
                        }
                        tags = self.model.cells[0]._tags(candidate)
                        trial = IncipientEquilibriumThroat(f, seed, active, tags, phase)
                        w = trial.initial
                        pp = candidate["pressure_pa"]
                        hh, ss, vv, _mu, bb, xx, _valid = trial.values(w, pp, z)
                        vp = trial.isentropic_volume_derivative(w, pp, z, ss)
                        parent = trial.parent.isentropic_volume_derivative(
                            trial.parent_coordinates(w, bb, xx), pp, z, ss
                        )
                        cfull = float((1 + 2 * (h0 - hh) * vp / vv.square()).cpu())
                        cparent = float((1 + 2 * (h0 - hh) * parent / vv.square()).cpu())
                        before, after = (
                            (cparent, cfull) if seed["_boundary_birth"] else (cfull, cparent)
                        )
                        if (
                            float(vp.cpu()) < 0
                            and float(parent.cpu()) < 0
                            and before >= -0.02
                            and after <= 0.02
                        ):
                            result[unit.name] = seed
                            break
                prior, prior_pressure = current, pressure
        return result

    @staticmethod
    def serialize(records):
        output = {}
        for name, row in records.items():
            tensor = {k: v for k, v in row.items() if isinstance(v, torch.Tensor) and v.ndim == 0}
            values = torch.stack(list(tensor.values())).detach().cpu().tolist()
            output[name] = {
                k: bool(v) if tensor[k].dtype == torch.bool else v for k, v in zip(tensor, values)
            }
            output[name]["enabled"] = row["enabled"]
            for key in ("phase_fractions", "phase_mole_fractions"):
                output[name][key] = row[key].detach().cpu().tolist() if key in row else []
            output[name]["phase_roots"] = list(row.get("phase_roots", ()))
        return output

    def pattern(self, offsets):
        support = set()
        for _unit, a, b, offset, _trace, _enabled, _opening, initial in self.entries:
            local = [
                *range(offsets[a], offsets[a + 1]),
                *range(offsets[b], offsets[b + 1]),
                *range(offset, offset + len(initial)),
            ]
            support.update((r, c) for r in local for c in local)
        return support
