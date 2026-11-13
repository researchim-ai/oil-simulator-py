"""Homogeneous, no-slip dense liquid mixtures in an equilibrium pump trace.

Phase allocations are Newton unknowns, not frozen inlet qualities. This closure
uses a liquid pump card for an explicitly declared well mixed liquid emulsion;
it does not describe gas handling or predict an emulsion-specific pump map.
"""

from __future__ import annotations

import torch

from .liquid import EquipmentOperatingError, HydraulicConvergenceError, PropertyModelError
from .peng_robinson import R
from .pr_hem_restriction import EquilibriumThroat
from .pr_pump_unit import G, pump_curve, pump_head_efficiency


class LiquidMixtureTrace:
    _prefer_homogeneous_seed = True

    def __init__(self, model, unit, source, seeds=None, active=None):
        self.model, self.unit = model, unit
        f = model.fluid
        z = source["n"] / source["n"].sum()
        self.active = torch.nonzero(z > 0).flatten() if active is None else active
        if not bool((source["q"]["density_kg_m3"] >= unit.minimum_density_kg_m3).all().cpu()):
            raise EquipmentOperatingError(
                f"Pump {unit.name!r} liquid-mixture closure requires every inlet phase to be dense"
            )
        base = {
            "pressure_pa": source["p"],
            "temperature_k": source["t"],
            "phase_count": len(source["phase_n"]),
            "phases": [
                {
                    **{k: v[j] for k, v in source["q"].items()},
                    "mole_fractions": source["x"][j],
                    "molar_fraction": source["phase_n"][j] / source["phase_n"].sum(),
                }
                for j in range(len(source["phase_n"]))
            ],
        }
        self.base_seed = base
        initial_seed = base
        if seeds is None and self._prefer_homogeneous_seed:
            fractions = source["phase_n"] / source["phase_n"].sum()
            # Only the algebraic starting branch changes. All source stocks
            # and trace components remain intact, and the full TPD audit must
            # restore any missing stable phase before a step can be accepted.
            if len(fractions) > 1 and float((1 - fractions.max()).cpu()) < 1e-6:
                homogeneous = f.tensor_homogeneous(source["p"], source["t"], z)
                if bool(
                    (
                        homogeneous["valid"]
                        & (homogeneous["density_kg_m3"] >= unit.minimum_density_kg_m3)
                    ).cpu()
                ):
                    initial_seed = {
                        **base,
                        "phase_count": 1,
                        "phases": [
                            {**homogeneous, "molar_fraction": f.tensor(1.0), "mole_fractions": z}
                        ],
                    }
        self.traces = []
        for seed in seeds or (initial_seed, initial_seed, initial_seed):
            trace = EquilibriumThroat(f, seed, self.active, model.cells[0]._tags(seed))
            self.traces.append(trace)
        self.count = 5 + sum(trace.nalloc for trace in self.traces)
        self.extra_initial = [trace.initial[1:] for trace in self.traces]

    def evaluate(self, unit, a, b, offset, qscale, curves, efficiency, y, data, speeds):
        f, model = self.model.fluid, self.model
        source, target = data[a], data[b]
        z = source["n"] / source["n"].sum()
        mw = (z * f.molar_masses).sum()
        inlet_beta = source["phase_n"] / source["phase_n"].sum()
        hin = (inlet_beta * source["q"]["molar_enthalpy_j_mol"]).sum()
        sin = (inlet_beta * source["q"]["molar_entropy_j_mol_k"]).sum()
        q, pressure = y[offset] * qscale, y[offset + 1].exp()
        temperatures = [y[offset + j] * f.tref for j in (2, 3, 4)]
        speed = (
            speeds[unit.name] / model._drives[unit.name].nominal_speed_rad_s
            if unit.name in speeds
            else q.new_tensor(unit.card.speed_ratio)
        )
        head, eta = pump_head_efficiency(unit.card, curves, efficiency, q, speed)
        gap = (target["p"] + unit.outlet_loss_pa_s2_m6 * q.square() - pressure) / source["p"]
        closed = gap > y[offset] if unit.card.check_valve else q.new_zeros((), dtype=torch.bool)
        valve = torch.minimum(y[offset], gap) if unit.card.check_valve else gap
        trace_values, chemical, snapshots = [], [], []
        cursor = offset + 5
        valid = (source["q"]["density_kg_m3"] >= unit.minimum_density_kg_m3).all()
        for j, (trace, pp, tt) in enumerate(
            zip(self.traces, (pressure, pressure, target["p"]), temperatures)
        ):
            w = torch.cat((y[offset + 2 + j].reshape(1), y[cursor : cursor + trace.nalloc]))
            cursor += trace.nalloc
            h, s, v, mu, beta, x, _valid = trace.values(w, pp, z)
            properties = [f.tensor_homogeneous(pp, tt, xx, tag) for xx, tag in zip(x, trace.tags)]
            # Trace liquid phases may be arbitrarily small: positivity and exact
            # component allocation, rather than a quality cutoff, define this branch.
            dense = (
                torch.stack(
                    [
                        prop["valid"] & (prop["density_kg_m3"] >= unit.minimum_density_kg_m3)
                        for prop in properties
                    ]
                ).all()
                & (beta > 0).all()
            )
            active_trace = ~closed if j else closed.new_ones(())
            valid = valid & (~active_trace | dense)
            chemical.append(mu)
            trace_values.append((h, s, v))
            snapshots.append((pp, tt, z, beta, x, trace.tags))
        (his, sis, _), (hp, _, _), (hd, sd, _) = trace_values
        scale = (mw * G * unit.card.shutoff_head_m).clamp_min(1.0)
        # Even with a shut check valve the equilibrium allocations of all
        # algebraic trace states remain determined. No artificial stocks exist.
        rows = torch.cat(
            (
                torch.stack(
                    (
                        (his - hin - mw * G * head) / scale,
                        (sis - sin) / R,
                        torch.where(
                            closed,
                            (temperatures[1] - temperatures[0]) / f.tref,
                            (hp - hin - (his - hin) / eta.clamp_min(1e-8)) / scale,
                        ),
                        torch.where(
                            closed, (temperatures[2] - target["t"]) / f.tref, (hd - hp) / scale
                        ),
                        valve,
                    )
                ),
                *chemical,
            )
        )
        actual = torch.where(closed, q * 0, q)
        mass = actual * source["rho"]
        molar = mass / mw
        phi = model.gravity_m_s2 * (model.nodes[a].elevation_m - model.reference_elevation_m)
        first = torch.cat((molar * z, y.new_zeros(1), (molar * (hin + mw * phi)).reshape(1)))
        last = torch.cat((molar * z, y.new_zeros(1), (molar * (hd + mw * phi)).reshape(1)))
        power, hydraulic = molar * (hd - hin), molar * (his - hin)
        entropy = molar * (sd - sin)
        sound2 = source["q"]["frozen_sound_speed_m_s"].square().min()
        valid = (
            valid
            & (eta >= 0)
            & (closed | (eta > 0))
            & (eta <= 1)
            & (q <= speed * unit.card.max_flow_m3_s)
            & (head >= 0)
            & (G * head / sound2 <= unit.maximum_compressibility_fraction)
            & (power >= -1e-8)
            & (entropy >= -1e-8)
            & torch.isfinite(power)
        )
        reduced = q / speed.clamp_min(1e-12)
        for table in curves.values():
            if table is not None:
                valid = valid & (closed | ((reduced >= table[0][0]) & (reduced <= table[0][-1])))
        if unit.card.max_shaft_power_w is not None:
            valid = valid & (power <= unit.card.max_shaft_power_w)
        npsh = q * 0
        if unit.vapor_pressure_pa is not None:
            npsh = (source["p"] - unit.vapor_pressure_pa) / (source["rho"] * G)
            required = q * 0 + (unit.card.npsh_required_m or 0) * speed.square()
            if curves["npsh_curve"] is not None:
                required = pump_curve(curves, "npsh_curve", reduced) * speed.square()
            valid = valid & (
                closed
                | (
                    (npsh >= required)
                    & (
                        (source["t"] - unit.vapor_pressure_temperature_k).abs()
                        <= unit.maximum_vapor_pressure_temperature_departure_k
                    )
                )
            )
        row = {
            "flow_m3_s": actual,
            "mass_rate_kg_s": mass,
            "head_m": head,
            "efficiency": eta,
            "shaft_power_w": power,
            "hydraulic_power_w": hydraulic,
            "pump_pressure_pa": pressure,
            "isentropic_temperature_k": temperatures[0],
            "pump_temperature_k": temperatures[1],
            "delivery_temperature_k": temperatures[2],
            "entropy_production_w_k": entropy,
            "valve_pressure_drop_pa": torch.where(closed, pressure * 0, pressure - target["p"]),
            "valve_closed": closed,
            "npsh_available_m": npsh,
        }
        # Only cancellation-prone enthalpy rows receive an arithmetic roundoff
        # scale. Chemical potentials, entropy, pressure and closed-valve
        # temperature equations retain the caller's original stopping budget.
        precision = (
            2 + max(len(inlet_beta), *(trace.count for trace in self.traces))
        ) * torch.finfo(y.dtype).eps
        hin_abs = (inlet_beta * source["q"]["molar_enthalpy_j_mol"].abs()).sum()
        zero = q * 0
        limits = torch.stack(
            (
                precision * (his.abs() + hin_abs + (mw * G * head).abs()) / scale,
                zero,
                torch.where(
                    closed,
                    zero,
                    precision
                    * (hp.abs() + hin_abs + (his.abs() + hin_abs) / eta.clamp_min(1e-8))
                    / scale,
                ),
                torch.where(closed, zero, precision * (hd.abs() + hp.abs()) / scale),
                zero,
            )
        ).clamp_max(1e-10)  # At most one tenth of the unchanged hard transport limit.
        limits = torch.cat((limits, y.new_zeros(self.count - 5)))
        return {a: -first, b: last}, rows, row, snapshots, valid, limits

    def replacements(self, snapshots, *, closed=False):
        """Audit solved PS/PH allocations; re-seed the common solve after a split.

        No nested flash participates in differentiation or alters solved flows.
        """
        replacements = []
        changed = False
        f = self.model.fluid
        for index, (p, t, z, beta, x, tags) in enumerate(snapshots):
            qs = [f.tensor_homogeneous(p, t, xx, tag) for xx, tag in zip(x, tags)]
            chemical = x.clamp_min(1e-100).log() + torch.stack(
                [q["log_fugacity_coefficients"] for q in qs]
            )
            if (
                float((chemical[:, self.active] - chemical[-1, self.active]).abs().max().cpu())
                > 1e-7
            ):
                raise PropertyModelError("Liquid-mixture pump has unequal fugacities")
            minimum, _, _ = f._stability(p, t, z, chemical[-1], self.active)
            seed = {
                "pressure_pa": p,
                "temperature_k": t,
                "phase_count": len(beta),
                "phases": [
                    {**q, "molar_fraction": weight, "mole_fractions": xx}
                    for q, weight, xx in zip(qs, beta, x)
                ],
            }
            duplicate = any(
                float((x[j] - x[k]).abs().max().cpu()) < 1e-8
                and float(
                    (qs[j]["compressibility_factor"] - qs[k]["compressibility_factor"]).abs().cpu()
                )
                < 1e-8
                for j in range(len(beta))
                for k in range(j)
            )
            if (not closed or index == 0) and (float(minimum.cpu()) < -1e-8 or duplicate):
                seed = f.flash_pt(p, t, z)
                changed = True
            if (not closed or index == 0) and any(
                float(q["density_kg_m3"].cpu()) < self.unit.minimum_density_kg_m3
                for q in seed["phases"]
            ):
                raise EquipmentOperatingError(
                    f"Pump {self.unit.name!r} equilibrium trace leaves the dense liquid domain"
                )
            replacements.append(seed)
        return replacements if changed else None

    def failed_branch_seeds(self, snapshots):
        """Rebuild algebraic phase branches before unrelated stock UV retries.

        A wrong branch can fail PS/PH before the converged-state TPD check is
        reached. Prefer the known inlet assemblage when a previous rebranch
        failed; otherwise PT-flash the trial coordinates only as a Newton seed.
        Every candidate is solved and audited again by the common network.
        """
        if any(trace.count != self.base_seed["phase_count"] for trace in self.traces):
            return [self.base_seed] * 3
        f = self.model.fluid
        seeds, changed = [], False
        for trace, (p, t, z, _beta, _x, _tags) in zip(self.traces, snapshots):
            try:
                seed = f.flash_pt(p, t, z)
            except (PropertyModelError, HydraulicConvergenceError):
                return None
            if any(
                float(q["density_kg_m3"].cpu()) < self.unit.minimum_density_kg_m3
                for q in seed["phases"]
            ):
                return None
            changed = changed or seed["phase_count"] != trace.count
            seeds.append(seed)
        return seeds if changed else None

    @staticmethod
    def serialize(snapshots):
        return [
            {
                "pressure_pa": float(p.cpu()),
                "temperature_k": float(t.cpu()),
                "phase_fractions": beta.cpu().tolist(),
                "phase_mole_fractions": x.cpu().tolist(),
                "phase_roots": list(tags),
            }
            for p, t, _z, beta, x, tags in snapshots
        ]
