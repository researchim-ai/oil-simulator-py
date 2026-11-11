"""Local UV capillary equilibrium on the actual resolved interface energy.

A coordinate transaction conserves each cell's N/V/M and the WHOLE line's
U+KE+F. Neighbours stay fixed while a cell changes phase support. Candidate
coexistence solves chemical equality, the variational pressure jump and UV
jointly; a bulk flash provides seeds only. This is instantaneous equilibrium,
not a calibrated finite nucleation rate or a global-minimum certificate.
"""

import math
from dataclasses import replace

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .peng_robinson import R
from .pr_capillary_pressure import chemical_potentials_tv
from .pr_homogeneous_uv import caloric_tv, recover_homogeneous_phases


def thermal_entropy(fluid, t, vm, x, q):
    # Algebraic TV form of the same caloric PR entropy (pressure cancels).
    b = q["b"]
    ratio = torch.log1p(2 * math.sqrt(2) * b / (vm + (1 - math.sqrt(2)) * b))
    rt = torch.sqrt(t[..., None] / fluid.tc)
    m = 1 + fluid.kappa * (1 - rt)
    ai = fluid.a0 * m.square()
    ait = -fluid.a0 * fluid.kappa * m / (fluid.tc * rt)
    aij = torch.sqrt(ai[..., :, None] * ai[..., None, :]) * (1 - fluid.kij)
    at = (
        x[..., :, None]
        * x[..., None, :]
        * aij
        * 0.5
        * (ait[..., :, None] / ai[..., :, None] + ait[..., None, :] / ai[..., None, :])
    ).sum((-2, -1))
    mixing = torch.where(x > 0, x * x.clamp_min(1e-300).log(), 0.0).sum(-1)
    return (
        (x * fluid.cp).sum(-1) * torch.log(t / fluid.tref)
        - R * mixing
        + R * torch.log((vm - b) * fluid.pref / (R * t))
        + at / (2 * math.sqrt(2) * b) * ratio
    )


class PRCapillaryEquilibrium:
    def __init__(self, source, *, maximum_iterations=48, maximum_seeds=12):
        if source.phase_support != "active":
            raise ValueError("Capillary equilibrium needs exact active phase support")
        if (
            source.wetting is not None
            and source.wetting.scale > 0
            and source.wetting.dry_extension is None
        ):
            raise ValueError("Phase changes require finite declared dry-wall energy")
        if (
            type(maximum_iterations) is not int
            or maximum_iterations < 1
            or type(maximum_seeds) is not int
            or maximum_seeds < 1
        ):
            raise ValueError("Positive capillary equilibrium solver budgets required")
        self.source, self.pipe, self.fluid = source, source.pipe, source.pipe.fluid
        expected_edges = tuple((i, i + 1) for i in range(len(self.pipe.lengths) - 1))
        if self.pipe.boundary_mode == "periodic":
            expected_edges += ((len(self.pipe.lengths) - 1, 0),)
        expected_fixed = tuple(
            0 if side == 0 else len(self.pipe.lengths) - 1
            for side, height in enumerate(source.boundary_heights)
            if height is not None
        )
        if (
            source.geometry_edges != expected_edges
            or tuple(source.fixed_indices.cpu().tolist()) != expected_fixed
        ):
            raise ValueError("Capillary UV equilibrium requires declared physical line geometry")
        self.maximum_iterations, self.maximum_seeds = maximum_iterations, maximum_seeds
        self._areas = tuple(length * source.width for length in self.pipe.lengths)
        stencils = [[] for _ in self.pipe.lengths]
        offsets = source.height_offsets.cpu().tolist()
        for edge, (left, right) in enumerate(source.geometry_edges):
            stencils[left].append((-1.0, right, offsets[edge], edge))
            stencils[right].append((1.0, left, offsets[edge], edge))
        for k, cell in enumerate(source.fixed_indices.cpu().tolist()):
            edge = len(source.geometry_edges) + k
            stencils[cell].append((1.0, None, offsets[edge], edge))
        self._stencils = tuple(tuple(rows) for rows in stencils)
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_capillary_equilibrium_uv_v1",
            "source": self.source.to_dict(),
            "pipe": self.pipe.to_dict(),
            "maximum_iterations": self.maximum_iterations,
            "maximum_seeds": self.maximum_seeds,
        }

    @classmethod
    def from_dict(cls, raw, *, device="cpu"):
        from .pr_active_multifluid_pipe import PRActiveMultifluidPipe
        from .pr_capillary_pressure import PRCapillaryPressure

        if (
            set(raw) != {"model", "source", "pipe", "maximum_iterations", "maximum_seeds"}
            or raw["model"] != "pr_capillary_equilibrium_uv_v1"
        ):
            raise ValueError("Unknown capillary equilibrium model")
        pipe = PRActiveMultifluidPipe.from_dict(raw["pipe"], device=device)
        return cls(
            PRCapillaryPressure.from_dict(pipe, raw["source"]),
            maximum_iterations=raw["maximum_iterations"],
            maximum_seeds=raw["maximum_seeds"],
        )

    def _geometry(self, state, index):
        s, f = self.source, self.fluid
        coefficients = []
        constants = []
        weights = []
        for sign, neighbour, offset, edge in self._stencils[index]:
            phase = dict(state.cells[neighbour]).get(s.liquid) if neighbour is not None else None
            height = (
                (phase.volume_m3 if phase is not None else 0.0) / self._areas[neighbour]
                if neighbour is not None
                else 0.0
            )
            coefficients.append(sign)
            constants.append(-sign * height + offset)
            weights.append(s.weights[edge])
        signs = f.tensor(coefficients)
        offsets = f.tensor(constants)
        weights = torch.stack(weights) if weights else f.tensor([])

        def energy(vl):
            h = vl / s.area[index]
            result = 0.5 * (weights * (signs * h + offsets).square()).sum()
            if s.wetting is not None:
                result = result + s.area[index] * s.wetting.potential(h)
            return result

        def pressure(vl):
            h = vl / s.area[index]
            result = (weights * signs * (signs * h + offsets)).sum() / s.area[index]
            if s.wetting is not None:
                result = result + s.wetting.divided_difference(h, h)
            return result

        energy.pressure = pressure
        return energy

    def _cell(self, state, index, heat):
        f, p, s = self.fluid, self.pipe, self.source
        row = dict(state.cells[index])
        names = (s.liquid, s.gas)
        nc = len(f.names)
        amounts = f.tensor(
            [math.fsum(ph.component_moles[j] for ph in row.values()) for j in range(nc)]
        )
        active = torch.nonzero(amounts > 0).flatten()
        na = amounts[active]
        total = amounts.sum()
        mass = (amounts * f.molar_masses).sum()
        volume = f.tensor(p.volumes[index])
        momentum = math.fsum(ph.momentum_kg_m_s for ph in row.values())
        velocity = f.tensor(momentum) / mass
        kinetic = 0.5 * mass * velocity.square()
        capillary = self._geometry(state, index)
        oldvl = f.tensor(row[s.liquid].volume_m3 if s.liquid in row else 0.0)
        oldsurface = capillary(oldvl)
        energy = (
            f.tensor(math.fsum(ph.total_energy_j for ph in row.values()) + heat)
            + oldsurface
            - kinetic
        )
        guess_t = math.fsum(
            ph.temperature_k * sum(ph.component_moles) for ph in row.values()
        ) / float(total.cpu())
        guess_p = math.fsum(ph.pressure_pa * ph.volume_m3 for ph in row.values()) / float(
            volume.cpu()
        )
        escale = max(1.0, abs(float(energy.cpu())), float(total.cpu()) * R * guess_t)
        pscale = max(1.0, guess_p)
        candidates = []
        failures = []
        seeds = []
        # Exact single-phase endpoint candidates: no N=V=0 EOS is evaluated.
        for j, root in enumerate(("liquid", "vapor")):
            vl = volume if j == 0 else volume * 0
            target = float((energy - capillary(vl) + kinetic).cpu())
            try:
                phase = recover_homogeneous_phases(
                    f,
                    [amounts.cpu().tolist()],
                    [float(volume.cpu())],
                    [momentum],
                    [target],
                    [root],
                    [guess_t],
                )[0]
                q = f.homogeneous(phase.pressure_pa, phase.temperature_k, amounts / total, root)
                # Below the distinct-root region, label a sole root by its
                # density relative to PR's critical-volume scale, not by F.
                other = f.homogeneous(
                    phase.pressure_pa, phase.temperature_k, amounts / total, ("vapor", "liquid")[j]
                )
                vm = volume / total
                distinct = abs(
                    float((q["molar_volume_m3_mol"] - other["molar_volume_m3_mol"]).cpu())
                ) > 1e-7 * float(vm.cpu())
                critical = (amounts / total * (0.3074 * R * f.tc / f.pc)).sum()
                if not distinct and ((j == 0) != bool((vm < critical).cpu())):
                    continue
                # Incipient-phase TPD uses the pressure on the OTHER side
                # of the interface, derived from the current surface energy.
                cap = capillary.pressure(vl)
                target_pressure = f.tensor(phase.pressure_pa) + (cap if j == 1 else -cap)
                if not f.min_pressure_pa <= float(target_pressure.cpu()) <= f.max_pressure_pa:
                    raise PropertyModelError("Incipient phase pressure outside EOS domain")
                composition = amounts / total
                reference = (
                    composition.clamp_min(1e-300).log()
                    + q["log_fugacity_coefficients"]
                    + torch.log(f.tensor(phase.pressure_pa) / target_pressure)
                )
                opposite = ("vapor", "liquid")[j]
                if len(active) == 1:
                    incipient = f.homogeneous(
                        target_pressure, phase.temperature_k, composition, opposite
                    )
                    value = (
                        composition
                        * (
                            composition.clamp_min(1e-300).log()
                            + incipient["log_fugacity_coefficients"]
                            - reference
                        )
                    ).sum()
                else:
                    from .equilibrium_properties import EquilibriumFluid

                    class OppositeRoot:
                        def __getattr__(self, name):
                            return getattr(f, name)

                        def tensor_homogeneous(self, p, t, x, root=opposite):
                            return f.tensor_homogeneous(p, t, x, root)

                    value, _composition, incipient = EquilibriumFluid._stability(
                        OppositeRoot(),
                        target_pressure,
                        f.tensor(phase.temperature_k),
                        composition,
                        reference,
                        active,
                    )
                critical_new = (incipient["mole_fractions"] * (0.3074 * R * f.tc / f.pc)).sum()
                dense = bool((incipient["molar_volume_m3_mol"] < critical_new).cpu())
                belongs_to_opposite = dense if opposite == "liquid" else not dense
                if belongs_to_opposite and float(value.cpu()) < -2e-8:
                    raise PropertyModelError(
                        "Homogeneous candidate unstable to capillary phase appearance"
                    )
                entropy = float((total * q["molar_entropy_j_mol_k"]).cpu())
                candidates.append(
                    (
                        entropy,
                        ((names[j], phase),),
                        {
                            "topology": root,
                            "incipient_tpd_rt": float(value.cpu()),
                            "scaled_residual": 0.0,
                            "newton_iterations": 0,
                        },
                    )
                )
            except (PropertyModelError, ConservationError, HydraulicConvergenceError) as err:
                failures.append(str(err))

        def seed_from(phases):
            liquid = phases.get(s.liquid)
            gas = phases.get(s.gas)
            if liquid is None or gas is None:
                return
            allocation = f.tensor(liquid.component_moles)[active] / na
            a = liquid.volume_m3 / float(volume.cpu())
            if not bool(((allocation > 0) & (allocation < 1)).all().cpu()) or not 0 < a < 1:
                return
            seeds.append(
                torch.cat(
                    (
                        torch.logit(allocation),
                        f.tensor([math.log(a / (1 - a)), math.log(liquid.temperature_k / guess_t)]),
                    )
                )
            )

        seed_from(row)
        # Pure-fluid cooling can cross coexistence far from the old hot
        # pressure. Saturation/lever-rule states are predictors, never the
        # accepted capillary solution. Rank them by the actual UV target.
        if len(active) == 1:
            component = f.components[int(active[0].cpu())]
            critical = component.critical_temperature_k
            low = max(f.min_temperature_k, component.min_temperature_k)
            high = min(f.max_temperature_k, critical * (1 - 1e-5))
            predicted = []
            if low < high:
                for temperature in [low + (high - low) * r for r in (0.1, 0.3, 0.5, 0.7, 0.9)]:
                    try:
                        psat = component.saturation_at_temperature(temperature)["pressure_pa"]
                        qs = [
                            f.homogeneous(psat, temperature, amounts / total, root)
                            for root in ("liquid", "vapor")
                        ]
                        vl0, vg0 = [float(q["molar_volume_m3_mol"].cpu()) for q in qs]
                        mean = float((volume / total).cpu())
                        if not vl0 < mean < vg0:
                            continue
                        beta = (vg0 - mean) / (vg0 - vl0)
                        fraction = beta * vl0 / mean
                        guess = torch.cat(
                            (
                                torch.logit(na * 0 + beta),
                                f.tensor(
                                    [
                                        math.log(fraction / (1 - fraction)),
                                        math.log(temperature / guess_t),
                                    ]
                                ),
                            )
                        )
                        predicted_energy = total * (
                            beta * qs[0]["molar_internal_energy_j_mol"]
                            + (1 - beta) * qs[1]["molar_internal_energy_j_mol"]
                        ) + capillary(volume * fraction)
                        predicted.append((abs(float((predicted_energy - energy).cpu())), guess))
                    except (PropertyModelError, HydraulicConvergenceError):
                        continue
            seeds.extend(guess for _, guess in sorted(predicted, key=lambda pair: pair[0]))
        else:
            # Bulk UV is ONLY a branch predictor. Every accepted split below is
            # re-solved with the actual F and its pressure derivative.
            try:
                from .pr_phase_equilibration import PRPhaseEquilibration

                bulk = PRPhaseEquilibration(f).recover(
                    amounts.cpu().tolist(),
                    float((energy - oldsurface + kinetic).cpu()),
                    momentum,
                    float(volume.cpu()),
                    pressure_guess_pa=guess_p,
                    temperature_guess_k=guess_t,
                )
                if len(bulk.phases) > 2 or (
                    len(bulk.phases) == 2 and all(root == "liquid" for root in bulk.phase_roots)
                ):
                    raise ValueError(
                        "Two-slot capillary VLE does not support this equilibrium phase topology"
                    )
                if len(bulk.phases) == 2:
                    phases = sorted(
                        bulk.phases,
                        key=lambda ph: sum(ph.component_moles) / ph.volume_m3,
                        reverse=True,
                    )
                    seed_from(dict(zip(names, phases)))
            except (PropertyModelError, ConservationError, HydraulicConvergenceError) as err:
                failures.append(str(err))
        # Distinct homogeneous roots at nearby temperatures provide an
        # additional predictor when the old registry contains only one phase.
        for factor in (1.0, 0.9, 1.1):
            t = guess_t * factor
            if not f.min_temperature_k <= t <= f.max_temperature_k:
                continue
            try:
                qs = [
                    f.homogeneous(guess_p, t, amounts / total, root) for root in ("liquid", "vapor")
                ]
                vl0, vg0 = [float(q["molar_volume_m3_mol"].cpu()) for q in qs]
                mean = float((volume / total).cpu())
                if not vl0 < mean < vg0 or vg0 - vl0 < 1e-7 * mean:
                    continue
                beta = (vg0 - mean) / (vg0 - vl0)
                a = beta * vl0 / mean
                seeds.append(
                    torch.cat(
                        (
                            torch.logit(na * 0 + beta),
                            f.tensor([math.log(a / (1 - a)), math.log(t / guess_t)]),
                        )
                    )
                )
            except (PropertyModelError, HydraulicConvergenceError):
                pass

        def evaluate(y):
            allocations = torch.sigmoid(y[: len(active)])
            nl = torch.zeros_like(amounts).index_copy(0, active, na * allocations)
            numbers = torch.stack((nl, amounts - nl))
            counts = numbers.sum(-1)
            alpha = torch.sigmoid(y[-2])
            volumes = volume * torch.stack((alpha, 1 - alpha))
            t = f.tensor(guess_t) * y[-1].exp()
            ts = t.expand(2)
            vm = volumes / counts
            x = numbers / counts[:, None]
            q = caloric_tv(f, ts, vm, x)
            mu = chemical_potentials_tv(f, ts, vm, x)
            vl = volumes[0]
            cap = capillary.pressure(vl)
            r = torch.cat(
                (
                    (mu[0, active] - mu[1, active]) / (R * t),
                    ((q["p"][0] - q["p"][1] - cap) / pscale).reshape(1),
                    ((counts * q["u"]).sum() + capillary(vl) - energy).reshape(1) / escale,
                )
            )
            valid = (
                torch.isfinite(r).all()
                & (t >= f.min_temperature_k)
                & (t <= f.max_temperature_k)
                & (counts > 0).all()
                & (volumes > 0).all()
                & (vm > q["b"]).all()
                & (q["dpdv"] < 0).all()
                & (q["cv"] > 0).all()
                & (q["p"] >= f.min_pressure_pa).all()
                & (q["p"] <= f.max_pressure_pa).all()
                & (vm[0] < vm[1] * (1 - 1e-7))
            )
            return r, valid, numbers, volumes, ts, q

        if len(active) == 1:
            # For a pure fluid with convex geometric F and no wall potential,
            # a stable endpoint needs no coexistence Newton search. Otherwise
            # prioritise the UV-nearest saturation predictors.
            if candidates and s.wetting is None:
                seeds = []
            else:
                seeds = sorted(seeds, key=lambda seed: abs(float(evaluate(seed)[0][-1].cpu())))[
                    : min(4, self.maximum_seeds)
                ]
        tried = 0
        for seed in seeds[: self.maximum_seeds]:
            tried += 1
            y = seed.detach()
            try:
                for iteration in range(self.maximum_iterations):
                    data = evaluate(y)
                    r = data[0]
                    norm = float(r.abs().max().cpu())
                    if not bool(data[1].cpu()):
                        raise PropertyModelError("Inadmissible capillary coexistence predictor")
                    if norm < 2e-13:
                        break
                    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
                    correction, info = torch.linalg.solve_ex(jac, -r)
                    if int(info.cpu()) or not bool(torch.isfinite(correction).all().cpu()):
                        raise HydraulicConvergenceError("Singular capillary equilibrium tangent")
                    for power in range(18):
                        trial = y + correction * 2.0 ** (-power)
                        test = evaluate(trial)
                        if bool(test[1].cpu()) and float(test[0].abs().max().cpu()) < norm:
                            y = trial.detach()
                            break
                    else:
                        raise HydraulicConvergenceError("Capillary equilibrium line search failed")
                else:
                    raise HydraulicConvergenceError(
                        "Capillary equilibrium iteration budget exhausted"
                    )
                _, _, numbers, volumes, temps, q = evaluate(y)
                if float(evaluate(y)[0].abs().max().cpu()) >= 2e-13:
                    raise HydraulicConvergenceError("Unconverged capillary candidate")

                temperature_coordinate = y[-1:].detach()

                def helmholtz_at_fixed_temperature(
                    z, temperature_coordinate=temperature_coordinate
                ):
                    yz = torch.cat((z, temperature_coordinate))
                    _, _, ns, vs, ts, qs = evaluate(yz)
                    xs = ns / ns.sum(-1)[:, None]
                    entropy = thermal_entropy(f, ts, vs / ns.sum(-1), xs, qs)
                    return (
                        (ns.sum(-1) * (qs["u"] - ts * entropy)).sum() + capillary(vs[0])
                    ) / escale

                hessian = torch.func.hessian(helmholtz_at_fixed_temperature)(y[:-1])
                eigenvalues = torch.linalg.eigvalsh((hessian + hessian.T) / 2)
                if float(eigenvalues.min().cpu()) < -2e-9 * max(
                    1.0, float(eigenvalues.abs().max().cpu())
                ):
                    raise PropertyModelError("Unstable capillary coexistence stationary point")
                masses = (numbers * f.molar_masses).sum(-1)
                energies = numbers.sum(-1) * q["u"] + 0.5 * masses * velocity.square()
                phases = recover_homogeneous_phases(
                    f,
                    numbers.cpu().tolist(),
                    volumes.cpu().tolist(),
                    (masses * velocity).cpu().tolist(),
                    energies.cpu().tolist(),
                    ["liquid", "vapor"],
                    temps.cpu().tolist(),
                )
                entropy = sum(
                    float(
                        (
                            numbers.sum(-1)[j]
                            * f.homogeneous(
                                ph.pressure_pa,
                                ph.temperature_k,
                                numbers[j] / numbers[j].sum(),
                                ("liquid", "vapor")[j],
                            )["molar_entropy_j_mol_k"]
                        ).cpu()
                    )
                    for j, ph in enumerate(phases)
                )
                candidates.append(
                    (
                        entropy,
                        tuple(zip(names, phases)),
                        {
                            "topology": "liquid_vapor",
                            "minimum_helmholtz_eigenvalue": float(eigenvalues.min().cpu()),
                            "scaled_residual": float(evaluate(y)[0].abs().max().cpu()),
                            "newton_iterations": iteration,
                        },
                    )
                )
                if len(active) == 1 and s.wetting is None:
                    break
            except (PropertyModelError, ConservationError, HydraulicConvergenceError) as err:
                failures.append(str(err))
        if not candidates:
            raise HydraulicConvergenceError(
                "No admissible capillary UV candidate: " + "; ".join(failures[-3:])
            )
        selected = max(candidates, key=lambda row: row[0])
        from dataclasses import asdict

        final_liquid = dict(selected[1]).get(s.liquid)
        final_vl = f.tensor(final_liquid.volume_m3 if final_liquid is not None else 0.0)
        return selected[1], {
            **selected[2],
            "cell": index,
            "surface_energy_change_j": float((capillary(final_vl) - oldsurface).cpu()),
            "phase_stocks": {name: asdict(phase) for name, phase in selected[1]},
            "candidate_count": len(candidates),
            "coexistence_predictors": tried,
            "failed_candidates": failures,
            "entropy_j_k": selected[0],
            "candidate_entropies_j_k": [c[0] for c in candidates],
            "phase_count_before": len(row),
            "phase_count_after": len(selected[1]),
        }

    def project(self, state, *, heat_j=None, heat_temperature_k=None):
        p, s = self.pipe, self.source
        if _digest(self.to_dict()) != self.model_id:
            raise ValueError("Changed capillary equilibrium model")
        before = p.inspect(state)
        surface0 = s.surface_energy(state)
        count = len(state.cells)
        heat = [0.0] * count if heat_j is None else [_number(v, "cell heat") for v in heat_j]
        if len(heat) != count:
            raise ValueError("One heat value per physical cell required")
        if any(heat) and heat_temperature_k is None:
            raise ValueError("External heat requires reservoir temperature")
        bath = (
            None
            if heat_temperature_k is None
            else _number(heat_temperature_k, "heat reservoir temperature", positive=True)
        )
        current = state
        reports = []
        for i, q in enumerate(heat):
            phases, report = self._cell(current, i, q)
            rows = list(current.cells)
            rows[i] = phases
            trial = replace(current, cells=tuple(rows))
            # Atomic local acceptance; geometry/entropy and actual stocks are
            # checked before the next coordinate can consume this state.
            initial_entropy = math.fsum(
                sum(phase.component_moles)
                * float(
                    self.fluid.homogeneous(
                        phase.pressure_pa,
                        phase.temperature_k,
                        self.fluid.tensor(phase.component_moles) / sum(phase.component_moles),
                        "liquid" if name == s.liquid else "vapor",
                    )["molar_entropy_j_mol_k"].cpu()
                )
                for name, phase in current.cells[i]
            )
            entropy = (
                report["entropy_j_k"] - initial_entropy - (q / bath if bath is not None else 0.0)
            )
            if (
                entropy
                < -256 * (math.ulp(initial_entropy) + math.ulp(report["entropy_j_k"])) - 1e-16
            ):
                raise ConservationError("Capillary phase equilibrium decreases entropy")
            report["entropy_production_j_k"] = entropy
            current = trial
            reports.append(report)
        after = p.inspect(current)
        surface1 = s.surface_energy(current)
        a, b = p._totals(state), p._totals(current)
        nc = len(self.fluid.names)
        residual = [b[j] - a[j] for j in range(nc)] + [
            b[-2] - a[-2] + surface1 - surface0 - math.fsum(heat),
            b[-1] - a[-1],
        ]
        phases = [ph for snapshot in (state, current) for c in snapshot.cells for _, ph in c]
        # The resultant momentum or reference-dependent energy can cancel.
        # Floating-point bounds must use their actual phasic operands.
        energy_operands = [ph.total_energy_j for ph in phases] + [*heat, surface0, surface1]
        momentum_operands = [ph.momentum_kg_m_s for ph in phases]
        allowances = [
            256 * math.fsum(math.ulp(ph.component_moles[j]) for ph in phases) + 1e-24
            for j in range(nc)
        ]
        allowances.extend(
            (
                512 * math.fsum(math.ulp(v) for v in energy_operands) + 1e-18,
                256 * math.fsum(math.ulp(v) for v in momentum_operands) + 1e-24,
            )
        )
        if any(abs(v) > bound for v, bound in zip(residual, allowances)):
            raise ConservationError(
                f"Capillary phase equilibrium violates N/M/(E+F): {residual}, {allowances}"
            )
        return current, {
            "model_id": self.model_id,
            "state": current.to_dict(),
            "cells": reports,
            "heat_j": heat,
            "heat_temperature_k": bath,
            "surface_energy_before_j": surface0,
            "surface_energy_after_j": surface1,
            "conservation_residuals": residual,
            "roundoff_allowances": allowances,
            "entropy_production_j_k": after["entropy"]
            - before["entropy"]
            - (math.fsum(heat) / bath if bath is not None else 0.0),
            "qualification": "Sequential local instantaneous capillary UV equilibrium; bounded multistart entropy selection, not global stability certification or finite-rate nucleation",
        }

    def equilibrium_residual(self, state):
        """Whole-line chemical, thermal and variational pressure defect."""
        s, f = self.source, self.fluid
        h = (
            f.tensor(
                [dict(c)[s.liquid].volume_m3 if s.liquid in dict(c) else 0.0 for c in state.cells]
            )
            / s.area
        )
        cap = s.discrete_pressure(h, h)
        indices = [i for i, c in enumerate(state.cells) if len(c) == 2]
        if not indices:
            return 0.0
        phases = [[dict(state.cells[i])[name] for name in (s.liquid, s.gas)] for i in indices]
        n = f.tensor([[p.component_moles for p in row] for row in phases])
        volumes = f.tensor([[p.volume_m3 for p in row] for row in phases])
        temperatures = f.tensor([[p.temperature_k for p in row] for row in phases])
        pressures = f.tensor([[p.pressure_pa for p in row] for row in phases])
        amounts = n.sum(-1)
        x = n / amounts[..., None]
        mu = chemical_potentials_tv(f, temperatures, volumes / amounts, x)
        supports = n.sum(1) > 0
        chemical = (
            torch.where(supports, (mu[:, 0] - mu[:, 1]) / (R * temperatures.mean(-1)[:, None]), 0.0)
            .abs()
            .max()
        )
        thermal = ((temperatures[:, 0] - temperatures[:, 1]) / temperatures.mean(-1)).abs().max()
        mechanical = (
            (
                (pressures[:, 0] - pressures[:, 1] - cap[indices])
                / pressures.abs().amax(-1).clamp_min(1.0)
            )
            .abs()
            .max()
        )
        return float(torch.stack((chemical, thermal, mechanical)).max().cpu())

    def equilibrate(self, state, *, heat_j=None, heat_temperature_k=None, maximum_sweeps=12):
        """Converge the complete nonlocal interface, retaining atomic stocks.

        Each local UV transaction conserves the augmented line energy. Heat
        is applied once, never once per sweep. Convergence also requires
        stationary support/volumes, so a pure cell is rechecked after a
        neighbour changes its interface. No partially converged state escapes.
        """
        if type(maximum_sweeps) is not int or maximum_sweeps < 1:
            raise ValueError("Positive capillary sweep budget required")
        tolerance = 2e-11
        current = state
        sweeps = []
        for iteration in range(maximum_sweeps):
            end, report = self.project(
                current,
                heat_j=heat_j if iteration == 0 else None,
                heat_temperature_k=heat_temperature_k if iteration == 0 else None,
            )
            sweeps.append(report)
            registry = all(
                tuple(dict(a)) == tuple(dict(b)) for a, b in zip(current.cells, end.cells)
            )
            change = max(
                abs(
                    (
                        dict(a)[self.source.liquid].volume_m3
                        if self.source.liquid in dict(a)
                        else 0.0
                    )
                    - (
                        dict(b)[self.source.liquid].volume_m3
                        if self.source.liquid in dict(b)
                        else 0.0
                    )
                )
                / volume
                for a, b, volume in zip(current.cells, end.cells, self.pipe.volumes)
            )
            defect = self.equilibrium_residual(end)
            current = end
            if registry and change <= tolerance and defect <= tolerance:
                heat = sweeps[0]["heat_j"]
                bath = sweeps[0]["heat_temperature_k"]
                a, b = self.pipe._totals(state), self.pipe._totals(current)
                f0, f1 = self.source.surface_energy(state), self.source.surface_energy(current)
                nc = len(self.fluid.names)
                residual = [b[j] - a[j] for j in range(nc)] + [
                    math.fsum((b[-2], -a[-2], f1, -f0, -math.fsum(heat))),
                    b[-1] - a[-1],
                ]
                allowances = [
                    math.fsum(row["roundoff_allowances"][j] for row in sweeps)
                    for j in range(nc + 2)
                ]
                allowances[-2] += 128 * math.fsum(
                    math.ulp(v) for v in (a[-2], b[-2], f0, f1, math.fsum(heat))
                )
                if any(abs(v) > bound for v, bound in zip(residual, allowances)):
                    raise ConservationError("Converged capillary line balance failed")
                return current, {
                    "model_id": self.model_id,
                    "state": current.to_dict(),
                    "equilibrium_sweeps": sweeps,
                    "converged": True,
                    "maximum_sweeps": maximum_sweeps,
                    "relative_tolerance": tolerance,
                    "maximum_equilibrium_residual": defect,
                    "maximum_fraction_change": change,
                    "heat_j": heat,
                    "heat_temperature_k": bath,
                    "surface_energy_before_j": f0,
                    "surface_energy_after_j": f1,
                    "conservation_residuals": residual,
                    "roundoff_allowances": allowances,
                    "entropy_production_j_k": math.fsum(
                        row["entropy_production_j_k"] for row in sweeps
                    ),
                    "qualification": "Converged nonlocal capillary UV coordinate equilibrium at fixed cell N/V/M and total augmented energy; bounded multistart, no global-minimum certificate or finite-rate nucleation claim",
                }
        raise HydraulicConvergenceError("Nonlocal capillary equilibrium sweep budget exhausted")
