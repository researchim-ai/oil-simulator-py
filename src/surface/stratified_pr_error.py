"""Filtered Euler/trapezoidal error probe; no extrapolated accepted stocks.

The primary trajectory retains two conservative backward Euler half steps.
A single linear correction on the second half estimates its difference from
a trapezoidal companion; twice that defect controls the full macro interval.
Wall temperature is eliminated analytically for that companion as well.
The caller bootstraps cold states/events and falls back on topology changes.
"""

import math
from dataclasses import replace

import torch


def endpoint_rates(model, probe):
    _state, _y, _order, _seeds, data, problems = probe
    rates = {}
    for name, vessel in model.vessels.items():
        point = data["vessels"][name]
        mass_out = (point["out_n"] * model.fluid.molar_masses).sum()
        energy_out = (
            point["out_e"] + mass_out * vessel.column.gravity_m_s2 * vessel.column.elevation_m
        )
        rates[name] = (
            (problems[name][6] - point["out_n"] + data["component_net"][name]).detach().clone(),
            float(
                (problems[name][7] - energy_out + point["heat"] + data["energy_net"][name]).cpu()
            ),
            float(((point["ambient"] - point["heat"]) / vessel.wall.heat_capacity_j_k).cpu()),
        )
    return rates


def temporal_probe(model, old, candidate, dt, commands, old_rates, probe, *, macro_start=None):
    """Return the original engineering error norm, or None to bootstrap safely."""
    _state, y, order, seeds, original, _problems = probe
    before, after = dict(old.vessels), dict(candidate.vessels)
    if macro_start is not None:
        start = dict(macro_start.vessels)
        if any(start[n].column.phase_roots != before[n].column.phase_roots for n in before):
            return None
    if any(before[n].column.phase_roots != after[n].column.phase_roots for n in before):
        return None
    anchors = {}
    for name, state in before.items():
        dn, de, dw = old_rates[name]
        n = model.fluid.tensor(state.column.component_moles) + 0.5 * dt * dn
        if (
            not bool(torch.isfinite(n).all().cpu())
            or bool((n < 0).any().cpu())
            or not torch.equal(n > 0, model.fluid.tensor(state.column.component_moles) > 0)
        ):
            return None
        # These are affine RHS anchors, not physical/validated vessel states.
        # They are consumed only by _problem; never inspected or published.
        anchors[name] = replace(
            state,
            column=replace(
                state.column,
                component_moles=tuple(n.cpu().tolist()),
                total_energy_j=state.column.total_energy_j + 0.5 * dt * de,
            ),
            wall_temperature_k=state.wall_temperature_k + 0.5 * dt * dw,
        )
    _initial, lower, upper, evaluate, _bodies = model._problem(
        anchors, seeds, dt / 2, order, commands
    )

    def residual(z):
        return evaluate(z)[0]

    with torch.no_grad():
        r = residual(y)
        matrix = torch.func.jacfwd(residual)(y)
        delta, info = torch.linalg.solve_ex(matrix, -r, check_errors=False)
        linear_error = matrix @ delta + r
        if not bool(
            (
                (info == 0)
                & torch.isfinite(delta).all()
                & torch.isfinite(linear_error).all()
                & (linear_error.abs().max() <= 1e-12 + 1e-8 * r.abs().max())
            ).cpu()
        ):
            return None
        trial = y + delta
        if not bool(((trial >= lower).all() & (trial <= upper).all()).cpu()):
            return None
        low = evaluate(trial)[1]
        if not all(bool(p["valid"].cpu()) for p in low["transfers"]):
            return None
        error = 0.0
        for name, vessel in model.vessels.items():
            column = vessel.column
            point = low["vessels"][name]
            q = point["data"]
            if not all(bool(p["valid"].all().cpu()) for p in q["properties"]):
                return None
            if not all(bool(p["valid"].cpu()) for p in point["ports"]):
                return None
            old_q = original["vessels"][name]["data"]
            n = model.fluid.tensor(after[name].column.component_moles)
            energy = (
                q["internal_energy_j"]
                + q["relative_potential_energy_j"]
                + (q["component_moles"] * model.fluid.molar_masses).sum()
                * column.gravity_m_s2
                * column.elevation_m
            )
            relative = (
                after[name].column.total_energy_j
                - float((n * model.fluid.molar_masses).sum().cpu())
                * column.gravity_m_s2
                * column.elevation_m
            )

            def mean_p(data, column=column):
                return (
                    sum((w * p).sum() for w, p in zip(data["weights"], data["pressures"]))
                    / column.geometry.volume_m3
                )

            pa, pb = mean_p(old_q), mean_p(q)
            ratios = [
                (q["temperature"] - after[name].column.temperature_k).abs()
                / model.temperature_tolerance_k,
                (point["wall"] - after[name].wall_temperature_k).abs()
                / model.temperature_tolerance_k,
                (
                    (q["component_moles"] - n).abs() / (1e-9 + model.relative_tolerance * n.sum())
                ).max(),
                (energy - after[name].column.total_energy_j).abs()
                / (1e-5 + model.relative_tolerance * max(1, abs(relative))),
                (pa - pb).abs() / (1 + model.relative_tolerance * torch.maximum(pa, pb)),
            ]
            if q["interfaces"].numel():
                ratios.append(
                    (
                        (q["interfaces"] - old_q["interfaces"]).abs()
                        / (1e-8 + model.relative_tolerance * column.geometry.height_m)
                    ).max()
                )
            ratio = float(torch.stack(ratios).max().cpu())
            if not math.isfinite(ratio):
                return None
            error = max(error, ratio)
    return error
