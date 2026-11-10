"""Independent scalar PR and finite-volume audit of capillary normal sources.

Uses no runtime source residual, force, chemical potential or energy helper.
This verifier does not qualify the missing axial capillary transport operator.
"""

import math

import numpy as np
from check_stratified_pr import Reference

from surface.peng_robinson import R


def close(actual, expected, label, *, atol=1e-22, rtol=1e-9):
    a, b = np.asarray(actual), np.asarray(expected)
    if (
        a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or np.any(abs(a - b) > atol + rtol * np.maximum(abs(a), abs(b)))
    ):
        raise ValueError("Capillary source audit differs: " + label)


def audit(model, before, after, report):
    if report["source_model_id"] != model.model_id or report["source_model"] != model.to_dict():
        from surface.operation import _digest

        if report["source_model_id"] != model.model_id or _digest(
            report["source_model"]
        ) != _digest(model.to_dict()):
            raise ValueError("Capillary source constitutive model differs")
    pipe = model.pipe
    f = pipe.fluid
    count = len(pipe.lengths)
    from surface.operation import _digest

    if (
        before.time_s != after.time_s
        or report["clock_advanced"]
        or _digest(report["state"]) != _digest(after.to_dict())
        or before.model_id != pipe.model_id
        or after.model_id != pipe.model_id
    ):
        raise ValueError("Capillary normal source clock/identity differs")
    dt = report["dt_s"]
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid capillary source interval")
    mw = np.array([c.molar_mass_kg_mol for c in f.components])
    selected = tuple(
        i for i, c in enumerate(before.cells) if set(dict(c)) == {model.liquid, model.gas}
    )
    if model.phase_support == "active" and report.get("active_cells") != list(selected):
        raise ValueError("Normal-source active registry differs")
    stocks = []
    entropy = []
    for state in (before, after):
        rows = []
        ent = []
        for cell in state.cells:
            if set(dict(cell)) - {model.liquid, model.gas} or (
                model.phase_support != "active" and set(dict(cell)) != {model.liquid, model.gas}
            ):
                raise ValueError("Capillary phase support differs")
            row = []
            for name, root in ((model.liquid, "liquid"), (model.gas, "vapor")):
                p = dict(cell).get(name)
                if p is None:
                    row.append((None, np.zeros(len(f.names)), np.zeros(len(f.names))))
                    continue
                n = np.array(p.component_moles)
                amount = n.sum()
                x = n / amount
                ref = Reference(f, np.flatnonzero(x > 0))
                q = ref.state(p.pressure_pa, p.temperature_k, x, root)
                energy = amount * q["u"] + 0.5 * p.momentum_kg_m_s**2 / np.dot(n, mw)
                close(p.volume_m3, amount * q["vm"], "scalar PR volume", atol=1e-25, rtol=1e-10)
                close(
                    p.total_energy_j,
                    energy,
                    "scalar PR energy",
                    atol=256 * math.ulp(energy) + 1e-22,
                    rtol=0,
                )
                ent.append(amount * q["s"])
                mu = np.zeros(len(x))
                active = ref.active
                ideal = np.array([c.ideal_cp_j_mol_k for c in f.components]) * (
                    p.temperature_k - f.tref - p.temperature_k * math.log(p.temperature_k / f.tref)
                )
                mu[active] = ideal[active] + R * p.temperature_k * (
                    np.log(x[active] * p.pressure_pa / f.pref) + q["phi"]
                )
                row.append((p, n, mu))
            rows.append(row)
        stocks.append(rows)
        entropy.append(math.fsum(ent))
    area = np.array(pipe.lengths) * model.width
    heights = [
        np.array([row[0][0].volume_m3 if row[0][0] is not None else 0.0 for row in state]) / area
        for state in stocks
    ]
    edges = (
        list(model.surface_edges)
        if hasattr(model, "surface_edges")
        else [(i, i + 1) for i in range(count - 1)]
    )
    if pipe.boundary_mode == "periodic" and not hasattr(model, "surface_edges"):
        edges.append((count - 1, 0))
    cap = np.zeros(count)
    surface = []
    for h in heights:
        surface.append(
            math.fsum(
                0.5
                * model.sigma
                * model.width
                / (0.5 * (pipe.lengths[i] + pipe.lengths[j]))
                * (h[j] - h[i]) ** 2
                for i, j in edges
            )
        )
    mid = (heights[0] + heights[1]) / 2
    for i, j in edges:
        force = (
            model.sigma
            * model.width
            / (0.5 * (pipe.lengths[i] + pipe.lengths[j]))
            * (mid[i] - mid[j])
        )
        cap[i] += force / area[i]
        cap[j] -= force / area[j]
    for side, height in enumerate(model.boundary_heights):
        if height is not None:
            index = 0 if side == 0 else count - 1
            coefficient = model.sigma * model.width / model.boundary_distances[side]
            for k, h in enumerate(heights):
                surface[k] += 0.5 * coefficient * (h[index] - height) ** 2
            cap[index] += coefficient * (mid[index] - height) / area[index]
    if model.wetting is not None:
        from reference_wetting_potential import potential as wall_energy
        from reference_wetting_potential import pressure as wall_pressure

        params = model.wetting.to_dict()
        for i, h in enumerate(heights):
            surface[i] += float(np.dot(area, wall_energy(params, model.sigma, h)))
        cap += wall_pressure(params, model.sigma, *heights)
    if selected:
        close(
            report["capillary_pressure_pa"], cap[list(selected)], "variational pressure", atol=1e-9
        )
    elif report["volume_transfer_m3"]:
        raise ValueError("Interface-free normal source transferred volume")
    dtotal = []
    lower = []
    for i, (old, new) in enumerate(zip(*stocks, strict=True)):
        if i not in selected:
            if before.cells[i] != after.cells[i]:
                raise ValueError("Normal source changed an interface-free cell")
            continue
        ri = selected.index(i)
        pl0, nl0, _ = old[0]
        pg0, ng0, _ = old[1]
        pl, nl, mul = new[0]
        pg, ng, mug = new[1]
        dv = pl.volume_m3 - pl0.volume_m3
        dn = nl - nl0
        close(nl + ng, nl0 + ng0, "component conservation", atol=1e-24, rtol=1e-14)
        close(
            pl.volume_m3 + pg.volume_m3,
            pl0.volume_m3 + pg0.volume_m3,
            "volume conservation",
            atol=1e-25,
            rtol=1e-14,
        )
        close(
            report["volume_transfer_m3"][ri],
            dv,
            "normal volume change",
            atol=2 * (math.ulp(pl0.volume_m3) + math.ulp(pl.volume_m3)),
            rtol=0,
        )
        # As for components below, use the accurately reported increment
        # after binding it to the canonical stocks in ULPs. Dividing the
        # cancellation-prone stock difference by dt*mobility invents a rate
        # error when a normal substep is much smaller than the cell volume.
        dv = report["volume_transfer_m3"][ri]
        # At acoustic substeps, subtraction of two large inventories loses
        # digits of the small increment. Bind the reported increment to those
        # inventories in ULPs, then audit its constitutive rate independently.
        reported_dn = np.asarray(report["component_transfer_to_liquid_mol"][ri])
        increment_roundoff = np.array(
            [2 * (math.ulp(a) + math.ulp(b)) for a, b in zip(nl0, nl, strict=True)]
        )
        close(reported_dn, dn, "component transfer", atol=increment_roundoff, rtol=0)
        close(
            pl.pressure_pa - pg.pressure_pa - cap[i],
            dv / (dt * model.mobility),
            "normal kinetic law",
            atol=1e-6,
            rtol=1e-6,
        )
        ui = 0.5 * (pl0.momentum_kg_m_s / np.dot(nl0, mw) + pg0.momentum_kg_m_s / np.dot(ng0, mw))
        kappa = mw * ui
        close(
            pl.momentum_kg_m_s - pl0.momentum_kg_m_s,
            np.dot(kappa, dn),
            "liquid transfer impulse",
            atol=1e-27,
        )
        close(
            pg.momentum_kg_m_s - pg0.momentum_kg_m_s,
            -np.dot(kappa, dn),
            "gas transfer impulse",
            atol=1e-27,
        )
        eps = np.array(model.transfer_energy)
        ul = pl.momentum_kg_m_s / np.dot(nl, mw)
        ug = pg.momentum_kg_m_s / np.dot(ng, mw)
        affinity = (eps - ul * kappa + 0.5 * ul * ul * mw - mul) / pl.temperature_k - (
            eps - ug * kappa + 0.5 * ug * ug * mw - mug
        ) / pg.temperature_k
        mobility = np.array(model.component_mobility)
        active = np.flatnonzero(mobility > 0)
        close(
            reported_dn[active] / (dt * mobility[active]),
            affinity[active],
            "chemical affinity law",
            atol=1e-8,
            rtol=1e-8,
        )
        close(
            dn[mobility == 0],
            np.zeros(np.sum(mobility == 0)),
            "inactive component transfer",
            atol=1e-24,
        )
        heat = dv * dv / (dt * model.mobility)
        q = dt * model.conductance * (pg.temperature_k - pl.temperature_k)
        transfer = q + np.dot(eps, dn)
        for target, previous, expected in (
            (pl, pl0, -pl.pressure_pa * dv + model.share * heat + transfer),
            (pg, pg0, pg.pressure_pa * dv + (1 - model.share) * heat - transfer),
        ):
            allowance = (
                256 * (math.ulp(target.total_energy_j) + math.ulp(previous.total_energy_j)) + 1e-22
            )
            close(
                target.total_energy_j - previous.total_energy_j,
                expected,
                "local phasic energy/work",
                atol=allowance,
                rtol=1e-8,
            )
        close(report["heat_transfer_to_liquid_j"][ri], q, "interphase heat", atol=1e-22)
        close(report["normal_dissipation_j"][ri], heat, "normal heat", atol=1e-22, rtol=1e-7)
        dtotal.extend(
            [pl.total_energy_j - pl0.total_energy_j, pg.total_energy_j - pg0.total_energy_j]
        )
        lower.append(
            heat * (model.share / pl.temperature_k + (1 - model.share) / pg.temperature_k)
            + q * (1 / pl.temperature_k - 1 / pg.temperature_k)
            + np.sum(dn[active] ** 2 / (dt * mobility[active]))
        )
    defect = math.fsum(dtotal) + surface[1] - surface[0]
    allowance = (
        256
        * math.fsum(
            math.ulp(p.total_energy_j)
            for state in (before, after)
            for cell in state.cells
            for _, p in cell
        )
        + 1e-22
    )
    close(defect, 0.0, "bulk plus surface energy", atol=allowance, rtol=0)
    production = entropy[1] - entropy[0]
    if production < math.fsum(lower) - 1e-20 - 256 * max(
        math.ulp(entropy[0]), math.ulp(entropy[1])
    ):
        raise ValueError("Independent capillary source entropy inequality failed")
    return {
        "accepted": True,
        "checked_cells": count,
        "energy_residual_j": defect,
        "entropy_production_j_k": production,
        "qualification": "Independent scalar PR, chemical affinities and surface work; no axial transport qualification",
    }


def main():
    import argparse
    import json
    from pathlib import Path

    from run_pr_active_surface import fingerprints

    from surface.operation import _digest
    from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
    from surface.pr_capillary_pressure import PRCapillaryPressure

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("report", type=Path)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    report = json.loads(args.report.read_text())
    if (
        report["request"] != config
        or report["source_sha256"] != fingerprints()
        or report["format"] != "pr_capillary_pressure_run_v1"
    ):
        raise ValueError("Stale or modified capillary normal-source request")
    pipe = PRActiveMultifluidPipe.from_dict(config["pipe"])
    model = PRCapillaryPressure.from_dict(pipe, config["source"])
    if _digest(report["normalized_model"]) != _digest(
        {"pipe": pipe.to_dict(), "source": model.to_dict()}
    ):
        raise ValueError("Capillary normal-source normalized model differs")
    before = PRActiveMultifluidState.from_dict(report["initial"])
    after = PRActiveMultifluidState.from_dict(report["final"])
    if (
        report["initial"] != config["initial"]
        or len(report["result"]["steps"]) != 1
        or report["result"]["steps"][0]["dt_s"] != config["dt_s"]
    ):
        raise ValueError("Capillary source initial state or interval differs")
    print(json.dumps(audit(model, before, after, report["result"]["steps"][0])))


if __name__ == "__main__":
    main()
