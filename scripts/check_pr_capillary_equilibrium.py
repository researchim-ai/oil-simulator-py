"""Independent scalar PR audit of capillary UV phase-support transactions."""

import math

import numpy as np
from check_pr_capillary_pressure import close
from check_stratified_pr import Reference
from reference_wetting_potential import potential, pressure

from surface.operation import _digest
from surface.peng_robinson import R
from surface.pr_interphase_relaxation import PRRelaxationPhase


def geometry(source, cells):
    areas = np.array(source.pipe.lengths) * source.width
    h = (
        np.array(
            [dict(c)[source.liquid].volume_m3 if source.liquid in dict(c) else 0.0 for c in cells]
        )
        / areas
    )
    mu = np.zeros(len(h))
    energy = 0.0
    for i, j in source.geometry_edges:
        coefficient = (
            source.sigma * source.width / (0.5 * (source.pipe.lengths[i] + source.pipe.lengths[j]))
        )
        jump = h[j] - h[i]
        energy += 0.5 * coefficient * jump**2
        mu[i] -= coefficient * jump / areas[i]
        mu[j] += coefficient * jump / areas[j]
    for side, height in enumerate(source.boundary_heights):
        if height is not None:
            i = 0 if side == 0 else len(h) - 1
            coefficient = source.sigma * source.width / source.boundary_distances[side]
            energy += 0.5 * coefficient * (h[i] - height) ** 2
            mu[i] += coefficient * (h[i] - height) / areas[i]
    if source.wetting is not None:
        params = source.wetting.to_dict()
        energy += float(areas @ potential(params, source.sigma, h))
        mu += pressure(params, source.sigma, h, h)
    return energy, mu


def parcel(fluid, phase, root):
    n = np.array(phase.component_moles)
    amount = n.sum()
    if amount <= 0 or phase.volume_m3 <= 0:
        raise ValueError("Empty owned phase must be omitted")
    x = n / amount
    q = Reference(fluid, np.flatnonzero(n > 0)).state(
        phase.pressure_pa, phase.temperature_k, x, root
    )
    mass = sum(v * c.molar_mass_kg_mol for v, c in zip(n, fluid.components))
    energy = amount * q["u"] + 0.5 * phase.momentum_kg_m_s**2 / mass
    close(
        phase.volume_m3, amount * q["vm"], "equilibrium scalar EOS volume", atol=1e-25, rtol=1e-10
    )
    close(
        phase.total_energy_j,
        energy,
        "equilibrium scalar EOS energy",
        atol=512
        * (
            math.ulp(amount * q["u"])
            + math.ulp(0.5 * phase.momentum_kg_m_s**2 / mass)
            + math.ulp(energy)
        )
        + 1e-21,
        rtol=0.0,
    )
    ideal = np.array([c.ideal_cp_j_mol_k for c in fluid.components]) * (
        phase.temperature_k
        - fluid.tref
        - phase.temperature_k * math.log(phase.temperature_k / fluid.tref)
    )
    mu = np.zeros(len(n))
    active = np.flatnonzero(n > 0)
    mu[active] = ideal[active] + R * phase.temperature_k * (
        np.log(x[active] * phase.pressure_pa / fluid.pref) + q["phi"]
    )
    return {
        "n": n,
        "mass": mass,
        "energy": phase.total_energy_j,
        "momentum": phase.momentum_kg_m_s,
        "entropy": amount * q["s"],
        "mu": mu,
    }


def audit(model, before, after, report):
    s, p, f = model.source, model.pipe, model.fluid
    if (
        report["model_id"] != model.model_id
        or _digest(report["state"]) != _digest(after.to_dict())
        or before.model_id != p.model_id
        or after.model_id != p.model_id
        or before.time_s != after.time_s
    ):
        raise ValueError("Capillary equilibrium identity or clock differs")
    if "equilibrium_sweeps" in report:
        sweeps = report["equilibrium_sweeps"]
        if (
            not report["converged"]
            or not sweeps
            or len(sweeps) > report["maximum_sweeps"]
            or report["relative_tolerance"] != 2e-11
            or report["maximum_equilibrium_residual"] > 2e-11
        ):
            raise ValueError("Unconverged capillary equilibrium line")
        from surface.pr_active_multifluid_pipe import PRActiveMultifluidState

        current = before
        checks = []
        last_before = before
        for index, sweep in enumerate(sweeps):
            if index == 0:
                if (
                    sweep["heat_j"] != report["heat_j"]
                    or sweep["heat_temperature_k"] != report["heat_temperature_k"]
                ):
                    raise ValueError("Initial equilibrium heat differs")
            elif any(sweep["heat_j"]) or sweep["heat_temperature_k"] is not None:
                raise ValueError("Equilibrium iteration applied external heat twice")
            end = PRActiveMultifluidState.from_dict(sweep["state"])
            checks.append(audit(model, current, end, sweep))
            last_before = current
            current = end
        if _digest(current.to_dict()) != _digest(after.to_dict()):
            raise ValueError("Final capillary equilibrium sweep differs")
        change = max(
            abs(
                (dict(a)[s.liquid].volume_m3 if s.liquid in dict(a) else 0.0)
                - (dict(b)[s.liquid].volume_m3 if s.liquid in dict(b) else 0.0)
            )
            / v
            for a, b, v in zip(last_before.cells, after.cells, p.volumes)
        )
        if change > 2e-11 or any(
            tuple(dict(a)) != tuple(dict(b)) for a, b in zip(last_before.cells, after.cells)
        ):
            raise ValueError("Nonstationary final capillary phase support")
        _, cap = geometry(s, after.cells)
        final_error = 0.0
        for i, cell in enumerate(after.cells):
            row = dict(cell)
            if len(row) != 2:
                continue
            liq, gas = row[s.liquid], row[s.gas]
            a = parcel(f, liq, "liquid")
            b = parcel(f, gas, "vapor")
            mask = a["n"] + b["n"] > 0
            errors = [
                abs(liq.pressure_pa - gas.pressure_pa - cap[i])
                / max(1.0, abs(liq.pressure_pa), abs(gas.pressure_pa)),
                abs(liq.temperature_k - gas.temperature_k)
                / max(liq.temperature_k, gas.temperature_k),
                float(np.max(abs(a["mu"][mask] - b["mu"][mask]) / (R * liq.temperature_k))),
            ]
            final_error = max(final_error, *errors)
        if final_error > 2e-9:
            raise ValueError("Independent whole-line phase equilibrium failed")
        close(
            report["maximum_fraction_change"],
            change,
            "reported final volume convergence",
            atol=1e-18,
            rtol=1e-8,
        )
        f0, _ = geometry(s, before.cells)
        f1, _ = geometry(s, after.cells)
        close(
            report["surface_energy_before_j"],
            f0,
            "initial equilibrium surface energy",
            atol=1e-20,
            rtol=1e-9,
        )
        close(
            report["surface_energy_after_j"],
            f1,
            "final equilibrium surface energy",
            atol=1e-20,
            rtol=1e-9,
        )
        nc = len(f.names)

        def totals(state):
            return [
                math.fsum(ph.component_moles[j] for c in state.cells for _, ph in c)
                for j in range(nc)
            ] + [
                math.fsum(ph.total_energy_j for c in state.cells for _, ph in c),
                math.fsum(ph.momentum_kg_m_s for c in state.cells for _, ph in c),
            ]

        a, b = totals(before), totals(after)
        defect = [y - x for x, y in zip(a, b)]
        defect[-2] = math.fsum((b[-2], -a[-2], f1, -f0, -math.fsum(report["heat_j"])))
        bounds = [
            1024 * len(sweeps) * len(after.cells) * (math.ulp(x) + math.ulp(y)) + 1e-23
            for x, y in zip(a, b)
        ]
        phases = [ph for state in (before, after) for cell in state.cells for _, ph in cell]
        bounds[-1] += 1024 * len(sweeps) * math.fsum(math.ulp(ph.momentum_kg_m_s) for ph in phases)
        bounds[-2] += 1024 * len(sweeps) * math.fsum(math.ulp(ph.total_energy_j) for ph in phases)
        bounds[-2] += (
            1024
            * len(sweeps)
            * math.fsum(math.ulp(v) for v in (f0, f1, math.fsum(report["heat_j"])))
            + 1e-18
        )
        close(defect, np.zeros(nc + 2), "whole-line N/M/(E+F)", atol=np.array(bounds), rtol=0.0)
        close(
            report["conservation_residuals"],
            defect,
            "reported whole-line conservation",
            atol=np.array(bounds),
            rtol=0.0,
        )
        entropy = math.fsum(c["entropy_production_j_k"] for c in checks)
        close(
            report["entropy_production_j_k"],
            entropy,
            "whole-line equilibrium entropy",
            atol=1e-15,
            rtol=1e-9,
        )
        return {
            "accepted": True,
            "checked_cells": len(after.cells),
            "checked_sweeps": len(sweeps),
            "maximum_equilibrium_residual": final_error,
            "maximum_local_augmented_energy_defect_j": max(
                c["maximum_local_augmented_energy_defect_j"] for c in checks
            ),
            "entropy_production_j_k": math.fsum(c["entropy_production_j_k"] for c in checks),
            "phase_counts": [len(c) for c in after.cells],
            "qualification": "Independent scalar PR audit of every UV transaction, one-time heat pulse and converged nonlocal capillary pressure/chemical/thermal equilibrium",
        }
    if len(report["cells"]) != len(before.cells) or len(report["heat_j"]) != len(before.cells):
        raise ValueError("Missing capillary equilibrium cell transactions")
    cells = list(before.cells)
    entropy = 0.0
    defects = []
    for i, record in enumerate(report["cells"]):
        if record["cell"] != i:
            raise ValueError("Capillary equilibrium transaction ordering differs")
        old = dict(cells[i])
        updated = {
            name: PRRelaxationPhase.from_dict(ph) for name, ph in record["phase_stocks"].items()
        }
        if not updated or set(updated) - {s.liquid, s.gas}:
            raise ValueError("Unknown equilibrium phase registry")
        oldq = {
            name: parcel(f, ph, "liquid" if name == s.liquid else "vapor")
            for name, ph in old.items()
        }
        newq = {
            name: parcel(f, ph, "liquid" if name == s.liquid else "vapor")
            for name, ph in updated.items()
        }
        n0 = sum(q["n"] for q in oldq.values())
        n1 = sum(q["n"] for q in newq.values())
        close(
            n1, n0, "local equilibrium components", atol=256 * np.spacing(abs(n0)) + 1e-24, rtol=0.0
        )
        close(
            sum(ph.volume_m3 for ph in updated.values()),
            p.volumes[i],
            "local equilibrium volume",
            atol=256 * math.ulp(p.volumes[i]),
            rtol=0.0,
        )
        close(
            sum(q["momentum"] for q in newq.values()),
            sum(q["momentum"] for q in oldq.values()),
            "local equilibrium momentum",
            atol=1e-22,
            rtol=1e-12,
        )
        surf0, _ = geometry(s, cells)
        cells[i] = tuple(updated.items())
        surf1, cap = geometry(s, cells)
        heat = report["heat_j"][i]
        terms = [
            *[q["energy"] for q in newq.values()],
            *[-q["energy"] for q in oldq.values()],
            surf1,
            -surf0,
            -heat,
        ]
        defect = math.fsum(terms)
        close(
            defect,
            0.0,
            "local equilibrium bulk plus surface energy",
            atol=1024 * sum(math.ulp(v) for v in terms) + 1e-18,
            rtol=0.0,
        )
        close(
            record["surface_energy_change_j"],
            surf1 - surf0,
            "reported equilibrium surface work",
            atol=1e-20,
            rtol=1e-8,
        )
        if len(updated) == 2:
            liquid, gas = updated[s.liquid], updated[s.gas]
            close(
                liquid.temperature_k,
                gas.temperature_k,
                "interphase thermal equilibrium",
                atol=1e-9,
                rtol=0.0,
            )
            close(
                (newq[s.liquid]["mu"] - newq[s.gas]["mu"])[n0 > 0] / (R * liquid.temperature_k),
                np.zeros(np.count_nonzero(n0 > 0)),
                "interphase chemical equilibrium",
                atol=2e-9,
                rtol=0.0,
            )
            close(
                liquid.pressure_pa - gas.pressure_pa,
                cap[i],
                "capillary mechanical equilibrium",
                atol=1e-5 + 1e-9 * max(liquid.pressure_pa, gas.pressure_pa),
                rtol=1e-8,
            )
            velocities = [q["momentum"] / q["mass"] for q in newq.values()]
            close(velocities[0], velocities[1], "equilibrium common velocity", atol=1e-12)
        bath = report["heat_temperature_k"]
        production = (
            sum(q["entropy"] for q in newq.values())
            - sum(q["entropy"] for q in oldq.values())
            - (heat / bath if bath is not None else 0.0)
        )
        tolerance = (
            512 * sum(math.ulp(q["entropy"]) for q in (*oldq.values(), *newq.values())) + 1e-18
        )
        if production < -tolerance:
            raise ValueError("Independent capillary equilibrium entropy decreased")
        close(
            record["entropy_production_j_k"],
            production,
            "reported equilibrium entropy",
            atol=tolerance,
            rtol=1e-10,
        )
        entropy += production
        defects.append(abs(defect))
    if _digest(
        [
            {name: ph.to_dict() if hasattr(ph, "to_dict") else vars(ph) for name, ph in c}
            for c in cells
        ]
    ) != _digest(after.to_dict()["cells"]):
        raise ValueError("Final equilibrium phase stocks differ from transactions")
    return {
        "accepted": True,
        "checked_cells": len(cells),
        "maximum_local_augmented_energy_defect_j": max(defects, default=0.0),
        "entropy_production_j_k": entropy,
        "phase_counts": [len(c) for c in after.cells],
        "qualification": "Independent scalar PR, chemical/thermal/mechanical equilibrium, per-coordinate N/V/M/(E+F) and Clausius audit; no global-minimum or kinetic calibration claim",
    }


def main():
    import argparse
    import json
    from pathlib import Path

    from run_pr_active_surface import fingerprints

    from surface.pr_active_multifluid_pipe import PRActiveMultifluidState
    from surface.pr_capillary_equilibrium import PRCapillaryEquilibrium

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    report = json.loads(args.report.read_text())
    if (
        report["format"] != "pr_capillary_equilibrium_run_v2"
        or report["request"] != config
        or report["source_sha256"] != fingerprints()
        or len(report["records"]) != len(config["heat_pulses"])
    ):
        raise ValueError("Stale or modified capillary equilibrium experiment")
    model = PRCapillaryEquilibrium.from_dict(config["model"])
    if _digest(model.to_dict()) != _digest(report["normalized_model"]):
        raise ValueError("Normalized capillary equilibrium model differs")
    old = PRActiveMultifluidState.from_dict(config["initial"])
    checks = []
    for pulse, record in zip(config["heat_pulses"], report["records"]):
        result = record["result"]
        if (
            not result.get("converged")
            or not result.get("equilibrium_sweeps")
            or result.get("maximum_sweeps") != 12
            or result["heat_j"] != pulse["heat_j"]
            or result["heat_temperature_k"] != pulse["heat_temperature_k"]
        ):
            raise ValueError("Capillary heat pulse differs from request")
        end = PRActiveMultifluidState.from_dict(result["state"])
        checks.append(audit(model, old, end, result))
        old = end
    if _digest(old.to_dict()) != _digest(report["final"]):
        raise ValueError("Capillary equilibrium final state differs")
    print(json.dumps({"accepted": True, "pulses": checks}))


if __name__ == "__main__":
    main()
