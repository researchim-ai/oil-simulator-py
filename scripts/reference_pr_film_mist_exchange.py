"""Independent scalar mass/energy/COM solution for a local morphology source."""

import math

import numpy as np
from reference_pr_active_multifluid import packets
from reference_wetting_potential import potential


def audit(pipe, before, after, report, law, entrainment_m3_s, deposition_m3_s, dt_s):
    if (
        report["model"] != law
        or report["entrainment_m3_s"] != list(entrainment_m3_s)
        or report["deposition_m3_s"] != list(deposition_m3_s)
        or report["dt_s"] != dt_s
        or report["state"] != after
    ):
        raise ValueError("Morphology source command/report differs")
    if law["film_surface_model"] not in ("planar_complete_wetting", "planar_long_wave_wetting"):
        raise ValueError("Unsupported film surface-energy model")
    a, b = packets(pipe, before), packets(pipe, after)
    nc = len(pipe.fluid.names)
    mw = np.array([c.molar_mass_kg_mol for c in pipe.fluid.components])
    film, mist, gas = (law[k] for k in ("film_slot", "mist_slot", "gas_slot"))
    coeff = 6 * law["surface_tension_n_m"] / law["droplet_diameter_m"]

    def close(x, y, label, atol=1e-12, rtol=1e-10):
        x, y = np.asarray(x), np.asarray(y)
        if (
            x.shape != y.shape
            or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))
            or np.any(np.abs(x - y) > atol + rtol * np.maximum(np.abs(x), np.abs(y)))
        ):
            raise ValueError("Morphology audit differs: " + label)

    diagnostics = {
        k: []
        for k in (
            "surface_energy_before_j",
            "surface_energy_after_j",
            "surface_creation_j",
            "deposition_heat_j",
            "kinetic_energy_removed_j",
            "irreversible_heat_j",
            "gas_to_mist_impulse_kg_m_s",
            "droplet_number",
        )
    }

    def row(p):
        return np.array(
            [*p["component_moles"], p["volume_m3"], p["momentum_kg_m_s"], p["total_energy_j"]]
        )

    def wall_energy(index, volume):
        material = law.get("wetting")
        if material is None:
            return 0.0
        area = pipe.lengths[index] * material["film_width_m"]
        if volume <= 0 and material["equilibrium_contact_angle_deg"] > 0:
            raise ValueError("Singular wetting potential on absent film")
        return float(area * potential(material, law["surface_tension_n_m"], volume / area))

    for i, (ca, cb) in enumerate(zip(before["cells"], after["cells"], strict=True)):
        for name in pipe.names:
            if name not in (film, mist, gas) and ca.get(name) != cb.get(name):
                raise ValueError("Uninvolved phase changed during morphology exchange")
        stocks = {n: row(ca[n]) if n in ca else np.zeros(nc + 3) for n in (film, mist, gas)}
        v0 = stocks[mist][nc]
        wall0 = wall_energy(i, stocks[film][nc])
        ve, vd = dt_s * entrainment_m3_s[i], dt_s * deposition_m3_s[i]
        if vd > stocks[mist][nc]:
            raise ValueError("Deposition without donor inventory")
        if vd:
            parcel = stocks[mist] * (vd / stocks[mist][nc])
            stocks[mist] = stocks[mist] * (1 - vd / stocks[mist][nc])
            stocks[film] += parcel
        wall_d = wall_energy(i, stocks[film][nc])
        deposition_heat = coeff * vd - (wall_d - wall0)
        stocks[film][-1] += deposition_heat
        if ve > stocks[film][nc]:
            raise ValueError("Entrainment without donor inventory")
        if ve:
            parcel = stocks[film] * (ve / stocks[film][nc])
            stocks[film] = stocks[film] * (1 - ve / stocks[film][nc])
            stocks[mist] += parcel
        wall_e = wall_energy(i, stocks[film][nc])
        cost = coeff * ve + wall_e - wall_d
        loss = max(cost, 0.0) / law["atomization_efficiency"]
        extra = loss - cost
        impulse = 0.0
        if loss:
            mg, md = stocks[gas][:nc] @ mw, stocks[mist][:nc] @ mw
            if mg <= 0 or md <= 0:
                raise ValueError("Missing kinetic reservoir")
            pg, pd = stocks[gas][-2], stocks[mist][-2]
            reduced = 1 / (1 / mg + 1 / md)
            relative = pg / mg - pd / md
            square = relative**2 - 2 * loss / reduced
            if square < 0:
                raise ValueError("Underfunded surface formation")
            # Solve for the two final momenta from conserved COM momentum
            # and the independently prescribed relative kinetic energy.
            new_relative = math.copysign(math.sqrt(square), relative)
            barycentric = (pg + pd) / (mg + md)
            ug = barycentric + md / (mg + md) * new_relative
            ud = barycentric - mg / (mg + md) * new_relative
            new_pg, new_pd = mg * ug, md * ud
            stocks[gas][-2], stocks[mist][-2] = new_pg, new_pd
            stocks[gas][-1] += (new_pg - pg) * (new_pg + pg) / (2 * mg)
            stocks[mist][-1] += (new_pd - pd) * (new_pd + pd) / (2 * md)
            impulse = pg - new_pg
        stocks[gas][-1] += (1 - law["heat_to_mist_fraction"]) * extra
        stocks[mist][-1] += law["heat_to_mist_fraction"] * extra
        for n, expected in stocks.items():
            if expected[nc] == 0:
                if n in cb:
                    raise ValueError("Ghost morphology phase")
                continue
            if n not in cb:
                raise ValueError("Missing actual morphology phase")
            actual = row(cb[n])
            close(actual[:nc], expected[:nc], n + " components", atol=1e-18)
            close(actual[nc], expected[nc], n + " volume", atol=1e-24)
            close(actual[-2], expected[-2], n + " momentum", atol=1e-15)
            close(
                actual[-1],
                expected[-1],
                n + " energy",
                atol=64 * math.ulp(float(expected[-1])) + 1e-18,
                rtol=0.0,
            )
        values = (
            coeff * v0 + wall0,
            coeff * stocks[mist][nc] + wall_e,
            cost,
            deposition_heat,
            loss,
            extra,
            impulse,
            stocks[mist][nc] / (math.pi / 6 * law["droplet_diameter_m"] ** 3),
        )
        for k, v in zip(diagnostics, values, strict=True):
            diagnostics[k].append(v)
    for k, expected in diagnostics.items():
        close(report[k], expected, k, atol=1e-12 if "impulse" not in k else 1e-15)
    # No physical clock is advanced by a local split source.
    bulk_energy = lambda s: math.fsum(p["total_energy_j"] for c in s["cells"] for p in c.values())
    actual_defect = (
        bulk_energy(after)
        - bulk_energy(before)
        + sum(diagnostics["surface_energy_after_j"])
        - sum(diagnostics["surface_energy_before_j"])
    )
    roundoff = (
        64
        * math.fsum(
            math.ulp(p["total_energy_j"])
            for s in (before, after)
            for c in s["cells"]
            for p in c.values()
        )
        + 1e-18
    )
    close(actual_defect, 0.0, "joint energy", atol=roundoff, rtol=0.0)
    close(
        report["joint_energy_residual_j"],
        actual_defect,
        "reported joint energy",
        atol=roundoff,
        rtol=0.0,
    )
    close(after["time_s"], before["time_s"], "source clock", atol=0, rtol=0)
    entropy = b["entropy"] - a["entropy"]
    if entropy < -1e-9:
        raise ValueError("Negative morphology entropy production")
    close(report["entropy_production_j_k"], entropy, "entropy production", atol=1e-9)
    return {
        "accepted": True,
        "checked_cells": len(before["cells"]),
        "qualification": "Prescribed rates, fixed drop diameter and explicitly declared planar wall energy; not a regime/onset model",
    }
