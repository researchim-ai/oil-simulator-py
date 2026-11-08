"""Independent NumPy face integration for finite supported PR junctions."""

import numpy as np
from reference_pr_multifluid_pipe import packets, path


def reconstruct(model, state, fluid):
    if model.get("hydrostatic_reference") is not None:
        raw = {k: v for k, v in model.items() if k != "hydrostatic_reference"}
        raw["model"] = "pr_multifluid_network_v2"
        result = reconstruct(raw, state, fluid)
        baseline = reconstruct(raw, hydrostatic_state(model, fluid), fluid)
        for key in ("phase_rate", "alpha_rate", "force", "wall_force_n"):
            result[key] = result[key] - baseline[key]
        result["faces"] = [
            {k: v - ref[k] for k, v in face.items()}
            for face, ref in zip(result["faces"], baseline["faces"])
        ]
        return result
    data = packets(model, state, fluid)
    lookup = {n: i for i, n in enumerate(model["cell_names"])}
    supported = set(model["supported_nodes"])
    nph, nc = len(data[0]["alpha"]), len(fluid.names)
    incoming = np.zeros((len(data), nph * (nc + 2) + nph - 1))
    force = np.zeros(3)
    faces = []
    elevations = model.get("elevations_m")
    gravity = model.get("gravity_m_s2", 0.0)
    volumes = np.array(model["volumes_m3"])
    for face in model["faces"]:
        a, b = lookup[face["source"]], lookup[face["target"]]
        left, right = data[a], data[b]
        minus, plus, flux = path(left, right, model["wave_speed_bound_m_s"])
        area = face["area_m2"]
        if elevations is not None:
            mw = np.array([c.molar_mass_kg_mol for c in fluid.components])
            dl = (
                0.0
                if face["source"] in supported
                else 0.5 * volumes[a] / area * face["direction"][2]
            )
            dr = (
                0.0
                if face["target"] in supported
                else 0.5 * volumes[b] / area * face["direction"][2]
            )
            rl = left["conserved"][:, :nc] @ mw
            rr = right["conserved"][:, :nc] @ mw
            pleft = left["pressure"] - gravity * rl * dl / left["alpha"]
            pright = right["pressure"] + gravity * rr * dr / right["alpha"]
            if (
                min(pleft.min(), pright.min()) < fluid.min_pressure_pa
                or max(pleft.max(), pright.max()) > fluid.max_pressure_pa
            ):
                raise ValueError("Hydrostatic port reconstruction outside PR domain")
            pressure = 0.5 * gravity * (rr * dr - rl * dl)
            correction = np.zeros_like(left["conserved"])
            correction[:, -2] = pressure
            correction[:, -1] = pressure * 0.5 * (left["velocity"] + right["velocity"])
            minus[: nph * (nc + 2)] += correction.ravel()
            plus[: nph * (nc + 2)] -= correction.ravel()
            flux["flux"] = flux["flux"] + correction
        faces.append(flux)
        incoming[a] += area * minus
        incoming[b] += area * plus
        support = 0.0
        for name, fluctuation in [(face["source"], minus), (face["target"], plus)]:
            if name in supported:
                support += fluctuation[: nph * (nc + 2)].reshape(nph, nc + 2)[:, -2].sum()
        wall_pressure = -(right["flux"][:, -2].sum() - left["flux"][:, -2].sum())
        force += area * (support + wall_pressure) * np.array(face["direction"])
        if elevations is not None:
            za = (
                elevations[a]
                if face["source"] in supported
                else elevations[a] + 0.5 * volumes[a] / area * face["direction"][2]
            )
            zb = (
                elevations[b]
                if face["target"] in supported
                else elevations[b] - 0.5 * volumes[b] / area * face["direction"][2]
            )
            zface = (za + zb) / 2
            mw = np.array([c.molar_mass_kg_mol for c in fluid.components])
            mass_flux = flux["flux"][:, :nc] @ mw
            for i, sign in [(a, 1.0), (b, -1.0)]:
                phase = incoming[i, : nph * (nc + 2)].reshape(nph, nc + 2)
                phase[:, -1] += gravity * area * sign * (zface - elevations[i]) * mass_flux
    if elevations is not None:
        for i, name in enumerate(model["cell_names"]):
            if name in supported:
                continue
            # Decode mass independently from scalar EOS stocks; directions are
            # recovered from topology rather than the runtime constant tensors.
            direction = np.array(
                next(f["direction"] for f in model["faces"] if f["source"] == name)
            )
            mw = np.array([c.molar_mass_kg_mol for c in fluid.components])
            mass = data[i]["conserved"][:, :nc] @ mw * volumes[i]
            load = gravity * mass * direction[2]
            incoming[i, : nph * (nc + 2)].reshape(nph, nc + 2)[:, -2] += load
            force -= load.sum() * direction
    power = 0.0
    for pump in model.get("pumps", []):
        i = lookup[pump["cell"]]
        k = pump["phase"]
        mapping = pump["map"]
        speed = pump["speed_rad_s"]
        area = next(f["area_m2"] for f in model["faces"] if f["source"] == pump["cell"])
        mw = np.array([c.molar_mass_kg_mol for c in fluid.components])
        rho = float(data[i]["conserved"][k, :nc] @ mw) / data[i]["alpha"][k]
        if rho < pump["minimum_density_kg_m3"]:
            raise ValueError("Liquid pump below declared density envelope")
        flow = data[i]["alpha"][k] * area * data[i]["velocity"][k]
        if abs(flow) > mapping["maximum_flow_m3_s"] or abs(speed) > mapping["maximum_speed_rad_s"]:
            raise ValueError("Pump operating point outside declared map")
        coupling = (
            mapping["euler_head_coefficient_m_s2"] * abs(speed)
            + mapping["linear_head_coefficient_m_s"]
        )
        head = coupling * speed - mapping["flow_loss_coefficient_s2_m5"] * flow * abs(flow)
        shaft = (
            rho
            * 9.80665
            * (coupling * flow + mapping["viscous_torque_coefficient_m4_s"] * speed)
            * speed
        )
        load = rho * 9.80665 * head * data[i]["alpha"][k] * area
        incoming[i, : nph * (nc + 2)].reshape(nph, nc + 2)[k, -2] -= load
        incoming[i, : nph * (nc + 2)].reshape(nph, nc + 2)[k, -1] -= shaft
        direction = np.array(
            next(f["direction"] for f in model["faces"] if f["source"] == pump["cell"])
        )
        force += load * direction
        power += shaft
    return {
        "external_power_w": power,
        "packets": data,
        "phase_rate": incoming[:, : nph * (nc + 2)].reshape(-1, nph, nc + 2),
        "alpha_rate": incoming[:, nph * (nc + 2) :] / np.array(model["volumes_m3"])[:, None],
        "faces": faces,
        "wall_fluxes": [],
        "wall_force_n": force,
        "force": force,
    }


def hydrostatic_state(model, fluid):
    """Independent primitive-to-stock conversion and Gibbs-potential audit."""
    import math

    from check_stratified_pr import Reference

    roots = model["sources"][0]["phase_roots"]
    specs = model["hydrostatic_reference"]
    cells = []
    invariants = []
    for i, phases in enumerate(specs):
        if len(phases) != len(roots):
            raise ValueError("Reference phase registry differs")
        output = []
        cell_invariants = []
        for phase, tag in zip(phases, roots):
            if phase["velocity_m_s"] != 0.0:
                raise ValueError("Moving hydrostatic reference")
            x = np.array(phase["mole_fractions"])
            q = Reference(fluid, np.flatnonzero(x > 0)).state(
                phase["pressure_pa"], phase["temperature_k"], x, tag
            )
            amount = phase["volume_m3"] / q["vm"]
            mw = sum(x[j] * c.molar_mass_kg_mol for j, c in enumerate(fluid.components))
            cell_invariants.append(
                (
                    phase["volume_m3"] / model["volumes_m3"][i],
                    phase["temperature_k"],
                    x,
                    (q["h"] - phase["temperature_k"] * q["s"]) / mw
                    + model["gravity_m_s2"] * (model["elevations_m"][i] - model["elevations_m"][0]),
                )
            )
            output.append(
                {
                    "component_moles": (amount * x).tolist(),
                    "volume_m3": phase["volume_m3"],
                    "pressure_pa": phase["pressure_pa"],
                    "temperature_k": phase["temperature_k"],
                    "momentum_kg_m_s": 0.0,
                    "total_energy_j": amount * q["u"],
                }
            )
        if abs(math.fsum(p["volume_m3"] for p in output) - model["volumes_m3"][i]) > 1e-12:
            raise ValueError("Hydrostatic reference does not fill cell")
        for edge, coordinate, tolerance in [
            ("pressure_conductance_m3_pa_s", "pressure_pa", 1e-6),
            ("heat_conductance_w_k", "temperature_k", 1e-8),
        ]:
            for a in range(len(roots)):
                for b in range(a):
                    if (
                        model["sources"][i][edge][a][b] > 0
                        and abs(phases[a][coordinate] - phases[b][coordinate]) > tolerance
                    ):
                        raise ValueError("Hydrostatic reference has a nonstationary local source")
        invariants.append(cell_invariants)
        cells.append({"phases": output})
    for row in invariants:
        for a, b in zip(row, invariants[0]):
            if (
                abs(a[0] - b[0]) > 1e-10
                or abs(a[1] - b[1]) > 1e-8
                or np.max(abs(a[2] - b[2])) > 1e-10
                or abs(a[3] - b[3]) > 1e-5
            ):
                raise ValueError("Reference fails independent PR hydrostatic audit")
    return {"cells": cells}
