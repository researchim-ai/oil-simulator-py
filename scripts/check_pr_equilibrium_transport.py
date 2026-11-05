"""Independent NumPy PR/HLLC/phase-drift/finite-aperture N/M/E audit.

Recomputes thermodynamic identities on serialized phase branches. No runtime
flow, flash, decode or inspect is invoked; global phase stability is not proved.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from check_pr_momentum_vessels import port_flux
from check_stratified_pr import Reference
from check_stratified_pr_flow import close
from check_stratified_pr_network import reference_darcy

from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork


def decode(state, node, model, fluid):
    n = np.asarray(state["component_moles"])
    amounts = np.asarray(state["phase_component_moles"])
    active = np.flatnonzero(n > 0)
    if amounts.shape != (len(state["phase_roots"]), len(n)) or np.any(n < 0) or np.any(amounts < 0):
        raise ValueError("Phase dimensions or signs differ")
    close(amounts.sum(0), n, atol=1e-9, rtol=2e-8)
    pn = amounts.sum(1)
    if np.any(pn <= 0):
        raise ValueError("Empty stored phase")
    x = amounts / pn[:, None]
    p = state["pressure_pa"]
    t = state["temperature_k"]
    ref = Reference(fluid, active)
    if (
        not fluid.min_pressure_pa <= p <= fluid.max_pressure_pa
        or not fluid.min_temperature_k <= t <= fluid.max_temperature_k
    ):
        raise ValueError("Primitive outside domain")
    q = [ref.state(p, t, xx, tag) for xx, tag in zip(x, state["phase_roots"])]
    vm = np.array([v["vm"] for v in q])
    rho = np.array([v["rho"] for v in q])
    h = np.array([v["h"] for v in q])
    uu = np.array([v["u"] for v in q])
    ss = np.array([v["s"] for v in q])
    mu = np.array([np.log(xx[active]) + v["phi"] for xx, v in zip(x, q)])
    if len(mu) > 1:
        close(mu[:-1], np.tile(mu[-1], (len(mu) - 1, 1)), atol=2e-7, rtol=0.0)
    close(pn @ vm, node["volume_m3"], atol=1e-12, rtol=2e-8)
    mw = np.array([c.molar_mass_kg_mol for c in fluid.components])
    phase_mass = amounts @ mw
    mass = n @ mw
    u = state["momentum_kg_m_s"] / mass
    if abs(u) > model["maximum_velocity_m_s"]:
        raise ValueError("Bulk velocity outside domain")
    drift = np.zeros(len(pn))
    if node["kind"] == "pipe_cell" and (
        node.get("distribution_strength", 0.0) or node.get("buoyancy_relaxation_s", 0.0)
    ):
        mean = phase_mass.sum() / (pn @ vm)
        scale = (
            node.get("distribution_strength", 0.0) * u
            + node.get("buoyancy_relaxation_s", 0.0)
            * model["gravity_m_s2"]
            * node["elevation_change_m"]
            / node["length_m"]
        )
        drift = scale * (mean / rho - 1.0)
    velocities = u + drift
    closure = node.get("drift_flux")
    if closure is not None and len(pn) > 1:
        alpha = pn * vm / (pn @ vm)
        gas = int(np.argmin(rho))
        ag = alpha[gas]
        rg = rho[gas]
        liquid = np.arange(len(pn)) != gas
        rl = np.dot(alpha[liquid], rho[liquid]) / alpha[liquid].sum()
        c0 = closure["distribution_parameter"]
        vd = closure["drift_velocity_m_s"]
        denominator = rl + ag * c0 * (rg - rl)
        if (
            denominator <= 0
            or not closure["minimum_gas_volume_fraction"]
            <= ag
            <= closure["maximum_gas_volume_fraction"]
            or rg / rl > closure["maximum_gas_liquid_density_ratio"]
        ):
            raise ValueError("Drift-flux regime envelope rejected")
        j = ((alpha @ rho) * u - ag * (rg - rl) * vd) / denominator
        vg = c0 * j + vd
        vl = ((alpha @ rho) * u - ag * rg * vg) / (alpha[liquid].sum() * rl)
        velocities = np.full(len(pn), vl)
        velocities[gas] = vg
        drift = velocities - u
    if np.max(np.abs(velocities)) > model["maximum_velocity_m_s"]:
        raise ValueError("Phase velocity outside domain")
    close(phase_mass @ drift, 0.0, atol=1e-10, rtol=0.0)
    kinetic = 0.5 * phase_mass @ velocities**2
    potential = (
        mass * model["gravity_m_s2"] * (node["elevation_m"] - model["reference_elevation_m"])
    )
    internal = pn @ uu
    entropy = pn @ ss
    stress = phase_mass @ drift**2 / node["volume_m3"]
    close(internal + kinetic + potential, state["total_energy_j"], atol=1e-6, rtol=2e-8)
    if node["kind"] == "vessel":
        close(state["momentum_kg_m_s"], 0.0, atol=1e-12, rtol=0.0)
    if node.get("wall") is not None:
        w = node["wall"]
        tw = state["wall_temperature_k"]
        if not w["min_temperature_k"] <= tw <= w["max_temperature_k"]:
            raise ValueError("Wall outside domain")
    elif state["wall_temperature_k"] is not None:
        raise ValueError("Unexpected wall state")
    return {
        "n": n,
        "pn": pn,
        "x": x,
        "p": p,
        "t": t,
        "u": u,
        "vm": vm,
        "rho_phase": rho,
        "h": h,
        "s_phase": ss,
        "rho": mass / node["volume_m3"],
        "mass": mass,
        "kinetic": kinetic,
        "potential": potential,
        "internal": internal,
        "entropy": entropy,
        "alpha": pn * vm / node["volume_m3"],
        "drift": drift,
        "velocities": velocities,
        "stress": stress,
    }


def liquid_level(volume, geometry):
    total = math.pi * geometry["diameter_m"] ** 2 / 4 * geometry["length_m"]
    if geometry["orientation"] == "vertical":
        return volume / total * geometry["length_m"]
    radius = geometry["diameter_m"] / 2

    def filled(height):
        theta = math.acos(np.clip((radius - height) / radius, -1, 1))
        return geometry["length_m"] * radius**2 * (theta - math.sin(2 * theta) / 2)

    a, b = 0.0, 2 * radius
    for _ in range(70):
        mid = (a + b) / 2
        if filled(mid) < volume:
            a = mid
        else:
            b = mid
    return (a + b) / 2


def disk_area(h, d):
    r = d / 2
    h = max(0.0, min(d, h))
    theta = math.acos(np.clip((r - h) / r, -1, 1))
    return r * r * (theta - math.sin(2 * theta) / 2)


def endpoint(node, data, link, side, model):
    weights = data["alpha"]
    height = link.get(side + "_height_m")
    if height is not None:
        geometry = node["geometry"]
        order = np.argsort(-data["rho_phase"], kind="stable")
        volumes = data["pn"] * data["vm"]
        volumes = volumes / volumes.sum() * node["volume_m3"]
        tops = np.cumsum(volumes[order])
        levels = np.array([liquid_level(v, geometry) for v in tops])
        bottom = np.r_[0.0, levels[:-1]]
        weights = np.zeros(len(volumes))
        diameter = link["diameter_m"]
        base = height - diameter / 2
        for k, i in enumerate(order):
            weights[i] = (
                disk_area(levels[k] - base, diameter) - disk_area(bottom[k] - base, diameter)
            ) / (math.pi * diameter**2 / 4)
    normal = 1.0 if link[side + "_port"] == "right" else -1.0
    projection = normal if side == "source" else -normal
    v = data["u"] * projection if node["kind"] == "pipe_cell" else 0.0
    w = data["drift"] * projection if node["kind"] == "pipe_cell" else np.zeros(len(weights))
    concentration = (weights[:, None] * data["x"] / data["vm"][:, None]).sum(0)
    rho = weights @ data["rho_phase"]
    phase_mass_mol = data["rho_phase"] * data["vm"]
    internal = np.sum(weights * (data["h"] - data["p"] * data["vm"]) / data["vm"])
    entropy = np.sum(weights * data["s_phase"] / data["vm"])
    z = node["elevation_m"] + (
        node["elevation_change_m"] * normal / 2 if node["kind"] == "pipe_cell" else 0.0
    )
    phi = model["gravity_m_s2"] * (z - model["reference_elevation_m"])
    stress = np.sum(weights * data["rho_phase"] * w * w)
    energy = internal + 0.5 * rho * v * v + rho * phi + 0.5 * stress
    uc = np.r_[concentration, rho * v, energy]
    ff = np.r_[concentration * v, rho * v * v + data["p"], (energy + data["p"]) * v]
    velocity = v + w
    concentration_phase = weights[:, None] * data["x"] / data["vm"][:, None]
    actual = np.column_stack(
        (
            concentration_phase * velocity[:, None],
            weights * data["rho_phase"] * velocity**2,
            weights
            / data["vm"]
            * (data["h"] + phase_mass_mol * (0.5 * velocity**2 + phi))
            * velocity,
        )
    )
    bulk = np.column_stack(
        (
            concentration_phase * v,
            weights * data["rho_phase"] * v * v,
            weights / data["vm"] * (data["h"] + phase_mass_mol * (0.5 * (v * v + w * w) + phi)) * v,
        )
    )
    extra = {
        "velocity": velocity,
        "u": v,
        "actual": actual,
        "bulk": bulk,
        "actual_entropy": weights / data["vm"] * data["s_phase"] * velocity,
        "bulk_entropy": weights / data["vm"] * data["s_phase"] * v,
    }
    return (uc, ff, data["p"], v, rho, entropy), weights, normal, extra


def audit(config, report):
    if config != report["config"]:
        raise ValueError("Transport request differs")
    parsed = PREquilibriumTransportNetwork.from_dict(config["model"])
    model = json.loads(json.dumps(parsed.to_dict()))
    fluid = parsed.fluid
    if model != report["model"]:
        raise ValueError("Normalized transport model differs")
    nodes = {n["name"]: n for n in model["nodes"]}
    names = tuple(nodes)
    nc = len(fluid.names)
    old = report["initial"]
    old_start = old
    checked = 0
    maximum = np.zeros(nc + 2)
    all_heat = 0.0
    olddata = {n: decode(old["nodes"][n], nodes[n], model, fluid) for n in names}
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        end = old["time_s"] + command["duration_s"]
        opens = {c["name"]: True for c in model["links"]}
        opens.update(command.get("openings", {}))
        if opens != interval["openings"] or not interval["steps"]:
            raise ValueError("Command or accepted interval differs")
        expected_loss = {
            name: command.get("loss_coefficients", {}).get(
                name, node.get("local_loss_coefficient", 0.0)
            )
            for name, node in nodes.items()
            if node["kind"] == "pipe_cell"
        }
        if expected_loss != interval["loss_coefficients"]:
            raise ValueError("Resistance command differs")
        interval_heat = interval_entropy = 0.0
        interval_start = old
        for step in interval["steps"]:
            dt = step["dt_s"]
            new = step["state"]
            if dt <= 0 or new["model_id"] != parsed.model_id or tuple(new["nodes"]) != names:
                raise ValueError("Invalid transport checkpoint")
            close(new["time_s"] - old["time_s"], dt, atol=1e-13, rtol=1e-10)
            data = {n: decode(new["nodes"][n], nodes[n], model, fluid) for n in names}
            net = {n: np.zeros(nc + 2) for n in names}
            occupied = set()
            for link in model["links"]:
                a, b = link["source"], link["target"]
                left, wa, na, da = endpoint(nodes[a], data[a], link, "source", model)
                right, wb, nb, db = endpoint(nodes[b], data[b], link, "target", model)
                area = math.pi * link["diameter_m"] ** 2 / 4
                if opens[link["name"]]:
                    flux, sf, contact, _ = port_flux(left, right, model["wave_speed_bound_m_s"])
                    for packet, forward in ((da, True), (db, False)):
                        mask = packet["velocity"] > 0 if forward else packet["velocity"] < 0
                        bulk = packet["u"] > 0 if forward else packet["u"] < 0
                        flux += packet["actual"][mask].sum(0) - (
                            packet["bulk"].sum(0) if bulk else 0.0
                        )
                        sf += packet["actual_entropy"][mask].sum() - (
                            packet["bulk_entropy"].sum() if bulk else 0.0
                        )
                else:
                    flux = np.zeros(nc + 2)
                    sf = contact = 0.0
                saved = step["links"][link["name"]]
                close(saved["flux"], area * flux, atol=1e-6, rtol=2e-7)
                close(
                    saved["source_fluid_reaction_impulse_kg_m_s"],
                    dt * area * flux[-2] if nodes[a]["kind"] == "vessel" else 0.0,
                    atol=1e-9,
                    rtol=2e-7,
                )
                close(
                    saved["target_fluid_reaction_impulse_kg_m_s"],
                    -dt * area * flux[-2] if nodes[b]["kind"] == "vessel" else 0.0,
                    atol=1e-9,
                    rtol=2e-7,
                )
                close(saved["entropy_flux_w_k"], area * sf, atol=1e-7, rtol=2e-7)
                close(saved["contact_velocity_m_s"], contact, atol=1e-8, rtol=2e-7)
                close(saved["source_area_fractions"], wa, atol=1e-8, rtol=2e-7)
                close(saved["target_area_fractions"], wb, atol=1e-8, rtol=2e-7)
                for side, n, normal, sign in [("source", a, na, -1.0), ("target", b, nb, 1.0)]:
                    update = np.r_[
                        sign * area * flux[:-2],
                        -normal * area * flux[-2] if nodes[n]["kind"] == "pipe_cell" else 0.0,
                        sign * area * flux[-1],
                    ]
                    if nodes[n]["kind"] == "pipe_cell":
                        occupied.add((n, link[side + "_port"]))
                        if not opens[link["name"]]:
                            update[-2] -= (
                                normal
                                * area
                                * (
                                    data[n]["p"]
                                    + data[n]["rho"] * data[n]["u"] ** 2
                                    + data[n]["stress"]
                                    + normal
                                    * model["wave_speed_bound_m_s"]
                                    * data[n]["rho"]
                                    * data[n]["u"]
                                )
                            )
                    net[n] += update
            production = 0.0
            external = ambient = 0.0
            for n in names:
                node = nodes[n]
                d = data[n]
                a = old["nodes"][n]
                b = new["nodes"][n]
                saved = step["nodes"][n]
                if node["kind"] == "pipe_cell":
                    area = math.pi * node["diameter_m"] ** 2 / 4
                    for port, normal in [("left", -1.0), ("right", 1.0)]:
                        if (n, port) not in occupied:
                            net[n][-2] -= (
                                normal
                                * area
                                * (
                                    d["p"]
                                    + d["rho"] * d["u"] ** 2
                                    + d["stress"]
                                    + normal * model["wave_speed_bound_m_s"] * d["rho"] * d["u"]
                                )
                            )
                    force = (
                        -d["mass"]
                        * model["gravity_m_s2"]
                        * node["elevation_change_m"]
                        / node["length_m"]
                    )
                    if node["dynamic_viscosity_pa_s"]:
                        force -= (
                            node["volume_m3"]
                            * reference_darcy(
                                d["u"],
                                d["rho"],
                                1.0,
                                node["diameter_m"],
                                node["dynamic_viscosity_pa_s"],
                                node["roughness_m"],
                            )[0]
                        )
                    coefficient = command.get("loss_coefficients", {}).get(
                        n, node.get("local_loss_coefficient", 0.0)
                    )
                    force -= 0.5 * area * coefficient * d["rho"] * d["u"] * abs(d["u"])
                    close(
                        b["momentum_kg_m_s"] - a["momentum_kg_m_s"],
                        dt * (net[n][-2] + force),
                        atol=1e-9,
                        rtol=3e-6,
                    )
                else:
                    force = 0.0
                close(saved["force_impulse_kg_m_s"], dt * force, atol=1e-9, rtol=2e-7)
                heat = command.get("heat_w", {}).get(n, 0.0)
                utility = command.get("utility_temperature_k", {}).get(n, a["temperature_k"])
                qh = qa = 0.0
                if node["wall"]:
                    wall = node["wall"]
                    tw = b["wall_temperature_k"]
                    qh = wall["fluid_ua_w_k"] * (tw - d["t"])
                    qa = wall["ambient_ua_w_k"] * (wall["ambient_temperature_k"] - tw)
                    close(
                        wall["heat_capacity_j_k"] * (tw - a["wall_temperature_k"]),
                        dt * (qa - qh),
                        atol=1e-6,
                        rtol=2e-7,
                    )
                    production += (
                        wall["heat_capacity_j_k"] * math.log(tw / a["wall_temperature_k"])
                        - dt * qa / wall["ambient_temperature_k"]
                    )
                close(saved["ambient_heat_j"], dt * qa, atol=1e-8, rtol=2e-7)
                close(saved["fluid_heat_j"], dt * qh, atol=1e-8, rtol=2e-7)
                dn = np.asarray(b["component_moles"]) - a["component_moles"] - dt * net[n][:-2]
                de = b["total_energy_j"] - a["total_energy_j"] - dt * (net[n][-1] + heat + qh)
                close(dn, np.zeros(nc), atol=1e-9, rtol=0.0)
                close(de, 0.0, atol=1e-5, rtol=0.0)
                close(saved["phase_volume_fractions"], d["alpha"], atol=1e-8, rtol=2e-7)
                close(saved["phase_velocity_m_s"], d["velocities"], atol=1e-7, rtol=2e-7)
                close(saved["kinetic_energy_j"], d["kinetic"], atol=1e-7, rtol=2e-7)
                close(saved["potential_energy_j"], d["potential"], atol=1e-7, rtol=2e-7)
                if saved["phase_count"] != len(d["pn"]):
                    raise ValueError("Phase count report differs")
                production += d["entropy"] - olddata[n]["entropy"] - dt * heat / utility
                external += dt * heat
                ambient += dt * qa
                maximum = np.maximum(maximum, np.r_[np.abs(dn), 0.0, abs(de)])
            if production < -1e-6 - 1e-9 * max(1, sum(abs(d["entropy"]) for d in olddata.values())):
                raise ValueError("Independent transport entropy rejected")
            close(step["entropy_production_j_k"], production, atol=2e-6, rtol=2e-6)
            close(step["ambient_heat_j"], ambient, atol=1e-7, rtol=2e-7)
            close(step["external_heat_j"], external, atol=1e-7, rtol=2e-7)
            interval_heat += ambient + external
            interval_entropy += production
            old = new
            olddata = data
            checked += 1
        final = interval["state"]
        last = json.loads(json.dumps(old))
        last["time_s"] = final["time_s"]
        last["next_step_s"] = final["next_step_s"]
        if last != final:
            raise ValueError("Published physical checkpoint differs")
        close(final["time_s"], end, atol=1e-12, rtol=0.0)
        close(interval["total_heat_j"], interval_heat, atol=1e-6, rtol=2e-7)
        close(interval["entropy_production_j_k"], interval_entropy, atol=2e-6, rtol=2e-6)
        close(
            interval["conservation_residuals"],
            ledger(interval_start, final, nodes, interval_heat),
            atol=1e-6,
            rtol=2e-7,
        )
        old = final
        all_heat += interval_heat
    close(report["total_heat_j"], all_heat, atol=1e-6, rtol=2e-7)
    close(
        report["conservation_residuals"],
        ledger(old_start, old, nodes, all_heat),
        atol=1e-6,
        rtol=2e-7,
    )
    return {
        "accepted": True,
        "checked_substeps": checked,
        "maximum_local_component_residual_mol": float(maximum[:nc].max()),
        "maximum_local_energy_residual_j": float(maximum[-1]),
    }


def ledger(before, after, nodes, heat):
    nc = len(next(iter(before["nodes"].values()))["component_moles"])
    dn = [
        math.fsum(s["component_moles"][i] for s in after["nodes"].values())
        - math.fsum(s["component_moles"][i] for s in before["nodes"].values())
        for i in range(nc)
    ]
    energy = 0.0
    for n, node in nodes.items():
        a, b = before["nodes"][n], after["nodes"][n]
        energy += b["total_energy_j"] - a["total_energy_j"]
        if node["wall"]:
            energy += node["wall"]["heat_capacity_j_k"] * (
                b["wall_temperature_k"] - a["wall_temperature_k"]
            )
    if any(abs(x) > 1e-8 for x in dn) or abs(energy - heat) > 1e-5 + 1e-9 * max(1, abs(heat)):
        raise ValueError("Independent global N/E ledger rejected")
    return dn + [energy - heat]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(json.loads(args.config.read_text()), json.loads(args.result.read_text())),
            indent=2,
        )
    )
