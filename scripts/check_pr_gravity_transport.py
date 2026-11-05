"""Independent NumPy PR / SciPy chemical-profile / HLLC graph audit.

SciPy is confined to this verifier. Runtime state decoding, inspection, Newton
and phase flash are never called. This is a numerical audit, not field validation.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from check_pr_equilibrium_transport import (
    audit_heat_links,
    decode,
    disk_area,
    endpoint,
    face_area,
    reference_sound,
    reservoir_trace,
)
from check_pr_momentum_vessels import port_flux
from check_stratified_pr_flow import close, profile
from check_stratified_pr_network import reference_darcy
from reference_hllc_local import wave_flux
from reference_pr_compressors import audit_compressors
from reference_pr_hydrostatic import attach_faces, gravity_force
from reference_pr_population import audit_population_balances, number_flux
from reference_pr_pumps import audit_pumps

from surface.pr_gravity_transport import PRGravityTransportNetwork


def correction(packet, forward, delta_phi=0.0, masses=None):
    mask = packet["velocity"] > 0 if forward else packet["velocity"] < 0
    bulk = packet["u"] > 0 if forward else packet["u"] < 0
    actual = packet["actual"].copy()
    base = packet["bulk"].copy()
    if delta_phi:
        actual[:, -1] += (actual[:, :-2] * masses).sum(-1) * delta_phi
        base[:, -1] += (base[:, :-2] * masses).sum(-1) * delta_phi
    flux = actual[mask].sum(0) - (base.sum(0) if bulk else 0.0)
    entropy = packet["actual_entropy"][mask].sum() - (packet["bulk_entropy"].sum() if bulk else 0.0)
    return flux, entropy


def component_balance(before, after, transfer):
    before, after, transfer = (np.asarray(v, dtype=float) for v in (before, after, transfer))
    tolerance = 1e-20 + 1e-18 * before.sum() + 1e-9 * np.abs(before) + 2e-6 * np.abs(transfer)
    residual = after - before - transfer
    if not np.all(np.isfinite(residual)) or np.any(np.abs(residual) > tolerance):
        raise ValueError("Independent component-specific transport balance rejected")


def audit(config, report):
    if config != report["config"]:
        raise ValueError("Request differs")
    parsed = PRGravityTransportNetwork.from_dict(config["model"])
    model = json.loads(json.dumps(parsed.to_dict()))
    if model != report["model"]:
        raise ValueError("Normalized model differs")
    tm = model["transport"]
    nodes = {n["name"]: n for n in tm["nodes"]}
    fluid = parsed.transport.fluid
    nc = len(fluid.names)
    mw = np.array([c.molar_mass_kg_mol for c in fluid.components])
    gravity = tm["gravity_m_s2"]
    datum = tm["reference_elevation_m"]
    old = report["initial"]
    first = old
    checked = 0
    heat_total = 0.0
    work_total = 0.0
    maximum = 0.0

    def identity(state):
        if (
            state["model_id"] != parsed.model_id
            or state["transport"]["model_id"] != parsed.transport.model_id
        ):
            raise ValueError("Checkpoint identity differs")
        if (
            set(state["columns"]) != set(model["columns"])
            or set(state["transport"]["nodes"]) != set(nodes)
            or set(state["walls"]) != set(model["walls"])
        ):
            raise ValueError("Checkpoint registry differs")
        for name, c in parsed.columns.items():
            if state["columns"][name]["model_id"] != c.model_id:
                raise ValueError("Column identity differs")

    def ledger(a, b, heat):
        def stocks(s):
            n = sum(
                (np.array(v["component_moles"]) for v in s["transport"]["nodes"].values()),
                np.zeros(nc),
            )
            n += sum((np.array(v["component_moles"]) for v in s["columns"].values()), np.zeros(nc))
            e = sum(v["total_energy_j"] for v in s["transport"]["nodes"].values())
            e += sum(
                v["total_energy_j"] - np.dot(v["component_moles"], mw) * gravity * datum
                for v in s["columns"].values()
            )
            e += sum(
                node["wall"]["heat_capacity_j_k"]
                * s["transport"]["nodes"][name]["wall_temperature_k"]
                for name, node in nodes.items()
                if node["wall"]
            )
            e += sum(
                w["heat_capacity_j_k"] * s["walls"][name] for name, w in model["walls"].items()
            )
            return n, e

        na, ea = stocks(a)
        nb, eb = stocks(b)
        r = np.r_[nb - na, eb - ea - heat]
        close(r, np.zeros(nc + 1), atol=1e-5, rtol=0.0)
        return r

    identity(old)

    def entropy_stock(state, decoded, profiles):
        value = sum(d["entropy"] for d in decoded.values()) + sum(p[-1] for p in profiles.values())
        value += sum(
            node["wall"]["heat_capacity_j_k"]
            * math.log(state["transport"]["nodes"][name]["wall_temperature_k"])
            for name, node in nodes.items()
            if node["wall"]
        )
        value += sum(
            wall["heat_capacity_j_k"] * math.log(state["walls"][name])
            for name, wall in model["walls"].items()
        )
        return value

    previous_entropy = entropy_stock(
        old,
        {
            name: decode(old["transport"]["nodes"][name], node, tm, fluid)
            for name, node in nodes.items()
        },
        {name: profile(old["columns"][name], c) for name, c in model["columns"].items()},
    )
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        start = old
        end = old["transport"]["time_s"] + command["duration_s"]
        interval_heat = 0.0
        interval_work = 0.0
        commands = {
            p["name"]: True
            for p in (
                tm["links"]
                + tm.get("heat_links", [])
                + model["ports"]
                + model.get("heat_links", [])
            )
        }
        commands.update({p["card"]["name"]: p["card"]["open"] for p in tm.get("pumps", [])})
        commands.update({unit["name"]: unit["open"] for unit in tm.get("compressors", [])})
        commands.update(command.get("openings", {}))
        if interval["openings"] != commands or not interval["steps"]:
            raise ValueError("Isolation command differs")
        losses = {
            n["name"]: n["local_loss_coefficient"] for n in tm["nodes"] if n["kind"] == "pipe_cell"
        }
        losses.update(command.get("loss_coefficients", {}))
        for step in interval["steps"]:
            new = step["state"]
            identity(new)
            dt = step["dt_s"]
            if dt <= 0:
                raise ValueError("Invalid substep")
            close(
                new["transport"]["time_s"] - old["transport"]["time_s"], dt, atol=1e-13, rtol=1e-10
            )
            data = {
                name: decode(new["transport"]["nodes"][name], node, tm, fluid)
                for name, node in nodes.items()
            }
            attach_faces(
                tm,
                nodes,
                data,
                step["transport"].get("hydrostatic_faces", {}),
                fluid,
                reference_sound,
            )
            net = {name: np.zeros(nc + 2) for name in nodes}
            particle_rates = {
                name: 0.0
                for name, node in nodes.items()
                if node.get("droplet_population") is not None
            }
            occupied = set()
            coln = {name: np.zeros(nc) for name in model["columns"]}
            cole = {name: 0.0 for name in coln}
            colentropy = {}
            profiles = {
                name: profile(new["columns"][name], c) for name, c in model["columns"].items()
            }
            current_entropy = entropy_stock(new, data, profiles)
            production = current_entropy - previous_entropy
            for link in tm["links"]:
                a, b = link["source"], link["target"]
                left_data, right_data = data[a], data[b]
                traces = step["transport"].get("reservoir_traces", {}).get(link["name"], {})
                for side, name, other in [("source", a, b), ("target", b, a)]:
                    eligible = (
                        tm.get("reservoir_ports", "riemann") == "stagnation_single_phase"
                        and commands[link["name"]]
                        and nodes[name]["kind"] == "vessel"
                        and nodes[name]["geometry"] is None
                        and nodes[other]["kind"] == "pipe_cell"
                        and not link[
                            ("target" if side == "source" else "source") + "_port"
                        ].startswith("side:")
                        and len(data[name]["pn"]) == 1
                        and len(data[other]["pn"]) == 1
                    )
                    if eligible:
                        if side not in traces:
                            raise ValueError("Reservoir stagnation trace missing")
                        point = reservoir_trace(data[name], traces[side], side, tm, fluid)
                        if side == "source":
                            left_data = point
                        else:
                            right_data = point
                    elif side in traces:
                        raise ValueError("Unexpected reservoir trace")
                left, _wa, na, da = endpoint(nodes[a], left_data, link, "source", tm)
                right, _wb, nb, db = endpoint(nodes[b], right_data, link, "target", tm)
                if commands[link["name"]]:
                    if (
                        tm.get("wave_fan", "fixed_bound")
                        in ("local_single_phase", "pressure_single_phase")
                        and len(data[a]["pn"]) == 1
                        and len(data[b]["pn"]) == 1
                    ):
                        flux, sflux, contact, _, fallback = wave_flux(
                            left,
                            right,
                            da["sound"],
                            db["sound"],
                            tm["wave_speed_bound_m_s"],
                            estimate="pvrs"
                            if tm.get("wave_fan") == "pressure_single_phase"
                            else "davis",
                        )
                    else:
                        flux, sflux, contact, _ = port_flux(left, right, tm["wave_speed_bound_m_s"])
                        fallback = False
                    base_flux = flux.copy()
                    ca, sa = correction(da, True)
                    cb, sb = correction(db, False)
                    flux += ca + cb
                    sflux += sa + sb
                else:
                    flux = np.zeros(nc + 2)
                    base_flux = flux.copy()
                    sflux = 0.0
                    contact = 0.0
                    fallback = False
                for side, trace in traces.items():
                    close(
                        trace["contact_outward_velocity_m_s"],
                        contact * (1.0 if side == "source" else -1.0),
                        atol=1e-8,
                        rtol=2e-7,
                    )
                if (
                    step["transport"]["links"][link["name"]].get("local_fan_fixed_fallback", False)
                    != fallback
                ):
                    raise ValueError("Acoustic fan fallback differs")
                area = math.pi * link["diameter_m"] ** 2 / 4
                family_nodes = [
                    nodes[n] for n in (a, b) if nodes[n].get("droplet_population") is not None
                ]
                saved_particles = step["transport"]["links"][link["name"]].get(
                    "droplet_number_fluxes", {}
                )
                if family_nodes:
                    family = family_nodes[0]["droplet_population"]["family"]
                    rate = area * number_flux(
                        base_flux,
                        contact,
                        da,
                        db,
                        family_nodes[0]["drift_flux"],
                        data[a]["population"],
                        data[b]["population"],
                        family_nodes[0]["droplet_population"],
                    )
                    if set(saved_particles) != {family}:
                        raise ValueError("Particle flux registry differs")
                    close(saved_particles[family], rate, atol=1e-5, rtol=2e-7)
                    if a in particle_rates:
                        particle_rates[a] -= rate
                    if b in particle_rates:
                        particle_rates[b] += rate
                elif saved_particles:
                    raise ValueError("Unexpected particle flux")
                close(
                    step["transport"]["links"][link["name"]]["flux"],
                    area * flux,
                    atol=1e-6,
                    rtol=2e-7,
                )
                for side, name, normal, sign in [("source", a, na, -1), ("target", b, nb, 1)]:
                    update = np.r_[
                        sign * area * flux[:-2],
                        -normal * area * flux[-2] if nodes[name]["kind"] == "pipe_cell" else 0.0,
                        sign * area * flux[-1],
                    ]
                    if nodes[name]["kind"] == "pipe_cell":
                        if normal == 0.0:
                            outward_mass = -sign * sum(
                                rate * component.molar_mass_kg_mol
                                for rate, component in zip(flux[:-2], fluid.components)
                            )
                            tangent = (
                                da["tangential_velocity"]
                                if side == "source"
                                else db["tangential_velocity"]
                            )
                            update[-2] -= area * max(0.0, outward_mass) * tangent
                        occupied.add((name, link[side + "_port"]))
                        if not commands[link["name"]]:
                            d = data[name]
                            wall = d.get("hydrostatic_faces", {}).get(link[side + "_port"], d)
                            update[-2] -= (
                                normal
                                * area
                                * (
                                    wall["p"]
                                    + d["stress"]
                                    + wall["rho"] * d["u"] ** 2
                                    + normal * tm["wave_speed_bound_m_s"] * wall["rho"] * d["u"]
                                )
                            )
                    net[name] += update
            audit_population_balances(
                nodes,
                old["transport"]["nodes"],
                new["transport"]["nodes"],
                data,
                particle_rates,
                step["transport"].get("droplet_populations", {}),
                dt,
            )
            order = step["quadrature_order"]
            if order == 0:
                if model["columns"] or model["ports"] or model.get("heat_links"):
                    raise ValueError("Missing quadrature for a gravitational or thermal coupling")
                gn, gw = np.empty(0), np.empty(0)
            else:
                gn, gw = np.polynomial.legendre.leggauss(order)
            for port in model["ports"]:
                name = port["column"]
                node = port["node"]
                system = model["columns"][name]
                state = new["columns"][name]
                _, geo, sample, _, _, entropy = profiles[name]
                colentropy[name] = entropy
                bounds = np.r_[0.0, state["interface_heights_m"], geo.height_m]
                fake = {
                    "diameter_m": port["diameter_m"],
                    "target_port": port["pipe_port"],
                    "target_height_m": None,
                }
                pe, _, normal, extra = endpoint(nodes[node], data[node], fake, "target", tm)
                occupied.add((node, port["pipe_port"]))
                radius = port["diameter_m"] / 2
                area = math.pi * radius**2
                total = np.zeros(nc + 2)
                fractions = []
                entropy_flux = 0.0
                for phase in range(len(state["phase_roots"])):
                    low = max(bounds[phase], port["height_m"] - radius)
                    high = max(low, min(bounds[phase + 1], port["height_m"] + radius))
                    fraction = (
                        disk_area(high - port["height_m"] + radius, port["diameter_m"])
                        - disk_area(low - port["height_m"] + radius, port["diameter_m"])
                    ) / area
                    theta_a = math.asin(np.clip((low - port["height_m"]) / radius, -1, 1))
                    theta_b = math.asin(np.clip((high - port["height_m"]) / radius, -1, 1))
                    theta = (theta_a + theta_b) / 2 + (theta_b - theta_a) / 2 * gn
                    raw = gw * np.cos(theta) ** 2
                    weights = fraction * raw / max(raw.sum(), 1e-30)
                    heights = port["height_m"] + radius * np.sin(theta)
                    phaseflux = np.zeros(nc + 2)
                    for h, w in zip(heights, weights):
                        if w <= 0:
                            continue
                        p, x, q = sample(h, phase)
                        phi = gravity * (system["elevation_m"] + h - datum)
                        uc = np.r_[x / q["vm"], 0.0, q["u"] / q["vm"] + q["rho"] * phi]
                        fc = np.r_[np.zeros(nc), p, 0.0]
                        ce = (uc, fc, p, 0.0, q["rho"], q["s"] / q["vm"])
                        pu, pf, pp, pv, pd, ps = pe
                        du = gravity * (h - port["height_m"])
                        pu = pu.copy()
                        pf = pf.copy()
                        pu[-1] += pd * du
                        pf[-1] += pd * du * pv
                        if commands[port["name"]]:
                            flux, sf, _, _ = port_flux(
                                ce, (pu, pf, pp, pv, pd, ps), tm["wave_speed_bound_m_s"]
                            )
                            cf, cs = correction(extra, False, du, mw)
                            flux += cf
                            sf += cs
                        else:
                            d = data[node]
                            pressure = (
                                d["p"]
                                + d["stress"]
                                + d["rho"] * d["u"] ** 2
                                + normal * tm["wave_speed_bound_m_s"] * d["rho"] * d["u"]
                            )
                            flux = np.r_[np.zeros(nc), pressure, 0.0]
                            sf = 0.0
                        phaseflux += w * flux
                        entropy_flux += area * w * sf
                    fractions.append(fraction)
                    total += area * phaseflux
                saved = step["columns"]["ports"][port["name"]]
                close(saved["flux"], total, atol=1e-6, rtol=2e-6)
                close(saved["area_fractions"], fractions, atol=1e-8, rtol=1e-6)
                close(saved["entropy_flux"], entropy_flux, atol=1e-7, rtol=2e-6)
                net[node] += np.r_[total[:-2], -normal * total[-2], total[-1]]
                if commands[port["name"]]:
                    coln[name] -= total[:-2]
                    cole[name] -= total[-1] + np.dot(total[:-2], mw) * gravity * datum
            temperatures = {
                **{name: d["t"] for name, d in data.items()},
                **{name: state["temperature_k"] for name, state in new["columns"].items()},
            }
            for links, saved in [
                (tm.get("heat_links", []), step["transport"].get("heat_links", {})),
                (model.get("heat_links", []), step["columns"].get("heat_links", {})),
            ]:
                for name, rate in audit_heat_links(links, temperatures, commands, saved).items():
                    if name in net:
                        net[name][-1] += rate
                    else:
                        cole[name] += rate
            pump_forces, pump_powers = audit_pumps(
                tm, data, commands, step["transport"].get("pumps", {})
            )
            for name, power in pump_powers.items():
                net[name][-1] += power
            compressor_net, compressor_power = audit_compressors(
                tm, data, commands, step["transport"].get("compressors", {}), fluid
            )
            for name, rate in compressor_net.items():
                net[name] += rate
            work = dt * (sum(pump_powers.values()) + compressor_power)
            close(step["transport"].get("external_work_j", 0.0), work, atol=1e-10, rtol=2e-7)
            heat = 0.0
            for name, node in nodes.items():
                d = data[name]
                a = old["transport"]["nodes"][name]
                b = new["transport"]["nodes"][name]
                force = 0.0
                if node["kind"] == "pipe_cell":
                    area = math.pi * node["diameter_m"] ** 2 / 4
                    for end_port, normal in [("left", -1), ("right", 1)]:
                        if (name, end_port) not in occupied:
                            wall = d.get("hydrostatic_faces", {}).get(end_port, d)
                            net[name][-2] -= (
                                normal
                                * face_area(node, end_port)
                                * (
                                    wall["p"]
                                    + d["stress"]
                                    + wall["rho"] * d["u"] ** 2
                                    + normal * tm["wave_speed_bound_m_s"] * wall["rho"] * d["u"]
                                )
                            )
                    force = gravity_force(node, d, tm, face_area)
                    if node["dynamic_viscosity_pa_s"]:
                        force -= node["volume_m3"] * reference_darcy(
                            d["u"],
                            d["rho"],
                            1.0,
                            node["diameter_m"],
                            node["dynamic_viscosity_pa_s"],
                            node["roughness_m"],
                        )
                    force -= 0.5 * area * losses[name] * d["rho"] * d["u"] * abs(d["u"])
                    force += pump_forces.get(name, 0.0)
                qh = qa = 0.0
                if node["wall"]:
                    w = node["wall"]
                    tw = b["wall_temperature_k"]
                    qh = w["fluid_ua_w_k"] * (tw - d["t"])
                    qa = w["ambient_ua_w_k"] * (w["ambient_temperature_k"] - tw)
                    close(
                        w["heat_capacity_j_k"] * (tw - a["wall_temperature_k"]),
                        dt * (qa - qh),
                        atol=1e-6,
                        rtol=2e-7,
                    )
                external = command.get("heat_w", {}).get(name, 0.0)
                if node["wall"]:
                    production -= dt * qa / node["wall"]["ambient_temperature_k"]
                production -= (
                    dt
                    * external
                    / command.get("utility_temperature_k", {}).get(name, a["temperature_k"])
                )
                heat += dt * (qa + external)
                re = (
                    b["total_energy_j"] - a["total_energy_j"] - dt * (net[name][-1] + qh + external)
                )
                component_balance(a["component_moles"], b["component_moles"], dt * net[name][:-2])
                close(re, 0.0, atol=1e-5, rtol=0.0)
                if node["kind"] == "pipe_cell":
                    close(
                        b["momentum_kg_m_s"] - a["momentum_kg_m_s"],
                        dt * (net[name][-2] + force),
                        atol=1e-9,
                        rtol=2e-6,
                    )
                maximum = max(maximum, abs(re))
            for name, system in model["columns"].items():
                a = old["columns"][name]
                b = new["columns"][name]
                qh = qa = 0.0
                if name in model["walls"]:
                    w = model["walls"][name]
                    tw = new["walls"][name]
                    qh = w["fluid_ua_w_k"] * (tw - b["temperature_k"])
                    qa = w["ambient_ua_w_k"] * (w["ambient_temperature_k"] - tw)
                    close(
                        w["heat_capacity_j_k"] * (tw - old["walls"][name]),
                        dt * (qa - qh),
                        atol=1e-6,
                        rtol=2e-7,
                    )
                external = command.get("heat_w", {}).get(name, 0.0)
                if name in model["walls"]:
                    production -= dt * qa / model["walls"][name]["ambient_temperature_k"]
                production -= (
                    dt
                    * external
                    / command.get("utility_temperature_k", {}).get(name, a["temperature_k"])
                )
                heat += dt * (qa + external)
                component_balance(a["component_moles"], b["component_moles"], dt * coln[name])
                re = b["total_energy_j"] - a["total_energy_j"] - dt * (cole[name] + qh + external)
                close(re, 0.0, atol=1e-5, rtol=0.0)
                maximum = max(maximum, abs(re))
            close(step["total_heat_j"], heat, atol=1e-6, rtol=2e-7)
            close(step.get("total_work_j", 0.0), work, atol=1e-10, rtol=2e-7)
            if production < -1e-6 - 1e-9 * max(1.0, abs(previous_entropy)):
                raise ValueError("Independent coupled entropy rejected")
            close(step["entropy_production_j_k"], production, atol=2e-6, rtol=2e-6)
            close(
                step["conservation_residuals"], ledger(old, new, heat + work), atol=1e-6, rtol=2e-7
            )
            interval_heat += heat
            interval_work += work
            checked += 1
            old = new
            previous_entropy = current_entropy
        close(old["transport"]["time_s"], end, atol=1e-13, rtol=0.0)
        if interval["state"] != old:
            # Adaptive driver updates the next step only, without changing stocks.
            x = json.loads(json.dumps(interval["state"]))
            y = json.loads(json.dumps(old))
            x["transport"]["next_step_s"] = y["transport"]["next_step_s"]
            # A parent controller canonicalizes the final reporting clock.
            # Permit only floating-point rounding of time; stocks, pressures,
            # temperatures and every other physical field still compare exactly.
            clock_scale = max(abs(end), abs(dt), 1e-300)
            close(
                x["transport"]["time_s"],
                y["transport"]["time_s"],
                atol=16 * np.finfo(float).eps * clock_scale,
                rtol=0.0,
            )
            x["transport"]["time_s"] = y["transport"]["time_s"]
            if x != y:
                raise ValueError("Final interval checkpoint differs")
            old = interval["state"]
        close(interval["total_heat_j"], interval_heat, atol=1e-6, rtol=2e-7)
        close(interval.get("total_work_j", 0.0), interval_work, atol=1e-10, rtol=2e-7)
        close(
            interval["conservation_residuals"],
            ledger(start, old, interval_heat + interval_work),
            atol=1e-6,
            rtol=2e-7,
        )
        heat_total += interval_heat
        work_total += interval_work
    close(
        report["conservation_residuals"],
        ledger(first, old, heat_total + work_total),
        atol=1e-6,
        rtol=2e-7,
    )
    close(report["total_heat_j"], heat_total, atol=1e-6, rtol=2e-7)
    close(report.get("total_work_j", 0.0), work_total, atol=1e-10, rtol=2e-7)
    return {
        "accepted": True,
        "checked_substeps": checked,
        "maximum_local_energy_residual_j": maximum,
    }


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
