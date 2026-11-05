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
from check_pr_equilibrium_transport import decode, disk_area, endpoint
from check_pr_momentum_vessels import port_flux
from check_stratified_pr_flow import close, profile
from check_stratified_pr_network import reference_darcy

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
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        start = old
        end = old["transport"]["time_s"] + command["duration_s"]
        interval_heat = 0.0
        commands = {p["name"]: True for p in tm["links"] + model["ports"]}
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
            net = {name: np.zeros(nc + 2) for name in nodes}
            occupied = set()
            coln = {name: np.zeros(nc) for name in model["columns"]}
            cole = {name: 0.0 for name in coln}
            colentropy = {}
            profiles = {
                name: profile(new["columns"][name], c) for name, c in model["columns"].items()
            }
            for link in tm["links"]:
                a, b = link["source"], link["target"]
                left, _wa, na, da = endpoint(nodes[a], data[a], link, "source", tm)
                right, _wb, nb, db = endpoint(nodes[b], data[b], link, "target", tm)
                if commands[link["name"]]:
                    flux, sflux, _, _ = port_flux(left, right, tm["wave_speed_bound_m_s"])
                    ca, sa = correction(da, True)
                    cb, sb = correction(db, False)
                    flux += ca + cb
                    sflux += sa + sb
                else:
                    flux = np.zeros(nc + 2)
                    sflux = 0.0
                area = math.pi * link["diameter_m"] ** 2 / 4
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
                        occupied.add((name, link[side + "_port"]))
                        if not commands[link["name"]]:
                            d = data[name]
                            update[-2] -= (
                                normal
                                * area
                                * (
                                    d["p"]
                                    + d["stress"]
                                    + d["rho"] * d["u"] ** 2
                                    + normal * tm["wave_speed_bound_m_s"] * d["rho"] * d["u"]
                                )
                            )
                    net[name] += update
            order = step["quadrature_order"]
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
                            net[name][-2] -= (
                                normal
                                * area
                                * (
                                    d["p"]
                                    + d["stress"]
                                    + d["rho"] * d["u"] ** 2
                                    + normal * tm["wave_speed_bound_m_s"] * d["rho"] * d["u"]
                                )
                            )
                    force = -d["mass"] * gravity * node["elevation_change_m"] / node["length_m"]
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
                heat += dt * (qa + external)
                component_balance(a["component_moles"], b["component_moles"], dt * coln[name])
                re = b["total_energy_j"] - a["total_energy_j"] - dt * (cole[name] + qh + external)
                close(re, 0.0, atol=1e-5, rtol=0.0)
                maximum = max(maximum, abs(re))
            close(step["total_heat_j"], heat, atol=1e-6, rtol=2e-7)
            close(step["conservation_residuals"], ledger(old, new, heat), atol=1e-6, rtol=2e-7)
            interval_heat += heat
            checked += 1
            old = new
        close(old["transport"]["time_s"], end, atol=1e-13, rtol=0.0)
        if interval["state"] != old:
            # Adaptive driver updates the next step only, without changing stocks.
            x = json.loads(json.dumps(interval["state"]))
            y = json.loads(json.dumps(old))
            x["transport"]["next_step_s"] = y["transport"]["next_step_s"]
            if x != y:
                raise ValueError("Final interval checkpoint differs")
            old = interval["state"]
        close(interval["total_heat_j"], interval_heat, atol=1e-6, rtol=2e-7)
        close(
            interval["conservation_residuals"],
            ledger(start, old, interval_heat),
            atol=1e-6,
            rtol=2e-7,
        )
        heat_total += interval_heat
    close(report["conservation_residuals"], ledger(first, old, heat_total), atol=1e-6, rtol=2e-7)
    close(report["total_heat_j"], heat_total, atol=1e-6, rtol=2e-7)
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
