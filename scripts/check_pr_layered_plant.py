"""Independent common-plant N/E/entropy, profile momentum and support audit."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from check_pr_equilibrium_transport import audit as ordinary_audit
from check_pr_equilibrium_transport import decode
from check_stratified_pr import Reference
from check_stratified_pr_flow import close, profile
from reference_inclined_cylinder import cut
from reference_pr_drives import initial_rotors
from reference_pr_layered_plant import exchanges
from scipy.optimize import brentq

from surface.peng_robinson import R
from surface.pr_layered_plant import PRLayeredPlantNetwork


def uniform_reference(fluid, requested, volume):
    """Scalar PR PT flash, independent of the runtime inventory initializer."""
    z = np.asarray(requested["mole_fractions"], dtype=float)
    if z.shape != (len(fluid.names),) or not np.isfinite(z).all() or np.any(z < 0):
        raise ValueError("Invalid requested initial composition")
    close(z.sum(), 1.0, atol=1e-10, rtol=0.0)
    active = np.flatnonzero(z > 0)
    ref = Reference(fluid, active)
    p, t = requested["pressure_pa"], requested["temperature_k"]
    if ref.pure is not None:
        phases = [(1.0, ref.state(p, t, z, "stable"))]
    else:
        flashed = ref.mixture.flash_pt(p, t, z[active])
        phases = []
        for phase in flashed["phases"]:
            x = np.zeros_like(z)
            x[active] = phase["mole_fractions"]
            candidates = [ref.state(p, t, x, tag) for tag in ("liquid", "vapor")]
            q = min(candidates, key=lambda q: abs(q["z"] - phase["compressibility_factor"]))
            phases.append((phase["molar_fraction"], q))
    total = volume / sum(beta * q["vm"] for beta, q in phases)
    return total * z, [(total * beta, q) for beta, q in phases]


def audit_initial(model, state, requested, fluid):
    """Bind requested uniform PT stocks to the reported equilibrated profiles."""
    if set(requested) == {"state"}:
        if requested["state"] != state:
            raise ValueError("Requested restart differs")
        return
    if set(requested) != {"transport", "layered_equilibrium"}:
        raise ValueError("Unknown requested plant initialization")
    transport = model["transport"]
    nodes = {n["name"]: n for n in transport["nodes"]}
    if set(requested["transport"]) != set(nodes) or set(state["transport"]["nodes"]) != set(nodes):
        raise ValueError("Requested initial node registry differs")
    if state["transport"]["time_s"] != 0.0 or state["layered"]["time_s"] != 0.0:
        raise ValueError("Requested initialization must start at zero time")
    initial_rotors(transport, state["transport"])
    for name, node in nodes.items():
        spec = requested["transport"][name]
        stock = state["transport"]["nodes"][name]
        d = decode(stock, node, transport, fluid)
        close(d["p"], spec["pressure_pa"], atol=1e-5, rtol=1e-9)
        close(d["t"], spec["temperature_k"], atol=1e-7, rtol=0.0)
        n, _ = uniform_reference(fluid, spec, node["volume_m3"])
        close(d["n"], n, atol=1e-8, rtol=2e-8)
        close(d["u"], spec.get("velocity_m_s", 0.0), atol=1e-10, rtol=1e-9)
        if node["wall"] is not None:
            close(
                stock["wall_temperature_k"],
                spec.get("wall_temperature_k", spec["temperature_k"]),
                atol=1e-8,
                rtol=0.0,
            )

    columns = model["layered"]["columns"]
    cells = state["layered"]["cells"]
    if len(cells) != len(columns):
        raise ValueError("Initial profile cell registry differs")
    geometry = columns[0]["geometry"]
    diameter, angle = geometry["diameter_m"], geometry["inclination_rad"]
    length = sum(c["geometry"]["length_m"] for c in columns)
    volume = math.pi * diameter**2 * length / 4
    height = length * abs(math.sin(angle)) + diameter * math.cos(angle)
    base = min(c["elevation_m"] for c in columns)
    gravity = columns[0]["gravity_m_s2"]
    expected_n, phases = uniform_reference(fluid, requested["layered_equilibrium"], volume)
    energy = entropy = cumulative_volume = prior_moment = 0.0
    for amount, q in sorted(phases, key=lambda v: -v[1]["rho"]):
        dv = amount * q["vm"]
        cumulative_volume = min(volume, cumulative_volume + dv)
        top = brentq(
            lambda h, target=cumulative_volume: cut(diameter, length, angle, h)[0] - target,
            0.0,
            height,
            xtol=1e-14 * height,
        )
        next_moment = cut(diameter, length, angle, top)[1]
        energy += amount * q["u"] + q["rho"] * gravity * (base * dv + next_moment - prior_moment)
        entropy += amount * q["s"]
        prior_moment = next_moment
    actual_n = np.zeros_like(expected_n)
    actual_energy = actual_entropy = 0.0
    common_constants = common_temperature = None
    mw = np.array([c.molar_mass_kg_mol for c in fluid.components])
    active = expected_n > 0
    for cfg, cell in zip(columns, cells, strict=True):
        stock = cell["profile"]
        actual_entropy += profile(stock, cfg)[-1]
        actual_n += stock["component_moles"]
        actual_energy += cell["total_energy_j"]
        close(cell["momentum_kg_m_s"], 0.0, atol=1e-12, rtol=0.0)
        close(cell["total_energy_j"], stock["total_energy_j"], atol=1e-7, rtol=1e-12)
        temperature = stock["temperature_k"]
        constants = np.asarray(stock["chemical_constants"])[active] + (
            mw[active] * gravity * (cfg["elevation_m"] - base) / (R * temperature)
        )
        if common_temperature is None:
            common_temperature, common_constants = temperature, constants
        else:
            close(temperature, common_temperature, atol=1e-7, rtol=0.0)
            close(constants, common_constants, atol=2e-7, rtol=0.0)
    # Independent scalar and tensor PT flashes have different convergence tolerances.
    close(actual_n, expected_n, atol=1e-8, rtol=2e-7)
    close(actual_energy, energy, atol=1e-5, rtol=2e-7)
    if actual_entropy < entropy - 1e-6 - 1e-10 * max(1.0, abs(entropy)):
        raise ValueError("Initial profile equilibrium lowered entropy")


def audit_profile_step(model, old, new, report, command, d):
    """Independently verify one branch, regardless of the surrounding graph."""
    dt = new["layered"]["time_s"] - old["layered"]["time_s"]
    if not dt > 0:
        raise ValueError("Nonpositive profile step")
    mw, g = d["molar_masses"], d["gravity_acceleration"]
    rates = d["rates"]
    pressure = d["wall_pressure"]
    heat = command.get("profile_heat_w", [0.0] * len(rates))
    if isinstance(heat, (int, float)):
        heat = [heat] * len(rates)
    heat = np.array(heat, dtype=float)
    close(report["heat_j"], dt * heat, atol=1e-9, rtol=1e-10)
    close(report["source_n"], d["source"], atol=2e-7, rtol=1e-5)
    close(d["source"], d["gravity"], atol=2e-7, rtol=1e-4)
    entropy_profile = 0.0
    old_entropy_profile = 0.0
    for i, (a, b, cfg) in enumerate(
        zip(
            old["layered"]["cells"],
            new["layered"]["cells"],
            model["layered"]["columns"],
            strict=True,
        )
    ):
        na = np.asarray(a["profile"]["component_moles"])
        nb = np.asarray(b["profile"]["component_moles"])
        close(nb - na, dt * rates[i, :-2], atol=1e-8, rtol=2e-5)
        energy_rate = rates[i, -1] + rates[i, :-2] @ mw * g * d["datum"] + heat[i]
        close(
            b["total_energy_j"] - a["total_energy_j"],
            dt * energy_rate,
            atol=2e-5,
            rtol=2e-5,
        )
        close(
            b["total_energy_j"],
            b["profile"]["total_energy_j"] + 0.5 * b["momentum_kg_m_s"] ** 2 / d["mass"][i],
            atol=1e-7,
            rtol=1e-12,
        )
        wall = (pressure[i, 0] if i == 0 and model["boundary_nodes"][0] is None else 0.0) - (
            pressure[i, 1] if i == len(rates) - 1 and model["boundary_nodes"][1] is None else 0.0
        )
        close(
            b["momentum_kg_m_s"] - a["momentum_kg_m_s"],
            dt * (rates[i, -2] + wall + d["source"][i] + d["friction_force_n"][i]),
            atol=2e-8,
            rtol=2e-5,
        )
        old_entropy_profile += profile(a["profile"], cfg)[-1]
        entropy_profile += profile(b["profile"], cfg)[-1]
    entropy_production = entropy_profile - old_entropy_profile + dt * d["incoming_entropy"]
    if np.any(heat):
        utilities = np.asarray(command["profile_utility_temperature_k"])
        entropy_production -= dt * np.sum(heat / utilities)
        for q, u, c in zip(heat, utilities, new["layered"]["cells"]):
            if q * (u - c["profile"]["temperature_k"]) < -1e-10:
                raise ValueError("Invalid passive profile utility")
    if entropy_production < -1e-6:
        raise ValueError("Independent profile entropy rejected")
    close(report["friction_force_n"], d["friction_force_n"], atol=1e-8, rtol=1e-6)
    close(report["wall_pressure"], d["wall_pressure"], atol=2e-7, rtol=1e-5)
    return entropy_production, heat


def audit(config, report):
    if config != report["config"]:
        raise ValueError("Common plant request differs")
    parsed = PRLayeredPlantNetwork.from_dict(config["model"])
    model = json.loads(json.dumps(parsed.to_dict()))
    if model != report["model"]:
        raise ValueError("Common plant normalized model differs")
    if len(config["schedule"]) != len(report["intervals"]):
        raise ValueError("Schedule length differs")
    t = model["transport"]
    names = [n["name"] for n in t["nodes"]]
    nodes = {n["name"]: n for n in t["nodes"]}
    nc = len(t["fluid"]["components"])
    mw = np.array([c["molar_mass_kg_mol"] for c in t["fluid"]["components"]])
    g = t["gravity_m_s2"]
    datum = t["reference_elevation_m"]
    old = report["initial"]
    audit_initial(model, old, config["initial"], parsed.transport.fluid)
    first = old
    cache = {}
    ordinary_intervals = []
    ordinary_commands = []
    total_heat = total_work = 0.0
    checked = 0

    def stocks(s):
        amounts = np.zeros(nc)
        energy = 0.0
        for name, node in nodes.items():
            v = s["transport"]["nodes"][name]
            amounts += v["component_moles"]
            energy += v["total_energy_j"]
            if node["wall"] is not None:
                energy += node["wall"]["heat_capacity_j_k"] * v["wall_temperature_k"]
        drives = {d["machine"]: d for d in t.get("drives", [])}
        energy += sum(
            0.5 * drives[n]["drive"]["inertia_kg_m2"] * w * w
            for n, w in s["transport"].get("rotors", [])
        )
        for cell in s["layered"]["cells"]:
            n = np.array(cell["profile"]["component_moles"])
            amounts += n
            energy += cell["total_energy_j"] - (n @ mw) * g * datum
        return amounts, energy

    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        if not interval["steps"]:
            raise ValueError("No accepted coupled steps")
        start = old
        ih = iw = 0.0
        ordinary_heat = ordinary_work = ordinary_entropy = 0.0
        ordinary_steps = []
        for step in interval["steps"]:
            new = step["state"]
            dt = step["dt_s"]
            close(
                new["transport"]["time_s"] - old["transport"]["time_s"],
                dt,
                atol=8
                * max(math.ulp(new["transport"]["time_s"]), math.ulp(old["transport"]["time_s"])),
                rtol=0.0,
            )
            if new["layered"]["time_s"] != new["transport"]["time_s"]:
                raise ValueError("Component clocks differ")
            data = {
                n: decode(new["transport"]["nodes"][n], nodes[n], t, parsed.transport.fluid)
                for n in names
            }
            d = exchanges(model, new, data)
            key = new["transport"]["time_s"]
            if key in cache:
                raise ValueError("Duplicate accepted timestamp")
            cache[key] = d
            close(
                step["vessel_support_force_n"],
                [d["support"][n] for n in names],
                atol=2e-6,
                rtol=2e-6,
            )
            entropy_production, heat = audit_profile_step(
                model, old, new, step["profile"], command, d
            )
            ordinary = step["transport"]
            ordinary_steps.append(ordinary)
            ordinary_heat += ordinary["ambient_heat_j"] + ordinary["external_heat_j"]
            ordinary_work += ordinary["external_work_j"]
            ordinary_entropy += ordinary["entropy_production_j_k"]
            sh = ordinary["ambient_heat_j"] + ordinary["external_heat_j"] + dt * sum(heat)
            sw = ordinary["external_work_j"]
            close(step["total_heat_j"], sh, atol=1e-8, rtol=2e-7)
            close(step["total_work_j"], sw, atol=1e-10, rtol=2e-7)
            close(
                step["entropy_production_j_k"],
                ordinary["entropy_production_j_k"] + entropy_production,
                atol=2e-6,
                rtol=2e-6,
            )
            na, ea = stocks(old)
            nb, eb = stocks(new)
            residual = np.r_[nb - na, eb - ea - sh - sw]
            close(residual, np.zeros(nc + 1), atol=2e-5, rtol=0.0)
            close(step["conservation_residuals"], residual, atol=1e-6, rtol=2e-7)
            old = new
            ih += sh
            iw += sw
            checked += 1
        close(
            old["transport"]["time_s"],
            start["transport"]["time_s"] + command["duration_s"],
            atol=8 * math.ulp(old["transport"]["time_s"]),
            rtol=0.0,
        )
        published = interval["state"]
        for key in ("transport", "layered"):
            aa = {k: v for k, v in old[key].items() if k != "next_step_s"}
            bb = {k: v for k, v in published[key].items() if k != "next_step_s"}
            if aa != bb:
                raise ValueError("Published component differs from accepted fine steps")
        old = published
        total_heat += ih
        total_work += iw
        close(interval["total_heat_j"], ih, atol=1e-7, rtol=2e-7)
        close(interval["total_work_j"], iw, atol=1e-10, rtol=2e-7)
        na, ea = stocks(start)
        nb, eb = stocks(old)
        close(
            interval["conservation_residuals"],
            np.r_[nb - na, eb - ea - ih - iw],
            atol=1e-6,
            rtol=2e-7,
        )
        allowed = {
            "duration_s",
            "openings",
            "heat_w",
            "utility_temperature_k",
            "loss_coefficients",
            "drive_torque_nm",
            "restriction_openings",
        }
        cmd = {k: v for k, v in command.items() if k in allowed}
        ordinary_commands.append(cmd)
        opened = {l["name"]: True for l in t["links"] + t.get("heat_links", [])}
        opened.update(
            {
                p["card"]["name"]: p["card"]["open"]
                for p in t.get("pumps", []) + t.get("pump_connections", [])
            }
        )
        opened.update(
            {c["name"]: c["open"] for c in t.get("compressors", []) + t.get("restrictions", [])}
        )
        opened.update(command.get("openings", {}))
        ordinary_intervals.append(
            {
                "state": old["transport"],
                "steps": ordinary_steps,
                "openings": opened,
                "loss_coefficients": {
                    n: command.get("loss_coefficients", {}).get(
                        n, v.get("local_loss_coefficient", 0.0)
                    )
                    for n, v in nodes.items()
                    if v["kind"] == "pipe_cell"
                },
                "total_heat_j": ordinary_heat,
                "total_work_j": ordinary_work,
                "entropy_production_j_k": ordinary_entropy,
                "conservation_residuals": [0.0] * (nc + 1),
            }
        )
    ordinary_config = {
        "model": t,
        "initial": {"state": first["transport"]},
        "schedule": ordinary_commands,
    }

    def external(_a, b, _data, _dt):
        d = cache[b["time_s"]]
        return {"net": d["ordinary_net"], "incoming_entropy": d["incoming_entropy"]}

    ordinary_report = {
        "config": ordinary_config,
        "model": t,
        "initial": first["transport"],
        "intervals": ordinary_intervals,
        "total_heat_j": sum(i["total_heat_j"] for i in ordinary_intervals),
        "total_work_j": sum(i["total_work_j"] for i in ordinary_intervals),
        "conservation_residuals": [0.0] * (nc + 1),
    }
    result = ordinary_audit(ordinary_config, ordinary_report, external_exchange=external)
    close(report["total_heat_j"], total_heat, atol=1e-7, rtol=2e-7)
    close(report["total_work_j"], total_work, atol=1e-10, rtol=2e-7)
    if report["state"] != old:
        raise ValueError("Final combined state differs")
    na, ea = stocks(first)
    nb, eb = stocks(old)
    close(
        report["conservation_residuals"],
        np.r_[nb - na, eb - ea - total_heat - total_work],
        atol=1e-6,
        rtol=2e-7,
    )
    return {"accepted": True, "checked_substeps": checked, "ordinary": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    report = json.loads(args.report.read_text())
    root = Path(__file__).resolve().parents[1]
    if not report.get("sources_unchanged_during_run") or not report.get("source_sha256"):
        raise ValueError("Missing source provenance")
    for p, h in report["source_sha256"].items():
        path = (root / p).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != h:
            raise ValueError("Stale common-plant sources")
    print(json.dumps(audit(config, report), indent=2))


if __name__ == "__main__":
    main()
