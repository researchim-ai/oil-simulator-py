"""Independent NumPy PR/32-node aperture and per-step N/E audit.

No Torch vessel solve or inspect. Does not certify global phase stability,
non-equilibrium physics, or empirical valve accuracy.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from check_stratified_pr import Reference, grid, softmax
from scipy.optimize import root

from surface.equilibrium_properties import EquilibriumFluid
from surface.peng_robinson import R
from surface.separator_geometry import SeparatorGeometry


def close(actual, expected, atol=1e-7, rtol=2e-6):
    a, b = np.asarray(actual), np.asarray(expected)
    if (
        a.shape != b.shape
        or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
        or np.max(np.abs(a - b) / (atol + rtol * np.maximum(np.abs(a), np.abs(b)))) > 1
    ):
        raise ValueError("Independent flow audit mismatch")


def profile(state, system):
    f = EquilibriumFluid.from_dict(system["fluid"])
    n = np.array(state["component_moles"])
    active = np.flatnonzero(n > 0)
    ref = Reference(f, active)
    count = len(active)
    order = state["quadrature_order"]
    phases = len(state["phase_roots"])
    variables = np.array(state["solution_variables"])
    offset = (phases * order + 2 * (phases - 1)) * count
    if variables.shape != (offset + count + phases,):
        raise ValueError("Malformed collocation state")
    local = variables[:offset].reshape(-1, count)
    close(
        variables[offset : offset + count],
        np.array(state["chemical_constants"])[active],
        atol=1e-9,
        rtol=0.0,
    )
    close(variables[offset + count] * f.tref, state["temperature_k"], atol=1e-8, rtol=0.0)
    interfaces = (
        np.cumsum(softmax(np.r_[variables[offset + count + 1 :], 0.0]))[:-1]
        * SeparatorGeometry(**system["geometry"]).height_m
    )
    if len(interfaces):
        close(interfaces, state["interface_heights_m"], atol=1e-8, rtol=0.0)
    t = state["temperature_k"]
    lam = np.array(state["chemical_constants"])[active]
    g = system["gravity_m_s2"]
    base = system["elevation_m"]
    geometry = SeparatorGeometry(**system["geometry"])
    heights, weights = grid(geometry, state["interface_heights_m"], order)
    total = np.zeros_like(n)
    energy = entropy = 0.0
    amounts = []
    densities = []
    for i, (hh, ww) in enumerate(zip(heights, weights)):
        phase = np.zeros_like(n)
        for j, (h, dv) in enumerate(zip(hh, ww)):
            p, x, q = ref.local(local[i * order + j], t, state["phase_roots"][i])
            close(
                np.log(x[active]) + q["phi"] + np.log(p / f.pref) + ref.mw * g * h / (R * t),
                lam,
                atol=2e-7,
                rtol=0.0,
            )
            phase += dv * x / q["vm"]
            energy += dv * (q["u"] / q["vm"] + q["rho"] * g * (base + h))
            entropy += dv * q["s"] / q["vm"]
        total += phase
        amounts.append(phase)
        densities.append(float(phase[active] @ ref.mw) / sum(ww))
    close(total, n, atol=1e-8, rtol=2e-8)
    close(energy, state["total_energy_j"], atol=1e-5, rtol=2e-8)

    def sample(h, i):
        def residual(v):
            p, x, q = ref.local(v, t, state["phase_roots"][i])
            return (
                np.log(x[active]) + q["phi"] + np.log(p / f.pref) + ref.mw * g * h / (R * t) - lam
            )

        solved = root(residual, local[i * order], tol=1e-10)
        if max(abs(residual(solved.x))) > 1e-7:
            raise ValueError("Independent port state unresolved")
        return ref.local(solved.x, t, state["phase_roots"][i])

    for i, h in enumerate(state["interface_heights_m"]):
        a, b = sample(h, i), sample(h, i + 1)
        close(a[0], b[0], atol=0.01, rtol=1e-8)
        if a[2]["rho"] <= b[2]["rho"]:
            raise ValueError("Unstable density order")
    return f, geometry, sample, np.array(amounts), np.array(densities), entropy


def audit(config, report):
    if report["config"] != config:
        raise ValueError("Config differs from supplied request")
    system = config["column"]
    wall = config["wall"]
    cw = wall["heat_capacity_j_k"]
    ua = wall["fluid_ua_w_k"]
    ambient_ua = wall.get("ambient_ua_w_k", 0.0)
    tamb = wall.get("ambient_temperature_k", 298.15)
    old = report["initial"]["state"]
    tw0 = config["wall_temperature_k"]
    old_entropy = profile(old, system)[-1]
    time = 0.0
    maximum_energy_error = 0.0
    checked = 0
    for interval, duration in zip(report["intervals"], config["durations_s"], strict=True):
        end = time + duration
        interval_heat = interval_feed = interval_out = 0.0
        for step in interval["steps"]:
            state = step["column_state"]
            dt = step["dt_s"]
            tw = step["wall_temperature_k"]
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("Invalid substep")
            f, geo, sample, amounts, density, s = profile(state, system)
            feed_n = sum(
                (np.array(v["stream"]["component_molar_rates_mol_s"]) for v in config["feeds"]),
                np.zeros(len(f.names)),
            )
            feed_e = math.fsum(v["stream"]["total_energy_rate_w"] for v in config["feeds"])
            out_n = np.zeros(len(f.names))
            out_e = out_s = 0.0
            mw = np.array([component.molar_mass_kg_mol for component in f.components])
            bounds = np.r_[0, state["interface_heights_m"], geo.height_m]
            mean_x = amounts / amounts.sum(-1, keepdims=True)
            nodes, weights = np.polynomial.legendre.leggauss(32)
            for port, saved in zip(config["outlets"], step["outlets"], strict=True):
                radius = port["diameter_m"] / 2
                center = port["height_m"]
                pieces = []
                mean_p = 0.0
                for i in range(len(amounts)):
                    lo = max(bounds[i], center - radius)
                    hi = min(bounds[i + 1], center + radius)
                    if hi <= lo:
                        continue
                    a = math.asin(np.clip((lo - center) / radius, -1, 1))
                    b = math.asin(np.clip((hi - center) / radius, -1, 1))
                    angles = (a + b) / 2 + (b - a) / 2 * nodes
                    fraction = ((b - a) + (math.sin(2 * b) - math.sin(2 * a)) / 2) / math.pi
                    ww = weights * np.cos(angles) ** 2
                    ww = ww / ww.sum() * fraction
                    for h, area in zip(center + radius * np.sin(angles), ww):
                        p, x, q = sample(h, i)
                        pieces.append((i, h, area, p, x, q))
                        mean_p += p * area
                opening = port.get("opening", 1.0)
                if port.get("pressure_setpoint_pa") is not None:
                    opening = np.clip(
                        port.get("opening_bias", 0.0)
                        + port.get("opening_gain_per_pa", 0.0)
                        * (mean_p - port["pressure_setpoint_pa"]),
                        0,
                        1,
                    )
                control = port.get("level_control")
                if control:
                    values = mean_x[:, f.names.index(control["component"])]
                    index = values.argmax()
                    opening = (
                        float(
                            np.clip(
                                control.get("opening_bias", 0.0)
                                + control["opening_gain_per_m"]
                                * (bounds[index + 1] - control["setpoint_m"]),
                                0,
                                1,
                            )
                        )
                        if values[index] > 0
                        and values[index] >= control.get("minimum_mole_fraction", 0.5)
                        else 0.0
                    )
                pn = np.zeros(len(f.names))
                pe = volume = 0.0
                for i, h, area, p, x, q in pieces:
                    selection = port.get("selection", "bulk")
                    selected = True
                    if selection == "densest":
                        selected = i == density.argmax()
                    if selection == "lightest":
                        selected = i == density.argmin()
                    if selection == "component_rich":
                        values = mean_x[:, f.names.index(port["component"])]
                        selected = (
                            i == values.argmax()
                            and values[i] > 0
                            and values[i] >= port.get("minimum_mole_fraction", 0.5)
                        )
                    vdot = (
                        port["volume_conductance_m3_s_pa"]
                        * opening
                        * area
                        * max(p - port["back_pressure_pa"], 0)
                        * selected
                    )
                    rate = vdot / q["vm"]
                    pn += rate * x
                    pe += rate * (
                        q["h"]
                        + float(x @ mw) * system["gravity_m_s2"] * (system["elevation_m"] + h)
                    )
                    out_s += rate * q["s"]
                    volume += vdot
                close(pn, saved["component_rates_mol_s"], atol=1e-12)
                close(pe, saved["total_energy_rate_w"], atol=1e-7)
                close(volume, saved["volume_rate_m3_s"], atol=1e-14)
                close(opening, saved["opening"], atol=1e-8)
                out_n += pn
                out_e += pe
            close(
                np.array(state["component_moles"]) - old["component_moles"],
                dt * (feed_n - out_n),
                atol=1e-8,
            )
            close(step["feed_component_moles"], dt * feed_n, atol=1e-10)
            close(step["outlet_component_moles"], dt * out_n, atol=1e-10)
            close(step["feed_energy_j"], dt * feed_e, atol=1e-7)
            close(step["outlet_energy_j"], dt * out_e, atol=1e-7)
            ambient = ambient_ua * (tamb - tw)
            heat = ua * (tw - state["temperature_k"])
            close(cw * (tw - tw0), dt * (ambient - heat), atol=1e-6)
            close(
                state["total_energy_j"] - old["total_energy_j"],
                dt * (feed_e - out_e + heat),
                atol=1e-6,
            )
            close(step["ambient_heat_j"], dt * ambient, atol=1e-7)
            error = (
                state["total_energy_j"]
                - old["total_energy_j"]
                + cw * (tw - tw0)
                - dt * (feed_e - out_e + ambient)
            )
            if not config["feeds"]:
                production = (
                    s - old_entropy + cw * math.log(tw / tw0) + dt * (out_s - ambient / tamb)
                )
                if production < -1e-6 - 1e-9 * max(1, abs(s), abs(old_entropy)):
                    raise ValueError("Independent entropy balance rejected")
                close(step["entropy_production_j_k"], production, atol=2e-6)
            old_entropy = s
            maximum_energy_error = max(maximum_energy_error, abs(error))
            interval_heat += dt * ambient
            interval_feed += dt * feed_e
            interval_out += dt * out_e
            old, tw0 = state, tw
            time += dt
            checked += 1
        close(time, end, atol=1e-10, rtol=1e-12)
        close(interval["ambient_heat_j"], interval_heat)
        close(interval["feed_energy_j"], interval_feed)
        close(interval["outlet_energy_j"], interval_out)
        if interval["state"]["column"] != old:
            raise ValueError("Interval final column differs")
        close(interval["state"]["wall_temperature_k"], tw0)
        close(interval["state"]["time_s"], time, atol=1e-10, rtol=1e-12)
    return {
        "accepted": True,
        "checked_substeps": checked,
        "reference_port_quadrature": 32,
        "maximum_energy_residual_j": maximum_energy_error,
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
