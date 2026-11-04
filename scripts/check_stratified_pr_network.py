"""Independent NumPy PR/32-point audit of closed direct-connector networks.

No Torch vessel/network solve or inspect. External material boundaries are
explicitly unsupported here; ambient heat and wall storage are included.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from check_stratified_pr_flow import close, profile
from scipy.optimize import brentq


def reference_darcy(velocity, density, length, diameter, viscosity, roughness):
    reynolds = density * abs(velocity) * diameter / viscosity
    laminar = 32 * viscosity * length / diameter**2 * velocity
    factor = (
        -1.8 * math.log10((roughness / (3.7 * diameter)) ** 1.11 + 6.9 / max(2000.0, reynolds))
    ) ** -2
    transition = max(0.0, min(1.0, (reynolds - 2000) / 2000))
    transition = transition**2 * (3 - 2 * transition)
    return (
        1 - transition
    ) * laminar + transition * factor * length / diameter * density * velocity * abs(
        velocity
    ) / 2, reynolds


def audit(config, report):
    if report["config"] != config:
        raise ValueError("Network config differs from supplied request")
    descriptions = config["vessels"]
    for d in descriptions.values():
        if d["model"]["feeds"] or d["model"]["outlets"]:
            raise ValueError(
                "Independent network auditor currently requires closed material boundaries"
            )
    old = report["initial"]
    time = old["time_s"]
    checked = 0
    maximum_error = 0.0
    names = tuple(sorted(descriptions))
    if tuple(sorted(old["vessels"])) != names:
        raise ValueError("Initial network vessel registry differs")
    previous = {
        name: profile(old["vessels"][name]["column"], d["model"]["column"])
        for name, d in descriptions.items()
    }
    nodes, weights = np.polynomial.legendre.leggauss(32)
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        if (
            not math.isfinite(command["duration_s"])
            or command["duration_s"] <= 0
            or not interval["steps"]
        ):
            raise ValueError("Invalid or empty network interval")
        target = time + command["duration_s"]
        expected_openings = {
            p["name"]: command.get("openings", {}).get(p["name"], p.get("opening", 1.0))
            for p in config["connections"]
        }
        ambient_interval = production_interval = 0.0
        initial_energy = math.fsum(
            old["vessels"][n]["column"]["total_energy_j"]
            + descriptions[n]["model"]["wall"]["heat_capacity_j_k"]
            * (
                old["vessels"][n]["wall_temperature_k"]
                - descriptions[n]["model"]["wall"].get("reference_temperature_k", 298.15)
            )
            for n in names
        )
        interval_n = {n: np.array(old["vessels"][n]["column"]["component_moles"]) for n in names}
        for step in interval["steps"]:
            if step["openings"] != expected_openings:
                raise ValueError("Network opening command differs")
            state = step["state"]
            dt = step["dt_s"]
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("Invalid network substep")
            if tuple(sorted(state["vessels"])) != names:
                raise ValueError("Network vessel registry differs")
            current = {
                name: profile(state["vessels"][name]["column"], d["model"]["column"])
                for name, d in descriptions.items()
            }
            nnet = {
                n: np.zeros_like(np.array(state["vessels"][n]["column"]["component_moles"]))
                for n in names
            }
            enet = {n: 0.0 for n in names}
            snet = {n: 0.0 for n in names}
            if len(step["transfers"]) != len(config["connections"]):
                raise ValueError("Network transfer registry differs")
            for connection, saved in zip(config["connections"], step["transfers"], strict=True):
                a, b = connection["source"], connection["target"]
                ca, cb = descriptions[a]["model"]["column"], descriptions[b]["model"]["column"]
                sa, sb = state["vessels"][a]["column"], state["vessels"][b]["column"]
                fa, ga, sample_a, *_ = current[a]
                _fb, gb, sample_b, *_ = current[b]
                ba = np.r_[0, sa["interface_heights_m"], ga.height_m] + ca["elevation_m"]
                bb = np.r_[0, sb["interface_heights_m"], gb.height_m] + cb["elevation_m"]
                center = ca["elevation_m"] + connection["source_height_m"]
                radius = connection["diameter_m"] / 2
                close(
                    center, cb["elevation_m"] + connection["target_height_m"], atol=1e-9, rtol=0.0
                )
                mw = np.array([c.molar_mass_kg_mol for c in fa.components])
                g = ca["gravity_m_s2"]
                if g != cb["gravity_m_s2"]:
                    raise ValueError("Connector gravity differs")
                flux = np.zeros(len(fa.names))
                energy = entropy = volume = 0.0
                fractions = []
                probes = []
                for i in range(len(ba) - 1):
                    for j in range(len(bb) - 1):
                        low = max(ba[i], bb[j], center - radius)
                        high = min(ba[i + 1], bb[j + 1], center + radius)
                        fraction = 0.0
                        if high > low:
                            low_angle = math.asin(np.clip((low - center) / radius, -1, 1))
                            high_angle = math.asin(np.clip((high - center) / radius, -1, 1))
                            theta = (low_angle + high_angle) / 2 + (
                                high_angle - low_angle
                            ) / 2 * nodes
                            fraction = (
                                (high_angle - low_angle)
                                + (math.sin(2 * high_angle) - math.sin(2 * low_angle)) / 2
                            ) / math.pi
                            ww = weights * np.cos(theta) ** 2
                            ww = ww / ww.sum() * fraction
                            for z, area in zip(center + radius * np.sin(theta), ww):
                                pa, xa, qa = sample_a(z - ca["elevation_m"], i)
                                pb, xb, qb = sample_b(z - cb["elevation_m"], j)
                                probes.append((z, area, pa, xa, qa, pb, xb, qb))
                        fractions.append(
                            {"source_phase": i, "target_phase": j, "fraction": fraction}
                        )
                is_darcy = connection.get("hydraulic_model", "linear") == "darcy_weisbach"
                opening = expected_openings[connection["name"]]
                n_atol, e_atol, v_atol = 1e-12, 1e-7, 1e-14
                if is_darcy:
                    pressure_a = math.fsum(v[1] * v[2] for v in probes)
                    pressure_b = math.fsum(v[1] * v[5] for v in probes)
                    drive = pressure_a - pressure_b
                    forward = drive >= 0
                    density = math.fsum(
                        v[1] * (v[4]["rho"] if forward else v[7]["rho"]) for v in probes
                    )

                    def loss(velocity, density=density, connection=connection):
                        return reference_darcy(
                            velocity,
                            density,
                            connection["length_m"],
                            connection["diameter_m"],
                            connection["dynamic_viscosity_pa_s"],
                            connection["roughness_m"],
                        )

                    if not connection.get("allow_reverse", True):
                        drive = max(drive, 0.0)
                    if opening not in (0.0, 1.0):
                        raise ValueError("Darcy audit requires binary isolation")
                    cap = connection["maximum_velocity_m_s"]
                    velocity = (
                        brentq(
                            lambda v, loss=loss, drive=drive: loss(v)[0] - drive,
                            -cap,
                            cap,
                            xtol=1e-13,
                        )
                        if opening
                        else 0.0
                    )
                    friction, re = loss(velocity)
                    h = saved["hydraulics"]
                    # Propagate pressure-subtraction roundoff through the
                    # independently differentiated friction law, not a fitted
                    # relative allowance for near-zero flows.
                    pressure_roundoff = (
                        64 * np.finfo(float).eps * max(abs(pressure_a), abs(pressure_b))
                    )
                    delta = max(1e-8, abs(velocity) * 1e-5)
                    slope = (loss(velocity + delta)[0] - loss(velocity - delta)[0]) / (2 * delta)
                    if not math.isfinite(slope) or slope <= 0:
                        raise ValueError("Independent Darcy sensitivity is invalid")
                    dv = pressure_roundoff / slope if opening else 0.0
                    section = math.pi * connection["diameter_m"] ** 2 / 4
                    n_gain = sum(
                        (
                            area * section * np.maximum(xa / qa["vm"], xb / qb["vm"])
                            for z, area, pa, xa, qa, pb, xb, qb in probes
                        ),
                        np.zeros(len(fa.names)),
                    )
                    e_gain = math.fsum(
                        area
                        * section
                        * max(
                            abs(qa["h"] + float(xa @ mw) * g * z) / qa["vm"],
                            abs(qb["h"] + float(xb @ mw) * g * z) / qb["vm"],
                        )
                        for z, area, pa, xa, qa, pb, xb, qb in probes
                    )
                    n_atol = n_atol + dv * n_gain
                    e_atol += dv * e_gain
                    v_atol += dv * section
                    for key, value in {
                        "velocity": velocity,
                        "pressure_source": pressure_a,
                        "pressure_target": pressure_b,
                        "pressure_drop": pressure_a - pressure_b,
                        "friction_loss": friction,
                        "reynolds": re,
                    }.items():
                        extra = dv if key == "velocity" else 0.0
                        if key == "reynolds":
                            extra = (
                                dv
                                * max(
                                    math.fsum(v[1] * v[4]["rho"] for v in probes),
                                    math.fsum(v[1] * v[7]["rho"] for v in probes),
                                )
                                * connection["diameter_m"]
                                / connection["dynamic_viscosity_pa_s"]
                            )
                        close(h[key], value, atol=1e-7 + extra)
                    reported_loss, reported_re = reference_darcy(
                        h["velocity"],
                        h["density"],
                        connection["length_m"],
                        connection["diameter_m"],
                        connection["dynamic_viscosity_pa_s"],
                        connection["roughness_m"],
                    )
                    close(h["friction_loss"], reported_loss, atol=1e-9, rtol=1e-9)
                    close(h["reynolds"], reported_re, atol=1e-10, rtol=1e-9)
                    # A shut/reverse-checked face has zero velocity; donor density
                    # then follows the runtime's documented nonnegative branch.
                    if velocity >= 0:
                        density = math.fsum(v[1] * v[4]["rho"] for v in probes)
                    if opening and abs(velocity) <= dv:
                        rho_a = math.fsum(v[1] * v[4]["rho"] for v in probes)
                        rho_b = math.fsum(v[1] * v[7]["rho"] for v in probes)
                        if min(
                            abs(h["density"] - rho_a), abs(h["density"] - rho_b)
                        ) > 1e-7 + 2e-6 * max(rho_a, rho_b):
                            raise ValueError("Independent near-zero donor density rejected")
                    else:
                        close(h["density"], density, atol=1e-7)
                for z, area, pa, xa, qa, pb, xb, qb in probes:
                    dp = pa - pb
                    if is_darcy:
                        signed = (
                            velocity * math.pi * connection["diameter_m"] ** 2 / 4 * area * opening
                        )
                        positive = velocity >= 0
                    else:
                        signed = (
                            connection["volume_conductance_m3_s_pa"]
                            * opening
                            * area
                            * (dp if connection.get("allow_reverse", True) else max(dp, 0.0))
                        )
                        positive = dp >= 0
                    x, q, donor = (xa, qa, pa) if positive else (xb, qb, pb)
                    if (
                        signed
                        and abs(dp) / donor
                        > connection.get("maximum_pressure_drop_fraction", 0.05) + 1e-9
                    ):
                        raise ValueError("Independent connector pressure domain rejected")
                    rate = signed / q["vm"]
                    flux += rate * x
                    energy += rate * (q["h"] + float(x @ mw) * g * z)
                    entropy += rate * q["s"]
                    volume += signed
                if saved["name"] != connection["name"] or len(fractions) != len(
                    saved["phase_pair_areas"]
                ):
                    raise ValueError("Network connector/phase-pair registry differs")
                for actual, expected in zip(saved["phase_pair_areas"], fractions):
                    if (
                        actual["source_phase"] != expected["source_phase"]
                        or actual["target_phase"] != expected["target_phase"]
                    ):
                        raise ValueError("Network phase pairing differs")
                    close(actual["fraction"], expected["fraction"], atol=1e-8)
                close(sum(v["fraction"] for v in fractions), 1.0, atol=1e-8)
                close(saved["component_rates_mol_s"], flux, atol=n_atol)
                close(saved["total_energy_rate_w"], energy, atol=e_atol)
                close(saved["volume_rate_m3_s"], volume, atol=v_atol)
                nnet[a] -= flux
                nnet[b] += flux
                enet[a] -= energy
                enet[b] += energy
                snet[a] -= entropy
                snet[b] += entropy
            production = error = ambient_total = 0.0
            nerror = np.zeros_like(nnet[names[0]])
            for name in names:
                model = descriptions[name]["model"]
                wall = model["wall"]
                before = old["vessels"][name]
                after = state["vessels"][name]
                col0, col1 = before["column"], after["column"]
                tw0, tw = before["wall_temperature_k"], after["wall_temperature_k"]
                cw = wall["heat_capacity_j_k"]
                tamb = wall.get("ambient_temperature_k", 298.15)
                heat = wall["fluid_ua_w_k"] * (tw - col1["temperature_k"])
                ambient = wall.get("ambient_ua_w_k", 0.0) * (tamb - tw)
                delta_n = np.array(col1["component_moles"]) - col0["component_moles"]
                close(delta_n, dt * nnet[name], atol=1e-8)
                close(
                    col1["total_energy_j"] - col0["total_energy_j"],
                    dt * (enet[name] + heat),
                    atol=1e-6,
                )
                close(cw * (tw - tw0), dt * (ambient - heat), atol=1e-6)
                local_production = (
                    current[name][-1]
                    - previous[name][-1]
                    + cw * math.log(tw / tw0)
                    - dt * (snet[name] + ambient / tamb)
                )
                if local_production < -2e-6 - 1e-9 * max(
                    1, abs(current[name][-1]), abs(previous[name][-1])
                ):
                    raise ValueError("Independent vessel entropy balance rejected")
                production += local_production
                nerror += delta_n - dt * nnet[name]
                error += (
                    col1["total_energy_j"]
                    - col0["total_energy_j"]
                    + cw * (tw - tw0)
                    - dt * (enet[name] + ambient)
                )
                ambient_total += dt * ambient
                close(after["time_s"], time + dt, atol=1e-10, rtol=1e-12)
            close(step["component_boundary_moles"], np.zeros_like(nerror), atol=1e-10)
            close(step["component_residuals_mol"], nerror, atol=1e-8, rtol=0.0)
            close(step["energy_residual_j"], error, atol=2e-6)
            close(step["entropy_production_j_k"], production, atol=2e-6)
            close(step["ambient_heat_j"], ambient_total, atol=1e-7)
            close(step["feed_energy_j"], 0.0)
            close(step["outlet_energy_j"], 0.0)
            maximum_error = max(maximum_error, abs(error))
            ambient_interval += ambient_total
            production_interval += production
            old = state
            previous = current
            time += dt
            checked += 1
            close(state["time_s"], time, atol=1e-10, rtol=1e-12)
        close(time, target, atol=1e-10, rtol=1e-12)
        close(interval["state"]["time_s"], target, atol=1e-10, rtol=1e-12)
        for name in names:
            child = interval["state"]["vessels"][name]
            if child["column"] != old["vessels"][name]["column"]:
                raise ValueError("Interval final network profile differs")
            close(
                child["wall_temperature_k"], old["vessels"][name]["wall_temperature_k"], atol=1e-8
            )
        final_energy = math.fsum(
            old["vessels"][n]["column"]["total_energy_j"]
            + descriptions[n]["model"]["wall"]["heat_capacity_j_k"]
            * (
                old["vessels"][n]["wall_temperature_k"]
                - descriptions[n]["model"]["wall"].get("reference_temperature_k", 298.15)
            )
            for n in names
        )
        close(
            interval["energy_residual_j"],
            final_energy - initial_energy - ambient_interval,
            atol=2e-6,
        )
        close(interval["ambient_heat_j"], ambient_interval, atol=1e-7)
        close(interval["entropy_production_j_k"], production_interval, atol=2e-6)
        close(interval["feed_energy_j"], 0.0)
        close(interval["outlet_energy_j"], 0.0)
        total_n = sum(
            (
                np.array(old["vessels"][n]["column"]["component_moles"]) - interval_n[n]
                for n in names
            ),
            np.zeros_like(nnet[names[0]]),
        )
        close(interval["component_residuals_mol"], total_n, atol=1e-8)
        old = interval["state"]
    return {
        "accepted": True,
        "checked_substeps": checked,
        "reference_connector_quadrature": 32,
        "maximum_energy_residual_j": maximum_error,
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
