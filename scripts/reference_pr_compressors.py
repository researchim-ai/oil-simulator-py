"""NumPy compressor-card and PR PS/PH audit; no runtime compressor calls."""

import math

import numpy as np
from check_stratified_pr import Reference
from check_stratified_pr_flow import close


def audit_compressors(model, data, openings, saved, fluid):
    units = model.get("compressors", [])
    if set(saved) != {unit["name"] for unit in units}:
        raise ValueError("Compressor registry differs")
    net, power_total = {}, 0.0
    masses = np.array([c.molar_mass_kg_mol for c in fluid.components])
    nodes = {node["name"]: node for node in model["nodes"]}
    for unit in units:
        if unit.get("signed_map") is not None:
            flows, power = audit_signed_compressor(
                model, unit, saved[unit["name"]], data, openings[unit["name"]], fluid
            )
            for name, flux in flows.items():
                net[name] = net.get(name, 0) + flux
            power_total += power
            continue
        point, card = saved[unit["name"]], unit["card"]
        a, b = unit["source"], unit["target"]
        inlet, receiver = data[a], data[b]
        if point["enabled"] is not openings[unit["name"]]:
            raise ValueError("Compressor isolation differs")
        if not openings[unit["name"]]:
            for key in ("beta", "mass_rate_kg_s", "shaft_power_w", "entropy_production_w_k"):
                close(point[key], 0.0, atol=1e-9, rtol=0)
            close(point["isentropic_temperature_k"], inlet["t"], atol=1e-7, rtol=1e-9)
            close(point["delivery_temperature_k"], inlet["t"], atol=1e-7, rtol=1e-9)
            close(point["pressure_ratio"], 1.0, atol=1e-12, rtol=0)
            close(point["efficiency"], 1.0, atol=1e-12, rtol=0)
            if point["map_choked"] is not False:
                raise ValueError("Isolated compressor cannot be choke-limited")
            continue
        theta = inlet["t"] / card["reference_temperature_k"]
        speed = unit["speed_rad_s"] / math.sqrt(theta)
        speeds, betas = np.asarray(card["corrected_speed_rad_s"]), np.asarray(card["beta"])
        beta = point["beta"]
        if not speeds[0] <= speed <= speeds[-1] or not 0 <= beta <= 1 or len(inlet["pn"]) != 1:
            raise ValueError("Compressor speed/beta/inlet-phase envelope exceeded")

        def table(key, beta_value, betas=betas, card=card, speed=speed, speeds=speeds):
            rows = [np.interp(beta_value, betas, row) for row in card[key]]
            return float(np.interp(speed, speeds, rows))

        ratio, eta = table("pressure_ratio", beta), table("isentropic_efficiency", beta)
        surge, choke = table("pressure_ratio", 0.0), table("pressure_ratio", 1.0)
        requested = receiver["p"] / inlet["p"]
        choked = requested < choke
        if requested > surge or not 0 < eta <= 1 or ratio < 1:
            raise ValueError("Compressor pressure/efficiency envelope exceeded")
        if point["map_choked"] is not choked:
            raise ValueError("Compressor choke branch differs")
        close(beta if choked else requested, 1.0 if choked else ratio, atol=1e-9, rtol=1e-9)
        x = inlet["x"][0]
        if abs(x - unit["reference_mole_fractions"]).sum() > unit["maximum_composition_departure"]:
            raise ValueError("Compressor reference gas composition exceeded")
        ref = Reference(fluid, np.flatnonzero(x > 0))
        qin = ref.state(inlet["p"], inlet["t"], x, "stable")
        qis = ref.state(inlet["p"] * ratio, point["isentropic_temperature_k"], x, "vapor")
        qout = ref.state(receiver["p"], point["delivery_temperature_k"], x, "vapor")
        if min(q["z"] for q in (qin, qis, qout)) < unit["minimum_compressibility_factor"]:
            raise ValueError("Dry gas compressor trace outside envelope")
        close(qis["s"], qin["s"], atol=1e-6, rtol=1e-9)
        close(qout["h"] - qin["h"], (qis["h"] - qin["h"]) / eta, atol=1e-5, rtol=1e-9)
        mass_rate = (
            table("corrected_mass_flow_kg_s", beta)
            * inlet["p"]
            / card["reference_pressure_pa"]
            / math.sqrt(theta)
        )
        mw = x @ masses
        molar = mass_rate / mw
        power, entropy = molar * (qout["h"] - qin["h"]), molar * (qout["s"] - qin["s"])
        if (
            power < -1e-8
            or entropy < -1e-8
            or (unit["maximum_shaft_power_w"] is not None and power > unit["maximum_shaft_power_w"])
        ):
            raise ValueError("Compressor work/entropy envelope exceeded")
        for key, value in {
            "mass_rate_kg_s": mass_rate,
            "shaft_power_w": power,
            "entropy_production_w_k": entropy,
            "pressure_ratio": ratio,
            "efficiency": eta,
        }.items():
            close(point[key], value, atol=1e-8, rtol=2e-7)
        phi = model["gravity_m_s2"] * (nodes[a]["elevation_m"] - model["reference_elevation_m"])
        nf = molar * x
        first = np.r_[nf, 0.0, molar * (qin["h"] + mw * phi)]
        last = np.r_[nf, 0.0, molar * (qout["h"] + mw * phi)]
        net[a] = net.get(a, 0) - first
        net[b] = net.get(b, 0) + last
        power_total += power
    return net, power_total


def audit_signed_compressor(model, unit, point, data, enabled, fluid):
    """Independent scalar/NumPy Euler, donor PS/PH, and physical-end audit."""
    card = unit["signed_map"]
    a, b = unit["source"], unit["target"]
    if point["enabled"] is not enabled:
        raise ValueError("Signed compressor isolation differs")
    if not enabled:
        for key in (
            "volume_rate_m3_s",
            "mass_rate_kg_s",
            "shaft_power_w",
            "load_torque_nm",
            "entropy_production_w_k",
        ):
            close(point[key], 0.0, atol=1e-9, rtol=0)
        return {}, 0.0
    flow = point["volume_rate_m3_s"]
    speed = point["speed_rad_s"]
    close(speed, unit["speed_rad_s"], atol=1e-10, rtol=1e-10)
    if not (abs(flow) <= card["maximum_flow_m3_s"] and abs(speed) <= card["maximum_speed_rad_s"]):
        raise ValueError("Signed compressor map envelope exceeded")
    sign = 1 if flow >= 0 else -1
    donor, receiver = (data[a], data[b]) if sign > 0 else (data[b], data[a])
    x = donor["x"][0]
    if len(donor["pn"]) != 1 or abs(x - unit["reference_mole_fractions"]).sum() > unit.get(
        "maximum_composition_departure", 0.01
    ):
        raise ValueError("Signed compressor donor/reference gas invalid")
    ref = Reference(fluid, np.flatnonzero(x > 0))
    qin = ref.state(donor["p"], donor["t"], x, "stable")
    mw = x @ np.array([c.molar_mass_kg_mol for c in fluid.components])
    # Reference density follows the independently evaluated PR molar volume.
    rho = qin["rho"]
    coupling = card["euler_coefficient"] * abs(speed) + card["linear_coupling"]
    head = coupling * speed - card["flow_loss_coefficient"] * flow * abs(flow)
    torque = rho * coupling * flow
    power = torque * speed
    stage = receiver["p"]
    close(point["stage_pressure_pa"], stage, atol=1e-5, rtol=1e-9)
    qis = ref.state(stage, point["isentropic_temperature_k"], x, "vapor")
    qout = ref.state(receiver["p"], point["delivery_temperature_k"], x, "vapor")
    for p, t, q in (
        (donor["p"], donor["t"], qin),
        (stage, point["isentropic_temperature_k"], qis),
        (receiver["p"], point["delivery_temperature_k"], qout),
    ):
        stable = ref.state(p, t, x, "stable")
        close(q["z"], stable["z"], atol=1e-8, rtol=1e-8)
        if q["z"] < unit.get("minimum_compressibility_factor", 0.5):
            raise ValueError("Signed compressor trace outside dry gas envelope")
    close(qis["s"], qin["s"], atol=1e-6, rtol=1e-9)
    close(qis["h"] - qin["h"], mw * sign * head, atol=1e-5, rtol=1e-9)
    close(qout["h"] - qin["h"], mw * sign * coupling * speed, atol=1e-5, rtol=1e-9)
    molar = rho * flow / mw
    entropy = abs(molar) * (qout["s"] - qin["s"])
    if entropy < -1e-8 or (
        unit.get("maximum_shaft_power_w") is not None and abs(power) > unit["maximum_shaft_power_w"]
    ):
        raise ValueError("Signed compressor entropy/power exceeded")
    for key, value in {
        "mass_rate_kg_s": rho * flow,
        "shaft_power_w": power,
        "load_torque_nm": torque,
        "entropy_production_w_k": entropy,
        "isentropic_head_j_kg": head,
        "dissipation_w": rho * card["flow_loss_coefficient"] * abs(flow) ** 3,
    }.items():
        close(point[key], value, atol=1e-8, rtol=2e-7)
    nodes = {n["name"]: n for n in model["nodes"]}

    def flux(name, h):
        phi = model["gravity_m_s2"] * (nodes[name]["elevation_m"] - model["reference_elevation_m"])
        return np.r_[molar * x, 0.0, molar * (h + mw * phi)]

    ha, hb = (qin["h"], qout["h"]) if sign > 0 else (qout["h"], qin["h"])
    return {a: -flux(a, ha), b: flux(b, hb)}, power
