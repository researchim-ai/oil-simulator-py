"""Independent NumPy PR PS/PH/check-valve audit of true-flow pumps."""

import numpy as np
from check_stratified_pr import Reference
from check_stratified_pr_flow import close
from reference_pr_pumps import curve


def audit_pump_connections(model, data, openings, saved, fluid):
    units = model.get("pump_connections", [])
    if set(saved) != {u["card"]["name"] for u in units}:
        raise ValueError("Pump connection registry differs")
    net, powers = {}, {}
    masses = np.array([c.molar_mass_kg_mol for c in fluid.components])
    nodes = {n["name"]: n for n in model["nodes"]}
    for unit in units:
        card = unit["card"]
        name, a, b = card["name"], card["start"], card["end"]
        row, source, target = saved[name], data[a], data[b]
        if row["enabled"] is not openings[name] or not isinstance(row["valve_closed"], bool):
            raise ValueError("Pump connection mode differs")
        q, p = row["flow_m3_s"], row["pump_pressure_pa"]
        if not openings[name]:
            for key in (
                "flow_m3_s",
                "mass_rate_kg_s",
                "shaft_power_w",
                "hydraulic_power_w",
                "head_m",
                "entropy_production_w_k",
                "valve_pressure_drop_pa",
                "npsh_available_m",
            ):
                close(row[key], 0.0, atol=1e-12, rtol=0.0)
            close(p, source["p"], atol=1e-6, rtol=1e-9)
            close(row["isentropic_temperature_k"], source["t"], atol=1e-7, rtol=1e-9)
            close(row["pump_temperature_k"], source["t"], atol=1e-7, rtol=1e-9)
            close(row["delivery_temperature_k"], target["t"], atol=1e-7, rtol=1e-9)
            if not row["valve_closed"]:
                raise ValueError("Disabled pump must be isolated")
            if unit.get("signed_map") is not None:
                for key in ("load_torque_nm", "dissipation_w", "mapped_speed_rad_s"):
                    close(row[key], 0.0, atol=1e-12, rtol=0.0)
            powers[name] = 0.0
            continue
        if unit.get("signed_map") is not None:
            local, power = audit_signed_connection(model, unit, row, source, target, fluid)
            for node, flux in local.items():
                net[node] = net.get(node, 0.0) + flux
            powers[name] = power
            continue
        speed, closed = card["speed_ratio"], row["valve_closed"]
        if q < 0 or q > speed * card["max_flow_m3_s"] * (1 + 1e-10):
            raise ValueError("Pump true flow outside card")
        reduced = q / max(speed, 1e-12)
        head = speed**2 * card["shutoff_head_m"] - card["curve_coefficient_s2_m5"] * q * q
        eta = card["efficiency"]
        if card["head_curve"] is not None:
            head = speed**2 * curve(card["head_curve"], reduced)
        if card["efficiency_curve"] is not None:
            eta = curve(card["efficiency_curve"], reduced)
        if head < 0 or not 0 < eta <= 1:
            raise ValueError("Pump head or efficiency outside envelope")
        for table in (card["head_curve"], card["efficiency_curve"], card["npsh_curve"]):
            if (
                table is not None
                and not closed
                and not table["flow_m3_s"][0] <= reduced <= table["flow_m3_s"][-1]
            ):
                raise ValueError("Pump table flow outside envelope")
        x = source["x"][0]
        ref = Reference(fluid, np.flatnonzero(x > 0))
        qin = ref.state(source["p"], source["t"], x, "stable")
        qis = ref.state(p, row["isentropic_temperature_k"], x, "liquid")
        qp = ref.state(p, row["pump_temperature_k"], x, "liquid")
        qd = ref.state(target["p"], row["delivery_temperature_k"], x, "liquid")
        mw = x @ masses
        close(qis["h"] - qin["h"], mw * 9.80665 * head, atol=1e-5, rtol=1e-8)
        close(qis["s"], qin["s"], atol=1e-7, rtol=1e-9)
        pressure_gap = target["p"] + unit["outlet_loss_pa_s2_m6"] * q * q - p
        if closed:
            close(q, 0.0, atol=1e-12, rtol=0.0)
            if not card["check_valve"] or pressure_gap < -1e-4:
                raise ValueError("Closed pump check-valve inequality fails")
            close(row["pump_temperature_k"], row["isentropic_temperature_k"], atol=1e-7, rtol=1e-9)
            close(row["delivery_temperature_k"], target["t"], atol=1e-7, rtol=1e-9)
        else:
            close(pressure_gap, 0.0, atol=1e-4, rtol=0.0)
            close(qp["h"] - qin["h"], (qis["h"] - qin["h"]) / eta, atol=1e-5, rtol=1e-8)
            close(qd["h"], qp["h"], atol=1e-5, rtol=1e-9)
        if (
            len(source["pn"]) != 1
            or min(qin["rho"], qis["rho"]) < unit["minimum_density_kg_m3"]
            or (not closed and min(qp["rho"], qd["rho"]) < unit["minimum_density_kg_m3"])
        ):
            raise ValueError("Pump leaves dense single-phase envelope")
        if 9.80665 * head / source["sound"] ** 2 > unit["maximum_compressibility_fraction"]:
            raise ValueError("Pump compressibility envelope exceeded")
        mass = q * source["rho"]
        molar = mass / mw
        power, hydraulic = molar * (qd["h"] - qin["h"]), molar * (qis["h"] - qin["h"])
        entropy = molar * (qd["s"] - qin["s"])
        if (
            power < -1e-8
            or entropy < -1e-8
            or (card["max_shaft_power_w"] is not None and power > card["max_shaft_power_w"])
        ):
            raise ValueError("Pump power/entropy envelope exceeded")
        npsh = 0.0
        if unit["vapor_pressure_pa"] is not None:
            npsh = (source["p"] - unit["vapor_pressure_pa"]) / (source["rho"] * 9.80665)
            required = (card["npsh_required_m"] or 0.0) * speed**2
            if card["npsh_curve"] is not None:
                required = curve(card["npsh_curve"], reduced) * speed**2
            if not closed and (
                npsh < required
                or abs(source["t"] - unit["vapor_pressure_temperature_k"])
                > unit["maximum_vapor_pressure_temperature_departure_k"]
            ):
                raise ValueError("Pump NPSH/temperature envelope exceeded")
        expected = {
            "flow_m3_s": q,
            "mass_rate_kg_s": mass,
            "head_m": head,
            "efficiency": eta,
            "shaft_power_w": power,
            "hydraulic_power_w": hydraulic,
            "pump_pressure_pa": p,
            "isentropic_temperature_k": row["isentropic_temperature_k"],
            "pump_temperature_k": row["pump_temperature_k"],
            "delivery_temperature_k": row["delivery_temperature_k"],
            "entropy_production_w_k": entropy,
            "valve_pressure_drop_pa": 0.0 if closed else p - target["p"],
            "valve_closed": closed,
            "npsh_available_m": npsh,
            "enabled": True,
        }
        if set(row) != set(expected):
            raise ValueError("Pump report schema differs")
        for key, value in expected.items():
            if not isinstance(value, bool):
                close(row[key], value, atol=1e-8, rtol=2e-7)
        phi = model["gravity_m_s2"] * (nodes[a]["elevation_m"] - model["reference_elevation_m"])
        net[a] = net.get(a, 0.0) - np.r_[molar * x, 0.0, molar * (qin["h"] + mw * phi)]
        net[b] = net.get(b, 0.0) + np.r_[molar * x, 0.0, molar * (qd["h"] + mw * phi)]
        powers[name] = power
    return net, powers


def audit_signed_connection(model, unit, row, left, right, fluid):
    """Independent NumPy constitutive and PR port ledger reconstruction."""
    m, card = unit["signed_map"], unit["card"]
    q, omega = row["flow_m3_s"], row["mapped_speed_rad_s"]
    if card["check_valve"] or row["valve_closed"]:
        raise ValueError("Signed pump cannot have a check valve")
    if (
        not np.isfinite(q + omega)
        or abs(q) > m["maximum_flow_m3_s"] * (1 + 1e-10)
        or abs(omega) > m["maximum_speed_rad_s"] * (1 + 1e-10)
    ):
        raise ValueError("Signed pump map envelope exceeded")
    close(
        omega,
        unit.get("mapped_speed_rad_s")
        if unit.get("mapped_speed_rad_s") is not None
        else card["speed_ratio"] * m["reference_speed_rad_s"],
        atol=1e-10,
        rtol=1e-10,
    )
    source, target = (left, right) if q >= 0 else (right, left)
    sign = 1.0 if q >= 0 else -1.0
    x = source["x"][0]
    masses = np.array([c.molar_mass_kg_mol for c in fluid.components])
    mw, rho, g = x @ masses, source["rho"], 9.80665
    coupling = m["euler_head_coefficient_m_s2"] * abs(omega) + m.get(
        "linear_head_coefficient_m_s", 0.0
    )
    head = coupling * omega - m["flow_loss_coefficient_s2_m5"] * q * abs(q)
    torque = rho * g * (coupling * q + m.get("viscous_torque_coefficient_m4_s", 0.0) * omega)
    shaft, hydraulic = torque * omega, rho * g * head * q
    heat = rho * g * m.get("viscous_torque_coefficient_m4_s", 0.0) * omega**2
    loss = heat + rho * g * m["flow_loss_coefficient_s2_m5"] * abs(q) ** 3
    ref = Reference(fluid, np.flatnonzero(x > 0))
    p = row["pump_pressure_pa"]
    qi = ref.state(source["p"], source["t"], x, "stable")
    qis = ref.state(p, row["isentropic_temperature_k"], x, "liquid")
    qp = ref.state(p, row["pump_temperature_k"], x, "liquid")
    qd = ref.state(target["p"], row["delivery_temperature_k"], x, "liquid")
    nodes = {n["name"]: n for n in model["nodes"]}
    a, b = card["start"], card["end"]
    phi_a, phi_b = [
        model["gravity_m_s2"] * (nodes[n]["elevation_m"] - model["reference_elevation_m"])
        for n in (a, b)
    ]
    dphi = sign * (phi_b - phi_a)
    close(qis["h"] - qi["h"], mw * (g * sign * head - dphi), atol=1e-5, rtol=1e-8)
    close(qis["s"], qi["s"], atol=1e-7, rtol=1e-9)
    close(qp["h"] - qi["h"], mw * (g * sign * coupling * omega - dphi), atol=1e-5, rtol=1e-8)
    close(qd["h"], qp["h"], atol=1e-5, rtol=1e-9)
    close(p, target["p"] + unit["outlet_loss_pa_s2_m6"] * q * q, atol=1e-4, rtol=1e-9)
    molar = q * rho / mw
    entropy = abs(molar) * (qd["s"] - qi["s"]) + heat / source["t"]
    from check_pr_equilibrium_transport import reference_sound

    sound = reference_sound(fluid, source["p"], source["t"], x, "liquid")
    if (
        len(source["pn"]) != 1
        or min(s["rho"] for s in (qi, qis, qp, qd)) < unit["minimum_density_kg_m3"]
        or g * abs(head) / sound**2 > unit["maximum_compressibility_fraction"]
        or entropy < -1e-8
    ):
        raise ValueError("Signed pump thermodynamic envelope exceeded")
    if card["max_shaft_power_w"] is not None and abs(shaft) > card["max_shaft_power_w"]:
        raise ValueError("Signed pump shaft rating exceeded")
    npsh = 0.0
    if unit["vapor_pressure_pa"] is not None:
        npsh = (source["p"] - unit["vapor_pressure_pa"]) / (rho * g)
        if (
            npsh < 0
            or abs(source["t"] - unit["vapor_pressure_temperature_k"])
            > unit["maximum_vapor_pressure_temperature_departure_k"]
        ):
            raise ValueError("Signed pump vapor pressure envelope exceeded")
    expected = {
        "flow_m3_s": q,
        "mass_rate_kg_s": q * rho,
        "head_m": head,
        "efficiency": 1.0,
        "shaft_power_w": shaft,
        "hydraulic_power_w": hydraulic,
        "pump_pressure_pa": p,
        "isentropic_temperature_k": row["isentropic_temperature_k"],
        "pump_temperature_k": row["pump_temperature_k"],
        "delivery_temperature_k": row["delivery_temperature_k"],
        "entropy_production_w_k": entropy,
        "valve_pressure_drop_pa": sign * (p - target["p"]),
        "valve_closed": False,
        "npsh_available_m": npsh,
        "load_torque_nm": torque,
        "dissipation_w": loss,
        "mapped_speed_rad_s": omega,
        "enabled": True,
    }
    if set(row) != set(expected):
        raise ValueError("Signed pump report schema differs")
    for key, value in expected.items():
        if not isinstance(value, bool):
            close(row[key], value, atol=1e-8, rtol=2e-7)
    ha, hb = (qi["h"], qd["h"]) if q >= 0 else (qd["h"], qi["h"])
    net = {
        a: -np.r_[molar * x, 0.0, molar * (ha + mw * phi_a)],
        b: np.r_[molar * x, 0.0, molar * (hb + mw * phi_b)],
    }
    net[a if q >= 0 else b][-1] += heat
    close(sum(v[-1] for v in net.values()), shaft, atol=1e-6, rtol=2e-7)
    return net, shaft
