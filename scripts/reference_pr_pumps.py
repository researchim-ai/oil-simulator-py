"""Independent scalar pump-card audit; no Torch or runtime card evaluations."""

import math

import numpy as np


def curve(table, query):
    x, y = np.asarray(table["flow_m3_s"]), np.asarray(table["values"])
    i = min(max(int(np.searchsorted(x, query, side="right")) - 1, 0), len(x) - 2)
    return y[i] + (y[i + 1] - y[i]) * (query - x[i]) / (x[i + 1] - x[i])


def signed_pump_values(mapping, q, omega, rho):
    """Independent scalar signed constitutive calculation."""
    for key in (
        "euler_head_coefficient_m_s2",
        "flow_loss_coefficient_s2_m5",
        "linear_head_coefficient_m_s",
        "viscous_torque_coefficient_m4_s",
    ):
        value = mapping.get(key, 0.0)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid signed pump dissipative coefficient")
    coupling = mapping["euler_head_coefficient_m_s2"] * abs(omega) + mapping.get(
        "linear_head_coefficient_m_s", 0.0
    )
    head = coupling * omega - mapping["flow_loss_coefficient_s2_m5"] * q * abs(q)
    torque = (
        rho * 9.80665 * (coupling * q + mapping.get("viscous_torque_coefficient_m4_s", 0.0) * omega)
    )
    loss = (
        rho
        * 9.80665
        * (
            mapping.get("viscous_torque_coefficient_m4_s", 0.0) * omega**2
            + mapping["flow_loss_coefficient_s2_m5"] * abs(q) ** 3
        )
    )
    return head, torque, torque * omega, rho * 9.80665 * head * q, loss


def audit_pumps(model, data, openings, saved):
    units = model.get("pumps", [])
    if set(saved) != {unit["card"]["name"] for unit in units}:
        raise ValueError("Pump registry differs")
    nodes = {node["name"]: node for node in model["nodes"]}
    forces, powers = {}, {}
    for unit in units:
        card, node = unit["card"], nodes[unit["node"]]
        name = card["name"]
        body, inlet = data[unit["node"]], data[card["start"]]
        area = math.pi * node["diameter_m"] ** 2 / 4
        q, rho = area * body["u"], body["rho"]
        head = power = hydraulic = npsh = compressibility = 0.0
        eta = card["efficiency"]
        if not openings[name]:
            force = (
                -0.5 * area * unit["stopped_loss_coefficient"] * rho * body["u"] * abs(body["u"])
            )
        elif unit.get("signed_map") is not None:
            mapping = unit["signed_map"]
            omega = unit.get("mapped_speed_rad_s")
            if omega is None:
                omega = mapping["reference_speed_rad_s"] * card["speed_ratio"]
            if abs(q) > mapping["maximum_flow_m3_s"] or abs(omega) > mapping["maximum_speed_rad_s"]:
                raise ValueError("Signed pump map envelope exceeded")
            for port in (unit["node"], card["start"], card["end"]):
                point = data[port]
                if len(point["pn"]) != 1 or point["rho"] < unit["minimum_density_kg_m3"]:
                    raise ValueError("Signed pump dense liquid domain exceeded")
                if unit["vapor_pressure_pa"] is not None and (
                    point["p"] < unit["vapor_pressure_pa"]
                    or abs(point["t"] - unit["vapor_pressure_temperature_k"])
                    > unit["maximum_vapor_pressure_temperature_departure_k"]
                ):
                    raise ValueError("Signed pump vapor pressure domain exceeded")
            head, torque, power, hydraulic, loss = signed_pump_values(mapping, q, omega, rho)
            compressibility = 9.80665 * abs(head) / body["sound"] ** 2
            if compressibility > unit["maximum_compressibility_fraction"]:
                raise ValueError("Signed pump compressibility exceeded")
            if card["max_shaft_power_w"] is not None and abs(power) > card["max_shaft_power_w"]:
                raise ValueError("Signed pump power exceeded")
            force = area * rho * 9.80665 * head
        else:
            speed = card["speed_ratio"]
            if (
                len(body["pn"]) != 1
                or len(inlet["pn"]) != 1
                or min(rho, inlet["rho"]) < unit["minimum_density_kg_m3"]
                or q < -64 * np.finfo(float).eps * card["max_flow_m3_s"]
                or q > speed * card["max_flow_m3_s"]
            ):
                raise ValueError("Pump liquid/flow operating envelope exceeded")
            if card["head_curve"] is None:
                head = speed**2 * card["shutoff_head_m"] - card[
                    "curve_coefficient_s2_m5"
                ] * q * abs(q)
            else:
                h = curve(card["head_curve"], abs(q) / speed)
                head = speed**2 * (
                    card["shutoff_head_m"] - math.copysign(1, q) * (card["shutoff_head_m"] - h)
                )
            if card["efficiency_curve"] is not None:
                eta = curve(card["efficiency_curve"], abs(q) / speed)
            if head < 0 or not 0 < eta <= 1:
                raise ValueError("Invalid accepted pump head/efficiency")
            compressibility = 9.80665 * head / body["sound"] ** 2
            if compressibility > unit["maximum_compressibility_fraction"]:
                raise ValueError("Liquid pump compressibility envelope exceeded")
            force = area * rho * 9.80665 * head
            hydraulic = rho * 9.80665 * head * q
            power = hydraulic / eta
            if card["max_shaft_power_w"] is not None and power > card["max_shaft_power_w"]:
                raise ValueError("Pump shaft power exceeded")
            if unit["vapor_pressure_pa"] is not None:
                if (
                    abs(inlet["t"] - unit["vapor_pressure_temperature_k"])
                    > unit["maximum_vapor_pressure_temperature_departure_k"]
                ):
                    raise ValueError("Vapor pressure temperature envelope exceeded")
                npsh = (inlet["p"] - unit["vapor_pressure_pa"]) / (inlet["rho"] * 9.80665) + (
                    inlet["u"]
                ) ** 2 / (2 * 9.80665)
                required = (card["npsh_required_m"] or 0) * speed**2
                if card["npsh_curve"] is not None:
                    required = curve(card["npsh_curve"], abs(q) / speed) * speed**2
                if npsh < required:
                    raise ValueError("Pump NPSH envelope exceeded")
        expected = {
            "enabled": openings[name],
            "flow_m3_s": q,
            "head_m": head,
            "efficiency": eta,
            "shaft_power_w": power,
            "hydraulic_power_w": hydraulic,
            "force_n": force,
            "npsh_available_m": npsh,
            "compressibility_fraction": compressibility,
        }
        if openings[name] and unit.get("signed_map") is not None:
            del expected["efficiency"], expected["npsh_available_m"]
            expected.update(load_torque_nm=torque, dissipation_w=loss)
        if set(saved[name]) != set(expected) or saved[name]["enabled"] is not openings[name]:
            raise ValueError("Pump report schema/mode differs")
        for key, value in expected.items():
            if key != "enabled" and not math.isclose(
                saved[name][key], value, rel_tol=2e-7, abs_tol=1e-9
            ):
                raise ValueError(f"Independent pump {key} differs")
        forces[unit["node"]], powers[unit["node"]] = force, power
    return forces, powers
