"""Independent population-moment, surface-energy and number-flux accounting."""

import math

import numpy as np
from check_stratified_pr_flow import close


def identify_roles(law, spec, x, z, rho):
    liquid = x[:, [law["oil_component_index"], law["water_component_index"]]]
    roles = liquid.argmax(-1) + 1
    i = int(np.argmin(rho))
    if z[i] >= spec["gas_compressibility_threshold"] and (
        len(rho) == 1 or rho[i] / max(rho) <= law["maximum_gas_liquid_density_ratio"]
    ):
        roles[i] = 0
    if len(rho) > 1 and (
        len(set(roles)) != len(roles)
        or np.any((roles != 0) & (liquid.max(-1) < law["minimum_marker_fraction"]))
    ):
        raise ValueError("Unresolved EOS/liquid marker phase roles")
    return roles


def population_properties(node, state, x, pn, vm, z, rho):
    spec = node.get("droplet_population")
    if spec is None:
        if state.get("droplet_number") is not None:
            raise ValueError("Unconfigured droplet stock")
        return None, None, 0.0
    law = dict(node["drift_flux"])
    if "droplet_number" not in state:
        raise ValueError("Missing droplet inventory")
    number = state["droplet_number"]
    roles = identify_roles(law, spec, x, z, rho)
    chosen = 1 if law["dispersed_liquid"] == "oil" else 2
    carrier = 2 if chosen == 1 else 1
    mask = roles == chosen
    present = bool(np.any(mask) and np.any(roles == carrier))
    volume = float(np.sum(pn * vm * mask)) if present else 0.0
    if (
        not np.isfinite(number)
        or number < 0
        or number > spec["maximum_number_density_m3"] * node["volume_m3"]
    ):
        raise ValueError("Invalid particle number")
    if present:
        if number <= 0:
            raise ValueError("Dispersed phase has no particles")
        diameter = (6 * volume / (math.pi * number)) ** (1 / 3)
        surface = law["surface_tension_n_m"] * math.pi * number * diameter**2
        law["droplet_diameter_m"] = diameter
    else:
        if number != 0:
            raise ValueError("Particle stock without dispersed/carrier pair")
        diameter = law["droplet_diameter_m"]
        surface = 0.0
    return (
        {
            "number": number,
            "diameter_m": diameter,
            "surface_energy_j": surface,
            "drop_volume_m3": volume,
            "drop_mask": mask,
            "present": present,
            "phase_roles": roles,
        },
        law,
        surface,
    )


def number_flux(base, contact, left, right, law, left_native, right_native, spec):
    index = (
        law["oil_component_index"]
        if law["dispersed_liquid"] == "oil"
        else law["water_component_index"]
    )

    def concentration(packet, native):
        x = packet["phase_x"]
        role = identify_roles(law, spec, x, packet["phase_z"], packet["phase_densities"])
        chosen = 1 if law["dispersed_liquid"] == "oil" else 2
        mask = (role == chosen) & (x[:, index] >= law["minimum_marker_fraction"])
        weight = float(packet["phase_weights"] @ mask)
        value = (
            weight / (math.pi * law["droplet_diameter_m"] ** 3 / 6)
            if native is None
            else weight * native["number"] / max(native["drop_volume_m3"], 1e-30)
        )
        return value, float(packet["velocity"] @ mask)

    nl, vl = concentration(left, left_native)
    nr, vr = concentration(right, right_native)
    donor = left if contact >= 0 else right
    number_density = nl if contact >= 0 else nr
    volume_rate = float(np.sum(base[:-2])) / float(np.sum(donor["phase_concentration"]))
    return (
        volume_rate * number_density
        + nl * (max(vl, 0.0) - max(float(left["u"]), 0.0))
        + nr * (min(vr, 0.0) - min(float(right["u"]), 0.0))
    )


def audit_population_balances(nodes, old, new, data, rates, saved, dt):
    expected = {name for name, node in nodes.items() if node.get("droplet_population") is not None}
    if set(saved) != expected:
        raise ValueError("Population report registry differs")
    for name in expected:
        node = nodes[name]
        d = data[name]
        p = d["population"]
        record = saved[name]
        number = p["number"]
        prior = old[name]["droplet_number"]
        net = rates[name]
        coag = (
            0.5
            * node["droplet_population"]["coalescence_kernel_m3_s"]
            * number**2
            / node["volume_m3"]
        )
        spec = node["droplet_population"]
        powered = spec.get("model") == "powered_monomoment_breakup"
        breakup = spec["breakup_rate_s"] * number if powered else 0.0
        if not p["present"]:
            close(number, 0.0, atol=1e-12, rtol=0.0)
        elif prior > 0 or net > 0:
            close(number, prior + dt * (net - coag + breakup), atol=1e-6, rtol=1e-9)
        else:
            expected_number = p["drop_volume_m3"] / (
                math.pi * node["drift_flux"]["droplet_diameter_m"] ** 3 / 6
            )
            close(number, expected_number, atol=1e-6, rtol=1e-9)
        for key in ("number", "diameter_m", "surface_energy_j", "drop_volume_m3"):
            close(record[key], p[key], atol=1e-15, rtol=2e-8)
        close(record["net_number_rate_s"], net, atol=1e-5, rtol=2e-7)
        close(record["coalescence_loss_rate_s"], coag, atol=1e-5, rtol=2e-8)
        if powered:
            close(record["breakup_gain_rate_s"], breakup, atol=1e-5, rtol=2e-8)
            close(record["agitation_power_w"], spec["agitation_power_w"], atol=1e-12, rtol=0.0)
            fraction = spec["breakup_rate_s"] * dt
            required = (
                p["surface_energy_j"] * (-math.expm1(math.log1p(-fraction) / 3)) / dt
                if p["present"] and fraction < 1
                else 0.0
            )
            if (
                p["present"]
                and (prior > 0 or net > 0)
                and (fraction >= 1 or required > spec["agitation_power_w"] * (1 + 1e-9))
            ):
                raise ValueError("Breakup consumed more surface work than the mixer supplied")
            close(record["breakup_surface_power_w"], required, atol=1e-10, rtol=2e-8)
        if abs(record["balance_residual"]) > 1e-8:
            raise ValueError("Population residual rejected")


def agitation_powers(nodes):
    return {
        name: node["droplet_population"]["agitation_power_w"]
        for name, node in nodes.items()
        if (node.get("droplet_population") or {}).get("model") == "powered_monomoment_breakup"
    }
