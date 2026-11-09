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
from reference_hllc_local import wave_flux
from reference_oil_water_drift import velocities as oil_water_velocities
from reference_oil_water_drift import wall_viscosity
from reference_pr_compressors import audit_compressors
from reference_pr_drives import audit_drives, initial_rotors, machine_model, rotor_energy
from reference_pr_hydrostatic import attach_faces, gravity_force
from reference_pr_population import (
    agitation_powers,
    audit_population_balances,
    number_flux,
    population_properties,
)
from reference_pr_pump_connections import audit_pump_connections
from reference_pr_pumps import audit_pumps
from reference_pr_restrictions import audit_restrictions

from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork


def face_area(node, side):
    diameter = node.get(side + "_diameter_m")
    return math.pi * (node["diameter_m"] if diameter is None else diameter) ** 2 / 4


def audit_heat_links(links, temperatures, openings, saved):
    if set(saved) != {link["name"] for link in links}:
        raise ValueError("Thermal connection registry differs")
    rates = {}
    for link in links:
        a, b = link["source"], link["target"]
        ta, tb = temperatures[a], temperatures[b]
        ua = link["conductance_w_k"] if openings[link["name"]] else 0.0
        q = ua * (ta - tb)
        production = ua * (ta - tb) ** 2 / (ta * tb)
        close(saved[link["name"]]["heat_rate_w"], q, atol=1e-9, rtol=2e-8)
        close(saved[link["name"]]["entropy_production_w_k"], production, atol=1e-10, rtol=2e-8)
        rates[a] = rates.get(a, 0.0) - q
        rates[b] = rates.get(b, 0.0) + q
    return rates


def reference_sound(fluid, p, t, x, tag="stable"):
    ref = Reference(fluid, np.flatnonzero(np.asarray(x) > 0))
    if ref.pure is not None:
        return ref.pure.state(p, t, tag)["sound_speed_m_s"]
    dp = max(1e-4, p * 1e-5)
    dt = t * 1e-5

    def values(p, t):
        q = ref.state(p, t, x, tag)
        return np.array([q["rho"], q["s"]])

    jp = (values(p + dp, t) - values(p - dp, t)) / (2 * dp)
    jt = (values(p, t + dt) - values(p, t - dt)) / (2 * dt)
    derivative = jp[0] - jt[0] * jp[1] / jt[1]
    if derivative <= 0:
        raise ValueError("Reference acoustic derivative rejected")
    return 1 / math.sqrt(derivative)


def reservoir_trace(data, saved, side, model, fluid):
    p, t = saved["pressure_pa"], saved["temperature_k"]
    x = np.asarray(saved["mole_fractions"])
    close(x, data["x"][0], atol=1e-10, rtol=2e-8)
    q = Reference(fluid, np.flatnonzero(x > 0)).state(p, t, x, "stable")
    contact = saved["contact_outward_velocity_m_s"]
    outward = max(contact, 0.0)
    close(saved["outward_velocity_m_s"], outward, atol=1e-8, rtol=2e-7)
    mode = "reversible_outflow" if contact >= 0 else "mixed_inflow"
    if saved["mode"] != mode:
        raise ValueError("Reservoir trace direction differs")
    h = (
        q["h"]
        + 0.5 * float(np.dot(x, [c.molar_mass_kg_mol for c in fluid.components])) * outward**2
    )
    if contact >= 0:
        close(q["s"], data["s_phase"][0], atol=1e-6, rtol=1e-8)
        close(h, data["h"][0], atol=1e-5, rtol=1e-8)
    else:
        close(p, data["p"], atol=1e-4, rtol=1e-8)
        close(t, data["t"], atol=1e-7, rtol=1e-9)
    close(saved["reservoir_entropy_j_mol_k"], data["s_phase"][0], atol=1e-6, rtol=1e-8)
    close(saved["trace_entropy_j_mol_k"], q["s"], atol=1e-6, rtol=1e-8)
    close(saved["reservoir_total_enthalpy_j_mol"], data["h"][0], atol=1e-5, rtol=1e-8)
    close(saved["trace_total_enthalpy_j_mol"], h, atol=1e-5, rtol=1e-8)
    return {
        **data,
        "p": p,
        "t": t,
        "x": x[None, :],
        "vm": np.array([q["vm"]]),
        "h": np.array([q["h"]]),
        "s_phase": np.array([q["s"]]),
        "rho_phase": np.array([q["rho"]]),
        "alpha": np.ones(1),
        "drift": np.zeros(1),
        "port_velocity": outward if side == "source" else -outward,
        "sound": reference_sound(fluid, p, t, x)
        if model.get("wave_fan") in ("local_single_phase", "pressure_single_phase")
        else None,
    }


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
    population, live_law, surface = population_properties(
        node, state, x, pn, vm, np.array([v["z"] for v in q]), rho
    )
    if population is not None:
        closure = live_law
    if (
        population is not None
        and not population["present"]
        and closure.get("model") != "dispersed_gas_liquid"
    ):
        closure = None
    if closure is not None and closure.get("model") == "dispersed_gas_liquid":
        from reference_gas_mist import velocities as mist_velocities

        velocities = mist_velocities(
            closure,
            pn * vm / (pn @ vm),
            rho,
            x,
            u,
            node["diameter_m"],
            model["gravity_m_s2"] * node["elevation_change_m"] / node["length_m"],
            model["gravity_m_s2"],
            p,
            t,
            mw,
            None if population is None else population["phase_roles"],
        )
        drift = velocities - u
    elif (
        closure is not None
        and len(pn) > 1
        and closure.get("model") in ("dispersed_oil_water", "concentrated_oil_water")
    ):
        alpha = pn * vm / (pn @ vm)
        velocities = oil_water_velocities(
            closure,
            alpha,
            rho,
            x,
            u,
            node["diameter_m"],
            model["gravity_m_s2"] * node["elevation_change_m"] / node["length_m"],
            model["gravity_m_s2"],
            p,
            None if population is None else population["phase_roles"],
        )
        drift = velocities - u
    elif closure is not None and len(pn) > 1 and closure.get("model") == "stratified_gas_liquid":
        from reference_stratified_drift import velocities as stratified_velocities

        if len(pn) != 2 or min(v["z"] for v in q) >= 0.5 or max(v["z"] for v in q) <= 0.5:
            raise ValueError("Stratified closure requires one dense liquid and one gas")
        velocities = stratified_velocities(
            closure,
            pn * vm / (pn @ vm),
            rho,
            u,
            node["diameter_m"],
            model["gravity_m_s2"] * node["elevation_change_m"] / node["length_m"],
            model["gravity_m_s2"],
        )
        drift = velocities - u
    elif closure is not None and len(pn) > 1:
        alpha = pn * vm / (pn @ vm)
        gas = int(np.argmin(rho))
        ag = alpha[gas]
        rg = rho[gas]
        liquid = np.arange(len(pn)) != gas
        rl = np.dot(alpha[liquid], rho[liquid]) / alpha[liquid].sum()
        if closure.get("model") == "holmes_gas_liquid":
            from reference_holmes_drift import parameters as holmes_parameters

            if len(pn) != 2:
                raise ValueError("Holmes closure requires two gas/liquid phases")
            c0, vd = holmes_parameters(
                closure,
                ag,
                rg,
                rl,
                u,
                node["diameter_m"],
                model["gravity_m_s2"] * node["elevation_change_m"] / node["length_m"],
                model["gravity_m_s2"],
            )
        elif closure.get("model") in ("relap7_vertical_pipe", "relap7_vertical_transition"):
            if len(pn) != 2:
                raise ValueError("RELAP-7 requires gas/liquid two-phase flow")
            axial = model["gravity_m_s2"] * node["elevation_change_m"] / node["length_m"]
            sigma, mu = closure["surface_tension_n_m"], closure["liquid_viscosity_pa_s"]
            length = math.sqrt(sigma / (abs(axial) * (rl - rg)))
            base = (sigma * abs(axial) * (rl - rg) / rl**2) ** 0.25
            c0 = 1.2 - 0.2 * math.sqrt(rg / rl)
            if closure.get("regime") == "dispersed_bubbly":
                vd = math.sqrt(2) * base
            else:
                diameter = min(30.0, node["diameter_m"] / length)
                viscosity = mu / math.sqrt(rl * sigma * length)
                vd = 0.0019 * diameter**0.809 * (rg / rl) ** (-0.157) * viscosity ** (-0.562) * base
                if closure.get("model") == "relap7_vertical_transition":
                    weight = max(0.0, min(1.0, 1 - ag / 0.1))
                    vd = weight * math.sqrt(2) * base + (1 - weight) * vd
            vd = math.copysign(vd, axial)
        else:
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
    variance = 0.0
    if population is not None and "coordinates" in population:
        from reference_pr_sections import velocities as section_velocities

        dispersion = section_velocities(
            node,
            population,
            rho,
            velocities,
            model["gravity_m_s2"] * node["elevation_change_m"] / node["length_m"],
            model["gravity_m_s2"],
            model["maximum_velocity_m_s"],
            p,
            t,
            float((x @ mw)[population["phase_roles"] == 0].sum()),
        )
        variance = float(phase_mass @ population["drop_mask"]) * dispersion
    kinetic = 0.5 * (phase_mass @ velocities**2 + variance)
    potential = (
        mass * model["gravity_m_s2"] * (node["elevation_m"] - model["reference_elevation_m"])
    )
    internal = pn @ uu
    entropy = pn @ ss
    stress = (phase_mass @ drift**2 + variance) / node["volume_m3"]
    close(internal + kinetic + potential + surface, state["total_energy_j"], atol=1e-6, rtol=2e-8)
    if node["kind"] == "vessel":
        close(state["momentum_kg_m_s"], 0.0, atol=1e-12, rtol=0.0)
    if node.get("wall") is not None:
        w = node["wall"]
        tw = state["wall_temperature_k"]
        if not w["min_temperature_k"] <= tw <= w["max_temperature_k"]:
            raise ValueError("Wall outside domain")
    elif state["wall_temperature_k"] is not None:
        raise ValueError("Unexpected wall state")
    sound = (
        reference_sound(fluid, p, t, x[0], state["phase_roots"][0])
        if (
            model.get("wave_fan", "fixed_bound") in ("local_single_phase", "pressure_single_phase")
            or any(pump["node"] == node["name"] for pump in model.get("pumps", []))
            or any(u["card"]["start"] == node["name"] for u in model.get("pump_connections", []))
        )
        and len(pn) == 1
        else None
    )
    return {
        "sound": sound,
        "n": n,
        "pn": pn,
        "x": x,
        "p": p,
        "t": t,
        "u": u,
        "vm": vm,
        "rho_phase": rho,
        "phase_z": np.array([v["z"] for v in q]),
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
        "surface": surface,
        "population": population,
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
    from reference_pr_sections import endpoint_data

    data = endpoint_data(data)
    data = data.get("hydrostatic_faces", {}).get(link[side + "_port"], data)
    weights = data["alpha"]
    height = link.get(side + "_height_m")
    if height is not None:
        geometry = node["geometry"]
        if node["kind"] == "pipe_cell":
            geometry = {
                "diameter_m": node["diameter_m"],
                "length_m": node["length_m"],
                "orientation": "horizontal"
                if abs(node["elevation_change_m"]) < 1e-12
                else "vertical",
            }
            extent = (
                geometry["diameter_m"]
                if geometry["orientation"] == "horizontal"
                else geometry["length_m"]
            )
            if max(data["rho_phase"]) * model["gravity_m_s2"] * extent / data["p"] > 1e-3:
                raise ValueError("Lateral pickup outside low-head envelope")
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
    port = link[side + "_port"]
    normal = 0.0 if port.startswith("side:") else (1.0 if port == "right" else -1.0)
    projection = normal if side == "source" else -normal
    v = data["u"] * projection if node["kind"] == "pipe_cell" else 0.0
    if "port_velocity" in data:
        v = data["port_velocity"]
    w = data["drift"] * projection if node["kind"] == "pipe_cell" else np.zeros(len(weights))
    concentration = (weights[:, None] * data["x"] / data["vm"][:, None]).sum(0)
    rho = weights @ data["rho_phase"]
    phase_mass_mol = data["rho_phase"] * data["vm"]
    internal = np.sum(weights * (data["h"] - data["p"] * data["vm"]) / data["vm"])
    surface_phase = np.zeros_like(weights)
    population = data.get("population")
    if population is not None:
        surface_phase = (
            weights
            * population["drop_mask"]
            * population["surface_energy_j"]
            / max(population["drop_volume_m3"], 1e-30)
        )
        if "surface_density_phase" in population:
            surface_phase = population["surface_density_phase"]
        internal += surface_phase.sum()
    if normal == 0.0 and node["kind"] == "pipe_cell":
        internal += 0.5 * np.sum(weights * data["rho_phase"] * (data["u"] + data["drift"]) ** 2)
    entropy = np.sum(weights * data["s_phase"] / data["vm"])
    z = node["elevation_m"] + (
        height
        - (node["diameter_m"] if abs(node["elevation_change_m"]) < 1e-12 else node["length_m"]) / 2
        if node["kind"] == "pipe_cell" and normal == 0.0 and height is not None
        else node["elevation_change_m"] * normal / 2
        if node["kind"] == "pipe_cell"
        else 0.0
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
        "phase_z": data["phase_z"],
        "phase_densities": data["rho_phase"],
        "phase_x": data["x"],
        "phase_weights": weights,
        "phase_concentration": concentration_phase,
        "sound": data["sound"],
        "tangential_velocity": float(
            np.sum(weights * data["rho_phase"] * (data["u"] + data["drift"])) / rho
        ),
        "velocity": velocity,
        "u": v,
        "actual": actual,
        "bulk": bulk,
        "actual_entropy": weights / data["vm"] * data["s_phase"] * velocity,
        "bulk_entropy": weights / data["vm"] * data["s_phase"] * v,
    }
    if population is not None:
        actual[:, -1] += surface_phase * velocity
        bulk[:, -1] += surface_phase * v
    return (uc, ff, data["p"], v, rho, entropy), weights, normal, extra


def audit(config, report, *, external_exchange=None):
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
    if set(config["initial"]) == {"state"}:
        if config["initial"]["state"] != old:
            raise ValueError("Requested restart checkpoint differs")
    else:
        initial_rotors(model, old)
    old_start = old
    checked = 0
    maximum = np.zeros(nc + 2)
    all_heat = 0.0
    all_work = 0.0
    all_exchange_n = np.zeros(nc)
    all_exchange_e = 0.0
    olddata = {n: decode(old["nodes"][n], nodes[n], model, fluid) for n in names}
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        end = old["time_s"] + command["duration_s"]
        opens = {c["name"]: True for c in model["links"] + model.get("heat_links", [])}
        opens.update(
            {
                p["card"]["name"]: p["card"]["open"]
                for p in [*model.get("pumps", []), *model.get("pump_connections", [])]
            }
        )
        opens.update({unit["name"]: unit["open"] for unit in model.get("compressors", [])})
        opens.update({u["name"]: u["open"] for u in model.get("restrictions", [])})
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
        interval_heat = interval_entropy = interval_work = 0.0
        interval_exchange_n = np.zeros(nc)
        interval_exchange_e = 0.0
        interval_start = old
        for step in interval["steps"]:
            dt = step["dt_s"]
            new = step["state"]
            if dt <= 0 or new["model_id"] != parsed.model_id or tuple(new["nodes"]) != names:
                raise ValueError("Invalid transport checkpoint")
            close(
                new["time_s"] - old["time_s"],
                dt,
                atol=8 * max(math.ulp(new["time_s"]), math.ulp(old["time_s"])),
                rtol=0.0,
            )
            data = {n: decode(new["nodes"][n], nodes[n], model, fluid) for n in names}
            attach_faces(
                model, nodes, data, step.get("hydrostatic_faces", {}), fluid, reference_sound
            )
            net = {n: np.zeros(nc + 2) for n in names}
            exchange_n = np.zeros(nc)
            exchange_e = exchange_entropy = 0.0
            if external_exchange is not None:
                exchanged = external_exchange(old, new, data, dt)
                if set(exchanged["net"]) != set(names):
                    raise ValueError("External exchange registry differs")
                for name, rate in exchanged["net"].items():
                    value = np.asarray(rate, dtype=float)
                    if value.shape != (nc + 2,) or not np.isfinite(value).all():
                        raise ValueError("Invalid independent external exchange")
                    net[name] += value
                exchange_n = dt * sum((v[:-2] for v in net.values()), np.zeros(nc))
                exchange_e = dt * sum(v[-1] for v in net.values())
                exchange_entropy = float(exchanged["incoming_entropy"])
            particle_rates = {
                n: 0.0 for n in names if nodes[n].get("droplet_population") is not None
            }
            occupied = set()
            for link in model["links"]:
                a, b = link["source"], link["target"]
                left_data, right_data = data[a], data[b]
                traces = step.get("reservoir_traces", {}).get(link["name"], {})
                for side, name, other in [("source", a, b), ("target", b, a)]:
                    eligible = (
                        model.get("reservoir_ports", "riemann") == "stagnation_single_phase"
                        and opens[link["name"]]
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
                        point = reservoir_trace(data[name], traces[side], side, model, fluid)
                        if side == "source":
                            left_data = point
                        else:
                            right_data = point
                    elif side in traces:
                        raise ValueError("Unexpected reservoir trace")
                left, wa, na, da = endpoint(nodes[a], left_data, link, "source", model)
                right, wb, nb, db = endpoint(nodes[b], right_data, link, "target", model)
                area = math.pi * link["diameter_m"] ** 2 / 4
                if opens[link["name"]]:
                    if (
                        model.get("wave_fan", "fixed_bound")
                        in ("local_single_phase", "pressure_single_phase")
                        and len(data[a]["pn"]) == 1
                        and len(data[b]["pn"]) == 1
                    ):
                        flux, sf, contact, _, fallback = wave_flux(
                            left,
                            right,
                            da["sound"],
                            db["sound"],
                            model["wave_speed_bound_m_s"],
                            estimate="pvrs"
                            if model.get("wave_fan") == "pressure_single_phase"
                            else "davis",
                        )
                    else:
                        flux, sf, contact, _ = port_flux(left, right, model["wave_speed_bound_m_s"])
                        fallback = False
                    base_flux = flux.copy()
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
                    base_flux = flux.copy()
                    sf = contact = 0.0
                    fallback = False
                for side, trace in traces.items():
                    close(
                        trace["contact_outward_velocity_m_s"],
                        contact * (1.0 if side == "source" else -1.0),
                        atol=1e-8,
                        rtol=2e-7,
                    )
                saved = step["links"][link["name"]]
                family_nodes = [
                    nodes[n] for n in (a, b) if nodes[n].get("droplet_population") is not None
                ]
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
                    if not opens[link["name"]]:
                        rate = rate * 0
                    if set(saved.get("droplet_number_fluxes", {})) != {family}:
                        raise ValueError("Particle flux registry differs")
                    if (
                        family_nodes[0]["droplet_population"].get("model")
                        == "sectional_droplet_population"
                    ):
                        if set(saved.get("droplet_section_fluxes", {})) != {family}:
                            raise ValueError("Sectional flux registry differs")
                        close(
                            saved["droplet_section_fluxes"][family][:-1],
                            rate[:-1],
                            atol=1e-5,
                            rtol=2e-7,
                        )
                        close(
                            saved["droplet_section_fluxes"][family][-1],
                            rate[-1],
                            atol=1e-20,
                            rtol=2e-7,
                        )
                        close(
                            saved["droplet_number_fluxes"][family],
                            rate[:-1].sum(),
                            atol=1e-5,
                            rtol=2e-7,
                        )
                    else:
                        close(saved["droplet_number_fluxes"][family], rate, atol=1e-5, rtol=2e-7)
                    if a in particle_rates:
                        particle_rates[a] -= rate
                    if b in particle_rates:
                        particle_rates[b] += rate
                elif saved.get("droplet_number_fluxes"):
                    raise ValueError("Unexpected particle flux")
                if saved.get("local_fan_fixed_fallback", False) != fallback:
                    raise ValueError("Acoustic fan fallback differs")
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
                        occupied.add((n, link[side + "_port"]))
                        if not opens[link["name"]]:
                            wall = (
                                data[n]
                                .get("hydrostatic_faces", {})
                                .get(link[side + "_port"], data[n])
                            )
                            update[-2] -= (
                                normal
                                * area
                                * (
                                    wall["p"]
                                    + wall["rho"] * data[n]["u"] ** 2
                                    + data[n]["stress"]
                                    + normal
                                    * model["wave_speed_bound_m_s"]
                                    * wall["rho"]
                                    * data[n]["u"]
                                )
                            )
                    net[n] += update
            audit_population_balances(
                nodes,
                old["nodes"],
                new["nodes"],
                data,
                particle_rates,
                step.get("droplet_populations", {}),
                dt,
            )
            heat_rates = audit_heat_links(
                model.get("heat_links", []),
                {n: d["t"] for n, d in data.items()},
                opens,
                step.get("heat_links", {}),
            )
            for name, rate in heat_rates.items():
                net[name][-1] += rate
            for name, rate in audit_restrictions(
                model,
                data,
                opens,
                command.get("restriction_openings", {}),
                step.get("restrictions", {}),
                fluid,
            ).items():
                net[name] += rate
            live_model = machine_model(model, old, new)
            pump_forces, pump_powers = audit_pumps(live_model, data, opens, step.get("pumps", {}))
            for name, power in pump_powers.items():
                net[name][-1] += power
            compressor_net, compressor_power = audit_compressors(
                live_model, data, opens, step.get("compressors", {}), fluid
            )
            for name, rate in compressor_net.items():
                net[name] += rate
            connection_net, connection_powers = audit_pump_connections(
                live_model, data, opens, step.get("pump_connections", {}), fluid
            )
            for name, rate in connection_net.items():
                net[name] += rate
            work = dt * (
                sum(pump_powers.values()) + compressor_power + sum(connection_powers.values())
            )
            correction, drive_heat, drive_entropy, drive_heat_sources = audit_drives(
                model,
                old,
                new,
                opens,
                command.get("drive_torque_nm", {}),
                dt,
                step.get("drives", {}),
                pump_powers,
                {**step.get("compressors", {}), **step.get("pump_connections", {})},
            )
            mixing = agitation_powers(nodes)
            for name, power in mixing.items():
                net[name][-1] += power
            work += dt * sum(mixing.values())
            if step.get("agitation_work_j", {}) != {name: dt * p for name, p in mixing.items()}:
                raise ValueError("Mechanical agitation work differs")
            work += correction
            for name, rate in drive_heat_sources.items():
                net[name][-1] += rate
            production = -dt * exchange_entropy
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
                            wall = d.get("hydrostatic_faces", {}).get(port, d)
                            net[n][-2] -= (
                                normal
                                * face_area(node, port)
                                * (
                                    wall["p"]
                                    + wall["rho"] * d["u"] ** 2
                                    + d["stress"]
                                    + normal * model["wave_speed_bound_m_s"] * wall["rho"] * d["u"]
                                )
                            )
                    force = gravity_force(node, d, model, face_area)
                    from reference_stratified_drift import wall_force as stratified_wall_force

                    force += stratified_wall_force(node, d, model)
                    viscosity = wall_viscosity(node, d)
                    if viscosity:
                        force -= (
                            node["volume_m3"]
                            * reference_darcy(
                                d["u"],
                                d["rho"],
                                1.0,
                                node["diameter_m"],
                                viscosity,
                                node["roughness_m"],
                            )[0]
                        )
                    coefficient = command.get("loss_coefficients", {}).get(
                        n, node.get("local_loss_coefficient", 0.0)
                    )
                    force -= 0.5 * area * coefficient * d["rho"] * d["u"] * abs(d["u"])
                    force += pump_forces.get(n, 0.0)
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
            production += drive_entropy
            ambient -= drive_heat
            close(step["entropy_production_j_k"], production, atol=2e-6, rtol=2e-6)
            close(step["ambient_heat_j"], ambient, atol=1e-7, rtol=2e-7)
            close(step["external_heat_j"], external, atol=1e-7, rtol=2e-7)
            close(step.get("external_work_j", 0.0), work, atol=1e-10, rtol=2e-7)
            close(
                step["conservation_residuals"],
                ledger(
                    old,
                    new,
                    nodes,
                    ambient + external + work + exchange_e,
                    model,
                    incoming_n=exchange_n,
                ),
                atol=1e-6,
                rtol=2e-7,
            )
            interval_heat += ambient + external
            interval_exchange_n += exchange_n
            interval_exchange_e += exchange_e
            interval_work += work
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
        close(final["time_s"], end, atol=8 * math.ulp(end), rtol=0.0)
        close(interval["total_heat_j"], interval_heat, atol=1e-6, rtol=2e-7)
        close(interval.get("total_work_j", 0.0), interval_work, atol=1e-10, rtol=2e-7)
        close(interval["entropy_production_j_k"], interval_entropy, atol=2e-6, rtol=2e-6)
        close(
            interval["conservation_residuals"],
            ledger(
                interval_start,
                final,
                nodes,
                interval_heat + interval_work + interval_exchange_e,
                model,
                incoming_n=interval_exchange_n,
            ),
            atol=1e-6,
            rtol=2e-7,
        )
        old = final
        all_heat += interval_heat
        all_work += interval_work
        all_exchange_n += interval_exchange_n
        all_exchange_e += interval_exchange_e
    close(report["total_heat_j"], all_heat, atol=1e-6, rtol=2e-7)
    close(report.get("total_work_j", 0.0), all_work, atol=1e-10, rtol=2e-7)
    close(
        report["conservation_residuals"],
        ledger(
            old_start,
            old,
            nodes,
            all_heat + all_work + all_exchange_e,
            model,
            incoming_n=all_exchange_n,
        ),
        atol=1e-6,
        rtol=2e-7,
    )
    return {
        "accepted": True,
        "checked_substeps": checked,
        "maximum_local_component_residual_mol": float(maximum[:nc].max()),
        "maximum_local_energy_residual_j": float(maximum[-1]),
    }


def ledger(before, after, nodes, heat, model=None, *, incoming_n=None):
    nc = len(next(iter(before["nodes"].values()))["component_moles"])
    dn = [
        math.fsum(s["component_moles"][i] for s in after["nodes"].values())
        - math.fsum(s["component_moles"][i] for s in before["nodes"].values())
        for i in range(nc)
    ]
    if incoming_n is not None:
        dn = (np.asarray(dn) - np.asarray(incoming_n)).tolist()
    energy = rotor_energy(after, model) - rotor_energy(before, model) if model is not None else 0.0
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
