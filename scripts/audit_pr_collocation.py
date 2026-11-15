"""Independent scalar-PR/HLLC audit of serialized conservative Gauss stages.

Uses the established NumPy EOS/port reference, never runtime flash, decode,
flow, Newton or transport inspection. It checks identities/balances, not
independent global TPD stability or experimental closure calibration.
"""

import math

import numpy as np
from check_pr_equilibrium_transport import decode, endpoint, face_area
from check_pr_momentum_vessels import port_flux
from check_stratified_pr_flow import close
from check_stratified_pr_network import reference_darcy
from reference_hllc_local import wave_flux
from reference_pr_hydrostatic import gravity_force


def reference_rates(model, states, fluid, commands):
    nodes = {n["name"]: n for n in model["nodes"]}
    data = {name: decode(states[name], node, model, fluid) for name, node in nodes.items()}
    net = {name: np.zeros(len(fluid.components) + 2) for name in nodes}
    occupied, faces = set(), {}
    for link in model["links"]:
        a, b = link["source"], link["target"]
        left, _, na, da = endpoint(nodes[a], data[a], link, "source", model)
        right, _, nb, db = endpoint(nodes[b], data[b], link, "target", model)
        area = math.pi * link["diameter_m"] ** 2 / 4
        if commands["openings"][link["name"]]:
            if (
                model.get("wave_fan", "fixed_bound")
                in ("local_single_phase", "pressure_single_phase")
                and len(data[a]["pn"]) == len(data[b]["pn"]) == 1
            ):
                flux, *_ = wave_flux(
                    left,
                    right,
                    da["sound"],
                    db["sound"],
                    model["wave_speed_bound_m_s"],
                    estimate="pvrs" if model["wave_fan"] == "pressure_single_phase" else "davis",
                )
            else:
                flux, *_ = port_flux(left, right, model["wave_speed_bound_m_s"])
            for packet, forward in ((da, True), (db, False)):
                mask = packet["velocity"] > 0 if forward else packet["velocity"] < 0
                bulk = packet["u"] > 0 if forward else packet["u"] < 0
                flux += packet["actual"][mask].sum(0) - (packet["bulk"].sum(0) if bulk else 0.0)
        else:
            flux = np.zeros(len(fluid.components) + 2)
        faces[link["name"]] = area * flux
        for side, name, normal, sign, packet in [
            ("source", a, na, -1.0, da),
            ("target", b, nb, 1.0, db),
        ]:
            update = np.r_[
                sign * area * flux[:-2],
                -normal * area * flux[-2] if nodes[name]["kind"] == "pipe_cell" else 0.0,
                sign * area * flux[-1],
            ]
            if nodes[name]["kind"] == "pipe_cell":
                if normal == 0.0:
                    outward_mass = -sign * sum(
                        q * c.molar_mass_kg_mol for q, c in zip(flux[:-2], fluid.components)
                    )
                    update[-2] -= area * max(0.0, outward_mass) * packet["tangential_velocity"]
                occupied.add((name, link[side + "_port"]))
                if not commands["openings"][link["name"]]:
                    d = data[name]
                    update[-2] -= (
                        normal
                        * area
                        * (
                            d["p"]
                            + d["rho"] * d["u"] ** 2
                            + d["stress"]
                            + normal * model["wave_speed_bound_m_s"] * d["rho"] * d["u"]
                        )
                    )
            net[name] += update
    for link in model.get("heat_links", []):
        q = (link["conductance_w_k"] if commands["openings"][link["name"]] else 0.0) * (
            data[link["source"]]["t"] - data[link["target"]]["t"]
        )
        net[link["source"]][-1] -= q
        net[link["target"]][-1] += q
    for name, node in nodes.items():
        d = data[name]
        if node["kind"] == "pipe_cell":
            for port, normal in [("left", -1.0), ("right", 1.0)]:
                if (name, port) not in occupied:
                    net[name][-2] -= (
                        normal
                        * face_area(node, port)
                        * (
                            d["p"]
                            + d["rho"] * d["u"] ** 2
                            + d["stress"]
                            + normal * model["wave_speed_bound_m_s"] * d["rho"] * d["u"]
                        )
                    )
            force = gravity_force(node, d, model, face_area)
            if node["dynamic_viscosity_pa_s"]:
                force -= (
                    node["volume_m3"]
                    * reference_darcy(
                        d["u"],
                        d["rho"],
                        1.0,
                        node["diameter_m"],
                        node["dynamic_viscosity_pa_s"],
                        node["roughness_m"],
                    )[0]
                )
            loss = commands.get("loss_coefficients", {}).get(
                name, node.get("local_loss_coefficient", 0.0)
            )
            force -= (
                0.5 * math.pi * node["diameter_m"] ** 2 / 4 * loss * d["rho"] * d["u"] * abs(d["u"])
            )
            net[name][-2] += force
        net[name][-1] += commands.get("heat_w", {}).get(name, 0.0)
    return net, faces


def audit_gauss_step(model, before, record, fluid, commands):
    if record["method"] != "gauss4" or len(record["stage_states"]) != 2:
        raise ValueError("Exactly two Gauss stages required")
    old = before["nodes"]
    new = record["state"]["nodes"]
    stages = record["stage_states"]
    dt = record["dt_s"]
    close(record["start_time_s"], before["time_s"], atol=1e-15, rtol=1e-13)
    close(record["state"]["time_s"] - before["time_s"], dt, atol=1e-15, rtol=1e-13)
    decoded = [reference_rates(model, stage, fluid, commands) for stage in stages]
    nodes = {n["name"]: n for n in model["nodes"]}
    for name, node in nodes.items():
        decode(old[name], node, model, fluid)
        decode(new[name], node, model, fluid)

    def vector(s):
        return np.r_[s["component_moles"], s["momentum_kg_m_s"], s["total_energy_j"]]

    q = math.sqrt(3) / 6
    coefficients = ((0.25, 0.25 - q), (0.25 + q, 0.25))
    for name, node in nodes.items():
        rates = [pair[0][name].copy() for pair in decoded]
        if node["kind"] == "vessel":
            for rate in rates:
                rate[-2] = 0.0
        for i, stage in enumerate(stages):
            expected = dt * sum(coefficients[i][j] * rates[j] for j in range(2))
            delta = vector(stage[name]) - vector(old[name])
            close(delta[:-2], expected[:-2], atol=1e-10, rtol=2e-6)
            close(delta[-2], expected[-2], atol=1e-9, rtol=2e-6)
            close(delta[-1], expected[-1], atol=1e-6, rtol=2e-6)
        expected = dt * (rates[0] + rates[1]) / 2
        delta = vector(new[name]) - vector(old[name])
        close(delta[:-2], expected[:-2], atol=1e-10, rtol=2e-6)
        close(delta[-2], expected[-2], atol=1e-9, rtol=2e-6)
        close(delta[-1], expected[-1], atol=1e-6, rtol=2e-6)
    if {r["name"] for r in record["links"]} != {l["name"] for l in model["links"]}:
        raise ValueError("Face registry differs")
    for row in record["links"]:
        expected = dt * (decoded[0][1][row["name"]] + decoded[1][1][row["name"]]) / 2
        close(row["integrated_flux"], expected, atol=1e-8, rtol=2e-6)
    for link in model.get("heat_links", []):
        ua = link["conductance_w_k"] if commands["openings"][link["name"]] else 0.0
        expected = (
            dt
            * ua
            * sum(
                stage[link["source"]]["temperature_k"] - stage[link["target"]]["temperature_k"]
                for stage in stages
            )
            / 2
        )
        close(record["heat_links"][link["name"]], expected, atol=1e-9, rtol=2e-6)
    return {"accepted": True, "stages": 2, "nodes": len(nodes), "faces": len(record["links"])}
