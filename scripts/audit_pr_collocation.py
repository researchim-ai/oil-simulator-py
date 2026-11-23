"""Independent scalar-PR/HLLC audit of serialized conservative Gauss stages.

Uses the established NumPy EOS/port reference, never runtime flash, decode,
flow, Newton or transport inspection. It checks identities/balances, not
independent global TPD stability or experimental closure calibration.
"""

import math

import numpy as np
from check_pr_equilibrium_transport import decode, endpoint, face_area, ledger
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
    wall_rates = {}
    for name, node in nodes.items():
        if node.get("wall"):
            w = node["wall"]
            tw = states[name]["wall_temperature_k"]
            qh = w["fluid_ua_w_k"] * (tw - data[name]["t"])
            qa = w["ambient_ua_w_k"] * (w["ambient_temperature_k"] - tw)
            net[name][-1] += qh
            wall_rates[name] = qa - qh
    return net, faces, wall_rates


def audit_gauss_step(model, before, record, fluid, commands, *, boundary_exchange=None):
    if record["method"] not in ("gauss4", "radau3") or len(record["stage_states"]) != 2:
        raise ValueError("Exactly two Gauss stages required")
    if (
        any(
            model.get(k)
            for k in ("pumps", "compressors", "pump_connections", "restrictions", "drives")
        )
        or model.get("reservoir_ports", "riemann") != "riemann"
        or any(
            node.get(k)
            for node in model["nodes"]
            for k in (
                "hydrostatic_reconstruction",
                "droplet_population",
                "drift_flux",
                "distribution_strength",
                "buoyancy_relaxation_s",
            )
        )
    ):
        raise ValueError("Model outside independently audited Gauss scope")
    old = before["nodes"]
    new = record["state"]["nodes"]
    stages = record["stage_states"]
    dt = record["dt_s"]
    close(record["start_time_s"], before["time_s"], atol=1e-15, rtol=1e-13)
    close(record["state"]["time_s"] - before["time_s"], dt, atol=1e-15, rtol=1e-13)
    decoded = [reference_rates(model, stage, fluid, commands) for stage in stages]
    nodes = {n["name"]: n for n in model["nodes"]}
    if any(set(s) != set(nodes) for s in [old, new, *stages]):
        raise ValueError("Gauss node registry differs")
    olddata = {name: decode(old[name], node, model, fluid) for name, node in nodes.items()}
    newdata = {name: decode(new[name], node, model, fluid) for name, node in nodes.items()}
    maximum = np.zeros(len(fluid.components) + 2)

    def vector(s):
        return np.r_[s["component_moles"], s["momentum_kg_m_s"], s["total_energy_j"]]

    q = math.sqrt(3) / 6
    coefficients = (
        ((0.25, 0.25 - q), (0.25 + q, 0.25))
        if record["method"] == "gauss4"
        else ((5 / 12, -1 / 12), (0.75, 0.25))
    )
    weights = (0.5, 0.5) if record["method"] == "gauss4" else (0.75, 0.25)

    def weighted(values):
        return sum(w * v for w, v in zip(weights, values))

    incoming_n, incoming_e, incoming_s = None, 0.0, 0.0
    if boundary_exchange is not None:
        packets = boundary_exchange["stage_net"]
        if len(packets) != 2 or len(boundary_exchange["stage_entropy"]) != 2:
            raise ValueError("Two boundary quadrature packets required")
        for k, packet in enumerate(packets):
            if set(packet) != set(nodes):
                raise ValueError("Boundary node registry differs")
            for name, value in packet.items():
                value = np.asarray(value, dtype=float)
                if value.shape != (len(fluid.components) + 2,) or not np.isfinite(value).all():
                    raise ValueError("Invalid boundary vector")
                decoded[k][0][name] += value
        incoming_n = dt * weighted(
            [
                sum((np.asarray(v)[:-2] for v in packet.values()), np.zeros(len(fluid.components)))
                for packet in packets
            ]
        )
        incoming_e = dt * weighted([sum(v[-1] for v in packet.values()) for packet in packets])
        incoming_s = dt * weighted(boundary_exchange["stage_entropy"])
        close(record["exchange_component_moles"], incoming_n, atol=1e-10, rtol=2e-7)
        close(record["exchange_energy_j"], incoming_e, atol=1e-7, rtol=2e-7)
    for name, node in nodes.items():
        rates = [pair[0][name].copy() for pair in decoded]
        if node["kind"] == "vessel":
            for rate in rates:
                rate[-2] = 0.0
        for i, stage in enumerate(stages):
            expected = dt * sum(coefficients[i][j] * rates[j] for j in range(2))
            delta = vector(stage[name]) - vector(old[name])
            # Subtracting large stored inventories can lose several ULPs even
            # at zero transfer (phase decoding includes exp/log and a sum).
            # Bound that arithmetic from the two published stocks, rather
            # than relaxing the relative tolerance on the exchanged amount.
            rounding = 16 * np.spacing(np.maximum(abs(vector(stage[name])), abs(vector(old[name]))))
            close(delta[:-2], expected[:-2], atol=1e-10 + rounding[:-2], rtol=2e-6)
            close(delta[-2], expected[-2], atol=1e-9 + rounding[-2], rtol=2e-6)
            close(delta[-1], expected[-1], atol=1e-6 + rounding[-1], rtol=2e-6)
            if node.get("wall"):
                actual = node["wall"]["heat_capacity_j_k"] * (
                    stage[name]["wall_temperature_k"] - old[name]["wall_temperature_k"]
                )
                target = dt * sum(coefficients[i][j] * decoded[j][2][name] for j in range(2))
                close(actual, target, atol=1e-6, rtol=2e-6)
        expected = dt * weighted(rates)
        delta = vector(new[name]) - vector(old[name])
        maximum = np.maximum(maximum, abs(delta - expected))
        rounding = 16 * np.spacing(np.maximum(abs(vector(new[name])), abs(vector(old[name]))))
        close(delta[:-2], expected[:-2], atol=1e-10 + rounding[:-2], rtol=2e-6)
        close(delta[-2], expected[-2], atol=1e-9 + rounding[-2], rtol=2e-6)
        close(delta[-1], expected[-1], atol=1e-6 + rounding[-1], rtol=2e-6)
        if node.get("wall"):
            w = node["wall"]
            actual = w["heat_capacity_j_k"] * (
                new[name]["wall_temperature_k"] - old[name]["wall_temperature_k"]
            )
            target = dt * weighted([pair[2][name] for pair in decoded])
            close(actual, target, atol=1e-6, rtol=2e-6)
    expected_walls = {name for name, node in nodes.items() if node.get("wall")}
    if set(record.get("wall_exchanges", {})) != expected_walls:
        raise ValueError("Wall exchange registry differs")
    ambient = 0.0
    for name in expected_walls:
        w = nodes[name]["wall"]
        qh = dt * sum(
            weight
            * w["fluid_ua_w_k"]
            * (stage[name]["wall_temperature_k"] - stage[name]["temperature_k"])
            for weight, stage in zip(weights, stages)
        )
        qa = dt * sum(
            weight
            * w["ambient_ua_w_k"]
            * (w["ambient_temperature_k"] - stage[name]["wall_temperature_k"])
            for weight, stage in zip(weights, stages)
        )
        change = w["heat_capacity_j_k"] * (
            new[name]["wall_temperature_k"] - old[name]["wall_temperature_k"]
        )
        for key, value in (
            ("fluid_heat_j", qh),
            ("ambient_heat_j", qa),
            ("wall_energy_change_j", change),
        ):
            close(record["wall_exchanges"][name][key], value, atol=1e-6, rtol=2e-6)
        ambient += qa
    close(record.get("ambient_heat_j", 0.0), ambient, atol=1e-6, rtol=2e-6)
    if len(record["links"]) != len(model["links"]) or {r["name"] for r in record["links"]} != {
        l["name"] for l in model["links"]
    }:
        raise ValueError("Face registry differs")
    for row in record["links"]:
        expected = dt * weighted([pair[1][row["name"]] for pair in decoded])
        close(row["integrated_flux"], expected, atol=1e-8, rtol=2e-6)
    for link in model.get("heat_links", []):
        ua = link["conductance_w_k"] if commands["openings"][link["name"]] else 0.0
        expected = (
            dt
            * ua
            * sum(
                weight
                * (stage[link["source"]]["temperature_k"] - stage[link["target"]]["temperature_k"])
                for weight, stage in zip(weights, stages)
            )
        )
        close(record["heat_links"][link["name"]], expected, atol=1e-9, rtol=2e-6)
    external = dt * sum(commands.get("heat_w", {}).values())
    production = (
        sum(newdata[name]["entropy"] - olddata[name]["entropy"] for name in nodes) - incoming_s
    )
    for name, node in nodes.items():
        heat = commands.get("heat_w", {}).get(name, 0.0)
        if heat:
            utility = commands.get("utility_temperature_k", {}).get(name)
            if (
                utility is None
                or utility <= 0
                or any(heat * (utility - stage[name]["temperature_k"]) < -1e-10 for stage in stages)
            ):
                raise ValueError("Gauss utility direction rejected")
            production -= dt * heat / utility
        if node.get("wall"):
            w = node["wall"]
            if any(
                abs(s[name]["wall_temperature_k"] - old[name]["wall_temperature_k"])
                > w["max_temperature_change_k"]
                for s in [*stages, new]
            ):
                raise ValueError("Gauss wall change exceeds domain")
            production += w["heat_capacity_j_k"] * math.log(
                new[name]["wall_temperature_k"] / old[name]["wall_temperature_k"]
            )
            qa = dt * sum(
                weight
                * w["ambient_ua_w_k"]
                * (w["ambient_temperature_k"] - stage[name]["wall_temperature_k"])
                for weight, stage in zip(weights, stages)
            )
            production -= qa / w["ambient_temperature_k"]
    if production < -1e-6 - 1e-10 * max(1.0, sum(abs(d["entropy"]) for d in olddata.values())):
        raise ValueError("Independent Gauss entropy rejected")
    close(record["external_heat_j"], external, atol=1e-7, rtol=2e-7)
    close(record["entropy_production_j_k"], production, atol=2e-6, rtol=2e-6)
    close(
        record["conservation_residuals"],
        ledger(
            before,
            record["state"],
            nodes,
            ambient + external + incoming_e,
            model,
            incoming_n=incoming_n,
        ),
        atol=1e-6,
        rtol=2e-7,
    )
    return {
        "accepted": True,
        "stages": 2,
        "nodes": len(nodes),
        "faces": len(record["links"]),
        "material_component_moles": incoming_n,
        "material_energy_j": incoming_e,
        "ambient_heat_j": ambient,
        "external_heat_j": external,
        "entropy_production_j_k": production,
        "maximum_local_residuals": maximum.tolist(),
    }
