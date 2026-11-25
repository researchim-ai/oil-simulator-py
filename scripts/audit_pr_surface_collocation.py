"""Independent NumPy/SciPy audit of common SDIRK2 controlled PR stages.

Reuses scalar EOS, chemical-profile, card and HLLC references, never runtime
decode, flux, sampling, inspection, residual or Newton. This checks numerical
identities and balances, not experimental closure calibration or global TPD.
"""

import json
import math

import numpy as np
from audit_pr_collocation import reference_rates
from audit_pr_collocation_machines import stage_rates, stored_energy
from check_pr_equilibrium_transport import decode, disk_area, endpoint
from check_pr_gravity_transport import correction
from check_pr_momentum_vessels import port_flux
from check_stratified_pr_flow import close, profile


def aperture_rates(plant, states, data, profiles, openings, order, nc):
    tm = plant["transport"]
    nodes = {n["name"]: n for n in tm["nodes"]}
    ordinary = {n: np.zeros(nc + 2) for n in nodes}
    columns = {n: np.zeros(nc + 1) for n in plant["columns"]}
    ports = {}
    gn, gw = np.polynomial.legendre.leggauss(order)
    for port in plant["ports"]:
        name, node = port["column"], port["node"]
        _, geo, sample, _, _, _ = profiles[name]
        state = states[name]
        bounds = np.r_[0.0, state["interface_heights_m"], geo.height_m]
        fake = {
            "diameter_m": port["diameter_m"],
            "target_port": port["pipe_port"],
            "target_height_m": None,
        }
        pe, _, normal, extra = endpoint(nodes[node], data[node], fake, "target", tm)
        radius = port["diameter_m"] / 2
        area = math.pi * radius**2
        total = np.zeros(nc + 2)
        for phase in range(len(state["phase_roots"])):
            low = max(bounds[phase], port["height_m"] - radius)
            high = max(low, min(bounds[phase + 1], port["height_m"] + radius))
            fraction = (
                disk_area(high - port["height_m"] + radius, port["diameter_m"])
                - disk_area(low - port["height_m"] + radius, port["diameter_m"])
            ) / area
            theta_a = math.asin(np.clip((low - port["height_m"]) / radius, -1, 1))
            theta_b = math.asin(np.clip((high - port["height_m"]) / radius, -1, 1))
            theta = (theta_a + theta_b) / 2 + (theta_b - theta_a) / 2 * gn
            raw = gw * np.cos(theta) ** 2
            weights = fraction * raw / max(raw.sum(), 1e-30)
            for height, weight in zip(port["height_m"] + radius * np.sin(theta), weights):
                if weight <= 0:
                    continue
                p, x, q = sample(height, phase)
                phi = tm["gravity_m_s2"] * (
                    plant["columns"][name]["elevation_m"] + height - tm["reference_elevation_m"]
                )
                uc = np.r_[x / q["vm"], 0.0, q["u"] / q["vm"] + q["rho"] * phi]
                ce = (uc, np.r_[np.zeros(nc), p, 0.0], p, 0.0, q["rho"], q["s"] / q["vm"])
                pu, pf, pp, pv, pd, ps = pe
                pu, pf = pu.copy(), pf.copy()
                du = tm["gravity_m_s2"] * (height - port["height_m"])
                pu[-1] += pd * du
                pf[-1] += pd * du * pv
                if openings[port["name"]]:
                    flux, *_ = port_flux(ce, (pu, pf, pp, pv, pd, ps), tm["wave_speed_bound_m_s"])
                    # The adapted runtime supports homogeneous cells only.
                    flux += correction(extra, False)[0]
                else:
                    d = data[node]
                    flux = np.r_[
                        np.zeros(nc),
                        d["p"]
                        + d["rho"] * d["u"] ** 2
                        + normal * tm["wave_speed_bound_m_s"] * d["rho"] * d["u"],
                        0.0,
                    ]
                total += area * weight * flux
        ordinary[node] += np.r_[total[:-2], -normal * total[-2], total[-1]]
        if openings[port["name"]]:
            columns[name] -= np.r_[total[:-2], total[-1]]
        ports[port["name"]] = total
    return ordinary, columns, ports


def audit(config, report):
    from surface.pr_surface_control import PRControlledSurface

    if config != report["config"]:
        raise ValueError("Common surface request differs")
    source = PRControlledSurface.from_dict(config["model"])
    parsed = getattr(source, "reference_surface", source)
    model = json.loads(json.dumps(parsed.to_dict()))
    plant, tm = model["plant"], model["plant"]["transport"]
    if plant.get("walls") or any(n.get("wall") for n in tm["nodes"]):
        raise ValueError("Common-stage wall-energy reference not yet adapted")
    if any(c.measurement == "inventory_level" for c in parsed.controllers):
        raise ValueError("Common-stage inventory-level reference not yet adapted")
    fluid = parsed.plant.transport.fluid
    nc = len(fluid.names)
    nodes = {n["name"]: n for n in tm["nodes"]}
    channels = parsed.channels
    controllers = {c.valve: c for c in parsed.controllers}
    gamma = 1 - 1 / math.sqrt(2)
    a = ((gamma, 0.0), (1 - gamma, gamma))
    weights = (1 - gamma, gamma)
    old = report["initial"]
    checked = 0

    def weighted(values):
        return sum(w * v for w, v in zip(weights, values))

    def node_vector(state, name, shaft):
        s = state[name]
        return np.r_[
            s["component_moles"], s["momentum_kg_m_s"], s["total_energy_j"] + shaft.get(name, 0.0)
        ]

    def column_vector(state):
        mass = sum(
            n * c.molar_mass_kg_mol for n, c in zip(state["component_moles"], fluid.components)
        )
        return np.r_[
            state["component_moles"],
            state["total_energy_j"] - mass * tm["gravity_m_s2"] * tm["reference_elevation_m"],
        ]

    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        for record in interval["steps"]:
            if record["method"] != "sdirk2-common":
                raise ValueError(
                    "This audit requires pure common stages; fallback needs the legacy audit"
                )
            step = record["plant"]
            if not 0 <= step["quadrature_error"] <= 1:
                raise ValueError("Missing spatial quadrature certificate")
            stages = step["transport"]["stage_states"]
            coupling = step["transport"]["stage_couplings"]
            rotors = step["transport"]["stage_rotors"]
            equipment = step["transport"]["stage_equipment"]
            dt = record["dt_s"]
            old_x, old_i = dict(old["positions"]), dict(old["integrals"])
            old_shaft = stored_energy(tm, dict(old["plant"]["transport"].get("rotors", [])))
            rates, column_rates, control_rates, work_rates, ambient_rates = [], [], [], [], []
            ports_at_stages, angular_at_stages = [], []
            for k, stage in enumerate(stages):
                c = coupling[k]["controls"]
                pos = dict(zip((v.name for v in channels), c["positions"]))
                vel = dict(zip((v.name for v in channels), c["velocities"]))
                ints = dict(zip((v.valve for v in parsed.controllers), c["integrals"]))
                profiles = {
                    name: profile(s, plant["columns"][name])
                    for name, s in coupling[k]["columns"].items()
                }
                data = {name: decode(stage[name], node, tm, fluid) for name, node in nodes.items()}
                openings = {
                    p["name"]: True
                    for p in [
                        *tm["links"],
                        *tm.get("heat_links", []),
                        *plant["ports"],
                        *plant.get("heat_links", []),
                    ]
                }
                openings.update(parsed.plant.transport.default_openings)
                openings.update(command.get("openings", {}))
                losses = {
                    n["name"]: command.get("loss_coefficients", {}).get(
                        n["name"], n["local_loss_coefficient"]
                    )
                    for n in tm["nodes"]
                    if n["kind"] == "pipe_cell"
                }
                torque = dict(command.get("drive_torque_nm", {}))
                integral_rates = {}
                for index, v in enumerate(channels):
                    spec = v.actuator
                    x = pos[v.name]
                    if not 0 <= x <= 1 or abs(x - old_x[v.name]) > spec.max_position_change * (
                        1 + 1e-10
                    ):
                        raise ValueError("Independent actuator domain/change rejected")
                    controller = controllers.get(v.name)
                    measurement = 0.0
                    if controller is not None:
                        if controller.measurement == "speed":
                            measurement = rotors[k][controller.sensor or v.port]
                        else:
                            port = parsed._ports[v.port]
                            if controller.measurement == "temperature":
                                measurement = data[controller.sensor or port.sensor_node]["t"]
                            elif controller.measurement == "pressure":
                                if port.column is None:
                                    measurement = data[controller.sensor or port.sensor_node]["p"]
                                else:
                                    s = coupling[k]["columns"][port.column]
                                    phase = sum(port.height_m > h for h in s["interface_heights_m"])
                                    measurement = profiles[port.column][2](port.height_m, phase)[0]
                            else:
                                name = controller.sensor or port.column
                                amounts = profiles[name][3]
                                fractions = amounts[
                                    :, fluid.names.index(controller.component)
                                ] / amounts.sum(-1)
                                phase = int(fractions.argmax())
                                bounds = [
                                    *coupling[k]["columns"][name]["interface_heights_m"],
                                    profiles[name][1].height_m,
                                ]
                                measurement = (
                                    bounds[phase]
                                    if fractions[phase] >= controller.minimum_mole_fraction
                                    else 0.0
                                )
                        proportional = (
                            controller.proportional_gain
                            * controller.action
                            * (measurement - controller.setpoint)
                        )
                        if v.name in command.get("targets", {}):
                            target = command["targets"][v.name]
                            close(
                                ints[v.name],
                                np.clip(
                                    target - controller.opening_bias - proportional,
                                    -controller.maximum_integral_abs,
                                    controller.maximum_integral_abs,
                                ),
                                atol=1e-9,
                                rtol=1e-8,
                            )
                        else:
                            target = np.clip(
                                controller.opening_bias + proportional + ints[v.name], 0, 1
                            )
                            derivative = (
                                controller.integral_gain_per_s
                                * controller.action
                                * (measurement - controller.setpoint)
                                + (x - controller.opening_bias - proportional - ints[v.name])
                                / controller.tracking_time_s
                            )
                            if (
                                ints[v.name] >= controller.maximum_integral_abs and derivative > 0
                            ) or (
                                ints[v.name] <= -controller.maximum_integral_abs and derivative < 0
                            ):
                                derivative = 0.0
                            integral_rates[v.name] = derivative
                    else:
                        target = command.get("targets", {}).get(v.name, old_x[v.name])
                    mode = command.get("actuator_modes", {}).get(v.name, "normal")
                    if mode == "fail_safe":
                        target = spec.fail_position
                    actual_v = vel[v.name]
                    expected_v = (
                        0.0
                        if mode == "stuck"
                        else np.clip(
                            (target - x) / spec.time_constant_s,
                            -spec.closing_rate_per_s,
                            spec.opening_rate_per_s,
                        )
                        if spec.time_constant_s
                        else np.clip(
                            actual_v + target - x, -spec.closing_rate_per_s, spec.opening_rate_per_s
                        )
                    )
                    close(actual_v, expected_v, atol=1e-9, rtol=1e-8)
                    close(
                        c["measurements"][index],
                        measurement,
                        atol=0.01 if controller and controller.measurement == "pressure" else 1e-8,
                        rtol=2e-7,
                    )
                    close(c["targets"][index], target, atol=1e-9, rtol=2e-7)
                    if hasattr(v, "machine"):
                        torque[v.machine] = v.minimum_torque_nm + x * (
                            v.maximum_torque_nm - v.minimum_torque_nm
                        )
                    elif openings[v.port]:
                        fraction = (
                            x
                            if v.characteristic == "linear"
                            else math.expm1(x * math.log(v.rangeability)) / (v.rangeability - 1)
                        )
                        losses[parsed._ports[v.port].node] += (
                            v.full_open_loss_coefficient / fraction**2
                        )
                live = {
                    **command,
                    "openings": openings,
                    "loss_coefficients": losses,
                    "drive_torque_nm": torque,
                }
                net, _, _ = reference_rates(
                    tm,
                    stage,
                    fluid,
                    live,
                    external_ports=[(p["node"], p["pipe_port"]) for p in plant["ports"]],
                )
                machinery = stage_rates(tm, data, rotors[k], equipment[k], fluid, live)
                pn, cn, ports = aperture_rates(
                    plant,
                    coupling[k]["columns"],
                    data,
                    profiles,
                    openings,
                    step["quadrature_order"],
                    nc,
                )
                ports_at_stages.append(ports)
                angular_at_stages.append(machinery[1])
                for name in net:
                    net[name] += machinery[0][name] + pn[name]
                temperatures = {
                    **{name: d["t"] for name, d in data.items()},
                    **{name: s["temperature_k"] for name, s in coupling[k]["columns"].items()},
                }
                for link in plant.get("heat_links", []):
                    q = (
                        link["conductance_w_k"]
                        * (temperatures[link["source"]] - temperatures[link["target"]])
                        if openings[link["name"]]
                        else 0.0
                    )
                    for name, sign in ((link["source"], -1), (link["target"], 1)):
                        (cn[name] if name in cn else net[name])[-1] += sign * q
                for name in cn:
                    cn[name][-1] += command.get("heat_w", {}).get(name, 0.0)
                rates.append(net)
                column_rates.append(cn)
                control_rates.append((vel, integral_rates))
                work_rates.append(machinery[2])
                ambient_rates.append(machinery[3])
            for k, stage in enumerate(stages):
                for name in angular_at_stages[k]:
                    close(
                        rotors[k][name] - dict(old["plant"]["transport"]["rotors"])[name],
                        dt * sum(a[k][j] * angular_at_stages[j][name] for j in range(2)),
                        atol=1e-8,
                        rtol=2e-6,
                    )
                shaft = stored_energy(tm, rotors[k])
                for name, node in nodes.items():
                    expected = dt * sum(a[k][j] * rates[j][name] for j in range(2))
                    if node["kind"] == "vessel":
                        expected[-2] = 0.0
                    delta = node_vector(stage, name, shaft) - node_vector(
                        old["plant"]["transport"]["nodes"], name, old_shaft
                    )
                    rounding = 32 * np.spacing(
                        np.maximum(
                            abs(node_vector(stage, name, shaft)),
                            abs(node_vector(old["plant"]["transport"]["nodes"], name, old_shaft)),
                        )
                    )
                    close(
                        delta,
                        expected,
                        atol=np.r_[np.full(nc, 1e-9), 1e-8, 2e-6] + rounding,
                        rtol=3e-6,
                    )
                for name in column_rates[k]:
                    delta = column_vector(coupling[k]["columns"][name]) - column_vector(
                        old["plant"]["columns"][name]
                    )
                    expected = dt * sum(a[k][j] * column_rates[j][name] for j in range(2))
                    close(delta, expected, atol=2e-6, rtol=3e-6)
                for index, v in enumerate(channels):
                    close(
                        coupling[k]["controls"]["positions"][index] - old_x[v.name],
                        dt * sum(a[k][j] * control_rates[j][0][v.name] for j in range(2)),
                        atol=1e-9,
                        rtol=2e-6,
                    )
                for index, c in enumerate(parsed.controllers):
                    if c.valve in control_rates[k][1]:
                        close(
                            coupling[k]["controls"]["integrals"][index] - old_i[c.valve],
                            dt * sum(a[k][j] * control_rates[j][1][c.valve] for j in range(2)),
                            atol=1e-9,
                            rtol=2e-6,
                        )
            close(record["total_work_j"], dt * weighted(work_rates), atol=1e-6, rtol=2e-6)
            for name, saved in step["columns"]["ports"].items():
                expected = weighted([p[name] for p in ports_at_stages])
                close(saved["flux"], expected, atol=1e-6, rtol=2e-6)
                close(saved["integrated_flux"], dt * expected, atol=1e-7, rtol=2e-6)
            close(
                record["total_heat_j"],
                dt * (weighted(ambient_rates) + sum(command.get("heat_w", {}).values())),
                atol=1e-6,
                rtol=2e-6,
            )
            final = record["state"]
            final_rotors = dict(final["plant"]["transport"].get("rotors", []))
            final_shaft = stored_energy(tm, final_rotors)
            stage_shaft = stored_energy(tm, rotors[-1])
            for name in nodes:
                close(
                    node_vector(final["plant"]["transport"]["nodes"], name, final_shaft),
                    node_vector(stages[-1], name, stage_shaft),
                    atol=2e-6,
                    rtol=1e-10,
                )
            for name in plant["columns"]:
                close(
                    column_vector(final["plant"]["columns"][name]),
                    column_vector(coupling[-1]["columns"][name]),
                    atol=2e-6,
                    rtol=1e-10,
                )
            close(list(final_rotors.values()), list(rotors[-1].values()), atol=1e-9, rtol=1e-10)
            close(
                [dict(final["positions"])[v.name] for v in channels],
                coupling[-1]["controls"]["positions"],
                atol=1e-10,
                rtol=1e-10,
            )
            close(
                [dict(final["integrals"])[c.valve] for c in parsed.controllers],
                coupling[-1]["controls"]["integrals"],
                atol=1e-10,
                rtol=1e-10,
            )
            close(
                final["plant"]["transport"]["time_s"] - old["plant"]["transport"]["time_s"],
                dt,
                atol=1e-14,
                rtol=1e-12,
            )
            old = record["state"]
            checked += 1
    return {
        "accepted": True,
        "steps": checked,
        "scope": "independent staged EOS/profile, HLLC, cards, PI, ideal limiter and N/E equations; no walls/fallback or experimental calibration",
    }
