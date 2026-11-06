"""Independent controller/actuator replay plus gravitational N/M/E audit.

Reconstructs measurements using NumPy PR and SciPy chemical-profile solves.
Runtime feedback, actuator integration and plant stepping are not invoked.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from check_pr_equilibrium_transport import decode
from check_pr_gravity_transport import audit as audit_plant
from check_stratified_pr_flow import close, profile

from surface.pr_surface_control import PRControlledSurface


def actuator(position, target, dt, spec, mode):
    target = spec["fail_position"] if mode == "fail_safe" else target
    if mode == "stuck" or target == position:
        return position
    gap = abs(target - position)
    rate = spec["opening_rate_per_s"] if target > position else spec["closing_rate_per_s"]
    tau = spec["time_constant_s"]
    if tau == 0:
        movement = min(gap, rate * dt)
    else:
        switch = max(0.0, gap / rate - tau)
        if dt <= switch:
            movement = rate * dt
        else:
            remaining = gap - rate * switch
            movement = gap - remaining * math.exp(-(dt - switch) / tau)
    return min(1.0, max(0.0, position + math.copysign(movement, target - position)))


def column_specs(plant):
    return {
        **plant.get("columns", {}),
        **{
            f"profile:{n}:{i}": col
            for n, spec in plant.get("profiles", {}).items()
            for i, col in enumerate(spec["pipe"]["columns"])
        },
    }


def column_measurement(old, port, plant, controller):
    name = port["column"]
    system = column_specs(plant)[name]
    if name in plant.get("columns", {}):
        state = old["plant"]["columns"][name]
    else:
        state = {
            f"profile:{n}:{i}": cell["profile"]
            for n, s in old["plant"]["profiles"].items()
            for i, cell in enumerate(s["cells"])
        }[name]
    fluid, geo, sample, amounts, _, _ = profile(state, system)
    if controller["measurement"] == "pressure":
        phase = sum(port["height_m"] > h for h in state["interface_heights_m"])
        return sample(port["height_m"], phase)[0]
    index = fluid.names.index(controller["component"])
    fractions = amounts[:, index] / amounts.sum(-1)
    phase = int(fractions.argmax())
    if fractions[phase] < controller["minimum_mole_fraction"]:
        return 0.0
    return (
        state["interface_heights_m"][phase]
        if phase < len(state["interface_heights_m"])
        else geo.height_m
    )


def audit(config, report):
    if config != report["config"]:
        raise ValueError("Controlled request differs")
    parsed = PRControlledSurface.from_dict(config["model"])
    model = json.loads(json.dumps(parsed.to_dict()))
    if model != report["model"]:
        raise ValueError("Controlled model normalization differs")
    old = report["initial"]
    if set(config["initial"]) == {"state"} and config["initial"]["state"] != old:
        raise ValueError("Requested controlled restart checkpoint differs")
    valves = {v["name"]: v for v in model["valves"]}
    valves.update({m["name"]: {**m, "port": m["machine"]} for m in model.get("motors", [])})
    trips = {t["name"]: t for t in model.get("trips", [])}
    controllers = {c["valve"]: c for c in model["controllers"]}
    plant = model["plant"]
    is_profile = plant["model"] in ("pr_profile_graph_v1", "pr_profile_graph_v2")
    endpoint_switches = {
        f"profile:{n}:{side}": (n, i)
        for n, spec in plant.get("profiles", {}).items()
        for i, side in enumerate(("left", "right"))
        if plant["model"] == "pr_profile_graph_v2" and spec["boundary_nodes"][i] is not None
    }
    ports = {p["name"]: p for p in plant.get("ports", [])}
    nodes = {n["name"]: n for n in plant["transport"]["nodes"]}
    links = {c["name"]: c for c in plant["transport"]["links"]}
    restrictions = {u["name"]: u for u in plant["transport"].get("restrictions", [])}
    machines = {
        p["card"]["name"]: p
        for p in [
            *plant["transport"].get("pumps", []),
            *plant["transport"].get("pump_connections", []),
        ]
    }
    machines.update({p["name"]: p for p in plant["transport"].get("compressors", [])})
    for valve in valves.values():
        if valve.get("model") == "motor_torque":
            unit = machines[valve["machine"]]
            node = unit.get("node") or unit.get("target") or unit["card"]["end"]
            ports[valve["port"]] = {"column": None, "node": node, "sensor_node": node}
            continue
        if valve.get("model") == "hem_throat_area":
            unit = restrictions[valve["port"]]
            ports[valve["port"]] = {
                "column": None,
                "node": unit["target"],
                "sensor_node": unit["target"],
                "restriction": True,
            }
            continue
        if valve["port"] in ports:
            continue
        link = links[valve["port"]]
        candidates = [
            name
            for side, name in (("source", link["source"]), ("target", link["target"]))
            if nodes[name]["kind"] == "pipe_cell"
            and link[side + "_port"] in ("left", "right")
            and not nodes[name].get("drift_flux")
            and not nodes[name]["distribution_strength"]
            and not nodes[name]["buoyancy_relaxation_s"]
        ]
        body = valve.get("resistance_node") or (candidates[0] if len(candidates) == 1 else None)
        if body not in candidates:
            raise ValueError("Ambiguous material valve body")
        ports[valve["port"]] = {
            "column": None,
            "node": body,
            "sensor_node": link["target"] if body == link["source"] else link["source"],
        }
    if old["model_id"] != parsed.model_id:
        raise ValueError("Controlled checkpoint identity differs")
    if set(config["initial"]) != {"state"}:
        expected_integrals = {name: 0.0 for name in controllers}
        expected_integrals.update(config["initial"]["integrals"])
        if (
            old["positions"] != config["initial"]["positions"]
            or old["integrals"] != expected_integrals
        ):
            raise ValueError("Requested initial controller state differs")
        if old.get("machine_feedback", {}) or old.get("inactive_feedback", []) != sorted(
            name for name, c in controllers.items() if c["measurement"] == "pump_flow"
        ):
            raise ValueError("Cold controller initialization fabricated feedback")
        if old.get("trips", {}) != {name: {"latched": False, "elapsed_s": 0.0} for name in trips}:
            raise ValueError("Cold protection state differs")
    initial_plant = old["plant"]
    flat = []
    schedule = []
    checked = 0
    heat_total = 0.0
    work_total = 0.0

    def compressor_measurement(state, name):
        unit = machines[name]
        card = unit["card"]
        transport = state["plant"]["transport"]
        source, target = transport["nodes"][unit["source"]], transport["nodes"][unit["target"]]
        speed = dict(transport.get("rotors", [])).get(name, unit["speed_rad_s"]) / math.sqrt(
            source["temperature_k"] / card["reference_temperature_k"]
        )
        axis = np.array(card["corrected_speed_rad_s"])
        if not axis[0] - 1e-12 <= speed <= axis[-1] + 1e-12:
            raise ValueError("Compressor feedback speed outside map")
        flow = np.array(
            [
                np.interp(speed, axis, column)
                for column in np.array(card["corrected_mass_flow_kg_s"]).T
            ]
        )
        ratios = np.array(
            [np.interp(speed, axis, column) for column in np.array(card["pressure_ratio"]).T]
        )
        requested = target["pressure_pa"] / source["pressure_pa"]
        if requested > ratios[0] + 1e-12:
            raise ValueError("Compressor feedback pressure outside stable map")
        return float(np.interp(requested, ratios[::-1], flow[::-1]) / flow[0] - 1)

    def trip_value(state, trip):
        transport = state["plant"]["transport"]
        if trip["measurement"] == "speed":
            return dict(transport["rotors"])[trip["sensor"]]
        return transport["nodes"][trip["sensor"]][
            "pressure_pa" if trip["measurement"] == "pressure" else "temperature_k"
        ]

    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        resets = command.get("reset_trips", [])
        if len(set(resets)) != len(resets) or set(resets) - set(trips):
            raise ValueError("Invalid protection reset registry")
        if resets:
            old = json.loads(json.dumps(old))
            for name in resets:
                t = trips[name]
                value = trip_value(old, t)
                allowed = (
                    value <= t["reset_threshold"]
                    if t["direction"] == "high"
                    else value >= t["reset_threshold"]
                )
                if not allowed:
                    raise ValueError("Unsafe protection reset")
                old["trips"][name] = {"latched": False, "elapsed_s": 0.0}
        end = old["plant"]["transport"]["time_s"] + command["duration_s"]
        interval_heat = 0.0
        interval_work = 0.0
        for step in interval["steps"]:
            new = step["state"]
            dt = step["dt_s"]
            if new["model_id"] != parsed.model_id or dt <= 0:
                raise ValueError("Invalid controlled checkpoint")
            close(
                new["plant"]["transport"]["time_s"] - old["plant"]["transport"]["time_s"],
                dt,
                atol=8
                * max(
                    math.ulp(new["plant"]["transport"]["time_s"]),
                    math.ulp(old["plant"]["transport"]["time_s"]),
                ),
                rtol=0.0,
            )
            manual = command.get("targets", {})
            modes = dict(command.get("actuator_modes", {}))
            isolation, forced = set(), set()
            if (
                set(old.get("trips", {})) != set(trips)
                or set(new.get("trips", {})) != set(trips)
                or set(step.get("protections", {})) != set(trips)
            ):
                raise ValueError("Protection state/report registry differs")
            for name, t in trips.items():
                prior = old["trips"][name]
                value = trip_value(old, t)
                unsafe = (
                    value >= t["trip_threshold"]
                    if t["direction"] == "high"
                    else value <= t["trip_threshold"]
                )
                active = prior["latched"] or (unsafe and t["delay_s"] == 0)
                if prior["latched"]:
                    latched, timer = True, prior["elapsed_s"]
                elif not unsafe:
                    latched, timer = False, 0.0
                else:
                    timer = min(t["delay_s"], math.fsum((prior["elapsed_s"], dt)))
                    latched = timer >= t["delay_s"] or math.isclose(
                        timer, t["delay_s"], rel_tol=1e-13, abs_tol=1e-15
                    )
                    if latched:
                        timer = t["delay_s"]
                if new["trips"][name]["latched"] is not latched:
                    raise ValueError("Protection latch differs")
                close(new["trips"][name]["elapsed_s"], timer, atol=1e-14, rtol=1e-10)
                record = step["protections"][name]
                close(record["measurement"], value, atol=1e-8, rtol=2e-7)
                close(record["elapsed_before_s"], prior["elapsed_s"], atol=1e-14, rtol=1e-10)
                close(record["elapsed_s"], timer, atol=1e-14, rtol=1e-10)
                if (
                    record["active"] is not active
                    or record["latched"] is not latched
                    or record["latched_before"] is not prior["latched"]
                ):
                    raise ValueError("Protection transition differs")
                blocked = active and modes.get(t["target"]) == "stuck"
                if record["actuator_blocked"] is not blocked:
                    raise ValueError("Stuck protection actuator differs")
                if active:
                    modes[t["target"]] = "stuck" if blocked else "fail_safe"
                    isolation.update(t["isolate_switches"])
                    forced.add(t["target"])
            targets = dict(manual)
            updated = dict(old["integrals"])
            inactive = []
            for name, c in controllers.items():
                port = ports[valves[name]["port"]]
                if c["measurement"] == "pump_flow":
                    unit = machines[c["sensor"]]
                    enabled = (
                        command.get("openings", {}).get(c["sensor"], unit["card"]["open"])
                        and c["sensor"] not in isolation
                    )
                    if not enabled or c["sensor"] not in old.get("machine_feedback", {}):
                        if step["measurements"][name] is not None:
                            raise ValueError("Unavailable pump has a fabricated flow measurement")
                        targets[name] = manual.get(name, old["positions"][name])
                        inactive.append(name)
                        continue
                    value = old["machine_feedback"][c["sensor"]]
                elif c["measurement"] == "compressor_margin":
                    enabled = (
                        command.get("openings", {}).get(c["sensor"], machines[c["sensor"]]["open"])
                        and c["sensor"] not in isolation
                    )
                    if not enabled:
                        if step["measurements"][name] is not None:
                            raise ValueError(
                                "Disabled compressor has a fabricated feedback reading"
                            )
                        targets[name] = manual.get(name, old["positions"][name])
                        inactive.append(name)
                        continue
                    value = compressor_measurement(old, c["sensor"])
                elif c["measurement"] == "speed":
                    value = dict(old["plant"]["transport"]["rotors"])[
                        c.get("sensor") or valves[name]["port"]
                    ]
                elif c["measurement"] == "temperature":
                    value = old["plant"]["transport"]["nodes"][
                        c.get("sensor") or port["sensor_node"]
                    ]["temperature_k"]
                elif c["measurement"] == "level" and c.get("sensor") in column_specs(plant):
                    value = column_measurement(old, {"column": c["sensor"]}, plant, c)
                elif port["column"] is None:
                    cells = old["plant"]["transport"]["nodes"]
                    if c["measurement"] == "pressure":
                        value = cells[c.get("sensor") or port["sensor_node"]]["pressure_pa"]
                    else:
                        fluid = parsed.plant.transport.fluid
                        index = fluid.names.index(c["component"])
                        volume = 0.0
                        path = plant["transport"]["pipe_paths"][c["sensor"]]
                        for node_name in path:
                            d = decode(
                                cells[node_name], nodes[node_name], plant["transport"], fluid
                            )
                            selected = d["x"][:, index] >= c["minimum_mole_fraction"]
                            volume += sum(d["pn"] * d["vm"] * selected)
                        value = volume / (math.pi * nodes[path[0]]["diameter_m"] ** 2 / 4)
                else:
                    value = column_measurement(old, port, plant, c)
                close(step["measurements"][name], value, atol=1e-7, rtol=2e-7)
                error = c["action"] * (value - c["setpoint"])
                proportional = c["proportional_gain"] * error
                if name in old.get("inactive_feedback", []):
                    updated[name] = min(
                        c["maximum_integral_abs"],
                        max(
                            -c["maximum_integral_abs"],
                            old["positions"][name] - c["opening_bias"] - proportional,
                        ),
                    )
                if name in manual:
                    targets[name] = manual[name]
                    iv = manual[name] - c["opening_bias"] - proportional
                else:
                    equilibrium = (
                        c["tracking_time_s"] * c["integral_gain_per_s"] * error
                        + old["positions"][name]
                        - c["opening_bias"]
                        - proportional
                    )
                    iv = equilibrium + (updated[name] - equilibrium) * math.exp(
                        -dt / c["tracking_time_s"]
                    )
                iv = min(c["maximum_integral_abs"], max(-c["maximum_integral_abs"], iv))
                updated[name] = iv
                if name not in manual:
                    targets[name] = min(1.0, max(0.0, c["opening_bias"] + proportional + iv))
            if new.get("inactive_feedback", []) != sorted(inactive):
                raise ValueError("Inactive controller state differs")
            for name, channel in valves.items():
                if name not in targets and (
                    channel.get("model") == "motor_torque"
                    or modes.get(name) in ("fail_safe", "stuck")
                ):
                    targets[name] = old["positions"][name]
            actual = {
                p["name"]: True
                for p in (
                    plant["transport"]["links"]
                    + plant["transport"].get("heat_links", [])
                    + plant.get("ports", [])
                    + plant.get("heat_links", [])
                )
            }
            actual.update(
                {
                    p["card"]["name"]: p["card"]["open"]
                    for p in [
                        *plant["transport"].get("pumps", []),
                        *plant["transport"].get("pump_connections", []),
                    ]
                }
            )
            actual.update(
                {unit["name"]: unit["open"] for unit in plant["transport"].get("compressors", [])}
            )
            actual.update({u["name"]: u["open"] for u in restrictions.values()})
            for switch, (branch, i) in endpoint_switches.items():
                actual[switch] = (
                    old["plant"].get("profile_openings", {}).get(branch, (True, True))[i]
                )
            branch_commands = {n: list(v) for n, v in command.get("profile_openings", {}).items()}
            for switch, (branch, i) in endpoint_switches.items():
                if branch in branch_commands:
                    actual[switch] = branch_commands[branch][i]
            for switch in isolation:
                if switch in endpoint_switches:
                    branch, i = endpoint_switches[switch]
                    if branch in branch_commands:
                        branch_commands[branch][i] = False
            actual.update(command.get("openings", {}))
            actual.update({name: False for name in isolation})
            torques = dict(command.get("drive_torque_nm", {}))
            if set(torques) & {m["machine"] for m in model.get("motors", [])}:
                raise ValueError("Conflicting controlled torque command")
            areas = dict(command.get("restriction_openings", {}))
            losses = {
                n["name"]: n["local_loss_coefficient"]
                for n in plant["transport"]["nodes"]
                if n["kind"] == "pipe_cell"
            }
            losses.update(command.get("loss_coefficients", {}))
            for name, v in valves.items():
                mode = modes.get(name, "normal")
                x = actuator(old["positions"][name], targets[name], dt, v["actuator"], mode)
                close(new["positions"][name], x, atol=1e-12, rtol=1e-10)
                saved = step["actuators"][name]
                close(saved["command"], targets[name], atol=1e-12, rtol=1e-10)
                if saved["mode"] != mode:
                    raise ValueError("Actuator mode differs")
                close(saved["initial_position"], old["positions"][name], atol=1e-12, rtol=1e-10)
                close(saved["final_position"], x, atol=1e-12, rtol=1e-10)
                if v.get("model") == "motor_torque":
                    torque = v["minimum_torque_nm"] + x * (
                        v["maximum_torque_nm"] - v["minimum_torque_nm"]
                    )
                    close(step["effective_motor_torque_nm"][name], torque, atol=1e-12, rtol=1e-10)
                    torques[v["machine"]] = torque
                    peak = max(
                        [abs(dict(old["plant"]["transport"]["rotors"])[v["machine"]])]
                        + [
                            abs(dict(s["state"]["transport"]["rotors"])[v["machine"]])
                            for s in step["plant"]["steps"]
                        ]
                    )
                    close(
                        step["motor_envelope"][name]["maximum_speed_rad_s"],
                        peak,
                        atol=1e-12,
                        rtol=1e-10,
                    )
                    close(
                        step["motor_envelope"][name]["motor_torque_nm"],
                        torque,
                        atol=1e-12,
                        rtol=1e-10,
                    )
                    continue
                opened = actual[v["port"]] and x > 0
                actual[v["port"]] = opened
                if v.get("model") == "hem_throat_area":
                    if v["port"] in command.get("restriction_openings", {}):
                        raise ValueError("Conflicting controlled throat command")
                    fraction = (
                        x
                        if v["characteristic"] == "linear"
                        else (v["rangeability"] ** x - 1) / (v["rangeability"] - 1)
                    )
                    area = restrictions[v["port"]]["opening"] * fraction if opened else 0.0
                    close(step["effective_throat_openings"][name], area, atol=1e-12, rtol=1e-10)
                    areas[v["port"]] = area
                    if step["effective_valve_K"][name] is not None:
                        raise ValueError("Critical throat cannot also apply a K resistance")
                    continue
                if opened:
                    area = (
                        x
                        if v["characteristic"] == "linear"
                        else (v["rangeability"] ** x - 1) / (v["rangeability"] - 1)
                    )
                    k = v["full_open_loss_coefficient"] / area**2
                    close(step["effective_valve_K"][name], k, atol=1e-10, rtol=1e-8)
                    losses[ports[v["port"]]["node"]] += k
                elif step["effective_valve_K"][name] is not None:
                    raise ValueError("Isolated valve cannot apply a body resistance")
            for name, c in controllers.items():
                if name in forced and step["measurements"][name] is not None:
                    proportional = (
                        c["proportional_gain"]
                        * c["action"]
                        * (step["measurements"][name] - c["setpoint"])
                    )
                    updated[name] = min(
                        c["maximum_integral_abs"],
                        max(
                            -c["maximum_integral_abs"],
                            new["positions"][name] - c["opening_bias"] - proportional,
                        ),
                    )
            for name, iv in updated.items():
                close(new["integrals"][name], iv, atol=1e-10, rtol=1e-8)
            physical = step["plant"]
            if physical["openings"] != actual:
                raise ValueError("Valve isolation not applied to plant")
            if set(physical["loss_coefficients"]) != set(losses):
                raise ValueError("Valve resistance registry differs")
            for node, expected in losses.items():
                # Independently reconstructed PR inventories and actuator
                # targets differ by roundoff. Do not demand bitwise equality
                # of K(x) evaluated from those independently computed states.
                close(physical["loss_coefficients"][node], expected, atol=1e-10, rtol=1e-10)
            sensors = {c["sensor"] for c in controllers.values() if c["measurement"] == "pump_flow"}
            if set(new.get("machine_feedback", {})) != sensors:
                raise ValueError("Pump feedback registry differs")
            for sensor in sensors:
                measured = physical["steps"][-1]["transport"]["pump_connections"][sensor][
                    "flow_m3_s"
                ]
                close(new["machine_feedback"][sensor], measured, atol=1e-12, rtol=1e-10)
            if physical["state"] != new["plant"]:
                raise ValueError("Controlled/physical checkpoint differs")
            physical_command = {
                k: v
                for k, v in command.items()
                if k
                in (
                    "heat_w",
                    "utility_temperature_k",
                    "drive_torque_nm",
                    "restriction_openings",
                    "profile_heat_w",
                    "profile_utility_temperature_k",
                    "profile_openings",
                )
            }
            if branch_commands:
                physical_command["profile_openings"] = branch_commands
            physical_command.update(duration_s=dt, openings=actual, loss_coefficients=losses)
            if areas:
                physical_command["restriction_openings"] = areas
            if torques:
                physical_command["drive_torque_nm"] = torques
            schedule.append(physical_command)
            flat.append(physical)
            close(step["total_heat_j"], physical["total_heat_j"], atol=1e-6, rtol=2e-7)
            close(
                step.get("total_work_j", 0.0),
                physical.get("total_work_j", 0.0),
                atol=1e-10,
                rtol=2e-7,
            )
            interval_heat += physical["total_heat_j"]
            interval_work += physical.get("total_work_j", 0.0)
            checked += 1
            old = new
        close(old["plant"]["transport"]["time_s"], end, atol=8 * math.ulp(end), rtol=0.0)
        x = json.loads(json.dumps(interval["state"]))
        y = json.loads(json.dumps(old))
        x["next_step_s"] = y["next_step_s"]
        # The report endpoint is canonically snapped to the requested time.
        # A summed final half-step can differ by one ULP; all other state
        # coordinates must still match exactly.
        close(
            x["plant"]["transport"]["time_s"],
            y["plant"]["transport"]["time_s"],
            atol=8 * math.ulp(end),
            rtol=0.0,
        )
        x["plant"]["transport"]["time_s"] = y["plant"]["transport"]["time_s"]
        if x != y:
            raise ValueError("Final controlled checkpoint differs")
        old = interval["state"]
        heat_total += interval_heat
        work_total += interval_work
        close(interval["total_heat_j"], interval_heat, atol=1e-6, rtol=2e-7)
        close(interval.get("total_work_j", 0.0), interval_work, atol=1e-10, rtol=2e-7)
    request = {
        "model": plant,
        "initial": {"state": config["initial"]["state"]["plant"]}
        if set(config["initial"]) == {"state"}
        else config["initial"]["plant"],
        "schedule": schedule,
    }
    physical_report = {
        "config": request,
        "model": plant,
        "initial": initial_plant,
        "intervals": flat,
        "total_heat_j": heat_total,
        "total_work_j": work_total,
        "conservation_residuals": report["conservation_residuals"],
    }
    if is_profile:
        from check_pr_profile_graph import audit as audit_profile_graph

        physical_report["state"] = old["plant"]
        result = audit_profile_graph(request, physical_report)
    else:
        result = audit_plant(request, physical_report)
    close(report["total_heat_j"], heat_total, atol=1e-6, rtol=2e-7)
    close(report.get("total_work_j", 0.0), work_total, atol=1e-10, rtol=2e-7)
    return {**result, "checked_control_substeps": checked}


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
