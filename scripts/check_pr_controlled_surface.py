"""Independent controller/actuator replay plus gravitational N/M/E audit.

Reconstructs measurements using NumPy PR and SciPy chemical-profile solves.
Runtime feedback, actuator integration and plant stepping are not invoked.
"""

import argparse
import json
import math
import sys
from pathlib import Path

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


def column_measurement(old, port, plant, controller):
    state = old["plant"]["columns"][port["column"]]
    system = plant["columns"][port["column"]]
    fluid, geo, sample, amounts, _, _ = profile(state, system)
    if controller["measurement"] == "pressure":
        phase = sum(port["height_m"] > h for h in state["interface_heights_m"])
        return sample(port["height_m"], phase)[0]
    index = fluid.names.index(controller["component"])
    fractions = amounts[:, index] / amounts.sum(-1)
    phase = int(fractions.argmax())
    if fractions[phase] < controller["minimum_mole_fraction"]:
        return 0.
    return state["interface_heights_m"][phase] if phase < len(state["interface_heights_m"]) else geo.height_m


def audit(config, report):
    if config != report["config"]:
        raise ValueError("Controlled request differs")
    parsed = PRControlledSurface.from_dict(config["model"])
    model = json.loads(json.dumps(parsed.to_dict()))
    if model != report["model"]:
        raise ValueError("Controlled model normalization differs")
    old = report["initial"]
    valves = {v["name"]: v for v in model["valves"]}
    controllers = {c["valve"]: c for c in model["controllers"]}
    plant = model["plant"]
    ports = {p["name"]: p for p in plant["ports"]}
    nodes = {n["name"]: n for n in plant["transport"]["nodes"]}
    links = {c["name"]: c for c in plant["transport"]["links"]}
    for valve in valves.values():
        if valve["port"] in ports:
            continue
        link = links[valve["port"]]
        candidates = [name for side, name in (("source", link["source"]), ("target", link["target"]))
                      if nodes[name]["kind"] == "pipe_cell"
                      and link[side + "_port"] in ("left", "right")
                      and not nodes[name].get("drift_flux")
                      and not nodes[name]["distribution_strength"]
                      and not nodes[name]["buoyancy_relaxation_s"]]
        body = valve.get("resistance_node") or (candidates[0] if len(candidates) == 1 else None)
        if body not in candidates:
            raise ValueError("Ambiguous material valve body")
        ports[valve["port"]] = {"column": None, "node": body,
                               "sensor_node": link["target"] if body == link["source"] else link["source"]}
    if old["model_id"] != parsed.model_id:
        raise ValueError("Controlled checkpoint identity differs")
    initial_plant = old["plant"]
    flat = []
    schedule = []
    checked = 0
    heat_total = 0.0
    work_total = 0.0
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
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
                atol=1e-13,
                rtol=1e-10,
            )
            manual = command.get("targets", {})
            modes = command.get("actuator_modes", {})
            targets = dict(manual)
            updated = dict(old["integrals"])
            for name, c in controllers.items():
                port = ports[valves[name]["port"]]
                if port["column"] is None:
                    cells = old["plant"]["transport"]["nodes"]
                    if c["measurement"] == "pressure":
                        value = cells[c.get("sensor") or port["sensor_node"]]["pressure_pa"]
                    else:
                        fluid = parsed.plant.transport.fluid
                        index = fluid.names.index(c["component"])
                        volume = 0.
                        path = plant["transport"]["pipe_paths"][c["sensor"]]
                        for node_name in path:
                            d = decode(cells[node_name], nodes[node_name], plant["transport"], fluid)
                            selected = d["x"][:, index] >= c["minimum_mole_fraction"]
                            volume += sum(d["pn"] * d["vm"] * selected)
                        value = volume / (math.pi * nodes[path[0]]["diameter_m"]**2 / 4)
                else:
                    value = column_measurement(old, port, plant, c)
                close(step["measurements"][name], value, atol=1e-7, rtol=2e-7)
                error = c["action"] * (value - c["setpoint"])
                proportional = c["proportional_gain"] * error
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
                    iv = equilibrium + (old["integrals"][name] - equilibrium) * math.exp(
                        -dt / c["tracking_time_s"]
                    )
                iv = min(c["maximum_integral_abs"], max(-c["maximum_integral_abs"], iv))
                updated[name] = iv
                if name not in manual:
                    targets[name] = min(1.0, max(0.0, c["opening_bias"] + proportional + iv))
            for name in valves:
                if name not in targets and modes.get(name) in ("fail_safe", "stuck"):
                    targets[name] = old["positions"][name]
            for name, iv in updated.items():
                close(new["integrals"][name], iv, atol=1e-10, rtol=1e-8)
            actual = {
                p["name"]: True for p in (
                    plant["transport"]["links"] + plant["transport"].get("heat_links", [])
                    + plant["ports"] + plant.get("heat_links", [])
                )
            }
            actual.update({p["card"]["name"]: p["card"]["open"] for p in plant["transport"].get("pumps", [])})
            actual.update({unit["name"]: unit["open"] for unit in plant["transport"].get("compressors", [])})
            actual.update(command.get("openings", {}))
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
                opened = actual[v["port"]] and x > 0
                actual[v["port"]] = opened
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
            if physical["state"] != new["plant"]:
                raise ValueError("Controlled/physical checkpoint differs")
            physical_command = {
                k: v for k, v in command.items() if k in ("heat_w", "utility_temperature_k")
            }
            physical_command.update(duration_s=dt, openings=actual, loss_coefficients=losses)
            schedule.append(physical_command)
            flat.append(physical)
            close(step["total_heat_j"], physical["total_heat_j"], atol=1e-6, rtol=2e-7)
            close(step.get("total_work_j", 0.0), physical.get("total_work_j", 0.0), atol=1e-10, rtol=2e-7)
            interval_heat += physical["total_heat_j"]
            interval_work += physical.get("total_work_j", 0.0)
            checked += 1
            old = new
        close(old["plant"]["transport"]["time_s"], end, atol=1e-13, rtol=0.0)
        x = json.loads(json.dumps(interval["state"]))
        y = json.loads(json.dumps(old))
        x["next_step_s"] = y["next_step_s"]
        if x != y:
            raise ValueError("Final controlled checkpoint differs")
        old = interval["state"]
        heat_total += interval_heat
        work_total += interval_work
        close(interval["total_heat_j"], interval_heat, atol=1e-6, rtol=2e-7)
        close(interval.get("total_work_j", 0.0), interval_work, atol=1e-10, rtol=2e-7)
    request = {"model": plant, "initial": config["initial"]["plant"], "schedule": schedule}
    physical_report = {
        "config": request,
        "model": plant,
        "initial": initial_plant,
        "intervals": flat,
        "total_heat_j": heat_total,
        "total_work_j": work_total,
        "conservation_residuals": report["conservation_residuals"],
    }
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
