"""Independent NumPy PR reconstruction of explicit pipe/equipment interfaces.

Does not call runtime flux, UV recovery or equipment residual implementations.
"""

import json

import numpy as np
from check_pr_equilibrium_transport import decode as decode_node
from check_stratified_pr_flow import close
from reference_pr_interphase_relaxation import decode as decode_cell


def interface_exchange(model, before, pipe_name, packet):
    """Reconstruct faces and opposite vessel loads from the old snapshot."""
    entry = model.pipes[pipe_name]
    pipe = entry["pipe"]
    nc = len(model.fluid.names)
    graph = dict(before.transport.nodes)
    cells = dict(before.pipes)[pipe_name]
    faces, entropy_faces = [None, None], [0.0, 0.0]
    net = {name: np.zeros(nc + 2) for name in model.transport.names}
    incoming_entropy = 0.0
    flux_roundoff = [None, None]
    for side, name in enumerate(entry["boundary_nodes"]):
        if name is None:
            continue
        node = model.transport.nodes[model.transport.names.index(name)]
        normalized = json.loads(json.dumps(model.transport.to_dict()))
        node_spec = next(v for v in normalized["nodes"] if v["name"] == name)
        d = decode_node(
            before.transport.to_dict()["nodes"][name], node_spec, normalized, model.fluid
        )
        vessel = np.r_[d["n"] / node.volume_m3, 0.0, d["internal"] / node.volume_m3]
        vessel_flux = np.zeros(nc + 2)
        vessel_flux[-2] = graph[name].pressure_pa
        i = 0 if side == 0 else -1
        cell, cell_flux = packet["conserved"][i], packet["flux"][i]
        data = decode_cell(model.fluid, cells.cells[i].to_dict(), cells.cells[i].phase_roots)
        cell_entropy = sum(q["entropy"] for q in data) / pipe.volumes[i]
        velocity = data[0]["velocity"]
        vessel_entropy = d["entropy"] / node.volume_m3
        left, right, lf, rf = (
            (vessel, cell, vessel_flux, cell_flux)
            if side == 0
            else (cell, vessel, cell_flux, vessel_flux)
        )
        faces[side] = 0.5 * (lf + rf) - 0.5 * pipe.wave_speed * (right - left)
        flux_roundoff[side] = (
            64
            * np.finfo(float).eps
            * (np.maximum(abs(lf), abs(rf)) + pipe.wave_speed * np.maximum(abs(left), abs(right)))
        )
        sl, sr = (vessel_entropy, cell_entropy) if side == 0 else (cell_entropy, vessel_entropy)
        entropy_faces[side] = 0.5 * cell_entropy * velocity - 0.5 * pipe.wave_speed * (sr - sl)
        sign = 1.0 if side == 0 else -1.0
        flow = -sign * pipe.area * faces[side]
        flow[-2] = 0.0
        net[name] += flow
        incoming_entropy -= sign * pipe.area * entropy_faces[side]
    return {
        "flux_roundoff": flux_roundoff,
        "boundary_fluxes": faces,
        "boundary_entropy_fluxes": entropy_faces,
        "net": net,
        "incoming_entropy": incoming_entropy,
    }


def audit_pipe_step(model, before, step, pipe_name, *, heat_rates_w=None, heat_temperatures_k=None):
    """Audit one coupled pipe update, including independently derived faces."""
    from check_pr_equilibrium_euler_pipe import audit

    pipe = model.pipes[pipe_name]["pipe"]
    old = dict(before.pipes)[pipe_name].to_dict()
    saved = step["pipes"][pipe_name]
    command = {"duration_s": saved["dt_s"]}
    if heat_rates_w is not None:
        command["heat_rates_w"] = heat_rates_w
    if heat_temperatures_k is not None:
        command["heat_temperatures_k"] = heat_temperatures_k
    config = {"model": pipe.to_dict(), "initial": {"state": old}, "schedule": [command]}
    report = {
        "config": config,
        "model": pipe.to_dict(),
        "initial": old,
        "state": saved["state"],
        "intervals": [
            {
                "state": saved["state"],
                "steps": [saved],
                "conservation_residuals": saved["conservation_residuals"],
            }
        ],
        "conservation_residuals": saved["conservation_residuals"],
    }

    def exchange(old_state, new_state, packet, dt):
        result = interface_exchange(model, before, pipe_name, packet)
        expected = {
            (pipe_name, side): (flux, result["boundary_entropy_fluxes"][side])
            for side, flux in enumerate(result["boundary_fluxes"])
            if flux is not None
        }
        actual = {(v["pipe"], v["side"]): v for v in step["interfaces"] if v["pipe"] == pipe_name}
        if set(actual) != set(expected):
            raise ValueError("Interface registry differs")
        for key, (flux, entropy) in expected.items():
            close(
                actual[key]["flux"],
                flux,
                atol=np.maximum(1e-6, result["flux_roundoff"][key[1]]),
                rtol=1e-7,
            )
            # PR entropy-density cancellation is amplified by the LF acoustic
            # envelope. Allow floating-point cancellation at that scale.
            cell_entropy_scale = (
                sum(
                    abs(q["entropy"])
                    for q in decode_cell(
                        model.fluid,
                        dict(before.pipes)[pipe_name].cells[0 if key[1] == 0 else -1].to_dict(),
                        dict(before.pipes)[pipe_name].cells[0 if key[1] == 0 else -1].phase_roots,
                    )
                )
                / pipe.volumes[0 if key[1] == 0 else -1]
            )
            entropy_roundoff = (
                64 * np.finfo(float).eps * pipe.wave_speed * max(1.0, cell_entropy_scale)
            )
            close(
                actual[key]["entropy_flux_w_m2_k"],
                entropy,
                atol=max(1e-7, entropy_roundoff),
                rtol=1e-7,
            )
            close(actual[key]["area_m2"], pipe.area, atol=1e-14, rtol=0.0)
            if actual[key]["node"] != model.pipes[pipe_name]["boundary_nodes"][key[1]]:
                raise ValueError("Interface node differs")
        return result

    config, report = json.loads(json.dumps([config, report]))
    return audit(config, report, external_exchange=exchange)


def audit_step(model, before, step, **commands):
    """Audit every spatial pipe and the implicit equipment step separately.

    Graph loads are reconstructed from the previous physical snapshot, never
    copied from the solver's reported interface fluxes.
    """
    from check_pr_equilibrium_euler_pipe import audit as audit_pipe
    from check_pr_equilibrium_transport import audit as audit_transport

    net = {name: np.zeros(len(model.fluid.names) + 2) for name in model.transport.names}
    incoming_entropy = 0.0
    checks = []
    for name, entry in model.pipes.items():
        checks.append(
            audit_pipe_step(
                model,
                before,
                step,
                name,
                heat_rates_w=commands.get("pipe_heat_w", {}).get(name),
                heat_temperatures_k=commands.get("pipe_heat_temperatures_k", {}).get(name),
            )
        )
        # Capture independently decoded old packets through the same public
        # auditor, whose callback runs after independent scalar PR decoding.
        pipe = entry["pipe"]
        old = dict(before.pipes)[name].to_dict()
        saved = step["pipes"][name]
        command = {"duration_s": saved["dt_s"]}
        for target, source in [
            ("heat_rates_w", "pipe_heat_w"),
            ("heat_temperatures_k", "pipe_heat_temperatures_k"),
        ]:
            if name in commands.get(source, {}):
                command[target] = commands[source][name]
        config = {"model": pipe.to_dict(), "initial": {"state": old}, "schedule": [command]}
        report = {
            "config": config,
            "model": pipe.to_dict(),
            "initial": old,
            "state": saved["state"],
            "intervals": [
                {
                    "state": saved["state"],
                    "steps": [saved],
                    "conservation_residuals": saved["conservation_residuals"],
                }
            ],
            "conservation_residuals": saved["conservation_residuals"],
        }

        def capture(old_state, new_state, packet, dt, name=name):
            nonlocal incoming_entropy
            exchange = interface_exchange(model, before, name, packet)
            for node, value in exchange["net"].items():
                net[node] += value
            incoming_entropy += exchange["incoming_entropy"]
            return exchange

        config, report = json.loads(json.dumps([config, report]))
        audit_pipe(config, report, external_exchange=capture)
    graph_model = json.loads(json.dumps(model.transport.to_dict()))
    gr = step["transport"]
    command = {"duration_s": step["dt_s"]}
    for source, target in [
        ("openings", "openings"),
        ("graph_heat_w", "heat_w"),
        ("graph_utility_temperature_k", "utility_temperature_k"),
        ("drive_torque_nm", "drive_torque_nm"),
        ("restriction_openings", "restriction_openings"),
    ]:
        if source in commands:
            command[target] = commands[source]
    opens = {v["name"]: True for v in graph_model["links"] + graph_model.get("heat_links", [])}
    opens.update(
        {
            v["card"]["name"]: v["card"]["open"]
            for v in graph_model.get("pump_connections", []) + graph_model.get("pumps", [])
        }
    )
    opens.update(
        {
            v["name"]: v["open"]
            for v in graph_model.get("compressors", []) + graph_model.get("restrictions", [])
        }
    )
    opens.update(command.get("openings", {}))
    heat = gr["ambient_heat_j"] + gr["external_heat_j"]
    interval = {
        "state": gr["state"],
        "steps": [gr],
        "openings": opens,
        "loss_coefficients": {},
        "total_heat_j": heat,
        "total_work_j": gr["external_work_j"],
        "entropy_production_j_k": gr["entropy_production_j_k"],
        "conservation_residuals": gr["conservation_residuals"],
    }
    initial = before.transport.to_dict()
    config = {"model": graph_model, "initial": {"state": initial}, "schedule": [command]}
    report = {
        "config": config,
        "model": graph_model,
        "initial": initial,
        "state": gr["state"],
        "intervals": [interval],
        "total_heat_j": heat,
        "total_work_j": gr["external_work_j"],
        "conservation_residuals": gr["conservation_residuals"],
    }
    config, report = json.loads(json.dumps([config, report]))
    graph_check = audit_transport(
        config,
        report,
        external_exchange=lambda *_: {"net": net, "incoming_entropy": incoming_entropy},
    )
    # Independent whole-system stocks: equipment fluid, thermal walls,
    # finite rotors and spatial conservative pipe inventories.
    from reference_pr_drives import rotor_energy

    def totals(state):
        graph = state["transport"]
        species = np.sum([v["component_moles"] for v in graph["nodes"].values()], axis=0)
        energy = sum(v["total_energy_j"] for v in graph["nodes"].values()) + rotor_energy(
            graph, graph_model
        )
        for node in graph_model["nodes"]:
            if node.get("wall") is not None:
                energy += (
                    node["wall"]["heat_capacity_j_k"]
                    * graph["nodes"][node["name"]]["wall_temperature_k"]
                )
        momentum = np.zeros(3)
        for name, entry in model.pipes.items():
            bulk = np.asarray(state["pipes"][name]["bulk_stocks"])
            species += bulk[:, :-2].sum(0)
            energy += bulk[:, -1].sum()
            momentum += bulk[:, -2].sum() * np.asarray(entry["direction"])
        return np.r_[species, energy, momentum]

    external = (
        heat + gr["external_work_j"] + sum(v["external_heat_j"] for v in step["pipes"].values())
    )
    impulse = sum(
        (
            np.asarray(entry["direction"]) * step["pipes"][name]["wall_impulse_kg_m_s"]
            for name, entry in model.pipes.items()
        ),
        start=np.zeros(3),
    )
    close(step["external_energy_j"], external, atol=1e-8, rtol=1e-7)
    close(step["external_impulse_kg_m_s"], impulse, atol=1e-10, rtol=1e-7)
    residual = (
        totals(step["state"])
        - totals(before.to_dict())
        - np.r_[np.zeros(len(model.fluid.names)), external, impulse]
    )
    close(residual, np.zeros_like(residual), atol=1e-5, rtol=0.0)
    close(step["conservation_residuals"], residual, atol=1e-6, rtol=1e-7)
    return {"accepted": True, "pipes": checks, "equipment": graph_check}


def audit(config, report):
    """Audit a complete source-bound IMEX schedule and published checkpoints."""
    from surface.pr_equilibrium_euler_plant import (
        PREquilibriumEulerPlant,
        PREquilibriumEulerPlantState,
    )

    if config != report["config"]:
        raise ValueError("Plant request differs")
    model = PREquilibriumEulerPlant.from_dict(config["model"])
    if json.loads(json.dumps(model.to_dict())) != report["model"]:
        raise ValueError("Plant normalized model differs")
    old = report["initial"]
    if config["initial"] != {"state": old} or len(config["schedule"]) != len(report["intervals"]):
        raise ValueError("Plant restart or schedule differs")
    checked = 0
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        start = old["transport"]["time_s"]
        physics = {
            k: v
            for k, v in command.items()
            if k not in ("duration_s", "maximum_step_s", "minimum_step_s", "maximum_attempts")
        }
        if not interval["steps"]:
            raise ValueError("Missing plant steps")
        for step in interval["steps"]:
            audit_step(model, PREquilibriumEulerPlantState.from_dict(old), step, **physics)
            if step["state"]["transport"] != step["transport"]["state"]:
                raise ValueError("Published equipment state differs")
            for name, value in step["state"]["pipes"].items():
                if value != step["pipes"][name]["state"]:
                    raise ValueError("Published pipe state differs")
            old = step["state"]
            checked += 1
        published = interval["state"]
        if {k: v for k, v in published.items() if k != "next_step_s"} != {
            k: v for k, v in old.items() if k != "next_step_s"
        }:
            raise ValueError("Published plant interval differs")
        close(
            published["transport"]["time_s"],
            start + command["duration_s"],
            atol=8 * np.spacing(max(1.0, abs(start + command["duration_s"]))),
            rtol=0.0,
        )
        old = published
    if old != report["state"]:
        raise ValueError("Final plant checkpoint differs")
    return {
        "accepted": True,
        "checked_substeps": checked,
        "closure": "explicit_Euler_faces_implicit_equipment",
    }
