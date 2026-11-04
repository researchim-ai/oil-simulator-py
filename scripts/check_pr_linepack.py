"""Independent compilation, Darcy-face and finite pipe inventory audit.

Reconstructs cell volumes/half-cell resistances from the requested pipe spec;
uses NumPy PR and bracketed scalar Darcy references, no dynamic Torch solve.
Currently requires closed external material boundaries, as the network auditor.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from check_stratified_pr import grid
from check_stratified_pr_flow import close
from check_stratified_pr_network import audit as network_audit

from surface.operation import _digest
from surface.pr_linepack import HorizontalPRPipe
from surface.stratified_pr_flow import StratifiedPRFlow
from surface.stratified_pr_network import StratifiedConnection


def expanded_model(request):
    if (
        set(request) != {"model", "vessels", "pipes", "integration"}
        or request["model"] != "quasisteady_pr_linepack_v1"
    ):
        raise ValueError("Unknown linepack request model")
    # Parsers normalize metadata only; no initialization or dynamic solves.
    terminals = {n: StratifiedPRFlow.from_dict(v).to_dict() for n, v in request["vessels"].items()}
    pipes = [HorizontalPRPipe.from_dict(p) for p in request["pipes"]]
    vessels = copy.deepcopy(terminals)
    links = []
    cell_names = {}
    face_names = {}
    integration = {
        "initial_step_s": 0.01,
        "minimum_step_s": 1e-6,
        "maximum_step_s": 1.0,
        "relative_tolerance": 1e-5,
        "temperature_tolerance_k": 0.002,
        "maximum_attempts": 1000,
    }
    if set(request["integration"]) - set(integration):
        raise ValueError("Unknown linepack integration controls")
    integration.update(request["integration"])
    for pipe in pipes:
        source = terminals[pipe.source]["column"]
        target = terminals[pipe.target]["column"]
        center = source["elevation_m"] + pipe.source_height_m
        close(center, target["elevation_m"] + pipe.target_height_m, atol=1e-9, rtol=0.0)
        dx = pipe.length_m / pipe.cells
        names = [f"pipe:{pipe.name}:cell:{i}" for i in range(pipe.cells)]
        faces = [f"pipe:{pipe.name}:face:{i}" for i in range(pipe.cells + 1)]
        cell_names[pipe.name] = names
        face_names[pipe.name] = faces
        for name in names:
            if name in vessels:
                raise ValueError("Linepack compiler cell collision")
            column = copy.deepcopy(source)
            column["geometry"] = {
                "diameter_m": pipe.diameter_m,
                "length_m": dx,
                "orientation": "horizontal",
            }
            column["elevation_m"] = center - pipe.diameter_m / 2
            wall = pipe.to_dict()["wall"]
            for key in ("heat_capacity_j_k", "fluid_ua_w_k", "ambient_ua_w_k"):
                wall[key] /= pipe.cells
            vessels[name] = {
                "model": "stratified_pr_flow_v1",
                "column": column,
                "wall": wall,
                "outlets": [],
                "feeds": [],
                "initial_step_s": 1.0,
                "minimum_step_s": 1e-5,
                "maximum_step_s": 60.0,
                "temperature_tolerance_k": 0.002,
                "relative_tolerance": 1e-5,
                "maximum_attempts": 1000,
            }
        sequence = [pipe.source, *names, pipe.target]
        for i, name in enumerate(faces):
            links.append(
                StratifiedConnection(
                    name,
                    sequence[i],
                    sequence[i + 1],
                    pipe.source_height_m if i == 0 else pipe.diameter_m / 2,
                    pipe.target_height_m if i == pipe.cells else pipe.diameter_m / 2,
                    pipe.diameter_m,
                    0.0,
                    allow_reverse=pipe.allow_reverse if i == 0 else True,
                    maximum_pressure_drop_fraction=pipe.maximum_pressure_drop_fraction,
                    hydraulic_model="darcy_weisbach",
                    length_m=dx / 2 if i in (0, pipe.cells) else dx,
                    dynamic_viscosity_pa_s=pipe.dynamic_viscosity_pa_s,
                    roughness_m=pipe.roughness_m,
                    maximum_velocity_m_s=pipe.maximum_velocity_m_s,
                ).to_dict()
            )
    return (
        {
            "model": "stratified_pr_network_v1",
            "vessels": dict(sorted(vessels.items())),
            "connections": links,
            **integration,
        },
        pipes,
        cell_names,
        face_names,
    )


def mean_pressure(state, column):
    from surface.separator_geometry import SeparatorGeometry

    geometry = SeparatorGeometry(**column["geometry"])
    active = sum(n > 0 for n in state["component_moles"])
    order = state["quadrature_order"]
    _heights, weights = grid(geometry, state["interface_heights_m"], order)
    y = np.array(state["solution_variables"])
    return (
        math.fsum(
            float(
                w
                @ np.exp(
                    y[i * order * active : (i + 1) * order * active].reshape(order, active)[:, 0]
                )
            )
            for i, w in enumerate(weights)
        )
        / geometry.volume_m3
    )


def audit(config, report):
    if set(config) != {"model", "initial", "schedule"} or report["config"] != config:
        raise ValueError("Linepack config differs from request")
    compiled, pipes, cells, faces = expanded_model(config["model"])
    if report["compiled_network"] != compiled:
        raise ValueError("Compiled linepack geometry/resistance differs from request")
    expected_wrapper = {
        "model": "quasisteady_pr_linepack_v1",
        "vessels": {n: compiled["vessels"][n] for n in config["model"]["vessels"]},
        "pipes": [p.to_dict() for p in pipes],
        "integration": {
            k: v for k, v in compiled.items() if k not in ("model", "vessels", "connections")
        },
    }
    if report["model"] != expected_wrapper:
        raise ValueError("Normalized linepack model differs from request")
    identity = _digest(compiled)

    def validate_ids(state):
        if state["model_id"] != identity:
            raise ValueError("Linepack network identity differs")
        for name, child in state["vessels"].items():
            model = compiled["vessels"][name]
            if child["model_id"] != _digest(model) or child["column"]["model_id"] != _digest(
                {"model": "stratified_pr_uv_v1", **model["column"]}
            ):
                raise ValueError("Linepack vessel/column identity differs")

    validate_ids(report["initial"])
    expanded_config = {
        "vessels": {n: {"model": v} for n, v in compiled["vessels"].items()},
        "connections": compiled["connections"],
        "schedule": config["schedule"],
    }
    numerical = network_audit(
        expanded_config,
        {"config": expanded_config, "initial": report["initial"], "intervals": report["intervals"]},
    )
    old = report["initial"]
    maximum_error = 0.0
    for interval in report["intervals"]:
        final = interval["state"]
        validate_ids(final)
        if set(interval["pipes"]) != {p.name for p in pipes}:
            raise ValueError("Reported pipe registry differs")
        for pipe in pipes:
            saved = interval["pipes"][pipe.name]
            members = cells[pipe.name]

            def inventory(state, members=members):
                children = state["vessels"]
                n = np.array([children[name]["column"]["component_moles"] for name in members]).sum(
                    0
                )
                fluid = math.fsum(children[name]["column"]["total_energy_j"] for name in members)
                wall = math.fsum(
                    compiled["vessels"][name]["wall"]["heat_capacity_j_k"]
                    * (
                        children[name]["wall_temperature_k"]
                        - compiled["vessels"][name]["wall"]["reference_temperature_k"]
                    )
                    for name in members
                )
                return n, fluid, wall

            n0, e0, w0 = inventory(old)
            n1, e1, w1 = inventory(final)
            incoming = np.zeros_like(n0)
            outgoing = np.zeros_like(n0)
            ein = eout = ambient = 0.0
            for step in interval["steps"]:
                validate_ids(step["state"])
                transfer = {p["name"]: p for p in step["transfers"]}
                a, b = transfer[faces[pipe.name][0]], transfer[faces[pipe.name][-1]]
                dt = step["dt_s"]
                incoming += dt * np.array(a["component_rates_mol_s"])
                outgoing += dt * np.array(b["component_rates_mol_s"])
                ein += dt * a["total_energy_rate_w"]
                eout += dt * b["total_energy_rate_w"]
                for name in members:
                    wall = compiled["vessels"][name]["wall"]
                    tw = step["state"]["vessels"][name]["wall_temperature_k"]
                    ambient += dt * wall["ambient_ua_w_k"] * (wall["ambient_temperature_k"] - tw)
            close(saved["component_moles"], n1, atol=1e-9)
            close(saved["fluid_total_energy_j"], e1, atol=1e-6)
            close(saved["wall_energy_j"], w1, atol=1e-6)
            close(saved["total_energy_j"], e1 + w1, atol=1e-6)
            close(saved["volume_m3"], math.pi * pipe.diameter_m**2 / 4 * pipe.length_m, atol=1e-12)
            components = compiled["vessels"][pipe.source]["column"]["fluid"]["components"]
            mass = float(n1 @ np.array([c["molar_mass_kg_mol"] for c in components]))
            close(saved["mass_kg"], mass, atol=1e-10)
            for key, value in {
                "source_component_transfer_mol": incoming,
                "target_component_transfer_mol": outgoing,
                "source_energy_transfer_j": ein,
                "target_energy_transfer_j": eout,
                "ambient_heat_j": ambient,
            }.items():
                close(saved[key], value, atol=1e-7)
            residual = n1 - n0 - incoming + outgoing
            error = e1 + w1 - e0 - w0 - ein + eout - ambient
            close(residual, np.zeros_like(residual), atol=1e-8, rtol=0.0)
            close(error, 0.0, atol=2e-6, rtol=0.0)
            close(saved["component_residuals_mol"], residual, atol=1e-8, rtol=0.0)
            close(saved["energy_residual_j"], error, atol=2e-6, rtol=0.0)
            maximum_error = max(maximum_error, abs(error))
            if [c["name"] for c in saved["cells"]] != members:
                raise ValueError("Axial cell registry differs")
            for i, description in enumerate(saved["cells"]):
                name = members[i]
                state = final["vessels"][name]["column"]
                column = compiled["vessels"][name]["column"]
                close(
                    description["axial_center_m"],
                    (i + 0.5) * pipe.length_m / pipe.cells,
                    atol=1e-12,
                )
                close(
                    description["mean_pressure_pa"],
                    mean_pressure(state, column),
                    atol=1e-5,
                    rtol=1e-9,
                )
                close(description["temperature_k"], state["temperature_k"], atol=1e-8)
                close(description["component_moles"], state["component_moles"], atol=1e-10)
                close(description["fluid_total_energy_j"], state["total_energy_j"], atol=1e-6)
                if description["phase_count"] != len(state["phase_roots"]):
                    raise ValueError("Reported pipe phase count differs")
        old = final
    return {
        **numerical,
        "checked_pipes": len(pipes),
        "maximum_pipe_energy_residual_j": maximum_error,
        "compiled_geometry_verified": True,
    }


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
