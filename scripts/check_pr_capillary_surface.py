"""Independent composition audit reusing scalar capillary and gravity verifiers."""

import json
import math

import numpy as np
from check_pr_capillary_column import aperture_area
from check_pr_capillary_facility import audit as capillary_audit
from check_pr_open_gravity_transport import audit as gravity_audit
from check_stratified_pr_flow import close
from reference_wetting_potential import potential

from surface.operation import _digest
from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe
from surface.pr_capillary_facility import PRCapillaryFacilityState
from surface.pr_capillary_pressure import PRCapillaryPressure


def capillary_stocks(system, state):
    """Scalar pipe N/U/KE/PE/surface/wetting stocks; shared columns excluded."""
    nc = len(next(iter(system["columns"].values()))["fluid"]["components"])
    amounts, energy, endpoints = [[] for _ in range(nc)], [], {}
    for name, spec in system["lines"].items():
        pipe = PRActiveMultifluidPipe.from_dict(spec["pipe"])
        source = PRCapillaryPressure(pipe, **spec["parameters"])
        ps = dict(state.pipes)[name]
        heights = np.array(
            [
                dict(cell)[source.liquid].volume_m3 if source.liquid in dict(cell) else 0.0
                for cell in ps.cells
            ]
        ) / (source.width * np.array(pipe.lengths))
        surface = source.sigma * source.width / (2 * pipe.lengths[0]) * sum(np.diff(heights) ** 2)
        attached = [p for p in spec["ports"] if p is not None]
        gravity = system["columns"][attached[0]["column"]]["gravity_m_s2"] if attached else 0.0
        for side, port in enumerate(spec["ports"]):
            if port is None:
                continue
            cs = dict(state.columns)[port["column"]]
            slots = dict(state.layer_slots)[port["column"]]
            if len(slots) != len(cs.phase_roots) or (
                len(slots) == 2 and slots != (source.liquid, source.gas)
            ):
                raise ValueError("Physical capillary layer registry differs")
            geometry = system["columns"][port["column"]]["geometry"]
            # Vertical profile coordinate is the actual column height.
            from surface.separator_geometry import SeparatorGeometry

            height = SeparatorGeometry(**geometry).height_m
            bounds = (0.0, *cs.interface_heights_m, height)
            center, radius = port["height_m"], pipe.diameter / 2
            fraction = math.fsum(
                aperture_area(
                    max(bounds[i], center - radius),
                    min(bounds[i + 1], center + radius),
                    center,
                    pipe.diameter,
                )
                / pipe.area
                for i, slot in enumerate(slots)
                if slot == source.liquid
            )
            exterior = fraction * pipe.area / source.width
            ci = 0 if side == 0 else -1
            work = (
                source.sigma * source.width / (2 * pipe.lengths[ci]) * (heights[ci] - exterior) ** 2
            )
            endpoints[name, side] = work
            surface += work
        if source.wetting is not None:
            surface += sum(
                source.width
                * np.array(pipe.lengths)
                * potential(source.wetting.to_dict(), source.sigma, heights)
            )
        energy.append(float(surface))
        for cell in ps.cells:
            for _, phase in cell:
                for j, n in enumerate(phase.component_moles):
                    amounts[j].append(n)
                mass = math.fsum(
                    n * c.molar_mass_kg_mol
                    for n, c in zip(phase.component_moles, pipe.fluid.components)
                )
                energy.append(phase.total_energy_j + mass * gravity * pipe.elevations[0])
    return np.array([math.fsum(v) for v in amounts]), math.fsum(energy), endpoints


def ordinary_stocks(graph, state):
    from reference_pr_drives import rotor_energy

    physical = graph["transport"]
    bodies = [*state["transport"]["nodes"].values(), *state["columns"].values()]
    n = np.sum([b["component_moles"] for b in bodies], axis=0)
    energy = math.fsum(b["total_energy_j"] for b in bodies) + rotor_energy(
        state["transport"], physical
    )
    energy += math.fsum(
        node["wall"]["heat_capacity_j_k"]
        * state["transport"]["nodes"][node["name"]]["wall_temperature_k"]
        for node in physical["nodes"]
        if node["wall"]
    )
    energy += math.fsum(
        w["heat_capacity_j_k"] * state["walls"][name] for name, w in graph["walls"].items()
    )
    return n, energy


def audit(model, initial, result):
    model, initial, result = json.loads(json.dumps([model, initial, result], allow_nan=False))
    if model["model"] != "pr_capillary_surface_plant_v1" or initial["model_id"] != _digest(model):
        raise ValueError("Capillary surface model identity differs")
    if model["surface"]["graph"]["transport"]["reference_elevation_m"] != 0:
        raise ValueError("This composition verifier requires absolute energy datum zero")
    cap = model["capillary"]
    before = PRCapillaryFacilityState.from_dict(result["capillary_initial"])
    middle = PRCapillaryFacilityState.from_dict(result["capillary_end"])
    after = PRCapillaryFacilityState.from_dict(result["capillary_final"])
    if _digest(before.to_dict()) != _digest(
        {
            "pipes": initial["capillary_pipes"],
            "columns": {n: initial["surface"]["columns"][n] for n in cap["columns"]},
            "layer_slots": initial["layer_slots"],
            "next_step_s": initial["next_step_s"],
            "model_id": _digest(cap),
        }
    ):
        raise ValueError("Initial shared column or capillary stock differs")
    if result["capillary"]["phase_controls"]:
        raise ValueError("Explicit side-port commands required in composition verifier")
    check_cap = capillary_audit(
        cap,
        before,
        middle,
        result["capillary"],
        normal_relaxation=result["capillary"]["normal_relaxation"],
    )
    expected_mid = json.loads(json.dumps(initial["surface"]))
    expected_mid["columns"].update({n: s.to_dict() for n, s in middle.columns})
    if _digest(result["midpoint"]) != _digest(expected_mid):
        raise ValueError("Shared midpoint differs from accepted capillary update")
    final = result["state"]
    if (
        final["model_id"] != _digest(model)
        or after.pipes != middle.pipes
        or _digest(final["capillary_pipes"]) != _digest({n: s.to_dict() for n, s in middle.pipes})
        or _digest(final["layer_slots"]) != _digest(dict(after.layer_slots))
        or set(dict(after.columns)) != set(cap["columns"])
        or any(
            _digest(s.to_dict()) != _digest(final["surface"]["columns"][n])
            for n, s in after.columns
        )
        or final["time_s"] != after.time_s
    ):
        raise ValueError("Final shared state or pipe history differs")
    _n, _e, start_work = capillary_stocks(cap, middle)
    _n, _e, end_work = capillary_stocks(cap, after)
    actual = {
        n: math.fsum(
            end_work[line, side] - start_work[line, side]
            for line, spec in cap["lines"].items()
            for side, p in enumerate(spec["ports"])
            if p is not None and p["column"] == n
        )
        for n in cap["columns"]
    }
    for n, value in actual.items():
        close(result["actual_geometry_work_j"][n], value, atol=1e-20, rtol=1e-8)
        tolerance = 8 * math.ulp(final["surface"]["columns"][n]["total_energy_j"]) + 1e-20
        if abs(value - result["geometry_work_j"][n]) > tolerance:
            raise ValueError("Unclosed endpoint surface work")
    raw = result["surface"]
    if final["surface"] != raw["state"]:
        raise ValueError("Final ordinary graph checkpoint differs")
    interval = {
        **raw,
        "steps": [raw],
        "openings": result["openings"],
        "open_model_id": _digest(model["surface"]),
        "port_openings": result["port_openings"],
        "boundary_component_moles": raw["material_component_moles"],
        "boundary_energy_j": raw["material_energy_j"],
    }
    check_gravity = gravity_audit(
        model["surface"],
        result["midpoint"],
        interval,
        openings=result["openings"],
        column_work_j={n: -v for n, v in result["geometry_work_j"].items()},
    )
    n0, e0, _ = capillary_stocks(cap, before)
    n1, e1, _ = capillary_stocks(cap, after)
    graph = model["surface"]["graph"]
    gn0, ge0 = ordinary_stocks(graph, initial["surface"])
    gn1, ge1 = ordinary_stocks(graph, final["surface"])
    heat = raw["total_heat_j"]
    work = raw["total_work_j"] + math.fsum(result["geometry_work_j"].values())
    close(result["total_heat_j"], heat, atol=1e-12, rtol=1e-10)
    close(result["total_work_j"], work, atol=1e-20, rtol=1e-10)
    close(
        result["material_component_moles"], raw["material_component_moles"], atol=1e-20, rtol=1e-10
    )
    close(result["material_energy_j"], raw["material_energy_j"], atol=1e-14, rtol=1e-10)
    residual = np.r_[
        gn1 + n1 - gn0 - n0 - np.array(raw["material_component_moles"]),
        ge1 + e1 - ge0 - e0 - raw["material_energy_j"] - heat - work,
    ]
    close(result["conservation_residuals"], residual, atol=1e-8, rtol=1e-8)
    if np.max(np.abs(residual[:-1])) > 1e-9 or abs(residual[-1]) > 1e-7:
        raise ValueError("Independent whole-plant balance rejected")
    close(
        result["entropy_production_j_k"],
        raw["entropy_production_j_k"] + result["capillary"]["entropy_production_j_k"],
        atol=1e-10,
        rtol=1e-9,
    )
    return {
        "accepted": True,
        "capillary": check_cap,
        "gravity": check_gravity,
        "capillary_initial_component_moles": n0.tolist(),
        "capillary_final_component_moles": n1.tolist(),
        "capillary_initial_energy_j": e0,
        "capillary_final_energy_j": e1,
        "qualification": "Scalar suboperator checks and shared-state/surface-work composition; no field calibration.",
    }
