"""Independent scalar open gravitational graph profiles, apertures and N/M/E/S."""

import json
import math

import numpy as np
from check_pr_gravity_transport import audit as gravity_audit
from check_stratified_pr_flow import close
from reference_pr_open_boundaries import reconstruct

from surface.operation import _digest
from surface.pr_gravity_transport import PRGravityTransportNetwork


def audit(model, initial, result, *, port_openings=None, column_work_j=None, **controls):
    model, initial, result = json.loads(json.dumps([model, initial, result], allow_nan=False))
    if model.get("model") != "pr_open_gravity_transport_v1" or result["open_model_id"] != _digest(
        model
    ):
        raise ValueError("Open gravitational model differs")
    graph = PRGravityTransportNetwork.from_dict(model["graph"])
    physical = model["graph"]["transport"]
    nc = len(graph.transport.fluid.names)
    openings = {p["name"]: p["opening"] for p in model["pressure_ports"]}
    if set(port_openings or {}) - set(openings):
        raise ValueError("Unknown material port command")
    openings.update(port_openings or {})
    if (
        any(not math.isfinite(v) or v < 0 or v > 1 for v in openings.values())
        or openings != result["port_openings"]
    ):
        raise ValueError("Material opening differs")
    steps = iter(result["steps"])
    totals = []

    def exchange(old, new, data, dt):
        step = next(steps)
        if step["state"] != new or step["dt_s"] != dt:
            raise ValueError("Open gravitational trajectory differs")
        material = step["columns"].get("material", step["transport"].get("coupling"))
        if (
            material["closure"] != "end_state_stirred_material_boundaries"
            or material["model_id"] != result["open_model_id"]
        ):
            raise ValueError("Material boundary identity differs")
        value = reconstruct(
            model, physical, graph.transport.fluid, data, material["ports"], openings
        )
        n = dt * sum((v[:-2] for v in value["net"].values()), np.zeros(nc))
        e = dt * sum(v[-1] for v in value["net"].values())
        totals.append((n, e))
        return value

    command = {
        "duration_s": result["state"]["transport"]["time_s"] - initial["transport"]["time_s"],
        **controls,
    }
    cfg = {"model": model["graph"], "initial": {"state": initial}, "schedule": [command]}
    report = {
        "config": cfg,
        "model": model["graph"],
        "initial": initial,
        "intervals": [result],
        "total_heat_j": result["total_heat_j"],
        "total_work_j": result["total_work_j"],
        "conservation_residuals": result["conservation_residuals"],
    }
    check = gravity_audit(
        cfg,
        report,
        external_exchange=exchange,
        reference_aperture_order=32,
        column_work_j=column_work_j,
    )
    n, e = sum((v[0] for v in totals), np.zeros(nc)), sum(v[1] for v in totals)
    close(result["boundary_component_moles"], n, atol=1e-10, rtol=2e-7)
    close(result["boundary_energy_j"], e, atol=1e-7, rtol=2e-7)
    return {
        **check,
        "minimum_reference_aperture_order": 32,
        "qualification": "Independent scalar PR chemical profiles, 32-point aperture rays, internal and external N/M/E/S balances. No field calibration or global stability proof.",
    }
