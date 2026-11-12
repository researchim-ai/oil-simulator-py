"""Independent scalar EOS and local N/M/E/S audit of open implicit PR graphs.

Reuses the existing NumPy internal-face/equipment audit, independently supplies
external material fluxes. No production flash, decoder or flow solve is used.
The scalar feed reference currently supports at most two equilibrium phases.
"""

import json
import math

import numpy as np
from check_pr_equilibrium_transport import audit as transport_audit
from check_stratified_pr_flow import close
from reference_pr_open_boundaries import reconstruct

from surface.operation import _digest
from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork


def audit(model, initial, result, *, port_openings=None, **controls):
    model, initial, result = json.loads(json.dumps([model, initial, result], allow_nan=False))
    if model.get("model") != "pr_open_transport_v1" or result["open_model_id"] != _digest(model):
        raise ValueError("Open model identity differs")
    parsed = PREquilibriumTransportNetwork.from_dict(model["transport"])
    f = parsed.fluid
    physical = json.loads(json.dumps(parsed.to_dict()))
    nc = len(f.names)
    openings = {p["name"]: p["opening"] for p in model["pressure_ports"]}
    if set(port_openings or {}) - set(openings):
        raise ValueError("Unknown external command")
    openings.update(port_openings or {})
    if any(not math.isfinite(x) or x < 0 or x > 1 for x in openings.values()):
        raise ValueError("Invalid opening")
    if result["port_openings"] != openings:
        raise ValueError("Published boundary commands differ")
    saved_steps = iter(result["steps"])
    boundaries = []

    def exchange(old, new, data, dt):
        step = next(saved_steps)
        if step["state"] != new or step["dt_s"] != dt:
            raise ValueError("Open boundary trajectory differs")
        coupling = step["coupling"]
        if (
            coupling["model_id"] != result["open_model_id"]
            or coupling["closure"] != "end_state_stirred_material_boundaries"
        ):
            raise ValueError("Boundary physical closure differs")
        saved = coupling["ports"]
        reconstructed = reconstruct(model, physical, f, data, saved, openings)
        net, entropy = reconstructed["net"], reconstructed["incoming_entropy"]
        total_n = dt * sum((v[:-2] for v in net.values()), np.zeros(nc))
        total_e = dt * sum(v[-1] for v in net.values())
        close(step["exchange_component_moles"], total_n, atol=1e-10, rtol=2e-7)
        close(step["exchange_energy_j"], total_e, atol=1e-7, rtol=2e-7)
        boundaries.append((total_n, total_e))
        return {"net": net, "incoming_entropy": entropy}

    command = {"duration_s": result["state"]["time_s"] - initial["time_s"], **controls}
    config = {"model": physical, "initial": {"state": initial}, "schedule": [command]}
    report = {
        "config": config,
        "model": physical,
        "initial": initial,
        "intervals": [result],
        "total_heat_j": result["total_heat_j"],
        "total_work_j": result["total_work_j"],
        "conservation_residuals": result["conservation_residuals"],
    }
    check = transport_audit(config, report, external_exchange=exchange)
    close(
        result["boundary_component_moles"],
        sum((v[0] for v in boundaries), np.zeros(nc)),
        atol=1e-10,
        rtol=2e-7,
    )
    close(result["boundary_energy_j"], sum(v[1] for v in boundaries), atol=1e-7, rtol=2e-7)
    return {
        **check,
        "qualification": "Independent scalar phase EOS, N/M/E/S and pipe/equipment faces; empirical low-dP external characteristics, no experimental calibration or global stability certificate.",
    }
