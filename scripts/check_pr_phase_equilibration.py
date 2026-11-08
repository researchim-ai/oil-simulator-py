"""Scalar PR audit of a conservative, variable-phase UV projection."""

import math

import numpy as np
from check_stratified_pr_flow import close
from reference_pr_interphase_relaxation import decode

from surface.equilibrium_properties import EquilibriumFluid
from surface.operation import _digest


def audit(model, before, after, report):
    if set(model) != {"model", "fluid"} or model["model"] != "pr_phase_equilibration_v1":
        raise ValueError("Unknown phase projection model")
    fluid = EquilibriumFluid.from_dict(model["fluid"])
    expected = _digest(model)
    for state in [before, after]:
        if (
            set(state) != {"phases", "phase_roots", "model_id", "time_s"}
            or state["model_id"] != expected
        ):
            raise ValueError("Phase checkpoint identity differs")
    if (
        before["time_s"] != after["time_s"]
        or not math.isfinite(before["time_s"])
        or before["time_s"] < 0
    ):
        raise ValueError("An instantaneous source cannot advance physical time")
    if (
        after != report["state"]
        or len(before["phases"]) != report["phase_count_before"]
        or len(after["phases"]) != report["phase_count_after"]
    ):
        raise ValueError("Phase count/checkpoint report differs")
    old = decode(fluid, before, before["phase_roots"])
    new = decode(fluid, after, after["phase_roots"])
    heat = report["heat_j"]
    bath = report["heat_temperature_k"]
    if (
        not math.isfinite(heat)
        or (bath is not None and (not math.isfinite(bath) or bath <= 0))
        or (heat and bath is None)
    ):
        raise ValueError("Missing physical heat reservoir")
    nc = len(fluid.components)
    residual = [
        math.fsum(q["n"][j] for q in new) - math.fsum(q["n"][j] for q in old) for j in range(nc)
    ]
    residual += [
        math.fsum(q["volume"] for q in new) - math.fsum(q["volume"] for q in old),
        math.fsum(p["total_energy_j"] for p in after["phases"])
        - math.fsum(p["total_energy_j"] for p in before["phases"])
        - heat,
        math.fsum(p["momentum_kg_m_s"] for p in after["phases"])
        - math.fsum(p["momentum_kg_m_s"] for p in before["phases"]),
    ]
    close(residual, [0.0] * (nc + 3), atol=np.array([1e-8] * nc + [1e-10, 1e-6, 1e-9]), rtol=0.0)
    close(
        report["conservation_residuals"],
        residual,
        atol=np.array([1e-10] * nc + [1e-12, 1e-8, 1e-10]),
        rtol=0.0,
    )
    entropy = (
        math.fsum(q["entropy"] for q in new)
        - math.fsum(q["entropy"] for q in old)
        - (0.0 if bath is None else heat / bath)
    )
    if entropy < -1e-7:
        raise ValueError("Phase projection violates total entropy")
    close(report["entropy_production_j_k"], entropy, atol=1e-7, rtol=1e-7)
    # Common pressure/temperature/velocity and equality of active fugacities.
    from check_stratified_pr import Reference

    chemical = []
    for q, phase, root in zip(new, after["phases"], after["phase_roots"]):
        close(
            [q["p"], q["t"], q["velocity"]],
            [new[0]["p"], new[0]["t"], new[0]["velocity"]],
            atol=np.array([1e-6, 1e-8, 1e-9]),
            rtol=1e-10,
        )
        x = q["n"] / q["n"].sum()
        active = np.flatnonzero(x > 0)
        state = Reference(fluid, active).state(q["p"], q["t"], x, root)
        chemical.append((active, np.log(x[active]) + state["phi"]))
    for a, b in zip(chemical, chemical[1:]):
        if not np.array_equal(a[0], b[0]):
            raise ValueError("Equilibrium phase component support differs")
        close(a[1], b[1], atol=1e-7, rtol=0.0)
    return {
        "accepted": True,
        "phase_count_before": len(old),
        "phase_count_after": len(new),
        "energy_residual_j": residual[nc + 1],
    }
