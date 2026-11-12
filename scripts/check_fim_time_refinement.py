"""Three-grid time self-convergence of actual source-bound coupled FIM runs.

Reuses the independent packet/capillary/gravity audits. No production step
is called. A three-grid trend is not an exact solution or field validation.
"""

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
from check_dynamic_receiver import audit


def difference_order(coarse, medium, fine):
    vectors = [np.asarray(v, dtype=float) for v in (coarse, medium, fine)]
    if len({v.shape for v in vectors}) != 1 or not all(np.isfinite(v).all() for v in vectors):
        raise ValueError("Finite common observable vectors required")
    first = float(np.linalg.norm(vectors[0] - vectors[1]))
    second = float(np.linalg.norm(vectors[1] - vectors[2]))
    floor = 128 * np.finfo(float).eps * max(1e-300, *(float(np.linalg.norm(v)) for v in vectors))
    resolved = bool(min(first, second) > floor)
    return {
        "coarse_medium_l2": first,
        "medium_fine_l2": second,
        "roundoff_floor": floor,
        "resolved": resolved,
        "decreases": resolved and second < first,
        "observed_order": math.log2(first / second) if resolved else None,
    }


def protocol(runs):
    if len(runs) != 3:
        raise ValueError("Exactly three refinement runs required")
    common, widths = [], []
    for run in runs:
        cfg = deepcopy(run["config"])
        cfg.pop("qualification")
        sim = cfg["simulator"]["simulation"]
        width = sim.pop("max_time_step_days") * 86400
        initial = sim.pop("initial_time_step_days") * 86400
        if (
            not math.isfinite(width)
            or width <= 0
            or not math.isclose(initial, width, rel_tol=1e-12)
            or sim.get("coupled_time_control") is not False
            or sim.get("dt_increase_factor") != 1.0
        ):
            raise ValueError("Explicit uniform-grid prediction-disabled protocol required")
        if any(
            not math.isclose(p["duration_s"] / width, round(p["duration_s"] / width), rel_tol=1e-12)
            for p in cfg["intervals"]
        ):
            raise ValueError("Reporting events must lie on every temporal grid")
        common.append(cfg)
        widths.append(width)
    if not common[0] == common[1] == common[2]:
        raise ValueError("Physics, tolerances, initial state or schedule changed during refinement")
    if not all(math.isclose(a / b, 2.0, rel_tol=1e-12) for a, b in zip(widths, widths[1:])):
        raise ValueError("Exactly halved ordered step widths required")
    return widths


def observables(run):
    state = run["records"][-1]["state"]
    cap = run["config"]["simulator"]["simulation"]["dynamic_receiver"]["receiver"]["capillary"]
    rows = {key: [] for key in ("velocity_m_s", "holdup", "pressure_pa", "temperature_k")}
    identities = []
    for name, pipe in sorted(state["capillary_pipes"].items()):
        weights = [
            v["molar_mass_kg_mol"] for v in cap["lines"][name]["pipe"]["fluid"]["components"]
        ]
        for index, cell in enumerate(pipe["cells"]):
            volume = math.fsum(p["volume_m3"] for p in cell.values())
            for slot, phase in sorted(cell.items()):
                identities.append((name, index, slot))
                mass = math.fsum(
                    n * m for n, m in zip(phase["component_moles"], weights, strict=True)
                )
                rows["velocity_m_s"].append(phase["momentum_kg_m_s"] / mass)
                rows["holdup"].append(phase["volume_m3"] / volume)
                rows["pressure_pa"].append(phase["pressure_pa"])
                rows["temperature_k"].append(phase["temperature_k"])
    return identities, rows


def audit_refinement(runs):
    widths = protocol(runs)
    checks = []
    for run, width in zip(runs, widths, strict=True):
        checks.append(audit(run["config"], run))
        if any(s["rejected_steps"] for s in run["time_step_statistics"]):
            raise ValueError("A chopped trajectory is not the requested uniform time grid")
        for record in run["records"]:
            if not math.isclose(record["dt_s"], width, rel_tol=1e-12):
                raise ValueError("Accepted trajectory is not the declared uniform grid")
    samples = [observables(run) for run in runs]
    if not samples[0][0] == samples[1][0] == samples[2][0]:
        raise ValueError("Phase/pipe topology differs: a smooth-regime order fit is not applicable")
    results = {
        name: difference_order(*(value[name] for _, value in samples)) for name in samples[0][1]
    }
    demonstrated = all(results[k]["decreases"] for k in ("velocity_m_s", "holdup"))
    return {
        "accepted": demonstrated,
        "step_widths_s": widths,
        "physical_end_time_s": runs[0]["records"][-1]["state"]["time_s"],
        "checked_runs": len(checks),
        "observables": results,
        "scope": "Three-grid endpoint self-convergence of all capillary-cell phase velocities, holdups, pressures and temperatures in actual coupled FIM. Reused scalar audits validate packets and suboperators; not an exact time solution, a reservoir-field convergence study or field calibration.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs=3)
    args = parser.parse_args()
    result = audit_refinement([json.loads(p.read_text()) for p in args.runs])
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["accepted"]:
        raise SystemExit("Temporal self-convergence was not demonstrated")
