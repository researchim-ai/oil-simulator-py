"""Reproduce exact-time event checks in a NEW output directory, never overwrite evidence."""

import argparse
import copy
import gzip
import json
from pathlib import Path

from audit_pr_controlled_mixed import audit
from check_pr_controlled_surface import actuator
from run_pr_controlled_surface import run

from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState

ROOT = Path(__file__).resolve().parent


def load(name):
    with gzip.open(ROOT / f"release-cycle-{name}/trajectory.json.gz", "rt") as stream:
        return json.load(stream)


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    a, b = load("reference"), load("optimized")
    model = PRControlledSurface.from_dict(a["config"]["model"])
    results = []
    for i in (1, 2):
        initial = copy.deepcopy(b["intervals"][i - 1]["state"])
        target = a["intervals"][i]["steps"][0]["state"]
        dt = target["plant"]["transport"]["time_s"] - initial["plant"]["transport"]["time_s"]
        cfg = {
            "model": a["config"]["model"],
            "initial": {"state": initial},
            "schedule": [{**a["config"]["schedule"][i], "duration_s": dt}],
        }
        report = json.loads(json.dumps(run(cfg, integrator="sdirk2-common")))
        certificate = audit(cfg, report)
        error = model._error(
            PRControlledSurfaceState.from_dict(target),
            PRControlledSurfaceState.from_dict(report["intervals"][-1]["state"]),
        )
        with gzip.open(output / f"event-replay-{i}.json.gz", "xt") as stream:
            json.dump(report, stream)
        results.append(
            {
                "interval": i,
                "time_s": target["plant"]["transport"]["time_s"],
                "dt_s": dt,
                "error_in_original_norm": error,
                "audit": certificate,
                "accepted": certificate["accepted"] and error <= 1,
            }
        )
    manual = {}
    for name, report in [("reference", a), ("optimized", b)]:
        initial = report["intervals"][0]["state"]
        t0 = initial["plant"]["transport"]["time_s"]
        errors = []
        for row in report["intervals"][1]["steps"]:
            state = row["state"]
            dt = state["plant"]["transport"]["time_s"] - t0
            for spec in a["config"]["model"]["motors"]:
                n = spec["name"]
                expected = actuator(initial["positions"][n], 0.0, dt, spec["actuator"], "normal")
                errors.append(abs(state["positions"][n] - expected) / model.position_tolerance)
        manual[name] = {"maximum_error_in_position_budget": max(errors), "checked": len(errors)}
    result = {
        "accepted": all(x["accepted"] for x in results)
        and all(x["maximum_error_in_position_budget"] <= 1 for x in manual.values()),
        "exact_time_replays": results,
        "independent_ideal_actuator_checks": manual,
        "scope": "direct requested-time integration and independent exact manual actuator law; no interpolation across events",
    }
    (output / "event-replay-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
