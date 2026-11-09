"""Independently audit every prescribed local morphology-source substep."""

import argparse
import hashlib
import json
from pathlib import Path

from reference_pr_film_mist_exchange import audit as audit_source

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe

ROOT = Path(__file__).resolve().parents[1]


def audit(config, report):
    if config != report["request"] or report["initial"] != config["initial"]:
        raise ValueError("Morphology run request/initial differs")
    for path, sha in report["source_sha256"].items():
        p = (ROOT / path).resolve()
        p.relative_to(ROOT)
        if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            raise ValueError("Changed morphology source")
    if len(config["schedule"]) != len(report["intervals"]):
        raise ValueError("Morphology schedule length differs")
    pipe = PRActiveMultifluidPipe.from_dict(config["pipe"])
    old = report["initial"]
    count = 0
    maximum = 0.0
    for period, interval in zip(config["schedule"], report["intervals"], strict=True):
        target = old["time_s"] + period["duration_s"]
        for step in interval["steps"]:
            r = step["source"]
            dt = r["dt_s"]
            if not 0 < dt <= period["maximum_step_s"] or old["time_s"] + dt > target:
                raise ValueError("Morphology substep outside report interval")
            audit_source(
                pipe,
                old,
                r["state"],
                r,
                config["exchange"],
                period["entrainment_m3_s"],
                period["deposition_m3_s"],
                dt,
            )
            expected = {**r["state"], "time_s": old["time_s"] + dt}
            if step["state"] != expected:
                raise ValueError("Morphology clock/frame differs")
            maximum = max(maximum, abs(r["joint_energy_residual_j"]))
            old = step["state"]
            count += 1
        if old != interval["state"] or abs(old["time_s"] - target) > 1e-14:
            raise ValueError("Incomplete morphology interval")
    if old != report["final"]:
        raise ValueError("Morphology final checkpoint differs")
    return {
        "accepted": True,
        "checked_steps": count,
        "maximum_reported_joint_energy_residual_j": maximum,
        "qualification": "Source conservation and independent PR EOS; no empirical rate qualification",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    print(json.dumps(audit(json.loads(a.config.read_text()), json.loads(a.report.read_text()))))


if __name__ == "__main__":
    main()
