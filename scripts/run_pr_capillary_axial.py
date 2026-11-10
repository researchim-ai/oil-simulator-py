"""Reproduce PR/capillary line dynamics and audit every substep."""

import argparse
import json
from pathlib import Path

from check_pr_capillary_axial import audit
from check_pr_capillary_pressure import audit as normal_audit
from run_pr_active_surface import fingerprints

from surface.pr_active_multifluid_pipe import PRActiveMultifluidState
from surface.pr_capillary_axial import PRCapillaryAxialPipe


def run(config, *, device="cpu"):
    if set(config) != {"model", "initial", "duration_s", "initial_step_s", "normal_relaxation"}:
        raise ValueError("Explicit axial model, stocks, duration, step and normal flag required")
    hashes = fingerprints()
    model = PRCapillaryAxialPipe.from_dict(config["model"], device=device)
    initial = PRActiveMultifluidState.from_dict(config["initial"])
    end, record = model.advance(
        initial,
        config["duration_s"],
        initial_step_s=config["initial_step_s"],
        normal_relaxation=config["normal_relaxation"],
    )
    old, audits = initial, []
    for step in record["steps"]:
        axial = step["axial"] if config["normal_relaxation"] else step
        transported = PRActiveMultifluidState.from_dict(axial["state"])
        row = {"axial": audit(model, old, transported, axial)}
        if config["normal_relaxation"]:
            final = PRActiveMultifluidState.from_dict(step["normal"]["state"])
            row["normal"] = normal_audit(model.source, transported, final, step["normal"])
        else:
            final = transported
        audits.append(row)
        old = final
    if old != end or hashes != fingerprints():
        raise ValueError("Joint axial history/source changed during execution")
    return {
        "format": "pr_capillary_axial_run_v1",
        "request": config,
        "device": device,
        "source_sha256": hashes,
        "initial": initial.to_dict(),
        "final": end.to_dict(),
        "result": record,
        "independent_audits": audits,
        "qualification": "Small uniform positive-support two-phase PR line with periodic, reflecting or maintained open faces; not a topology-changing graph or a field benchmark",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = run(json.loads(args.config.read_text()), device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "accepted_substeps": len(result["result"]["steps"]),
                "rejected_substeps": len(result["result"]["rejected"]),
                "end_time_s": result["final"]["time_s"],
            }
        )
    )


if __name__ == "__main__":
    main()
