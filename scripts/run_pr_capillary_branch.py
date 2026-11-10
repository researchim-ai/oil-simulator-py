"""Run finite-chamber PR graphs, auditing each transport/source/isolation step."""

import argparse
import json
import time
from pathlib import Path

from check_pr_capillary_branch import audit, audit_switch
from check_pr_capillary_pressure import audit as normal_audit
from run_pr_active_surface import fingerprints

from surface.pr_active_multifluid_pipe import PRActiveMultifluidState
from surface.pr_capillary_branch import PRCapillaryBranchNetwork, PRCapillaryBranchState


def run(config, *, device="cpu"):
    if set(config) != {"model", "initial", "schedule"}:
        raise ValueError("Explicit branch model, initial stocks and schedule required")
    schedule = config["schedule"]
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("Nonempty operating schedule required")
    hashes = fingerprints()
    model = PRCapillaryBranchNetwork.from_dict(config["model"], device=device)
    state = PRCapillaryBranchState.from_dict(config["initial"])
    model._view(state)
    intervals = []
    start = time.perf_counter()
    for period in schedule:
        if set(period) - {"duration_s", "initial_step_s", "normal_relaxation", "openings"} or not {
            "duration_s", "initial_step_s", "normal_relaxation"
        } <= set(period):
            raise ValueError("Period requires duration, step, normal-source flag and optional ports")
        if type(period["normal_relaxation"]) is not bool:
            raise ValueError("Normal-source flag must be boolean")
        event = None
        if "openings" in period:
            changed, switched, event = model.switch_ports(state, period["openings"])
            event_audit = audit_switch(model, state, changed, switched, event)
            model, state = changed, switched
        else:
            event_audit = None
        initial = state
        state, report = model.advance(
            initial, period["duration_s"], initial_step_s=period["initial_step_s"],
            normal_relaxation=period["normal_relaxation"],
        )
        old = model._view(initial)
        checks = []
        for step in report["advance"]["steps"]:
            axial = step["axial"] if period["normal_relaxation"] else step
            transported = PRActiveMultifluidState.from_dict(axial["state"])
            check = {"axial": audit(model, old, transported, axial)}
            if period["normal_relaxation"]:
                final = PRActiveMultifluidState.from_dict(step["normal"]["state"])
                check["normal"] = normal_audit(model.source, transported, final, step["normal"])
            else:
                final = transported
            checks.append(check)
            old = final
        if old != model._view(state):
            raise ValueError("Branch history does not terminate at actual graph stocks")
        intervals.append({
            "model": model.to_dict(), "initial": initial.to_dict(),
            "isolation_event": event, "isolation_audit": event_audit,
            "result": report, "independent_audits": checks,
        })
    if hashes != fingerprints():
        raise RuntimeError("Physical source changed during graph execution")
    return {
        "format": "pr_capillary_branch_run_v1", "request": config, "device": device,
        "source_sha256": hashes, "elapsed_with_audits_s": time.perf_counter() - start,
        "intervals": intervals, "final": state.to_dict(), "final_model": model.to_dict(),
        "qualification": (
            "Synthetic horizontal positive-support PR graph; finite stationary mixing chambers, "
            "declared strip/wall energy and reversible binary isolation. "
            "No calibrated equipment, dry-state transitions or field-scale performance claim."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing result")
    result = run(json.loads(args.config.read_text()), device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "end_time_s": result["final"]["time_s"],
        "accepted_substeps": sum(len(p["result"]["advance"]["steps"]) for p in result["intervals"]),
        "elapsed_with_audits_s": result["elapsed_with_audits_s"],
    }))


if __name__ == "__main__":
    main()
