"""Journal accepted PR macrosteps without changing stepping or acceptance.

The recorder observes the unchanged coarse/fine trial and acceptance event.
Each line contains the two accepted physical halfsteps and the exact restart
state. An interrupted run can therefore retain independently auditable work.
"""

import argparse
import hashlib
import json
import math
import os
import signal
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from run_pr_controlled_surface import run

from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_plant_registry import align_physical_clock
from surface.pr_surface_control import PRControlledSurface

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        ROOT / "scripts/run_pr_controlled_surface.py",
        Path(__file__),
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def journal_run(config, output, *, device="cpu", budget_s=3600):
    """Return completion metadata; a prefix is never called a complete cycle."""
    if not math.isfinite(budget_s) or budget_s <= 0:
        raise ValueError("Finite positive wall-time budget required")
    if set(config.get("initial", {})) != {"state"}:
        raise ValueError("Journaling requires an explicit initial checkpoint")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = output.with_suffix(".accepted.jsonl")
    trajectory = output.with_suffix(".trajectory.json")
    if any(p.exists() for p in (output, journal, trajectory)):
        raise FileExistsError("Use a fresh journal/output path")
    pending = {}
    original = PRControlledSurface._temporal_trial
    hashes = source_hashes()
    start = time.perf_counter()
    status = {
        "completed": False,
        "source_sha256": hashes,
        "journal": journal.name,
        "simulated_time_s": config["initial"]["state"]["plant"]["transport"]["time_s"],
        "accepted_macrosteps": 0,
        "compiled_cpu_chemical": EquilibriumFluid._compiled_chemical_cpu,
    }

    def observed(self, *args, **kwargs):
        answer = original(self, *args, **kwargs)
        pending.update(trial=answer)
        return answer

    with journal.open("x") as handle:
        handle.write(
            json.dumps({"kind": "header", "config": config, "source_sha256": hashes}) + "\n"
        )
        handle.flush()

        def progress(event):
            print(json.dumps(event), flush=True)
            if event["event"] == "attempt":
                pending["interval_end_s"] = event["end_time_s"]
            if event["event"] != "accepted":
                return
            _, fine, one, two = pending.pop("trial")
            current = replace(fine, next_step_s=event["next_step_s"])
            if current.time_s != event["time_s"]:
                current = replace(
                    current, plant=align_physical_clock(current.plant, event["time_s"])
                )
            record = {
                "kind": "accepted",
                "interval_end_s": pending["interval_end_s"],
                "event": event,
                "state": current.to_dict(),
                "steps": [one, two],
            }
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            status.update(
                simulated_time_s=current.time_s,
                accepted_macrosteps=status["accepted_macrosteps"] + 1,
            )

        def expired(signum, _frame):
            reason = (
                "Wall-time budget exhausted" if signum == signal.SIGALRM else "Process interrupted"
            )
            raise TimeoutError(reason + "; accepted journal retained")

        old = signal.signal(signal.SIGALRM, expired)
        old_term = signal.signal(signal.SIGTERM, expired)
        signal.setitimer(signal.ITIMER_REAL, budget_s)
        try:
            with patch.object(PRControlledSurface, "_temporal_trial", observed):
                result = run(config, device=device, progress=progress)
            trajectory.write_text(json.dumps(result, allow_nan=False) + "\n")
            status.update(completed=True, trajectory=trajectory.name)
        except (TimeoutError, RuntimeError, ValueError, KeyboardInterrupt) as exc:
            status["failure"] = f"{type(exc).__name__}: {exc}"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
            signal.signal(signal.SIGTERM, old_term)
    status.update(
        elapsed_s=time.perf_counter() - start, sources_unchanged=source_hashes() == hashes
    )
    status["completed"] &= status["sources_unchanged"]
    output.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    return status


def accepted_prefix(journal):
    """Recover only durable accepted work, ready for the independent checker.

    A final incomplete line (process killed during write) is ignored; corruption
    of any earlier line is an error. The remaining request keeps original stage
    boundaries, so an interruption cannot skip isolation or restart commands.
    """
    lines = Path(journal).read_text().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines.pop()
    records = [json.loads(line) for line in lines]
    if not records or records[0].get("kind") != "header":
        raise ValueError("Journal header missing")
    header, rows = records[0], records[1:]
    config = header["config"]
    if any(row.get("kind") != "accepted" for row in rows):
        raise ValueError("Unknown journal record")
    if not rows:
        return None, config
    model = PRControlledSurface.from_dict(config["model"])
    initial = config["initial"]["state"]
    clock = initial["plant"]["transport"]["time_s"]
    cursor = 0
    intervals, commands, remaining = [], [], []
    previous = initial
    for index, command in enumerate(config["schedule"]):
        end = clock + command["duration_s"]
        selected = []
        while cursor < len(rows) and rows[cursor]["interval_end_s"] == end:
            selected.append(rows[cursor])
            cursor += 1
        if not selected:
            remaining.extend(config["schedule"][index:])
            break
        steps = [step for row in selected for step in row["steps"]]
        state = selected[-1]["state"]
        reached = state["plant"]["transport"]["time_s"]
        if not clock < reached <= end:
            raise ValueError("Journal clock outside requested interval")
        heat = math.fsum(step["total_heat_j"] for step in steps)
        work = math.fsum(step["total_work_j"] for step in steps)
        from surface.pr_surface_control import PRControlledSurfaceState

        before = PRControlledSurfaceState.from_dict(previous)
        after = PRControlledSurfaceState.from_dict(state)
        intervals.append(
            {
                "state": state,
                "steps": steps,
                "total_heat_j": heat,
                "total_work_j": work,
                "conservation_residuals": model.plant._ledger(
                    before.plant, after.plant, heat + work
                ),
            }
        )
        commands.append({**command, "duration_s": reached - clock})
        previous = state
        if reached < end:
            resumed = {**command, "duration_s": end - reached}
            resumed.pop("reset_trips", None)  # already applied at this interval's start
            remaining = [resumed, *config["schedule"][index + 1 :]]
            break
        clock = end
    if cursor != len(rows):
        raise ValueError("Journal records do not match schedule boundaries")
    heat = math.fsum(r["total_heat_j"] for r in intervals)
    work = math.fsum(r["total_work_j"] for r in intervals)
    from surface.pr_surface_control import PRControlledSurfaceState

    first = PRControlledSurfaceState.from_dict(initial)
    last = PRControlledSurfaceState.from_dict(previous)
    request = {**config, "schedule": commands}
    report = {
        "config": request,
        "model": json.loads(json.dumps(model.to_dict())),
        "initial": initial,
        "intervals": intervals,
        "total_heat_j": heat,
        "total_work_j": work,
        "conservation_residuals": model.plant._ledger(first.plant, last.plant, heat + work),
    }
    continuation = (
        {**config, "initial": {"state": previous}, "schedule": remaining} if remaining else None
    )
    return report, continuation


def merge_operating_reports(original, prefix, continuation):
    """Reunite an interrupted report interval without changing physical records."""
    if continuation["initial"] != prefix["intervals"][-1]["state"]:
        raise ValueError("Continuation does not start at the recorded accepted state")
    if continuation["model"] != prefix["model"]:
        raise ValueError("Continuation model differs")
    from surface.pr_surface_control import PRControlledSurfaceState

    model = PRControlledSurface.from_dict(original["model"])
    chunks = [*prefix["intervals"], *continuation["intervals"]]
    initial = prefix["initial"]
    before = PRControlledSurfaceState.from_dict(initial)
    clock = before.time_s
    intervals = []
    cursor = 0
    for command in original["schedule"]:
        end = clock + command["duration_s"]
        selected = []
        while cursor < len(chunks):
            chunk = chunks[cursor]
            reached = chunk["state"]["plant"]["transport"]["time_s"]
            if reached > end + 8 * math.ulp(end):
                break
            selected.append(chunk)
            cursor += 1
            if abs(reached - end) <= 8 * math.ulp(end):
                break
        if not selected or abs(reached - end) > 8 * math.ulp(end):
            raise ValueError("Merged reports do not cover an original operating stage")
        state = selected[-1]["state"]
        after = PRControlledSurfaceState.from_dict(state)
        steps = [step for chunk in selected for step in chunk["steps"]]
        heat = math.fsum(chunk["total_heat_j"] for chunk in selected)
        work = math.fsum(chunk["total_work_j"] for chunk in selected)
        intervals.append(
            {
                "state": state,
                "steps": steps,
                "total_heat_j": heat,
                "total_work_j": work,
                "conservation_residuals": model.plant._ledger(
                    before.plant, after.plant, heat + work
                ),
            }
        )
        before, clock = after, end
    if cursor != len(chunks):
        raise ValueError("Extra report interval beyond the original request")
    heat = math.fsum(x["total_heat_j"] for x in intervals)
    work = math.fsum(x["total_work_j"] for x in intervals)
    first = PRControlledSurfaceState.from_dict(initial)
    return {
        **continuation,
        "config": original,
        "initial": initial,
        "intervals": intervals,
        "total_heat_j": heat,
        "total_work_j": work,
        "conservation_residuals": model.plant._ledger(first.plant, before.plant, heat + work),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path, nargs="?")
    p.add_argument(
        "--recover-journal",
        type=Path,
        help="Export independently audited accepted prefix and remaining request",
    )
    p.add_argument(
        "--resume-journal",
        type=Path,
        help="Resume durable accepted work and qualify the original complete cycle",
    )
    p.add_argument(
        "--allow-solver-change",
        action="store_true",
        help="Explicit source migration after independent prefix audit; physical model/state must remain compatible",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--budget-s", type=float, default=3600)
    p.add_argument("--compiled-cpu-chemical", action="store_true")
    args = p.parse_args()
    if args.recover_journal and args.resume_journal:
        p.error("Choose recovery or resumed integration")
    if args.recover_journal:
        if args.config is not None:
            p.error("Recovery uses the original request in the journal header")
        report, remaining = accepted_prefix(args.recover_journal)
        if report is None:
            p.error("No accepted macrostep in journal")
        from check_pr_controlled_surface import audit

        checked = audit(report["config"], report)
        if checked.get("accepted") is not True:
            raise ValueError("Independent accepted-prefix audit rejected")
        paths = [
            args.output,
            args.output.with_suffix(".trajectory.json"),
            args.output.with_suffix(".remaining.json"),
        ]
        if any(path.exists() for path in paths):
            raise FileExistsError("Use a fresh recovery path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        paths[1].write_text(json.dumps(report, allow_nan=False) + "\n")
        paths[2].write_text(json.dumps(remaining, allow_nan=False) + "\n")
        result = {
            "complete_operating_cycle_qualified": False,
            "accepted_prefix_audit": checked,
            "remaining_request": paths[2].name,
            "trajectory": paths[1].name,
        }
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2))
        return 0
    prefix = original_request = None
    restart_changes = {}
    if args.resume_journal:
        if args.config is not None:
            p.error("Resume uses the request in the journal header")
        with args.resume_journal.open() as stream:
            header = json.loads(stream.readline())
        hashes = source_hashes()
        restart_changes = {
            k: {"recorded": v, "current": hashes.get(k)}
            for k, v in header["source_sha256"].items()
            if (k.startswith("src/surface/") or k == "scripts/run_pr_controlled_surface.py")
            and hashes.get(k) != v
        }
        if restart_changes and not args.allow_solver_change:
            raise ValueError(
                "Solver source differs from accepted journal; qualify changes separately"
            )
        original_request = header["config"]
        prefix, config = accepted_prefix(args.resume_journal)
        if prefix is None or config is None:
            p.error("Resume needs a nonempty incomplete accepted prefix")
        from check_pr_controlled_surface import audit

        if audit(prefix["config"], prefix).get("accepted") is not True:
            raise ValueError("Independent prefix audit rejected")
    elif args.config is None:
        p.error("Configuration, --resume-journal or --recover-journal required")
    else:
        config = json.loads(args.config.read_text())
    if not math.isfinite(args.budget_s) or args.budget_s <= 0:
        p.error("Finite positive wall-time budget required")
    if args.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        p.error("Explicit CUDA_VISIBLE_DEVICES required")
    EquilibriumFluid._compiled_chemical_cpu = args.compiled_cpu_chemical
    result = journal_run(config, args.output, device=args.device, budget_s=args.budget_s)
    result["restart_source_changes"] = restart_changes
    result["explicit_source_migration"] = bool(restart_changes) and args.allow_solver_change
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if result["completed"]:
        from check_pr_controlled_surface import audit
        from validate_pr_integrated_facility import operational_audit

        report = json.loads(args.output.with_suffix(".trajectory.json").read_text())
        if prefix is not None:
            report = merge_operating_reports(original_request, prefix, report)
            full_path = args.output.with_suffix(".full-cycle.trajectory.json")
            if full_path.exists():
                raise FileExistsError(full_path)
            full_path.write_text(json.dumps(report, allow_nan=False) + "\n")
            result["trajectory"] = full_path.name
            result["prefix_journal"] = str(args.resume_journal)
        try:
            result["physical_audit"] = audit(report["config"], report)
            result["operation"] = operational_audit(report["config"], report)
            result["sources_unchanged"] &= source_hashes() == result["source_sha256"]
            result["full_operating_cycle_qualified"] = (
                result["physical_audit"].get("accepted") is True and result["sources_unchanged"]
            )
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            result["full_operating_cycle_qualified"] = False
            result["audit_failure"] = f"{type(exc).__name__}: {exc}"
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))
    return 0 if result.get("full_operating_cycle_qualified", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
