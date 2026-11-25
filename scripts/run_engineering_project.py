"""Run an explicitly selected engineering study with its existing physics/audit.

Mode selection never changes tolerances, edits input files or converts another
mode's physical checkpoint. Dry-run resolves inputs and capabilities only.
"""

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_journal import atomic_json

from simulator.engineering_modes import BACKENDS, MODES, load_case


def _execute(plan, device, progress):
    request, options = plan.request(), plan.options()
    if plan.backend == "fim_facility":
        from check_fim_facility_forecast import audit
        from run_fim_blackoil_surface import run_case

        boundaries = [
            n
            for n in request["simulation"]["physical_surface_network"]["network"]["nodes"]
            if n.get("pressure_pa") is not None
        ]
        if len(boundaries) != 1:
            raise ValueError("Forecast backend requires one production receiver")
        boundaries[0]["pressure_pa"] = options["outlet_pressure_pa"]
        result = run_case(
            request,
            options["outlet_pressure_pa"],
            options["duration_days"],
            options["report_days"],
            device,
            facility_schedule=options["events"],
            progress_callback=progress,
        )

        def check(raw):
            return audit(
                request,
                raw,
                expected_schedule=options["events"],
                expected_days=options["duration_days"],
            )
    elif plan.backend == "pr_linepack":
        from check_pr_linepack import audit
        from run_pr_linepack import run

        result = run(request, device=device)

        def check(raw):
            return audit(request, raw)
    else:
        from run_pr_controlled_surface import run

        result = run(request, device=device, progress=progress, integrator=options["integrator"])
        if options["integrator"] == "sdirk2-common":
            from audit_pr_surface_collocation import audit
        else:
            from check_pr_controlled_surface import audit

        def check(raw):
            return audit(request, raw)

    return request, result, check


def run_project(project, case, output, *, device="cpu", progress=None):
    # These checks happen before output publication or GPU initialization.
    plan = load_case(project, case)
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda")
    if device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Select CUDA_VISIBLE_DEVICES explicitly for a CUDA run")
    if progress is not None and not callable(progress):
        raise TypeError("Progress callback must be callable")
    plan.verify_sources()
    root = Path(__file__).resolve().parents[1]
    sources = [
        p for folder in (root / "src", root / "scripts") for p in sorted(folder.rglob("*.py"))
    ]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    description = plan.describe()
    description.update(requested_device=device, source_sha256=hashes)
    atomic_json(output / "plan.json", description)
    atomic_json(
        output / "status.json",
        {"status": "running", "case": case, "mode": plan.mode, "stage": "solve"},
    )
    stage = "solve"
    try:
        started = time.perf_counter()
        effective_request, result, audit = _execute(plan, device, progress)
        wall = time.perf_counter() - started
        stage = "persist_trajectory"
        with gzip.open(output / "trajectory.json.gz", "wt") as stream:
            json.dump(
                {"case": description, "effective_request": effective_request, "result": result},
                stream,
                allow_nan=False,
            )
        stage = "numerical_audit"
        # Audits read the same JSON representation that is actually persisted.
        result = json.loads(json.dumps(result, allow_nan=False))
        certificate = audit(result)
        if certificate.get("accepted") is not True:
            raise ValueError("Backend numerical audit rejected the calculation")
        stage = "source_guard"
        plan.verify_sources()
        for p in sources:
            if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(root))]:
                raise RuntimeError(f"Source changed during engineering run: {p}")
        summary = {
            "status": "complete",
            "case": case,
            "mode": plan.mode,
            "backend": plan.backend,
            "fingerprint": plan.fingerprint,
            "requested_device": device,
            "execution": result.get("execution", result.get("runtime", {})),
            "run_wall_s": wall,
            "numerical_audit": certificate,
            "audit_scope": BACKENDS[plan.backend]["audit"],
            "physical_calibration": "not_established_by_this_run",
            "sources_unchanged": True,
            "trajectory_sha256": hashlib.sha256(
                (output / "trajectory.json.gz").read_bytes()
            ).hexdigest(),
            "simulation_horizon": description["simulation_horizon"],
            "omitted_physics": description["omitted_physics"],
        }
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "status.json", summary)
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        atomic_json(
            output / "status.json",
            {
                "status": "failed",
                "case": case,
                "mode": plan.mode,
                "stage": stage,
                "exception": type(exc).__name__,
                "reason": str(exc),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, nargs="?")
    parser.add_argument("--case")
    parser.add_argument("--describe-modes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.describe_modes:
        print(json.dumps({"modes": MODES, "backends": BACKENDS}, indent=2))
        return
    if args.project is None or args.case is None:
        parser.error("Provide project and --case, or use --describe-modes")
    if args.dry_run:
        print(json.dumps(load_case(args.project, args.case).describe(), indent=2))
        return
    if args.output is None:
        parser.error("--output is required for execution")
    print(
        json.dumps(
            run_project(
                args.project,
                args.case,
                args.output,
                device=args.device,
                progress=lambda e: print(json.dumps(e), file=sys.stderr, flush=True),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
