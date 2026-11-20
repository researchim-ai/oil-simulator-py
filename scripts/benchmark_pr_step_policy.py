"""Paired accepted interval: certified Newton floor and split probe reuse.

Both policies use identical physics/tolerances/initial stocks and current PR
kernels. Independent physical audit runs outside solver timing. The complete
requested interval must finish; rejected trials alone cannot pass this probe.
"""

import argparse
import gzip
import hashlib
import json
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from check_pr_controlled_surface import audit
from run_pr_controlled_surface import run

from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_subcycled_surface import PRSubcycledSurface
from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState
from surface.torch_nonlinear import TorchNewton

ROOT = Path(__file__).resolve().parents[1]


def measure(config, enabled):
    calls = []
    original = TorchNewton.solve

    def counted(self, *args, **kwargs):
        answer = original(self, *args, **kwargs)
        calls.append(
            {
                "unknowns": args[1].numel(),
                "evaluations": answer[3],
                "jacobian_builds": self.jacobian_builds,
                "stop": self.stop_reason,
            }
        )
        return answer

    with ExitStack() as context:
        context.enter_context(patch.object(TorchNewton, "_early_certified_roundoff", enabled))
        context.enter_context(patch.object(PRControlledSurface, "_share_probe_quadrature", enabled))
        context.enter_context(patch.object(PRSubcycledSurface, "_share_probe_quadrature", enabled))
        context.enter_context(patch.object(TorchNewton, "solve", counted))
        start = time.perf_counter()
        report = run(config)
        elapsed = time.perf_counter() - start
    report = json.loads(json.dumps(report))
    checked = audit(config, report)
    if checked.get("accepted") is not True:
        raise ValueError("Independent physical audit rejected")
    return {"wall_s": elapsed, "report": report, "audit": checked, "solves": calls}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--compiled-cpu-chemical", action="store_true")
    p.add_argument("--accelerated-first", action="store_true")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    EquilibriumFluid._compiled_chemical_cpu = args.compiled_cpu_chemical
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        Path(__file__),
        ROOT / "scripts/run_pr_controlled_surface.py",
    ]
    hashes = {str(q.relative_to(ROOT)): hashlib.sha256(q.read_bytes()).hexdigest() for q in paths}
    results = {}
    sequence = [("reference", False), ("accelerated", True)]
    if args.accelerated_first:
        sequence.reverse()
    for name, enabled in sequence:
        r = measure(config, enabled)
        with gzip.open(args.output_dir / (name + ".json.gz"), "wt") as f:
            json.dump(r, f, allow_nan=False)
        results[name] = r
    a, b = (results[k] for k in ("reference", "accelerated"))
    model = PRControlledSurface.from_dict(config["model"])
    old = PRControlledSurfaceState.from_dict(a["report"]["intervals"][-1]["state"])
    new = PRControlledSurfaceState.from_dict(b["report"]["intervals"][-1]["state"])
    difference = model._error(old, new)
    if old.time_s != new.time_s:
        raise ValueError("End clocks differ")

    def stats(r):
        return {
            "wall_s": r["wall_s"],
            "nonlinear_solves": len(r["solves"]),
            "residual_evaluations": sum(x["evaluations"] for x in r["solves"]),
            "jacobian_builds": sum(x["jacobian_builds"] for x in r["solves"]),
            "attempts": sum(x["attempts"] for x in r["report"]["intervals"]),
        }

    summary = {
        "scope": "One paired CPU accepted interval including initialization; shared host, not a statistical speedup guarantee",
        "config": str(args.config),
        "reference": stats(a),
        "accelerated": stats(b),
        "observed_speedup": a["wall_s"] / b["wall_s"],
        "end_time_s": new.time_s,
        "difference_in_original_error_norm": difference,
        "agree": difference < 0.01,
        "independent_audits_passed": True,
        "source_sha256": hashes,
        "complete_operating_cycle_qualified": False,
    }
    if any(
        hashlib.sha256(q.read_bytes()).hexdigest() != hashes[str(q.relative_to(ROOT))]
        for q in paths
    ):
        raise RuntimeError("Source changed during paired measurement")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, indent=2))
    return 0 if summary["agree"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
