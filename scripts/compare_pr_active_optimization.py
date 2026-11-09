"""Compare saved before/after physical checkpoints, not timing alone.

Historical source hashes must match an explicitly supplied source archive.
Current sources are checked separately. No stale-source bypass is allowed.
"""

import argparse
import hashlib
import json
from pathlib import Path

from summarize_pr_active_benchmark import ROOT, compare


def load(path, source_root):
    report = json.loads(Path(path).read_text())
    for name, digest in report["source_sha256"].items():
        candidate = source_root / name
        if not candidate.is_file():
            # Unchanged config files may be shared with the current checkout.
            candidate = ROOT / name
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source archive/config does not match: {name}")
    return report


def run(before_paths, after_paths, archive, cells):
    results = []
    for old_path, new_path in zip(before_paths, after_paths, strict=True):
        old, new = load(old_path, Path(archive)), load(new_path, ROOT)
        if old["device"] != new["device"] or old["torch"] != new["torch"]:
            raise ValueError("Different device or Torch version")
        old_cases = {c["cells"]: c for c in old["cases"]}
        selected = set()
        for b in new["cases"]:
            if b["cells"] not in cells:
                continue
            if b["cells"] in selected:
                raise ValueError("Duplicate selected scaling point")
            selected.add(b["cells"])
            a = old_cases[b["cells"]]
            for k in (
                "branches",
                "columns",
                "equipment_vessels",
                "pump_rotors",
                "active_phases",
                "physical_step_s",
                "simulated_duration_s",
            ):
                if a[k] != b[k]:
                    raise ValueError(f"Different workload: {k}")
            if not a["global_ledger_accepted"] or not b["global_ledger_accepted"]:
                raise ValueError("Global balance rejected")
            worst = compare(a["final_state"], b["final_state"])
            results.append(
                {
                    "device": new["device"],
                    "hardware": new["hardware"],
                    "cells": b["cells"],
                    "before_step_seconds": a["step_seconds"],
                    "after_step_seconds": b["step_seconds"],
                    "before_mean_s": a["mean_step_seconds"],
                    "after_mean_s": b["mean_step_seconds"],
                    "speedup": a["mean_step_seconds"] / b["mean_step_seconds"],
                    "final_state_parity_accepted": True,
                    "maximum_parity_tolerance_fraction": worst[0],
                    "worst_parity_path": worst[1],
                    "old_source_sha256": old["source_sha256"],
                    "new_source_sha256": new["source_sha256"],
                    "old_report_sha256": hashlib.sha256(Path(old_path).read_bytes()).hexdigest(),
                    "new_report_sha256": hashlib.sha256(Path(new_path).read_bytes()).hexdigest(),
                }
            )
        if selected != set(cells):
            raise ValueError("Missing requested cell count in after report")
    return {
        "format": "pr_active_optimization_comparison_v1",
        "scope": "Two short steps per point; no long-horizon throughput or confidence interval claim.",
        "requested_cell_counts": cells,
        "parity_atol": 1e-8,
        "parity_rtol": 1e-8,
        "cases": results,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, nargs="+", required=True)
    p.add_argument("--after", type=Path, nargs="+", required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--cells", type=int, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args.before, args.after, args.archive, args.cells)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for row in result["cases"]:
        print({k: v for k, v in row.items() if "sha256" not in k})


if __name__ == "__main__":
    main()
