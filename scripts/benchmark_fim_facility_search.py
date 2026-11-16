"""Paired same-physics FIM/facility search timings with bracket certification.

Only search_method changes. Every candidate solves the same reservoir step;
all trajectories are persisted and independently accounted before completion.
"""

import argparse
import copy
import gzip
import hashlib
import json
import statistics
from pathlib import Path

from benchmark_journal import BenchmarkJournal
from check_fim_facility_forecast import audit
from run_fim_blackoil_surface import run_case

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = {
    "final_pressure_pa": 100.0,
    "final_s_w": 1e-4,
    "final_s_g": 1e-4,
    "final_rs": 0.01,
    "withdrawal_relative": 0.001,
}


def errors(a, b):
    result = {
        key: max(abs(x - y) for x, y in zip(a[key], b[key], strict=True))
        for key in BUDGETS
        if key != "withdrawal_relative"
    }
    result["withdrawal_relative"] = max(
        abs(x - y) / max(abs(x), 1)
        for x, y in zip(
            a["withdrawn_component_mass_kg"], b["withdrawn_component_mass_kg"], strict=True
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/fim_blackoil_facility_coupled.json"
    )
    parser.add_argument(
        "--schedule", type=Path, default=ROOT / "configs/fim_facility_forecast_schedule.json"
    )
    parser.add_argument("--days", type=float, default=30)
    parser.add_argument("--report-days", type=float, default=5)
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["fluid"]["pvt_path"] = str(ROOT / config["fluid"]["pvt_path"])
    schedule = json.loads(args.schedule.read_text())
    paths = [
        *sorted((ROOT / "src/simulator").glob("*.py")),
        *sorted((ROOT / "src/surface").glob("*.py")),
        Path(__file__),
        ROOT / "scripts/run_fim_blackoil_surface.py",
        ROOT / "scripts/check_fim_facility_forecast.py",
        args.config.resolve(),
        args.schedule.resolve(),
        Path(config["fluid"]["pvt_path"]),
    ]

    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    before = hashes()
    journal = BenchmarkJournal(
        args.output,
        {
            "scope": __doc__,
            "source_sha256": before,
            "budgets": BUDGETS,
            "days": args.days,
            "report_days": args.report_days,
            "repeats": args.repeats,
            "device": "cpu",
        },
    )
    rows, comparisons = [], []
    for pair in range(args.repeats):
        matched = {}
        for mode in ("bisect", "safeguarded") if pair % 2 == 0 else ("safeguarded", "bisect"):
            c = copy.deepcopy(config)
            c["simulation"]["coupled_facility"].setdefault("control", {})["search_method"] = mode
            receiver = c["simulation"]["coupled_facility"]["facility"]["separator_stages"][0][
                "pressure_pa"
            ]
            print(json.dumps({"event": "run", "pair": pair, "mode": mode}), flush=True)
            r = run_case(
                c, receiver, args.days, args.report_days, "cpu", facility_schedule=schedule
            )
            filename = f"pair{pair}_{mode}.json.gz"
            path = args.output / filename
            with gzip.open(path, "wt") as handle:
                json.dump(
                    {"config": c, "schedule": schedule, "requested_days": args.days, "result": r},
                    handle,
                    allow_nan=False,
                )
            checked = audit(c, r, expected_schedule=schedule, expected_days=args.days)
            row = {
                "pair": pair,
                "mode": mode,
                "elapsed_s": r["solve_wall_s"],
                "audit": checked,
                "trajectory": filename,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            rows.append(row)
            journal.checkpoint(rows, comparisons, {})
            if hashes() != before:
                raise RuntimeError("Source changed during benchmark")
            matched[mode] = r
            print(json.dumps(row), flush=True)
        delta = errors(matched["bisect"], matched["safeguarded"])
        comparison = {
            "pair": pair,
            "errors": delta,
            "accepted": all(delta[k] <= BUDGETS[k] for k in delta),
        }
        comparisons.append(comparison)
        journal.checkpoint(rows, comparisons, {})
        if not comparison["accepted"]:
            raise RuntimeError("Search methods exceed declared equivalence budgets")
    median = {
        mode: statistics.median(r["elapsed_s"] for r in rows if r["mode"] == mode)
        for mode in ("bisect", "safeguarded")
    }
    journal.complete(
        {
            "scope": __doc__,
            "runs": rows,
            "comparisons": comparisons,
            "budgets": BUDGETS,
            "median_elapsed_s": median,
            "speedup": median["bisect"] / median["safeguarded"],
            "source_sha256": before,
            "sources_unchanged": hashes() == before,
        }
    )
    print(
        json.dumps(
            {"median_elapsed_s": median, "speedup": median["bisect"] / median["safeguarded"]}
        )
    )


if __name__ == "__main__":
    main()
