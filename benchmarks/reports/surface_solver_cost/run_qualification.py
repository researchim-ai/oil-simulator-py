"""Sequential CPU qualification: no competing benchmark children."""
import argparse
import gzip
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=2)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("positive pair count required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    def execute(script, stem, extra=(), timeout=240):
        path = output / (stem+".json")
        print("Running "+stem, flush=True)
        with (output/(stem+".log")).open("w") as log:
            subprocess.run([sys.executable, str(HERE/script), "--output", str(path), *extra],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)
        return json.loads(path.read_text())
    root = execute("compare_pr_root.py", "pr_root")
    liquid = execute("compare_liquid_newton.py", "liquid_newton", ("--pairs", str(args.pairs)))
    rows = []
    source = None
    for pair in range(args.pairs):
        reports = []
        for split in ((True, False) if pair%2 == 0 else (False, True)):
            stem = f"pr_pair{pair}_"+("split" if split else "joint")
            row = execute("run_pr_profile.py", stem, ("--split-characteristics",) if split else ())
            if source is None:
                source = row["source_sha256"]
            assert row["source_sha256"] == source
            row["pair"] = pair
            rows.append(row)
            raw = output/(stem+".report.json")
            report = json.loads(raw.read_text())
            reports.append(report)
            with gzip.open(str(raw)+".gz", "wb") as f:
                f.write(raw.read_bytes())
            raw.unlink()
        # The accepted physical steps and endpoint must be bitwise identical.
        assert reports[0]["intervals"] == reports[1]["intervals"]
    medians = {label:statistics.median(r["elapsed_s"] for r in rows if r["split_characteristics"] == split)
               for label,split in (("split",True), ("joint",False))}
    summary = dict(pr_profile_medians_s=medians, pr_profiles=rows,
                   liquid_newton_medians_s=liquid["medians_s"], pr_root=root,
                   accepted_pr_histories_bitwise_identical=True)
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in summary.items() if k not in ("pr_profiles", "pr_root")}), flush=True)


if __name__ == "__main__":
    main()
