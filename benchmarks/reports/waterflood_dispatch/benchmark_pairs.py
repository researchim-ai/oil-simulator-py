"""Sequential opposite-order execution pairs, preserving guarded raw studies."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "matched_pairs"

if OUT.exists():
    raise FileExistsError(OUT)
OUT.mkdir()
completed = []
for name, kind in [("baseline_first", "baseline"), ("fast_second", "fast"),
                   ("fast_first", "fast"), ("baseline_second", "baseline")]:
    with (OUT / (name + ".log")).open("x") as stream:
        subprocess.run(
            [sys.executable, "scripts/run_fim_waterflood.py", "--config",
             f"configs/fim_waterflood_field192_dispatch_{kind}.json", "--days", "0.1",
             "--report-days", "0.05", "--save-fields", "--output", str(OUT / name)],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True,
        )
    report = json.loads((OUT / name / "summary.json").read_text())
    completed.append({"name": name, "kind": kind, "wall_s": report["solve_wall_s"]})
    (OUT / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")
    print(json.dumps(completed[-1]), flush=True)
