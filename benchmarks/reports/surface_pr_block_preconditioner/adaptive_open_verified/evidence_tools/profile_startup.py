"""Bounded diagnostic profile, not a solver speed or completion benchmark."""

import cProfile
import hashlib
import json
import pstats
import signal
from pathlib import Path

from run_pr_controlled_surface import run


class Deadline(BaseException):
    pass


def expired(_signum, _frame):
    raise Deadline()


root = Path.cwd()
config = root / "configs/surface_pr_integrated_facility.json"
paths = [
    *sorted((root / "src/surface").glob("*.py")),
    root / "scripts/run_pr_controlled_surface.py",
    config,
]
hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
events = []
prof = cProfile.Profile()
signal.signal(signal.SIGALRM, expired)
status = "interrupted_at_diagnostic_budget"
prof.enable()
signal.setitimer(signal.ITIMER_REAL, 30)
try:
    run(json.loads(config.read_text()), progress=events.append)
    status = "completed"
except Deadline:
    pass
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    prof.disable()
stats = pstats.Stats(prof)
rows = [
    {
        "file": str(Path(k[0]).relative_to(root))
        if str(k[0]).startswith(str(root) + "/")
        else k[0],
        "line": k[1],
        "function": k[2],
        "primitive_calls": v[0],
        "calls": v[1],
        "self_s": v[2],
        "cumulative_s": v[3],
    }
    for k, v in stats.stats.items()
]
result = {
    "scope": __doc__,
    "status": status,
    "budget_s": 30,
    "events": events,
    "source_sha256": hashes,
    "sources_unchanged": all(
        hashlib.sha256((root / p).read_bytes()).hexdigest() == h for p, h in hashes.items()
    ),
    "top_cumulative": sorted(rows, key=lambda r: r["cumulative_s"], reverse=True)[:60],
    "top_self": sorted(rows, key=lambda r: r["self_s"], reverse=True)[:40],
}
out = (
    root
    / "benchmarks/reports/surface_pr_block_preconditioner/adaptive_open_verified/full_facility_profile.json"
)
with out.open("x") as f:
    json.dump(result, f, indent=2, allow_nan=False)
print(
    json.dumps(
        {
            "status": status,
            "last_event": events[-1] if events else None,
            "top": result["top_cumulative"][:18],
        },
        indent=2,
    )
)
