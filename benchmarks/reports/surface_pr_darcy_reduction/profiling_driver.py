import cProfile
import hashlib
import json
import pstats
import signal
from pathlib import Path

from run_pr_controlled_surface import run

ROOT = Path.cwd()
OUT = ROOT / "benchmarks/reports/surface_pr_darcy_reduction"
config_path = OUT / "reduced_config.json"
paths = sorted((ROOT / "src/surface").glob("*.py")) + [
    ROOT / "scripts/run_pr_controlled_surface.py",
    config_path,
]
fingerprints = lambda: {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
}
before = fingerprints()


class BudgetExpired(BaseException):
    pass


def stop(*_):
    raise BudgetExpired()


progress = []
profile = cProfile.Profile()
old = signal.signal(signal.SIGALRM, stop)
signal.setitimer(signal.ITIMER_REAL, 45)
status = "incomplete"
try:
    profile.enable()
    run(json.loads(config_path.read_text()), device="cpu", progress=progress.append)
    status = "completed"
except BudgetExpired:
    status = "profiling_budget_exhausted"
finally:
    profile.disable()
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, old)
rows = []
for (filename, line, name), (primitive, calls, own, cumulative, callers) in pstats.Stats(
    profile
).stats.items():
    rows.append(
        {
            "file": filename,
            "line": line,
            "function": name,
            "primitive_calls": primitive,
            "calls": calls,
            "own_s": own,
            "cumulative_s": cumulative,
        }
    )
result = {
    "scope": "Instrumented 45-second CPU profile of the complete reduced facility request; not a timing benchmark or accepted trajectory.",
    "status": status,
    "progress": progress,
    "source_sha256": before,
    "sources_unchanged": fingerprints() == before,
    "top_cumulative": sorted(rows, key=lambda x: x["cumulative_s"], reverse=True)[:45],
    "top_own": sorted(rows, key=lambda x: x["own_s"], reverse=True)[:25],
}
with (OUT / "full_cycle_profile.json").open("x") as handle:
    json.dump(result, handle, indent=2)
print(status)
