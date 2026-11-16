"""Isolate the first temporal error using the actual production error operators.

No tolerances or physics are changed. Hybrid comparisons isolate differences
between real coarse/fine states; they are never integrated or accepted states.
"""

import argparse
import gzip
import hashlib
import json
import signal
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from benchmark_journal import BenchmarkJournal
from run_pr_controlled_surface import run

from surface.pr_surface_control import PRControlledSurface

ROOT = Path(__file__).resolve().parents[1]


class ProbeFinished(BaseException):
    pass


class ProbeTimeout(BaseException):
    pass


def error_sources(model, coarse, fine):
    plant, transport = model.plant, model.plant.transport
    a, b = coarse.plant, fine.plant
    groups = {}
    for i, (name, node) in enumerate(a.transport.nodes):
        nodes = list(b.transport.nodes)
        nodes[i] = (name, node)
        hybrid = replace(b.transport, nodes=tuple(nodes))
        groups[f"transport_node:{name}"] = transport._error(hybrid, b.transport)
    groups["rotors"] = transport._error(
        replace(b.transport, rotors=a.transport.rotors), b.transport
    )
    for name, column in a.columns:
        columns = tuple((n, column if n == name else s) for n, s in b.columns)
        groups[f"column:{name}"] = plant._error(replace(b, columns=columns), b)
    groups["column_walls"] = plant._error(replace(b, walls=a.walls), b)
    for field in ("positions", "integrals", "trips"):
        hybrid = replace(fine, **{field: getattr(coarse, field)})
        if field == "integrals":
            hybrid = replace(hybrid, inactive_feedback=coarse.inactive_feedback)
        groups[field] = model._error(hybrid, fine)
    actual = model._error(coarse, fine)
    assert actual == max(groups.values()), (actual, groups)
    return actual, dict(sorted(groups.items(), key=lambda item: item[1], reverse=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-s", type=int, default=90)
    args = parser.parse_args()
    if not 1 <= args.budget_s <= 300:
        parser.error("Budget must be 1..300 seconds")
    config = ROOT / "configs/surface_pr_integrated_facility.json"
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), Path(__file__), config]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    journal = BenchmarkJournal(args.output, {"scope": __doc__, "source_sha256": hashes})
    original = PRControlledSurface._temporal_trial
    result = {}

    def observed(self, *arguments, **kwargs):
        coarse, fine, one, two = original(self, *arguments, **kwargs)
        error, sources = error_sources(self, coarse, fine)
        result.update(error=error, sources=sources, coarse=coarse.to_dict(), fine=fine.to_dict())
        with gzip.open(args.output / "half_steps.json.gz", "wt") as handle:
            json.dump({"model": self.to_dict(), "steps": [one, two]}, handle, allow_nan=False)
        raise ProbeFinished()

    def expired(*_args):
        raise ProbeTimeout()

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.budget_s)
    try:
        with patch.object(PRControlledSurface, "_temporal_trial", observed):
            run(json.loads(config.read_text()))
    except ProbeFinished:
        pass
    except ProbeTimeout:
        journal.checkpoint([], [], {})
        raise SystemExit("Diagnostic budget exhausted; no error attribution qualified") from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert result, "No temporal trial captured"
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    journal.complete(result)
    print(json.dumps({"error": result["error"], "sources": result["sources"]}, indent=2))


if __name__ == "__main__":
    main()
