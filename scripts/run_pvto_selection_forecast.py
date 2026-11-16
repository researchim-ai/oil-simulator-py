"""Full current FIM: benchmark only all-table versus selected-table PVTO lookup.

Both modes retain batched admission, the same integral and physical model.
Reference builds the old all-table matrix once per column for both gathers.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import run_structured_well_forecast as driver
from pvto_reference import AllTablesPVTO

from surface import blackoil_pipe_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup", choices=["all_tables", "selected"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [Path(__file__).resolve(), Path(__file__).with_name("pvto_reference.py").resolve()]
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    original = blackoil_pipe_batch.PackedPVTO
    base = AllTablesPVTO if args.lookup == "all_tables" else original
    calls = 0

    class Counted(base):
        def interpolate(self, *a, **kw):
            nonlocal calls
            calls += 1
            return super().interpolate(*a, **kw)

    argv = sys.argv
    blackoil_pipe_batch.PackedPVTO = Counted
    try:
        sys.argv = [
            str(Path(driver.__file__)),
            "--config",
            str(args.config),
            "--days",
            "1",
            "--report-days",
            "1",
            "--output",
            str(args.output),
        ]
        driver.main()
    finally:
        sys.argv = argv
        blackoil_pipe_batch.PackedPVTO = original
    if before != hashes() or not calls:
        raise RuntimeError("PVTO benchmark source changed or policy did not execute")
    with (args.output / "lookup_policy.json").open("x") as handle:
        json.dump(
            {
                "scope": __doc__,
                "policy": args.lookup,
                "lookup_calls": calls,
                "source_sha256": before,
                "sources_unchanged": True,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
