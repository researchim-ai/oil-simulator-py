"""Same complete FIM: repeated versus shared pressure-interval searches.

Property arithmetic, knots, quadrature and physical tolerances are identical.
Only pressure search reuse within one property evaluation changes in memory.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import run_structured_well_forecast as driver

from surface import blackoil_pipe_batch, blackoil_pvto_batch
from surface.blackoil_pressure_lookup import PressureLookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup", choices=["repeated", "shared"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve()
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    searches = 0
    queries = 0

    class Counted(PressureLookup):
        def __init__(self, *a, **kw):
            nonlocal searches
            searches += 1
            super().__init__(*a, **kw)

        def interpolate(self, values):
            nonlocal queries
            queries += 1
            return super().interpolate(values)

    class Repeated:
        def __init__(self, *a, **kw):
            self.a, self.kw = a, kw

        def interpolate(self, values):
            return Counted(*self.a, **self.kw).interpolate(values)

    selected = Counted if args.lookup == "shared" else Repeated
    saved = (blackoil_pipe_batch.PressureLookup, blackoil_pvto_batch.PressureLookup)
    argv = sys.argv
    blackoil_pipe_batch.PressureLookup = selected
    blackoil_pvto_batch.PressureLookup = selected
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
        blackoil_pipe_batch.PressureLookup, blackoil_pvto_batch.PressureLookup = saved
    if hashlib.sha256(source.read_bytes()).hexdigest() != before or not searches:
        raise RuntimeError("Shared lookup benchmark changed or policy did not execute")
    with (args.output / "lookup_policy.json").open("x") as handle:
        json.dump(
            {
                "scope": __doc__,
                "policy": args.lookup,
                "pressure_interval_searches": searches,
                "property_column_queries": queries,
                "wrapper_source": str(source),
                "wrapper_sha256": before,
                "sources_unchanged": True,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
