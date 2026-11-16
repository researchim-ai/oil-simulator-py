"""Full FIM reference comparison: global versus stream-local pressure knots.

Both variants use current PVTO lookup/admission and identical physical inputs.
Only the grid factory is changed in memory for the benchmark process.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import run_structured_well_forecast as driver

from surface import blackoil_pipe_batch


class GlobalGrid:
    def __init__(self, model):
        self.knots = model._fixed_pressures
        self.maximum_fixed_knots = len(model._fixed_values)

    def fixed(self, masses):
        return self.knots[None, :].expand(len(masses), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=["global", "stream"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--days", type=float, default=1)
    parser.add_argument("--report-days", type=float, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve()
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    original = blackoil_pipe_batch.get_pressure_grid
    calls = 0
    fixed_calls = 0
    maximum_fixed_knots = 0
    maximum_fixed_tensor_entries = 0

    class Counted:
        def __init__(self, grid):
            self.grid = grid
            self.maximum_fixed_knots = grid.maximum_fixed_knots

        def fixed(self, masses):
            nonlocal fixed_calls, maximum_fixed_tensor_entries
            result = self.grid.fixed(masses)
            fixed_calls += 1
            maximum_fixed_tensor_entries = max(maximum_fixed_tensor_entries, result.numel())
            return result

    def factory(model):
        nonlocal calls, maximum_fixed_knots
        calls += 1
        grid = GlobalGrid(model) if args.grid == "global" else original(model)
        maximum_fixed_knots = max(maximum_fixed_knots, grid.maximum_fixed_knots)
        return Counted(grid)

    argv = sys.argv
    blackoil_pipe_batch.get_pressure_grid = factory
    try:
        sys.argv = [
            str(Path(driver.__file__)),
            "--config",
            str(args.config),
            "--days",
            str(args.days),
            "--report-days",
            str(args.report_days),
            "--output",
            str(args.output),
        ]
        driver.main()
    finally:
        sys.argv = argv
        blackoil_pipe_batch.get_pressure_grid = original
    if hashlib.sha256(source.read_bytes()).hexdigest() != before or not fixed_calls:
        raise RuntimeError("Grid policy source changed or did not execute")
    with (args.output / "grid_policy.json").open("x") as handle:
        json.dump(
            {
                "scope": __doc__,
                "policy": args.grid,
                "factory_calls": calls,
                "fixed_calls": fixed_calls,
                "maximum_fixed_knots": maximum_fixed_knots,
                "maximum_fixed_tensor_entries": maximum_fixed_tensor_entries,
                "wrapper_source": str(source),
                "wrapper_sha256": before,
                "sources_unchanged": True,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
