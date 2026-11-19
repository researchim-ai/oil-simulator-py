"""Prepare an explicit engineering model with conservative Darcy replacements."""

import argparse
import json
from pathlib import Path

# Public compatibility imports used by existing validation scripts.
from surface.pr_model_reduction import merge_pipe_stocks, reduce_cell, reduce_cells  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["gas_recycle_0"],
        help="Explicit horizontal pipe cells to replace; no automatic selection",
    )
    args = parser.parse_args()
    request, mapping = reduce_cells(json.loads(args.config.read_text()), args.cells)
    with args.output.open("x") as handle:
        json.dump(request, handle, indent=2, allow_nan=False)
    with args.output.with_suffix(".mapping.json").open("x") as handle:
        json.dump(mapping, handle, indent=2, allow_nan=False)
    print(json.dumps(mapping))


if __name__ == "__main__":
    main()
