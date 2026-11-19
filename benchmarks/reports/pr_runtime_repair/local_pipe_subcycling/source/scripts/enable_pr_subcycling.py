"""Explicitly select locally subcycled pipes without changing physical stocks."""

import argparse
import json
from pathlib import Path

from surface.pr_subcycled_surface import PRSubcycledSurface
from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState


def enable(config, pipe_nodes, *, local_error_fraction=0.1):
    if set(config) != {"model", "initial", "schedule"} or set(config["initial"]) != {"state"}:
        raise ValueError("Supply a controlled-surface configuration with an explicit checkpoint")
    model = PRSubcycledSurface(
        PRControlledSurface.from_dict(config["model"]),
        pipe_nodes,
        local_error_fraction=local_error_fraction,
    )
    state = model.adopt(PRControlledSurfaceState.from_dict(config["initial"]["state"]))
    return {
        "model": model.to_dict(),
        "initial": {"state": state.to_dict()},
        "schedule": config["schedule"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--pipes", nargs="+", required=True)
    parser.add_argument("--local-error-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = enable(
        json.loads(args.config.read_text()),
        args.pipes,
        local_error_fraction=args.local_error_fraction,
    )
    # Never overwrite the source checkpoint or another prepared configuration.
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
