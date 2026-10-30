"""Run the physical single-liquid surface network; this is not a reservoir run."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import HydraulicConvergenceError, LiquidNetwork, NetworkResult


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, help="JSON result; stdout if omitted")
    parser.add_argument(
        "--initial-state", type=Path, help="Previous result as Newton initial guess"
    )
    args = parser.parse_args()
    try:
        network = LiquidNetwork.from_dict(json.loads(args.config.read_text()))
        initial = (
            NetworkResult.from_dict(json.loads(args.initial_state.read_text()))
            if args.initial_state
            else None
        )
        result = network.solve(initial=initial)
    except (OSError, KeyError, TypeError, ValueError, HydraulicConvergenceError) as error:
        parser.error(str(error))
    text = json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
