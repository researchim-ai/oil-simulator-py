"""Held-out shock amplitudes/void fractions; same strict local T criterion."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from check_pipe_shock import run_case, symmetric_shock_reference


def evaluate(spec):
    cells, void, velocity = spec
    result = run_case(cells, void_fraction=void, inward_velocity_m_s=velocity)
    oracle = symmetric_shock_reference(void_fraction=void, inward_velocity_m_s=velocity)
    result["local_passed"] = result["temperature_error_k"] < 0.02
    result["pressure_global_passed"] = (
        result["pressure_L1_error_pa"] < 0.005 * oracle["pressure_pa"]
    )
    result["temperature_global_passed"] = result["temperature_L1_error_k"] < 0.1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--matrix", choices=("regression", "calibration", "validation", "all"), default="regression"
    )
    args = parser.parse_args()
    if args.cells < 2 or args.cells > 1000 or args.cells % 2 or not 1 <= args.workers <= 4:
        parser.error("Need even cells in [2,1000] and 1..4 workers")
    regression = [
        (args.cells, 0.97, 40.0),
        (args.cells, 0.99, 40.0),
        (args.cells, 0.98, 30.0),
        (args.cells, 0.98, 60.0),
    ]
    # Consumed by coefficient selection after a marginal failure at chi=1.5.
    # These are numerical calibration states, NOT independent validation.
    calibration = [
        (args.cells, 0.975, 35.0),
        (args.cells, 0.985, 55.0),
        (args.cells, 0.97, 50.0),
        (args.cells, 0.99, 30.0),
    ]
    # Reserved after freezing chi=1.4; do not label consumed states held-out.
    validation = [
        (args.cells, 0.972, 45.0),
        (args.cells, 0.982, 40.0),
        (args.cells, 0.988, 50.0),
        (args.cells, 0.978, 55.0),
    ]
    jobs = {
        "regression": regression,
        "calibration": calibration,
        "validation": validation,
        "all": regression + calibration + validation,
    }[args.matrix]
    if args.workers == 1:
        rows = [evaluate(spec) for spec in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(evaluate, jobs))
    passed = all(
        r["local_passed"] and r["pressure_global_passed"] and r["temperature_global_passed"]
        for r in rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": "Regression/validation shock states in the same frozen EOS; local T limit unchanged at 0.02 K, L1 T <0.1 K and L1 P <0.5%",
                "matrix": args.matrix,
                "passed": passed,
                "cases": rows,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    if not passed:
        parser.exit(1, "Held-out shock accuracy gate failed\n")
    print("Held-out shock accuracy gate passed")


if __name__ == "__main__":
    main()
