"""Check actual controlled-plant support against uncompressed reverse AD."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from audit_pr_controlled_mixed import audit
from run_pr_controlled_surface import run

from surface.pr_collocation import PRCollocation
from surface.torch_linear import ReverseColoredJacobian


def check(config):
    original = PRCollocation._solve
    checked = {}

    def verify(self, fn, initial, lo, hi, pattern, update):
        key = (len(initial), tuple(pattern[0]), tuple(pattern[1]))
        if key not in checked:
            planner = ReverseColoredJacobian(len(initial), *pattern, "cpu")
            errors = []
            for point in (
                initial,
                (initial + torch.linspace(-1e-9, 1e-9, len(initial), dtype=initial.dtype)).clamp(
                    lo, hi
                ),
            ):
                expected = torch.func.jacrev(fn)(point)
                recovered = planner.dense(fn, point)
                torch.testing.assert_close(recovered, expected, rtol=2e-12, atol=2e-12)
                errors.append(float((recovered - expected).abs().max()))
            checked[key] = {
                "unknowns": len(initial),
                "declared_coordinates": len(pattern[0]),
                "reverse_directions": planner.color_count,
                "maximum_absolute_errors": errors,
                "support_sha256": hashlib.sha256(repr(key).encode()).hexdigest(),
            }
        return original(self, fn, initial, lo, hi, pattern, update)

    PRCollocation._solve = verify
    try:
        report = json.loads(json.dumps(run(config, integrator="sdirk2-common")))
    finally:
        PRCollocation._solve = original
    certificate = audit(config, report)
    if not certificate["accepted"] or not checked:
        raise ValueError("Graph audit failed")
    return {"accepted": True, "graphs": list(checked.values()), "independent_audit": certificate}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = check(json.loads(args.request.read_text()))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
