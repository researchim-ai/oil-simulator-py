"""Verify reusable PR linearization across a changed physical state and dt.

This diagnostic uses a plain FX backend to check graph guards and numerical
identity. It does not qualify a production fused Jacobian or real-time speed.
Only the pure aperture autograd primitive bypasses Dynamo introspection; all
its tensors are explicit inputs, with no captured tensors or mutable state.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from surface.pr_gravity_transport import (
    PRGravityTransportNetwork,
    PRGravityTransportState,
    _ColumnCoupling,
)
from surface.three_phase_properties import _DiskBandArea

ROOT = Path(__file__).resolve().parents[1]


def problem(model, state, case, dt):
    t = model.transport
    coupling = _ColumnCoupling(
        model, state, dt, case["order"], case["openings"], case["heat_w"], case["utility"]
    )
    seeds = [t._seed(i, s) for i, (_, s) in enumerate(state.transport.nodes)]
    y, lo, hi, evaluate, offsets, *_ = t._problem(
        state.transport,
        seeds,
        dt,
        {n: case["openings"][n] for n in t.default_openings},
        [case["heat_w"].get(n, 0.0) for n in t.names],
        case["losses"],
        external_ports=coupling.external_ports,
        active_override=coupling.active,
        guesses_override=coupling.ordinary_guesses,
        drive_torque_nm=case["drive_torque_nm"],
    )
    y, lo, hi, evaluate, _ = coupling.prepare(y, lo, hi, evaluate, offsets, dt, None)

    def residual(z):
        return evaluate(z)[0]

    return y, residual


def linearize(function, z):
    return torch.func.jacrev(function)(z)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fixture", type=Path, default=ROOT / "tests/data/surface/pr_restart_check_valve.json"
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), Path(__file__), args.fixture]
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    case = json.loads(args.fixture.read_text())
    model = PRGravityTransportNetwork.from_dict(case["model"])
    state = PRGravityTransportState.from_dict(case["state"])
    # Process-local diagnostic registration, not a global production policy.
    torch.compiler.allow_in_graph(_DiskBandArea)
    graphs = []

    def backend(graph, inputs):
        graphs.append(len(list(graph.graph.nodes)))
        return graph.forward

    compiled = torch.compile(linearize, fullgraph=True, backend=backend)
    dt = case["dt_s"]
    with torch.no_grad():
        y, function = problem(model, state, case, dt)
        start = time.perf_counter()
        actual = compiled(function, y)
        trace_s = time.perf_counter() - start
        torch.testing.assert_close(actual, linearize(function, y), rtol=2e-10, atol=3e-10)
        new, _ = model._step(
            state,
            dt,
            case["openings"],
            case["heat_w"],
            case["utility"],
            case["losses"],
            drive_torque_nm=case["drive_torque_nm"],
        )
        y2, function2 = problem(model, new, case, 1.3 * dt)
        start = time.perf_counter()
        actual2 = compiled(function2, y2)
        reused_s = time.perf_counter() - start
        torch.testing.assert_close(actual2, linearize(function2, y2), rtol=2e-10, atol=3e-10)
    if any(hashlib.sha256(f.read_bytes()).hexdigest() != hashes[str(f)] for f in paths):
        raise RuntimeError("Sources changed during graph verification")
    result = {
        "accepted": len(graphs) == 1,
        "graph_builds": len(graphs),
        "graph_nodes": graphs,
        "unknown_counts": [len(y), len(y2)],
        "first_trace_s": trace_s,
        "reused_call_s": reused_s,
        "changed_physical_state_and_dt": True,
        "independent_eager_ad_agreement": True,
        "backend": "plain_fx",
        "fused_jacobian_qualified": False,
        "source_sha256": hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
