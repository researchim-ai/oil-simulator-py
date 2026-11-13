"""Measure the complete ordinary-network stability audit, scalar vs spatial.

This is the stability stage, not a complete time integration benchmark. Both
paths must retain the same stable assemblage at every supplied checkpoint node.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import torch

    from surface import equilibrium_properties as properties
    from surface.equilibrium_cache import cached_flashes, cached_stability_checks
    from surface.pr_equilibrium_transport import (
        PREquilibriumTransportNetwork,
        PREquilibriumTransportState,
    )
    from surface.torch_backend import runtime_fingerprint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--legacy-curvature",
        action="store_true",
        help="Include the former logit curvature test in scalar reference",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Positive repetition count required")
    if args.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Explicit CUDA_VISIBLE_DEVICES required")
    config = args.config.resolve()
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), Path(__file__).resolve(), config]

    def fingerprint():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    before = fingerprint()
    output = args.output.resolve()
    output.parent.mkdir(exist_ok=True, parents=True)
    if output.exists():
        raise FileExistsError("Use a fresh immutable result path")
    cfg = json.loads(config.read_text())
    model = PREquilibriumTransportNetwork.from_dict(
        cfg["model"]["plant"]["transport"], device=args.device
    )
    state = PREquilibriumTransportState.from_dict(cfg["initial"]["state"]["plant"]["transport"])
    data = [model._state_data(i, s) for i, (_, s) in enumerate(state.nodes)]
    times = {"scalar": [], "spatial": []}
    original_kernel = properties.stability_candidates

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    with (
        cached_flashes(model.fluid, enabled=False),
        cached_stability_checks(model.fluid, enabled=False),
    ):
        # Alternating order limits drift from thermal/host conditions. Warm
        # each implementation separately, then measure the unchanged points.
        for repetition in range(args.repeats + 1):
            modes = ("scalar", "spatial") if repetition % 2 == 0 else ("spatial", "scalar")
            for mode in modes:
                model._batch_stability = mode == "spatial"
                properties.stability_candidates = (
                    (lambda *a, **kw: original_kernel(*a, **kw, _scaled_curvature=False))
                    if args.legacy_curvature and mode == "scalar"
                    else original_kernel
                )
                synchronize()
                start = time.perf_counter()
                result = model._stable_all(data)
                synchronize()
                elapsed = time.perf_counter() - start
                if any(q is not None for q in result):
                    raise ValueError("Supplied checkpoint is not on a stable admissible branch")
                print(
                    json.dumps({"mode": mode, "warmup": repetition == 0, "elapsed_s": elapsed}),
                    file=sys.stderr,
                    flush=True,
                )
                if repetition:
                    times[mode].append(elapsed)
    properties.stability_candidates = original_kernel
    model._batch_stability = True
    model.inspect(state)
    unchanged = fingerprint() == before
    result = {
        "format": "surface_spatial_stability_benchmark_v1",
        "accepted": unchanged,
        "scalar_reference_curvature": "legacy_logits"
        if args.legacy_curvature
        else "scaled_square_root",
        "scope": "Complete ordinary-node stability audit only; not timestep throughput or experimental validation",
        "device": args.device,
        "nodes": len(data),
        "component_registry": list(model.fluid.names),
        "times_s": times,
        "speedup": statistics.median(times["scalar"]) / statistics.median(times["spatial"]),
        "all_points_keep_same_admissible_stable_branch": True,
        "source_sha256": before,
        "sources_unchanged": unchanged,
        "runtime": runtime_fingerprint(),
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("accepted", "nodes", "speedup")}, indent=2))
    if not unchanged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
