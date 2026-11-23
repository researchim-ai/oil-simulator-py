"""Exclusive call-tree costs of the accepted detailed pipe/facility schedule.

No per-kernel synchronization is inserted. Inclusive timers overlap; exclusive
costs subtract nested instrumented calls. Independent audit is outside timing.
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import torch
from check_pr_controlled_surface import audit
from run_pr_controlled_surface import run

from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork
from surface.pr_gravity_transport import PRGravityTransportNetwork, _ColumnCoupling
from surface.pr_subcycled_surface import PRSubcycledSurface
from surface.pr_surface_control import PRControlledSurface
from surface.stratified_pr import StratifiedPRColumn
from surface.torch_nonlinear import TorchNewton

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), Path(__file__), args.config]
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    EquilibriumFluid._compiled_chemical_cpu = True
    TorchNewton._dense_jacobian_mode = "reverse"
    stack, costs = [], defaultdict(lambda: {"inclusive_s": 0.0, "exclusive_s": 0.0, "calls": 0})

    def instrument(original, name):
        @wraps(original)
        def wrapped(*a, **kw):
            if torch.compiler.is_compiling():
                return original(*a, **kw)
            frame = [time.perf_counter(), 0.0]
            stack.append(frame)
            try:
                return original(*a, **kw)
            finally:
                elapsed = time.perf_counter() - frame[0]
                stack.pop()
                costs[name]["inclusive_s"] += elapsed
                costs[name]["exclusive_s"] += elapsed - frame[1]
                costs[name]["calls"] += 1
                if stack:
                    stack[-1][1] += elapsed

        return wrapped

    targets = [
        (PRSubcycledSurface, ("_fast_stage", "_slow_stage")),
        (PRControlledSurface, ("_temporal_trial", "inspect")),
        (PRGravityTransportNetwork, ("inspect", "_step")),
        (
            PREquilibriumTransportNetwork,
            ("inspect", "_state_data", "_stable_sets", "_step", "_problem"),
        ),
        (_ColumnCoupling, ("prepare", "accept")),
        (StratifiedPRColumn, ("inspect", "sample")),
        (EquilibriumFluid, ("flash_pt", "flash_uv", "_stability_many", "_stability")),
        (TorchNewton, ("solve",)),
    ]
    # Compiler/property startup is not attributed to the warm run.
    run(config, device=args.device)
    with ExitStack() as scope:
        for cls, names in targets:
            for name in names:
                scope.enter_context(
                    patch.object(
                        cls, name, instrument(getattr(cls, name), cls.__name__ + "." + name)
                    )
                )
        start = time.perf_counter()
        report = run(config, device=args.device)
        elapsed = time.perf_counter() - start
    report = json.loads(json.dumps(report))
    checked = audit(config, report)
    if checked.get("accepted") is not True:
        raise ValueError("Independent audit rejected the measured schedule")
    if any(hashlib.sha256(f.read_bytes()).hexdigest() != hashes[str(f)] for f in paths):
        raise RuntimeError("Sources changed during profiling")
    result = {
        "wall_s": elapsed,
        "device": args.device,
        "simulated_duration_s": sum(c["duration_s"] for c in config["schedule"]),
        "costs": dict(sorted(costs.items(), key=lambda kv: -kv[1]["exclusive_s"])),
        "audit": checked,
        "source_sha256": hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
