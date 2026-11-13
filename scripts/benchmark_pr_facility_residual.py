"""Scalar/grouped EOS and aperture comparison in the actual coupled plant Newton residual.

This measures residual/Jacobian assembly, not accepted simulation time or field
throughput. Both paths use identical states, root selections and tolerances.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState
from surface.torch_backend import runtime_fingerprint
from surface.torch_nonlinear import TorchNewton


def fingerprints():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), *sorted((root / "src/surface").glob("*.py"))]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class _Captured(RuntimeError):
    pass


def capture(config, device, batch):
    model = PRControlledSurface.from_dict(config["model"], device=device)
    state = PRControlledSurfaceState.from_dict(config["initial"]["state"])
    model.inspect(state)
    transport = model.plant.transport
    for column in model.plant.columns.values():
        column._batch_column_eos = batch
    original_problem = transport._problem
    transport._problem = lambda *a, **kw: original_problem(*a, **kw, _batch_phase_eos=batch)
    original_step = model.plant._step
    model.plant._step = lambda *a, **kw: original_step(*a, **kw, _batch_aperture_rays=batch)
    original_solve = TorchNewton.solve
    captured = []

    def solve(solver, function, initial, *args, **kwargs):
        if initial.numel() > 50:
            captured.append((function, initial.detach().clone()))
            raise _Captured()
        return original_solve(solver, function, initial, *args, **kwargs)

    TorchNewton.solve = solve
    try:
        model._step(state, 1e-4, {}, {}, {})
    except _Captured:
        pass
    finally:
        TorchNewton.solve = original_solve
    if len(captured) != 1:
        raise RuntimeError("No unique common plant Newton problem captured")
    return captured[0]


def run(config, device="cpu", repeats=3):
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError("Use 1..10 timed repetitions")
    before = fingerprints()
    scalar, y = capture(config, device, False)
    batch, by = capture(config, device, True)
    if not torch.equal(y, by):
        raise ValueError("Scalar and batch initial coordinates differ")
    cuda = y.device.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize(y.device)

    results = {}
    for kind in ("residual", "forward_jacobian"):
        answers = {}
        times = {}
        for label, fn in (("scalar", scalar), ("batch", batch)):
            evaluate = fn if kind == "residual" else torch.func.jacfwd(fn)
            answers[label] = evaluate(y).detach()
            values = []
            for _ in range(repeats):
                sync()
                start = time.perf_counter()
                evaluate(y)
                sync()
                values.append(time.perf_counter() - start)
            times[label] = values
        a, b = answers["scalar"], answers["batch"]
        scale = 1e-10 + 1e-10 * torch.maximum(a.abs(), b.abs())
        worst = float(((a - b).abs() / scale).max().cpu())
        if not bool((torch.isfinite(a).all() & torch.isfinite(b).all()).cpu()) or worst > 1:
            raise ValueError(f"{kind} differs between scalar and grouped EOS: {worst}")
        results[kind] = {
            "times_s": times,
            "scalar_median_s": statistics.median(times["scalar"]),
            "batch_median_s": statistics.median(times["batch"]),
            "speedup": statistics.median(times["scalar"]) / statistics.median(times["batch"]),
            "maximum_tolerance_fraction": worst,
            "checked_scalars": a.numel(),
        }
    if before != fingerprints():
        raise RuntimeError("Sources changed during benchmark")
    return {
        "format": "pr_facility_residual_benchmark_v1",
        "accepted": True,
        "qualification": "Coupled plant residual/Jacobian assembly only; not whole-network integration qualification or experimental validation",
        "unknowns": y.numel(),
        "device": device,
        "runtime": runtime_fingerprint(),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "source_sha256": before,
        "results": results,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output path")
    result = run(json.loads(args.config.read_text()), args.device, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
