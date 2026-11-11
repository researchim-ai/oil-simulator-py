"""Scalar vs spatial-batch physical stability checks on distinct PR states.

Property-kernel scaling only: these point counts are not reservoir sizes or a
whole-network throughput measure. Both paths retain all TPD starts/fallbacks.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from run_dynamic_receiver import fingerprints

from surface.equilibrium_properties import EquilibriumFluid
from surface.torch_backend import runtime_fingerprint


def run(fluid_spec, *, device="cpu", sizes=(1, 8, 32, 128), repeats=3):
    if (
        not sizes
        or any(type(n) is not int or not 1 <= n <= 4096 for n in sizes)
        or type(repeats) is not int
        or not 1 <= repeats <= 10
    ):
        raise ValueError("Explicit positive point counts and repeat budget required")
    hashes = fingerprints()
    f = EquilibriumFluid.from_dict(fluid_spec, device=device)
    if len(f.names) != 2:
        raise ValueError("Declared manufactured binary-state distribution requires two species")
    results = []
    for n in sizes:
        p = torch.linspace(9e5, 2e6, n, device=f.device, dtype=torch.float64)
        t = torch.linspace(310, 350, n, device=f.device, dtype=torch.float64)
        xa = torch.linspace(0.1, 0.9, n, device=f.device, dtype=torch.float64)
        x = torch.stack((xa, 1 - xa), -1)
        with torch.no_grad():
            props = torch.vmap(f.tensor_homogeneous)(p, t, x)
        phi = props["log_fugacity_coefficients"]

        def scalar(indices, p=p, t=t, x=x, phi=phi):
            valid = []
            for i in indices:
                eq = f.flash_pt(
                    float(p[i].cpu()), float(t[i].cpu()), x[i], _allow_pure_coexistence=True
                )
                phase = eq["phases"][0]
                residual = (
                    (
                        torch.log(x[i])
                        + phi[i]
                        - torch.log(phase["mole_fractions"])
                        - phase["log_fugacity_coefficients"]
                    )
                    .abs()
                    .max()
                )
                valid.append(float(residual.cpu()) <= 2e-7)
            return valid

        def batch(p=p, t=t, x=x, phi=phi, scalar=scalar):
            mask = f.homogeneous_stability_mask(p, t, x, phi)
            missing = torch.nonzero(~mask).flatten().cpu().tolist()
            fallback = scalar(missing)
            if not all(fallback):
                raise RuntimeError("Unstable supplied homogeneous state")
            return len(missing)

        # Untimed warmup exercises the same operators on both paths. State
        # construction and independent reported audits stay outside timing.
        assert scalar([0]) == [True]
        batch()
        scalar_times, batch_times, fallbacks = [], [], []
        for _ in range(repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            accepted = scalar(range(n))
            if device == "cuda":
                torch.cuda.synchronize()
            scalar_times.append(time.perf_counter() - start)
            if not all(accepted):
                raise RuntimeError("Scalar reference rejected benchmark state")
            start = time.perf_counter()
            fallback = batch()
            if device == "cuda":
                torch.cuda.synchronize()
            batch_times.append(time.perf_counter() - start)
            fallbacks.append(fallback)
        results.append(
            {
                "points": n,
                "scalar_seconds": scalar_times,
                "spatial_batch_seconds": batch_times,
                "median_speedup": statistics.median(scalar_times) / statistics.median(batch_times),
                "reference_fallback_points": fallbacks,
                "all_checks_accepted": True,
            }
        )
    if fingerprints() != hashes:
        raise RuntimeError("Property benchmark sources changed")
    return {
        "format": "pr_spatial_stability_scaling_v1",
        "device": device,
        "model": f.to_dict(),
        "runtime": runtime_fingerprint(),
        "source_sha256": hashes,
        "state_distribution": {
            "pressure_pa": [9e5, 2e6],
            "temperature_k": [310, 350],
            "first_species_mole_fraction": [0.1, 0.9],
            "construction": "inclusive linspace per axis",
        },
        "repeats": repeats,
        "cases": results,
        "qualification": __doc__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--config", type=Path, default=Path("configs/fim_dynamic_pr_receiver.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite benchmark")
    raw = json.loads(args.config.read_text())
    spec = raw["simulator"]["simulation"]["dynamic_receiver"]["receiver"]["column"]["fluid"]
    result = run(spec, device=args.device, sizes=args.sizes, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "device": args.device,
                "cases": [
                    {"points": c["points"], "median_speedup": c["median_speedup"]}
                    for c in result["cases"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
