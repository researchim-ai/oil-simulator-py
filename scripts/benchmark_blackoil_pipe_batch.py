"""Heterogeneous independent-pipe CPU benchmark with unchanged scalar physics.

Seeded geometry/composition variation, identical inputs and physical checks.
This kernel timing is separate from whole-reservoir qualifications.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from benchmark_journal import BenchmarkJournal

from surface.blackoil_network import BlackOilProductionNetwork
from surface.blackoil_pipe import BlackOilPipe
from surface.blackoil_pipe_batch import PipeBatch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipes", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.pipes <= 256 or args.repeats < 1 or torch.get_num_threads() != 1:
        parser.error("2..256 pipes, positive repeats and one Torch thread required")
    config = ROOT / "configs/fim_blackoil_surface.json"
    spec = json.loads(config.read_text())["simulation"]["physical_surface_network"]["network"]
    model = BlackOilProductionNetwork.from_dict(spec).model
    rng = np.random.default_rng(57)
    pipes = [
        BlackOilPipe(
            f"p{i}",
            float(rng.uniform(100, 450)),
            float(rng.uniform(0.07, 0.14)),
            float(rng.uniform(0, 40)),
            float(rng.uniform(1e-6, 3e-5)),
            float(rng.uniform(0, 3)),
        )
        for i in range(args.pipes)
    ]
    boundary = rng.uniform(1e6, 1.5e6, args.pipes).tolist()
    # Excess gas compositions exercise the saturated branch with different
    # gas fractions; dedicated regression covers moving bubble points.
    rates = [
        [float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.5, 2.0)), float(rng.uniform(0.08, 0.15))]
        for _ in pipes
    ]
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        ROOT / "src/simulator/pvt.py",
        config,
        Path(__file__),
    ]
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    journal = BenchmarkJournal(
        args.output,
        {
            "scope": __doc__,
            "source_sha256": before,
            "device": "cpu",
            "threads": 1,
            "seed": 57,
            "pipes": [vars(p) for p in pipes],
            "boundary_pa": boundary,
            "masses_kg_s": rates,
            "repeats": args.repeats,
        },
    )
    kernel = PipeBatch(pipes, model)
    # Explicit warm-up of both paths. Never time the two solvers concurrently.
    pipes[0].required_inlet(boundary[0], rates[0], model)
    kernel.solve(boundary, rates)
    samples = {"scalar": [], "torch_batch": []}
    for _ in range(args.repeats):
        started = time.perf_counter()
        scalar = [p.required_inlet(b, m, model) for p, b, m in zip(pipes, boundary, rates)]
        samples["scalar"].append(time.perf_counter() - started)
        started = time.perf_counter()
        batch = kernel.solve(boundary, rates)
        samples["torch_batch"].append(time.perf_counter() - started)
    pressure = max(
        abs(a["inlet_pressure_pa"] - b["inlet_pressure_pa"]) for a, b in zip(scalar, batch)
    )
    derivative = max(
        abs(a["pressure_derivatives"][k] - b["pressure_derivatives"][k])
        / max(1.0, abs(a["pressure_derivatives"][k]))
        for a, b in zip(scalar, batch)
        for k in a["pressure_derivatives"]
    )
    energy = max(
        abs(a["friction_dissipation_w"] - b["friction_dissipation_w"])
        / max(1.0, abs(a["friction_dissipation_w"]))
        for a, b in zip(scalar, batch)
    )
    # Fixed, declared tolerances. They are not derived from observed errors.
    if pressure > 0.001 or derivative > 1e-7 or energy > 1e-8:
        raise RuntimeError("Batch/scalar equivalence budget exceeded")
    if before != hashes():
        raise RuntimeError("Source changed during qualification")
    medians = {k: statistics.median(v) for k, v in samples.items()}
    journal.complete(
        {
            "scope": __doc__,
            "sources_unchanged": True,
            "source_sha256": before,
            "samples_s": samples,
            "medians_s": medians,
            "speedup_ratio": medians["scalar"] / medians["torch_batch"],
            "maximum_pressure_difference_pa": pressure,
            "maximum_scaled_derivative_difference": derivative,
            "maximum_scaled_dissipation_difference": energy,
            "pipes": len(pipes),
            "scalar_retries": 0,
            "accepted": True,
        }
    )


if __name__ == "__main__":
    main()
