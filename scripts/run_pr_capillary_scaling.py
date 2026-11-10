"""Measure the actual sparse PR/capillary step on repeated periodic domains.

The dense small periodic pattern supplies a numerical reference for every
large cell. Physical cell size and time step stay fixed; domain length and
owned fluid inventory grow. This is not a field/Norne performance benchmark.
"""

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
from run_pr_active_surface import fingerprints

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
from surface.pr_capillary_axial import PRCapillaryAxialPipe
from surface.pr_capillary_pressure import PRCapillaryPressure


def run(config, *, device="cpu"):
    if set(config) != {"template", "cell_counts", "step_s"}:
        raise ValueError("Explicit periodic template, cell counts and step required")
    if set(config["template"]) != {"model", "initial"}:
        raise ValueError("The periodic template contains exactly a model and initial stocks")
    base = PRCapillaryAxialPipe.from_dict(config["template"]["model"], device=device)
    seed = PRActiveMultifluidState.from_dict(config["template"]["initial"])
    if base.pipe.boundary_mode != "periodic" or any(v is not None for v in base.boundaries):
        raise ValueError("Repeated reference domain must be periodic")
    counts = config["cell_counts"]
    if not counts or any(
        type(n) is not int or n < len(seed.cells) or n % len(seed.cells) for n in counts
    ):
        raise ValueError("Scaling cell counts must be positive whole reference patterns")
    hashes = fingerprints()
    reference, _ = base.step_coupled(seed, config["step_s"])
    cuda = base.pipe.fluid.device.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize(base.pipe.fluid.device)

    rows = []
    for count in counts:
        gc.collect()
        if cuda:
            torch.cuda.reset_peak_memory_stats(base.pipe.fluid.device)
        sync()
        start = time.perf_counter()
        raw = base.pipe.to_dict()
        raw["lengths_m"] = [base.pipe.lengths[0]] * count
        raw["face_elevations_m"] = (
            None if base.pipe.elevations is None else [base.pipe.elevations[0]] * (count + 1)
        )
        p = PRActiveMultifluidPipe.from_dict(raw, device=device)
        params = {
            k: v for k, v in base.source.to_dict().items() if k not in ("model", "pipe_model_id")
        }
        params["maximum_unknowns"] = (3 + len(base.source.active)) * count
        params["linear_solver"] = "sparse"
        source = PRCapillaryPressure(p, **params)
        old = p.initialize(
            [dict(seed.cells[i % len(seed.cells)]) for i in range(count)], time_s=seed.time_s
        )
        model = PRCapillaryAxialPipe(
            source,
            maximum_unknowns=(2 * len(p.fluid.names) + 5) * count,
            volume_diffusion=base.volume_diffusion,
            linear_solver="sparse",
        )
        # Compile topology and warm the same operation without changing old stocks.
        model.step_coupled(old, config["step_s"])
        sync()
        setup_time = time.perf_counter() - start
        start = time.perf_counter()
        end, report = model.step_coupled(old, config["step_s"])
        sync()
        elapsed = time.perf_counter() - start
        worst = 0.0
        for i, cell in enumerate(end.cells):
            expected = dict(reference.cells[i % len(seed.cells)])
            for name, phase in cell:
                other = expected[name]
                quantities = [
                    *(
                        (a, b, 1e-22 + 1e-10 * max(abs(a), abs(b)))
                        for a, b in zip(phase.component_moles, other.component_moles)
                    ),
                    (phase.volume_m3, other.volume_m3, 1e-24 + 1e-10 * other.volume_m3),
                    (
                        phase.momentum_kg_m_s,
                        other.momentum_kg_m_s,
                        1e-22 + 1e-8 * abs(other.momentum_kg_m_s),
                    ),
                    (phase.temperature_k, other.temperature_k, 1e-8),
                    (
                        phase.total_energy_j,
                        other.total_energy_j,
                        128 * max(math.ulp(phase.total_energy_j), math.ulp(other.total_energy_j))
                        + 1e-22,
                    ),
                ]
                for a, b, tol in quantities:
                    if not math.isfinite(a) or abs(a - b) > tol:
                        raise ValueError("Scaled state differs from the dense periodic reference")
                    worst = max(worst, abs(a - b) / tol)
        rows.append(
            {
                "cells": count,
                "setup_and_warmup_s": setup_time,
                "coupled_step_s": elapsed,
                "peak_process_cuda_bytes": torch.cuda.max_memory_allocated() if cuda else None,
                "axial": report["axial"]["linear_solver"],
                "normal": report["normal"]["linear_solver"],
                "maximum_reference_tolerance_fraction": worst,
                "component_energy_balance_accepted": True,
            }
        )
        del model, source, p, old, end, report
    if hashes != fingerprints():
        raise RuntimeError("Source changed during scaling run")
    return {
        "format": "pr_capillary_sparse_scaling_v1",
        "request": config,
        "source_sha256": hashes,
        "device": device,
        "device_name": torch.cuda.get_device_name() if cuda else "cpu",
        "rows": rows,
        "qualification": "Fixed cell size and step, repeated synthetic periodic domain; dense-reference agreement and actual coupled solve; not field throughput or a large gravitational-column benchmark",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = run(json.loads(args.config.read_text()), device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["rows"]))


if __name__ == "__main__":
    main()
