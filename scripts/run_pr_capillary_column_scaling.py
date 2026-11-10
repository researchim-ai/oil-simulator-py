"""Time finite-column / nonperiodic capillary-line coupling, with scalar audits.

Cell size, physical column inventory and time step stay fixed. Increasing
line length adds actual pipe stock, not copies of the boundary reservoir.
This qualifies one finite column and one line, not an arbitrary facility.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from check_pr_capillary_column import audit
from run_pr_active_surface import fingerprints

from surface.pr_capillary_column import PRCapillaryColumnNetwork, PRCapillaryColumnState


def run(config, *, device="cpu"):
    if set(config) != {"template", "cell_counts", "step_s"}:
        raise ValueError("Explicit coupled template, cell counts and step required")
    template = config["template"]
    if set(template) != {"model", "initial"}:
        raise ValueError("Exactly model and initial stocks required in template")
    base = PRCapillaryColumnNetwork.from_dict(template["model"], device=device)
    seed = PRCapillaryColumnState.from_dict(template["initial"])
    base.inspect(seed)
    counts = config["cell_counts"]
    if not counts or any(type(n) is not int or n < len(seed.pipe.cells) for n in counts):
        raise ValueError("Cell counts must be integers at least as large as the seed line")
    hashes = fingerprints()
    cuda = base.pipe.fluid.device.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize(base.pipe.fluid.device)

    rows = []
    for count in counts:
        gc.collect()
        if cuda:
            torch.cuda.reset_peak_memory_stats(base.pipe.fluid.device)
        raw = base.to_dict()
        raw["parameters"] = dict(raw["parameters"])
        raw["pipe"]["lengths_m"] = [base.pipe.lengths[0]] * count
        raw["pipe"]["face_elevations_m"] = [base.elevation] * (count + 1)
        nc = len(base.pipe.fluid.names)
        raw["maximum_unknowns"] = (2 * nc + 5) * count
        raw["linear_solver"] = "sparse"
        raw["parameters"]["maximum_unknowns"] = (3 + nc) * count
        raw["parameters"]["linear_solver"] = "sparse"
        model = PRCapillaryColumnNetwork.from_dict(raw, device=device)
        old = model.initialize(
            model.pipe.initialize(
                [dict(seed.pipe.cells[i % len(seed.pipe.cells)]) for i in range(count)],
                time_s=seed.time_s,
            ),
            seed.column,
            next_step_s=seed.next_step_s,
            layer_slots=seed.layer_slots,
        )
        sync()
        start = time.perf_counter()
        model.step(old, config["step_s"])
        sync()
        warmup = time.perf_counter() - start
        start = time.perf_counter()
        final, record = model.step(old, config["step_s"])
        sync()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() if cuda else None
        # Deliberately outside the timed solver: a separate scalar PR EOS,
        # profile integration and complete N/M/E/volume/entropy equations.
        independent = audit(model.to_dict(), old, final, record)
        rows.append(
            {
                "cells": count,
                "physical_model": model.to_dict(),
                "warmup_s": warmup,
                "coupled_step_s": elapsed,
                "peak_process_cuda_bytes": peak,
                "axial": record["pipe"]["axial"]["linear_solver"],
                "normal": record["pipe"]["normal"]["linear_solver"],
                "column_energy_iterations": record["column_energy_iterations"],
                "conservation_residuals": record["conservation_residuals"],
                "energy_roundoff_allowance_j": record["energy_roundoff_allowance_j"],
                "independent_audit": independent,
            }
        )
        del model, old, final, record
    if hashes != fingerprints():
        raise RuntimeError("Physical source changed during coupled scaling")
    return {
        "format": "pr_capillary_column_scaling_v1",
        "request": config,
        "source_sha256": hashes,
        "device": device,
        "device_name": torch.cuda.get_device_name() if cuda else "cpu",
        "rows": rows,
        "qualification": "Synthetic nonperiodic line and one actual finite gravity column; fixed dx/dt/column inventory, increasing pipe length; not arbitrary facility or field throughput",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a completed coupled benchmark")
    result = run(json.loads(args.config.read_text()), device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps([{k: row[k] for k in ("cells", "coupled_step_s")} for row in result["rows"]]))


if __name__ == "__main__":
    main()
