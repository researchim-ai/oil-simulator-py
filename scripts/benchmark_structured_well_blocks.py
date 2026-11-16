"""Large algebraic well block, checked against a closed-form rank-one inverse.

No PVT, reservoir integration or dense reference matrices are allocated here.
Actual producing-well qualifications are separate run_structured_well_forecast runs.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from benchmark_journal import BenchmarkJournal

from simulator.tree_well_system import TreeWellSystem
from surface.tree_pressure_tangent import TreePressureTangent

ROOT = Path(__file__).resolve().parents[1]


def payload_bytes(operator):
    seen = set()

    def size(x):
        if torch.is_tensor(x):
            storage = x.untyped_storage()
            key = (str(x.device), storage.data_ptr())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(x, (TreeWellSystem, TreePressureTangent)):
            return sum(size(v) for v in vars(x).values())
        if isinstance(x, (tuple, list)):
            return sum(size(v) for v in x)
        return 0

    return size(operator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wells", type=int, default=4096)
    parser.add_argument("--perforations-per-well", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.wells < 2
        or args.perforations_per_well < 1
        or args.repeats < 1
        or torch.get_num_threads() != 1
    ):
        parser.error("wells>=2, perforations>=1, repeats>=1 and one Torch CPU thread required")
    nw = args.wells
    nc = nw * args.perforations_per_well
    cells = nw * 2
    names = [f"p{i}" for i in range(nw)] + ["header", "root"]
    spec = {
        "model": "blackoil_tree_pressure_tangent_v1",
        "node_names": names,
        "parent_names": ["header"] * nw + ["root", None],
        "source_names": names[:-2],
        "pressure_coefficients": [1.0] * (nw + 1) + [0.0],
        "mass_coefficients": [[0.001, 0.002, 0.003]] * nw + [[0.5, 0.2, 0.8], [0.0, 0.0, 0.0]],
        "valve_names": [],
        "valve_nodes": [],
        "valve_coefficients": [],
    }
    paths = [
        *sorted((ROOT / "src/simulator").glob("*.py")),
        *sorted((ROOT / "src/surface").glob("*.py")),
        Path(__file__),
    ]

    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    before = hashes()
    journal = BenchmarkJournal(
        args.output,
        {
            "scope": __doc__,
            "source_sha256": before,
            "wells": nw,
            "perforations": nc,
            "cells": cells,
            "seed": 27,
            "spec": spec,
            "repeats": args.repeats,
            "device": "cpu",
            "threads": 1,
        },
    )
    owners = torch.arange(nc) % nw
    conn_cell = torch.arange(nc) % cells
    rng = torch.Generator().manual_seed(27)
    dq = torch.randn(nc, 3, 3, dtype=torch.float64, generator=rng) * 1e-5
    cv = torch.full((nc, 3), -1e-5, dtype=torch.float64)
    weight = torch.ones_like(cv)
    rate = torch.zeros(nw, dtype=torch.bool)
    surface = ~rate
    started = time.perf_counter()
    op = TreeWellSystem(
        spec, names[:-2], owners, conn_cell, dq, cv, weight, rate, surface, has_surface=True
    )
    op.prepare()
    construction = time.perf_counter() - started
    U = torch.zeros(nw, 3, dtype=torch.float64).index_add_(0, owners, cv)
    local = torch.tensor([0.001, 0.002, 0.003], dtype=torch.float64)
    header = torch.tensor([0.5, 0.2, 0.8], dtype=torch.float64)
    diagonal = 1 - (U * local).sum(1)
    v = (U * header).sum(1)
    rhs = torch.sin(torch.arange(nw, dtype=torch.float64))
    expected = rhs / diagonal + (1 / diagonal) * (
        (v * rhs / diagonal).sum() / (1 - (v / diagonal).sum())
    )
    torch.testing.assert_close(op.solve(rhs), expected, rtol=1e-10, atol=1e-10)
    dx = torch.randn(cells, 3, dtype=torch.float64, generator=rng)
    dm = (dq * dx[conn_cell, None, :]).sum(2)
    source_dm = torch.zeros(nw, 3, dtype=torch.float64).index_add_(0, owners, dm)
    expected_b = -(source_dm * local).sum(1) - (source_dm.sum(0) * header).sum()
    torch.testing.assert_close(op.Bx(dx), expected_b, rtol=1e-10, atol=1e-12)
    samples = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        op.solve(rhs)
        op.Bx(dx)
        samples.append(time.perf_counter() - started)
    if hashes() != before:
        raise RuntimeError("Source changed during qualification")
    result = {
        "scope": __doc__,
        "source_sha256": before,
        "sources_unchanged": True,
        "wells": nw,
        "perforations": nc,
        "nodes": nw + 2,
        "constructor_and_admission_s": construction,
        "median_solve_plus_Bx_s": statistics.median(samples),
        "samples_s": samples,
        "tensor_payload_bytes_including_borrowed_inputs": payload_bytes(op),
        "hypothetical_dense_B_and_D_bytes": 8 * (nw * nc * 3 + nw * nw),
        "dense_B_D_allocated": False,
        "closed_form_inverse_accepted": True,
        "closed_form_Bx_accepted": True,
        "iterative_fallback": op.iterative_fallback,
        "factor_bad": bool(op.factor_bad),
        "maximum_scaled_residual": float(op.maximum_relative_residual),
    }
    if result["factor_bad"]:
        raise RuntimeError("Structured residual guard failed")
    journal.complete(result)
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
