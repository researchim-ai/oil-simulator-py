"""Compare final-state annular root solvers at identical synthetic inputs.

This times only the fixed-inventory momentum source, excluding PR recovery,
transport, ports and the network. Bisection shares the constitutive kernel;
independent physical/numerical references remain in the source tests.
"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch

from surface.annular_film import annular_response, interfacial_conductance
from surface.annular_implicit_source import implicit_annular_velocities

ROOT = Path(__file__).resolve().parents[1]


def bisection(alpha, diameter, mu_l, mu_g, rho, ug0, ul0, mg, ml, lengths, dt, roughness):
    eta, _, c, perimeter = annular_response(alpha, diameter, mu_l)
    denominator = ml + dt * lengths * c
    drive0 = ug0 - eta * ml / denominator * ul0
    coupling = dt * lengths * perimeter * (1 / mg + eta.square() / denominator)
    lo, hi = torch.zeros_like(drive0), drive0.abs()
    for _ in range(56):
        mid = (lo + hi) / 2
        e, *_ = interfacial_conductance(
            alpha,
            diameter,
            mu_l,
            mu_g,
            rho,
            mid,
            torch.zeros_like(mid),
            dry_consistent=True,
            roughness_m=roughness,
        )
        residual = mid + coupling * e * mid - drive0.abs()
        lo, hi = torch.where(residual < 0, mid, lo), torch.where(residual < 0, hi, mid)
    drive = torch.copysign((lo + hi) / 2, drive0)
    e, *_ = interfacial_conductance(
        alpha,
        diameter,
        mu_l,
        mu_g,
        rho,
        drive,
        torch.zeros_like(drive),
        dry_consistent=True,
        roughness_m=roughness,
    )
    k = lengths * perimeter * e
    share = dt * k / (mg + dt * k)
    ul = (ml * ul0 + eta * mg * share * ug0) / (denominator + eta.square() * mg * share)
    ug = mg / (mg + dt * k) * ug0 + share * eta * ul
    return ug, ul


def run(batch_size=256, repeats=9, device="cpu"):
    if batch_size < 2 or repeats < 3:
        raise ValueError("At least two cells and three timing repeats required")

    def synchronize():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    spec = {"dtype": torch.float64, "device": device}
    alpha = torch.logspace(-4, math.log10(0.2), batch_size, **spec)
    dt = torch.logspace(-8, 6, batch_size, **spec).roll(batch_size // 3)
    ug0 = torch.linspace(-20, 20, batch_size, **spec)
    ul0 = 0.1 * torch.sin(torch.arange(batch_size, **spec))
    area = math.pi * 0.1**2 / 4
    args = (
        alpha,
        0.1,
        0.001,
        1.8e-5,
        1.2,
        ug0,
        ul0,
        1.2 * area * (1 - alpha),
        998 * area * alpha,
        torch.ones_like(alpha),
        dt,
        0.0002,
    )
    methods = {"bisection_56": bisection, "safeguarded_newton_16": implicit_annular_velocities}
    results = {name: method(*args)[:2] for name, method in methods.items()}
    error = max(float((a - b).abs().max().cpu()) for a, b in zip(*results.values()))
    if error > 1e-10:
        raise ValueError(f"Root algorithms disagree: {error} m/s")
    samples = {name: [] for name in methods}
    rng = random.Random(917)
    for _ in range(repeats):
        names = list(methods)
        rng.shuffle(names)
        for name in names:
            synchronize()
            start = time.perf_counter()
            methods[name](*args)
            synchronize()
            samples[name].append(time.perf_counter() - start)
    medians = {
        name: float(torch.tensor(values, dtype=torch.float64).median())
        for name, values in samples.items()
    }
    return {
        "scope": "synthetic fixed-inventory momentum source only",
        "device": device,
        "dtype": "float64",
        "cells": batch_size,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "physical_substep_range_s": [1e-8, 1e6],
        "timing_samples_s": samples,
        "median_wall_s": medians,
        "source_only_speedup": medians["bisection_56"] / medians["safeguarded_newton_16"],
        "maximum_final_velocity_difference_m_s": error,
        "experimental_validation": False,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/surface/annular_film.py",
                "src/surface/annular_implicit_source.py",
                "src/surface/pr_active_pipe_friction.py",
                "scripts/benchmark_annular_implicit.py",
            )
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.cells, args.repeats, args.device)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("timing_samples_s", "source_sha256")},
            indent=2,
        )
    )
