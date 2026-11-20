"""Time the complete configured coupled schedule with equivalent root solvers.

The reference temporarily replaces only the paired annular root algorithm in
this single-process benchmark. Models, source law, boundary commands and all
other numerical settings remain identical. This is a short synthetic network,
not a reservoir or long-horizon facility performance estimate.
"""

import argparse
import copy
import json
import math
import random
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from benchmark_annular_implicit import bisection
from run_pr_active_surface import fingerprints
from run_pr_active_surface import run as run_network

import surface.annular_implicit_source as kernel
from surface.annular_film import interfacial_conductance
from surface.pr_active_annular_friction import IMPLICIT_MODEL, PRActiveAnnularFriction
from surface.pr_active_surface import PRActiveSurfaceNetwork
from surface.stratified_pr import StratifiedPRState

ROOT = Path(__file__).resolve().parents[1]


def parallel_config(config, branches, device):
    """Parallel pipes interact through one shared inventory and finite ports.

    Scale vertical-vessel cross section and inventories by the same factor;
    reconstruct column equilibrium and initial states through physical APIs.
    """
    if type(branches) is not int or not 1 <= branches <= 16:
        raise ValueError("Between one and sixteen parallel branches required")
    result = copy.deepcopy(config)
    if branches == 1:
        return result
    system = result["system"]
    if (len(system["pipes"]), len(system["columns"]), len(system["ports"])) != (
        1,
        1,
        1,
    ) or system.get("equipment") is not None:
        raise ValueError("Parallel benchmark expects one prototype pipe, column and port")
    old_name = next(iter(system["pipes"]))
    prototype = system["pipes"][old_name]
    port = system["ports"][0]
    if prototype["boundary_nodes"] != [port["name"], None]:
        raise ValueError("Expected one finite inlet and a closed far end")
    column_name = next(iter(system["columns"]))
    geometry = system["columns"][column_name]["geometry"]
    if geometry["orientation"] != "vertical":
        raise ValueError("Cross-section scaling requires a vertical vessel")
    geometry["diameter_m"] *= math.sqrt(branches)
    system["pipes"] = {}
    system["ports"] = []
    port_names = []
    for i in range(branches):
        name = f"riser_{i:02d}"
        port_name = f"pickup_{i:02d}"
        port_names.append(port_name)
        system["ports"].append({**port, "name": port_name})
        system["pipes"][name] = copy.deepcopy(prototype)
        system["pipes"][name]["boundary_nodes"] = [port_name, None]
    for period in result["schedule"]:
        positions = period.get("commands", {}).get("port_positions", {})
        if positions:
            period["commands"]["port_positions"] = {
                name: positions[port["name"]] for name in port_names
            }
    model = PRActiveSurfaceNetwork.from_dict(system, device=device)
    old = config["initial"]["columns"][column_name]
    pressure = config["initial"]["pipes"][old_name]["cells"][0]["gas"]["pressure_pa"]
    equilibrium = model.columns[column_name].equilibrate(
        [branches * n for n in old["component_moles"]],
        branches * old["total_energy_j"],
        pressure_guess_pa=pressure,
        temperature_guess_k=old["temperature_k"],
    )
    column_state = StratifiedPRState.from_dict(equilibrium["state"])
    pipe_states = {
        name: e["pipe"].initialize(config["initial"]["pipes"][old_name]["cells"])
        for name, e in model.pipes.items()
    }
    result["initial"] = model.initialize({column_name: column_state}, pipe_states).to_dict()
    return result


def reference_root(alpha, diameter, mu_l, mu_g, rho, ug0, ul0, mg, ml, lengths, dt, roughness):
    ug, ul = bisection(alpha, diameter, mu_l, mu_g, rho, ug0, ul0, mg, ml, lengths, dt, roughness)
    e, eta, beta, c, perimeter, re = interfacial_conductance(
        alpha,
        diameter,
        mu_l,
        mu_g,
        rho,
        ug,
        ul,
        dry_consistent=True,
        roughness_m=roughness,
    )
    return ug, ul, e, eta, beta, c, perimeter, re


@contextmanager
def measured_algorithm(algorithm):
    original_apply = PRActiveAnnularFriction.apply
    target = kernel.implicit_annular_velocities if algorithm == "newton" else reference_root
    times = {"friction_calls": 0, "friction_inclusive_host_s": 0.0}

    def timed_apply(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return original_apply(self, *args, **kwargs)
        finally:
            times["friction_calls"] += 1
            times["friction_inclusive_host_s"] += time.perf_counter() - start

    with (
        patch.object(kernel, "implicit_annular_velocities", target),
        patch.object(PRActiveAnnularFriction, "apply", timed_apply),
    ):
        yield times


def compare_states(a, b):
    """Compare physical fields only; reject topology/clock differences."""
    maxima = {
        "pressure_pa": 0.0,
        "temperature_k": 0.0,
        "momentum_kg_m_s": 0.0,
        "total_energy_j": 0.0,
        "volume_m3": 0.0,
        "component_moles": 0.0,
    }

    def walk(x, y, key=None):
        if isinstance(x, dict):
            if not isinstance(y, dict) or x.keys() != y.keys():
                raise ValueError("Network result fields differ")
            for name in x:
                walk(x[name], y[name], name)
        elif isinstance(x, (list, tuple)):
            if not isinstance(y, (list, tuple)) or len(x) != len(y):
                raise ValueError("Network result registry differs")
            for u, v in zip(x, y):
                walk(u, v, key)
        elif key in maxima:
            maxima[key] = max(maxima[key], abs(x - y))
            if abs(x - y) > 1e-9 + 1e-10 * max(abs(x), abs(y)):
                raise ValueError(f"Root algorithms changed final {key}")
        elif x != y:
            raise ValueError(f"Network results differ in {key}")

    walk(a, b)
    return maxima


def run(config_path, output, *, repeats=3, device="cpu", branches=1):
    if repeats < 3:
        raise ValueError("At least three measured runs per algorithm required")
    config = parallel_config(json.loads(Path(config_path).read_text()), branches, device)
    if any(
        e["pipe"].get("wall_friction", {}).get("model") != IMPLICIT_MODEL
        for e in config["system"]["pipes"].values()
    ):
        raise ValueError("Benchmark requires the fully implicit annular source")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(json.dumps(config, indent=2) + "\n")
    source_hashes = fingerprints()

    def synchronize():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    # Warm both paths including EOS and quadrature before measuring.
    for algorithm in ("bisection", "newton"):
        with measured_algorithm(algorithm):
            run_network(config, device=device)
    samples = {a: [] for a in ("bisection", "newton")}
    final = {}
    rng = random.Random(3197)
    for _ in range(repeats):
        order = list(samples)
        rng.shuffle(order)
        for algorithm in order:
            with measured_algorithm(algorithm) as timing:
                synchronize()
                start = time.perf_counter()
                report = run_network(config, device=device)
                synchronize()
                elapsed = time.perf_counter() - start
            samples[algorithm].append({"complete_run_wall_s": elapsed, **timing})
            final[algorithm] = report
    error = compare_states(
        final["bisection"]["result"]["state"], final["newton"]["result"]["state"]
    )
    if fingerprints() != source_hashes:
        raise RuntimeError("Source changed during paired network benchmark")
    medians = {
        a: statistics.median(r["complete_run_wall_s"] for r in rows) for a, rows in samples.items()
    }
    result = {
        "scope": "entire configured short coupled schedule; no reservoir",
        "device": device,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "pipes": len(config["system"]["pipes"]),
        "columns": len(config["system"]["columns"]),
        "cells": sum(len(e["pipe"]["lengths_m"]) for e in config["system"]["pipes"].values()),
        "simulated_duration_s": final["newton"]["result"]["state"]["time_s"]
        - config["initial"]["time_s"],
        "accepted_steps": {a: len(r["result"]["steps"]) for a, r in final.items()},
        "attempts": {a: r["result"]["attempts"] for a, r in final.items()},
        "samples": samples,
        "median_complete_run_wall_s": medians,
        "complete_run_speedup": medians["bisection"] / medians["newton"],
        "maximum_final_state_differences": error,
        "conservation_residuals": {
            a: r["result"]["conservation_residuals"] for a, r in final.items()
        },
        "experimental_validation": False,
        "source_hashes": source_hashes,
        "reference_override": "paired source root only, in-process context restored after each run",
    }
    for a, report in final.items():
        (output / (a + "_network.json")).write_text(json.dumps(report, allow_nan=False) + "\n")
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/surface_pr_active_surface_annular_implicit.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--branches", type=int, default=1)
    args = parser.parse_args()
    result = run(
        args.config,
        args.output_dir,
        repeats=args.repeats,
        device=args.device,
        branches=args.branches,
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("samples", "source_hashes")}, indent=2
        )
    )
