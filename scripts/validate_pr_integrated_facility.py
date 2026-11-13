"""Bounded whole-facility acceptance: connected topology, operation and independent balances.

A timeout, unsupported equipment state or incomplete operating cycle is a failed
qualification, never a successful benchmark. The diagnostic trajectory is retained.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def coverage(config):
    """Check actual graph connectivity including the common separator and machines."""
    plant = config["model"]["plant"]
    transport = plant["transport"]
    vertices = {n["name"] for n in transport["nodes"]} | set(plant["columns"])
    edges = [(x["source"], x["target"]) for x in transport["links"]]
    edges += [(x["column"], x["node"]) for x in plant["ports"]]
    edges += [(x["card"]["start"], x["card"]["end"]) for x in transport["pump_connections"]]
    edges += [(x["source"], x["target"]) for x in transport["compressors"]]
    reached = {next(iter(vertices))}
    while True:
        extended = (
            reached | {b for a, b in edges if a in reached} | {a for a, b in edges if b in reached}
        )
        if extended == reached:
            break
        reached = extended
    if reached != vertices:
        raise ValueError(f"Disconnected facility: {sorted(vertices - reached)}")
    machines = {x["card"]["name"] for x in transport["pump_connections"]} | {
        x["name"] for x in transport["compressors"]
    }
    if (
        len(transport["pump_connections"]) < 2
        or not transport["compressors"]
        or not plant["columns"]
    ):
        raise ValueError("Acceptance needs liquid pumps, gas compression and a shared separator")
    if {x["machine"] for x in config["model"]["motors"]} != machines:
        raise ValueError("All machines require represented motor controls")
    return {
        "connected_vertices": len(vertices),
        "connections": len(edges),
        "pipe_cells": sum(n["kind"] == "pipe_cell" for n in transport["nodes"]),
        "separator_ports": len(plant["ports"]),
        "machines": sorted(machines),
    }


def requested_operation(config):
    """Reject incomplete qualification schedules before any expensive solve."""
    if len(config["schedule"]) != 3:
        raise ValueError("Expected startup, isolation and restart intervals")
    times = [float(c["duration_s"]) for c in config["schedule"]]
    if not all(math.isfinite(t) and t > 0 for t in times) or math.fsum(times) < 1:
        raise ValueError("Qualification requires at least one simulated second across all stages")
    return times


def operational_audit(config, report):
    """Require nonzero transport, isolation and restarted delivery, not just a zero-flow balance."""
    names = coverage(config)["machines"]
    times = requested_operation(config)
    delivered, ports = [], []
    for interval in report["intervals"]:
        totals = dict.fromkeys(names, 0.0)
        aperture = {}
        for control in interval["steps"]:
            for physical in control["plant"]["steps"]:
                t = physical["transport"]
                machines = {**t["pump_connections"], **t["compressors"]}
                dt = physical["dt_s"]
                for name in names:
                    totals[name] += dt * abs(machines[name]["mass_rate_kg_s"])
                for name, row in physical["columns"]["ports"].items():
                    aperture[name] = aperture.get(name, 0.0) + dt * math.fsum(
                        abs(x) for x in row["flux"][:-2]
                    )
        delivered.append(totals)
        ports.append(aperture)
    if len(delivered) != 3:
        raise ValueError("Incomplete facility operating cycle")
    for name in names:
        if delivered[0][name] <= 0 or delivered[2][name] <= 0:
            raise ValueError(f"No startup/restarted flow through {name}")
        if delivered[1][name] != 0:
            raise ValueError(f"Isolated machine {name} transferred mass")
    expected_ports = {p["name"] for p in config["model"]["plant"]["ports"]}
    for stage in (0, 2):
        if set(ports[stage]) != expected_ports or any(v <= 0 for v in ports[stage].values()):
            raise ValueError("Every feed/takeoff must participate in startup and restart")
    if any(v != 0 for v in ports[1].values()):
        raise ValueError("Isolated separator transferred material")
    start = report["initial"]["plant"]["transport"]["time_s"]
    end = report["intervals"][-1]["state"]["plant"]["transport"]["time_s"]
    if abs(end - start - math.fsum(times)) > 8 * math.ulp(end):
        raise ValueError("Facility did not reach the requested report clock")
    return {
        "simulated_s": end - start,
        "machine_throughput_kg": delivered,
        "separator_aperture_throughput_mol": ports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/surface_pr_integrated_facility.json"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("Finite positive wall-time budget required")
    if args.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Explicit CUDA_VISIBLE_DEVICES required")
    config = json.loads(args.config.read_text())
    topology = coverage(config)
    requested_operation(config)
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        Path(__file__),
        args.config.resolve(),
        ROOT / "scripts/run_pr_controlled_surface.py",
        *sorted((ROOT / "scripts").glob("check_pr_*.py")),
    ]

    def fingerprints():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    before = fingerprints()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    trajectory = output.with_suffix(".trajectory.json")
    log = output.with_suffix(".log")
    if any(p.exists() for p in (output, trajectory, log)):
        raise FileExistsError("Use a fresh result path; completed evidence is immutable")
    result = {
        "format": "pr_integrated_facility_acceptance_v1",
        "accepted": False,
        "device": args.device,
        "topology": topology,
        "source_sha256": before,
        "trajectory": trajectory.name,
        "log": log.name,
        "qualification": "Synthetic finite-inventory facility within declared equipment envelopes; no empirical calibration or reservoir/injection-well qualification",
    }
    start = time.perf_counter()
    try:
        with log.open("x") as handle:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_pr_controlled_surface.py"),
                    str(args.config),
                    "--device",
                    args.device,
                    "--output",
                    str(trajectory),
                    "--progress",
                ],
                cwd=ROOT,
                stdout=handle,
                stderr=handle,
                timeout=args.timeout_s,
                check=False,
            )
        if proc.returncode:
            raise RuntimeError(
                f"Facility solver exited with code {proc.returncode}; inspect {log.name}"
            )
        # The independent checker uses scalar/NumPy thermodynamics, not runtime stepping.
        checked = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_pr_controlled_surface.py"),
                str(args.config),
                str(trajectory),
            ],
            cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
            timeout=max(0.001, args.timeout_s - (time.perf_counter() - start)),
            check=False,
        )
        if checked.returncode:
            raise RuntimeError("Independent physical audit failed: " + checked.stderr[-2000:])
        report = json.loads(trajectory.read_text())
        result["physical_audit"] = json.loads(checked.stdout)
        if result["physical_audit"].get("accepted") is not True:
            raise ValueError("Independent physical audit rejected the trajectory")
        result["operation"] = operational_audit(config, report)
        result["accepted"] = True
    except (
        subprocess.TimeoutExpired,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_s"] = time.perf_counter() - start
    result["sources_unchanged"] = fingerprints() == before
    result["accepted"] &= result["sources_unchanged"]
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
