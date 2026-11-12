"""Source-bound actual FIM/finite PR receiver run with independent scalar audits."""

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from check_dynamic_receiver import audit

from simulator.fluid import Fluid
from simulator.reservoir import Reservoir
from simulator.simulation import Simulator
from simulator.well import WellManager
from surface.torch_backend import runtime_fingerprint


def fingerprints():
    root = Path(__file__).resolve().parents[1]
    paths = [*sorted((root / "src").rglob("*.py")), *sorted((root / "scripts").glob("*.py"))]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(config, device="cpu"):
    if set(config) != {"simulator", "intervals", "qualification"} or not config["intervals"]:
        raise ValueError("Simulator, nonempty schedule and declared physical scope required")
    raw = deepcopy(config["simulator"])
    if raw["simulation"].get("dynamic_receiver") is None:
        raise ValueError("An actual FIM/dynamic receiver configuration is required")
    spec = raw["simulation"]["dynamic_receiver"]
    initial = spec["initial_state"]
    clock = (
        initial["transport"]["time_s"]
        if spec["model"] == "dynamic_pr_gravity_transport_fim_v1"
        else initial["time_s"]
    )
    if clock != 0.0:
        raise ValueError("Fresh runs start at zero; restore a full simulator checkpoint to resume")
    hashes = fingerprints()
    dev = torch.device(device)
    reservoir = Reservoir(raw["reservoir"], device=dev)
    fluid = Fluid(raw["fluid"], reservoir, device=dev)
    sim = Simulator(
        reservoir, fluid, WellManager(raw["wells"], reservoir), raw["simulation"], device=dev
    )
    records, timings, step_statistics = [], [], []
    sim.substep_hook = lambda _time: records.append(deepcopy(sim.facility_coupling.report))
    for interval in config["intervals"]:
        if set(interval) - {"connection_openings"} != {"duration_s", "outlet_openings"}:
            raise ValueError("Explicit duration and outlet commands required")
        sim.facility_coupling.outlet_openings = dict(interval["outlet_openings"])
        sim.facility_coupling.connection_openings = dict(interval.get("connection_openings", {}))
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        if not sim.run_step(interval["duration_s"]):
            raise RuntimeError(f"Coupled interval failed: {sim.facility_coupling.last_attempt}")
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        step_statistics.append(deepcopy(sim.last_fim_interval_stats))
    if fingerprints() != hashes:
        raise RuntimeError("Coupled run sources changed")
    result = {
        "format": "dynamic_pr_receiver_run_v1",
        "config": config,
        "device": str(dev),
        "source_sha256": hashes,
        "runtime": runtime_fingerprint(),
        "interval_seconds": timings,
        "time_step_statistics": step_statistics,
        "records": records,
        "qualification": config["qualification"],
    }
    result["independent_audit"] = audit(config, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a completed run")
    result = run(json.loads(args.config.read_text()), args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "accepted_intervals": len(result["records"]),
                "independent_audit_accepted": result["independent_audit"]["accepted"],
                "interval_seconds": result["interval_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
