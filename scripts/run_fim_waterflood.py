"""Black-oil FIM plus production/facility/treated-water/injection tree study."""

import argparse
import contextlib
import copy
import gzip
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_journal import BenchmarkJournal, atomic_json
from check_fim_waterflood import audit

from simulator.fluid import Fluid
from simulator.reservoir import Reservoir
from simulator.simulation import Simulator
from simulator.well import WellManager


def run(config, days, report_days, events=(), device="cpu", progress=None, save_fields=False):
    if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in (days, report_days)):
        raise ValueError("Positive finite horizon/report interval required")
    events = copy.deepcopy(list(events))
    previous = -1
    for event in events:
        if (
            set(event) != {"time_days", "controls"}
            or isinstance(event["time_days"], bool)
            or not math.isfinite(event["time_days"])
            or not previous < event["time_days"] < days
            or event["time_days"] < 0
        ):
            raise ValueError("Events must be ordered in [0,horizon)")
        previous = event["time_days"]
    config = copy.deepcopy(config)
    records = []
    fields = []
    log = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(log):
        dev = torch.device(device)
        r = Reservoir(config["reservoir"], device=dev)
        f = Fluid(config["fluid"], r, device=dev)
        sim = Simulator(r, f, WellManager(config["wells"], r), config["simulation"], device=dev)
        driver = sim.facility_coupling
        controls = copy.deepcopy(driver.controls)
        try:
            for event in events:
                driver.set_controls(**event["controls"])
        finally:
            driver.controls = controls
        initial = driver._inventory()

        def accepted(time_days):
            stage = {
                "time_days": time_days,
                "newton_iterations": sim.fim_solver.last_newton_iters,
                "coupling": copy.deepcopy(driver.report),
                "surface": copy.deepcopy(sim.fim_solver.wells.surface_report),
            }
            records.append(stage)
            if progress is not None:
                progress(stage, time.perf_counter() - started)

        sim.substep_hook = accepted
        index = 0
        while sim.current_time < days * 86400:
            while index < len(events) and events[index]["time_days"] * 86400 <= sim.current_time:
                driver.set_controls(**events[index]["controls"])
                index += 1
            target = min(days * 86400, sim.current_time + report_days * 86400)
            if index < len(events):
                target = min(target, events[index]["time_days"] * 86400)
            if not sim.run_step(target - sim.current_time):
                raise RuntimeError(
                    f"Waterflood failed at {sim.current_time / 86400} days: {driver.last_attempt}"
                )
            if save_fields:
                fields.append(
                    {
                        "time_days": sim.current_time / 86400,
                        "pressure_pa": f.pressure.flatten().detach().cpu().tolist(),
                        "s_w": f.s_w.flatten().detach().cpu().tolist(),
                        "s_g": f.s_g.flatten().detach().cpu().tolist(),
                    }
                )
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        final = driver._inventory()
    return {
        "requested_days": days,
        "report_days": report_days,
        "schedule": events,
        "solve_wall_s": time.perf_counter() - started,
        "initial_component_mass_kg": initial,
        "final_component_mass_kg": final,
        "accepted_substeps": records,
        "reservoir_snapshots": fields,
        "final_pressure_pa": f.pressure.flatten().detach().cpu().tolist(),
        "final_s_w": f.s_w.flatten().detach().cpu().tolist(),
        "final_s_g": f.s_g.flatten().detach().cpu().tolist(),
        "execution": {
            "reservoir_device": str(dev),
            "injection_tangents": "Torch float64",
            "injection_equipment_properties": "host scalar laws",
            "production_facility": "existing host model",
            "torch_threads": torch.get_num_threads(),
        },
        "solver_log": log.getvalue(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/fim_blackoil_waterflood.json"
    )
    parser.add_argument("--days", type=float, default=2)
    parser.add_argument("--report-days", type=float, default=0.5)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--save-fields", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Explicit CUDA_VISIBLE_DEVICES required")
    config = json.loads(args.config.read_text())
    config["fluid"]["pvt_path"] = str(ROOT / config["fluid"]["pvt_path"])
    events = [] if args.schedule is None else json.loads(args.schedule.read_text())
    paths = (
        sorted((ROOT / "src").rglob("*.py"))
        + sorted((ROOT / "scripts").glob("*.py"))
        + [args.config.resolve(), Path(config["fluid"]["pvt_path"])]
        + ([] if args.schedule is None else [args.schedule.resolve()])
    )
    hashes = lambda: {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    before = hashes()
    journal = BenchmarkJournal(
        args.output,
        {
            "scope": __doc__,
            "days": args.days,
            "report_days": args.report_days,
            "schedule": events,
            "source_sha256": before,
            "device": args.device,
        },
    )
    journal.checkpoint([], [], {})

    def progress(stage, wall):
        journal.checkpoint([stage], [], {"wall_s": wall})
        print(
            json.dumps(
                {
                    "time_days": stage["time_days"],
                    "wall_s": wall,
                    "water": stage["coupling"]["water_rates_kg_s"],
                }
            ),
            flush=True,
            file=sys.__stdout__,
        )

    try:
        result = run(
            config, args.days, args.report_days, events, args.device, progress, args.save_fields
        )
        with gzip.open(args.output / "trajectory.json.gz", "wt") as stream:
            json.dump({"config": config, "result": result}, stream, allow_nan=False)
        certificate = audit(config, result)
        if before != hashes():
            raise RuntimeError("Waterflood study source changed during calculation")
        summary = {
            "status": "complete",
            "audit": certificate,
            "solve_wall_s": result["solve_wall_s"],
            "sources_unchanged": True,
            "source_sha256": before,
            "trajectory_sha256": hashlib.sha256(
                (args.output / "trajectory.json.gz").read_bytes()
            ).hexdigest(),
            "execution": result["execution"],
            "scope": __doc__,
        }
        journal.complete(summary)
        atomic_json(
            args.output / "progress.json",
            {
                "status": "complete",
                "time_days": args.days,
                "accepted_substeps": len(result["accepted_substeps"]),
            },
        )
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        atomic_json(
            args.output / "failure.json", {"exception": type(error).__name__, "reason": str(error)}
        )
        atomic_json(args.output / "progress.json", {"status": "failed", "reason": str(error)})
        raise


if __name__ == "__main__":
    main()
