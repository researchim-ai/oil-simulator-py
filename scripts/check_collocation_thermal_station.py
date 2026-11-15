"""End-to-end control clock, conservative ODE and audited restart fixture."""

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from check_liquid_thermal_collocation import ConservativeReference
from run_collocation_thermal import run

from surface import CollocationThermalStation
from surface.torch_liquid import TorchLiquidEquations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = ROOT / "configs/surface_collocation_thermal_control.json"
    config = json.loads(path.read_text())
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        path,
        Path(__file__).resolve(),
        ROOT / "scripts/check_liquid_thermal_collocation.py",
        ROOT / "scripts/run_collocation_thermal.py",
    ]
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    start = time.perf_counter()
    full = run(config)
    elapsed = time.perf_counter() - start
    partial = run(config, stop_after=1)
    resumed = run(config, json.loads(json.dumps(partial)))
    assert resumed == full
    system = CollocationThermalStation.from_dict(config["system"])
    p = system.physics.initialize(**config["initial"]["physical"])
    h = system.heat.initialize(
        p, config["initial"]["temperatures_k"], config["initial"]["wall_temperatures_k"]
    )
    ref = ConservativeReference(system.heat)
    reference = ref.encode(p, h)
    e = TorchLiquidEquations(system.physics)
    names = [
        "pressure:h",
        "pressure:t",
        "pipe_flow:line",
        "speed:a",
        "speed:b",
        *system.heat.names,
        *[n + ":wall" for n in system.heat.wall_names],
        *["mass:" + n for n in ref.reservoirs],
    ]
    budget = np.r_[
        [1.0, 1.0, 1e-8, 1e-4, 1e-4],
        [1e-5] * (len(system.heat.names) + len(system.heat.wall_names)),
        [1e-7] * len(ref.reservoirs),
    ]
    errors = np.zeros(len(names))
    previous = 0.0
    samples = []
    events = []
    states = []
    combined = []
    last_segment = None
    for period in full["records"]:
        samples.extend(period["samples"])
        events.extend(period["events"])
        for segment in period["segments"]:
            start = segment["hydraulics"]["initial"]["time_s"]
            stop = segment["state"]["hydraulics"]["time_s"]
            assert start == previous
            if last_segment:
                assert last_segment["state"]["hydraulics"] == segment["hydraulics"]["initial"]
                assert last_segment["state"]["heat"] == segment["initial_heat"]
            command = (5.0, 7.0) if start < 0.1 - 1e-14 else (0.0, 0.0)
            assert segment["hydraulics"]["controls"]["motor_torques_nm"] == dict(
                zip(("a", "b"), command)
            )
            assert segment["thermal_controls"]["heat_duties_w"] == {"line::0": 100.0}
            rhs = ConservativeReference(
                system.heat, motor_torques_nm=command, heat_duties_w={"line::0": 100.0}
            )
            expected = solve_ivp(
                rhs,
                (start, stop),
                reference,
                method="DOP853",
                rtol=1e-12,
                atol=1e-9,
                dense_output=True,
            )
            assert expected.success
            for hy, th in zip(segment["hydraulics"]["steps"], segment["thermal_steps"]):
                assert hy["time_s"] == th["time_s"] and hy["end_time_s"] == th["end_time_s"]
                z = np.array(hy["scaled_state"])
                actual = np.r_[
                    z[[0, 1, 2, 6, 7]] * np.r_[[e.ps] * 2, e.qs, [100, 110]],
                    [th["temperatures_k"][n] for n in system.heat.names],
                    [th["wall_temperatures_k"][n] for n in system.heat.wall_names],
                    [th["masses_kg"][n] for n in ref.reservoirs],
                ]
                error = abs(actual - rhs.observe(expected.sol(hy["end_time_s"])))
                assert np.all(error < budget), dict(zip(names, error))
                errors = np.maximum(errors, error)
                states.append({"time_s": hy["end_time_s"], "observables": actual.tolist()})
                combined.append(abs(th["combined_energy_residual_j"]))
            reference = expected.y[:, -1]
            previous = stop
            last_segment = segment
    assert [s["tick"] for s in samples] == list(range(6))
    assert np.allclose([s["time_s"] for s in samples], np.arange(6) * 0.05, rtol=0.0, atol=1e-14)
    assert len(events) == 1 and events[0]["type"] == "temperature_trip"
    assert abs(events[0]["time_s"] - 0.1) < 1e-14
    assert full["state"]["next_tick"] == 6
    assert abs(previous - math.fsum(p["duration_s"] for p in config["periods"])) < 1e-14
    for sample in samples:
        assert sample["controllers"]["heater"]["requested_duty_w"] == 1000.0
        assert sample["controllers"]["heater"]["applied_duty_w"] == 100.0
    assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == v for n, v in sources.items())
    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output / "run.json.gz", "wt") as stream:
        json.dump(
            {
                "config": config,
                "full": full,
                "partial": partial,
                "observables": names,
                "trajectory": states,
            },
            stream,
            allow_nan=False,
        )
    summary = {
        "scope": "Sampled thermostats/trips on constant-property liquid hydraulics, caloric stocks and wall; not full PR",
        "simulated_duration_s": previous,
        "diagnostic_elapsed_s": elapsed,
        "timing_is_benchmark_comparison": False,
        "samples": len(samples),
        "events": events,
        "accepted_steps": len(states),
        "maximum_observable_errors": dict(zip(names, errors.tolist())),
        "budgets": dict(zip(names, budget.tolist())),
        "max_combined_energy_residual_j": max(combined),
        "resume_bitwise_identical": True,
        "state_continuity_at_command_changes_checked": True,
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "passed": True,
        "source_sha256": sources,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
