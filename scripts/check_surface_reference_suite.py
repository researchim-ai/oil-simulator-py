"""Run independent surface reference gates with explicit local-accuracy status.

Frozen CoolProp fixtures avoid adding a runtime dependency. This is numerical
verification in declared model domains, not industrial certification.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-local-temperature", action="store_true")
    parser.add_argument("--include-shock-matrix", action="store_true")
    parser.add_argument("--include-compressor", action="store_true")
    parser.add_argument("--shock-workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("pr", "check_pr_reference.py", []),
        ("isothermal_acoustics", "check_pipe_acoustics.py", []),
        ("thermal_acoustics", "check_pipe_acoustics.py", ["--thermal"]),
        ("thermal_shock", "check_pipe_shock.py", []),
    ]
    if args.include_shock_matrix:
        jobs.append(
            (
                "thermal_shock_matrix",
                "check_pipe_shock_matrix.py",
                ["--matrix", "all", "--workers", str(args.shock_workers)],
            )
        )
    if args.include_compressor:
        jobs.append(("compressor_train", "check_compressor_train.py", []))
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    records = []
    for name, script, extra in jobs:
        output = args.output_dir / (name + ".json")
        output.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(root / "scripts" / script),
            *extra,
            "--output",
            str(output.resolve()),
        ]
        with (args.output_dir / (name + ".log")).open("w") as log:
            result = subprocess.run(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        artifact_valid = False
        artifact_error = None
        try:
            payload = json.loads(output.read_text())
            if name == "pr":
                artifact_valid = (
                    payload["passed"] and payload["section"] == "all" and payload["checks"] > 0
                )
            elif name == "compressor_train":
                artifact_valid = (
                    payload["passed"]
                    and len(payload["blowdown"]) == 3
                    and len(payload["equilibria"]) == 2
                    and all(row["passed"] for row in payload["equilibria"])
                )
            elif name == "thermal_shock":
                artifact_valid = payload["passed"] and len(payload["cases"]) >= 2
            elif name == "thermal_shock_matrix":
                artifact_valid = (
                    payload["passed"]
                    and payload["matrix"] == "all"
                    and len(payload["cases"]) == 12
                    and all(
                        row["local_passed"]
                        and row["pressure_global_passed"]
                        and row["temperature_global_passed"]
                        for row in payload["cases"]
                    )
                )
            else:
                artifact_valid = len(payload["cases"]) >= 2
        except (OSError, ValueError, KeyError, TypeError) as error:
            artifact_error = str(error)
        records.append(
            {
                "name": name,
                "returncode": result.returncode,
                "output": str(output),
                "command": command,
                "artifact_valid": bool(artifact_valid),
                "artifact_error": artifact_error,
            }
        )
    succeeded = all(r["returncode"] == 0 and r["artifact_valid"] for r in records)
    shock_path = args.output_dir / "thermal_shock.json"
    local = False
    shock_record = next(r for r in records if r["name"] == "thermal_shock")
    if shock_path.exists() and shock_record["returncode"] == 0:
        local = bool(json.loads(shock_path.read_text())["stagnation_temperature_gate_passed"])
    passed = succeeded and (not args.require_local_temperature or local)
    files = [
        root / "tests/data/surface/pr_caloric_coolprop.json",
        root / "src/surface/peng_robinson.py",
        root / "src/surface/peng_robinson_mixture.py",
        root / "src/surface/single_phase_nozzle.py",
        root / "src/surface/thermal_pipe.py",
        root / "src/surface/thermal_flux.py",
        root / "src/surface/transient_pipe.py",
        Path(__file__),
        *(
            root / "scripts" / name
            for name in [
                "pr_reference_support.py",
                "check_pr_reference.py",
                "check_pipe_acoustics.py",
                "check_pipe_shock.py",
                "check_pipe_shock_matrix.py",
            ]
        ),
    ]
    if args.include_compressor:
        files.extend(
            root / "src/surface" / name
            for name in (
                "compressor_map.py",
                "compressor_train.py",
                "compressor_schedule.py",
                "gas.py",
                "gas_valve.py",
                "actuator.py",
                "pid.py",
                "rotating_drive.py",
            )
        )
        files.extend(
            [
                root / "scripts/check_compressor_train.py",
                root / "configs/surface_compressor_train.json",
            ]
        )
    report = {
        "scope": "Independent PR, homogeneous-pipe and selected optional dry-gas compressor reference gates",
        "passed": passed,
        "standard_gates_passed": succeeded,
        "strict_local_temperature_passed": succeeded and local,
        "local_temperature_required": args.require_local_temperature,
        "shock_matrix_included": args.include_shock_matrix,
        "compressor_included": args.include_compressor,
        "status": "failed"
        if not passed
        else ("passed" if local else "passed_with_local_temperature_limit"),
        "known_limits": []
        if local
        else [
            "Local stagnation temperature error does not satisfy 0.02 K criterion at the tested resolution; global L1 convergence is a separate property."
        ],
        "not_validated": [
            "Experimental calibration",
            "Full thermal slip closure",
            "Water/three phases",
            "Thermal reservoir-network coupling",
            "Field-scale/GPU performance",
            "Universal local temperature accuracy outside the explicitly tested shock states and resolutions",
        ],
        "jobs": records,
        "sha256": {
            str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(report["status"])
    if not passed:
        parser.exit(1, "Selected surface reference gates failed; see summary.json\n")


if __name__ == "__main__":
    main()
