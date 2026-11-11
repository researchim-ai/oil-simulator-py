"""Source-bound caloric capillary phase-transition experiment, with audits."""

import argparse
import json
import time
from pathlib import Path

import torch
from check_pr_capillary_equilibrium import audit
from run_pr_active_surface import fingerprints

from surface.operation import _digest
from surface.pr_active_multifluid_pipe import PRActiveMultifluidState
from surface.pr_capillary_equilibrium import PRCapillaryEquilibrium


def run(config, *, device="cpu"):
    if set(config) != {"model", "initial", "heat_pulses"} or not config["heat_pulses"]:
        raise ValueError("Explicit model, initial stocks and nonempty heat pulses required")
    hashes = fingerprints()
    model = PRCapillaryEquilibrium.from_dict(config["model"], device=device)
    initial = current = PRActiveMultifluidState.from_dict(config["initial"])
    model.pipe.inspect(initial)
    records = []
    for pulse in config["heat_pulses"]:
        if set(pulse) != {"heat_j", "heat_temperature_k"}:
            raise ValueError("Heat pulse needs energy and reservoir temperature")
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        end, report = model.equilibrate(current, **pulse)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        check = audit(model, current, end, report)
        records.append({"result": report, "audit": check, "solver_seconds": elapsed})
        current = end
    if fingerprints() != hashes or _digest(initial.to_dict()) != _digest(config["initial"]):
        raise RuntimeError("Capillary experiment sources or initial state changed")
    return {
        "format": "pr_capillary_equilibrium_run_v2",
        "request": config,
        "device": device,
        "source_sha256": hashes,
        "normalized_model": model.to_dict(),
        "records": records,
        "final": current.to_dict(),
        "qualification": "Instantaneous local UV phase-equilibrium heat pulses, no report clock advancement; not a calibrated nucleation kinetic model",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a completed experiment")
    result = run(json.loads(args.config.read_text()), device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "accepted_pulses": len(result["records"]),
                "phase_history": [
                    [len(c) for c in record["result"]["state"]["cells"]]
                    for record in result["records"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
