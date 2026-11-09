"""Source-bound normal/chemical capillary transaction on active PR stocks."""

import argparse
import json
from pathlib import Path

from run_pr_active_surface import fingerprints

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
from surface.pr_capillary_pressure import PRCapillaryPressure


def run(config, *, device="cpu"):
    if set(config) != {"pipe", "source", "initial", "dt_s"}:
        raise ValueError("Explicit pipe/source/stocks and normal interval required")
    sha = fingerprints()
    pipe = PRActiveMultifluidPipe.from_dict(config["pipe"], device=device)
    source = PRCapillaryPressure.from_dict(pipe, config["source"])
    initial = PRActiveMultifluidState.from_dict(config["initial"])
    end, record = source.apply(initial, config["dt_s"])
    if sha != fingerprints():
        raise RuntimeError("Capillary source changed during execution")
    return {
        "format": "pr_capillary_pressure_run_v1",
        "family": "capillary_normal_source",
        "request": config,
        "normalized_model": {"pipe": pipe.to_dict(), "source": source.to_dict()},
        "source_sha256": sha,
        "device": device,
        "initial": initial.to_dict(),
        "final": end.to_dict(),
        "result": {"steps": [record]},
        "qualification": "Normal/chemical source only; no axial surface transport qualification",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    r = run(json.loads(args.config.read_text()), device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "newton_iterations": r["result"]["steps"][0]["newton_iterations"],
                "energy_residual_j": r["result"]["steps"][0]["energy_residual_j"],
            }
        )
    )


if __name__ == "__main__":
    main()
