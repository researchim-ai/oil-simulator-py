"""Verify the conservative material/thermal film split; no momentum closure."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from surface.liquid import ConservationError, HydraulicConvergenceError, PropertyModelError
from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
from surface.pr_phase_volume_transport import transfer_film_increment
from surface.thin_film_dynamics import ThinFilmDynamics, ThinFilmState

ROOT = Path(__file__).resolve().parents[1]


def fingerprints():
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            *sorted((ROOT / "src/surface").glob("*.py")),
            *sorted((ROOT / "scripts").glob("*.py")),
        ]
    }


def run(config, *, device="cpu"):
    if set(config) != {
        "pipe",
        "film",
        "initial",
        "initial_film",
        "duration_s",
        "maximum_step_s",
        "liquid_slot",
        "gas_slot",
    } and set(config) != {
        "pipe",
        "film",
        "initial",
        "initial_film",
        "duration_s",
        "maximum_step_s",
        "liquid_slot",
        "gas_slot",
        "minimum_step_s",
    }:
        raise ValueError("Unknown or incomplete film transfer request")
    before = fingerprints()
    pipe = PRActiveMultifluidPipe.from_dict(config["pipe"], device=device)
    film = ThinFilmDynamics.from_dict(config["film"], device=device)
    state = PRActiveMultifluidState.from_dict(config["initial"])
    fs = ThinFilmState.from_dict(config["initial_film"])
    initial = state
    initial_film = fs
    duration = float(config["duration_s"])
    maximum = float(config["maximum_step_s"])
    minimum = float(config.get("minimum_step_s", 1e-10))
    if (
        not all(math.isfinite(v) and v > 0 for v in (duration, maximum, minimum))
        or minimum > maximum
    ):
        raise ValueError("Positive ordered transfer time controls required")
    end_time = state.time_s + duration
    steps = []
    rejected = 0
    dt = maximum
    while state.time_s < end_time:
        if len(steps) + rejected >= 10000:
            raise HydraulicConvergenceError("Film transfer attempt limit")
        trial = min(dt, end_time - state.time_s)
        try:
            new, nf, r = transfer_film_increment(
                pipe,
                state,
                film,
                fs,
                trial,
                liquid_slot=config["liquid_slot"],
                gas_slot=config["gas_slot"],
            )
        except (HydraulicConvergenceError, PropertyModelError, ConservationError):
            dt = trial / 2
            rejected += 1
            if dt < minimum:
                raise
            continue
        state, fs = new, nf
        steps.append(r)
        dt = min(maximum, trial * 1.5)
    if before != fingerprints():
        raise RuntimeError("Film transfer sources changed during run")
    return {
        "request": config,
        "source_sha256": before,
        "device": device,
        "qualification": "Material/thermal split only; capillary momentum traction and network integration absent",
        "initial": initial.to_dict(),
        "initial_film": initial_film.to_dict(),
        "final": state.to_dict(),
        "final_film": fs.to_dict(),
        "steps": steps,
        "rejected_steps": rejected,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    report = run(json.loads(a.config.read_text()), device=a.device)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"steps": len(report["steps"]), "rejected_steps": report["rejected_steps"]}))


if __name__ == "__main__":
    main()
