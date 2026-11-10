"""Source-only PR morphology schedule; no spatial network/onset prediction."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from surface.liquid import _number
from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
from surface.pr_film_mist_exchange import PRFilmMistExchange

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
    if set(config) != {"pipe", "exchange", "initial", "schedule"} or not config["schedule"]:
        raise ValueError("Complete explicit morphology schedule required")
    sha = fingerprints()
    pipe = PRActiveMultifluidPipe.from_dict(config["pipe"], device=device)
    law = PRFilmMistExchange.from_dict(config["exchange"])
    initial = state = PRActiveMultifluidState.from_dict(config["initial"])
    pipe.inspect(initial)
    intervals = []
    for period in config["schedule"]:
        if set(period) != {"duration_s", "maximum_step_s", "entrainment_m3_s", "deposition_m3_s"}:
            raise ValueError("Unknown morphology schedule command")
        duration = _number(period["duration_s"], "source duration", positive=True)
        maximum = _number(period["maximum_step_s"], "source maximum step", positive=True)
        target = state.time_s + duration
        steps = []
        while state.time_s < target:
            if len(steps) >= 10000:
                raise ValueError("Morphology source step limit")
            dt = min(maximum, target - state.time_s)
            source, r = law.apply(
                pipe,
                state,
                dt,
                entrainment_m3_s=period["entrainment_m3_s"],
                deposition_m3_s=period["deposition_m3_s"],
            )
            state = replace(source, time_s=state.time_s + dt)
            steps.append({"source": r, "state": state.to_dict()})
        intervals.append({"steps": steps, "state": state.to_dict()})
    if fingerprints() != sha:
        raise RuntimeError("Morphology sources changed during run")
    return {
        "request": config,
        "source_sha256": sha,
        "device": device,
        "qualification": "Local fixed-diameter exchange; prescribed rates and explicit planar wall energy; not a spatial network or empirical onset model",
        "initial": initial.to_dict(),
        "intervals": intervals,
        "final": state.to_dict(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    r = run(json.loads(a.config.read_text()), device=a.device)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"steps": sum(len(s["steps"]) for s in r["intervals"]), "time_s": r["final"]["time_s"]}
        )
    )


if __name__ == "__main__":
    main()
