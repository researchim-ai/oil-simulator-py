"""Source-bound joint capillary mechanics, optionally with PR material stocks."""

import argparse
import hashlib
import json
from pathlib import Path

from surface.inertial_capillary_film import InertialCapillaryFilm, InertialCapillaryState
from surface.pr_isothermal_capillary_film import (
    PRIsothermalCapillaryFilm,
    PRIsothermalCapillaryState,
)

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
    allowed = {
        "model",
        "initial_thickness_m",
        "initial_face_velocity_m_s",
        "initial_state",
        "duration_s",
        "initial_step_s",
        "minimum_step_s",
    }
    if (
        set(config) - allowed
        or ("initial_state" in config) == ("initial_thickness_m" in config)
        or ("initial_state" in config and "initial_face_velocity_m_s" in config)
    ):
        raise ValueError("One complete mechanical initial condition or checkpoint required")
    sha = fingerprints()
    is_pr = config["model"]["model"] == "pr_isothermal_capillary_film_v1"
    cls = PRIsothermalCapillaryFilm if is_pr else InertialCapillaryFilm
    model = cls.from_dict(config["model"], device=device)
    if "initial_state" in config:
        initial = (PRIsothermalCapillaryState if is_pr else InertialCapillaryState).from_dict(
            config["initial_state"]
        )
    else:
        initial = model.initialize(
            config["initial_thickness_m"], config.get("initial_face_velocity_m_s")
        )
    model.inspect(initial)
    state, result = model.advance(
        initial,
        config["duration_s"],
        initial_step_s=config["initial_step_s"],
        minimum_step_s=config.get("minimum_step_s", 1e-12),
    )
    if fingerprints() != sha:
        raise RuntimeError("Joint capillary sources changed during run")
    return {
        "format": "inertial_capillary_run_v1",
        "request": config,
        "normalized_model": model.to_dict(),
        "source_sha256": sha,
        "device": device,
        "family": "pr_isothermal" if is_pr else "mechanical",
        "qualification": "Periodic planar single-liquid coating; passive exterior; neutral wetting; not general gas/liquid network",
        "initial": initial.to_dict(),
        "final": state.to_dict(),
        "result": result,
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
            {
                "steps": len(r["result"]["steps"]),
                "rejected_steps": r["result"]["rejected_steps"],
                "time_s": r["final"].get("time_s", r["final"].get("mechanics", {}).get("time_s")),
            }
        )
    )


if __name__ == "__main__":
    main()
