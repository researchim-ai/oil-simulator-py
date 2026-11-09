"""Source-bound periodic long-wave wall-film experiment (no PR-network bridge)."""

import argparse
import hashlib
import json
from pathlib import Path

from surface.thin_film_dynamics import ThinFilmDynamics

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
    if set(config) - {
        "model",
        "initial_thickness_m",
        "duration_s",
        "initial_step_s",
        "minimum_step_s",
        "tangential_gravity_m_s2",
        "interface_shear_pa",
    }:
        raise ValueError("Unknown film run fields")
    before = fingerprints()
    model = ThinFilmDynamics.from_dict(config["model"], device=device)
    initial = model.initialize(config["initial_thickness_m"])
    state, result = model.advance(
        initial,
        config["duration_s"],
        initial_step_s=config["initial_step_s"],
        minimum_step_s=config.get("minimum_step_s", 1e-12),
        **{k: config[k] for k in ("tangential_gravity_m_s2", "interface_shear_pa") if k in config},
    )
    if fingerprints() != before:
        raise RuntimeError("Film sources changed during run")
    return {
        "format": "long_wave_film_run_v1",
        "request": config,
        "source_sha256": before,
        "device": device,
        "initial": initial.to_dict(),
        "final": state.to_dict(),
        "result": result,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    report = run(json.loads(args.config.read_text()), device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "steps": len(report["result"]["steps"]),
                "rejected_steps": report["result"]["rejected_steps"],
            }
        )
    )


if __name__ == "__main__":
    main()
