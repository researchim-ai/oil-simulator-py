"""Reproducible PR confined-gas / film pressure-volume trajectory."""

import argparse
import json
from pathlib import Path

from run_inertial_capillary_film import fingerprints

from surface.pr_gas_loaded_capillary import PRGasLoadedCapillaryFilm, PRGasLoadedCapillaryState


def run(config, *, device="cpu"):
    allowed = {
        "model",
        "initial_state",
        "initial_thickness_m",
        "initial_face_velocity_m_s",
        "gas_mole_fractions",
        "gas_pressures_pa",
        "duration_s",
        "initial_step_s",
        "minimum_step_s",
    }
    if set(config) - allowed or ("initial_state" in config) == ("initial_thickness_m" in config):
        raise ValueError("One gas-loaded initial condition required")
    if "initial_state" in config and set(config) & {
        "initial_face_velocity_m_s",
        "gas_mole_fractions",
        "gas_pressures_pa",
    }:
        raise ValueError("Checkpoint cannot be mixed with initial fields")
    sha = fingerprints()
    model = PRGasLoadedCapillaryFilm.from_dict(config["model"], device=device)
    initial = (
        PRGasLoadedCapillaryState.from_dict(config["initial_state"])
        if "initial_state" in config
        else model.initialize(
            config["initial_thickness_m"],
            config["gas_mole_fractions"],
            face_velocity_m_s=config.get("initial_face_velocity_m_s"),
            gas_pressures_pa=config.get("gas_pressures_pa"),
        )
    )
    end, result = model.advance(
        initial,
        config["duration_s"],
        initial_step_s=config["initial_step_s"],
        minimum_step_s=config.get("minimum_step_s", 1e-12),
    )
    if fingerprints() != sha:
        raise RuntimeError("Gas-loaded film sources changed during run")
    return {
        "format": "pr_gas_loaded_capillary_run_v1",
        "request": config,
        "normalized_model": model.to_dict(),
        "source_sha256": sha,
        "device": device,
        "family": "closed_gas_chambers",
        "initial": initial.to_dict(),
        "final": end.to_dict(),
        "result": result,
        "qualification": "Closed isothermal PR gas chambers, periodic liquid film; no axial gas flow",
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
                "steps": len(r["result"]["steps"]),
                "rejected_steps": r["result"]["rejected_steps"],
                "time_s": r["final"]["film"]["mechanics"]["time_s"],
            }
        )
    )


if __name__ == "__main__":
    main()
