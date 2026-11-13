"""Reproduce one fixed-volume gas/film source step from complete physical stocks.

This is a local operator verification runner. dt_s is a source duration; the
film transport clock is unchanged and no whole-pipe transient is implied.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from run_inertial_capillary_film import fingerprints

from surface.pr_capillary_gas_exchange import PRCapillaryGasExchange
from surface.pr_interphase_relaxation import PRRelaxationPhase
from surface.pr_isothermal_capillary_film import PRIsothermalCapillaryState


def run(config, *, device="cpu"):
    if set(config) != {"source", "initial_film", "initial_gas", "dt_s", "conductance_kg_s"}:
        raise ValueError("Complete explicit gas/film source request required")
    hashes = fingerprints()
    source = PRCapillaryGasExchange.from_dict(config["source"], device=device)
    before = PRIsothermalCapillaryState.from_dict(config["initial_film"])
    gas = tuple(PRRelaxationPhase.from_dict(p) for p in config["initial_gas"])
    after, new_gas, report = source.apply(before, gas, config["dt_s"], config["conductance_kg_s"])
    if fingerprints() != hashes:
        raise RuntimeError("Gas/film source inputs changed during run")
    return {
        "format": "pr_capillary_gas_exchange_run_v1",
        "request": config,
        "device": device,
        "source_sha256": hashes,
        "final_film": after.to_dict(),
        "final_gas": [asdict(p) for p in new_gas],
        "source_report": report,
        "qualification": "Local fixed-volume source, supplied conductance, no clock advancement or external experimental qualification",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    result = run(json.loads(args.config.read_text()), device=args.device)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "interface_model": result["source_report"]["interface_model"],
                "energy_residual_j": result["source_report"]["energy_residual_j"],
                "momentum_residual_kg_m_s": result["source_report"]["momentum_residual_kg_m_s"],
                "clock_advanced": False,
            }
        )
    )
