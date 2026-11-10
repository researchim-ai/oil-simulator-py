"""Replay and independently audit a shared capillary source on actual graph stocks."""

import argparse
import json
from pathlib import Path

from check_pr_capillary_chain import audit
from run_pr_active_surface import fingerprints

from surface.pr_active_surface import PRActiveSurfaceNetwork, PRActiveSurfaceState


def run(config, *, device="cpu"):
    if set(config) != {"network", "initial", "pipe_order", "parameters", "dt_s", "port_openings"}:
        raise ValueError("Explicit graph, inventories, source parameters and openings required")
    hashes = fingerprints()
    model = PRActiveSurfaceNetwork.from_dict(config["network"], device=device)
    old = PRActiveSurfaceState.from_dict(config["initial"])
    end, record = model.relax_capillary_chain(
        old,
        config["dt_s"],
        config["pipe_order"],
        config["parameters"],
        port_openings=config["port_openings"],
    )
    verified = audit(model, old, end, record, config["parameters"])
    if hashes != fingerprints():
        raise RuntimeError("Source changed during capillary graph transaction")
    return {
        "format": "pr_capillary_chain_run_v1",
        "request": config,
        "device": device,
        "source_sha256": hashes,
        "initial": old.to_dict(),
        "final": end.to_dict(),
        "result": record,
        "independent_audit": verified,
        "qualification": "Shared normal/chemical source only; axial capillary transport and phase topology changes are not qualified",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()
    result = run(json.loads(args.config.read_text()), device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "accepted": result["independent_audit"]["accepted"],
                "shared_edges": result["independent_audit"]["shared_edges"],
                "segment_energy_residuals_j": result["result"]["segment_energy_residuals_j"],
            }
        )
    )


if __name__ == "__main__":
    main()
