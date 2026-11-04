"""Run a gravitational PR vessel with finite apertures and prescribed feeds."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), *sorted((root / "src/surface").glob("*.py"))]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    from surface.stratified_pr import StratifiedPRState
    from surface.stratified_pr_flow import StratifiedPRFlow
    from surface.stratified_pr_thermal import StratifiedThermalState
    from surface.torch_backend import runtime_fingerprint

    cfg = json.loads(args.config.read_text())
    if set(cfg) != {
        "column",
        "wall",
        "integration",
        "initial",
        "wall_temperature_k",
        "durations_s",
        "outlets",
        "feeds",
    }:
        raise ValueError("Unknown or missing thermal config fields")
    if not isinstance(cfg["durations_s"], list) or not cfg["durations_s"]:
        raise ValueError("Nonempty interval schedule required")
    model = StratifiedPRFlow.from_dict(
        {
            "model": "stratified_pr_flow_v1",
            "column": cfg["column"],
            "wall": cfg["wall"],
            "outlets": cfg["outlets"],
            "feeds": cfg["feeds"],
            **cfg["integration"],
        },
        device=args.device,
    )
    column = model.column
    initial = column.initialize(**cfg["initial"])
    state = model.initialize(
        StratifiedPRState.from_dict(initial["state"]), cfg["wall_temperature_k"]
    )
    intervals = []
    for duration in cfg["durations_s"]:
        report = model.advance(state, duration)
        state = StratifiedThermalState.from_dict(report["state"])
        intervals.append(report)
    result = {
        "config": cfg,
        "model_id": model.model_id,
        "initial": initial,
        "intervals": intervals,
        "runtime": runtime_fingerprint(),
        "source_sha256": hashes,
    }
    for p in sources:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(root))]:
            raise RuntimeError("Sources changed during thermal run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, args.output)
        temporary = None
    finally:
        if temporary:
            os.unlink(temporary)
    print(args.output)


if __name__ == "__main__":
    main()
