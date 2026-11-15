"""Run sampled thermal controls on coupled Torch Gauss4 with audited restart."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import CollocationThermalStation, run_collocation_thermal_schedule
from surface.torch_backend import liquid_backend_config


def run(data, checkpoint=None, stop_after=None, device=None):
    if set(data) != {"system", "initial", "periods"}:
        raise ValueError("Expected exactly system, initial and periods")
    spec = data["initial"]
    if "physical" not in spec or set(spec) - {"physical", "temperatures_k", "wall_temperatures_k"}:
        raise ValueError("Expected physical initialization and optional fluid/wall temperatures")
    system = CollocationThermalStation.from_dict(
        liquid_backend_config(data["system"], None, device)
    )
    physical = system.physics.initialize(**spec["physical"])
    heat = system.heat.initialize(
        physical, spec.get("temperatures_k"), spec.get("wall_temperatures_k")
    )
    initial = system.initialize(physical, heat)
    return run_collocation_thermal_schedule(
        system, initial, data["periods"], checkpoint, stop_after
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--device", help="Explicit Torch device; never silently falls back")
    args = parser.parse_args()
    temporary = None
    try:
        result = run(
            json.loads(args.config.read_text()),
            json.loads(args.checkpoint.read_text()) if args.checkpoint else None,
            args.stop_after,
            args.device,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=args.output.parent,
            delete=False,
            prefix="." + args.output.name,
            suffix=".tmp",
        ) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
        temporary = None
    except (ValueError, TypeError, KeyError, RuntimeError, OSError) as error:
        parser.exit(2, f"{error}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
