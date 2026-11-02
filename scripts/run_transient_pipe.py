"""Run explicit isothermal drift-flux pipe periods with checked JSON restart."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.pipe_schedule import run_pipe_schedule
from surface.thermal_pipe import ThermalHomogeneousPipe
from surface.transient_pipe import DriftFluxPipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    temporary = None
    try:
        data = json.loads(args.config.read_text())
        if not {"pipe", "initial", "periods"} <= set(data) or set(data) - {
            "pipe",
            "initial",
            "periods",
            "stepper",
            "model",
        }:
            raise ValueError("Invalid pipe configuration fields")
        models = {
            "isothermal_drift_flux": DriftFluxPipe,
            "thermal_homogeneous": ThermalHomogeneousPipe,
        }
        kind = data.get("model", "isothermal_drift_flux")
        if kind not in models:
            raise ValueError("Unknown distributed pipe model")
        pipe = models[kind](**data["pipe"])
        initial = pipe.initialize(**data["initial"])
        result = run_pipe_schedule(
            pipe,
            initial,
            data["periods"],
            data.get("stepper"),
            json.loads(args.checkpoint.read_text()) if args.checkpoint else None,
            args.stop_after,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=args.output.parent,
            delete=False,
            prefix="." + args.output.name,
            suffix=".tmp",
        ) as output:
            temporary = Path(output.name)
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, args.output)
        temporary = None
    except (ValueError, TypeError, KeyError, RuntimeError, OSError) as error:
        parser.exit(2, f"{error}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
