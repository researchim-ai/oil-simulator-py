"""Run controlled pressure/network/shaft periods with audited JSON restart (PyTorch CPU/CUDA)."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import ControlledLiquidStation, run_controlled_liquid_schedule
from surface.torch_backend import liquid_backend_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--device", help="Explicit Torch device, e.g. cpu or cuda:0")
    parser.add_argument("--backend", choices=("torch", "scipy_reference"))
    parser.add_argument(
        "--hydraulic-state-transfer",
        choices=("interval", "solve"),
        help="Mechanical state transfer strategy; default interval",
    )
    parser.add_argument("--hydraulic-journal", choices=("full", "compact"))
    parser.add_argument("--hydraulic-journal-branches", nargs="*")
    args = parser.parse_args()
    temporary = None
    try:
        data = json.loads(args.config.read_text())
        if set(data) != {"system", "initial", "periods"}:
            raise ValueError("Controlled liquid config requires exactly system, initial, periods")
        data["system"] = liquid_backend_config(data["system"], args.backend, args.device)
        if args.hydraulic_state_transfer is not None:
            data["periods"] = [
                {**p, "hydraulic_state_transfer": args.hydraulic_state_transfer}
                for p in data["periods"]
            ]
        if args.hydraulic_journal is not None:
            data["periods"] = [
                {**p, "hydraulic_journal": args.hydraulic_journal} for p in data["periods"]
            ]
        if args.hydraulic_journal_branches is not None:
            data["periods"] = [
                {**p, "hydraulic_journal_branches": args.hydraulic_journal_branches}
                for p in data["periods"]
            ]
        drive = ControlledLiquidStation.from_dict(data["system"])
        initial = drive.initialize(**data["initial"])
        result = run_controlled_liquid_schedule(
            drive,
            initial,
            data["periods"],
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
