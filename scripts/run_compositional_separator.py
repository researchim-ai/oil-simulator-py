"""Run monolithic PR separator dynamics on Torch CPU or CUDA."""

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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    temporary = None
    try:
        root = Path(__file__).resolve().parents[1] / "src/surface"
        # Snapshot before importing physical modules; verify again before commit.
        names = (
            "phase_geometry.py",
            "separator_geometry.py",
            "three_phase_properties.py",
            "compositional_separator.py",
            "compositional_separator_schedule.py",
            "equilibrium_properties.py",
            "equilibrium_inventory.py",
            "component_energy.py",
            "peng_robinson.py",
            "peng_robinson_mixture.py",
            "torch_linear.py",
            "vessel_wall.py",
            "liquid.py",
            "operation.py",
            "torch_backend.py",
        )
        hashes = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}
        own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        from surface.compositional_separator import CompositionalSeparator
        from surface.compositional_separator_schedule import run_compositional_separator_schedule

        data = json.loads(args.config.read_text())
        if set(data) != {"system", "initial", "periods"}:
            raise ValueError("Config requires exactly system, initial, periods")
        model = CompositionalSeparator.from_dict(data["system"], device=args.device)
        initial = model.initialize(**data["initial"])
        checkpoint = json.loads(args.checkpoint.read_text()) if args.checkpoint else None
        result = run_compositional_separator_schedule(
            model, initial, data["periods"], checkpoint, args.stop_after
        )
        if (
            hashes != {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}
            or own_hash != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ):
            raise RuntimeError("Sources changed during separator calculation; refusing publication")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=args.output.parent,
            prefix="." + args.output.name,
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
        temporary = None
    except (ValueError, TypeError, KeyError, RuntimeError, OSError) as error:
        parser.exit(2, f"{error}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
