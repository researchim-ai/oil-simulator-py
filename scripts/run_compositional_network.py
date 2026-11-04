"""Run a coupled Torch PR vessel network, publishing only a complete result."""

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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    temporary = None
    try:
        root = Path(__file__).resolve().parents[1] / "src/surface"
        # Snapshot before importing physical modules. No stale imported solver
        # can publish a result attributed to newly overwritten source files.
        names = (
            "phase_geometry.py",
            "separator_geometry.py",
            "three_phase_properties.py",
            "compositional_network.py",
            "compositional_network_schedule.py",
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
        own = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        from surface.compositional_network import CompositionalNetwork
        from surface.compositional_network_schedule import run_compositional_network_schedule

        data = json.loads(args.config.read_text())
        if set(data) != {"system", "initial", "periods"}:
            raise ValueError("Config requires exactly system, initial, periods")
        model = CompositionalNetwork.from_dict(data["system"], device=args.device)
        initial = model.initialize(data["initial"])
        checkpoint = json.loads(args.checkpoint.read_text()) if args.checkpoint else None
        result = run_compositional_network_schedule(
            model, initial, data["periods"], checkpoint, args.stop_after
        )
        if (
            hashes != {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}
            or own != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ):
            raise RuntimeError("Sources changed during network calculation; refusing publication")
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
