"""Solve a conserved gravitational PR column and sample local port states."""

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SOURCES = (
    "stratified_pr.py",
    "equilibrium_properties.py",
    "peng_robinson.py",
    "peng_robinson_mixture.py",
    "component_energy.py",
    "separator_geometry.py",
    "three_phase_properties.py",
    "torch_linear.py",
    "torch_backend.py",
    "liquid.py",
    "operation.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    temporary = None
    try:
        root = Path(__file__).resolve().parents[1] / "src/surface"
        hashes = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in SOURCES}
        own = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        import numpy as np

        from surface.liquid import _number
        from surface.operation import _digest
        from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState
        from surface.torch_backend import runtime_fingerprint

        cfg = json.loads(args.config.read_text())
        if set(cfg) != {"system", "initial", "samples"}:
            raise ValueError("Config requires exactly system, initial, samples")
        model = StratifiedPRColumn.from_dict(cfg["system"], device=args.device)
        initial = cfg["initial"]
        if set(initial) == {"pressure_pa", "temperature_k", "mole_fractions"}:
            command = model.initialize
        elif set(initial) == {
            "component_moles",
            "total_energy_j",
            "pressure_guess_pa",
            "temperature_guess_k",
        }:
            command = model.equilibrate
        else:
            raise ValueError("Initial state needs PT/composition or conserved N/E with guesses")
        if not isinstance(cfg["samples"], list):
            raise TypeError("samples must be a list")
        for item in cfg["samples"]:
            if not isinstance(item, dict) or set(item) - {
                "height_m",
                "interface_index",
                "side",
                "molar_rate_mol_s",
            }:
                raise ValueError("Unknown sample field")
            if ("height_m" in item) == ("interface_index" in item):
                raise ValueError("Sample needs height OR interface index")
            if "interface_index" in item and (
                isinstance(item["interface_index"], bool)
                or not isinstance(item["interface_index"], int)
                or item["interface_index"] < 0
            ):
                raise ValueError("Invalid interface index")
            if item.get("side") not in (None, "below", "above"):
                raise ValueError("Unknown sample side")
            if "interface_index" in item and item.get("side") not in ("below", "above"):
                raise ValueError("Interface sample requires an explicit side")
            if "height_m" in item:
                height = _number(item["height_m"], "sample height", nonnegative=True)
                if height > model.geometry.height_m:
                    raise ValueError("Sample outside column")
            if "molar_rate_mol_s" in item:
                _number(item["molar_rate_mol_s"], "molar rate", nonnegative=True)
        result = command(**initial)
        state = StratifiedPRState.from_dict(result["state"])
        samples = []
        for item in cfg["samples"]:
            if "interface_index" in item:
                index = item["interface_index"]
                if index >= len(state.interface_heights_m):
                    raise ValueError("Requested interface does not exist")
                height = state.interface_heights_m[index]
            else:
                height = item["height_m"]
            point = model.sample(state, height, side=item.get("side"))
            row = {"request": copy.deepcopy(item), "point": point}
            if "molar_rate_mol_s" in item:
                row["stream"] = model.stream_at(
                    state, height, item["molar_rate_mol_s"], side=item.get("side")
                ).to_dict()
            samples.append(row)
        payload = {
            "format": "stratified_pr_column_v1",
            "system": model.to_dict(),
            "initial": initial,
            "result": result,
            "samples": samples,
            "provenance": {
                "source_hashes": hashes,
                "runner_sha256": own,
                "runtime": runtime_fingerprint(),
                "device": str(model.fluid.device),
                "dtype": "float64",
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
        }
        payload = {**payload, "checksum": _digest(payload)}
        if (
            hashes != {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in SOURCES}
            or own != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ):
            raise RuntimeError(
                "Source files changed during column calculation; refusing publication"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=args.output.parent,
            prefix="." + args.output.name,
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
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
