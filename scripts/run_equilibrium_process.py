"""Run explicit assay + thermal well boundaries through a common PR process."""

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
    args = parser.parse_args()
    temporary = None
    try:
        root = Path(__file__).resolve().parents[1]
        sources = [
            "src/surface/equilibrium_properties.py",
            "src/surface/component_energy.py",
            "src/surface/equilibrium_inventory.py",
            "src/surface/equilibrium_process.py",
            "src/surface/peng_robinson.py",
            "src/surface/peng_robinson_mixture.py",
            "src/simulator/thermal_surface_boundary.py",
            "scripts/run_equilibrium_process.py",
            "src/surface/liquid.py",
            "src/surface/operation.py",
            "src/surface/torch_backend.py",
        ]
        source_hashes = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in sources}
        from simulator.thermal_surface_boundary import characterize_well_thermal_boundaries
        from surface.component_energy import StockComponentAssay
        from surface.equilibrium_process import run_equilibrium_process
        from surface.equilibrium_properties import EquilibriumFluid
        from surface.operation import _digest
        from surface.torch_backend import runtime_fingerprint

        data = json.loads(args.config.read_text())
        if set(data) != {
            "fluid",
            "stock_mass_fractions",
            "wells",
            "thermal_conditions",
            "transfers",
            "mixer",
        }:
            raise ValueError("Unexpected equilibrium process config fields")
        fluid = EquilibriumFluid.from_dict(data["fluid"], device=args.device)
        assay = StockComponentAssay(fluid.names, data["stock_mass_fractions"], fluid.model_id)
        boundary = characterize_well_thermal_boundaries(
            data["wells"], data["thermal_conditions"], fluid, assay
        )
        result = run_equilibrium_process(
            fluid, boundary.pop("streams"), data["transfers"], data["mixer"]
        )
        result = {
            "boundary": boundary,
            "process": result,
            "configuration": data,
            "runtime": runtime_fingerprint(),
            "device": args.device,
            "source_sha256": source_hashes,
        }
        if source_hashes != {
            n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in sources
        }:
            raise RuntimeError("Source files changed during the calculation; refusing publication")
        result["checksum"] = _digest(result)
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
