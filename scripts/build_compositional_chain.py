"""Build a reproducible finite PR chain for sparse-path qualification."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.compositional_network import CompositionalConnection, CompositionalNetwork
from surface.compositional_separator import CompositionalSeparator
from surface.equilibrium_properties import EquilibriumFluid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vessels", type=int, default=36)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.vessels <= 1000:
        parser.error("vessels must be in [2,1000]; generation is not a scaling qualification")
    root = Path(__file__).resolve().parents[1]
    example = json.loads((root / "configs/surface_compositional_network.json").read_text())
    fluid = EquilibriumFluid.from_dict(example["system"]["vessels"]["a"]["fluid"])
    names = [f"v{i:02d}" for i in range(args.vessels)]
    network = CompositionalNetwork(
        {n: CompositionalSeparator(fluid, 0.001) for n in names},
        [
            CompositionalConnection(f"e{i:02d}", names[i], names[i + 1], 1e-10)
            for i in range(args.vessels - 1)
        ],
        maximum_dt_s=0.01,
        temporal_error_control=False,
    )
    initial = {
        n: {
            "pressure_pa": 1e6 - 8750 * i / (args.vessels - 1),
            "temperature_k": 320,
            "mole_fractions": [0.5, 0.5],
        }
        for i, n in enumerate(names)
    }
    data = {"system": network.to_dict(), "initial": initial, "periods": [{"duration_s": 0.01}]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
