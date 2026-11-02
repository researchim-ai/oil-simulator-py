"""Paired small CPU timings; no coupled-field or GPU performance claim."""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import ThermalHomogeneousPipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    modes = {
        "rusanov": {"numerical_flux": "rusanov"},
        "constant_entropy": {
            "thermal_entropy_dissipation": 0.75,
            "adaptive_entropy_dissipation": False,
        },
        "lagged_entropy": {},
    }
    rows = []
    for repeat in range(3):
        for name, kw in modes.items():
            p = ThermalHomogeneousPipe(10, 0.1, 64, 850, 500, 300, 1, 0, 0.02, **kw)
            s = p.initialize(1e6, 0.2)
            b = {
                "pressure_pa": 1e6,
                "gas_void_fraction": 0.2,
                "mixture_velocity_m_s": 0.0,
                "temperature_k": 300.0,
            }
            start = time.perf_counter()
            r = p.advance(s, 0.02, left={**b, "pressure_pa": 1010000}, right=b, initial_dt_s=0.01)
            wall = time.perf_counter() - start
            rows.append(
                {
                    "mode": name,
                    "repeat": repeat,
                    "wall_s": wall,
                    "steps": len(r["steps"]),
                    "failed": r["failed_steps"],
                    "evaluations": sum(x["nonlinear_evaluations"] for x in r["steps"]),
                }
            )
    d = {
        "scope": "Paired CPU 64 cells/two steps, initialization excluded, 3 alternating samples; concurrent background checks not isolated; not a field/GPU benchmark",
        "rows": rows,
        "medians_s": {
            n: statistics.median(x["wall_s"] for x in rows if x["mode"] == n) for n in modes
        },
        "sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in [
                Path(__file__).resolve().parents[1] / "src/surface/thermal_pipe.py",
                Path(__file__).resolve().parents[1] / "src/surface/thermal_flux.py",
                Path(__file__).resolve(),
            ]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(d["medians_s"])


if __name__ == "__main__":
    main()
