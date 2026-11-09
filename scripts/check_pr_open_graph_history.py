"""Unit-aware CPU/CUDA trajectory parity for audited open active PR graphs."""

import argparse
import json
import math
from pathlib import Path

from check_pr_surface_device_parity import audit as checkpoint_audit

TOLERANCES = {
    "volume_m3": (1e-24, 1e-10),
    "component_moles": (1e-22, 1e-9),
    "phase_component_moles": (1e-22, 1e-9),
    "total_energy_j": (1e-20, 1e-10),
    "momentum_kg_m_s": (1e-22, 1e-9),
    "pressure_pa": (1e-5, 1e-10),
    "temperature_k": (1e-8, 1e-10),
    "wall_temperature_k": (1e-8, 1e-10),
    "time_s": (1e-18, 1e-12),
    "next_step_s": (1e-18, 1e-12),
}


def audit(cpu_directory, cuda_directory, *, root):
    checkpoint_audit(cpu_directory, cuda_directory, root=root, atol=1e-17, rtol=1e-9)
    cpu_directory, cuda_directory = Path(cpu_directory), Path(cuda_directory)
    summaries = [
        json.loads((p / "summary.json").read_text()) for p in (cpu_directory, cuda_directory)
    ]
    rows = {}
    for name in summaries[1]["cases"]:
        reports = [
            json.loads((directory / summary["cases"][name]["report"]).read_text())
            for directory, summary in zip((cpu_directory, cuda_directory), summaries, strict=True)
        ]
        a, b = reports
        if a.get("format") != "pr_active_surface_run_v1" or not a["system"].get("reservoirs"):
            continue
        if a["source_hashes"] != b["source_hashes"]:
            raise ValueError("Open graph trajectory provenance differs")
        stats = {"checked_state_scalars": 0, "maximum_tolerance_fraction": 0.0, "worst_path": ""}

        def compare(x, y, path, quantity=None, *, stats=stats):
            if isinstance(x, dict):
                if not isinstance(y, dict) or x.keys() != y.keys():
                    raise ValueError("Open graph registry differs: " + path)
                for k, v in x.items():
                    compare(v, y[k], path + "/" + k, k)
            elif isinstance(x, list):
                if not isinstance(y, list) or len(x) != len(y):
                    raise ValueError("Open graph phase count differs: " + path)
                for i, (v, w) in enumerate(zip(x, y, strict=True)):
                    compare(v, w, path + "/" + str(i), quantity)
            elif isinstance(x, (int, float)) and not isinstance(x, bool):
                if (
                    not isinstance(y, (int, float))
                    or isinstance(y, bool)
                    or not math.isfinite(x)
                    or not math.isfinite(y)
                    or quantity not in TOLERANCES
                ):
                    raise ValueError("Unknown or invalid physical quantity: " + path)
                absolute, relative = TOLERANCES[quantity]
                fraction = abs(x - y) / (absolute + relative * max(abs(x), abs(y)))
                if fraction > 1:
                    raise ValueError("Open graph trajectory parity failed: " + path)
                stats["checked_state_scalars"] += 1
                if fraction > stats["maximum_tolerance_fraction"]:
                    stats.update(maximum_tolerance_fraction=fraction, worst_path=path)
            elif x != y:
                raise ValueError("Open graph metadata differs: " + path)

        sa, sb = a["result"]["steps"], b["result"]["steps"]
        if len(sa) != len(sb):
            raise ValueError("Open graph accepted histories differ")
        compare(a["initial"], b["initial"], "initial")
        for i, (x, y) in enumerate(zip(sa, sb, strict=True)):
            compare(x["state"], y["state"], "steps/" + str(i))
        compare(a["result"]["state"], b["result"]["state"], "final")
        rows[name] = {"accepted": True, **stats}
    if not rows:
        raise ValueError("No audited open graph trajectories")
    return {
        "accepted": True,
        "qualification": "All accepted states of open graphs, field-specific physical units; independent physical audits separate",
        "tolerances": TOLERANCES,
        "cases": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cpu_directory", type=Path)
    p.add_argument("cuda_directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(
        args.cpu_directory, args.cuda_directory, root=Path(__file__).resolve().parents[1]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
