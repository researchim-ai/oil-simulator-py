"""Live independent liquid-network comparison; not multiphase plant validation.

Reuses the existing exporter and archived cases, preserving their budgets.
Additional operating states are specified before running either solver.
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cases():
    for name, flow, head in (
        ("ring", 1e-5, .02),
        ("booster", 2e-4, .002),
        ("mapped_pump_valve", 1e-10, .01),
        ("parallel_pumps", 3e-5, .001),
    ):
        fixture = json.loads((ROOT / f"tests/data/surface/epanet_22_{name}.json").read_text())
        yield name, fixture["input"], flow, head
        if name == "ring":
            reverse = copy.deepcopy(fixture["input"])
            # Reverse the orientation of every branch: identical physical network.
            for pipe in reverse["pipes"]:
                pipe["start"], pipe["end"] = pipe["end"], pipe["start"]
            yield "ring_reversed", reverse, flow, head
            isolated = copy.deepcopy(fixture["input"])
            isolated["pipes"][3]["open"] = False
            yield "ring_isolated_bypass", isolated, flow, head
        if name == "mapped_pump_valve":
            throttled = copy.deepcopy(fixture["input"])
            throttled["valves"][0]["opening"] = .6
            yield "mapped_throttled", throttled, flow, head
            restored = copy.deepcopy(fixture["input"])
            yield "mapped_restored", restored, flow, head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ref = args.reference_root.resolve()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = json.loads((ref / "source_manifest.json").read_text())
    if digest(ref / "source.tar.gz") != manifest["sha256"]:
        raise ValueError("Reference archive checksum mismatch")
    paths = [*sorted((ROOT / "src/surface").glob("*.py")),
             ROOT / "scripts/check_surface_epanet.py", Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    rows = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for name, config, flow, head in cases():
        config_path = out / f"{name}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        command = [sys.executable, str(ROOT / "scripts/check_surface_epanet.py"), str(ref),
                   "--config", str(config_path), "--output-dir", str(out),
                   "--match-si-local-loss",
                   "--max-flow-error-kg-s", str(flow), "--max-head-error-m", str(head)]
        with (out / f"{name}.log").open("w") as log:
            run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        artifact = out / f"{name}_comparison.json"
        result = json.loads(artifact.read_text()) if artifact.exists() else {}
        rows.append({"case": name, "returncode": run.returncode,
                     "accepted": run.returncode == 0 and result.get("accepted") is True,
                     "comparison": artifact.name,
                     "flow_error_kg_s": result.get("max_flow_difference_kg_s"),
                     "head_error_m": result.get("max_head_difference_m"),
                     "power_error_scaled": result.get("max_power_scaled_difference"),
                     "status_mismatches": result.get("status_mismatches")})
    unchanged = all(digest(ROOT / p) == v for p, v in hashes.items())
    result = {"scope": "Steady single-liquid networks; operating snapshots, not transient integration",
              "whole_multiphase_facility_validated": False,
              "reference": manifest,
              "reference_library_sha256": digest(ref / "build/lib/libepanet2.so"),
              "source_sha256": hashes, "source_unchanged": unchanged,
              "cases": rows, "accepted": unchanged and all(r["accepted"] for r in rows)}
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
