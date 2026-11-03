"""Extract a real period boundary and qualify full GPU restart by physics replay."""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from surface.operation import _digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/surface_thermal_liquid.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefix-periods", type=int, default=3)
    parser.add_argument("--thermal-execution", choices=("auto", "eager", "graph"))
    parser.add_argument("--hydraulic-journal", choices=("full", "compact"))
    parser.add_argument("--hydraulic-journal-branches", nargs="*")
    args = parser.parse_args()
    data = json.loads(args.full.read_text())
    candidate = copy.deepcopy(data)
    checksum = candidate.pop("checksum")
    if checksum != _digest(candidate):
        parser.error("Invalid full-run checksum")
    if not 0 < args.prefix_periods < data["next_period"]:
        parser.error("Prefix must stop inside the calculated schedule")
    prefix = {k: copy.deepcopy(v) for k, v in data.items() if k != "checksum"}
    prefix["records"] = prefix["records"][: args.prefix_periods]
    prefix["next_period"] = args.prefix_periods
    prefix["state"] = copy.deepcopy(prefix["records"][-1]["state"])
    prefix["checksum"] = _digest(prefix)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "prefix_extracted.json"
    if args.full.resolve() in {checkpoint.resolve(), (args.output_dir / "resumed.json").resolve()}:
        parser.error("Full input must differ from generated checkpoint/resume paths")
    checkpoint.write_text(json.dumps(prefix, indent=2, allow_nan=False) + "\n")
    resumed = args.output_dir / "resumed.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_thermal_liquid.py"),
        str(args.config),
        "--backend",
        "torch",
        "--device",
        args.device,
        "--hydraulic-heat-transfer",
        "device",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(resumed),
    ]
    if args.thermal_execution is not None:
        command.extend(["--thermal-execution", args.thermal_execution])
    if args.hydraulic_journal is not None:
        command.extend(["--hydraulic-journal", args.hydraulic_journal])
    if args.hydraulic_journal_branches is not None:
        command.extend(["--hydraulic-journal-branches", *args.hydraulic_journal_branches])
    subprocess.run(command, check=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = {
        "byte_identical": args.full.read_bytes() == resumed.read_bytes(),
        "full_sha256": sha(args.full),
        "resumed_sha256": sha(resumed),
        "prefix_periods": args.prefix_periods,
        "prefix_origin": "extracted from calculated schedule at a period boundary; resume replays the entire prefix",
    }
    (args.output_dir / "restart_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["byte_identical"]:
        raise SystemExit("Restart mismatch")


if __name__ == "__main__":
    main()
