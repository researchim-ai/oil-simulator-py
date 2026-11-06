"""Compare audited CPU/CUDA final checkpoints from the functional matrix.

This checks numerical parity, not physical accuracy or GPU throughput.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(cpu_directory, cuda_directory, *, root, atol=1e-9, rtol=1e-8):
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol <= 0 or rtol <= 0:
        raise ValueError("Positive finite parity tolerances required")
    directories = [Path(cpu_directory).resolve(), Path(cuda_directory).resolve()]
    root = Path(root).resolve()
    summaries = [json.loads((p / "summary.json").read_text()) for p in directories]
    cpu, gpu = summaries
    if cpu["device"] != "cpu" or gpu["device"] != "cuda":
        raise ValueError("Expected CPU and CUDA summaries, in that order")
    for summary in summaries:
        for path, digest in summary["source_sha256"].items():
            source = (root / path).resolve()
            source.relative_to(root)
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Changed source/config: {path}")
    for path, digest in gpu["source_sha256"].items():
        if cpu["source_sha256"].get(path) != digest:
            raise ValueError(f"Different source/config: {path}")
    if not gpu["cases"] or set(gpu["cases"]) - set(cpu["cases"]):
        raise ValueError("CUDA cases must be a nonempty subset of CPU cases")
    rows = {}
    for name in gpu["cases"]:
        reports = []
        for summary, directory in zip(summaries, directories):
            record = summary["cases"][name]
            if record["audit"].get("accepted") is not True:
                raise ValueError(f"Physical audit did not accept {name}")
            path = (directory / record["report"]).resolve()
            path.relative_to(directory)
            reports.append(json.loads(path.read_text()))
        worst = [0.0, ""]
        count = 0

        def compare(x, y, path, worst=worst, absolute=atol):
            nonlocal count
            if isinstance(x, dict):
                if not isinstance(y, dict) or x.keys() != y.keys():
                    raise ValueError(f"Checkpoint registry differs: {path}")
                for key in x:
                    # Device provenance is checked above; opaque model IDs are
                    # not physical observables. All state coordinates are checked.
                    if key != "model_id":
                        compare(x[key], y[key], f"{path}/{key}")
            elif isinstance(x, list):
                if not isinstance(y, list) or len(x) != len(y):
                    raise ValueError(f"Checkpoint shape differs: {path}")
                for i, (a, b) in enumerate(zip(x, y)):
                    compare(
                        a,
                        b,
                        f"{path}/{i}",
                        absolute=1e-20
                        if path.endswith("/droplet_sections") and i == len(x) - 1
                        else atol,
                    )
            elif isinstance(x, (float, int)) and not isinstance(x, bool):
                if isinstance(y, bool) or not isinstance(y, (float, int)):
                    raise TypeError(f"Checkpoint type differs: {path}")
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError(f"Nonfinite checkpoint: {path}")
                scaled = abs(x - y) / (absolute + rtol * max(abs(x), abs(y)))
                count += 1
                if scaled > worst[0]:
                    worst[:] = [scaled, path]
                if scaled > 1:
                    raise ValueError(f"Device parity failed: {path}: {x} vs {y}")
            elif x != y:
                raise ValueError(f"Checkpoint value differs: {path}")

        a, b = reports
        if a["config"] != b["config"] or a["model"] != b["model"]:
            raise ValueError(f"Requested/normalized models differ: {name}")
        if len(a["intervals"]) != len(b["intervals"]) or not a["intervals"]:
            raise ValueError(f"Report intervals differ: {name}")
        for i, (x, y) in enumerate(zip(a["intervals"], b["intervals"])):
            compare(x["state"], y["state"], f"intervals/{i}/state")
        rows[name] = {
            "accepted": True,
            "checked_state_scalars": count,
            "maximum_tolerance_fraction": worst[0],
            "worst_path": worst[1],
        }
    return {
        "accepted": True,
        "qualification": "Checkpoint device parity; independent physical audits are separate",
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_directory", type=Path)
    parser.add_argument("cuda_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.cpu_directory, args.cuda_directory, root=Path(__file__).resolve().parents[1]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
