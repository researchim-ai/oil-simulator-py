"""Release gate for independent PR fixtures; no CoolProp runtime dependency."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

from pr_reference_support import cases, validate_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/data/surface/pr_caloric_coolprop.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--section", choices=("all", "homogeneous", "vle", "inverse"), default="all"
    )
    args = parser.parse_args()
    data = json.loads(args.reference.read_text())
    generator = Path(__file__).with_name("generate_pr_caloric_reference.py")
    if (
        data["schema"] != 1
        or data["backend"] != "PR"
        or data["generator_version"] != "7.2.0"
        or data["generator_sha256"] != hashlib.sha256(generator.read_bytes()).hexdigest()
    ):
        parser.error("Reference schema/version/generator mismatch; regenerate explicitly")
    rows = [
        validate_one(data, *item)
        for item in cases(data)
        if args.section == "all" or item[0] == args.section
    ]
    root = Path(__file__).resolve().parents[1]
    result = {
        "scope": "Independent EOS implementation verification under explicit constant-Cp conventions, not experimental calibration or all-fluid validation",
        "reference": str(args.reference),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                root / "src/surface/peng_robinson.py",
                root / "src/surface/peng_robinson_mixture.py",
                root / "src/surface/single_phase_nozzle.py",
                Path(__file__),
                Path(__file__).with_name("pr_reference_support.py"),
            ]
        },
        "checks": len(rows),
        "section": args.section,
        "passed": all(r["passed"] for r in rows),
        "failures": sum(not r["passed"] for r in rows),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        parser.exit(1, f"{result['failures']} independent PR checks failed; see {args.output}\n")
    print(f"{len(rows)} independent PR checks passed")


if __name__ == "__main__":
    main()
