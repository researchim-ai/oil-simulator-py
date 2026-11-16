"""Like-for-like FIM benchmark of scalar versus batched PVT admission only.

Both modes run the current torch_batch pipe solver. 'legacy' restores just the
previous per-stream admission in this process, never a different pipe model.
The ordinary forecast driver supplies accounting and source guards; this
wrapper additionally records its own source, selected policy and unchanged hash.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import run_structured_well_forecast as driver
import torch

from surface import blackoil_pipe_batch
from surface.liquid import PropertyModelError


def legacy_validate_inputs(model, boundaries, rates, count):
    masses = torch.stack([model.validate(p, m) for p, m in zip(boundaries, rates, strict=True)])
    boundary = model.rho.new_tensor(boundaries)
    if masses.shape != (count, 3):
        raise ValueError("One boundary and composition per pipe required")
    if (
        len(model.oil_rs) < 2 or not float(model.oil_rs[0]) <= 0 <= float(model.oil_rs[-1])
    ) and bool(((masses[:, 0] == 0) & (masses[:, 2] == 0)).any()):
        raise PropertyModelError(
            "Pure-water composition derivatives require covering zero-GOR oil PVTO"
        )
    return boundary, masses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", choices=["legacy", "batch"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--days", type=float, default=1)
    parser.add_argument("--report-days", type=float, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (
        config["simulation"]["physical_surface_network"]["network"].get("hydraulic_backend")
        != "torch_batch"
    ):
        parser.error("Both policies require the same torch_batch hydraulic backend")
    source = Path(__file__).resolve()
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    original = blackoil_pipe_batch.validate_inputs
    argv = sys.argv
    policy = legacy_validate_inputs if args.admission == "legacy" else original
    calls = 0

    def admitted(*a, **kw):
        nonlocal calls
        calls += 1
        return policy(*a, **kw)

    blackoil_pipe_batch.validate_inputs = admitted
    try:
        sys.argv = [
            str(Path(driver.__file__)),
            "--config",
            str(args.config),
            "--days",
            str(args.days),
            "--report-days",
            str(args.report_days),
            "--output",
            str(args.output),
        ]
        driver.main()
    finally:
        sys.argv = argv
        blackoil_pipe_batch.validate_inputs = original
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise RuntimeError("Admission benchmark wrapper changed during run")
    if not calls:
        raise RuntimeError("Selected admission policy did not execute")
    metadata = {
        "scope": __doc__,
        "admission_policy": args.admission,
        "wrapper_source": str(source),
        "wrapper_sha256": before,
        "wrapper_unchanged": True,
        "admission_calls": calls,
    }
    with (args.output / "admission_policy.json").open("x") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
