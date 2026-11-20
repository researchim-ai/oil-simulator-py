"""Paired first-trial comparison, retaining every stability start and tolerance.

Reference uses row-wise vmap/forward AD and uncompacted BFGS histories.
The first trial can be rejected physically; comparing algorithms does not
qualify an operating cycle. No full integration is launched by this probe.
"""

import argparse
import gzip
import hashlib
import json
import signal
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from run_pr_controlled_surface import run

import surface.chemical_kernel as chemical
import surface.equilibrium_minimization_batch as minimization
from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_phase_batch import PhaseEOSGroups
from surface.pr_surface_control import PRControlledSurface

ROOT = Path(__file__).resolve().parents[1]


class Captured(BaseException):
    pass


def trial(config, reference):
    saved = {}
    original = PRControlledSurface._temporal_trial
    minimize = minimization._minimize

    def observed(self, *args, **kwargs):
        start = time.perf_counter()
        coarse, fine, *_ = original(self, *args, **kwargs)
        saved.update(wall_s=time.perf_counter() - start, model=self, coarse=coarse, fine=fine)
        saved["temporal_error"] = self._error(coarse, fine)
        raise Captured()

    def legacy_batch(fluid, p, t, x):
        return torch.vmap(
            lambda pp, tt, xx: fluid.tensor_homogeneous(pp, tt, xx, _chemical_only=True)
        )(p, t, x)

    with ExitStack() as stack:
        stack.enter_context(patch.object(PRControlledSurface, "_temporal_trial", observed))
        if reference:
            stack.enter_context(patch.object(EquilibriumFluid, "_lean_allocation", False))
            stack.enter_context(patch.object(EquilibriumFluid, "_analytic_chemical", False))
            stack.enter_context(patch.object(EquilibriumFluid, "_warm_equilibrium", False))
            stack.enter_context(
                patch.object(EquilibriumFluid, "tensor_chemical_batch", legacy_batch)
            )
            stack.enter_context(patch.object(chemical, "jacobian", chemical._jacobian))
            stack.enter_context(patch.object(PhaseEOSGroups, "_native_cpu", False))
            stack.enter_context(
                patch.object(minimization, "_minimize", lambda *a: minimize(*a, _compact=False))
            )
        try:
            run(config)
        except Captured:
            return saved
    raise ValueError("No first temporal trial captured")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiled-cpu-chemical", action="store_true")
    parser.add_argument("--budget-s", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.budget_s <= 600:
        parser.error("Explicit wall budget in 1..600 s required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "configs/surface_pr_integrated_facility.json").read_text())
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    def expired(*_):
        raise TimeoutError("Paired first-trial budget exhausted")

    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.budget_s)
    try:
        reference = trial(config, True)
        with patch.object(EquilibriumFluid, "_compiled_chemical_cpu", args.compiled_cpu_chemical):
            actual = trial(config, False)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    differences = {
        key: actual["model"]._error(actual[key], reference[key]) for key in ("coarse", "fine")
    }
    for key in ("coarse", "fine"):
        if actual[key].time_s != reference[key].time_s:
            raise ValueError("Algorithm comparison clocks differ")
    result = {
        "compiled_cpu_chemical": args.compiled_cpu_chemical,
        "reference_wall_s": reference["wall_s"],
        "native_wall_s": actual["wall_s"],
        "observed_trial_speedup": reference["wall_s"] / actual["wall_s"],
        "temporal_error": {
            "reference": reference["temporal_error"],
            "native": actual["temporal_error"],
        },
        "coarse_fine_difference_in_original_error_norm": differences,
        "algorithms_agree": max(differences.values()) <= 1e-5,
        "full_operating_cycle_qualified": False,
        "source_sha256": hashes,
        "timing_scope": "One paired CPU first-trial observation, excludes initialization; shared host, no statistical speedup claim",
    }
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))]
        for p in paths
    ):
        raise RuntimeError("Source changed during paired trial")
    with gzip.open(args.output_dir / "states.json.gz", "wt") as handle:
        json.dump(
            {
                "config": config,
                "reference": {k: reference[k].to_dict() for k in differences},
                "native": {k: actual[k].to_dict() for k in differences},
            },
            handle,
            allow_nan=False,
        )
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))
    return 0 if result["algorithms_agree"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
