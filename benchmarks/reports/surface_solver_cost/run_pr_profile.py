"""CPU cost attribution on the original PR facility; never disables audits."""

import argparse
from collections import defaultdict
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch
from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork as Network
from surface.torch_nonlinear import TorchNewton
from surface.torch_linear import ColoredJacobian
from surface import torch_nonlinear
from run_pr_controlled_surface import run
from check_pr_controlled_surface import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-characteristics", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.split_characteristics:
        # Reproduce only the former implementation without editing sources.
        source = textwrap.dedent(inspect.getsource(Network._characteristics))
        old = "a, b = torch.func.jacfwd(quantities)(y0)"
        assert source.count(old) == 1
        source = source.replace(old, "a = torch.func.jacfwd(lambda y: quantities(y)[0])(y0)\n    b = torch.func.jacfwd(lambda y: quantities(y)[1])(y0)")
        namespace = {}
        exec(source, Network._characteristics.__globals__, namespace)
        Network._characteristics = namespace["_characteristics"]
    costs = defaultdict(lambda: {"calls": 0, "inclusive_s": 0., "exclusive_s": 0.})
    stack = []
    patches = []

    def instrument(owner, name, label):
        original = getattr(owner, name)
        @wraps(original)
        def measured(*a, **kw):
            frame = [time.perf_counter(), 0.]
            stack.append(frame)
            try:
                return original(*a, **kw)
            finally:
                elapsed = time.perf_counter() - frame[0]
                stack.pop()
                costs[label]["calls"] += 1
                costs[label]["inclusive_s"] += elapsed
                costs[label]["exclusive_s"] += elapsed - frame[1]
                if stack:
                    stack[-1][1] += elapsed
        setattr(owner, name, measured)
        patches.append((owner, name, original))

    instrument(TorchNewton, "solve", "newton_other")
    instrument(ColoredJacobian, "values", "colored_jacobian")
    instrument(EquilibriumFluid, "_stability", "tpd_scalar")
    instrument(EquilibriumFluid, "_stability_many", "tpd_batch_other")
    instrument(Network, "_characteristics", "wave_characteristics")
    instrument(torch_nonlinear, "gmres_device", "gmres")
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((ROOT / "src/surface").glob("*.py"))}
    config = json.loads((ROOT / "configs/surface_pr_integrated_facility_liquid_mixture.json").read_text())
    config["schedule"] = [{"duration_s": 1e-6}]
    start = time.perf_counter()
    try:
        report = run(config, device="cpu")
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)
    elapsed = time.perf_counter() - start
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in sources.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".report.json").write_text(json.dumps(report, allow_nan=False))
    # The independent auditor consumes the serialized public JSON schema.
    report = json.loads(args.output.with_suffix(".report.json").read_text())
    # Audit timing is separate, and is not attributed to the solver run.
    checked = audit(report["config"], report)
    result = dict(elapsed_s=elapsed, split_characteristics=args.split_characteristics,
                  costs=costs, audit=checked, source_sha256=sources,
                  torch_version=torch.__version__, threads=torch.get_num_threads(),
                  simulated_time_s=1e-6)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
