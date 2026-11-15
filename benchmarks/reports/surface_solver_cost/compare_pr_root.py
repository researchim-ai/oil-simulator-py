"""Same first captured PR residual and exact AD tangent, three root solvers.

Not a full integration benchmark. SciPy is an independent optional comparator.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
import numpy as np
from scipy.optimize import least_squares
import torch
from run_pr_controlled_surface import run
from surface.torch_nonlinear import TorchNewton
from surface.torch_linear import ColoredJacobian


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    captured = []
    class Captured(Exception):
        pass
    original = TorchNewton.solve
    def capture(self, fn, x, lo, hi, **kw):
        captured.append((fn, x.clone(), lo.clone(), hi.clone(), kw))
        raise Captured
    TorchNewton.solve = capture
    try:
        config = json.loads((ROOT/"configs/surface_pr_integrated_facility_liquid_mixture.json").read_text())
        config["schedule"] = [{"duration_s":1e-6}]
        try:
            run(config, device="cpu")
        except Captured:
            pass
    finally:
        TorchNewton.solve = original
    assert len(captured) == 1
    fn, x, lo, hi, kw = captured[0]
    rows = []
    for dense in (128, 768):
        solver = TorchNewton("cpu", maximum_iterations=40, dense_limit=dense)
        start = time.perf_counter()
        with torch.no_grad():
            z, r, ok, count = solver.solve(fn, x.clone(), lo, hi, **kw)
        rows.append(dict(method="torch_sparse" if dense == 128 else "torch_dense",
                         elapsed_s=time.perf_counter()-start, maximum_residual=float(r.abs().max()),
                         qualified=bool(ok), evaluations=count, jacobians=solver.jacobian_builds))
    coloring = ColoredJacobian(len(x), *kw["pattern"], "cpu")
    count = {"residual":0, "jacobian":0}
    def fun(v):
        count["residual"] += 1
        return fn(torch.tensor(v, dtype=x.dtype)).detach().numpy()
    def jac(v):
        count["jacobian"] += 1
        values = coloring.values(fn, torch.tensor(v, dtype=x.dtype)).detach().numpy()
        matrix = np.zeros((len(x),len(x)))
        matrix[coloring.rows,coloring.cols] = values
        return matrix
    start = time.perf_counter()
    result = least_squares(fun, x.numpy(), jac=jac, bounds=(lo.numpy(),hi.numpy()),
                           method="trf", x_scale="jac", ftol=None, gtol=1e-14, xtol=1e-14, max_nfev=80)
    elapsed = time.perf_counter()-start
    r = fn(torch.tensor(result.x, dtype=x.dtype))
    # The benchmark checks the SAME final row budgets; xtol alone is insufficient.
    bounds = kw["roundoff_tolerance"]()
    budget = torch.maximum(bounds, torch.full_like(r, kw["stagnation_tolerance"]))
    rows.append(dict(method="scipy_trf_exact_torch_ad", elapsed_s=elapsed,
                     maximum_residual=float(r.abs().max()), qualified=bool((r.abs() <= budget).all()),
                     evaluations=count["residual"], jacobians=count["jacobian"],
                     scipy_success=result.success, message=result.message))
    output = dict(scope="First nonlinear PR system only; not a full physical trajectory",
                  unknowns=len(x), colors=len(coloring.seeds), results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
