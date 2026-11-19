"""Same prepared state and tolerances; legacy versus existing optimized Newton.

Run with PYTHONPATH=src:scripts:tests, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1.
"""
import contextlib
import io
import json
import pickle
import time
from pathlib import Path

from check_dynamic_receiver import audit_step
from test_fim_receiver_operating_point import case

from simulator.dynamic_receiver_coupling import DynamicReceiverCoupledFIM
from surface import pr_operating_point, stratified_pr, stratified_pr_flow, stratified_pr_thermal
from surface.torch_linear import TorchNewton as Legacy
from surface.torch_nonlinear import TorchNewton as Optimized

modules = (stratified_pr, stratified_pr_flow, stratified_pr_thermal, pr_operating_point)
for module in modules:
    module.TorchNewton = Legacy
with contextlib.redirect_stdout(io.StringIO()):
    base = case()
    state, _ = base.prepare_operating_point()
    snapshot = pickle.dumps(base)
results = []
for name, cls in (("legacy", Legacy), ("optimized", Optimized)):
    counters = {"solves": 0, "residual_evaluations": 0}
    class Measured(cls):
        def solve(self, *args, _counters=counters, **kwargs):
            result = super().solve(*args, **kwargs)
            _counters["solves"] += 1
            _counters["residual_evaluations"] += result[3]
            return result
    for module in modules:
        module.TorchNewton = Measured
    with contextlib.redirect_stdout(io.StringIO()):
        restored = pickle.loads(snapshot)
        model = DynamicReceiverCoupledFIM(
            restored.solver, restored.receiver, state, restored.assay,
            restored.thermal_conditions, restored.inlet_height_m,
            {"surface_subcycling": True},
        )
        model.outlet_openings = {"export": .9}
        started = time.perf_counter()
        accepted = model.step(.05)
        elapsed = time.perf_counter() - started
    row = {"implementation": name, "accepted": accepted, "wall_s": elapsed,
               "counters": dict(counters), "attempt": model.last_attempt}
    if accepted:
        row["state"] = model.receiver_state.to_dict()
        row["report"] = model.report
        row["audit"] = audit_step({**model.to_dict(), "initial_state": state.to_dict()}, model.report)
        row["reservoir"] = {key: getattr(model.solver.fluid, key).cpu().tolist()
                            for key in ("pressure", "s_w", "s_g", "rs")}
    results.append(row)
    Path(__file__).with_name("comparison.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    print(name, accepted, elapsed, counters, flush=True)
