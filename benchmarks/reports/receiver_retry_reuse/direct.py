"""Same physical model without nested local integration; temporal refinement."""
import contextlib
import io
import json
import pickle
import time
from pathlib import Path

from check_dynamic_receiver import audit_step
from test_fim_receiver_operating_point import case

from surface import pr_operating_point, stratified_pr, stratified_pr_flow, stratified_pr_thermal
from surface.torch_linear import TorchNewton as Legacy
from surface.torch_nonlinear import TorchNewton as Optimized

modules = (pr_operating_point, stratified_pr, stratified_pr_flow, stratified_pr_thermal)
for module in modules:
    module.TorchNewton = Legacy
with contextlib.redirect_stdout(io.StringIO()):
    base = case()
    state, _ = base.prepare_operating_point()
    base.receiver_state = state
    base.outlet_openings = {'export': .9}
    snapshot = pickle.dumps(base)
for module in modules:
    module.TorchNewton = Optimized
rows = []
for dt, count in ((.005, 10), (.0025, 20)):
    model = pickle.loads(snapshot)
    integration_s, audits, reports = 0., [], []
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(count):
            before = model.receiver_state
            tick = time.perf_counter()
            accepted = model.step(dt, current_time=before.time_s)
            integration_s += time.perf_counter() - tick
            assert accepted, model.last_attempt
            audits.append(audit_step({**model.to_dict(), 'initial_state': state.to_dict()}, model.report))
            reports.append(model.report)
    rows.append({'dt_s': dt, 'intervals': count, 'integration_s': integration_s,
                 'wall_including_audits_s': time.perf_counter() - started,
                 'state': model.receiver_state.to_dict(), 'audits': audits, 'reports': reports,
                 'inlet_pressure_pa': model._pressure(model.receiver_state),
                 'reservoir': {key: getattr(model.solver.fluid, key).cpu().tolist()
                               for key in ('pressure', 's_w', 's_g', 'rs')}})
    Path(__file__).with_name('direct.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    print(dt, integration_s, 'accepted', len(audits), flush=True)
