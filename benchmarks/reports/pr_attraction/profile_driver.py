"""Instrument one complete facility request; EOS time is not whole-run speedup."""
import json
import time
from pathlib import Path
from surface.equilibrium_properties import EquilibriumFluid
from run_pr_controlled_surface import run
from audit_pr_surface_collocation import audit

root = Path(__file__).resolve().parents[3]
config = json.loads((root / 'benchmarks/reports/pr_prepared_facility/qualified/request.json').read_text())
original = EquilibriumFluid.tensor_homogeneous
calls = elapsed = depth = 0

def instrument(self, *args, **kwargs):
    global calls, elapsed, depth
    outer = depth == 0
    depth += 1
    start = time.perf_counter()
    try:
        return original(self, *args, **kwargs)
    finally:
        depth -= 1
        if outer:
            calls += 1
            elapsed += time.perf_counter() - start

EquilibriumFluid.tensor_homogeneous = instrument
try:
    start = time.perf_counter()
    result = run(config, integrator='sdirk2-common')
    total = time.perf_counter() - start
finally:
    EquilibriumFluid.tensor_homogeneous = original
checked = audit(config, json.loads(json.dumps(result)))
share = elapsed / total
out = {'accepted':checked['accepted'], 'eos_calls':calls, 'instrumented_total_s':total,
       'instrumented_eos_s':elapsed, 'eos_wall_fraction':share,
       'ideal_speedup_if_eos_were_free':1/(1-share),
       'coverage':'tensor_homogeneous only; direct chemical kernels and phase-search orchestration excluded',
       'scope':'diagnostic instrumentation includes overhead and initialization; one short operating interval, not a timing benchmark or phase stability-only profile'}
Path(__file__).with_name('eos_share.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
