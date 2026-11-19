import json,time,signal
from dataclasses import replace
from pathlib import Path
from surface.pr_surface_control import PRControlledSurface,PRControlledSurfaceState
from surface.pr_subcycled_surface import PRSubcycledSurface
from run_pr_controlled_surface import run
from check_pr_controlled_surface import audit
root=Path('/tmp/pr-local-subcycles');c=json.loads(Path('configs/surface_pr_integrated_facility.json').read_text())
c['model'].update(initial_step_s=5e-5,maximum_step_s=5e-5)
b=PRControlledSurface.from_dict(c['model']);s=replace(PRControlledSurfaceState.from_dict(c['initial']['state']),model_id=b.model_id,next_step_s=5e-5)
m=PRSubcycledSurface(b,['gas_recycle_0'],local_error_fraction=.025)
c['model']=m.to_dict();c['initial']={'state':m.adopt(s).to_dict()};c['schedule']=[{'duration_s':1e-4}]
t=time.perf_counter();r=run(c,progress=lambda p:print(json.dumps(p),flush=True));seconds=time.perf_counter()-t
r=json.loads(json.dumps(r));c=r['config'];(root/'refinement-raw.json').write_text(json.dumps({'seconds':seconds,'report':r}));checked=audit(c,r)
original=json.loads((root/'result.json').read_text());a=PRControlledSurfaceState.from_dict(original['intervals'][-1]['state']);fine=PRControlledSurfaceState.from_dict(r['intervals'][-1]['state']);norm=m._error(a,fine)
(root/'refinement.json').write_text(json.dumps({'seconds':seconds,'audit':checked,'difference_original_norm':norm,'report':r}));print(json.dumps({'seconds':seconds,'audit':checked,'difference_original_norm':norm}))
