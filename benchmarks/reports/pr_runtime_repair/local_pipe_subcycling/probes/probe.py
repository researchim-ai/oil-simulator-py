import json,time,signal
from pathlib import Path
from surface.pr_surface_control import PRControlledSurface,PRControlledSurfaceState
from surface.pr_subcycled_surface import PRSubcycledSurface
from run_pr_controlled_surface import run
c=json.loads(Path('configs/surface_pr_integrated_facility.json').read_text())
m=PRSubcycledSurface(PRControlledSurface.from_dict(c['model']),['gas_recycle_0'])
s=m.adopt(PRControlledSurfaceState.from_dict(c['initial']['state']))
c['model']=m.to_dict();c['initial']={'state':s.to_dict()}
Path('/tmp/pr-local-subcycles/config.json').write_text(json.dumps(c,indent=2))
c['schedule']=[{'duration_s':1e-4}]
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('300s probe budget')))
signal.alarm(300)
start=time.perf_counter()
r=run(c,progress=lambda p:print(json.dumps(p),flush=True))
Path('/tmp/pr-local-subcycles/result.json').write_text(json.dumps(r,indent=2))
print('seconds',time.perf_counter()-start,flush=True)
