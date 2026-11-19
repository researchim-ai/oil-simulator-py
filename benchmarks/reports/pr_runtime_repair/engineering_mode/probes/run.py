import json,time
from pathlib import Path
from surface.pr_model_reduction import reduce_cells
from run_pr_controlled_surface import run
from check_pr_controlled_surface import audit
out=Path('/tmp/pr-engineering-mode')
c=json.loads(Path('configs/surface_pr_integrated_facility.json').read_text())
c['schedule']=[{'duration_s':.001}]
c,mapping=reduce_cells(c,['gas_recycle_0']);c=json.loads(json.dumps(c))
(out/'config.json').write_text(json.dumps(c,indent=2));(out/'mapping.json').write_text(json.dumps(mapping,indent=2))
t=time.perf_counter()
def progress(p):
 print(json.dumps(p),flush=True)
 with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(p)+'\n')
r=run(c,progress=progress);seconds=time.perf_counter()-t
r=json.loads(json.dumps(r));(out/'result.json').write_text(json.dumps(r));checked=audit(c,r)
(out/'summary.json').write_text(json.dumps({'seconds':seconds,'audit':checked},indent=2));print(json.dumps({'seconds':seconds,'audit':checked}))
