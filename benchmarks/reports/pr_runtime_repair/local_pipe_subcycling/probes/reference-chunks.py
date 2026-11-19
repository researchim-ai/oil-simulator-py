import json,time
from pathlib import Path
from run_pr_controlled_surface import run
from check_pr_controlled_surface import audit
root=Path('/tmp/pr-local-subcycles')
base=json.loads(Path('configs/surface_pr_integrated_facility.json').read_text())
files=sorted(root.glob('reference-chunk-*.json'))
index=len(files)
ends=[1e-5,2e-5,3e-5,4e-5,7e-5,1e-4]
if index>=len(ends): raise SystemExit('reference complete')
if files: base['initial']={'state':json.loads(files[-1].read_text())['report']['intervals'][-1]['state']}
t0=base['initial']['state']['plant']['transport']['time_s']
base['schedule']=[{'duration_s':ends[index]-t0}]
t=time.perf_counter()
r=run(base)
wall=time.perf_counter()-t
r=json.loads(json.dumps(r))
checked=audit(base,r)
out={'seconds':wall,'report':r,'audit':checked}
p=root/f'reference-chunk-{index:02}.json';p.write_text(json.dumps(out))
print(json.dumps({'chunk':index+1,'seconds':wall,'time_s':r['intervals'][-1]['state']['plant']['transport']['time_s'],'attempts':r['intervals'][0]['attempts'],'audited':True}),flush=True)
