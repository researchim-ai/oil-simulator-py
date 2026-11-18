import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'scripts'))
from check_norne_run import compare_same_source_summaries
root=Path('.scratch/norne_cpr_validation')
a,b=root/'baseline',root/'replay_fixed'
manifests=[json.loads((p/'manifest.json').read_text()) for p in (a,b)]
for key in ('input_files','reference_files','profile','device','threads','cuda_visible_devices','requested_days'):
 assert manifests[0][key]==manifests[1][key],key
changed=[k for k in manifests[0]['source_files'] if manifests[0]['source_files'][k]!=manifests[1]['source_files'][k]]
assert changed==['src/simulator/cpr_replay.py'],changed
r={'declared_source_changes':changed,'vectors':[]}
with np.load(a/'summary.npz') as x,np.load(b/'summary.npz') as y:
 assert set(x.files)==set(y.files) and np.array_equal(x['TIME'],y['TIME'])
 for key in x.files:
  xx,yy=x[key],y[key];valid=np.isfinite(xx)&np.isfinite(yy)
  assert np.array_equal(np.isfinite(xx),np.isfinite(yy)),key
  delta=np.abs(xx[valid]-yy[valid]);scale=max(1.,float(np.max(np.abs(xx[valid]),initial=0)),float(np.max(np.abs(yy[valid]),initial=0)))
  r['vectors'].append({'vector':key,'max_absolute_difference':float(np.max(delta,initial=0)),'max_scaled_difference':float(np.max(delta,initial=0)/scale),'exact':bool(np.array_equal(xx,yy,equal_nan=True))})
metas=[json.loads((p/'meta.json').read_text()) for p in (a,b)]
for p in (a,b):
 audit=json.loads((p/'validation.json').read_text())
 assert audit['status']=='passed_recorded_audit'
r['runs']=[]
for p,m in zip((a,b),metas):
 samples=[json.loads(l) for l in (p/'execution_samples.jsonl').read_text().splitlines()]
 last=json.loads((p/'audit_status.json').read_text())
 r['runs'].append({**{k:m[k] for k in ('wall_s','invocation_solver_s','substeps','failed_substeps','newton_iters','days','device')},'workspace':last['cpr_replay_workspace'],'memory_samples':len(samples),'peak_allocated_mb':max(x['cuda_peak_allocated_mb'] for x in samples),'max_reserved_mb':max(x['cuda_reserved_mb'] for x in samples),'final_allocated_mb':samples[-1]['cuda_allocated_mb'],'final_reserved_mb':samples[-1]['cuda_reserved_mb']})
r['observed_wall_ratio']=metas[0]['wall_s']/metas[1]['wall_s']
r['observed_solver_ratio']=metas[0]['invocation_solver_s']/metas[1]['invocation_solver_s']
x,y=[np.load(p/'substeps.npz')['data'] for p in (a,b)]
r['exact_substep_decisions']=x.shape==y.shape and np.array_equal(x[:,:4],y[:,:4])
r['fields']={}
with np.load(a/'fields.npz') as x,np.load(b/'fields.npz') as y:
 r['field_snapshot_dates_equal']=np.array_equal(x['t'],y['t'])
 if r['field_snapshot_dates_equal']:
  for k in ('p','sw','sg','so','rs'):
   r['fields'][k]={'max_abs':float(np.max(np.abs(x[k].astype(float)-y[k].astype(float)))),'exact':bool(np.array_equal(x[k],y[k]))}
r['scope']='One sequential full Norne pair, same GPU; declared pool-lifecycle source change; field snapshots float32, summary float64. Descriptive comparison, not independent validation or isolated repeated timing.'
(root/'comparison.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='vectors'},indent=2))
