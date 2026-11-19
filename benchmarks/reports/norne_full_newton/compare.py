import json
from pathlib import Path
import numpy as np
new=Path('.scratch/norne_full_newton/run')
old=Path('.scratch/norne_cpr_validation/baseline')
r={'current':str(new),'historical':str(old),'note':'Different source versions; historical comparison is not an isolated optimization benchmark.'}
a=json.loads((new/'manifest.json').read_text());b=json.loads((old/'manifest.json').read_text())
r['inputs_equal']=a['input_files']==b['input_files']
r['changed_sources']=[k for k in sorted(set(a['source_files'])|set(b['source_files'])) if a['source_files'].get(k)!=b['source_files'].get(k)]
for filename,clock in [('summary.npz','TIME'),('fields.npz','t')]:
 with np.load(new/filename) as x,np.load(old/filename) as y:
  result={'clocks_equal':np.array_equal(x[clock],y[clock]),'vectors':{}}
  for k in sorted(set(x.files)&set(y.files)):
   u,v=x[k],y[k]
   if u.shape!=v.shape or (k not in [clock,'pv'] and not result['clocks_equal']):
    result['vectors'][k]={'comparable':False};continue
   valid=np.isfinite(u)&np.isfinite(v)
   delta=np.abs(u[valid].astype(float)-v[valid].astype(float))
   scale=max(1.,float(np.max(np.abs(v[valid])))) if valid.any() else 1.
   result['vectors'][k]={'equal':bool(np.array_equal(u,v,equal_nan=True)),'max_absolute_difference':float(delta.max()) if delta.size else None,'max_scaled_difference':float(delta.max()/scale) if delta.size else None,'finite_mask_mismatches':int(np.count_nonzero(np.isfinite(u)!=np.isfinite(v)))}
  r[filename]=result
r['timing']={label:json.loads((p/'meta.json').read_text()) for label,p in [('current',new),('historical',old)]}
(new/'historical_comparison.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k not in ['summary.npz','timing']},indent=2))
print('SUMMARY unequal',sum(not v.get('equal',False) for v in r['summary.npz']['vectors'].values()))
print('SUMMARY worst scaled',sorted(r['summary.npz']['vectors'].items(),key=lambda kv:kv[1].get('max_scaled_difference') or 0,reverse=True)[:10])
