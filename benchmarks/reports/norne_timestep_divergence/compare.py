"""Compare controlled Norne grids; saved full-run pressure is in bar."""
import json
from pathlib import Path
import numpy as np
root=Path('.scratch/norne_local_refinement')
report={}
for label,baseline in [('current_step','.scratch/norne_full_newton/run'),('historical_step','.scratch/norne_cpr_validation/baseline')]:
 with np.load(root/label/'fields.npz') as a,np.load(Path(baseline)/'fields.npz') as b:
  mask=b['pv'].reshape(-1)>0
  j=int(np.flatnonzero(np.isclose(b['t'],755))[0]);r={}
  for k in ['p','sw','sg','so','rs']:
   u=a[k].reshape(-1);v=b[k][j].reshape(-1)[mask]
   if k=='p':u=u/1e5
   d=u.astype(np.float32).astype(float)-v.astype(float)
   r[k]={'max_abs':float(abs(d).max()),'unit':'bar' if k=='p' else 'dimensionless' if k in ['sw','sg','so'] else 'surface_volume_ratio'}
  report[label+'_vs_full']=r
for label in ['historical_step','half_step']:
 if not (root/label/'fields.npz').exists():continue
 with np.load(root/'current_step'/'fields.npz') as a,np.load(root/label/'fields.npz') as b:
  w=a['pv'];r={}
  for k in ['p','sw','sg','so','rs']:
   d=a[k].reshape(-1)-b[k].reshape(-1)
   r[k]={'max_abs':float(abs(d).max()),'pv_rms':float(np.sqrt(np.dot(w,d*d)/w.sum())),'unit':'Pa' if k=='p' else 'dimensionless' if k in ['sw','sg','so'] else 'surface_volume_ratio'}
  report['current_vs_'+label]=r
report['runs']={p.parent.name:json.loads(p.read_text()) for p in root.glob('*/status.json')}
(root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))
