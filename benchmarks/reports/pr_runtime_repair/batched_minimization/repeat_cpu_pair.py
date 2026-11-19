import json,time,runpy
from pathlib import Path
from unittest.mock import patch
import surface.equilibrium_properties as ep
from surface.pr_surface_control import PRControlledSurface
from run_pr_controlled_surface import run
from diagnose_pr_facility_error import ProbeFinished
original=ep.stability_candidates
from surface.equilibrium_stability_batch import stability_candidates as baseline
trial=PRControlledSurface._temporal_trial
results=[]
for label,fn in [('baseline',baseline),('batched_minimization',original)]:
    record={'variant':label}
    def observed(self,*a,**kw):
        start=time.perf_counter()
        coarse,fine,one,two=trial(self,*a,**kw)
        record.update(trial_seconds=time.perf_counter()-start,error=self._error(coarse,fine),coarse=coarse.to_dict(),fine=fine.to_dict())
        raise ProbeFinished()
    start=time.perf_counter()
    with patch.object(ep,'stability_candidates',fn),patch.object(PRControlledSurface,'_temporal_trial',observed):
        try:run(json.loads(Path('configs/surface_pr_integrated_facility.json').read_text()))
        except ProbeFinished:pass
    record['total_seconds']=time.perf_counter()-start
    results.append(record)
    Path('/tmp/pr-batched-minimization-repeat.json').write_text(json.dumps(results,indent=2))
    print({k:v for k,v in record.items() if k not in ('coarse','fine')},flush=True)
print('identical',all(results[0][k]==results[1][k] for k in ('coarse','fine','error')),flush=True)
