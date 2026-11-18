"""Isolated transaction bookkeeping; not a reservoir solve benchmark."""
import json, statistics, time
from types import SimpleNamespace
import torch
from simulator.fim_transaction import _StepState

torch.set_num_threads(1)
torch.cuda.set_per_process_memory_fraction(.08)
rows=[]
for n in (262144,1048576):
    fluid=SimpleNamespace(**{k:torch.full((n,),v,dtype=torch.float64,device='cuda') for k,v in [('pressure',3e6),('s_w',.2),('s_o',.77)]})
    wells=SimpleNamespace(wells=[],p_bh=torch.ones(64,device='cuda'))
    solver=SimpleNamespace(fluid=fluid,wells=wells)
    measurements={m:[] for m in ('recapture','reusable')}
    for round_ in range(5):
        for mode in (('recapture','reusable') if round_%2==0 else ('reusable','recapture')):
            state=_StepState(solver)
            for i in range(10):
                state.restore(reusable=mode=='reusable')
                if mode=='recapture':state=_StepState(solver)
            torch.cuda.synchronize();start=time.perf_counter()
            for i in range(100):
                state.restore(reusable=mode=='reusable')
                if mode=='recapture':state=_StepState(solver)
            torch.cuda.synchronize()
            measurements[mode].append((time.perf_counter()-start)/100)
    rows.append({'cells':n,'removed_reservoir_copy_bytes_per_restore':n*24,'seconds_per_restore':measurements,'median_seconds':{k:statistics.median(v) for k,v in measurements.items()}})
print(json.dumps({'scope':__doc__,'device':torch.cuda.get_device_name(),'shared_gpu':True,'rows':rows},indent=2))
