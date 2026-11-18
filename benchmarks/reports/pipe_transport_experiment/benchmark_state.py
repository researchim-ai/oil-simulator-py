import json,statistics,torch
from torch.utils.benchmark import Timer
from test_surface_blackoil_properties import model
from surface.blackoil_pipe_batch import tensor_state
from surface.blackoil_pvto_batch import PackedPVTO

torch.set_num_threads(1)
results={}
for device in ('cpu','cuda'):
 if device=='cuda':torch.cuda.set_per_process_memory_fraction(.12)
 m=model().to(device); packed=PackedPVTO(m)
 p=torch.linspace(1.2e6,3.8e6,128,dtype=torch.float64,device=device).expand(32,-1).clone().requires_grad_(True)
 masses=torch.tensor([[.85,2.,.06]],dtype=torch.float64,device=device).expand(32,-1).clone().requires_grad_(True)
 def run(transport):
  state=tensor_state(m,p,masses,pvto=packed,transport_only=transport)
  vp=torch.autograd.grad(state['specific_volume'].sum(),p,create_graph=True)[0]
  return vp,state['viscosity_pa_s']
 measurements={False:[],True:[]}
 for repeat in range(4):
  for transport in ((False,True) if repeat%2==0 else (True,False)):
   t=Timer('run(transport)',globals={'run':run,'transport':transport},num_threads=1).blocked_autorange(min_run_time=.2)
   measurements[transport].append(t.median)
 results[device]={'full_seconds':measurements[False],'transport_seconds':measurements[True],'ratio':statistics.median(measurements[False])/statistics.median(measurements[True])}
results['scope']='32 streams x 128 pressures including dv/dp AD; repeated short local microbenchmark, not full-model speedup.'
print(json.dumps(results,indent=2))
