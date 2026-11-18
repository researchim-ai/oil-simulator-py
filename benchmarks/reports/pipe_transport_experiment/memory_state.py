import gc,json,torch
from torch.profiler import profile,ProfilerActivity
from test_surface_blackoil_properties import model
from surface.blackoil_pipe_batch import tensor_state
from surface.blackoil_pvto_batch import PackedPVTO

torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.12)
m=model().to('cuda');packed=PackedPVTO(m)
p=torch.linspace(1.2e6,3.8e6,512,dtype=torch.float64,device='cuda').expand(128,-1).clone().requires_grad_(True)
masses=torch.tensor([[.85,2.,.06]],dtype=torch.float64,device='cuda').expand(128,-1).clone().requires_grad_(True)
def run(mode):
 state=tensor_state(m,p,masses,pvto=packed,transport_only=mode)
 vp=torch.autograd.grad(state['specific_volume'].sum(),p,create_graph=True)[0]
 return state,vp
for mode in (False,True):
 v=run(mode);del v
torch.cuda.synchronize();gc.collect();results={}
for mode in (False,True):
 torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
 v=run(mode);torch.cuda.synchronize()
 results[str(mode)]={'peak_increment_bytes':torch.cuda.max_memory_allocated()-base,'live_increment_bytes':torch.cuda.memory_allocated()-base}
 del v;gc.collect();torch.cuda.synchronize()
 with profile(activities=[ProfilerActivity.CPU]) as prof:
  v=run(mode)
 results[str(mode)]['dispatch_counts']={e.key:e.count for e in prof.key_averages() if e.key in ('aten::stack','aten::mul','aten::add','aten::sub')}
 del v;gc.collect();torch.cuda.synchronize()
results['scope']='128 streams x 512 pressure points, current properties plus differentiable dv/dp; CUDA tensor allocation excluding preexisting inputs and allocator reserve.'
print(json.dumps(results,indent=2))
