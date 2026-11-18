import json,torch
from pathlib import Path
from simulator.cpr_gpu import CPRPreconditionerGPU
torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.2)
p=Path('.scratch/norne_cpr_validation');state=torch.load(p/'baseline/latest_checkpoint.pt',map_location='cuda:0',weights_only=False)
f=state['sim'].fim_solver
assert getattr(f,'_cpr_replay_workspace',None) is None
f.cpr_reuse_graph=True
f.cpr_graph_input_budget_bytes=128*1024**2
original=CPRPreconditionerGPU.matvec; records=[]
def check(self,r):
 actual=original(self,r);expected=self._matvec_eager(r)
 records.append({'abs':float((actual-expected).abs().max()),'relative_norm':float((actual-expected).norm()/expected.norm().clamp_min(1e-300))})
 return actual
CPRPreconditionerGPU.matvec=check
ok=f.step(86400,current_time=state['t']*86400)
r={'checkpoint_day':state['t'],'step_ok':ok,'calls':len(records),'max_absolute':max(x['abs'] for x in records),'max_relative_norm':max(x['relative_norm'] for x in records),'workspace':f._cpr_replay_workspace.report(),'scope':'One actual Norne step: graph CPR versus fresh eager application at every RHS; not a proof for all states.'}
(p/'matvec_probe.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
