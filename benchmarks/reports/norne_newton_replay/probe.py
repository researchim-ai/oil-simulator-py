import os,sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path('src').resolve()))
import torch
import numpy as np
torch.use_deterministic_algorithms(True, warn_only=True)
p=Path('.scratch/norne_restart_diagnostics')
mode=sys.argv[1]
snapshot=torch.load(p/'half/prestep_484.pt',weights_only=False)
s=snapshot['solver'];torch.set_num_threads(1)
if mode=='iteration':s.amg_rebuild='iteration'
if mode=='limited':s.ds_max=min(s.ds_max,.05)
if mode=='budget':s.max_newton=80
if mode=='direction':
 original_update=s._apply_update
 def update(p,sw,x,sat,ug,dx,dbg):
  dp,dsw,dxx=dx.unbind(1)
  fp=(torch.clamp(s.dp_max_rel*p.abs(),min=s.dp_max_abs)/dp.abs().clamp_min(1e-300)).clamp(max=1.)
  dsg=torch.where(sat,dxx,torch.zeros_like(dxx))
  fs=(s.ds_max/torch.stack([dsw.abs(),dsg.abs(),(dsw+dsg).abs()]).amax(0).clamp_min(1e-300)).clamp(max=1.)
  limit=torch.full_like(dxx,s.drs_max_rel*max(s.pvt.rs_max,1e-12))
  if s.wet:limit=torch.where(ug,s.drs_max_rel*max(s.pvt.rv_max,1e-12),limit)
  fx=torch.where(sat,torch.ones_like(dxx),(limit/dxx.abs().clamp_min(1e-300)).clamp(max=1.))
  factor=torch.minimum(fp,torch.minimum(fs,fx))
  return original_update(p,sw,x,sat,ug,dx*factor[:,None],dbg)
 s._apply_update=update
trace=[];orig=s.convergence
def convergence(*a,**kw):
 ok,detail=orig(*a,**kw);trace.append(detail);return ok,detail
s.convergence=convergence
start=time.monotonic();ok=s.step(snapshot['dt'],current_time=snapshot['current_time']);elapsed=time.monotonic()-start
r={'mode':mode,'ok':ok,'wall_s':elapsed,'iterations':s.last_newton_iters,'criteria':trace,'linear':s._last_linear_stats}
(p/(mode+'.json')).write_text(json.dumps(r,indent=2))
print('RESULT',mode,ok,elapsed)
