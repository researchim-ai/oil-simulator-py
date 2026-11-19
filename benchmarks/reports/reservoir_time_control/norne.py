import sys,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path('src').resolve()))
import numpy as np
import torch
from simulator.eclipse_run import iter_deck
factor=float(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
config=dict(pressure_tolerance_pa=10000*factor,saturation_tolerance=.001*factor,
 component_relative_tolerance=.005*factor,withdrawal_relative_tolerance=.005*factor,
 withdrawal_absolute_tolerance_kg=1.,startup_step_days=.5)
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').rglob('*.py')}
(out/'manifest.json').write_text(json.dumps(dict(config=config,sources=sources,requested_days=25,device=torch.cuda.get_device_name(0)),indent=2))
ctx={};rows=[];start=time.monotonic();state='failed'
try:
 it=iter_deck('.scratch/ot2/norne/NORNE_ATW2013.DATA',device=torch.device('cuda:0'),
  max_days=25,quiet=True,yield_start=True,context=ctx,
  sim_overrides={'convergence_profile':'strict','newton_line_search_fallback':True,
  'initial_time_step_days':.5,'max_time_step_days':5.,'min_time_step_days':1e-6,'max_time_step_attempts':20,'reservoir_time_control':config})
 next(it);d=ctx['sim'].reservoir_time_control;s=ctx['solver'];original=d.step
 initial_mass=d._masses().sum(1).cpu().numpy();initial_net=d._withdrawal().sum(0).numpy()
 def checked(dt,current_time=None):
  ok=original(dt,current_time=current_time)
  row=dict(day=current_time/86400,dt=dt/86400,accepted=ok,attempt=dict(d.last_attempt))
  if ok:
   balance=(d._masses().sum(1).cpu().numpy()-initial_mass+d._withdrawal().sum(0).numpy()-initial_net)/np.maximum(abs(initial_mass),1.)
   assert np.isfinite(balance).all() and np.max(abs(balance))<1e-4
   assert d.report['indicator']<=1
   row['balance']=balance.tolist()
  rows.append(row)
  if len(rows)%10==0: print('attempt',len(rows),'day',current_time/86400,'accepted',ok,flush=True)
  return ok
 d.step=checked
 for _,day,summary in it:print('day',day,'attempts',len(rows),flush=True)
 np.savez_compressed(out/'summary.npz',**summary)
 np.savez_compressed(out/'fields.npz',**{k:v.detach().cpu().numpy() for k,v in d._fields().items()})
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items())
 assert abs(day-25)<1e-8
 state='complete'
finally:
 (out/'steps.json').write_text(json.dumps(rows,indent=2))
 (out/'status.json').write_text(json.dumps(dict(status=state,wall_s=time.monotonic()-start,
  accepted=sum(r['accepted'] for r in rows),rejected=sum(not r['accepted'] for r in rows)),indent=2))
