import hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path('/home/user/workspace/oil-simulator-py-impes');sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_pr_controlled_surface import run
from surface.torch_nonlinear import TorchNewton
from surface.pr_surface_control import PRControlledSurface
from surface.pr_gravity_transport import PRGravityTransportNetwork
cfg=json.loads((ROOT/'configs/surface_pr_integrated_facility_liquid_mixture.json').read_text())
cfg['schedule']=[{'duration_s':1e-6}]
paths=[*sorted((ROOT/'src/surface').glob('*.py')),ROOT/'scripts/run_pr_controlled_surface.py']
source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
solves=[];physical=[];errors=[]
original_solve=TorchNewton.solve; original_step=PRGravityTransportNetwork._step; original_error=PRControlledSurface._error

def solve(self,function,x,*args,**kwargs):
 start=time.perf_counter();ans=original_solve(self,function,x,*args,**kwargs)
 solves.append({'unknowns':x.numel(),'elapsed_s':time.perf_counter()-start,'iterations':self.iterations,'evaluations':ans[3],'jacobian_builds':self.jacobian_builds,'jacobian_reuses':self.jacobian_reuses,'retry_refreshes':self.jacobian_retry_refreshes,'stop_reason':self.stop_reason,'maximum_residual':float(ans[1].abs().max())})
 return ans

def step(self,state,dt,*args,**kwargs):
 first=len(solves);start=time.perf_counter();ans=original_step(self,state,dt,*args,**kwargs)
 physical.append({'dt_s':dt,'elapsed_s':time.perf_counter()-start,'newton_solves':len(solves)-first,'newton_unknowns':[v['unknowns'] for v in solves[first:]],'accepted_quadrature_order':ans[1]['quadrature_order'],'quadrature_error':ans[1].get('quadrature_error')})
 return ans

def error(self,a,b):
 result=original_error(self,a,b);t=self.plant.transport;scores={};mass=t.fluid.molar_masses.tolist()
 for (name,x),(_,y) in zip(a.plant.transport.nodes,b.plant.transport.nodes):
  pre='node:'+name+':'
  scores[pre+'pressure']=abs(x.pressure_pa-y.pressure_pa)/(1+t.relative_tolerance*max(x.pressure_pa,y.pressure_pa))
  scores[pre+'temperature']=abs(x.temperature_k-y.temperature_k)/t.temperature_tolerance_k
  scores[pre+'energy']=abs(x.total_energy_j-y.total_energy_j)/(1e-6+t.relative_tolerance*max(1,abs(x.total_energy_j),abs(y.total_energy_j)))
  for i,(u,v) in enumerate(zip(x.component_moles,y.component_moles)):scores[pre+'component:'+str(i)]=abs(u-v)/(1e-10+t.relative_tolerance*max(abs(u),abs(v)))
  ux=x.momentum_kg_m_s/sum(n*m for n,m in zip(x.component_moles,mass));uy=y.momentum_kg_m_s/sum(n*m for n,m in zip(y.component_moles,mass))
  scores[pre+'velocity']=abs(ux-uy)/t.velocity_tolerance_m_s
 for (name,x),(_,y) in zip(a.plant.transport.rotors,b.plant.transport.rotors):
  d=t._drives[name];scores['rotor:'+name]=abs(x-y)/(1e-8+d.relative_speed_tolerance*d.nominal_speed_rad_s)
 for (name,x),(_,y) in zip(a.plant.columns,b.plant.columns):
  c=self.plant.columns[name];pre='column:'+name+':'
  scores[pre+'phase_count']=2. if len(x.phase_roots)!=len(y.phase_roots) else 0.
  scores[pre+'temperature']=abs(x.temperature_k-y.temperature_k)/t.temperature_tolerance_k
  for i,(u,v) in enumerate(zip(x.component_moles,y.component_moles)):scores[pre+'component:'+str(i)]=abs(u-v)/(1e-10+t.relative_tolerance*max(abs(u),abs(v)))
  for i,(u,v) in enumerate(zip(x.interface_heights_m,y.interface_heights_m)):scores[pre+'interface:'+str(i)]=abs(u-v)/(1e-8+c.relative_tolerance*c.geometry.height_m)
 for (name,x),(_,y) in zip(a.positions,b.positions):scores['actuator:'+name]=abs(x-y)/self.position_tolerance
 unavailable=set(a.inactive_feedback)|set(b.inactive_feedback)
 for (name,x),(_,y) in zip(a.integrals,b.integrals):
  if name not in unavailable:scores['integral:'+name]=abs(x-y)/self.integral_tolerance
 assert not self.trips and not a.plant.walls
 order=sorted(scores.items(),key=lambda x:x[1],reverse=True)
 errors.append({'actual_error':result,'manual_max':order[0][1],'manual_matches':math.isclose(result,order[0][1],rel_tol=1e-8,abs_tol=1e-12),'largest_contributors':order[:16],'locally_predicted_dt_s':1e-6*0.9/math.sqrt(max(result,1e-30))})
 return result
TorchNewton.solve=solve;PRGravityTransportNetwork._step=step;PRControlledSurface._error=error
start=time.perf_counter()
try:result=run(cfg,device='cpu')
finally:TorchNewton.solve=original_solve;PRGravityTransportNetwork._step=original_step;PRControlledSurface._error=original_error
elapsed=time.perf_counter()-start
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==v for name,v in source.items())
result['source_sha256']=source
Path('/tmp/surface_audit_instrumented_report.json').write_text(json.dumps(result,allow_nan=False)+'\n')
out={'scope':'Instrumented CPU one-microsecond startup. Times are diagnostic, not performance claims.','elapsed_s':elapsed,'global_newton_solves':solves,'physical_trials':physical,'temporal_error':errors,'sources_unchanged':True,'source_sha256':source}
Path('/tmp/surface_audit_instrumentation.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'},indent=2))
