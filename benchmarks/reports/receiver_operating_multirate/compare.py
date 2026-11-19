import contextlib,io,json,pickle,time
from pathlib import Path
from test_fim_receiver_operating_point import case
from simulator.dynamic_receiver_coupling import DynamicReceiverCoupledFIM
from check_dynamic_receiver import audit_step
with contextlib.redirect_stdout(io.StringIO()):
 base=case(); state,initialization=base.prepare_operating_point()
result={'initialization':initialization,'runs':[]}
import simulator.receiver_subcycling as sub
from surface.stratified_pr_flow import ReceiverFeedPressureError
original_advance=sub.advance_receiver_locally
def traced(*a,**kw):
 t=time.perf_counter()
 try:
  x=original_advance(*a,**kw)
  print('local accepted',a[2],len(x[1]['surface_steps']),time.perf_counter()-t,file=__import__('sys').stderr,flush=True)
  return x
 except ReceiverFeedPressureError as e:
  print('pressure lower bound',e.receiver_pressure_pa,time.perf_counter()-t,file=__import__('sys').stderr,flush=True)
  raise
sub.advance_receiver_locally=traced

for enabled in (False,True):
 with contextlib.redirect_stdout(io.StringIO()):
  solver=pickle.loads(pickle.dumps(base.solver))
  m=DynamicReceiverCoupledFIM(solver,base.receiver,state,base.assay,
      base.thermal_conditions,base.inlet_height_m,{'surface_subcycling':enabled})
  m.outlet_openings={'export':.9}
  calls=[]; actual=solver.step
  def count(dt,*a,**kw):
   calls.append(dt); return actual(dt,*a,**kw)
  solver.step=count
  start=time.perf_counter(); accepted=m.step(.05); elapsed=time.perf_counter()-start
 row={'subcycling':enabled,'accepted':accepted,'wall_s':elapsed,'reservoir_calls':calls,'attempt':m.last_attempt}
 if accepted:
  row['report']=m.report;row['audit']=audit_step({**m.to_dict(),'initial_state':state.to_dict()},m.report)
 result['runs'].append(row)
 Path('/tmp/receiver-multirate/comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print({k:v for k,v in row.items() if k not in ('report','audit')},flush=True)
