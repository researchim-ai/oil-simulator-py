import contextlib,io,json,time
from pathlib import Path
from test_fim_receiver_operating_point import case
from check_dynamic_receiver import audit_step
with contextlib.redirect_stdout(io.StringIO()):
 m=case(); state,initialization=m.prepare_operating_point(); m.receiver_state=state
 m.outlet_openings={'export':.9}; records=[]; calls=[]; actual=m.solver.step
 def count(dt,*a,**kw):
  calls.append(dt); return actual(dt,*a,**kw)
 m.solver.step=count
 start=time.perf_counter()
 for i in range(5):
  before=m.receiver_state
  assert m.step(.01,current_time=before.time_s),m.last_attempt
  records.append({'report':m.report,'audit':audit_step({**m.to_dict(),'initial_state':state.to_dict()},m.report)})
 elapsed=time.perf_counter()-start
result={'wall_s':elapsed,'reservoir_calls':calls,'records':records,'state':m.receiver_state.to_dict(),'reservoir_pressure_pa':m.solver.fluid.pressure.flatten().tolist()}
Path('/tmp/receiver-multirate/reference.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print({'wall_s':elapsed,'reservoir_calls':len(calls),'accepted_steps':len(records)},flush=True)
