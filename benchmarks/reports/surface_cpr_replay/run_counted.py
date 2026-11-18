import sys,runpy,json
from simulator.cpr_gpu import CPRPreconditionerGPU
source=sys.argv.pop(1); original=CPRPreconditionerGPU.matvec
counts=dict(captures=0,replays=0,eager=0); last={}
def counted(self,r):
 before=[self.graph_captures,self.graph_replays,self.graph_eager_calls]
 result=original(self,r)
 after=[self.graph_captures,self.graph_replays,self.graph_eager_calls]
 for key,a,b in zip(counts,before,after): counts[key]+=b-a
 w=getattr(self,'replay_workspace',None)
 if w is not None:last.update(w.report())
 return result
CPRPreconditionerGPU.matvec=counted
try:runpy.run_path(source+'/scripts/run_fim_facility_forecast.py',run_name='__main__')
finally:print('CPR_COUNTERS',json.dumps({'calls':counts,'workspace':last}),flush=True)
