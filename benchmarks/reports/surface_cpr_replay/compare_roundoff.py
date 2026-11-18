"""Strict final/accepted states, with explicit roundoff gate for trial traces only."""
import copy,json,sys
from pathlib import Path
from check_facility_refinement_pair import completed
from benchmark_initial_cell_cache import require_equal
path=Path(sys.argv[1]);days=float(sys.argv[2]); a=completed(path/'reference',days)[1]['result'];b=completed(path/'candidate',days)[1]['result']
x,y=copy.deepcopy(a),copy.deepcopy(b); diffs=[]
def trace(a,b,path):
 if isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:trace(a[k],b[k],path+'/'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),path
  for i,(aa,bb) in enumerate(zip(a,b)):trace(aa,bb,path+'/'+str(i))
 elif a!=b:
  assert type(a) is float and type(b) is float,(path,a,b)
  error=abs(a-b);relative=error/max(abs(a),abs(b),1e-30)
  assert relative<=2e-14,(path,a,b,relative)
  diffs.append(dict(path=path,reference=a,candidate=b,relative=relative))
for i,(aa,bb) in enumerate(zip(x['accepted_substeps'],y['accepted_substeps'])):
 trace(aa['facility_coupling']['trace'],bb['facility_coupling']['trace'],str(i))
 bb['facility_coupling']['trace']=copy.deepcopy(aa['facility_coupling']['trace'])
require_equal(x,y)
print(json.dumps(dict(accepted=True,exact_final_and_accepted_states=True,trial_trace_relative_gate=2e-14,trial_trace_differences=diffs),indent=2,allow_nan=False))
