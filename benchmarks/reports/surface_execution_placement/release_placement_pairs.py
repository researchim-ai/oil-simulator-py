import json,os,shutil,statistics,subprocess,sys
from pathlib import Path
ROOT=Path('/home/user/workspace/oil-simulator-py-impes');work=ROOT/'.scratch/surface_next'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from check_facility_refinement_pair import completed
from compare_fim_facility_forecast import compare as compare_fields
from check_long_facility_trajectory import compare as compare_wells
from check_facility_capacity_search import FIELD_BUDGETS,WELL_BUDGETS
candidate=work/'release_source'
base=json.loads((work/'config.json').read_text())
hybrid=json.loads(json.dumps(base))
hybrid['simulation']['physical_surface_network']['network_backend']='cpu_reference'
(work/'hybrid_config.json').write_text(json.dumps(hybrid,indent=2))
expected=json.loads(json.dumps(hybrid))
expected['simulation']['physical_surface_network']['network_backend']='torch_device'
assert expected==base
rows=[];checks=[]
for repeat in range(3):
 outputs={}
 for enabled in ((False,True) if repeat%2==0 else (True,False)):
  name='candidate' if enabled else 'reference';source=candidate if enabled else work/'reference_source'
  out=work/f'release_placement{repeat}_{name}';outputs[enabled]=out
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-2f9ce359-f3f4-accc-bf9c-31bc88bb1cb7',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/pr_forecast_mpl',PYTHONPATH=f'{source/"src"}:{source/"scripts"}')
  cmd=[sys.executable,str(source/'scripts/run_fim_facility_forecast.py'),'--config',str(work/('hybrid_config.json' if enabled else 'config.json')),'--days','.00005','--report-days','.00005','--device','cuda','--cuda-memory-fraction','.08','--output',str(out)]
  with (work/f'release_placement{repeat}_{name}.log').open('w') as log:
   subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
  summary,bundle,path=completed(out,.00005)
  row={'repeat':repeat,'candidate':enabled,'solve_wall_s':summary['solve_wall_s'],'audit':summary['audit'],'execution':summary['execution'],'temporal_work':bundle['result']['temporal_work']}
  rows.append(row);print(json.dumps(row),flush=True)
 fields=compare_fields(outputs[False]/'trajectory.json.gz',outputs[True]/'trajectory.json.gz',FIELD_BUDGETS,allow_numerical_policy=True)
 wells=compare_wells(outputs[False]/'trajectory.json.gz',outputs[True]/'trajectory.json.gz',allow_numerical_policy=True)
 wells['budgets']=WELL_BUDGETS;wells['accepted']=wells['mode_mismatches']==0 and all(wells['errors'][k]<=v for k,v in WELL_BUDGETS.items())
 check={'repeat':repeat,'fields':fields,'wells':wells,'accepted':fields['accepted'] and wells['accepted']}
 checks.append(check)
 if not check['accepted']:raise RuntimeError('Fixed strict GPU regression failed')
 print(json.dumps(check),flush=True)
medians={str(c):statistics.median(r['solve_wall_s'] for r in rows if r['candidate'] is c) for c in (False,True)}
result={'accepted':True,'scope':'Three alternating source-guarded GPU gathering vs CPU gathering pairs, reservoir/linear system on RTX3090Ti, allocator8%, shared/busy device; 1024cells,16wells,.00005days. Unchanged physics/FIM/time/search. Fixed strict regression budgets; not full-month/Norne or isolated GPU scaling.','rows':rows,'checks':checks,'median_s':medians,'observed_ratio':medians['False']/medians['True']}
(work/'release_placement_pairs.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('rows','checks')}),flush=True)
