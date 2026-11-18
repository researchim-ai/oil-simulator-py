import json,os,shutil,statistics,subprocess,sys
from pathlib import Path
ROOT=Path('/home/user/workspace/oil-simulator-py-impes');work=ROOT/'.scratch/surface_next'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from benchmark_initial_cell_cache import require_equal
from check_facility_refinement_pair import completed
candidate=work/'candidate_source'
for name in ('src','scripts'):
 shutil.copytree(ROOT/name,candidate/name,ignore=shutil.ignore_patterns('__pycache__'))
config=json.loads((ROOT/'configs/fim_blackoil_facility_depletion_fast.json').read_text())
config['fluid']['pvt_path']=str(ROOT/config['fluid']['pvt_path'])
(work/'config.json').write_text(json.dumps(config,indent=2))
rows=[]
for repeat in range(3):
 pair={}
 for enabled in ((False,True) if repeat%2==0 else (True,False)):
  name='candidate' if enabled else 'reference';source=candidate if enabled else work/'reference_source'
  out=work/f'pair{repeat}_{name}'
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/pr_forecast_mpl',PYTHONPATH=f'{source/"src"}:{source/"scripts"}')
  cmd=[sys.executable,str(source/'scripts/run_fim_facility_forecast.py'),'--config',str(work/'config.json'),'--days','.003','--report-days','.003','--device','cpu','--output',str(out)]
  with (work/f'pair{repeat}_{name}.log').open('w') as log:
   subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
  summary,bundle,_=completed(out,.003);pair[enabled]=bundle
  row={'repeat':repeat,'candidate':enabled,'solve_wall_s':summary['solve_wall_s'],'audit':summary['audit'],'trajectory_sha256':summary['trajectory_sha256']}
  rows.append(row);print(json.dumps(row),flush=True)
 require_equal(pair[False]['result'],pair[True]['result'])
 print('All accepted-stage fields/reports and final fields exact',repeat,flush=True)
medians={str(c):statistics.median(r['solve_wall_s'] for r in rows if r['candidate'] is c) for c in (False,True)}
result={'accepted':True,'scope':'Three alternating source-guarded CPU pairs on shared host; 1024 cells /16 wells /.003 days. Same FIM, time policy and search. Exact accepted-stage reports and fields. Not a full-month speed measurement.','rows':rows,'median_s':medians,'observed_ratio':medians['False']/medians['True']}
(work/'pairs.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
