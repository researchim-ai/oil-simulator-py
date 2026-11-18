import copy,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path('/home/user/workspace/oil-simulator-py-impes');w=root/'.scratch/capacity_iqi';source=w/'executed_source'
for folder in ('src','scripts'):shutil.copytree(root/folder,source/folder,ignore=shutil.ignore_patterns('__pycache__'))
sys.path[:0]=[str(source/'src'),str(source/'scripts')]
from check_capacity_certificate_policy import compare
base=json.loads((root/'configs/fim_blackoil_facility_depletion_gpu_hybrid_amg.json').read_text())
base['fluid']['pvt_path']=str(root/base['fluid']['pvt_path'])
for version in (1,2):
 c=copy.deepcopy(base);c['simulation']['coupled_facility']['control']['inactive_cap_certificate_version']=version
 (w/f'config{version}.json').write_text(json.dumps(c,indent=2)+'\n')
results=[]
for name,days,repeats in [('startup',.00005,2),('evolution',.005,1)]:
 for repeat in range(repeats):
  for v in ((1,2) if repeat%2==0 else (2,1)):
   out=w/f'{name}_{repeat}_v{v}'
   env=dict(os.environ,PYTHONPATH=f'{source/"src"}:{source/"scripts"}')
   cmd=[sys.executable,str(source/'scripts/run_fim_facility_forecast.py'),'--config',str(w/f'config{v}.json'),'--days',str(days),'--report-days',str(days),'--device','cuda','--cuda-memory-fraction','.08','--output',str(out)]
   with (w/f'{name}_{repeat}_v{v}.log').open('w') as log:subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
   print('DONE',name,repeat,v,flush=True)
  result=compare(w/f'{name}_{repeat}_v1',w/f'{name}_{repeat}_v2',days)
  assert result['accepted']
  results.append({'window':name,'repeat':repeat,**result})
  (w/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
  print('COMPARE',name,{k:result[k] for k in ['fewer_fim_solves','reference_solve_wall_s','candidate_solve_wall_s']},result['reference_work']['reservoir_evaluations'],result['candidate_work']['reservoir_evaluations'],flush=True)
