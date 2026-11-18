import json,os,shutil,subprocess,sys
from pathlib import Path
r=Path('/home/user/workspace/oil-simulator-py-impes');w=r/'.scratch/capacity_warm';s=w/'interior'
for f in ['src','scripts']:shutil.copytree(w/'reference'/f,s/f,ignore=shutil.ignore_patterns('__pycache__'))
p=s/'src/surface/capacity_bracket.py';t=p.read_text();needle='        candidate = self.low + ratio * (self.high - self.low)'
assert t.count(needle)==1
t=t.replace(needle,needle+'\n        candidate += self.tolerance / 4')
p.write_text(t)
for label,source,days in [('interior',s,.00005),('reference_evolution',w/'reference',.001),('interior_evolution',s,.001)]:
 out=w/f'startup_{label}';env=dict(os.environ,PYTHONPATH=f'{source/"src"}:{source/"scripts"}')
 cmd=[sys.executable,str(source/'scripts/run_fim_facility_forecast.py'),'--config',str(w/'config.json'),'--days',str(days),'--report-days',str(days),'--device','cuda','--cuda-memory-fraction','.08','--output',str(out)]
 with (w/f'{label}.log').open('w') as log:subprocess.run(cmd,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
 print('DONE',label,flush=True)
