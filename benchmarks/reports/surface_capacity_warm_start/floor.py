import os,shutil,subprocess,sys
from pathlib import Path
r=Path('/home/user/workspace/oil-simulator-py-impes');w=r/'.scratch/capacity_warm';s=w/'warm_floor'
for f in ['src','scripts']:shutil.copytree(w/'candidate'/f,s/f,ignore=shutil.ignore_patterns('__pycache__'))
p=s/'src/simulator/blackoil_fim.py';t=p.read_text()
t=t.replace("        if seed is not None and all(abs(a-b)","        warm_used = False\n        if seed is not None and all(abs(a-b)")
t=t.replace("            p, sw, x, sat, ug, p_bh = [v.clone() for v in seed[1]]","            warm_used = True\n            p, sw, x, sat, ug, p_bh = [v.clone() for v in seed[1]]")
t=t.replace('            if ok and k >= self.min_newton:', '            if ok and k >= max(self.min_newton, int(warm_used)):')
p.write_text(t)
env=dict(os.environ,PYTHONPATH=f'{s/"src"}:{s/"scripts"}')
cmd=[sys.executable,str(s/'scripts/run_fim_facility_forecast.py'),'--config',str(w/'config.json'),'--days','.00005','--report-days','.00005','--device','cuda','--cuda-memory-fraction','.08','--output',str(w/'startup_warm_floor')]
with (w/'warm_floor.log').open('w') as log:subprocess.run(cmd,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
print('DONE',flush=True)
