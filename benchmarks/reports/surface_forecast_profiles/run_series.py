import os,subprocess,sys
from pathlib import Path
root=Path('/home/user/workspace/oil-simulator-py-impes');w=root/'.scratch/engineering_profile';source=w/'executed_source'
label,device=sys.argv[1:];env=dict(os.environ,PYTHONPATH=f'{source/"src"}:{source/"scripts"}')
for i in range(3):
 out=w/f'{label}_run{i}';cmd=[sys.executable,str(source/'scripts/run_fim_facility_forecast.py'),'--config',str(w/f'{label}_{i}.json'),'--schedule',str(w/'schedule.json'),'--days','30','--report-days','1' if label=='depletion' else '5','--device',device,'--output',str(out)]
 if device=='cuda':cmd+=['--cuda-memory-fraction','.12']
 with (w/f'{label}_run{i}.log').open('w') as log:
  subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
 print('COMPLETE',label,i,flush=True)
