import os,subprocess,sys
from pathlib import Path
root=Path.cwd();out=root/'.scratch/surface_assembly_reuse'
for name,source in [('reference',out/'reference'),('candidate',root)]:
 env=dict(os.environ,PYTHONPATH=f'{source}/src:{source}/scripts')
 command=[sys.executable,'-m','cProfile','-o',str(out/(name+'.prof')),str(source/'scripts/run_fim_facility_forecast.py'),'--config',str(root/'.scratch/norne_cpr_validation/profile_config.json'),'--days','5','--report-days','5','--device','cuda','--cuda-memory-fraction','.12','--output',str(out/('profile_'+name))]
 with (out/('profile_'+name+'.log')).open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
