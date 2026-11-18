import os,sys,subprocess,json
from pathlib import Path
from check_facility_refinement_pair import completed
from benchmark_initial_cell_cache import require_equal
root=Path.cwd();out=root/'.scratch/pipe_root_reuse/strong_reverse';out.mkdir(exist_ok=False)
results={}
for name,source in [('candidate',root),('reference',root/'.scratch/pipe_port_reuse/reference')]:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=f'{source}/src:{source}/scripts')
 cmd=[sys.executable,str(source/'scripts/run_fim_facility_forecast.py'),'--config',str(root/'.scratch/pipe_root_reuse/strong/config.json'),'--days','2','--report-days','2','--device','cpu','--output',str(out/name)]
 with (out/(name+'.log')).open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 results[name]=completed(out/name,2)
require_equal(results['reference'][1]['result'],results['candidate'][1]['result'])
(out/'result.json').write_text(json.dumps({'exact_all_accepted_stages':True,'seconds':{k:v[0]['solve_wall_s'] for k,v in results.items()}},indent=2)+'\n')
