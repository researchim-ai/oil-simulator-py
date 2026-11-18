import json,os,subprocess,sys
from pathlib import Path
w=Path('.scratch/capacity_history')
for enabled in (False,True):
 c=json.loads(Path('configs/fim_blackoil_facility_depletion_forecast.json').read_text())
 c['simulation']['coupled_facility']['control']['reuse_capacity_history']=enabled
 p=w/f'config_{enabled}.json';p.write_text(json.dumps(c,indent=2))
 with (w/f'short_{enabled}.log').open('w') as log:
  subprocess.run([sys.executable,'scripts/run_fim_facility_forecast.py','--config',str(p),'--days','1','--report-days','1','--device','cpu','--output',str(w/f'short_{enabled}')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/pr_forecast_mpl',PYTHONPATH='src:scripts'),stdout=log,stderr=subprocess.STDOUT,check=True)
 s=json.loads((w/f'short_{enabled}/summary.json').read_text()); print(enabled,s['solve_wall_s'],s['audit']['reservoir_evaluations'],flush=True)
