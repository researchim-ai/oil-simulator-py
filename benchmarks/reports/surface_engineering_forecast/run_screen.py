import json,os,subprocess,sys,time
from pathlib import Path
w=Path('.scratch/engineering_tolerances')
base=json.loads(Path('configs/fim_blackoil_facility_depletion_forecast.json').read_text())
cases={'outer':(1e-9,1e-12,.1,1e-5,1e-7),'balanced':(1e-6,1e-10,100.,1e-5,1e-7),'engineering':(1e-4,1e-8,100.,1e-4,1e-7),'fast':(1e-3,1e-7,1000.,1e-4,1e-6)}
(w/'planned_cases.json').write_text(json.dumps(cases,indent=2))
(w/'budgets.json').write_text(Path('benchmarks/reports/surface_forecast_profiles/budgets.json').read_text())
for name,(cnv,mb,bhp,rate,outer) in cases.items():
 c=json.loads(json.dumps(base));s=c['simulation']
 s.update(newton_tol_cnv=cnv,newton_tol_mb=mb,newton_tol_well_bhp=bhp,newton_tol_well_rate=rate)
 s['coupled_facility']['control']['fraction_tolerance']=outer
 p=w/f'{name}.json';p.write_text(json.dumps(c,indent=2))
 with (w/f'{name}.log').open('w') as log:
  r=subprocess.run([sys.executable,'scripts/run_fim_facility_forecast.py','--config',str(p),'--days','1','--report-days','1','--device','cpu','--output',str(w/name)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/pr_forecast_mpl',PYTHONPATH='src:scripts'),stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:
  print(name,'FAIL',json.loads((w/name/'failure.json').read_text()),flush=True)
 else:
  s=json.loads((w/name/'summary.json').read_text());print(name,s['solve_wall_s'],s['audit']['reservoir_evaluations'],s['audit']['mass_balance_error_kg'],flush=True)
