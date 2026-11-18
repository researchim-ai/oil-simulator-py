import os,sys,subprocess,json
from pathlib import Path
root=Path.cwd();out=root/'.scratch/pipe_root_reuse';records={}
for repeat in range(2):
 for candidate in ((False,True) if repeat==0 else (True,False)):
  source=root if candidate else root/'.scratch/pipe_port_reuse/reference';name=f'batch_{repeat}_{candidate}'
  with (out/(name+'.json')).open('w') as f:subprocess.run([sys.executable,str(out/'benchmark_batch.py')],env=dict(os.environ,PYTHONPATH=str(source/'src')),stdout=f,check=True)
  records[name]=json.loads((out/(name+'.json')).read_text())
first=next(iter(records.values()))['rows'];assert all(r['rows']==first for r in records.values())
(out/'batch_comparison.json').write_text(json.dumps({'exact_rows':True,'runs':{k:{f:v[f] for f in ('median','peak_extra_bytes','checked_integrals_per_solve')} for k,v in records.items()}},indent=2)+'\n')
