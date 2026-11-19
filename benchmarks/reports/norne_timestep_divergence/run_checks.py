import subprocess,os,json
from pathlib import Path
p=Path('.scratch/norne_local_refinement')
results=[]
for name,grid,mode in [('current_step','current','step'),('historical_step','historical','step'),('current_ref','current','cnv_ref')]:
 cmd=['/home/user/miniconda3/bin/python','scripts/benchmark_norne_accuracy.py','--deck','.scratch/ot2/norne/NORNE_ATW2013.DATA','--grid',str(p/(grid+'_grid.npz')),'--out',str(p/name),'--mode',mode]
 with (p/(name+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 results.append({'name':name,'exit_code':r.returncode});(p/'batch.json').write_text(json.dumps(results,indent=2))
 print(name,r.returncode,flush=True)
