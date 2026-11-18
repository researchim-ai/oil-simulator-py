import json,os,shutil,subprocess,sys
from pathlib import Path
root=Path('/home/user/workspace/oil-simulator-py-impes');w=root/'.scratch/capacity_warm'
for name in ['reference','candidate']:
 s=w/name
 for folder in ['src','scripts']:shutil.copytree(root/folder,s/folder,ignore=shutil.ignore_patterns('__pycache__'))
 if name=='candidate':
  p=s/'src/simulator/blackoil_fim.py';text=p.read_text()
  needle='        self._p_bh_cur = p_bh\n        dbg.begin_step(dt, current_time, p_old, sw_old)'
  replacement='''        scope = getattr(self, '_capacity_amg_scope', None)
        caps = tuple(w.control_value for w in self.wells.wells)
        seed = getattr(scope, '_trial_guess', None)
        if seed is not None and all(abs(a-b) <= .1*max(abs(a),abs(b)) for a,b in zip(caps,seed[0])):
            p, sw, x, sat, ug, p_bh = [v.clone() for v in seed[1]]
            state = self.cell_state(p, sw, x, sat, ug)
            scope._warm_hits = getattr(scope, '_warm_hits', 0) + 1
        self._p_bh_cur = p_bh
        dbg.begin_step(dt, current_time, p_old, sw_old)'''
  assert text.count(needle)==1;text=text.replace(needle,replacement)
  needle='                log("  Converged!")\n                return True'
  replacement='''                if scope is not None:
                    scope._trial_guess = (caps, tuple(v.detach().clone() for v in (p,sw,x,sat,ug,p_bh)))
                log("  Converged!")
                return True'''
  assert text.count(needle)==1;text=text.replace(needle,replacement);p.write_text(text)
c=json.loads((root/'configs/fim_blackoil_facility_depletion_gpu_certified.json').read_text());c['fluid']['pvt_path']=str(root/c['fluid']['pvt_path'])
(w/'config.json').write_text(json.dumps(c,indent=2)+'\n')
for name in ['reference','candidate']:
 s=w/name;out=w/f'startup_{name}';env=dict(os.environ,PYTHONPATH=f'{s/"src"}:{s/"scripts"}')
 cmd=[sys.executable,str(s/'scripts/run_fim_facility_forecast.py'),'--config',str(w/'config.json'),'--days','.00005','--report-days','.00005','--device','cuda','--cuda-memory-fraction','.08','--output',str(out)]
 with (w/f'{name}.log').open('w') as f:subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
 print('DONE',name,flush=True)
