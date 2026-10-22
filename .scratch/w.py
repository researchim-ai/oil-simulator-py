import sys,time,pathlib;sys.path.insert(0,'src');sys.path.insert(0,'tests')
from test_wet_gas import _wet_deck
from simulator.eclipse_run import run_deck
p=pathlib.Path('.scratch/wet');p.mkdir(exist_ok=True)
ov=eval(sys.argv[1]) if len(sys.argv)>1 else None
t=time.time();r,_=run_deck(_wet_deck(p,1e-4),max_days=1500,sim_overrides=ov,quiet=True)
print("ok",time.time()-t,r['FGPR'][-1],file=sys.__stdout__)
