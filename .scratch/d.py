import sys,shutil,pathlib;sys.path.insert(0,'src')
import numpy as np
from simulator.eclipse_run import run_deck
S='tests/data/spe1/SPE1CASE1.DATA'
t=open(S).read().replace("\nGRID","\nENDSCALE\n/\nGRID",1)
pathlib.Path('.scratch/e.DATA').write_text(t)
a,_=run_deck(S,max_days=200);b,_=run_deck('.scratch/e.DATA',max_days=200)
print(a['FGPR'][:8]);print(b['FGPR'][:8])
