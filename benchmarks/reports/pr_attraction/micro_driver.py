"""Full EOS batches, increasing synthetic component registries; not a plant benchmark."""
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
import torch
from surface.equilibrium_properties import EquilibriumFluid

root=Path(__file__).resolve().parents[3]
data=json.loads((root/'tests/data/surface/pr_restart_check_valve.json').read_text())
seed=EquilibriumFluid.from_dict(data['model']['transport']['fluid'])
rows=[]
for nc in (3,6,12):
    components=[replace(seed.components[i%3],name=f'c{i}') for i in range(nc)]
    fluid=EquilibriumFluid(components,[[0.]*nc for _ in range(nc)])
    count=256
    p=fluid.tensor(torch.linspace(1e5,4e6,count))
    t=fluid.tensor(torch.linspace(320.,380.,count))
    gen=torch.Generator().manual_seed(789)
    x=torch.rand((count,nc),generator=gen,dtype=torch.float64);x/=x.sum(-1,keepdim=True)
    values={};times={False:[],True:[]}
    for enabled in (False,True):
        fluid._factored_attraction=enabled
        values[enabled]=fluid.tensor_homogeneous(p,t,x)
    assert values[True]['valid'].all()
    for key in values[False]:
        torch.testing.assert_close(values[True][key],values[False][key],rtol=2e-10,atol=2e-8,equal_nan=True)
    for repeat in range(6):
        for enabled in ((False,True) if repeat%2==0 else (True,False)):
            fluid._factored_attraction=enabled
            start=time.perf_counter()
            for _ in range(10): fluid.tensor_homogeneous(p,t,x)
            times[enabled].append((time.perf_counter()-start)/10)
    med={str(k):statistics.median(v) for k,v in times.items()}
    row={'components':nc,'points':count,'median_call_s':med,'speedup':med['False']/med['True'],'samples_s':{str(k):v for k,v in times.items()},'properties_match':True}
    rows.append(row); print(json.dumps(row),flush=True)
out={'accepted':True,'scope':'full EOS value batches, CPU one thread, 6 alternating paired blocks of 10 calls; synthetic registries, no AD or plant integration timing','rows':rows}
Path(__file__).with_name('eos_batches.json').write_text(json.dumps(out,indent=2)+'\n')
