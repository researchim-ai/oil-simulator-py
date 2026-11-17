"""Compare only initial pressure proposal on the same solved physical cases."""
import importlib.util,json,statistics,sys,time
from pathlib import Path
import torch
sys.path.insert(0,'tests')
from test_surface_blackoil_properties import model
from surface.blackoil_pipe import BlackOilPipe
from surface.blackoil_pipe_batch import PipeBatch
spec=importlib.util.spec_from_file_location('surface._pipe_batch_before_seed','.scratch/blackoil_pipe_batch_before_seed.py')
legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
cases=[('water',BlackOilPipe('water',80,.03),2.4e6,[0,.001,0]),('phase',BlackOilPipe('phase',300,.08,30,1e-5),2.72e6,[.85,2,.06]),('gas',BlackOilPipe('gas',100,.15,5,2e-5,2),1.1e6,[.85,1,.144])]
rows=[]
for name,pipe,boundary,rates in cases:
    m=model();times={};solutions={}
    for label,klass in [('midpoint',legacy.PipeBatch),('known_port',PipeBatch)]:
        kernel=klass([pipe],m);kernel.solve([boundary],[rates]);samples=[]
        for _ in range(7):
            t=time.perf_counter();row=kernel.solve([boundary],[rates])[0];samples.append(time.perf_counter()-t)
        times[label]=statistics.median(samples);solutions[label]=row
    a,b=solutions.values()
    delta=max(abs(a['pressure_derivatives'][k]-b['pressure_derivatives'][k]) for k in a['pressure_derivatives'])
    rows.append({'case':name,'median_s':times,'pressure_evaluations':{k:v['pressure_evaluations'] for k,v in solutions.items()},'inlet_delta_pa':abs(a['inlet_pressure_pa']-b['inlet_pressure_pa']),'maximum_derivative_abs_delta':delta})
print(json.dumps(rows,indent=2));Path('.scratch/pipe_seed_micro.json').write_text(json.dumps({'scope':__doc__+' CPU float64, seven repetitions, shared host; no universal speed claim.','rows':rows},indent=2)+'\n')
