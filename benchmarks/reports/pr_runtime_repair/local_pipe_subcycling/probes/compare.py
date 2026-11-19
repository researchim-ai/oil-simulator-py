import json,math
from pathlib import Path
from surface.pr_surface_control import PRControlledSurface,PRControlledSurfaceState
root=Path('/tmp/pr-local-subcycles')
files=sorted(root.glob('reference-chunk-*.json'))
last=json.loads(files[-1].read_text())
b=PRControlledSurfaceState.from_dict(last['report']['intervals'][-1]['state'])
assert b.time_s==1e-4
r=json.loads((root/'result.json').read_text())
a=PRControlledSurfaceState.from_dict(r['intervals'][-1]['state'])
m=PRControlledSurface.from_dict(last['report']['model'])
t=m.plant.transport
nodes={}
for (name,x),(_,y) in zip(a.plant.transport.nodes,b.plant.transport.nodes):
 nodes[name]={'pressure_pa':abs(x.pressure_pa-y.pressure_pa),'temperature_k':abs(x.temperature_k-y.temperature_k),'velocity_m_s':float(abs(t._state_velocity(x)-t._state_velocity(y))), 'energy_j':abs(x.total_energy_j-y.total_energy_j),'component_moles_max':max(abs(v-w) for v,w in zip(x.component_moles,y.component_moles))}
out={'time_s':a.time_s,'original_error_norm':m._error(a,b),'nodes':nodes,
     'reference_wall_seconds':sum(json.loads(p.read_text())['seconds'] for p in files),
     'reference_reporting_intervals':len(files),'reference_attempts':sum(json.loads(p.read_text())['report']['intervals'][0]['attempts'] for p in files),
     'reference_timing_caveat':'Additional reporting boundaries and model reconstruction; not a clean speedup measurement.'}
(root/'comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
