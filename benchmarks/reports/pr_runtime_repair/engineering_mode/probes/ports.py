import json,time,gzip
from pathlib import Path
from surface.pr_subcycled_surface import PRSubcycledSurface
from surface.pr_surface_control import PRControlledSurface,PRControlledSurfaceState
from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork
from run_pr_equilibrium_transport import run
from check_pr_equilibrium_transport import audit
root=Path('/tmp/pr-engineering-mode')
base=json.loads(Path('configs/surface_pr_integrated_facility.json').read_text());m=PRSubcycledSurface(PRControlledSurface.from_dict(base['model']),['gas_recycle_0'])
s=base['initial']['state']['plant']['transport'];s['nodes']={k:v for k,v in s['nodes'].items() if k in m.fast.names};s.pop('rotors',None)
for port in ['stagnation_single_phase']:
 spec=m.fast.to_dict();spec['reservoir_ports']=port
 local=PREquilibriumTransportNetwork.from_dict(spec);s['model_id']=local.model_id
 c=json.loads(json.dumps({'model':spec,'initial':{'state':s},'schedule':[{'duration_s':.001}]}))
 t=time.perf_counter();r=run(c);seconds=time.perf_counter()-t;r=json.loads(json.dumps(r))
 with gzip.open(root/(port+'.json.gz'),'wt') as f:json.dump(r,f)
 checked=audit(c,r)
 final=r['intervals'][-1]['state']['nodes']['gas_recycle_0'];mass=sum(v*x.molar_mass_kg_mol for v,x in zip(final['component_moles'],local.fluid.components))
 summary={'seconds':seconds,'audit':checked,'final_velocity_m_s':final['momentum_kg_m_s']/mass}
 (root/(port+'-summary.json')).write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
