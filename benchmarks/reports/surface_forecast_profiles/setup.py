import copy,json,shutil
from pathlib import Path
root=Path('/home/user/workspace/oil-simulator-py-impes');w=root/'.scratch/engineering_profile'
budgets={'fields':{'final_pressure_pa':1000.,'final_s_w':.0002,'final_s_g':.0002,'final_rs':.05,'withdrawal_relative':.002},'wells':{'bhp_abs_pa':1000.,'component_rate_scaled':.002,'cumulative_mass_relative':.002},'scope':'Predeclared numerical refinement targets for these synthetic models; not universal industrial tolerances or experimental validation.'}
(w/'budgets.json').write_text(json.dumps(budgets,indent=2)+'\n')
schedule=[{'time_days':10.,'controls':{'gas_valve_max_opening':.8}},{'time_days':20.,'controls':{}}]
(w/'schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
for label,path,steps in [('depletion','configs/fim_blackoil_facility_depletion_gpu_certified.json',[.5,.25,.125]),('large','benchmarks/reports/surface_gpu_year/field256_wells64_conservative.json',[5.,2.5,1.25])]:
 c=json.loads((root/path).read_text());c['fluid']['pvt_path']=str(root/c['fluid']['pvt_path'])
 s=c['simulation'];f=s['coupled_facility'];f.pop('temporal_control',None);f.pop('time_control',None)
 s['physical_surface_network']['network_backend']='cpu_reference'
 f['control'].update(commit_method='state_restore',search_method='facet_safeguarded',reuse_inactive_caps=True,reuse_initial_cell_state=True,reuse_capacity_amg=True,inactive_cap_certificate_version=2)
 for i,dt in enumerate(steps):
  a=copy.deepcopy(c);a['simulation']['max_time_step_days']=dt
  a['simulation']['initial_time_step_days']=.00005 if label=='depletion' else min(1.,dt)
  (w/f'{label}_{i}.json').write_text(json.dumps(a,indent=2)+'\n')
source=w/'executed_source'
for folder in ['src','scripts']:shutil.copytree(root/folder,source/folder,ignore=shutil.ignore_patterns('__pycache__'))
print('Frozen predeclared budgets, configurations and source')
