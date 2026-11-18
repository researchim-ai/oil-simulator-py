"""Qualify optional historical proposals with fixed tight execution budgets."""
import argparse
import copy
import json
import math
from pathlib import Path

from check_facility_refinement_pair import completed
from check_long_facility_trajectory import compare as compare_wells
from compare_fim_facility_forecast import compare as compare_fields

FIELDS = {'final_pressure_pa': .1, 'final_s_w': 1e-8, 'final_s_g': 1e-8,
          'final_rs': 1e-5, 'withdrawal_relative': 1e-5}
WELLS = {'bhp_abs_pa': .1, 'component_rate_scaled': 1e-5, 'cumulative_mass_relative': 1e-5}


def compare(reference, candidate, days):
    sa,a,pa=completed(reference,days)
    sb,b,pb=completed(candidate,days)
    configs=[copy.deepcopy(x['config']) for x in (a,b)]
    flags=[c['simulation']['coupled_facility']['control'].pop('reuse_capacity_history',False) for c in configs]
    if flags != [False,True] or any(type(v) is not bool for v in flags) or configs[0]!=configs[1]:
        raise ValueError('Only enabling capacity history may differ')
    if a['schedule']!=b['schedule'] or sa['device']!=sb['device']:
        raise ValueError('Identical schedule and device required')
    protocols=[json.loads((Path(p)/'protocol.json').read_text()) for p in (reference,candidate)]
    for key in ('cuda_visible_devices','cuda_memory_fraction','report_days'):
        if protocols[0].get(key)!=protocols[1].get(key):
            raise ValueError('Identical GPU selection, allocator limit and report dates required')
    pvt=str(Path(a['config']['fluid']['pvt_path']).resolve())
    pvt_hashes=[p['source_sha256'].get(pvt) for p in protocols]
    if pvt_hashes[0] is None or pvt_hashes[0]!=pvt_hashes[1]:
        raise ValueError('Identical recorded PVT contents required')
    for key in ('torch_version','torch_threads','device_name','gathering_backend','hydraulic_backend'):
        if sa['execution'][key]!=sb['execution'][key]:
            raise ValueError('Identical execution backend required')
    fields=compare_fields(pa,pb,FIELDS,allow_numerical_policy=True)
    wells=compare_wells(pa,pb,allow_numerical_policy=True)
    wells['budgets']=WELLS
    wells['accepted']=wells['mode_mismatches']==0 and all(wells['errors'][k]<=v for k,v in WELLS.items())
    work=[sum(s['facility_coupling']['reservoir_evaluations'] for s in x['result']['accepted_substeps']) for x in (a,b)]
    rejected=[sum(r['time_step_statistics']['rejected_steps'] for r in x['result']['interval_timings']) for x in (a,b)]
    grids=[[(s['time_days'],s['dt_s']) for s in x['result']['accepted_substeps']] for x in (a,b)]
    if grids[0]!=grids[1]:
        raise ValueError('Execution comparison requires the same accepted time grid')
    stage_errors=dict.fromkeys(WELLS,0.)
    amounts=[[0.]*3,[0.]*3]
    modes=0
    for stages in zip(a['result']['accepted_substeps'],b['result']['accepted_substeps'],strict=True):
        for i,s in enumerate(stages):
            amounts[i]=[math.fsum((v,q)) for v,q in zip(amounts[i],s['withdrawn_component_mass_kg'],strict=True)]
        wa,wb=[s['surface']['wells'] for s in stages]
        if wa.keys()!=wb.keys():
            raise ValueError('Different accepted-stage well registry')
        for name,x in wa.items():
            y=wb[name];modes+=x['mode']!=y['mode']
            stage_errors['bhp_abs_pa']=max(stage_errors['bhp_abs_pa'],abs(x['bhp_pa']-y['bhp_pa']))
            stage_errors['component_rate_scaled']=max(stage_errors['component_rate_scaled'],
                *(abs(v-q)/max(abs(v),1e-8) for v,q in zip(x['component_mass_rates_kg_s'],y['component_mass_rates_kg_s'],strict=True)))
        stage_errors['cumulative_mass_relative']=max(stage_errors['cumulative_mass_relative'],
            *(abs(v-q)/max(abs(v),1.) for v,q in zip(*amounts,strict=True)))
    stage_ok=modes==0 and all(math.isfinite(v) and v<=WELLS[k] for k,v in stage_errors.items())
    return {'accepted':fields['accepted'] and wells['accepted'] and stage_ok, 'fields':fields, 'wells':wells,
            'all_accepted_stage_wells':{'accepted':stage_ok,'errors':stage_errors,'budgets':WELLS,'mode_mismatches':modes},
            'accepted_fim_evaluations':work,'rejected_steps':rejected,
            'fewer_fim_solves':work[1]<work[0] and rejected[1]<=rejected[0],
            'solve_wall_s':[sa['solve_wall_s'],sb['solve_wall_s']],
            'scope':'Shared-host execution comparison, not independent physical validation or isolated timing.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--days',type=float,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=compare(args.reference,args.candidate,args.days)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['accepted'] else 1)
