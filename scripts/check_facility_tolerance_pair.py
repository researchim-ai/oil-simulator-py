"""Engineering accuracy gate for declared convergence/search policies.

Keeps the independent mass audit at its existing 0.01 kg + 1e-8 of production.
Changing that audit or any physical configuration is not permitted here.
"""
import argparse
import copy
import json
import math
from pathlib import Path

from check_facility_engineering_profile import validate_budgets
from check_facility_refinement_pair import completed
from check_long_facility_trajectory import compare as compare_wells
from compare_fim_facility_forecast import NEWTON_TOLERANCES
from compare_fim_facility_forecast import compare as compare_fields


def policy_config(config, time_refinement):
    c=copy.deepcopy(config);s=c['simulation']
    for key,default in NEWTON_TOLERANCES.items():
        value=s.pop(key,default)
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
            raise ValueError('Invalid nonlinear tolerance')
    if time_refinement:
        for key in ('max_time_step_days','initial_time_step_days'):
            value=s.pop(key,None)
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('Explicit positive time-step bounds required')
    control=s['coupled_facility']['control']
    if control.pop('reuse_capacity_history',False) is not False:
        raise ValueError('Historical polishing is outside this declared policy comparison')
    method=control.pop('search_method','safeguarded')
    if method not in ('facet_safeguarded','continuation_safeguarded'):
        raise ValueError('Undeclared search method in tolerance comparison')
    tol=control.pop('fraction_tolerance',1e-4)
    if type(tol) not in (int,float) or not math.isfinite(tol) or not 0<tol<1:
        raise ValueError('Invalid fraction tolerance')
    interval=control.pop('continuation_scan_interval',16)
    if type(interval) is not int or interval<1:
        raise ValueError('Invalid scan period')
    return c


def compare(reference,candidate,budgets,days,*,time_refinement=False):
    validate_budgets(budgets)
    sa,a,pa=completed(reference,days);sb,b,pb=completed(candidate,days)
    if policy_config(a['config'],time_refinement)!=policy_config(b['config'],time_refinement):
        raise ValueError('Only declared convergence/search policies may differ')
    if a['schedule']!=b['schedule'] or sa['device']!=sb['device']:
        raise ValueError('Identical schedule and device required')
    protocols=[json.loads((Path(p)/'protocol.json').read_text()) for p in (reference,candidate)]
    for key in ('cuda_visible_devices','cuda_memory_fraction','report_days'):
        if protocols[0].get(key)!=protocols[1].get(key):
            raise ValueError('Identical execution protocol required')
    for key in ('torch_version','torch_threads','device_name','gathering_backend','hydraulic_backend'):
        if sa['execution'][key]!=sb['execution'][key]:
            raise ValueError('Identical execution backend required')
    pvt=str(Path(a['config']['fluid']['pvt_path']).resolve())
    digests=[p['source_sha256'].get(pvt) for p in protocols]
    if digests[0] is None or digests[0]!=digests[1]:
        raise ValueError('Identical PVT contents required')
    fields=compare_fields(pa,pb,budgets['fields'],allow_numerical_policy=True)
    wells=compare_wells(pa,pb,allow_numerical_policy=True)
    wells['budgets']=budgets['wells']
    wells['accepted']=wells['mode_mismatches']==0 and all(math.isfinite(wells['errors'][k]) and wells['errors'][k]<=v for k,v in budgets['wells'].items())
    work=[]
    for bundle,summary in ((a,sa),(b,sb)):
        stages=bundle['result']['accepted_substeps']
        work.append({'fim':sum(s['facility_coupling']['reservoir_evaluations'] for s in stages),
                     'linear':sum(s['facility_coupling']['capacity_amg_work']['linear_attempts'] for s in stages),
                     'accepted_steps':len(stages),'rejected_steps':sum(t['time_step_statistics']['rejected_steps'] for t in bundle['result']['interval_timings']),
                     'solve_wall_s':summary['solve_wall_s']})
    return {'accepted':fields['accepted'] and wells['accepted'],'fields':fields,'wells':wells,'work':work,
            'time_refinement':time_refinement,'declared_budgets':budgets,
            'scope':'Final fields and wells at common report dates; unchanged strict independent mass audit. Shared-host timings, not an exact-solution error bound.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--budgets',type=Path,required=True);p.add_argument('--days',type=float,required=True)
    p.add_argument('--time-refinement',action='store_true');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=compare(args.reference,args.candidate,json.loads(args.budgets.read_text()),args.days,time_refinement=args.time_refinement)
    with args.output.open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
    print(json.dumps(r,indent=2));raise SystemExit(0 if r['accepted'] else 1)
