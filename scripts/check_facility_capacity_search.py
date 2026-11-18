"""Regression for capacity-search changes with fixed, predeclared error budgets.

Only the search method and auxiliary sparse-value budget may differ. This is
not independent physical validation or exact execution equivalence.
"""
import argparse
import copy
import json
from pathlib import Path

from check_facility_refinement_pair import completed
from check_long_facility_trajectory import compare as compare_wells
from compare_fim_facility_forecast import compare as compare_fields

FIELD_BUDGETS = {'final_pressure_pa': .1, 'final_s_w': 1e-8, 'final_s_g': 1e-8,
                 'final_rs': 1e-5, 'withdrawal_relative': 1e-7}
WELL_BUDGETS = {'bhp_abs_pa': .1, 'component_rate_scaled': 1e-7, 'cumulative_mass_relative': 1e-7}


def compare(reference, candidate, days):
    sa, a, pa = completed(reference, days)
    sb, b, pb = completed(candidate, days)
    configs = [copy.deepcopy(x['config']) for x in (a, b)]
    methods = []
    for c in configs:
        control = c['simulation']['coupled_facility']['control']
        methods.append(control.pop('search_method', 'safeguarded'))
        budget = control.pop('sensitivity_sparse_budget_bytes', 256*1024**2)
        if type(budget) is not int or budget <= 0:
            raise ValueError('Invalid sensitivity sparse budget')
    if methods != ['facet_safeguarded', 'sensitivity_safeguarded'] or configs[0] != configs[1]:
        raise ValueError('Only capacity-search method and auxiliary memory budget may differ')
    if a['schedule'] != b['schedule'] or sa['device'] != sb['device']:
        raise ValueError('Identical commands and execution device required')
    fields = compare_fields(pa, pb, FIELD_BUDGETS, allow_numerical_policy=True)
    wells = compare_wells(pa, pb, allow_numerical_policy=True)
    wells['budgets'] = WELL_BUDGETS
    wells['accepted'] = wells['mode_mismatches'] == 0 and all(
        wells['errors'][k] <= v for k, v in WELL_BUDGETS.items())
    work = [x['result']['temporal_work'] for x in (a, b)]
    return {'scope': __doc__, 'accepted': fields['accepted'] and wells['accepted'],
            'fewer_fim_solves': work[1]['reservoir_evaluations'] < work[0]['reservoir_evaluations'],
            'required_days': days, 'fields': fields, 'wells': wells,
            'reference_work': work[0], 'candidate_work': work[1],
            'reference_solve_wall_s': sa['solve_wall_s'], 'candidate_solve_wall_s': sb['solve_wall_s'],
            'observed_ratio': sa['solve_wall_s']/sb['solve_wall_s'],
            'timing_scope': 'Shared host; these are observations, not isolated hardware speedups.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--days', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.reference, args.candidate, args.days)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    return 0 if result['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
