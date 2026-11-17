"""Compare completed forecasts differing only in inactive-rate-cap reuse.

This is an optimization regression, not independent physical validation or a
new same-source time-refinement qualification. Timings may share a busy host.
"""

import argparse
import copy
import json
from pathlib import Path

from check_facility_refinement_pair import completed
from check_long_facility_trajectory import compare as compare_wells
from compare_fim_facility_forecast import compare as compare_fields

FIELD_BUDGETS = {
    'final_pressure_pa': .1,
    'final_s_w': 1e-8,
    'final_s_g': 1e-8,
    'final_rs': 1e-5,
    'withdrawal_relative': 1e-7,
}
WELL_BUDGETS = {
    'bhp_abs_pa': .1,
    'component_rate_scaled': 1e-7,
    'cumulative_mass_relative': 1e-7,
}


def compare(baseline, optimized, days):
    sa, a, pa = completed(baseline, days)
    sb, b, pb = completed(optimized, days)
    configs = [copy.deepcopy(x['config']) for x in (a, b)]
    flags = [c['simulation']['coupled_facility']['control'].pop('reuse_inactive_caps', False)
             for c in configs]
    if any(type(f) is not bool for f in flags) or flags != [False, True] or configs[0] != configs[1]:
        raise ValueError('Only inactive-cap reuse may differ; require disabled/enabled pair')
    if a['schedule'] != b['schedule']:
        raise ValueError('Identical operating commands required')
    if sa['device'] != sb['device']:
        raise ValueError('Compare the same execution device')
    fields = compare_fields(pa, pb, FIELD_BUDGETS, allow_numerical_policy=True)
    wells = compare_wells(pa, pb, allow_numerical_policy=True)
    wells['budgets'] = WELL_BUDGETS
    wells['accepted'] = wells['mode_mismatches'] == 0 and all(
        wells['errors'][k] <= limit for k, limit in WELL_BUDGETS.items())
    work = []
    for bundle in (a, b):
        w = bundle['result']['temporal_work']
        work.append({
            'capacity_evaluations': w['capacity_evaluations'],
            'reservoir_evaluations': w.get('reservoir_evaluations', w['capacity_evaluations']),
            'coupled_solves': w['coupled_solves'],
            'macro_attempts': w['macro_attempts'],
            'rejected_macro_steps': w['rejected_macro_steps'],
            'accepted_stages': len(bundle['result']['accepted_substeps']),
        })
    fewer = work[1]['reservoir_evaluations'] < work[0]['reservoir_evaluations']
    clocks = [[s['time_days'] for s in x['result']['accepted_substeps']] for x in (a, b)]
    return {
        'scope': __doc__, 'accepted': fields['accepted'] and wells['accepted'] and fewer,
        'required_days': days, 'final_fields': fields, 'well_trajectories': wells,
        'baseline_work': work[0], 'optimized_work': work[1],
        'identical_accepted_clocks': clocks[0] == clocks[1],
        'reservoir_solve_reduction_fraction': 1 - work[1]['reservoir_evaluations'] / work[0]['reservoir_evaluations'],
        'baseline_solve_wall_s': sa['solve_wall_s'], 'optimized_solve_wall_s': sb['solve_wall_s'],
        'observed_wall_ratio': sa['solve_wall_s'] / sb['solve_wall_s'],
        'timing_scope': 'Observed CPU/device timings; host isolation not established.',
        'baseline_trajectory_sha256': sa['trajectory_sha256'],
        'optimized_trajectory_sha256': sb['trajectory_sha256'],
        'same_source': sa['source_sha256'] == sb['source_sha256'],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline', type=Path)
    p.add_argument('optimized', type=Path)
    p.add_argument('--days', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.baseline, args.optimized, args.days)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    return 0 if result['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
