"""Exact regression for old-state property reuse in completed capacity forecasts.

This validates the optimization, not experimental physics or a new same-source
time-refinement pair. Timings are observations on a shared host.
"""

import argparse
import copy
import json
from pathlib import Path

from check_facility_refinement_pair import completed
from check_fim_facility_forecast import audit
from check_long_facility_trajectory import compare as compare_wells
from compare_fim_facility_forecast import compare as compare_fields


def compare(baseline, optimized, days):
    sa, a, pa = completed(baseline, days)
    sb, b, pb = completed(optimized, days)
    configs = [copy.deepcopy(x['config']) for x in (a, b)]
    flags = []
    for config in configs:
        control = config['simulation']['coupled_facility']['control']
        flags.append(control.pop('reuse_initial_cell_state', False))
        control.setdefault('initial_cell_state_cache_bytes', 64 * 1024**2)
    if any(type(f) is not bool for f in flags) or flags != [False, True] or configs[0] != configs[1]:
        raise ValueError('Only initial cell state reuse may differ; require disabled/enabled pair')
    if a['schedule'] != b['schedule'] or sa['device'] != sb['device']:
        raise ValueError('Identical commands and execution device required')
    fields = compare_fields(pa, pb, dict.fromkeys(
        ('final_pressure_pa', 'final_s_w', 'final_s_g', 'final_rs', 'withdrawal_relative'), 0.),
        allow_numerical_policy=True)
    wells = compare_wells(pa, pb, allow_numerical_policy=True)
    aa, bb = [x['result']['accepted_substeps'] for x in (a, b)]
    exact = len(aa) == len(bb)
    if exact:
        for x, y in zip(aa, bb, strict=True):
            exact &= all(x[key] == y[key] for key in
                         ('time_days', 'dt_s', 'surface', 'withdrawn_component_mass_kg', 'newton_iterations'))
            exact &= all(x['facility_coupling'][key] == y['facility_coupling'][key] for key in
                         ('trace', 'material_account', 'facility', 'fraction'))
    work = []
    for bundle in (a, b):
        checked = audit(bundle['config'], bundle['result'], expected_schedule=bundle['schedule'],
                        expected_days=days)
        actual = checked['reservoir_evaluations']
        work.append({'accepted_reservoir_evaluations': actual,
                     'accepted_initial_cell_evaluations': checked['accepted_initial_cell_evaluations'],
                     'accepted_initial_cell_reuses': checked['accepted_initial_cell_reuses']})
    fewer = work[1]['accepted_initial_cell_evaluations'] < work[0]['accepted_initial_cell_evaluations']
    return {'scope': __doc__, 'accepted': fields['accepted'] and wells['accepted'] and exact and fewer,
            'required_days': days, 'exact_all_accepted_stages': exact, 'final_fields': fields,
            'well_trajectories': wells, 'baseline_work': work[0], 'optimized_work': work[1],
            'baseline_solve_wall_s': sa['solve_wall_s'], 'optimized_solve_wall_s': sb['solve_wall_s'],
            'observed_wall_ratio': sa['solve_wall_s'] / sb['solve_wall_s'],
            'baseline_trajectory_sha256': sa['trajectory_sha256'],
            'optimized_trajectory_sha256': sb['trajectory_sha256'],
            'same_source': sa['source_sha256'] == sb['source_sha256']}


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
