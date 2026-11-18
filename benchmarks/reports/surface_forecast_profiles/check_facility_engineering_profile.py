"""Time-refinement qualification against explicitly declared engineering budgets.

Reuses the strict source/PVT/configuration and conservation checks. This is
numerical qualification of a declared model, not experimental validation.
"""
import argparse
import json
import math
from pathlib import Path

from check_facility_refinement_pair import completed, qualify


def validate_budgets(budgets):
    expected = {
        'fields': {'final_pressure_pa', 'final_s_w', 'final_s_g', 'final_rs', 'withdrawal_relative'},
        'wells': {'bhp_abs_pa', 'component_rate_scaled', 'cumulative_mass_relative'},
    }
    for group, keys in expected.items():
        values = budgets.get(group)
        if not isinstance(values, dict) or set(values) != keys:
            raise ValueError(f'Explicit complete {group} budgets required')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in values.values()):
            raise ValueError('Budgets must be finite positive numbers')


def check(candidate, reference, budgets, *, days=30.):
    validate_budgets(budgets)
    summaries = [completed(p, days)[0] for p in (candidate, reference)]
    protocols = [json.loads((Path(p) / 'protocol.json').read_text()) for p in (candidate, reference)]
    for key in ('device', 'cuda_memory_fraction'):
        if summaries[0].get(key) != summaries[1].get(key):
            raise ValueError('Time refinement requires the same execution device and allocator limit')
    if protocols[0].get('cuda_visible_devices') != protocols[1].get('cuda_visible_devices'):
        raise ValueError('Time refinement requires the same selected GPU')
    for key in ('torch_version', 'torch_threads', 'device_name', 'gathering_backend', 'hydraulic_backend'):
        if summaries[0]['execution'].get(key) != summaries[1]['execution'].get(key):
            raise ValueError('Time refinement requires the same execution backend')
    result = qualify(candidate, reference, budgets['fields'], required_days=days)
    wells = result['well_trajectories']
    wells['budgets'] = dict(budgets['wells'])
    wells['accepted'] = wells['mode_mismatches'] == 0 and all(
        math.isfinite(wells['errors'][key]) and wells['errors'][key] <= value
        for key, value in budgets['wells'].items())
    result['accepted'] = result['final_fields']['accepted'] and wells['accepted']
    result['scope'] = __doc__
    result['declared_budgets'] = budgets
    result['accuracy_scope'] = ('Final reservoir fields and well histories at common report dates; '
                                'two discretizations, not a bound against the exact solution.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--budgets', type=Path, required=True)
    parser.add_argument('--days', type=float, default=30.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.candidate, args.reference, json.loads(args.budgets.read_text()), days=args.days)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
