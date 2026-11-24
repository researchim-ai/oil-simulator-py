"""Compare complete saved Norne outputs; historical timing is not an isolated A/B."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CURRENT = ROOT / '.scratch/norne_release_20261124_gpu/run'
HISTORIES = {
    'preceding_same_policy': ROOT / '.scratch/norne_full_safeguard/run',
    'full_newton': ROOT / '.scratch/norne_full_newton/run',
    'older_baseline': ROOT / '.scratch/norne_cpr_validation/baseline',
}


def arrays(first, second, clock):
    with np.load(first) as a, np.load(second) as b:
        result = {'keys_equal': set(a.files) == set(b.files),
                  'clocks_equal': np.array_equal(a[clock], b[clock]), 'vectors': {}}
        for key in sorted(set(a.files) & set(b.files)):
            x, y = a[key], b[key]
            if x.shape != y.shape or (key not in (clock, 'pv') and not result['clocks_equal']):
                result['vectors'][key] = {'comparable': False}
                continue
            valid = np.isfinite(x) & np.isfinite(y)
            delta = abs(x[valid].astype(float) - y[valid].astype(float))
            result['vectors'][key] = {
                'equal': bool(np.array_equal(x, y, equal_nan=True)),
                'max_absolute_difference': float(delta.max()) if delta.size else None,
                'finite_mask_mismatches': int(np.count_nonzero(np.isfinite(x) != np.isfinite(y))),
            }
        result['all_arrays_equal'] = result['keys_equal'] and all(
            row.get('equal', False) for row in result['vectors'].values())
        return result


def main():
    current_meta = json.loads((CURRENT / 'meta.json').read_text())
    validation = json.loads((CURRENT / 'validation.json').read_text())
    assert validation['status'] == 'passed_recorded_audit' and current_meta['days'] == 3312
    manifest = json.loads((CURRENT / 'manifest.json').read_text())
    reports = {}
    for name, folder in HISTORIES.items():
        meta = json.loads((folder / 'meta.json').read_text())
        previous = json.loads((folder / 'manifest.json').read_text())
        assert meta['status'] == 'done' and meta['days'] == 3312
        with np.load(CURRENT / 'substeps.npz') as a, np.load(folder / 'substeps.npz') as b:
            step_grid_equal = np.array_equal(a['data'][:, :4], b['data'][:, :4])
        reports[name] = {
            'path': str(folder.relative_to(ROOT)),
            'inputs_equal': manifest['input_files'] == previous['input_files'],
            'references_equal': manifest['reference_files'] == previous['reference_files'],
            'changed_sources': [k for k in sorted(set(manifest['source_files']) | set(previous['source_files']))
                                if manifest['source_files'].get(k) != previous['source_files'].get(k)],
            'historical_wall_s': meta['wall_s'], 'current_wall_s': current_meta['wall_s'],
            'observed_wall_ratio': meta['wall_s'] / current_meta['wall_s'],
            'observed_percent_less_time': 100 * (1 - current_meta['wall_s'] / meta['wall_s']),
            'physical_step_grid_and_accepted_newton_equal': bool(step_grid_equal),
            'summary': arrays(CURRENT / 'summary.npz', folder / 'summary.npz', 'TIME'),
            'fields': arrays(CURRENT / 'fields.npz', folder / 'fields.npz', 't'),
            'historical_accepted_steps': meta['substeps'] - meta['failed_substeps'],
            'historical_rejected_attempts': meta['failed_substeps'],
            'historical_accepted_newton': meta['newton_iters'],
            'historical_manifest_policy': {k: previous.get(k) for k in
                ('profile', 'amg_rebuild', 'cpr_reuse_graph', 'newton_line_search_fallback')},
        }
    result = {
        'scope': 'Complete strict 3312-day reservoir/wells, no detailed surface facility. Different historical sources and host conditions; ratios are observations, not isolated or repeated optimization A/Bs.',
        'current_run': str(CURRENT.relative_to(ROOT)), 'current_meta': current_meta,
        'current_audit': validation['audit'], 'field_cumulative_vs_eclipse': validation['field_cumulative'],
        'historical_comparisons': reports,
    }
    destination = Path(__file__).with_name('comparison.json')
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: {k: row[k] for k in ('historical_wall_s', 'current_wall_s', 'observed_wall_ratio', 'physical_step_grid_and_accepted_newton_equal')}
                      | {'summary_equal': row['summary']['all_arrays_equal'], 'fields_equal': row['fields']['all_arrays_equal']}
                      for name, row in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
