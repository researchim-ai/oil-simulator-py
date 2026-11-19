"""Audit nested fixed-grid Norne runs before reporting endpoint trends.

Observed order is a self-convergence diagnostic, not an absolute error bound.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from check_fim_time_refinement import difference_order

ROOT = Path(__file__).resolve().parents[1]


def original_grid_path(manifest):
    path = Path(manifest['arguments']['grid'])
    return (path if path.is_absolute() else ROOT/path).resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_nested(grids):
    require(len(grids) == 3, 'Exactly three grids required')
    for g in grids:
        require(g.ndim == 2 and g.shape[1] >= 4 and len(g) > 0,
                'Nonempty step arrays required')
        require(np.isfinite(g[:, :4]).all() and np.all(g[:, 1] > 0)
                and np.all(g[:, 2] == 1), 'Only accepted finite positive steps allowed')
        require(np.allclose(g[1:, 0], g[:-1, 0] + g[:-1, 1], rtol=0, atol=1e-8),
                'Grid has gaps or overlap')
    for a, b in zip(grids, grids[1:]):
        require(len(b) == 2 * len(a), 'Each step must split into two')
        require(np.allclose(b[::2, 0], a[:, 0], rtol=0, atol=1e-8)
                and np.allclose(b[1::2, 0], a[:, 0] + a[:, 1]/2, rtol=0, atol=1e-8)
                and np.allclose(b[:, 1], np.repeat(a[:, 1]/2, 2), rtol=0, atol=1e-8),
                'Grids must be nested exact halvings')


def protocol(manifests):
    require(len(manifests) == 3, 'Exactly three manifests required')
    for key in ('sources', 'launcher_sha256', 'effective_settings'):
        require(manifests[0][key] == manifests[1][key] == manifests[2][key],
                f'Different {key}')
    inputs = []
    for m in manifests:
        grid = str(original_grid_path(m))
        inputs.append({k: v for k, v in m['inputs'].items() if k != grid})
    require(inputs[0] == inputs[1] == inputs[2], 'Different physical inputs or initial checkpoint')


def audit(paths):
    paths = [Path(p) for p in paths]
    manifests = [json.loads((p/'manifest.json').read_text()) for p in paths]
    protocol(manifests)
    grids, fields, summaries, statuses = [], [], [], []
    for p, m in zip(paths, manifests):
        status = json.loads((p/'status.json').read_text())
        require(status['status'] == 'complete' and status['source_and_inputs_verified'],
                'Incomplete or unverified run')
        grid_path = p/'grid.npz'
        original_grid = original_grid_path(m)
        if not grid_path.exists():
            grid_path = original_grid
        require(hashlib.sha256(grid_path.read_bytes()).hexdigest() == m['inputs'][str(original_grid)],
                'Grid checksum differs')
        with np.load(grid_path) as z:
            g = z['data']
        rows = json.loads((p/'steps.json').read_text())
        require(len(rows) == len(g) == status['accepted'], 'Recorded step count differs')
        require(np.allclose([r['day'] for r in rows], g[:, 0], rtol=0, atol=1e-8)
                and np.allclose([r['dt'] for r in rows], g[:, 1], rtol=0, atol=1e-8),
                'Actual trajectory differs from grid')
        require(abs(status['start'] - g[0, 0]) < 1e-8
                and abs(status['day'] - g[-1, 0] - g[-1, 1]) < 1e-8, 'Wrong horizon')
        for key, limit in [('cnv', m['effective_settings']['tol_cnv']),
                           ('mb', m['effective_settings']['tol_mb']), ('balance', 1e-4)]:
            values = np.asarray([r[key] for r in rows])
            require(np.isfinite(values).all() and np.max(abs(values)) <= limit,
                    f'Failed {key} audit')
            if key != 'balance':
                require((values >= 0).all(), f'Negative {key} norm')
        with np.load(p/'fields.npz') as z:
            f = {k: z[k].reshape(-1) for k in ('p', 'sw', 'sg', 'so', 'pv')}
        require(all(np.isfinite(v).all() for v in f.values()) and (f['pv'] > 0).all()
                and (f['p'] > 0).all(),
                'Invalid fields or PV')
        require(all(v.shape == f['pv'].shape for v in f.values()), 'Field shape mismatch')
        require(all((f[k] >= -1e-10).all() and (f[k] <= 1+1e-10).all() for k in ('sw','sg','so'))
                and np.allclose(f['sw']+f['sg']+f['so'], 1, rtol=0, atol=1e-10), 'Invalid saturations')
        with np.load(p/'summary.npz') as z:
            summaries.append({k: z[k] for k in ('TIME', 'FOPT', 'FWPT', 'FGPT')})
        grids.append(g); fields.append(f); statuses.append(status)
    check_nested(grids)
    require(len({f['p'].shape for f in fields}) == 1, 'Different reservoir shape')
    # Use one fixed positive norm for all three solutions.
    root_weight = np.sqrt(fields[-1]['pv']/fields[-1]['pv'].sum())
    trends = {}
    for k in ('p', 'sw', 'sg', 'so'):
        values = [f[k] for f in fields]
        result = difference_order(*(root_weight*v for v in values))
        result.update(coarse_medium_max=float(abs(values[0]-values[1]).max()),
                      medium_fine_max=float(abs(values[1]-values[2]).max()),
                      unit='Pa' if k == 'p' else 'fraction')
        trends[k] = result
    cumulative = {}
    for summary, status in zip(summaries, statuses):
        require(np.isfinite(summary['TIME']).all() and np.all(np.diff(summary['TIME']) > 0)
                and abs(summary['TIME'][-1]-status['day']) < 1e-8, 'Invalid SUMMARY clock')
    for k in ('FOPT', 'FWPT', 'FGPT'):
        cumulative[k] = difference_order(*[[s[k][-1]] for s in summaries])
    return dict(protocol_verified=True, days=statuses[0]['day'],
                accepted_steps=[len(g) for g in grids], fields=trends,
                all_field_rms_differences_decrease=all(v['decreases'] for v in trends.values()),
                cumulative=cumulative, runs=[str(p) for p in paths],
                scope='Three nested temporal grids at fixed physics, sources and tolerances. '
                      'Endpoint PV-weighted RMS uses finest-grid PV for all vectors. '
                      'No absolute error bound, calibration or asymptotic convergence claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs=3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.runs)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))
