"""Same-source cross-GPU trajectory differences, separate from timing claims."""
import json
from pathlib import Path

import numpy as np
from compare import ROOT, arrays


def main():
    first = ROOT / '.scratch/norne_release_20261124_gpu/run'
    second = ROOT / '.scratch/norne_release_20261124_gpu_ti/run'
    manifests = [json.loads((d / 'manifest.json').read_text()) for d in (first, second)]
    for key in ('source_files', 'input_files', 'reference_files'):
        assert manifests[0][key] == manifests[1][key], key
    validations = [json.loads((d / 'validation.json').read_text()) for d in (first, second)]
    assert all(v['status'] == 'passed_recorded_audit' for v in validations)
    meta = [json.loads((d / 'meta.json').read_text()) for d in (first, second)]
    fields = arrays(first / 'fields.npz', second / 'fields.npz', 't')
    summary = arrays(first / 'summary.npz', second / 'summary.npz', 'TIME')
    with np.load(first / 'substeps.npz') as a, np.load(second / 'substeps.npz') as b:
        x, y = a['data'][:, :4], b['data'][:, :4]
        n = min(len(x), len(y))
        bad = np.flatnonzero((abs(x[:n, :2] - y[:n, :2]).max(1) > 1e-7)
                            | np.any(x[:n, 2:] != y[:n, 2:], axis=1))
        i = int(bad[0]) if len(bad) else None
        steps = {'same_shape': x.shape == y.shape,
                 'equal': bool(x.shape == y.shape and np.array_equal(x, y)),
                 'first_material_difference_index': i,
                 'material_clock_tolerance_days': 1e-7,
                 'rtx3090_row': x[i].tolist() if i is not None else None,
                 'rtx3090_ti_row': y[i].tolist() if i is not None else None}
    with np.load(first / 'summary.npz') as a, np.load(second / 'summary.npz') as b:
        cumulative = {k: {'rtx3090': float(a[k][-1]), 'rtx3090_ti': float(b[k][-1]),
                         'ti_minus_3090_percent': float(100 * (b[k][-1] - a[k][-1]) / a[k][-1])}
                      for k in ('FOPT', 'FWPT', 'FGPT')}
    peaks = {}
    with np.load(first / 'fields.npz') as a, np.load(second / 'fields.npz') as b:
        assert np.array_equal(a['t'], b['t'])
        for key in ('p', 'sw', 'sg', 'so', 'rs'):
            delta = abs(a[key].astype(float) - b[key].astype(float))
            index = np.unravel_index(delta.argmax(), delta.shape)
            peaks[key] = {'max_absolute_difference': float(delta[index]),
                          'saved_day': float(a['t'][index[0]]),
                          'zero_based_grid_index': [int(v) for v in index[1:]],
                          'unit': 'bar' if key == 'p' else 'fraction' if key in ('sw', 'sg', 'so') else 'SM3/SM3'}
            if key == 'p':
                peaks[key]['max_absolute_difference_mpa'] = float(delta[index]) * 0.1
    result = {'scope': 'Two complete strict 3312-day runs with identical frozen sources, inputs and references, different GPU models. Shared-host timings and different adaptive trajectories are not an isolated optimization speed comparison.',
              'same_sources_inputs_references': True, 'both_recorded_audits_pass': True,
              'rtx3090': meta[0], 'rtx3090_ti': meta[1],
              'summary': summary, 'fields': fields, 'attempts': steps,
              'cumulative_cross_device': cumulative, 'saved_field_peaks': peaks,
              'limitation': 'Strict nonlinear convergence and material audits do not prove temporal accuracy or cross-device field equivalence. These runs do not establish which trajectory is more accurate.'}
    destination = Path(__file__).with_name('cross_device_comparison.json')
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('attempts', 'cumulative_cross_device', 'saved_field_peaks')}, indent=2))


if __name__ == '__main__':
    main()
