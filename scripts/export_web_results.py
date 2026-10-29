"""Export the audited Norne run as small, portable website verification files.

NORNE_DIR=opm-tests/norne NORNE_OUT=results/norne python scripts/export_web_results.py
Requires a complete audited bundle: comparison.json, manifest.json, meta.json,
resume_validation.json, summary.npz, fields.npz and substeps.npz.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from simulator.ecl_io import EclSummary

root = Path(os.environ.get('NORNE_OUT', 'results/norne'))
output = Path(os.environ.get('NORNE_WEB_DATA', 'web/data'))
report = json.loads((root/'comparison.json').read_text())
meta = json.loads((root/'meta.json').read_text())
manifest = json.loads((root/'manifest.json').read_text())
restart = json.loads((root/'resume_validation.json').read_text())
if not report['complete'] or report['days'] != 3312 or manifest['profile'] != 'strict':
    raise ValueError('Publish only a completed strict Norne validation')
if max(report['audit']['max_cnv'].values()) > 1e-3 or max(report['audit']['max_mb'].values()) > 1e-7:
    raise ValueError('Accepted steps violate strict convergence')
output.mkdir(parents=True, exist_ok=True)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
keys = ['FOPT','FWPT','FGPT','FOPR','FWPR','FGPR','FWIR','FGIR']
public = dict(schema_version=1, run_id=root.name,
              validation_date=datetime.fromtimestamp(manifest['started_unix'], timezone.utc).date().isoformat(),
              reference='Eclipse 2014.2, NORNE_ATW2013', days=report['days'],
              units='standard m3; rates in standard m3/day; pressure in bar',
              convergence=meta['convergence'], wall_s=meta['wall_s'], device=meta['device'],
              cpu=meta['cpu'], threads=meta['threads'], torch=meta['torch'],
              newton_iterations=meta['newton_iters'], audit=report['audit'],
              field_cumulative=report['field_cumulative'], pressure=report['pressure'],
              wells_cumulative=report['wells_cumulative'], execution_notes=report['execution_notes'],
              restart={k:restart[k] for k in ['start_day','end_day','vectors_checked','max_scaled_error']},
              mean_normalized_errors=[r for r in report['metrics']['mean'] if r['vector'] in keys],
              error_definition='final: 100*(ours-reference)/reference; mean: mean absolute error/reference scale on report dates, not pointwise relative error',
              provenance=dict(source_files=manifest['source_files'], input_files=manifest['input_files'],
                              reference_files=manifest['reference_files'],
                              cublas_workspace_config=manifest['cublas_workspace_config'],
                              files_sha256={name:sha(root/name) for name in ['summary.npz','fields.npz','substeps.npz','manifest.json','comparison.json']}))
(output/'norne-results.json').write_text(json.dumps(public,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
with (output/'norne-wells.csv').open('w',newline='') as stream:
    rows=report['wells_cumulative'];writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
reference=EclSummary(Path(os.environ.get('NORNE_DIR','opm-tests/norne'))/'ECL.2014.2/NORNE_ATW2013')
summary=np.load(root/'summary.npz')
with (output/'norne-field.csv').open('w',newline='') as stream:
    writer=csv.writer(stream);writer.writerow(['day']+[p+'_'+key for key in keys for p in ['ours','eclipse']])
    for i,day in enumerate(summary['TIME']):
        row=[float(day)]
        for key in keys:
            a=float(summary[key][i]);b=float(np.interp(day,reference.days,reference.get(key)))
            row.extend([a if np.isfinite(a) else '',b if np.isfinite(b) else ''])
        writer.writerow(row)
print('Exported',output)
