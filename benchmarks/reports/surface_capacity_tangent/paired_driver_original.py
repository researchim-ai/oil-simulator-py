import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path('/home/user/workspace/oil-simulator-py-impes')
work = ROOT / '.scratch/facility_tangent'
source = work / 'final_source'
sys.path[:0] = [str(source / 'scripts'), str(source / 'src')]
from check_facility_capacity_search import compare

env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           MPLCONFIGDIR='/tmp/pr_forecast_mpl', PYTHONPATH=f'{source / "src"}:{source / "scripts"}')
rows = []
checks = []
for repeat in range(3):
    for candidate in ((False, True) if repeat % 2 == 0 else (True, False)):
        name = 'candidate' if candidate else 'reference'
        directory = work / f'final_pair{repeat}_{name}'
        command = [sys.executable, str(source / 'scripts/run_fim_facility_forecast.py'),
                   '--config', str(work / f'{name}.json'), '--days', '.003', '--report-days', '.003',
                   '--device', 'cpu', '--output', str(directory)]
        with (work / f'final_pair{repeat}_{name}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=180)
        s = json.loads((directory / 'summary.json').read_text())
        row = {'repeat': repeat, 'candidate': candidate, 'solve_wall_s': s['solve_wall_s'],
               'directory': str(directory), 'audit': s['audit']}
        rows.append(row)
        print(json.dumps(row), flush=True)
    check = compare(work / f'final_pair{repeat}_reference', work / f'final_pair{repeat}_candidate', .003)
    checks.append(check)
    if not check['accepted'] or not check['fewer_fim_solves']:
        raise RuntimeError(f'Pair {repeat} failed accuracy/work gate: {check}')
medians = {str(c): statistics.median(r['solve_wall_s'] for r in rows if r['candidate'] is c)
           for c in (False, True)}
result = {'accepted': True, 'scope': 'Three alternating CPU pairs on a shared host; '
          '1024 cells, 16 wells, 0.003 days. Fixed strict regression budgets, unchanged FIM/time policy.',
          'rows': rows, 'checks': checks, 'median_s': medians,
          'observed_ratio': medians['False']/medians['True']}
(work / 'final_pairs.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('rows','checks')}, indent=2), flush=True)
