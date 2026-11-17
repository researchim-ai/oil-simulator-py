"""Alternate two source trees on one model; require exact accepted trajectories.

Each child forecast must complete its source guard and accounting audit. This
is an execution regression on a shared host, not independent physical validation.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

from benchmark_initial_cell_cache import require_equal
from check_facility_refinement_pair import completed

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--candidate-root', type=Path, default=ROOT)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--schedule', type=Path)
    p.add_argument('--days', type=float, required=True)
    p.add_argument('--pairs', type=int, default=3)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--cuda-memory-fraction', type=float, default=.08)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not math.isfinite(args.days) or args.days <= 0 or args.pairs < 1 or args.threads < 1:
        p.error('Positive finite days and positive pair/thread counts required')
    if args.device == 'cuda' and (not os.environ.get('CUDA_VISIBLE_DEVICES') or
            not math.isfinite(args.cuda_memory_fraction) or not 0 < args.cuda_memory_fraction <= 1):
        p.error('CUDA requires explicit CUDA_VISIBLE_DEVICES and a fraction in (0, 1]')
    sources = [args.reference_root.resolve(), args.candidate_root.resolve()]
    for source in sources:
        if not (source / 'scripts/run_fim_facility_forecast.py').is_file() or not (source / 'src').is_dir():
            p.error(f'Source tree lacks forecast runner or src: {source}')
    config = json.loads(args.config.read_text())
    config['fluid']['pvt_path'] = str((ROOT / config['fluid']['pvt_path']).resolve())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    model_path = out / 'config.json'
    model_path.write_text(json.dumps(config, indent=2) + '\n')
    schedule_path = None
    if args.schedule:
        schedule_path = out / 'schedule.json'
        schedule_path.write_text(args.schedule.read_text())
    rows = []
    try:
        for repeat in range(args.pairs):
            pair = {}
            for candidate in ((False, True) if repeat % 2 == 0 else (True, False)):
                source = sources[int(candidate)]
                name = f'pair{repeat}_{"candidate" if candidate else "reference"}'
                env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                           PYTHONPATH=f'{source / "src"}:{source / "scripts"}')
                if args.device == 'cpu':
                    env['CUDA_VISIBLE_DEVICES'] = ''
                command = [sys.executable, str(source / 'scripts/run_fim_facility_forecast.py'),
                           '--config', str(model_path), '--days', str(args.days), '--report-days',
                           str(args.days), '--device', args.device, '--output', str(out / name)]
                if schedule_path:
                    command += ['--schedule', str(schedule_path)]
                if args.device == 'cuda':
                    command += ['--cuda-memory-fraction', str(args.cuda_memory_fraction)]
                with (out / f'{name}.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                summary, bundle, _ = completed(out / name, args.days)
                pair[candidate] = bundle
                rows.append({'repeat': repeat, 'candidate': candidate, 'directory': name,
                             'solve_wall_s': summary['solve_wall_s'],
                             'trajectory_sha256': summary['trajectory_sha256'], 'audit': summary['audit']})
                print(json.dumps(rows[-1]), flush=True)
            if pair[False]['config'] != pair[True]['config'] or pair[False]['schedule'] != pair[True]['schedule']:
                raise ValueError('Different physical/numerical configuration or operating commands')
            require_equal(pair[False]['result'], pair[True]['result'])
        medians = {str(c): statistics.median(r['solve_wall_s'] for r in rows if r['candidate'] is c)
                   for c in (False, True)}
        result = {'accepted': True, 'scope': __doc__, 'rows': rows, 'device': args.device,
                  'days': args.days, 'threads': args.threads, 'exact_all_accepted_stages': True,
                  'median_s': medians, 'observed_ratio': medians['False'] / medians['True']}
        (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))
    except BaseException as error:
        (out / 'failure.json').write_text(json.dumps({'accepted': False, 'reason': str(error)}) + '\n')
        raise


if __name__ == '__main__':
    main()
