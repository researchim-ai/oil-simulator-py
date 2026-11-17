"""Alternating, source-guarded CPU/CUDA pairs for exact initial-cell-state reuse."""

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import torch
from check_fim_facility_forecast import audit
from run_fim_blackoil_surface import run_case

ROOT = Path(__file__).resolve().parents[1]


def require_equal(a, b):
    for key in ('final_pressure_pa', 'final_s_w', 'final_s_g', 'final_rs', 'withdrawn_component_mass_kg'):
        if a[key] != b[key]:
            raise ValueError(f'Initial cell reuse changed {key}')
    if len(a['accepted_substeps']) != len(b['accepted_substeps']):
        raise ValueError('Initial cell reuse changed the number of accepted stages')
    for x, y in zip(a['accepted_substeps'], b['accepted_substeps'], strict=True):
        for key in ('time_days', 'dt_s', 'surface', 'withdrawn_component_mass_kg', 'newton_iterations'):
            if x[key] != y[key]:
                raise ValueError(f'Initial cell reuse changed accepted-stage {key}')
        for key in ('trace', 'material_account', 'facility', 'fraction'):
            if x['facility_coupling'][key] != y['facility_coupling'][key]:
                raise ValueError(f'Initial cell reuse changed coupled {key}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--days', type=float, required=True)
    p.add_argument('--pairs', type=int, default=3)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--cuda-memory-fraction', type=float, default=.08)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not math.isfinite(args.days) or args.days <= 0 or args.pairs < 1:
        p.error('Positive finite days and positive pair count required')
    if args.output.exists():
        p.error('Output already exists')
    if args.device == 'cuda':
        if not os.environ.get('CUDA_VISIBLE_DEVICES'):
            p.error('Select explicit CUDA_VISIBLE_DEVICES')
        if not math.isfinite(args.cuda_memory_fraction) or not 0 < args.cuda_memory_fraction <= 1:
            p.error('CUDA memory fraction must be in (0, 1]')
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, 0)
    config = json.loads(args.config.read_text())
    config['fluid']['pvt_path'] = str(ROOT / config['fluid']['pvt_path'])
    receiver = config['simulation']['coupled_facility']['facility']['separator_stages'][0]['pressure_pa']
    paths = [*sorted((ROOT / 'src').rglob('*.py')), *sorted((ROOT / 'scripts').glob('*.py')),
             args.config.resolve(), Path(config['fluid']['pvt_path'])]
    def hashes():
        return {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    before = hashes()
    rows = []
    for repeat in range(args.pairs):
        pair = {}
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            c = copy.deepcopy(config)
            c['simulation']['coupled_facility']['control']['reuse_initial_cell_state'] = enabled
            r = run_case(c, receiver, args.days, args.days, args.device)
            checked = audit(c, r, expected_schedule=[], expected_days=args.days)
            pair[enabled] = r
            rows.append({'repeat': repeat, 'enabled': enabled, 'solve_wall_s': r['solve_wall_s'],
                         'accepted_stages': len(r['accepted_substeps']), 'audit': checked})
            print(json.dumps(rows[-1]), flush=True)
        require_equal(pair[False], pair[True])
    if hashes() != before:
        raise RuntimeError('Benchmark sources/config/PVT changed')
    medians = {str(enabled): statistics.median(r['solve_wall_s'] for r in rows if r['enabled'] is enabled)
               for enabled in (False, True)}
    result = {'accepted': True, 'scope': __doc__, 'required_days': args.days, 'device': args.device,
              'torch_threads': torch.get_num_threads(),
              'cuda_memory_fraction': args.cuda_memory_fraction if args.device == 'cuda' else None,
              'rows': rows, 'median_s': medians, 'observed_ratio': medians['False'] / medians['True'],
              'exact_all_accepted_stages': True, 'sources_unchanged': True, 'source_sha256': before,
              'timing_scope': 'Alternating pairs on a shared host; not an isolated hardware speed claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows', 'source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
