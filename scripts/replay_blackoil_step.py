"""Replay a trusted local pre-step snapshot from benchmark_norne_accuracy.

The snapshot is a Python pickle: never load files from untrusted sources.
No schedule, timestep or convergence settings are changed by this runner.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def replay(snapshot, output, allow_source_change=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = torch.load(snapshot, weights_only=False)
    execution = data['execution']
    changed = [name for name, expected in data['source_hashes'].items()
               if not (ROOT / name).exists()
               or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected]
    if changed and not allow_source_change:
        raise ValueError('Snapshot sources differ; use --allow-source-change for an explicit experiment')
    torch.set_num_threads(execution['threads'])
    torch.use_deterministic_algorithms(execution['deterministic'], warn_only=execution['warn_only'])
    torch.backends.cuda.matmul.allow_tf32 = execution['matmul_tf32']
    torch.backends.cudnn.allow_tf32 = execution['cudnn_tf32']
    solver = data['solver']
    original = solver.convergence
    trace = []

    def convergence(*args, **kwargs):
        ok, criteria = original(*args, **kwargs)
        trace.append(dict(criteria))
        return ok, criteria

    solver.convergence = convergence
    report = dict(status='running', day=data['current_time'] / 86400,
                  dt_days=data['dt'] / 86400, execution=execution, changed_sources=changed,
                  snapshot_sha256=hashlib.sha256(Path(snapshot).read_bytes()).hexdigest())
    started = time.monotonic()
    try:
        ok = solver.step(data['dt'], current_time=data['current_time'])
        report.update(status='accepted' if ok else 'rejected',
                      # last_newton_iters can describe a previous step on failure.
                      newton_corrections=solver.last_newton_iters if ok else None,
                      linear=solver._last_linear_stats)
    except Exception as exc:
        report.update(status='error', error=repr(exc))
        raise
    finally:
        solver.__dict__.pop('convergence', None)
        report.update(wall_s=time.monotonic() - started, criteria=trace)
        (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', help='Trusted local pre-step .pt file')
    parser.add_argument('--out', required=True)
    parser.add_argument('--allow-source-change', action='store_true')
    args = parser.parse_args()
    result = replay(args.snapshot, args.out, args.allow_source_change)
    print(result['status'], f"{result['wall_s']:.3f}s")
    sys.exit(0 if result['status'] == 'accepted' else 1)
