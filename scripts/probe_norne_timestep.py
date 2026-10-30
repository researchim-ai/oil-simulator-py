"""Controlled Norne continuation with an optional timestep cap.

Run with CUDA_VISIBLE_DEVICES selecting the physical GPU. Outputs are new-only;
checkpoint history is preserved, and accepted substeps are audited strictly.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from simulator.blackoil_fim import BlackOilFIMSolver
from simulator.eclipse_run import iter_deck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deck', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--end', required=True, type=float)
    parser.add_argument('--cap-days', type=float)
    parser.add_argument('--eclipse-grid', help='Align steps to accepted times printed in an Eclipse PRT (rounded)')
    parser.add_argument('--debug-timings', action='store_true', help='Record synchronized solver stage timings')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not math.isfinite(args.end) or args.end <= 0:
        parser.error('--end must be finite and positive')
    if args.cap_days is not None and (not math.isfinite(args.cap_days) or args.cap_days <= 0):
        parser.error('--cap-days must be finite and positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint)
    manifest = dict(arguments=vars(args), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted((ROOT / 'src').rglob('*.py'))},
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=torch.__version__)
    if args.eclipse_grid:
        manifest['eclipse_prt_sha256'] = hashlib.sha256(Path(args.eclipse_grid).read_bytes()).hexdigest()
    manifest['launcher_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / 'launcher.py').write_text(Path(__file__).read_text())
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    original_step = BlackOilFIMSolver.step
    original_convergence = BlackOilFIMSolver.convergence
    samples = []
    status = dict(status='loading')
    started = time.monotonic()
    criteria = {}

    def convergence(self, *a, **kw):
        ok, detail = original_convergence(self, *a, **kw)
        if ok:
            if detail['cnv_relaxed'] or max(detail['cnv']) > 1e-3 or max(detail['mb']) > 1e-7:
                raise RuntimeError('Accepted substep violates strict convergence')
            criteria.update(cnv=detail['cnv'], mb=detail['mb'])
        return ok, detail

    def step(self, dt, current_time=None):
        wall = time.monotonic()
        ok = original_step(self, dt, current_time=current_time)
        samples.append([(current_time or 0) / 86400, dt / 86400, int(ok),
                        self.last_newton_iters if ok else -1, time.monotonic() - wall])
        return ok

    BlackOilFIMSolver.step = step
    BlackOilFIMSolver.convergence = convergence
    context = {}
    debug = None
    try:
        steps = iter_deck(args.deck, device=torch.device(args.device), resume=str(checkpoint),
                          max_days=args.end, context=context, yield_start=True)
        _, start, summary = next(steps)
        solver, sim = context['solver'], context['sim']
        manifest['execution_options'] = dict(
            deterministic=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        if solver.tol_cnv != 1e-3 or solver.tol_mb != 1e-7 or solver.relaxed_max_pv_fraction != 0:
            raise RuntimeError('Checkpoint is not strict')
        if not np.any(np.abs(np.cumsum(context['case'].report_days) - args.end) < 1e-7):
            raise RuntimeError('End must coincide with a report date')
        if args.debug_timings:
            from simulator.solver_debug import SolverDebugger
            debug = SolverDebugger(dict(enabled=True, level=1, console=False, dump_on_failure=False,
                                        output_dir=str(out / 'debug'), run_id='timings'),
                                   solver.debug.dims, torch.device(args.device))
            debug.eq_names = solver.debug.eq_names
            solver.debug = debug
        if args.end <= start:
            raise RuntimeError('End day must follow checkpoint')
        if args.cap_days is not None:
            sim.sim_params['max_time_step_days'] = args.cap_days
            overrides = set(sim.sim_params.get('_deck_tuning_overrides', ()))
            overrides.add('max_time_step_days')
            sim.sim_params['_deck_tuning_overrides'] = tuple(overrides)
        if args.eclipse_grid:
            text = Path(args.eclipse_grid).read_text(errors='replace')
            grid = np.unique([float(t) for t in re.findall(
                r'^ STEP\s+\d+\s+TIME=\s*([\d.]+)\s+DAYS', text, re.MULTILINE)])
            if not len(grid) or grid[-1] < args.end:
                raise RuntimeError('Eclipse printed grid does not cover requested end')
            run_step = sim.run_step

            def aligned_step(dt):
                begin = sim.current_time / 86400.
                end = begin + dt / 86400.
                targets = [*grid[(grid > begin + 1e-7) & (grid < end - 1e-7)], end]
                cap = sim.sim_params.get('max_time_step_days')
                try:
                    for target in targets:
                        days = float(target - sim.current_time / 86400.)
                        limit = days if args.cap_days is None else min(days, args.cap_days)
                        sim.sim_params['max_time_step_days'] = limit
                        sim._fim_dt = limit * 86400.
                        if not run_step(days * 86400.):
                            return False
                    return True
                finally:
                    if cap is None:
                        sim.sim_params.pop('max_time_step_days', None)
                    else:
                        sim.sim_params['max_time_step_days'] = cap

            sim.run_step = aligned_step
        status.update(status='running', start_day=start, day=start)
        for _, day, summary in steps:
            status.update(day=day, wall_s=time.monotonic() - started,
                          accepted=sum(r[2] for r in samples), failed=sum(1-r[2] for r in samples),
                          last_criteria=criteria)
            np.savez_compressed(out / 'summary.npz', **summary)
            np.savez_compressed(out / 'substeps.npz', data=np.asarray(samples))
            (out / 'status.json').write_text(json.dumps(status, indent=2))
            print(f'day={day:g} wall={status["wall_s"]:.1f}s accepted={status["accepted"]}', flush=True)
        if abs(status['day'] - args.end) > 1e-7:
            raise RuntimeError('End must coincide with a report date')
        arrays = {key: getattr(solver.fluid, attr).detach().cpu().numpy()
                  for key, attr in [('p', 'pressure'), ('sw', 's_w'), ('sg', 's_g'),
                                    ('so', 's_o'), ('rs', 'rs'), ('rv', 'rv')]}
        arrays.update({key: getattr(solver.wells, key).detach().cpu().numpy()
                       for key in ['conn_cell', 'conn_well', 'conn_wi', 'conn_head']})
        arrays['well_names'] = np.array([w.name for w in solver.wells.wells])
        np.savez_compressed(out / 'final_fields.npz', **arrays)
        status['status'] = 'done'
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        if debug is not None:
            debug.close()
        BlackOilFIMSolver.step = original_step
        BlackOilFIMSolver.convergence = original_convergence
        (out / 'status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
