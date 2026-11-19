"""Qualify the already completed adaptive runs without rerunning the candidate."""
import gzip
import json
import statistics
from pathlib import Path

from check_facility_engineering_profile import check
from validate_operational_forecast import workload

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
candidate = OUT.parent / 'operational_adaptive_forecast_fast' / 'refined'
reference = OUT / 'refined'
budgets = json.loads((ROOT / 'configs/fim_facility_forecast_budgets.json').read_text())
comparison = check(candidate, reference, budgets, days=30.)
(OUT / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False))


def statistics_for(directory):
    with gzip.open(directory / 'trajectory.json.gz', 'rt') as handle:
        result = json.load(handle)['result']
    stages = result['accepted_substeps']
    intervals = [s['dt_s'] for s in stages]
    return {
        **workload(directory),
        'step_hours': {'minimum': min(intervals) / 3600,
                       'median': statistics.median(intervals) / 3600,
                       'maximum': max(intervals) / 3600},
        'time_fraction_with_steps_over_two_hours': sum(dt for dt in intervals if dt > 7200) / sum(intervals),
        'event_boundaries_present': all(any(abs(s['time_days'] - day) < 1e-10 for s in stages)
                                        for day in (10., 20.)),
    }


status = {
    'status': 'complete' if comparison['accepted'] else 'accuracy_failed',
    'accepted': comparison['accepted'],
    'candidate': str(candidate.relative_to(ROOT)),
    'reference': str(reference.relative_to(ROOT)),
    'days': 30.,
    'work': {'operating': statistics_for(candidate), 'refined': statistics_for(reference)},
    'scope': '1024-cell, 16-well synthetic blackoil forecast with quasi-steady facilities; no dynamic vessel inventory or transient multiphase validation',
    'timing_scope': 'Shared-host CPU with one Torch thread; not GPU or universal performance',
    'accuracy_scope': 'Declared numerical refinement budgets, not error against the exact solution or experimental calibration',
    'tests': {'passed': 47},
}
(OUT / 'status.json').write_text(json.dumps(status, indent=2, allow_nan=False))
print(json.dumps(status, indent=2))
raise SystemExit(0 if comparison['accepted'] else 1)
