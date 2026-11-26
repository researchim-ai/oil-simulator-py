# Waterflood dispatch execution reduction

The 30-day same-physics / same-time-policy comparison passes all predeclared
execution budgets at 60 report dates. This is an internal numerical comparison,
not an ECLIPSE/OPM performance claim or experimental physical calibration.

## Result

- Guarded month: 2434.481 → 432.384 s, observed speedup 5.63×. The reference
  is the existing archived `../waterflood_temporal/controlled/` trajectory;
  it was not unnecessarily rerun. Exact archived source/input hashes are
  checked before accepting this cross-version comparison.
- Whole-system temporal solves: 882 → 882; accepted half stages: 588 → 588.
- Actual nested reservoir FIM calls: 25,548 → 4384 (5.83× fewer).
- Same-code short opposite-order pairs: 20.936/3.923 s and 21.082/3.832 s.
  Each short pair uses 186 → 30 FIM calls and passes the execution gate.
  `matched_pairs/` preserves guarded raw studies. These runs overlapped the
  month calculation on a shared host, so wall-time ratios are observed,
  not clean-device speed guarantees.
- Month maximum differences: cell pressure 0.13294 Pa, physical BHP
  0.38167 Pa, Sw 2.20e-8, Sg 8.12e-8, component rates 1.051e-6 scaled,
  cumulative transported masses 1.911e-6 relative and shaft work 2.100e-7
  relative. Well modes match. Individual completions differ by at most
  0.09129 Pa and 7.339e-7 scaled rates; valve states match.
- Regression: 194 passed, 6 CUDA-required checks skipped. All studies here
  use one CPU thread; GPU performance is not measured.

The retained first-order time estimator still solves the full step and two
half steps at their own durations/states. Those are different mathematical
problems. This change removes repeated final solves and redundant capacity
search/property work within each of them; it does not claim one FIM call
for an entire time interval or the absolute minimum possible work.

## Changes

`WaterfloodDispatchControl` defaults to safeguarded search, solved-state
restore and a bounded 64 MiB initial-cell cache. It reuses existing
`CapacityBracket`, `InitialCellStateCache` and `_StepState` implementations.
Scalar brackets now also admit a zero-injection lower anchor.

Measured constraint facets propose trial fractions. Every proposed point
is physically solved; absent/opaque constraints retain bisection safeguards.
Both fraction width and actual common physical BHP span must still pass the
original tolerances. No interpolated reservoir state or scaled well flow is
accepted. Pump operating envelopes are unchanged; diagnostic facets do not
extend the admissible property maps.

Only one selected converged state is retained, including well cumulatives.
It replaces the old final replay. Each old-state property cache is confined
to one old state/dt/time and is bypassed if its memory budget is exceeded.
Nonlinear iterates always get their own properties. Water-tree geometry is
cached per device, while rates, properties, pressure losses and tangents are
evaluated for their current values.

Report v2 identifies the actually solved committed trace row and the FIM/cache
work. Its independent audit rejects duplicate points in state-restore mode,
wrong selected-state evidence and inconsistent work/cache counters.

## Evidence

- `month_execution.json`, `completion_execution.json`: strict execution gates.
- `inherited_time_quality.json`: conservative triangle bounds combining the
  old accepted temporal refinement with this tight execution comparison.
  This preserves the prior time-quality budgets without claiming that a new
  stricter temporal month was run. Relative bounds use identical denominator
  floors and `e_new_fine <= e_old_fine + e_new_old*(1+e_old_fine)`.
- `short_execution.json`, `day_execution.json`, `matched_pair_*.json`:
  same-code and archived-version bridges.
- `regression.log`: replay equality, well cumulatives, zero anchor, opaque
  fallback, BHP stop criteria, cache-budget bypass, rollback/restart and
  existing pump/power/NPSH/control scenarios.
- `engineering.csv`, `perforations.csv`: all accepted month stages/completions.
- `validation.json`, `source_manifest.json`, `sources.tar.gz`, `checksums.json`:
  consolidated status and the exact new source/input snapshot.

## Reproduce

From the repository root, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
`CUDA_VISIBLE_DEVICES=''` and `PYTHONPATH=src:scripts:tests`:

```bash
python scripts/run_fim_waterflood.py \
  --config configs/fim_waterflood_field192_dispatch_fast.json \
  --days 30 --report-days 0.5 --schedule configs/fim_waterflood_schedule.json \
  --save-fields --output /tmp/waterflood-dispatch-fast
python scripts/compare_fim_waterflood.py \
  /tmp/waterflood-dispatch-fast benchmarks/reports/waterflood_temporal/controlled \
  --execution --reference-source-archive benchmarks/reports/waterflood_temporal/sources.tar.gz \
  --budgets benchmarks/reports/waterflood_dispatch/execution_budgets.json \
  --output /tmp/waterflood-dispatch-execution.json
```

`benchmark_pairs.py` creates new sequential pairs and refuses to overwrite an
existing `matched_pairs/` directory. `dispatch_baseline` selects the old
bisection/replay/no-cache policy inside the new source version.
Pure incompressible injected water, explicit nonreturn completions, the
producer completion restriction and the quasi-steady surface scope remain.
