# Temporal control for mixed production / water-injection FIM

The full 30-day pair **passes the unchanged global refinement gate** at all
60 report dates. Both independent material/equipment/control audits pass.
This is numerical qualification of the stated synthetic scope, not external
physical calibration or a guaranteed absolute global error bound.

## Full-month result

The controlled/reference studies accepted 588/766 actual half stages.
Maximum differences are cell pressure 110.922 Pa, Sw 5.130e-6, Sg 2.032e-5,
physical well BHP 83.318 Pa, component rates 0.024882%, cumulative transported
masses 0.025892% and cumulative shaft work 0.006160%. Well modes match.
Individual injection completions also pass: pressure 16.859 Pa, rates
0.022757%, with no valve-state mismatches. Both studies execute all seven
events. Maximum whole-trajectory component balance residual is 1.316e-5 kg.

`refinement.json` and `perforation_refinement.json` are the complete gates;
`validation.json` consolidates status, scope and work. The earlier uncontrolled
6/3-hour monthly pair remains rejected in `../waterflood_multidepth/`.

CPU solve wall times were 2434.48 / 3145.11 s (40.57 / 52.42 minutes) under
the shared-host execution described below. The strict controller is opt-in;
this campaign does not establish a production-throughput improvement.

## Implementation

`WaterfloodCoupledFIM` reuses the existing transactional `temporal_step`
integrator and `FacilityTemporalControl` policy. Three whole-system solves
compare one full BE step against two consecutive BE half steps. Only the
two half steps survive; all full-probe reservoir/well state and cumulative
well volumes are rolled back. Each whole-system solve includes the actual
production/water-resource/injection dispatch, not a scaled final flow.

The mixed-driver hooks preserve well/component identity when comparing
integrated fluxes. Positive production cannot cancel negative injection.
Cell-component inventories prevent cancellation across cells and avoid
comparing undefined solution ratios at phase disappearance. Other defects
cover P/Sw/Sg, external inputs/outputs, all five water transfers, shaft work
and optional connected BHP. Isolated mathematical pressure gauges are excluded.
See `method.json` for the error scales and `docs/SURFACE_CONFIG_REFERENCE.md`
for configuration and report semantics.

Dispatch pressure resolution must be at least ten times tighter than the
local BHP budget. Algebraic tolerances are tightened temporarily and restored
after success, error rejection or an exception. The maximum measured local
defect must not exceed one. The controller proposes the next interval from
the measured defect and restarts its history on a control change.

The runner emits both actual half-stage ledgers. `driver.report` describes
the last half stage, not the entire requested macro interval; integrate
`accepted_stages` for macro totals. The audit checks pairs, exact clocks,
control/event boundaries and coarse-work evidence; it recomputes well
withdrawal, resource-transfer, shaft-work and BHP defects from stored data.
The independent ledger audit does not re-solve the reservoir field equations.

## Verification and artifacts

- `final_regression.log`: 122 passed, 6 CUDA-required checks skipped.
- `cumulative_ledger_test.log`: discarded probes do not enter well cumulatives;
  cumulative stock volumes equal the accepted half-stage flux integrals.
- `restart_test.log`: serialized state preserves the next solution/report
  and the temporal proposal history.
- `pilot_day/`: one-day implementation pilot. Its exact earlier source
  revision is preserved in `pilot_sources.tar.gz` and its manifest. The
  subsequent changes add work counters and report metadata; this pilot is
  not the source snapshot used by the full-month pair.
- `controlled/` and `reference/`: the full 30-day source-guarded studies.
  They share the same physical inputs, events and 6-hour macro ceiling.
  The reference halves local P/Sw/Sg/BHP/component-relative tolerances.
  Actual accepted stages are half the proposed macro intervals.
- `budgets.json`: copied unchanged from the rejected old 6/3-hour study.
- `live_report_comparisons.json`: preliminary observed common-date well
  reports, with incomplete sampling. This is not the complete refinement gate.
- `environment.json`: dependency versions and execution scope.
- `engineering.csv` / `perforations.csv`: all controlled accepted stages and
  all individual completion histories, with isolated pressure gauges blank.
- `sources.tar.gz` / `source_manifest.json`: current guarded source/input
  snapshot; `checksums.json` covers the stored artifacts except itself.

Scope: 192 cells (8×8×3), two producers and one three-depth water injector,
synthetic equipment maps, pure incompressible injected water, explicit
nonreturn completions and quasi-steady surface trees/facility. The production
adapter still has its single-datum completion restriction. No experimental
calibration, bare-completion reverse oil/gas flow, full thermal energy balance
or long-term field-history validation is claimed.

The controller is opt-in. Accurate temporal estimation costs additional
whole-system solves, each of which can include many reservoir dispatch trials.
`temporal_work` counts that work, including rejected estimates. These are CPU
single-thread runs partly concurrent on a shared host, not clean performance
or GPU benchmarks. Local tolerances are not accumulated global error bounds.

## Reproduce

From the repository root, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
`CUDA_VISIBLE_DEVICES=''` and `PYTHONPATH=src:scripts:tests`:

```bash
python scripts/run_fim_waterflood.py \
  --config configs/fim_waterflood_field192_controlled.json \
  --days 30 --report-days 0.5 --schedule configs/fim_waterflood_schedule.json \
  --save-fields --output /tmp/waterflood-controlled
python scripts/run_fim_waterflood.py \
  --config configs/fim_waterflood_field192_controlled_reference.json \
  --days 30 --report-days 0.5 --schedule configs/fim_waterflood_schedule.json \
  --save-fields --output /tmp/waterflood-controlled-reference
python scripts/compare_fim_waterflood.py \
  /tmp/waterflood-controlled /tmp/waterflood-controlled-reference \
  --budgets benchmarks/reports/waterflood_temporal/budgets.json \
  --output /tmp/waterflood-controlled-refinement.json
```
