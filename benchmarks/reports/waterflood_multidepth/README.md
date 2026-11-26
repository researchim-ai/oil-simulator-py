# Multi-depth water-injection coupling

The JSON mixed waterflood adapter now reuses the existing standard-well
hydrostatics for pure incompressible water. Injection perforations may be
above and below the BHP datum. This is not an open-completion crossflow,
compressible-water or multisegment-well implementation.

The 8×8×3 synthetic reservoir has layer thicknesses 20/30/50 m, injection
perforations at 10/35/75 m and BHP datum 40 m. Two producing wells retain
single completions at their BHP datum. Reservoir, producers, preparation,
limited makeup, disposal, pumping and injection use the main FIM/Schur path.

## What passed and what did not

- `regression.log`: 79 passed, 2 CUDA-required checks skipped. Includes
  hydrostatic equilibrium, invariance under a change of BHP datum, mixed
  check-valve states, finite-difference B/D/Cv, coupled shutdown/restart,
  a forged-head audit rejection and a vapor-pressure domain rejection.
- `coarse/` and `reference/`: completed 30 days and seven operating events
  with 122/242 accepted steps. Both independent material/equipment/control
  audits pass. Configured maximum steps are 6/3 hours.
- **`refinement.json` is rejected.** Errors exceed predeclared budgets:
  BHP 1085.89 Pa, cell pressure 1244.60 Pa, Sg 3.0521e-4, component rates
  0.3873%, transported masses 0.3340%. Do not present these month trajectories
  as qualified to the requested time-accuracy budgets.
- `perforation_refinement.json` is also rejected: completion pressure
  252.42 Pa, component rates 0.3400%, with no valve-state disagreements.
- `time_error_diagnostic.json` locates the maximum well error near day four
  on P2. This is not exclusively a first-step error or an injection-head
  discrepancy. Further qualification requires tighter time resolution or
  a separately verified temporal controller for the mixed driver.
- `fine_day/` and `finest_day/` are a separate one-day resolution diagnostic
  at 1.5 hours/45 minutes, without the month's operating events. Their gate
  **passes**: BHP 73.10 Pa, component rates 0.01972%, completion pressure
  5.37 Pa and completion rates 0.01683%. It is stored separately and cannot
  qualify the 30-day event study.

These are numerical checks of a synthetic model. There are no measured
equipment maps, experimental calibration or external field-history reference.
The production adapter's multi-depth restriction remains. The water-column
model excludes friction, storage and acceleration between datum and perforations.
Unsupported vapor-pressure crossing is rejected rather than solved as flashing.

## Artifacts

- `engineering.csv`: every accepted coarse stage, actual water transfers,
  operating modes, power and cumulative shaft work. It inherits the failed
  temporal qualification of the coarse month run.
- `perforations.csv`: 366 completion rows, with depths, heads, actual fluxes,
  valve states and reservoir pressure. Isolated BHP gauges are blank.
- `validation.json`: consolidated status and scope, including failed gates.
- `source_manifest.json` / `sources.tar.gz`: guarded source/input snapshot.
- `checksums.json`: SHA-256 of all stored artifacts except itself.

All runs are CPU, one Torch thread each, partly concurrent on a shared host.
The month solve wall times are not clean performance benchmarks, nor GPU timings.

## Reproduce

From the repository root, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
`CUDA_VISIBLE_DEVICES=''` and `PYTHONPATH=src:scripts:tests`:

```bash
python scripts/run_fim_waterflood.py \
  --config configs/fim_waterflood_field192_coarse.json \
  --days 30 --report-days 0.5 --schedule configs/fim_waterflood_schedule.json \
  --save-fields --output /tmp/waterflood-3d-coarse
python scripts/run_fim_waterflood.py \
  --config configs/fim_waterflood_field192_reference.json \
  --days 30 --report-days 0.5 --schedule configs/fim_waterflood_schedule.json \
  --save-fields --output /tmp/waterflood-3d-reference
python scripts/compare_fim_waterflood.py \
  /tmp/waterflood-3d-coarse /tmp/waterflood-3d-reference \
  --budgets benchmarks/reports/waterflood_multidepth/budgets.json \
  --output /tmp/waterflood-3d-refinement.json
```

The last command is expected to reject this 6/3-hour pair. For the one-day
diagnostic, use the `fine`/`finest` configs, `--days 1`, and omit `--schedule`.
Exporter and completion comparison scripts in this directory require an intact
completed trajectory and the same independent accounting audit.
