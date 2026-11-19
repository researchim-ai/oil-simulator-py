# Coupled operating-point preparation and local PR receiver steps

## Implemented scope

- `fixed_reservoir_wells` solves the existing well/gathering equations at fixed reservoir fields, with the existing well tolerances and full rollback. No reservoir time step or cumulative production is used during initialization.
- `DynamicReceiverCoupledFIM.prepare_operating_point` iterates well rates, explicit stock-to-species/thermal characterization, and the stationary receiver pressure. It returns a proposed initial state without installing it. Reservoir fields remain unchanged; the reservoir is not claimed stationary.
- `scripts/prepare_dynamic_receiver.py` exports a separate run configuration and initialization ledger. `prepared/` is an actual CLI output; its full 1.02 s schedule was NOT executed here.
- Opt-in `control.surface_subcycling` integrates one PR flow vessel locally inside each reservoir pressure trial. Exact integrated feeds match the constant implicit well rates of that trial. Outer full/two-half error control, endpoint pressure checks, and rollback remain active.
- The scalar auditor checks all accepted local vessel stages, their sums, and coupled end pressures. A tampered local export is rejected.
- Partial trajectory pressure failures are treated as bounds, not complete endpoint residuals. Predictors use a completed trajectory and the original feed-pressure admissibility bound. No physical acceptance threshold was relaxed.

Scope is deliberately restricted: one vessel for local integration, one phase for operating-point preparation, existing positive producing-well controls. No arbitrary-graph operating-point initializer, thermal reservoir, injection, higher-order time integration, or field calibration was added. This does not replace the quasi-steady operational facility driver.

## Completed transition comparison

Manufactured three-cell/two-producer coupled case; 0.1 m diameter, 20 m tall receiver; outlet conductance 3e-6 m3/(s Pa), unchanged 5% pressure-drop validity domain. All cases start at the same prepared operating point and set export opening to 0.9 for 0.05 s.

- Existing method, one 0.05 s attempt: rejected, temporal ratio 5.262676, nine reservoir calls, about 5.57 s CPU.
- Local surface steps, one 0.05 s attempt: accepted, temporal ratio 0.036193, sixteen reservoir calls, about 284.30 s CPU; independent audit accepted. Accepted fine stages contain 20 and 16 local vessel steps.
- Existing method, five 0.01 s accepted steps: 35 reservoir calls, about 47.52 s; each step independently audited.
- Existing method, ten 0.005 s accepted steps: 70 reservoir calls, about 42.31 s; each step independently audited.

Differences of the locally integrated receiver versus the 0.01 s reference: pressure 12.3443 Pa, temperature 0.0007390 K, maximum component stock 0.00043447 mol. Against the 0.005 s reference: pressure 3.27744 Pa, temperature 0.000194864 K, maximum component stock 0.00011556 mol. These are differences between discretizations, not certified errors against an exact solution. Full reservoir field comparisons were not recorded in this benchmark.

**This is not a demonstrated overall speedup.** The local scheme accepts a larger reservoir interval and reduces reservoir calls compared with completing the interval at smaller steps, but it is much slower on this tiny case. Surface integration and repeated pressure trials dominate. Runs overlapped some CPU tests; wall times are observations, not isolated benchmark ratios. Large-reservoir/GPU benefit remains unverified. The mode stays off by default.

Two earlier diagnostic local runs were interrupted while investigating pressure-bracket behavior. Their partial logs are kept separately; they are not successful benchmarks. `comparison.json` is the completed final run.

## Validation and reproduction

The main regression suite passed 41 CPU tests (one GPU test skipped, one spatial-batch test excluded). Additional overlapping suites passed 16 tests and, after the final pressure-bound change, 13 tests. A forced-failure rollback test also passed. Logs are retained. Ruff and whitespace checks passed.

`compare.py`, `reference.py`, and `reference_finer.py` reuse repository test fixtures. From the repository root, create `/tmp/receiver-multirate` and run with:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts:tests python benchmarks/reports/receiver_operating_multirate/compare.py
```

These diagnostic scripts write into `/tmp/receiver-multirate`. The production initialization CLI refuses an existing output directory. Source hashes are in `source_sha256.json`; initialization has its own source manifest.

Remaining performance work: cheaper local integration and pressure response evaluation, followed by a large coupled-model benchmark. General graph/multiphase initialization and higher-order integration remain separate unfinished work.
