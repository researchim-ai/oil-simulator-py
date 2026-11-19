# Shared Newton for the gravitational receiver

The gravitational column, thermal vessel, finite-aperture flow vessel and
operating-point initializer previously imported `TorchNewton` from
`surface.torch_linear`. They now use the existing
`surface.torch_nonlinear.TorchNewton`, already used by other surface models.
No new nonlinear algorithm or physical closure was introduced.

This enables guarded Jacobian/factor reuse within one solve, early acceptance
of a sufficiently improving full step, and avoidance of the unused regularized
normal solve on healthy CPU Jacobians. Numeric factors do not cross solves or
phase topologies. The nonlinear solver enforces the caller's requested tolerance;
physical, quadrature, temporal-error and independent audit checks remain active.

## Reproduction

From the repository root:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src:scripts:tests /home/user/miniconda3/bin/python \
benchmarks/reports/receiver_newton_reuse/compare.py
```

`compare.py` prepares one common state using the legacy implementation, then
clones the complete model for each run. Both runs simulate 0.05 s with the
export opening reduced to 0.9 and local subcycling enabled. The file retains
complete receiver and reservoir states, ledgers, audit results, timings and
Newton residual-evaluation counts. Audit time is outside the integration timer.
This is a CPU comparison on a shared host, not a clean GPU or full-Norne result.
Some regression tests overlapped the legacy run; timings are indicative.

## Completed comparison

Both variants completed the same 0.05 s and passed the independent scalar audit.
Observed CPU wall time: **301.843 → 244.339 s** (1.235×, 19.1% less time).
Newton residual evaluations: **35,488 → 15,590** (56.1% fewer).
Nonlinear solve calls stayed at **2,218**: this patch reduces work within solves,
not the number of pressure replays or adaptive integration stages.

Final reservoir pressure, water/gas saturation and dissolved-gas fields matched
exactly. Maximum receiver component difference was 7.0e-12 mol, energy difference
5.5e-9 J and temperature difference 2.62e-11 K. Wall temperature and phase
interfaces matched exactly. See `summary.json` and complete `comparison.json`.
One shared-host pair is not a production speed guarantee; the earlier five
0.01 s reference without local subcycling was much faster. Local subcycling
therefore remains disabled by default.

## Validation

- 35 receiver/column/thermal/initialization/coupling tests passed (summary in
  `receiver-tests.json`). These include two/three-phase columns, moving
  interfaces, finite apertures, conservation, restart and failed-step rollback.
- 29 graph/momentum/network tests passed (`graph-tests.log`).
- Shared-Newton behavior checks are recorded in `newton-tests.log`.
- `profile-before.json` profiles the subcycling test module, including setup
  and independent audits. Its cumulative times overlap and must not be summed.
  It is not a direct timestep performance measurement.

## Remaining limits

This change does not remove the outer pressure replay or nested step doubling.
Local subcycling remains opt-in. It does not qualify performance for arbitrary
networks, large field forecasts, or GPU execution. It does not replace detailed
thermodynamics with an approximation or establish experimental calibration.
