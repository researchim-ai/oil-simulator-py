# Exact half-step reuse after temporal rejection

The thermal/flow vessel integrator estimates time error by comparing one full
backward-Euler step with two half steps. When that estimate rejects the attempt,
the next attempted dt is halved. Its coarse solve is exactly the first half step
that was just computed from the same initial state.

Reuse that validated state instead of solving it again. The memo holds one
state and is confined to this `advance` call, this initial state identity and
this exact dt. It is populated only after all three solves and physical checks
succeed and the temporal estimate rejects the attempt. Nonlinear, EOS or
conservation failures do not populate it. Neither discarded coarse ledgers nor
reused states are added to accepted histories. Accepted fine stages are still
computed and audited normally. No tolerance, growth factor or numerical scheme
changes.

## Checks

`tests/test_surface_thermal_retry_reuse.py` compares enabled/disabled reuse on
a real heated PR vessel with temporal rejections. Complete result dictionaries,
including time grid and ledgers, match exactly. It verifies one fewer step solve
per temporal rejection and fresh computation after an injected physical failure.
The broader regression output is in `tests.log`; final retry tests in
`retry-tests.log`.

## Reproduction

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src:scripts:tests /home/user/miniconda3/bin/python \
benchmarks/reports/receiver_retry_reuse/compare.py
python benchmarks/reports/receiver_retry_reuse/summarize.py
```

The full coupled 0.05 s transition uses the same prepared state, export command
and tolerances as `../receiver_pressure_probe`. Its archived completed run is
the reference. Both use optimized Newton and completed pressure probes. Timing
is a shared-host CPU observation with some overlapping regression work, not an
isolated GPU speed guarantee. `comparison.json` retains the complete state,
ledger and independent audit; `summary.json` contains field deltas and counts.

This removes exact redundant work on rejected attempts. It does not eliminate
full/two-half work on accepted attempts, pressure coupling iterations, or the
need to qualify an alternative higher-order time integrator. Local surface
subcycling remains opt-in; this test is not a full-field performance claim.

## Completed local-integration comparison

The interval completed and passed independent audit. Compared with the archived
pressure-probe run: **170.806 → 164.404 s**, nonlinear solves **1,574 → 1,526**,
residual evaluations **11,064 → 10,680**. All compared reservoir and receiver
physical fields matched exactly; accepted pressure-trial counts stayed 5/2.
This is a modest measured change (3.7% less elapsed time), not a radical speedup.

`direct.py` additionally evaluates the same physical model without nested local
subcycling at two outer timesteps. It uses the same legacy-prepared initial
state and optimized Newton. `direct.json` separates integration time from
independent audits and retains every accepted report. This comparison changes
the temporal discretization, not the physical model; it must be interpreted
with the refinement differences, not as identical trajectories.

## Ordinary integration and refinement

With local subcycling disabled (the existing default), the SAME physical model
completed the 0.05 s transition in **33.867 s** at dt=0.005 s and **56.437 s**
at dt=0.0025 s. All 30 intervals passed independent audits. The largest reported
whole-system temporal-error ratio was 0.203414 (acceptance limit 1).

Halving dt changed final inlet pressure by 4.674 Pa (operating pressure about
1 MPa), temperature by 0.0002805 K, component inventories by at most
0.0001644 mol and energy by 0.1496 J. The refined ordinary run differed from
the local run by 1.396 Pa and 0.00008564 K. These are discretization differences,
not certified errors against an exact solution. Reservoir pressure differences
between the two ordinary runs were 2.33e-9 Pa.

Thus ordinary integration is faster for this THREE-CELL reservoir. This is not
a policy to disable local integration for large reservoirs: more frequent
reservoir solves may dominate there. No fidelity reduction, tolerance change,
or automatic mode selection was introduced. The result supports retaining the
existing non-subcycled default while the expensive nested algorithm is developed.
