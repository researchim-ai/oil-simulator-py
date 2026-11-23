# Restart failure: discontinuous pump check-valve thermal equations

## Captured failure and cause

The original integrated-facility checkpoint at 0.4 s is preserved without
changing inventories, geometry, commands, wave fan or tolerance budgets.
`first_failed_candidate.json` captures the first unsuccessful Newton candidate
in the slow coupled separator/pipe/equipment operator, at dt=0.004443094602 s.
The reconstruction matches the original initial nonlinear vector exactly.

The legacy solver stalls after 17 iterations and 355 residual evaluations,
with maximum residual 0.0014300953 in the gas takeoff momentum equation. The
physical-validity flag is true. Refreshing every Jacobian and removing update
limits reproduce the same failure: neither fixes the cause.

A Newton direction changes water-pump check-valve activity. The old residual
selects closed/open thermal equations on each trial using `gap > scaled_flow`.
Those equations have different values at the switching surface: rows 136/137
jump from zero to approximately -0.220523/-0.007109. The 0.2205 jump remains
at alpha=2**-24. The Newton linear system is solved accurately, but line search
cannot cross this finite residual discontinuity. The larger momentum residual
is therefore a symptom, not the original defective equation.

`direction_check.log`, `jump_check.log` and `branch_replay.log` retain the
finite-direction checks and the failed legacy/fresh-Jacobian/unlimited-update
comparisons. The diagnostic used eager CPU chemical kernels to separate solver
behavior from compilation. Its timing is not a full-facility speed benchmark.

## Repair

Pump check valves now use a fixed active branch within a Newton solve:

- Closed: zero flow; the existing closed thermal constraints remain.
- Open: zero pressure gap; the existing PS/PH and isenthalpic delivery
  constraints remain.
- A negative pressure gap on a closed branch requests opening.
- An open branch blocked at its nonnegative-flow bound with a positive
  pressure gap requests closure.

A changed active set starts another solve of the coupled system before any
state is accepted. All other selected valve branches survive phase/restriction
retries. The active-set retry budget is bounded; exhaustion remains a failure.
The same treatment covers single-liquid and equilibrium liquid-mixture pumps.
Signed pumps keep their existing reversible mapping.

The final residual, phase stability, equipment envelope, component/energy
balances and independent audit remain mandatory. No tolerance is relaxed.
The comparison-only class switch `PumpConnectionBundle._fixed_check_valves`
retains the old residual for regression evidence. Runtime reports expose the
active-set policy, selected branches and retry count. Failure messages now
include Newton stop reason, physical validity and the largest residual's row
and ordinary-node equation when available.

## Verification

The saved smooth branch converges in 8–10 residual evaluations after repair;
the original dynamic-branch residual takes 355 and fails. This is a branch
solve comparison, not an accepted-interval speedup claim: required active-set
updates and spatial refinement must also finish.

The full repaired slow step, including independent column quadrature
refinement to order 4, is accepted with quadrature error 3.92e-5. Both oil and
water check valves are open in the final physical state. The water flow is
1.6149e-8 m3/s; the original failure sat on its nearly zero-flow switch.
The independently audited energy residual is within 1.2e-7 J locally.

`repaired_slow_step_audit.json` uses the separate NumPy PR/equipment/flux
checker. `cuda_slow_step_audit.json` independently accepts the same captured
step on RTX 3090. CPU/CUDA endpoint difference is 8.8443e-9 in the original
network error norm, at exactly the same clock. GPU execution includes cold
kernel/compilation work and is not a device speed comparison.

Tests:

- 109 regression tests passed, one skipped: pumps, mixtures, signed operation,
  the actual restart branch, subcycling, shared quadrature and journals.
- Separate CPU/CUDA restart suite: seven tests passed (overlaps CPU suite).
- The captured legacy discontinuity is deliberately reproduced in a regression
  test; the fixed branch must have a small full-step residual instead.
- Tests require complementarity switching in both directions and rejection
  rather than acceptance when the active-set budget is exhausted.

Reproduction:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts \
  /home/user/miniconda3/bin/python -m pytest -q \
  tests/test_surface_pr_check_valve_active_set.py
```

The compact physical checkpoint and failing nonlinear vector are in
`tests/data/surface/pr_restart_check_valve.json`.

## Full-cycle scope

`cycle_prefix.accepted.jsonl` retains the original startup/isolation history
through 0.4 s with explicit segment-level source provenance. It is independently
audited before the continuation in `repaired_launch.json` starts. The new
900-second budget excludes earlier prefix runtime.

The repaired full scenario passes the original Newton/phase-branch failure.
Its initial 0.008886 s probe is now rejected for actual combined temporal error
8.805, rather than nonlinear failure; smaller probes remain independently
checked. This repair does not remove the expensive stiff fast-pipe transient,
operator-splitting error or repeated temporal probes. Full operating-cycle
completion/qualification must be read from `repaired_cycle.json`; interval and
GPU checks above do not imply it.

## Completed operating cycle

The original one-second startup/isolation/restart schedule is now complete.
`final_cycle.json` records `completed: true`,
`full_operating_cycle_qualified: true` and `sources_unchanged: true`.
`final_cycle.full-cycle.trajectory.json` contains the reunited original request
and all actual accepted states. The source provenance of the earlier prefix
is retained explicitly; this is a restarted qualification, not a cold timed run
on one solver revision from time zero.

Independent whole-cycle audit:

- 270 accepted split halfsteps, 2,418 local pipe steps, 270 slow steps checked.
- Largest absolute total component residual: 2.765e-10 mol.
- Total energy residual: -1.510e-6 J.
- Both liquid pumps and the gas compressor transfer mass during startup and
  restart, and exactly zero during isolation.
- All five separator feed/takeoff ports participate during startup and restart,
  with exactly zero transfer during isolation.
- The final report clock is exactly 1.0 s.

The first post-repair continuation used its 900 s wall budget and retained 34
accepted macrosteps through 0.5145568972 s. The checked continuation driver
then resumed the exact accepted state, after confirming unchanged source and
model, and completed the remaining interval in 605.04 s. These observed timings
exclude earlier startup/isolation work and prefix audits and include shared-host
conditions and initial temporal rejections. They do not establish a cold-cycle
performance speedup. Detailed dynamics remains too expensive for a fast
long-horizon forecast; this change fixes a real nonlinear restart failure.

No full-cycle temporal-refinement study or full-facility CUDA cycle is claimed.
The CUDA evidence above covers the captured accepted coupled slow step.
