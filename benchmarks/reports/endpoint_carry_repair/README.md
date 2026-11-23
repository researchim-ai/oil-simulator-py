# Carrying the local step proposal across interval boundaries

The fast transport integrator previously used an endpoint-clipped last step as
its next-step proposal. In the original nominal facility interval, a remaining
0.1316 ms tail replaced a 1.947 ms proposal with 0.197 ms. Subsequent Strang
stages then paid for repeated growth from this artificial short step.

`PREquilibriumTransportNetwork.advance` now remembers the proposal before
clock clipping. Only an accepted clipped trial with error < 0.25 may retain
that proposal. An error rejection/nonlinear failure replaces it with the cut
step, and error >= 0.25 retains the original conservative update. Maximum and
minimum step bounds remain. Every later interval independently checks the
coarse/fine error, phase stability, nonlinear equations and physical balances.
No geometry, inventory, operating command or acceptance tolerance is changed.

The reference policy is available through
`run_pr_controlled_surface.py --shrink-at-interval-end`. Runtime reports record
`preserve_transport_endpoint_step`.

## Work and timing on an accepted interval

Both `nominal_pair/` and `nominal_reverse_pair/` use the same 0.02 s request
starting at 0.1665991401 s, with the earlier certified Newton/probe optimizations
enabled for BOTH policies. The paired harness toggles only endpoint carry.

- Nonlinear solves: 114 -> 84.
- Residual evaluations: 703 -> 531.
- Jacobian builds: 193 -> 136.
- Both policies finish in one outer attempt at exactly the requested clock.
- Both pass independent component/energy/trajectory audits.
- Reference-first timing: 39.88 -> 21.21 s (1.880x observed).
- Accelerated-first timing: 27.40 -> 25.49 s (1.075x observed).

These shared-host measurements include initialization/compiler warmup. The
first pair overlapped regression testing. The wide order-dependent timing
spread does not establish a reproducible 1.88x speedup. The deterministic work
reductions reproduce in both orders. These are complete accepted-interval
measurements, not rejected-trial or isolated-flash timings.

## Accuracy qualification, including the failed same-path comparison

Both paired summaries intentionally retain `agree: false`, and their harness
exits 1: the original strict same-path threshold was 0.01 of the original
error norm, whereas the changed adaptive path differs by 0.0197319586.
That initial comparison is not relabeled as passing.

A separate `validate_pr_endpoint_carry.py` qualification tightens ONLY the
local temporal-error fraction from 0.1 to 0.025 and 0.00625 (4x and 16x).
Physical plant, initial stocks and commands are identical; only the numerical
model identity is explicitly updated for the stricter local error budget.
All refined trajectories pass independent audits.

Differences from the 16x-refined result, measured in the ORIGINAL full-network
error norm, are:

- Original endpoint policy: 0.00103098157.
- Endpoint-carry policy: 0.0187009771.
- 4x-refined policy: 0.0162563058.

The two refined endpoints agree within 0.025 (the 4x local budget), and both
original-budget policies are within 0.1 (the existing local-error allocation)
of the stricter reference. `nominal_refinement/summary.json` records this
separate qualification. This is local temporal-refinement evidence for this
interval, not proof of monotone convergence or unchanged global forecast error.
The adaptive path does change the numerical endpoint.

## Verification and continuation

- 68 transport, phase-change, subcycling, shared-probe and journal tests passed.
- 16 clock/endpoint tests passed on CPU before adding the reference-policy test.
- Final endpoint-policy suite: 16 CPU/CUDA cases passed on RTX 3090, including
  error/rejection guards, original-policy reproduction and a real closed vessel.
  These counts overlap and must not be added as unique cases.
- Ruff and diff whitespace checks passed.

The previous whole-cycle continuation ended interrupted at 0.3291256402 s,
with `sources_unchanged: true`. Its accepted steps show the shutdown step grows
from 12 microseconds to 16 milliseconds: there is no permanently stuck
microstep in that recorded segment. This does not explain/qualify the entire
initial shutdown transient.

`cycle_prefix.accepted.jsonl` combines the original startup prefix and the
recorded shutdown continuation after exact checkpoint/model continuity checks.
It retains segment-level source provenance and explicitly marks mixed solver
sources. `continued_launch.json` starts a new independently audited continuation
from 0.3291256402 s, with a 600 s internal wall-time budget. Timing excludes the
old prefix. Full-cycle completion and qualification must be checked in the
result; they are not implied by the interval/refinement checks above.

## Restart bottleneck diagnosis

The bounded diagnostic in `restart_local_diagnostic.json` identifies velocity
in `gas_recycle_0` as the largest local temporal-error coordinate. At restart,
finite vessel pressures are 1,019,980.3 and 999,730.8 Pa, while the pipe is at
1,009,992.3 Pa and effectively at rest. The original 1 mm pipe and binary
opening commands are unchanged.

For the first fast interval of 0.0022215473 s, coarse/fine pipe velocities are
0.26199446 and 0.26730276 m/s. Their 0.00530830 m/s difference is 53.083 times
the existing 0.0001 m/s local velocity tolerance. Halving is initially
non-monotone; the first recorded passing local trial is 1.0847399e-6 s with
error 0.313044. It remains essential to distinguish this actual temporal-error
rejection from the artificial endpoint-proposal shrink repaired above.

`restart_eager_profile_top.txt` records a bounded 20 s cProfile diagnostic:
70 transport `_step` calls occur in the first fast stage; the largest costs
include EOS evaluation, nonlinear residuals and characteristic validation.
CPU chemical compilation was disabled ONLY in this diagnostic to separate
physical/local-integration work from compiler overhead. Its times are not a
production speed benchmark. The main continuation retains CPU compilation.

An earlier diagnostic combining cProfile, compiler activity and asynchronous
faulthandler stack dumping terminated with SIGSEGV in that separate child.
`restart_profile_status.json` retains this instrumentation failure; it produced
no accepted solver trajectory and must not be counted as validation. The
independent main continuation was not changed by either diagnostic.

The fixed-bound wave fan is still the original configured 4000 m/s fan.
A different acoustic fan is a distinct numerical discretization and is not
silently substituted to obtain a faster result. Whether that conservative fan
adds avoidable stiffness requires a separate accuracy/refinement comparison.

`accepted_prefix_audit.json` independently qualifies the newly accepted segment
through 0.4 s (isolation complete), with energy residual -7.44e-8 J and largest
component residual 2.91e-11 mol. It preserves the remaining restart request.
On the subsequent restart, the first 0.0088861892 s macrotrial was rejected
with the existing generic Newton/stable-phase/HLLC branch-budget message;
the integrator retries 0.0044430946 s. That message does not identify which of
those gates failed. No accepted restart endpoint or complete-cycle qualification
is claimed. Endpoint-step carry does not resolve this separate branch failure.

Final continuation status: the 600 s budget expired at the last accepted clock
0.4 s, after five newly accepted macrosteps. `continued_cycle.json` records
`completed: false`, `sources_unchanged: true` and the retained accepted journal.
No accepted macrostep after restart was obtained. The continuation process has
finished; it is not still running. This is a release-relevant remaining problem,
not a successful complete operating-cycle benchmark.
