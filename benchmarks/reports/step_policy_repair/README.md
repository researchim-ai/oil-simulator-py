# Certified Newton stopping and split temporal-probe quadrature

Two changes remove repeated numerical work without changing physical or
integration tolerances:

1. Newton can stop at a caller-certified arithmetic floor. Every residual row
   above the strict target must fit its own finite, nonnegative roundoff bound;
   every uncertified row still meets the original strict target. Final residual,
   budget, finite-value and network physical audits remain mandatory. A plain
   stagnation tolerance cannot authorize this shortcut.
2. The locally subcycled surface can reuse the independently certified slow-
   plant quadrature order for the discarded full temporal probe. Both accepted
   halfsteps still qualify their spatial quadrature independently, and fast
   pipe operators retain their own local temporal checks. Missing/mixed/invalid
   certificates or a failed direct probe retain the original refinement path.

The reference numerical policies remain available through
`run_pr_controlled_surface.py --confirm-roundoff-plateau --refine-discarded-probe`.
The latter now controls both monolithic and subcycled implementations. Runtime
reports expose the selected policies.

## Accepted connected-network comparisons

`nominal_request.json` restarts at 0.1665991401 s and requests 0.02 s of the
same integrated facility. No geometry, stocks, connections, commands, time-
error budget or physical tolerance is changed between policies. Independent
physical checks run outside solver timing and must pass for both policies.

- `nominal_pair/`: reference 45.18 s, accelerated 27.01 s; observed 1.673×.
- `nominal_reverse_pair/`: accelerated runs first; reference 40.93 s,
  accelerated 29.67 s; observed 1.380×.
- Endpoint difference is 7.31755e-9 in the ORIGINAL full-network error norm in
  both pairs; both requests finish in one macro attempt at the same clock.
- Total residual evaluations drop 998 → 703; Jacobian builds 215 → 193.
  For the coupled 268/376-unknown systems specifically, residual evaluations
  drop 354 → 59 and Jacobian builds 34 → 12. The 99 local 18-unknown solves
  and their 578 residual evaluations remain unchanged.

These are two paired shared-host observations with opposite execution order,
including initialization. They are not statistical confidence bounds or a
claim of production-speed whole-network simulation. In particular, the
remaining 27–30 seconds for this accepted 0.02 s interval is still too slow.

`shutdown_request.json` applies the original isolation/motor commands at
0.2 s over the previously accepted first shutdown interval of 8.93005e-6 s.
`shutdown_pair/` reports 35.45 → 29.40 s (observed 1.206×), identical endpoint,
one macro attempt and both independent audits passing. It is a short event
accuracy regression, not evidence that tiny steps are appropriate for an
entire operating forecast or that the complete shutdown/restart cycle passed.

Compressed reports contain actual states, solver counters, independent audits
and complete requests. Summaries fingerprint the source used for each pair.
The CLI metadata-only change between pairs is retained in their hashes.

## Verification and limitations

The complete CPU regression set passed 89 tests, with one CUDA-only test skipped;
a separate CPU/CUDA Newton set passed 61 tests on RTX 3090. These suites overlap.
Tests cover wrong/nonfinite/negative roundoff certificates, strict uncovered
rows, attainable targets, phase/flow reversals, isolation, conservation,
checkpoint/restart, spatial probe fallback and rejection of unverified probes
in accepted histories. The full CPU worker ended with exit code zero.

`cuda_coupled_check.json` additionally covers a small coupled separator/pipe/
controller interval on RTX 3090: both policies pass independent audits and
have zero difference in the original error norm. It is not a full-facility
GPU speed benchmark.

The old full-cycle run was stopped and journaled before source edits. Neither
these interval comparisons nor the prior PR microbenchmarks qualify the full
one-second startup/isolation/restart cycle or its temporal refinement. The
next major unchanged work is local fast-pipe integration and repeated local
thermodynamic/flux derivative assembly; that requires separate profiling and
verification rather than extrapolating the global-solve savings.

`continued_launch.json` identifies the resumed full-cycle check on the new
policy. Its independently audited prefix preserves 32 accepted macrosteps
through 0.20000893005 s, with segment-level source provenance. New accepted
states are journaled from that clock onward. The 1200 s budget and continuation
timing exclude the previously computed prefix; they must not be presented as
cold full-cycle runtime. Whole-cycle qualification remains pending until the
original schedule and its independent audit actually complete.
