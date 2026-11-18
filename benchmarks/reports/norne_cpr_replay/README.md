# Full Norne CPR replay qualification

**Decision: do not promote persistent CPR replay for Norne.** Default remains
`cpr_reuse_graph=false`. Both complete runs passed the existing strict per-step
and accumulated mass audit, but the candidate was slower and changed the
adaptive trajectory. Passing the physical audit is not an equivalence claim.

## Completed paired runs

Norne ATW2013, 3312 days, reservoir and wells only; no additional surface network.
Sequential runs on the same explicitly selected RTX 3090, one PyTorch CPU thread.
Baseline: 648.594 s wall / 620.137 s inside solver, 599 accepted substeps,
33 rejected attempts, 3103 Newton iterations.
Fixed candidate: 673.050 s wall / 631.908 s inside solver, 604 accepted substeps,
29 rejected attempts, 3166 Newton iterations.
Observed candidate overhead: 3.77% wall, 1.90% solver. One pair on a shared host
is not a statistically isolated performance result. Baseline SUMMARY vectors
exactly reproduce the archived 2026-11-16 run, whose wall time was different.

Final candidate workspace counters: 3234 captures, 475 refreshes, 47242 replays,
zero fallbacks. Reuse is too infrequent to justify default promotion here.

## Memory defect found and fixed

The first candidate attempt repeatedly created a new CUDA stream and private
pool when the sparse pattern changed. Reserved allocator memory grew toward
23 GiB despite much lower live tensor memory; the agent stopped only this own
experiment. It is explicitly **incomplete**, retained under `stopped_pool_churn/`.

The fix retains one stream and one registered graph pool across recaptures,
with an anchor graph that is never replayed after replacement. Complete fixed
run maximum reserve: 914 MiB; baseline: 958 MiB. Peak live allocation:
637.06 versus 631.18 MiB. No accumulating reservation was observed across this
full run. Input-buffer budget still does not bound total CUDA memory.
A regression exercises 100 alternating-pattern recaptures; 23 GPU tests passed.

## Numerical comparison

Full `comparison.json` includes every SUMMARY vector, saved field maxima and
step statistics. Across all float32 field snapshots, maximum absolute changes:
pressure **0.432861 bar = 0.0432861 MPa**, Sw 0.0055691, Sg 0.0109879,
So 0.0106657, Rs 10.1831. These are local maxima over the saved history, not
field-average or exclusively final errors. Largest normalized SUMMARY-vector
change is about 2.03% (WGOR:B-1H, normalized by the vector maximum, not pointwise).
Maximum normalized accumulated field-production changes: oil 0.00519%,
water 0.01358%, gas 0.00311%.

First differences in adaptive step durations are at roundoff scale, but later
step decisions differ. A separate one-day checkpoint probe at day 3251 compared
all 34 actual CPR applications against fresh eager evaluation: maximum relative
norm difference 2.04e-17; step converged. This supports a roundoff-sensitive
trajectory explanation at that state, but does not prove the cause of every
later discrepancy or equivalence of all states. No tolerance was loosened to
accept the candidate. Norne quality-preserving equivalence remains unqualified.

## Evidence and reproduction

Run `scripts/validate_norne.py` with a new NORNE_OUT, NORNE_DIR pointing to the
Norne deck, NORNE_DAYS=3312, NORNE_DEVICE=cuda:0, NORNE_THREADS=1,
NORNE_CPR_REUSE_GRAPH=0 or 1, explicit CUDA_VISIBLE_DEVICES UUID and
CUBLAS_WORKSPACE_CONFIG=:4096:8. Then run `scripts/check_norne_run.py` on each
output. Both source archives and hashed manifests are retained. The sole source
difference between completed runs is `src/simulator/cpr_replay.py`, the declared
pool-lifecycle fix; the graph policy flag also differs intentionally.
Raw field snapshots and checkpoints remain at `artifact_locations.json` paths
with field hashes. The comparative script uses those raw snapshots. Audits,
SUMMARY files, step records and execution memory samples are archived here.

A separate five-day synthetic coupled-model CPU-call profile is included under
`coupled_profile*`. It is **not a GPU-kernel profile or a Norne speed result**.
Observed cumulative Python-call times include reservoir assembly 1.67 s,
well/surface assembly 1.56 s and legacy CPR graph calls 0.62 s; nested totals
must not be added. 1209 autograd backward calls and 367 tensor `.cpu()` calls
were recorded. These identify candidates for focused profiling, not proof that
all those transfers or derivatives are unnecessary. Next priority: coupled
assembly/derivative costs and repeated reservoir solves, preserving physical
and nonlinear tolerances.
