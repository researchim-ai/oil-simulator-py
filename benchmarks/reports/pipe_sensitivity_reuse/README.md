# Reuse converged pipe integrals and evaluated port states

The batched pipe solver previously discarded its converged Newton integral,
then repeated the same adaptive quadrature with all inputs differentiable to
obtain implicit sensitivities. It now builds the complete local input vector
on each trial, differentiates only the root view for Newton proposals, and uses
the accepted trial tape for the final full implicit Jacobian. Rejected tapes
are explicitly released before the next trial. No numerical state is cached
between solves; quadrature tolerances, pressure residuals, choking checks and
mechanical balances are unchanged.

The scalar pipe reuses the full two-port flash already evaluated for the
subcritical acceleration check when building reports. Both endpoint admission
checks remain. Its final sensitivity backward no longer retains quadrature
tensors for a second backward that never occurs; subsequent accounting only
detaches the already computed quantities.

## Correctness and measured scope

The accepted-tape Jacobian matches a freshly evaluated full-input integral
exactly on CPU and CUDA. Tests include heterogeneous gas/oil/water rows, moving
phase boundaries and collapsed quadrature intervals. Port reports match separate
public flash evaluations exactly for both pressure-solution directions, mixed
phases and pure water. This does not expand the model to physical reverse flow.
Final GPU suite: 46 passed. Additional CPU quadrature checks and full source-pair
forecasts pass; test counts across logs overlap and must not be added.

Two opposite-order 64-pipe CUDA microbenchmark pairs reduce checked integrals
from **5 to 4 per solve**. All report fields are exactly equal. Combined median
**42.975 → 40.519 ms**, 5.7% less time. Additional peak live tensor allocation
is effectively unchanged: 3,684,352 → 3,683,840 bytes. This is a homogeneous
Rs-only synthetic pipe batch, not a field-scale network or a GPU-kernel-only
measurement.

The stronger coupled CPU case spans 2 days, 42 accepted substeps and 161
reservoir solutions. Opposite-order pairs are **21.525 → 20.208 s** and
**19.944 → 18.916 s**; combined median **20.734 → 19.562 s**, 5.7% less time.
Every accepted state, trial trace, active constraint and material account
compares exactly with the source snapshot from the start of this work.

**Full GPU coupled acceleration is not established.** The final 30-day,
65,536-cell/64-well GPU pairs give 7.726 → 7.957 s median. An earlier version
before explicitly dropping rejected tapes gave 7.760 → 7.296 s; its outputs
also match exactly, but these different source versions/timing batches are not
pooled into a claimed speedup. All runs use a shared host. Synthetic cases report
at the interval end with operating events; these times cannot be compared to
previous five-day reporting protocols or Norne.

## Evidence

`month_final/`, `strong/`, `strong_reverse/`: final source-guarded audited runs,
configs, trajectories and exact comparisons. `month/`: preliminary version
before explicit rejected-tape cleanup. `batch_*`: all GPU microbenchmark
samples, reports and memory counters. `port_micro_comparison.json` shows that
port reuse alone did not establish a speedup; the final mechanism also reuses
the accepted integral. `before_source.tar.gz`, `after_source.tar.gz`, incremental
patch and test snapshots are included. `qualification.json` distinguishes
measured local/CPU benefits from the unconfirmed full GPU benefit.

Reproduce full-case pairs with `scripts/benchmark_facility_source_pair.py`, the
before-source tree, archived config/schedule, a fresh output and explicit GPU
UUID. Use 30 days/CUDA for the month, 2 days/CPU for the strong case. Adjust
absolute PVT paths when relocating. Microbenchmark drivers retain the recorded
workspace paths; `pvt.json` is the common synthetic input. No commit was made.

The separate `pipe_transport_experiment` was rejected and is not in production.
Persistent CPR remains off. Next larger performance targets require profiling
of the complete GPU-coupled path and reducing repeated reservoir evaluations;
these tests do not establish independent physical calibration or Norne speed.
