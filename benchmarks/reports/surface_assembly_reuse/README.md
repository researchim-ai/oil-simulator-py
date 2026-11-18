# Surface assembly work reduction, preserving evaluated states

Implemented three execution changes without changing equations, tolerances,
quadrature orders, active-set logic or timestep policy:

- Scalar black-oil PVT columns share the current pressure interval lookup.
  Each column retains its own pressure subtraction and original arithmetic,
  preserving the pressure AD accumulation graph. Undersaturated PVTO columns
  likewise share only their current interval indices/table coordinates.
- ComponentStream and StockDensities validation reads dataclass fields directly,
  avoiding recursive `asdict` copies. Validation still includes inherited fields.
- A network solve computes each node's own source masses once and uses them
  again for its independent component balance. One immutable empty stream is
  constructed per solve. There is no cross-trial numerical cache.

## Verification

Value, Jacobian and Hessian comparisons against the original per-column
interpolation are exact on CPU and CUDA, including saturated/undersaturated
states and a breakpoint. CPU property/pipe/network checks: 46 passed, 4 skipped
before adding the CUDA parameterization. Final GPU-selected suite: 40 passed.
Process/thermal/FIM integration suite: 39 passed. Counts overlap; do not add them.

An existing pressure-search-count test failed on the frozen original source as
well as the candidate: its saturated CPU input skips unused PVTO tables, so it
correctly makes one search, not four. The test now explicitly checks saturated
(one search) and undersaturated (four searches) cases. Original failure log is
retained; no physical tolerance or accepted result was changed to fix a test.

Four alternating GPU pairs on the synthetic 65,536-cell/64-well coupled model,
30 days with operating events at days 10 and 20, passed exact comparisons of
final fields, every accepted state, trial traces, active constraints and material
accounts. Both sources used `cpr_reuse_graph=false` and identical configs.
Time pairs, seconds: 7.465 → 6.732; 6.818 → 6.710;
7.921 → 7.774; 8.435 → 7.452. Combined median 7.693 → 7.092 s,
7.8% less time (1.085x). RTX 3090; shared host, not an isolated universal speedup.
The first batch overlapped short local CPU checks; the repeat batch did not.

A stronger CPU case over two days, 42 accepted substeps and 161 reservoir
solutions, also passed exact trajectory/accounting comparison. Timing
19.021 → 19.247 s does **not** establish a CPU speedup.

These month runs request reporting at the interval endpoint, with intermediate
operating events. They take 8 accepted steps and 27 reservoir evaluations;
previous reports using five-day reporting are not direct timing comparators.
This is not a new Norne result, independent physical validation, or evidence of
an order-of-magnitude acceleration.

## Profile evidence

A separate five-day paired CPU-call profile of the GPU case passes the same
exact-state check. Torch interval searches 7110 → 3366; ComponentStream
validation calls 14976 → 9867; mass-rate conversions 5370 → 2796;
`asdict` calls 15217 → 239. Stream-validation cumulative CPU time
0.1464 → 0.0545 s. Whole surface-assembly cumulative timing 1.522 → 1.546 s
in this single profiled pair did not improve; profiling instrumentation and
shared-host variation prevent treating that timing as proof of assembly speedup.
The repeated unprofiled full-case pairs above are the timing evidence.
This profile measures Python calls, not GPU kernel occupancy or synchronization
attribution. Unchanged autograd remains necessary for the implicit sensitivities.

## Reproduction and scope

`before_source.tar.gz`, `after_source.tar.gz`, `source.patch`, test snapshots,
run protocols, configs, source hashes, trajectories and profiles are included.
Use `scripts/benchmark_facility_source_pair.py` with the before-source tree,
`month/config.json`, `month/schedule.json`, `--days 30 --pairs 2 --device cuda
--cuda-memory-fraction .12`, an explicit GPU UUID and a fresh output directory.
Relocate absolute PVT paths when moving the archive. CPU strong-case config and
one paired run are in `strong_pair/`; short profiles in `profile_*`.

Next larger targets need their own profiling/accuracy qualification: implicit
surface derivatives, repeated coupled reservoir solutions, and larger networks.
Experimental persistent CPR remains off, following its failed Norne promotion.
