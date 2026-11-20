# Exact thermodynamic kernel acceleration and facility diagnosis

The paired first coarse/fine trial of the original integrated facility took
33.2739 s with the row-wise reference and 23.6800 s with native batching,
local reverse derivatives, compacted BFGS histories and opt-in fused CPU
chemical kernels (1.405× in this single observation). Both resulting states
match exactly in the original error norm; both temporal errors are 22.27425.
This rejected first trial is a performance/equivalence probe, not a successful
transient calculation. See `fused_comparison/comparison.json` and its source
hashes and compressed states. Timing includes trial work, excludes model
initialization, and is not a statistical speedup claim on a dedicated host.

No EOS, phase-stability starts, curvature checks, final full-property acceptance
or physical tolerances were removed. CPU fusion is explicit opt-in; CUDA keeps
its existing compiled route. Input/model sensitivities bypass detached fused
trials. Compilation failures are reported and retain exact eager computation
on the same device. Native local composition derivatives store B*(C−1)^2
entries, not a dense derivative across all spatial samples.

`initial_hydraulics.json` is a read-only checkpoint diagnosis. The recycle is
0.2 m long and 1 mm in diameter, with 0.157 mL inventory volume, initial speed
zero and about 20.1 kPa between its finite vessel boundaries. Frozen acoustic
transit is 0.559 ms; the configured 4000 m/s wave bound corresponds to 0.05 ms;
a laminar momentum relaxation estimate is 2.904 ms. These screening scales
do not establish hydraulic equilibrium or replace multiphase characteristics.
The original transient must not be silently replaced with a steady initial
state or a larger pipe.

The locally subcycled request in
`configs/surface_pr_integrated_facility_subcycled.json` keeps original stocks,
geometry, connections, commands and tolerances. Its requested startup,
isolation and restart still total one second. The bounded fused attempt logged
13 accepted macrosteps through 0.0101432 s, then execution ended with code 143
before the validator serialized a trajectory. This is incomplete, not a pass;
`bounded_fused_observation.json` describes only logged progress. The earlier
`subcycled_cycle.json` was an intentionally interrupted diagnostic before CPU
fusion was installed, not a completed cycle either.

For continued qualification, `scripts/run_pr_checkpointed_facility.py` journals
each accepted pair of physical halfsteps and the exact restart state. Recovery
runs the independent physical audit on the accepted prefix and exports the
remaining schedule without skipping isolation/restart commands. Process-level
interruption cannot erase all accepted work. A complete integration and a
complete qualified operating cycle are reported separately.

The full one-second detailed operating cycle, its refinement and external
experimental qualification remain open until their actual checks pass.

The journaled run was deliberately stopped after 0.00025 s to correct JSON
normalization in the recovery utility; solver sources were unchanged. Recovery
then passed the independent audit: four split halfsteps, 176 local records,
four slow records, energy residual 4.4194e-8 J. See
`checkpointed_prefix.json` and its trajectory. Its remaining request preserves
the remaining startup interval, isolation and restart. This confirms durable
accepted-step recovery and the short physical prefix, not the full cycle.
