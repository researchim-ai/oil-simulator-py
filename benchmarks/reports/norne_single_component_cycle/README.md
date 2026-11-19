# Norne: single-component Newton cycle and final correction acceptance

## Reproduced cause

On the fixed step starting at day 2510, gas CNV alternated approximately
0.0108769 and 0.0666838. The main compact cell was 28291 (full grid I/J/K =
23/39/17, one-based). Sw alternated 0.9560843 ↔ 0.9674580, pressure changed
by about 1355.5 Pa. Its phase flags stayed `sat=False, ug=True`: this was
not repeated phase switching. Adjacent compact cells 28290 and 27051 also
contributed. Oil and water component equations had already met the requested
criteria, but the oscillation detector required two oscillating components.
Consequently it never damped this single remaining unconverged component.

The detector now retains the two-component rule when several components are
unconverged, but allows detection when only one remains. No tolerance,
constitutive law, phase switch, linear solver or time-grid change is involved.

## Second termination defect

With detection fixed, relaxation activates at iteration 16 (omega=0.9),
and gas residual decreases. The former loop nevertheless failed at its
iteration limit: it applied the final permitted correction and assembled
its residual, but never checked that resulting state for acceptance.
The loop now checks convergence after the final allowed correction and
refuses any further linear solve at the budget. Tests verify that the number
of linear solves does not exceed the budget and that a genuinely insufficient
budget still rejects and rolls back physical state.

## Verification protocol

20 focused CPU/CUDA tests passed: detector regressions, an actual black-oil
solve with exactly its required correction budget and one less, fluid/Jacobian
regressions, adaptive-AMG retry behavior. The early and late Norne windows
use identical prescribed substep grids, checkpoint, physical settings and
acceptance criteria to the earlier accuracy experiment. Four variants per
window: ordinary AMG, adaptive AMG, CNV=1e-4 and CNV=1e-5 references (MB=1e-8).
Well criteria are unchanged. Each completed run checks component balance,
finite/bounded fields, exact time coverage and source/input integrity.

The pre-fix fine reference failed at day 2510 even with 80 corrections in
the prior experiment. The detector-only intermediate candidate broke the
cycle but exposed final-correction acceptance; its failed run is retained in
`after/`. That directory is not the final implementation. Completed final
runs are named `fixed_*`, and `change.patch` contains both retained fixes.

These tests concern numerical convergence at the existing discretization,
not external physical validation or a full nine-year performance claim.

## Final result

All nine post-fix window runs completed with 17 prescribed steps each and
no timestep subdivision: eight regular/refinement regressions plus the late
CNV=1e-5 / MB=1e-9 reference that previously failed. The critical fine-reference
step (2510, dt=9.66368 days) used exactly 20 corrections; gas CNV after the
last correction was **5.50882e-6**, max MB **6.08783e-11**, cumulative component
balance error below **8.2e-9**. No tolerance or correction budget was relaxed.

For all six previously successful variants (both ordinary/adaptive windows
and the two early refined references), saved final fields, SUMMARY arrays,
step clocks and Newton counts matched exactly. See `regression.json`.
All completed runs verify numerical/source/input invariants; `completed_runs.json`
records actual settings and work. Timings are individual instrumented runs,
not a new paired speed benchmark. The full 3312-day case was not rerun.

Late ordinary/adaptive max errors against the fine reference are now about
40.97/40.90 Pa and Sg 2.126e-5/2.432e-5. Differences between the two refined
references reach 55.20 Pa and Sg 8.413e-6; a general accuracy ranking remains
unjustified. Adaptive AMG remains experimental and off by default.
This fix does not claim to address the separate early-window MB plateau
at approximately 1.16e-9 or every possible nonlinear convergence failure.
