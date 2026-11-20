# Analytic PR derivatives and audited warm phase flashes

This change accelerates thermodynamic calculations. It does not qualify the
whole detailed facility as a fast production model.

## Implementation

- Analytic local composition tangents replace repeated general AD in the
  CPU chemical Newton kernel. Analytic constrained Gibbs curvature runs on
  CPU/CUDA, with optional CPU and default CUDA kernel fusion.
- The root tangent differentiates the same detached cubic-root Newton
  correction used by the existing EOS, including its residual correction.
  Mechanical/caloric root selection and final full-property audits remain.
- Phase allocation iterations evaluate chemical properties only; full caloric
  properties are completed for the final physical result. Newton polishing
  uses eager chemical values and the final acceptance uses full eager EOS.
- Transaction-scoped previous phase allocations provide conservative initial
  guesses at nearby P/T/composition. They are not approximate cached results.
  The new allocation preserves every component total. Equal fugacities,
  positive stocks, phase distinctness, full multistart TPD, material balance
  and Gibbs energy are checked. A failed guess falls back to the cold solve.
- Model tensor versions invalidate hints. Hints are bounded, own their storage
  and are removed at transaction exit. No old phase count is imposed after
  a phase transition. Exact result caching remains separate.

Sources consulted for the approach:
[analytic cubic/PR composition derivatives in Thermo](https://thermo.readthedocs.io/thermo.eos_mix.html#thermo.eos_mix.PRMIX.dlnphis_dzs),
[scoped warm starts in NeqSim](https://equinor.github.io/neqsim/development/performance_tuning.html).
The derivative implementation here is local Torch tensor algebra; it does not
introduce a Java/native thermodynamics dependency or copy another solver's
phase-stability short-circuit policy.

## Paired measurements

`final_flash_sequences.json` measures eight distinct scalar PT requests in each
sequence, including its first cold flash. Compilation warmup is outside timing;
no exact-result cache hits occurred in the measured sequences. Both paths use
the same model, states, tolerances and complete stability tests.

- Two-phase hydrocarbon VLE: 44.55 → 8.23 ms per request, observed 5.41×.
  Maximum phase-composition difference 3.10e-10; phase-fraction difference
  4.73e-10; scaled bulk-property difference 8.25e-10.
- Synthetic three-liquid mixture: 175.91 → 47.50 ms, observed 3.70×.
  Composition difference 1.91e-14; fraction difference 1.12e-16;
  scaled bulk-property difference 2.66e-14.

Seven of eight calls used successfully audited warm allocations in each
sequence. These are single paired observations on a shared host, not a
statistical performance guarantee or external experimental validation of the
synthetic three-liquid model. `flash_sequences.json` is an earlier pre-lean-
allocation observation and is kept separately.

`final_trial/comparison.json` compares the original integrated facility's first
coarse/fine attempt: 30.17 → 21.48 s, observed 1.404×; both physical states and
error 22.27425 match exactly. The attempt is rejected by temporal control.
Its performance ratio is not the flash-kernel ratio and is not a completed
operating-cycle speedup. The empty earlier `first_trial` directory had no
accepted evidence: the source guard rejected that measurement after edits.

## Verification and current operating-cycle check

87 CPU tests passed, including independent AD/finite-difference derivatives,
trace fractions, two/three-phase flashes, cache invalidation, cold fallback,
caloric identities and phase-group derivatives. A separate CPU/CUDA suite
passed 72 tests on RTX 3090, including compiled GPU derivatives and independent
CoolProp/symmetric-mixture equilibrium references. These suites overlap;
the counts must not be added as unique tests.

`recorded_prefix.accepted.jsonl` combines unchanged physical records from the
previous two solver journals through 0.1065991401 s. Each segment retains its
source provenance. `accelerated_launch.json` identifies the bounded continued
cycle using the new solver. Restart across numerical source changes is explicit:
the runner independently audits the prefix, validates the model/checkpoint,
and records old/new source hashes. The original complete 1 s schedule and its
future isolation/restart commands remain the final acceptance target.

Until its result reports complete-cycle qualification, the whole-cycle check
is pending. No claim of production-speed whole-network simulation, temporal
refinement or external experimental qualification follows from these kernel
measurements.

`accelerated_prefix.json` independently audits the new accepted interval
0.1065991401–0.1665991401 s: six split halfsteps, 52 local records and six slow
records. Component residual magnitudes are below 1.46e-11 mol; energy residual
is -6.7479e-8 J. Macrosteps are 0.02 s and pass unchanged temporal control.
This is an accepted connected-network prefix, not the completed cycle.
