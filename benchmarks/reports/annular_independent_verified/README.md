# Independent diagnosis of the annular experimental discrepancy

The current interface-velocity annular closure was rerun on all 16 unchanged
Hannover input windows. Experimental pressure qualification still fails:
MAE approximately 58.84 / 48.28 Pa for the two predefined roughness baselines.
No fit, changed samples or widened acceptance bounds were used.

An independent scalar calculation now solves these exact measured-input states:
radial SciPy quadrature, a separate interface-slip Brent solve, and a separate
thin-film-root Brent solve. It imports no runtime mobility or friction formula.
The production Torch solution is compared only after solving the reference.
All 16 pass fixed numerical budgets of 1e-8 in holdup and 1e-5 Pa over the
3 m pressure span. See diagnosis.json for every window and experiment.json for
raw-data hashes and the unchanged external scoring.

This rules out numerical error of the tested magnitude in the steady film
equations as an explanation for the tens-of-pascals mismatch. It does not prove
which missing physical effect dominates. Full circumferential wetting is not
confirmed; the measured pressure span includes liquid application; waves and
entrainment are not identified by this steady fully-wetted closure.

The experimental gate remains FAILED. An alternative closure needs independent
applicability evidence and validation data; modifying coefficients to pass these
same windows would not establish predictive accuracy. No production physics
or time-step tolerance was changed by this diagnosis.

```
PYTHONPATH=src:scripts python scripts/diagnose_annular_experiment.py \
  --output-dir /tmp/fresh-annular-diagnosis
```

Exit zero means the numerical diagnosis passed, not experimental qualification.
The two statuses are separately recorded. Existing radial BVP, force balance,
energy, thin-film limits and dynamic-source checks: 28 passed, 2 CUDA skips.

## Zero-film limit audit (20 November 2026)

`scripts/audit_annular_dry_limit.py` isolates the turbulent zero-film offset:
Wallis' Fanning baseline .005 tends to Darcy .020, while the dry branch uses
Reynolds number and wall roughness. The production thin-film limit is checked
in `tests/test_annular_dry_limit_audit.py` without using measured pressures.

On the same 16 windows, the 3 m offset is -1.88..-1.26 Pa for a smooth wall,
and -13.20..-10.42 Pa for the declared .2 mm roughness sensitivity. Subtracting
this offset algebraically changes MAE from 58.84/48.28 Pa to 60.37 Pa in both
cases. This is NOT a re-solved alternative closure or an accepted validation
score. It rules out this baseline difference alone as the explanation of the
experimental discrepancy. See `dry_limit_audit.json`; qualification stays false.
No production coefficient, equation, roughness or experimental bound changed.

The phase-disappearance friction mismatch and the experimental accuracy of the
fully wetted model are separate unresolved issues. A replacement interfacial
law needs an explicit domain, dry-limit consistency, and independent validation;
subtracting the diagnostic offset from runtime pressure is not such a law.

## Applicability and independent dry-reference follow-up

`applicability.json` records a re-solved superficial-gas-velocity sensitivity,
separate dry-run comparisons and conditional contact-angle thresholds from the
existing wetting criterion. It does not fit material properties, alter runtime
closures, or change qualification. The superficial variant still fails all 16
instrument-only pressure comparisons (MAE 34.96 Pa). The existing interface
variant remains 58.84 Pa. Neither dry reference series explains the discrepancy.

`author_dry_reference.json` reads 13 cached author dry-air comparison points
from the additional unchanged source workbook, authenticated against the
original archive. The independent dry CFD curve differs from the legacy dry
baseline by -4.55..+8.13 Pa over the original 20 wet windows. It is a separate
campaign/model reference, not replacement measured truth. Full-domain pressure
and pure distributed friction must not be conflated.

Reproduce with `scripts/audit_annular_applicability.py --output PATH` and
`scripts/audit_hannover_air_reference.py --output PATH`, using
`PYTHONPATH=src:scripts`. Their exit zero means diagnostic completion, not
experimental qualification; every report keeps qualification false.

The runtime steady solver now exposes the source superficial-velocity law
explicitly. `superficial_reference.json` is the matched experimental challenge,
not a replacement for the interface-source gate. `comparison_contract` records
what is and is not reproduced. The previously used `held_out=true` label was
incorrect for repeatedly inspected development data and is now false in fresh
reports. Historical report bytes are retained as provenance, not rewritten.

`surface_suite/summary.json` records the completed combined external suite on
unchanged source hashes after adding the superficial reference. Its exit code
is 1: the experimental gate remains false. All 17 VLE states were computed;
Hannover and the existing FETRig qualification requirements remain unmet. The
new reference gives 34.96/24.40 Pa MAE for the two predeclared dry baselines.
No runtime gate was replaced with this lower-error diagnostic result.

`tap_span.json` adds a conditional contraction/pipe/expansion measurement
operator. Its known terms and assumptions are recorded separately. Maximum
dry discrepancy versus author CFD is 17.66%, so the rig is not fully identified.
For every matched wet window the model overpredicts the incremental drop;
a positive added injection resistance cannot remove the 25.99..44.63 Pa
residual. This excludes that specific one-term repair, not all possible
wet/dry changes in local flow structure. No negative loss was inserted.


The additional workbook audit now reads explicit geometry from
`Uebersicht_Experiment!S4:T5`, rather than inferring it solely from velocities.
Those velocities remain an independent consistency check. Its instrument notes
(C65:C70), acquisition dates and unconfirmed wet-campaign equivalence are
included in `author_dry_reference.json` and `tap_span.json`. The September 2024
manometer reading uncertainty and changed inlet connection are not silently
transferred to the original wet/dry data or used to widen the original gate.

`fetrig_middle_station.json` adds six public FETRig cases at Y=350 mm (19
input-steady windows), acquired with all six nominal gas settings before scoring.
It includes both gas-unit interpretations and source hashes. The no-gas
conditional thickness difference is −0.78% to −0.45%; this is not an experimental
qualification because units, clock pairing and sensor effects remain unresolved.
Re-run the common external suite to include this station; historical suite files
above predate this addition and are retained as snapshots.

The runtime now also offers `laminar_annular_colebrook_implicit_v3` with
final-state resistance. See `docs/SURFACE_CONFIG_REFERENCE.md` and
`configs/surface_pr_active_surface_annular_implicit.json` for a reproducible
coupled example. This temporal-integration improvement does not change the
experimental qualification status of Colebrook/Wallis.

`implicit_source_network.json` is a source-bound six-substep coupled example
using the implicit source. Recheck it with:

```
PYTHONPATH=src:scripts python scripts/check_pr_active_surface.py configs/surface_pr_active_surface_annular_implicit.json benchmarks/reports/annular_independent_verified/implicit_source_network.json
```

A later source change intentionally invalidates that snapshot's fingerprint;
regenerate via `scripts/run_pr_active_surface.py` to a new output before checking.

`implicit_source_speed.json` compares the former 56-bisection paired source
with the 16-iteration safeguarded Newton source at identical synthetic inputs.
On CPU, 256 cells / 9 timing repeats gave 2.67x source-only acceleration and
maximum final velocity difference 1.98e-13 m/s. This excludes PR recovery,
transport and network work. Reproduce with `scripts/benchmark_annular_implicit.py`.
`implicit_source_network.json` has been regenerated and independently checked
with the optimized source; the six-substep energy residual remains 4.55e-13 J.

Current geometry-reuse measurements:

- `implicit_source_speed_geometry.json`: 256-state source-only comparison,
  3.77x faster than the former 56-bisection implementation, max velocity
  difference 1.98e-13 m/s. `implicit_source_speed.json` is the older measurement
  before invariant geometry was reused and is kept as a historical snapshot.
- `parallel_source_benchmark/`: four pipes / eight cells connected to one common
  vessel through finite ports, 3 microseconds of simulated time, six substeps.
  Complete-run medians 3.023 s (bisection) and 2.314 s (optimized Newton),
  identical final states and energy residual 1.82e-12 J. Both full traces passed
  independent checking. This short prototype has a single-phase vessel stock
  and two-phase pipes; it is not a whole-field or long-horizon performance claim.
- `gas_profile_contract.json`: 160 public **dry-channel** velocity observations.
  No liquid interface or confirmed wall-zero coordinate is present, so this
  source cannot qualify a wet interfacial shear law.

Recheck the parallel results with `scripts/check_pr_active_surface.py`, passing
`parallel_source_benchmark/request.json` and either complete network report.
The one-pipe `implicit_source_network.json` has also been regenerated with the
current geometry-reuse source.
