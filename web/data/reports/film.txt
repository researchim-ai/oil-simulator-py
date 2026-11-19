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
