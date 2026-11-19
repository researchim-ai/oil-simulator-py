# Current independent surface reference checks

Command (CPU, one Torch thread):

```
PYTHONPATH=src:scripts python scripts/check_surface_reference_suite.py \
  --include-compressor --require-local-temperature --output-dir /tmp/surface-references
```

All five selected jobs passed. The suite required the local stagnation-temperature
criterion, in addition to the spatial refinement criteria.

- PR: 148 checks against archived independent CoolProp PR results passed. This
  verifies the implementation with declared caloric conventions, not real-fluid
  experimental accuracy for every composition.
- Isothermal and thermal acoustics: three spatial/time resolutions each passed.
  Thermal pressure L1 errors decreased 3.361 → 1.803 → 0.884 Pa.
- Symmetric compression: Rankine–Hugoniot reference, 32/128/512 cells. At 512
  cells the stagnation-temperature error is 0.003625 K, below 0.02 K; pressure
  error at the centre is 95.84 Pa. This is a frozen homogeneous two-phase model,
  not a validation of phase slip or experimental flow-regime correlations.
- Compressor train: analytical choked ideal-gas blowdown with 0.04/0.02/0.01 s
  steps. Mass error decreases by factors 1.991 and 1.996; finest error is
  0.0003821 kg and temperature error 0.01429 K. Manufactured partial/full
  recycle equilibria preserve inventory, energy and shaft speed.

The compressor reference checks equipment coupling and energy accounting under
ideal-gas assumptions, not calibration of real compressor maps. These are
separate component/subsystem benchmarks, not a complete independent field-network
comparison. All per-job reports, commands, logs and hashes are in this directory.
No production equations or tolerances were changed in this run.
