# Live EPANET network comparison

Eight steady operating snapshots were calculated with our current LiquidNetwork
and the locally built OpenWaterAnalytics EPANET 2.2 shared library. All passed.
This is a fresh external solver run, not comparison only against stored outputs.
The archive checksum is checked; the library and current source hashes are saved.

Cases: ring, reversed branch orientations, isolated bypass, oil booster,
mapped pump/valve, throttled valve, restored valve, parallel pumps with a blocked
weak pump. The restoration is a new steady solve, not a transient/restart test.

Maximum differences across cases: flow 0.0001594 kg/s; hydraulic head
0.01420 m; pump shaft power 0.1166% (normalization has a 1 W floor).
No open/closed state mismatches. Each case uses its existing archived-test
flow/head budget; new operating snapshots inherit the parent case budget.
Pump power budget is 0.2%. The original budgets were not relaxed.

## Reference convention and fixes

EPANET hydcoeffs.c uses the rounded local-loss coefficient 0.02517 with feet
and cubic feet per second. Production uses K*v²/(2*g), g=9.80665 m/s².
`--match-si-local-loss` exports K multiplied by
`8*0.3048/(pi²*9.80665*0.02517)`. This is a dimensional conversion derived
from the reference source, not a fit. Viscosity and fluid density are matched.
Distributed pipe friction still reflects EPANET's gravity convention.

Before this explicit conversion, the new throttled-valve case differed by
0.01402 m against its 0.01 m budget. Afterwards the difference is 0.000228 m.
The first run is retained in ../surface_live_epanet_matrix. The checker also
now handles zero power for blocked pumps, checks shaft power and branch states,
and rejects non-finite comparisons. Production hydraulic equations were unchanged.

## Reproduction and limits

```
PYTHONPATH=src:scripts python scripts/validate_surface_epanet_matrix.py \
  .scratch/epanet_220_20261031 --output-dir /tmp/fresh-epanet-matrix
```

Output directory must be new. The reference root contains the verified source
archive, source_manifest.json and build/lib/libepanet2.so. See summary.json and
per-case input, EPANET report, comparison and logs.

This qualifies steady incompressible single-liquid hydraulic comparisons only.
It does not qualify gas compression, phase separation, thermal/multiphase
transients, a full coupled Norne run, or real equipment calibration.
The eight-case matrix is not a throughput benchmark. Existing hydraulic,
pump, control and map tests: 86 passed on CPU.
