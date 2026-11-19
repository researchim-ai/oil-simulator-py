# Explicit engineering reduction of selected connections

The production API now lives in `surface.pr_model_reduction`; the old
`reduce_pr_recycle.py` functions remain compatibility imports. Multiple
explicitly selected, simple horizontal pipe cells can be replaced by existing
Darcy connections. Components, total energy and volume are relocated into
finite receiving vessels, followed by UV flash. Initial momentum must be zero.
Unselected dynamics and equipment/control definitions are preserved.

This changes the physical model. It removes selected line inertia and spatial
storage; it does not merely accelerate the same equations. Scheduled heat,
variable pipe resistance, control/physical references, geometric boundary
states, varying port areas and partial isolation are rejected. The map reports
the original port model and the reduced boundary assumptions. No automatic
selection or hidden tolerance relaxation was introduced.

## Integrated facility, 0–0.001 s

`result.json.gz` contains the complete run; independent audit passed. Five
macro attempts were accepted, giving ten accepted fine records, in163.8434 CPU
seconds. The previously archived locally subcycled detailed run took369.59 s
with the same endpoint and original tolerances. These are separate shared-host
runs and different physical models, not a controlled kernel speed benchmark.

Endpoint differences in common finite vessels/pipe nodes:

- maximum pressure:0.25008 Pa;
- maximum temperature:2.3458e-5 K;
- maximum grouped component inventory:5.4856e-6 mol;
- maximum grouped total energy:0.003193 J.

Grouped inventories include the removed detailed pipe in the corresponding
reference receiver. These close inventories do NOT establish matching flow:
the detailed recycle mean velocity is0.265812 m/s, while the reduced algebraic
velocity is25.104276 m/s. The original detailed boundary model is `riemann`;
the reduced connection uses pressure-driven ideal-entry/mixed-exit assumptions.
The frozen-density laminar inertia estimate is2.904 ms, longer than this
1 ms interval. Both startup and boundary-model differences require separate
investigation. This result does not qualify interchangeable startup models.

## Settled single-phase benchmark

`settled/` reruns the current reduced calculation to0.02 s and independently
re-audits archived, request/hash-matched detailed trajectories with1/4/8 cells.
Those detailed reference trajectories were reused, not rerun or retimed.
The new reduced run took3.3068 CPU seconds. Relative outlet flow differences
are2.808%,0.679%,0.322% respectively. Against8 cells, endpoint differences are
0.02721 Pa,2.195e-6 K,7.026e-7 mol,0.001203 J. All predetermined budgets passed.
This qualifies one synthetic pure-methane case after startup, not all mixtures,
multiphase regimes or the complete integrated facility.

Final combined suite:70 CPU regressions passed in118.40 s, covering
subcycling, explicit boundary selection, conservative reduction, Darcy
connections and existing surface control. No GPU timing or experimental calibration is claimed.

`comparison.json` is reproduced with `scripts/compare_pr_engineering_mode.py`;
it deliberately reports diagnostics rather than manufacturing an acceptance
verdict. `source/` records the implementation after adding explanatory mapping
metadata; physical equations were unchanged by that metadata addition.

## Paired boundary-condition experiment on the facility mixture

Two isolated three-node runs start from exactly the same original finite gas
vessels and pipe inventory. Same fluid, geometry, tolerances and1 ms duration;
only `reservoir_ports` changes. `riemann` gives0.265802316 m/s in79.47 CPU s;
`stagnation_single_phase` gives9.071310255 m/s in265.20 CPU s. Both independent
transport audits pass (170 and320 accepted fine records). These results
separate the boundary effect from the rest of the facility. They do not
establish steady-state equivalence at1 ms or a universal choice of boundary
model for all geometries and phase regimes.

The existing stagnation physics can now be selected explicitly in a detailed
subcycled region using `fast_reservoir_ports`. No default changes; slow graph
ports remain unchanged; the choice changes the model identity and is included
in independent audit reconstruction. This permits physically declared
comparisons without quietly changing the boundary of the whole installation.
