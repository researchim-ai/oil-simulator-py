# Signed actual-flow pump connections

`PRPumpConnection` selects its bidirectional mode only when `signed_map` is an
explicit `SignedPumpMap`. The Pump card must have `check_valve: false`.
`mapped_speed_rad_s` optionally specifies a fixed signed speed; otherwise the
card speed ratio multiplies the map reference speed. A connected rotor supplies
its evaluation speed directly, including zero and negative values. Forward NPSH
curves are rejected in signed mode. Explicit vapor pressure and its temperature
can still limit the live donor state.

The supplied example coefficients are **synthetic**. They exercise the equations;
they are not a field-validated map. Each application needs calibrated coefficients
and appropriate flow, speed, density, acoustic, and property limits.

For map coupling C = a |omega| + c:

- H = C omega - b q |q|, oriented from the named start to end.
- T = rho g (C q + d omega), the actual signed shaft load at every speed.
- Pshaft = T omega; Phydraulic = rho g H q.
- Pshaft - Phydraulic = rho g (d omega² + b |q|³) >= 0.

Positive q donates from start; negative q donates from end. Composition, density,
enthalpy, entropy, temperature, and sound speed come from that live donor.
The PS trace targets specific reversible work g sign(q) H minus the change in
potential energy. The pump PH trace targets g sign(q) C omega minus that same
potential change. The delivery trace throttles isenthalpically into the receiver.
The oriented outlet loss is R q |q|. `pump_pressure_pa` denotes the intermediate
pressure on the actual receiver side; `valve_pressure_drop_pa` is signed in the
fixed start-to-end orientation.

Both physical port energy fluxes include their own elevations. Viscous loss
rho g d omega² enters the actual donor as heat, without division by flow or
speed. Consequently the net connection energy addition is exactly the same
signed shaft power reported to the rotor/work ledger, even at zero flow.
At zero speed and nonzero flow, torque may be finite while shaft power is zero.
The entropy diagnostic includes donor viscous heat divided by donor temperature.
Acceptance checks the live donor and PS/PH liquid states, absolute-head acoustic
compression, entropy production, map bounds, and absolute shaft rating.
Accepted trace validation also flashes the live donor to check phase stability.

At exactly zero flow the start is the donor convention. Upwinding is piecewise
smooth: use directional Jacobian checks at the switching surface. A centered
finite difference crossing donors is not a derivative for unequal port states.
Disabled connections isolate both ends and report zero work and load.
The legacy path is unchanged when `signed_map` is absent.

## Integration and compatibility

- Normalized dictionaries omit `signed_map` and `mapped_speed_rad_s` when `None` during
  model serialization, preserving legacy schema and model identity.
- Mapped pump connection names are included in signed rotor initialization, bounds,
  drive evaluation and independent rotor audit. Consume `load_torque_nm`
  directly at zero speed. Do not apply legacy free-coast logic to signed maps.
- Active component supports are united in both directions for open signed connections,
  including transitions where a donor contains previously absent components.
- The bundle owns signed initial flow and receiver-side trace pressure guesses;
  there is no additional core guess hook.
- The connection reference audit independently reconstructs the live donor sound
  speed; it does not require a core audit decoder change.

Dedicated tests cover all flow/speed quadrants, zero speed/flow, donor changes,
common Newton Jacobian and sparsity, conservation and independent tamper checks.
