# Explicit signed finite-cell pump closure

`PRPumpUnit.signed_map` opts a homogeneous finite pump cell into `SignedPumpMap`
from `surface.pr_pump_four_quadrant`. Legacy cards retain their forward-only
acceptance domain. The signed map and rotor are integrated into the common PR
Newton and checkpoint model.

The map is a reduced constitutive model with user-supplied coefficients, not a
validated industrial performance dataset. A speed/flow rectangle declares its
acceptance domain; there is no extrapolation acceptance or inferred reverse
compressor performance. Runtime calculations use Torch and support autograd.

Positive q is left-to-right flow, positive h raises pressure left-to-right,
and positive T is torque absorbed from the shaft. For signed angular speed w:

    h = (a*abs(w) + c)*w - b*q*abs(q)
    T = rho*g*((a*abs(w) + c)*q + d*w)
    Pshaft = T*w
    Phydraulic = rho*g*h*q
    Pshaft - Phydraulic = rho*g*(d*w*w + b*abs(q)**3) >= 0

Coefficients a, b, c, d are nonnegative. Their fields and units are respectively
`euler_head_coefficient_m_s2`, `flow_loss_coefficient_s2_m5`,
`linear_head_coefficient_m_s`, and `viscous_torque_coefficient_m4_s`.
The optional c term permits flow-driven torque at rest. With c=0, the closure
cannot represent hydraulic starting torque. Head and torque are continuous
with finite first derivatives at zero speed and zero flow; no efficiency
quotient or division by speed defines the map. Forward map/efficiency/NPSH
curves on the legacy Pump are not reverse calibration. Signed mode rejects
legacy NPSH requirements rather than claiming those certify reverse operation.

For prescribed signed operation, set `mapped_speed_rad_s` on the unit.
Otherwise speed is map reference speed times the Pump speed ratio. A coupled
drive overrides prescribed speed and must have nominal speed equal to the map
reference speed. Its rotor residual consumes the explicit map load torque.
Finite-cell momentum receives area*rho*g*h and total energy receives Pshaft;
dissipation converts mechanical energy internally and is not extra external
heat. Regenerative shaft work is negative. Shaft power rating applies to its
absolute value. Both endpoint stocks and the pump body must be single-phase
dense liquid, and the absolute g*h/a_sound² compressibility bound applies.
If vapor pressure is specified, every stock must remain above it in the stated
temperature range. This is a domain check, not a cavitation/NPSH kinetic model.

Map tests use synthetic coefficients solely to test signs, finite derivatives,
dissipation, envelopes, serialization, finite-cell conservation, and
fluid-driven rotor acceleration. They do not validate industrial operating
accuracy. Calibrate head AND torque over the intended signed envelope before
using the closure for engineering predictions. The constrained formula does
not reproduce every pump's measured four-quadrant topology.

## Dynamic signed shafts and independent audit

`RotorBundle` infers reversal capability only from a finite-body pump's
non-null `signed_map`. Those units receive symmetric +/- drive speed bounds,
explicit load torque through rest, and no positive-midpoint gate. Legacy pumps
and compressors retain nonnegative bounds, their positive running-speed gate,
and the existing isolated stop event. `PRMachineDrive` accepts a finite initial
speed within the absolute drive rating; network inspection determines whether
that machine is permitted to rotate in reverse.

An isolated signed shaft follows I*dw/dt = torque - drag*w through zero without
an artificial stop. Entire-function integrals give its exact angular travel,
motor work and drag heat, including zero drag. Open shafts retain the common
Newton midpoint balance and its discrete kinetic-energy identity. Electrical
rating checks use the signed endpoint products torque*speed. On a zero crossing,
electrical energy accounts separately for positive motor work divided by motor
efficiency and negative motor work multiplied by regeneration efficiency. Using
net mechanical work alone would miss losses when a step both draws and exports.
For open shafts this split integrates the linear midpoint speed trajectory;
for isolated shafts it integrates the exact affine trajectory. The physical
meaning of positive/negative torque is fixed by the drive's signed torque limits.

Checkpoint capability checks:

- `PREquilibriumTransportState.__post_init__`: rotor scalars may be any finite
  signed values; this unbound state does not know machine capabilities.
- `PREquilibriumTransportNetwork.inspect`: enforce abs(speed)<=drive maximum,
  allowing negative speed only for names in `self.pumps` with `signed_map`.
  Keep compressors and legacy pump/connection shafts nonnegative.

No other core integration is needed for finite-body reversal. Pump units already
receive midpoint shaft speed and the rotor/pump contributions already share the
Newton system and energy ledger. The drive nominal speed must match the map
reference speed. Restart retains signed inventories.

`configs/surface_pr_signed_pump_reversal.json` is a synthetic closed finite-stock
example: higher right-plenum pressure produces negative fluid flow; negative
motor torque reverses an initially positive open shaft, positive torque reverses
it again in hydraulic isolation, and reopening with negative torque reverses it
once more. These are transient verification cases, not industrial data.

Run and independently audit it with:

    python scripts/run_pr_equilibrium_transport.py configs/surface_pr_signed_pump_reversal.json --output /tmp/pr_signed_pump_reversal.json
    python scripts/check_pr_equilibrium_transport.py configs/surface_pr_signed_pump_reversal.json /tmp/pr_signed_pump_reversal.json

`reference_pr_pumps.py` recomputes signed head, torque, power and dissipation
using scalar arithmetic; `reference_pr_drives.py` independently checks signed
rotor inventories, open torque balance, isolated motion by SciPy ODE integration,
and electrical energy by numerical quadrature split at zero speed. Its adjusted
`machine_model` sets explicit `mapped_speed_rad_s`, preserving the legacy card's
positive `speed_ratio`. Audits reject altered torque/dissipation reports.

`PRPumpConnection` remains a forward-only PS/PH/check-valve model. Reverse
flow there needs direction-dependent donor composition/enthalpy, both sets
of liquid trace equations, signed valve loss and unknown bounds, and a
consistent head/torque energy closure. Simply relaxing its q>=0 bound is not
physical. Use the finite body representation for the signed pump feature.

Legacy forward pump configurations omit unused signed-map fields during
serialization, preserving their checkpoint model identity. Speed PI setpoints
remain nonnegative; signed reversals in this example use explicit drive-torque
commands. Reverse compressor maps are not inferred from forward data.
