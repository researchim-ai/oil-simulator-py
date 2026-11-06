# Explicit reversible dry-gas connection

Opt in with `PRCompressorUnit.signed_map: SignedGasMap` from
`surface.pr_compressor_gasmap`, with `card=None`. Legacy forward cards retain
their equations and record schema. This is a specified, bounded, quasi-steady
dry-gas constitutive map, not surge dynamics or a universal measured curve.
Coefficients must be supplied explicitly; reverse behavior is never inferred.

The map includes nominal `reference_speed_rad_s`, `maximum_speed_rad_s`,
`maximum_flow_m3_s`, `euler_coefficient` a, `linear_coupling` c, and
`flow_loss_coefficient` b. Speed and geometric volumetric flow Q are signed
along source-to-target. Both absolute speed and absolute Q are bounded.
C=a|omega|+c; h_iso=C*omega-b*Q*|Q| is specific J/kg directly.
Load torque is rho*C*Q, shaft power is torque*omega, and
Pshaft-rho*Q*h_iso=rho*b*|Q|^3 >= 0. Coefficients a, c, b respectively have
units m², m²/s, and 1/m⁴ (radians dimensionless). There is no gravity factor.
Windage belongs to the finite drive, never to a heat term divided by Q.

Each bundle entry has **three unknowns**: signed Q, T_is/Tref, T_out/Tref.
The entry tuple layout and offset at index 3 are unchanged. Both PS and PH
use receiving pressure. PS imposes h_is-h_in=MW*sign(Q)*h_iso and s_is=s_in;
PH imposes h_out-h_in=MW*sign(Q)*C*omega. Internal b accounts for all passive
loss. Additional outlet losses require a separate network branch.

Q selects the physical donor, including density and composition. Fluxes use
that composition at both physical ends, with donor enthalpy at the donating
end and delivery enthalpy at the receiving end. The receiving inventory's
composition does not replace the donor gas. Donors and both traces must pass
the dry-gas, reference-composition, compressibility and stable vapor checks.
Entropy production must be nonnegative. At Q=0 all fluxes and external work
vanish, even though shutoff head can be nonzero. Q=0 selects the source donor.

`CompressorBundle(..., donor_signs={name: +1 or -1})` optionally freezes a
branch for Newton; negative sign*Q makes that branch inadmissible. A parent
solver can try the two branches separately when a donor composition jump
prevents convergence. Default selection follows Q. Inventory guesses do not
invent reverse rates from forward cards; parent active-support logic must
permit new donor species in both directions.

The graph and drive integration registers signed compressors and pump
connections for bidirectional rotor domains and component support. Legacy
card speed guards apply only to legacy cards. The three-row block and
sparsity allocation remain unchanged. Legacy serialization omits
`signed_map=None`; this compressor change introduces no other new serialized
defaults. SignedGasMap fields exist only in opt-in cards. The optional
`donor_signs` argument is solver state and is not serialized.

The independent audit dispatches through `audit_signed_compressor` in
scripts/reference_pr_compressors.py. It checks speed, Euler power, donor
PS/PH, torque, loss, entropy, and both physical-end fluxes independently.

Run the complete synthetic two-plenum facility with:

```sh
python scripts/run_pr_equilibrium_transport.py configs/surface_pr_signed_compressor.json --output /tmp/signed-compressor.json
python scripts/check_pr_equilibrium_transport.py configs/surface_pr_signed_compressor.json /tmp/signed-compressor.json
```

The facility has parallel machines with opposite geometric orientation and
opposite signed rotor speeds. Both compress dry gas from delivery to suction:
the forward unit has positive Q and the reverse unit negative Q. Their finite
drives include viscous drag, motor losses and regeneration limits. The three
short schedule intervals exercise operation, isolation and reopening. The two
plenums start with slightly different gas compositions inside the explicit
reference envelope. This checks signed orientation, not a transient surge or
a single rotor crossing through zero speed. Unit tests separately cover
zero flow, zero speed, reversed donors, reference guards, JacFD and independent
audit rejection of damaged records.

The smaller configs/pr_compressor_signed_unit.json remains a standalone
unit-card example, not a runner input.
