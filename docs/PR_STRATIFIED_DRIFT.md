# Bounded stratified gas/liquid slip

`src/surface/stratified_drift.py` provides `StratifiedGasLiquidClosure`,
`model="stratified_gas_liquid"`. This is an explicitly selected, local,
quasi-steady branch for **two continuous layers** in a circular, near-horizontal
pipe. The caller establishes gas/liquid identities; the lower-density phase is
placed above the higher-density phase. Density contrast alone does not identify
gas. Oil/water, three layers, dispersed phases, annular flow, entrainment,
intermittency, capillary-dominated films, and phase appearance/disappearance are
outside this closure. No automatic regime selection is implemented.

The gas fraction comes from local **volume stocks**, not superficial fluxes or a
holdup correlation. Neither geometry nor thermodynamic stocks are adjusted to
make a prescribed pair of phase fluxes fit.

## Source and constitutive choices

Primary reference: Y. Taitel and A. E. Dukler (1976), *A model for predicting flow
regime transitions in horizontal and near horizontal gas-liquid flow*, AIChE
Journal 22(1), 47–55,
[DOI 10.1002/aic.690220105](https://doi.org/10.1002/aic.690220105),
[requested paper scan](https://electronicsandbooks.com/edt/manual/Magazine/A/AIChE%20Journal%20US/1976%20v.22/01%20%281-207%29/047-055.pdf).
The two-fluid separated-layer force-balance framework is the starting point;
the signed equations below are rederived explicitly for this implementation's
axis convention and mass-average constraint. This implementation does **not**
reproduce the paper's complete transition map, finite-wave transition criteria,
or original friction correlations. The equations below define this implementation
explicitly; they are not a transcription of the original transition map.

Wall turbulent friction uses the Darcy Haaland expression from S. E. Haaland
(1983), *Simple and Explicit Formulas for the Friction Factor in Turbulent Pipe
Flow*, J. Fluids Eng. 105(1), 89–90,
[DOI 10.1115/1.3240948](https://doi.org/10.1115/1.3240948).
Hydraulic-diameter substitution, the laminar/turbulent envelope, viscous
interface term, and long-wave screening described below are declared engineering
choices, not newly attributed Taitel–Dukler equations or experimental validation.

## Coordinates, geometry, and mass frame

All inputs are SI Torch scalars/vectors on one device. Positive x is the pipe
axis. The public `velocities()` API takes `axial_gravity_m_s2=+g*dz/dx`,
positive uphill, exactly like the existing core. Its wrapper negates this before
calling `parameters()`. The diagnostic `parameters()` API instead takes actual
signed gravitational acceleration `g_x=-g*dz/dx`. The independent audit script's
`velocities()` **and** `parameters()` both take the public positive-uphill scalar.
No sign adapter is needed in the parent public closure call.

Let `a_g=V_g/(V_g+V_l)`, `a_l=1-a_g`, `A=pi*D^2/4` and
`m_g=a_g*rho_g/(a_g*rho_g+a_l*rho_l)`, `m_l=1-m_g` (mass fractions).
With slip `s=v_g-v_l`, use

```
v_g = U + m_l*s
v_l = U - m_g*s
```

Then `sum(V_k*rho_k*(v_k-U))=0` by construction, including countercurrent
flow at U=0. U is the imposed **mass barycentric** velocity; it is not the
volume flux `j=a_g*v_g+a_l*v_l` returned in diagnostics. Reversing U and axial
gravity reverses velocities and pressure gradient. At horizontal stop both
phase velocities vanish. On a slope U=0 can have counterflow.

For liquid segment half-angle `theta` in `(0,pi)`, geometry is

```
a_l = (theta - sin(theta)*cos(theta))/pi
A_l = a_l*A;                 A_g = a_g*A
S_l = D*theta;               S_g = D*(pi-theta)
S_i = D*sin(theta);          h_l = D*(1-cos(theta))/2
D_l = 4*A_l/S_l;             D_g = 4*A_g/(S_g+S_i)
```

These are local constitutive cross-section calculations. No parent pipe or
separator geometry is modified. Liquid hydraulic diameter excludes the free
surface; gas hydraulic diameter includes the moving interface. Laminar
hydraulic-diameter wall drag is an approximation for a circular segment, not
an exact cross-sectional viscous solution.

## Signed common-pressure force balance

`tau_wk` has the sign of `v_k`. `tau_i` has the sign of `s`: it acts against
the gas and with the liquid. For `p_x=dp/dx` and axial acceleration `g_x`,
the per-unit-length balances are

```
0 = -A_g*p_x + rho_g*A_g*g_x - tau_wg*S_g - tau_i*S_i
0 = -A_l*p_x + rho_l*A_l*g_x - tau_wl*S_l + tau_i*S_i
```

Eliminating the common pressure gradient gives the scalar root

```
F(s) = (rho_l-rho_g)*g_x + tau_wg*S_g/A_g - tau_wl*S_l/A_l
       + tau_i*S_i*(1/A_g+1/A_l) = 0.
```

The module reports both reconstructed pressure gradients, their difference F,
and their mean. Summing the two force balances cancels interface drag exactly.
The diagnostic pressure gradient is not a second pipe momentum equation or an
instruction to replace the parent's pressure unknown.

`gas_dynamic_viscosity_pa_s` and `liquid_dynamic_viscosity_pa_s` are mandatory,
positive constructor fields. They are assigned after density sorting, so core
call sites need not supply a viscosity context. Optional `phase_viscosities_pa_s`
overrides both with live tensors **in input phase order**. Supply real dynamic
viscosities from an appropriate property model; no density-to-viscosity conversion
is used. Live tensor overrides retain derivatives. The parent must reject wrong
gas identities using EOS/compressibility role checks before density sorting;
density contrast is only an ordering rule, not a phase-role classifier.

With `Re=rho*abs(v)*D_h/mu`, `Re_t=max(Re,2300)`, and explicitly supplied wall
roughness `epsilon=wall_roughness_m`, wall drag is

```
f_D = [-1.8*log10((epsilon/(3.7*D_h))^1.11 + 6.9/Re_t)]^-2
tau_w = v * max(8*mu/D_h, rho*f_D*abs(v)/8)
```

The laminar term gives a finite linear drag at stop; the maximum is a monotone
engineering transition envelope. Derivatives are branchwise; do not assume a
classical second derivative at laminar/turbulent branch ties.

Interface drag is explicitly parameterized by positive, mandatory
`interfacial_friction_factor=f_i` in the **Fanning** convention:

```
tau_i = (8*mu_g/D_g)*s + (f_i*rho_g/2)*s*abs(s).
```

The first term represents an approximate gas-side viscous shear scale; the
second is specified turbulent/interface drag. `f_i` must be chosen/calibrated
for the intended interface. No wave roughness is inferred and wall roughness
is not silently reused as interface roughness. The liquid and gas wall terms
and interface term each dissipate energy: `tau_wk*v_k>=0`, `tau_i*s>=0`.

## Bounded root and derivatives

F is strictly increasing on the declared friction envelope:

```
dF/ds = tau'_wg*m_l*S_g/A_g + tau'_wl*m_g*S_l/A_l
        + tau'_i*S_i*(1/A_g+1/A_l) > 0.
```

Thus any bracketed root is unique for fixed stocks and geometry, including
reversal and stop. A fixed-count device-local bisection locates the root in
`[-maximum_slip_m_s,+maximum_slip_m_s]`. A root outside the bracket is rejected;
a clipped endpoint is never accepted as a balanced solution. Two live Newton
corrections recover first and second implicit derivatives from the detached
seed: `ds/dq=-F_q/F_s`. Segment geometry uses the same method with the exact
slope `da_l/dtheta=2*sin(theta)^2/pi`. No host synchronization, SciPy solver,
or differentiation through bisection choices is used in the runtime path.
`torch.func.jacfwd`, `jacrev`, nested forward AD, and `vmap` are tested.
Second derivatives at exactly zero signed quadratic drag need not exist.

## Envelope and KH rejection

Defaults are gas volume fraction 0.02–0.98, gas/liquid density ratio at most
0.2, at most 5 degrees from horizontal, slip at most 100 m/s, and each phase
speed at most 100 m/s. The angle envelope can be explicitly configured up to
10 degrees, and the gas-fraction envelope must stay within `[1e-4,1-1e-4]`.
Roughness divided by either hydraulic diameter must not exceed 0.05. These
are implementation restrictions, not universal experimental boundaries.
All volumes, densities, viscosities, diameter, and gravity must be positive
and finite. The two-layer solve requires exactly two phases (three or more raise
`ValueError`). Public `velocities()` provides the existing one-phase limit
`v=U, valid=True` without requiring geometry context; diagnostic `parameters()`
remains a two-phase routine.

The additional inviscid, rigid-lid **long-wave** screen uses
`g_n=sqrt(g^2-g_x^2)` and equivalent layer depths `H_k=A_k/S_i`:

```
s_critical^2 = (rho_l-rho_g)*g_n*(A_g/rho_g + A_l/rho_l)/S_i
kh_margin_m2_s2 = s_critical^2 - s^2
```

For clarity, this follows by requiring real wave speeds c in the linear
hydrostatic two-layer relation
`rho_g*(c-v_g)^2/H_g + rho_l*(c-v_l)^2/H_l = (rho_l-rho_g)*g_n`.
Its discriminant gives the stated slip bound. Nonpositive margin is rejected.
`long_wave_stable=True` means only that this particular discriminant is
positive. Finite wavelengths, finite-amplitude waves, viscosity-induced
instability, surface tension, entrainment and slug initiation are not analyzed;
passing this screen does **not** certify physical stability or global
hyperbolicity of the coupled thermodynamic transport system. In particular it
is not the full Taitel–Dukler finite-wave criterion. No alternative regime is
selected on rejection.

## API and parent integration handoff

```python
from surface.stratified_drift import StratifiedGasLiquidClosure

closure = StratifiedGasLiquidClosure(
    interfacial_friction_factor=0.01,  # illustrative, not a calibration
    wall_roughness_m=1e-5,
    gas_dynamic_viscosity_pa_s=1.8e-5,
    liquid_dynamic_viscosity_pa_s=1e-3,
)
v, valid = closure.velocities(
    phase_volumes, phase_densities, U,
    phase_viscosities_pa_s=dynamic_viscosities,  # optional live override
    diameter_m=diameter,
    axial_gravity_m_s2=+g * elevation_change / length,  # core convention
    gravity_m_s2=g,
)
diagnostics = closure.parameters(
    phase_volumes, phase_densities, U,
    phase_viscosities_pa_s=dynamic_viscosities,  # optional live override
    diameter_m=diameter,
    axial_gravity_m_s2=-g * elevation_change / length,  # actual acceleration
    gravity_m_s2=g,
)
```

`valid` is a scalar device boolean combining `state_valid`,
`inclination_valid`, `root_valid`, and `long_wave_stable`. Rejected inputs
return diagnostic candidates; callers must check validity before using them as
physical states. Safe numerical surrogates for invalid arguments do not expand
the validity envelope. Extreme overflowing inputs are not guaranteed finite
candidates. There is no stateful continuation or hidden accepted fallback.

Core integration is implemented in `PREquilibriumTransportNetwork` and its
pipe builder/factory. Accepted two-phase states require one dense-liquid and
one gas EOS root (compressibility factors straddle 0.5). This threshold is
a bounded implementation screen, not a universal phase-identity classifier. The common momentum
residual includes the total wall reaction `-L*(tau_wg*S_g+tau_wl*S_l)`;
interfacial shear cancels internally. A second mixture wall viscosity is
rejected to avoid double-counting. The one-phase limit uses the appropriate
stored phase viscosity and ordinary Darcy wall loss. Stationary wall friction
remains internal heating in the total-energy balance, not an external energy
sink. The common EOS/flux characteristic checks still apply.

`surface_pr_stratified_drift.json` exercises two finite cells, a live material
connection, isolation, and reopening. The independent transport and gravity
auditors reconstruct both slip and total wall force. The common Jacobian is
checked against finite differences, and a checkpoint is continued independently.
This is a bounded constitutive branch, not a complete regime selector.

## Dedicated verification

Run `PYTHONPATH=src:scripts pytest -q tests/test_surface_stratified_drift.py`.
The suite checks an independent NumPy/SciPy pressure-balance reference,
mixture momentum cancellation, dissipative signs, circular segments, mass
frame conservation, phase permutations, horizontal stop, slope counterflow,
axis reversal, first/second derivatives against finite differences, forward
and reverse AD agreement, batched Jacobians, viscosity requirements, explicit
roughness/interface sensitivity, and domain/KH/bracket rejection. CPU float32
and float64 are checked; the same dtype checks run on CUDA when available.
Numerical consistency is not experimental calibration or coupled PR integration
validation.

The independent `scripts/reference_stratified_drift.py` imports only math, NumPy,
and SciPy. `velocities(closure_dict, alpha, rho, bulk, diameter, axial_g, gravity,
phase_viscosities_pa_s=None)` accepts input-order volume fractions or stocks
and densities and the **positive-uphill** scalar. It returns NumPy velocities
and raises `ValueError` on rejected states. `parameters` has the same arguments
and returns diagnostics, including `valid`, for KH/envelope inspection. It uses
independent scalar circular geometry and a SciPy Brent force-balance solve.
