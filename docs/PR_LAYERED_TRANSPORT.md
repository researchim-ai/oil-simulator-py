# Inclined profile-cell Euler transport

`src/surface/pr_layered_transport.py` implements a native chain of straight,
coaxial, constant-diameter cells. Each cell contains its actual finite
`InclinedCylinderGeometry` gravity profile, one shared temperature, one shared
axial velocity, integrated component inventories, one axial momentum `M u`,
and total energy `U + PE + M u²/2`. Layers have instantaneous local gravity
and chemical equilibrium with flat horizontal interfaces. This is a bounded
shared-velocity closure: there is no layer inertia, phase slip, transverse
momentum, variable area, or lateral-port support. Exactly vertical cylinders
are outside this transverse-disk implementation.

## Joint equations and geometry

`StratifiedPRColumn._problem` supplies the collocation chemistry and interface
pressure equations, live integrated N/U/PE, and profile variables. The native
residual preserves those equations, replaces the final component/energy rows
with backward-Euler balances, and adds one momentum row per cell. It appends
pressure/composition trace variables at every physical face quadrature point.
Trace chemistry uses that cell's temperature and chemical constants at the
same absolute height on both sides. No residual calls flash, `equilibrate`,
`sample`, or another Newton solve. Acceptance calls `_accept` to validate the
joint profile iterate against the flux/heat balance targets.

At each shared axial disk, intersections of every left and right phase
interval with the disk span `D cos(theta)` partition the face. All phase pairs,
including zero-area pairs, remain allocated during a Newton solve. Moving
interface heights change the live intersection weights. Angular Gauss
quadrature integrates disk strips. Exact strip area and first height moment
corrections protect the subtraction of large pressure integrals; very thin
strips bypass the moment correction to avoid cancellation. One HLLC fan per
common quadrature point supplies N/momentum/E flux; the integrated flux is
scattered with opposite signs. Frozen homogeneous acoustic speeds are checked
against the configured fan bound. This is not an equilibrium-mixture sound
speed claim or a proof of the reduced closure's hyperbolicity.

The momentum source is the current cell profile's own right pressure integral
minus its own left pressure integral. It is independent of the Riemann flux.
Each closed exterior patch uses a reflected HLLC fan: the ghost retains
species, density, pressure, internal energy, kinetic energy and potential
energy, reverses axial momentum/velocity and species/energy flux, and retains
momentum flux. Left walls use ghost/interior and right walls interior/ghost.
The resulting momentum traction is stored separately in `wall_pressure`
(integrated force, N, not Pa). Its contact has zero velocity; N/E and entropy
transfer are exactly zero, with cancellation roundoff projected out. At rest
this equals the own-profile pressure integral; during motion its acoustic
reaction damps the cell momentum and conserves total energy as kinetic energy
converts to internal energy. `pressure` remains exclusively the thermodynamic
integral for the gravity source; it is never replaced by wall-fan pressure.
`wall_pressure` is zero on open ends and non-exterior cell faces, and the global
`wall_force_n` uses its left-minus-right traction.
Reports compare the pressure source to `-M g sin(theta)` from the independent
volume inventory and reject a quadrature mismatch larger than
`1e-7 N + 1e-4 |M g sin(theta)|`. Energy includes gravitational potential and
kinetic energy; gravity needs no extra total-energy source. Local numerical
datums reduce elevation cancellation. `source_frozen: false` means this source
is evaluated from the current Newton iterate, including derivatives.

## Declared effective-mixture wall friction

The saved optional `dynamic_viscosity_pa_s >= 0` and `0 <= roughness_m < D/2`
(defaults both zero) enable a stationary-wall Darcy force. With positive
viscosity, each cell uses
`friction_force_n = -V * darcy_pressure_loss(u, M/V, 1, D, mu, roughness)[0]`.
Zero viscosity explicitly disables the term, including when roughness is set.
This is a declared effective-mixture approximation using the existing laminar,
Haaland, and transitional-blend implementation. No per-phase viscosity table,
wetted-layer shear model, or universal multiphase friction closure is inferred.

Friction enters only the momentum balance: total E remains conserved, so the
lost kinetic energy becomes profile energy and contributes to entropy. Signed
`friction_force_n` is separate from gravity `source_n` and reflected-wall
`wall_pressure`; the global momentum ledger sums all three plus boundary
transfers. The NumPy reference uses the existing independent `reference_darcy`.
Tests prescribe synthetic viscosities for analytic laminar damping checks,
compare to the inviscid reflected-wall case, verify both velocity signs,
rest invariance, N/E closure, entropy increase, model round-trip, parameter
rejection, and finite-difference derivatives. These test viscosities are not
estimates for the PR mixtures. The config serializes the two zero defaults;
the native auditor also normalizes omitted zero defaults in raw model input.

## Initialization, phase changes, and time stepping

`initialize(profiles, velocities_m_s=None)` attaches existing cell profiles.
`initialize_equilibrium(P,T,x)` creates one gravity-equilibrated cylinder
covering the chain and restricts its absolute chemical constants to each
actual cell. This produces consistent two/three-layer rest states across
cells and faces instead of initializing neighboring profiles independently.

Phase topology is fixed within each Newton. On convergence/domain/stability
rejection, `step(..., heat_w=...)` can obtain a new `_seed` from predicted
conserved N and total E minus predicted kinetic energy and try another joint
Newton with different profile/trace dimensions. The original stocks are the
balance targets in every retry. There is no post-step re-equilibration of an
accepted state. Retries are bounded and can fail; there is no universal phase
transition guarantee. Tests demonstrate closed-cell heat-driven birth and death.

`advance(state, duration_s, ...)` compares one coarse step with two half steps
in N, datum-relative E, momentum and interface heights. It accepts the fine
state, adjusts the next step, and halves rejected steps. The serializable
`LayeredTransportState` contains cells, time, model identity, and next step.
Adaptive failures raise rather than returning a partial success. Ordinary
phase support must initially agree between native cells.

## Sparse Newton

Saved `dense_limit` (default 256) chooses the Torch dense/sparse path;
`max_unknowns` (default 16384) bounds allocation. Unknowns and residuals both
have the order all profiles, all velocities, then all traces. Physical blocks
combine each profile with its velocity and keep each local/shared trace block
separate. Blocks are disjoint, exhaustive, and bounded by 256 unknowns.

The conservative structural pattern includes each cell and adjacent cells,
all local/adjacent face traces in its balance rows, and both participating
bodies plus the own trace block in shared chemistry rows. It never infers
zeros from a resting numerical Jacobian. Tests check off-pattern derivatives
and compare moving two-cell dense/sparse solutions. This is a sparse-capable
release kernel, not a large-plant scaling benchmark; the configured maximum
is a bound rather than demonstrated production capacity.

## Public layered plant bridge

Multiple named chains, shared junctions, rings and endpoint isolation use
[`PRProfileGraphNetwork`](PR_PROFILE_GRAPH.md). The single-chain API below
remains supported with its existing model and checkpoint schema.

`PRLayeredPlantNetwork` in `src/surface/pr_layered_plant.py` joins a native
profile chain to an existing `PREquilibriumTransportNetwork` through actual
finite vessels. Its public `initialize`, `step`, `advance`, `to_dict`, and
`from_dict` methods carry a `LayeredPlantState` containing the ordinary and
layered states with common time and model identity. The ordinary graph retains
its branches, loops, machine/drive equations, walls and heat links;
finite vessels provide real-stock junctions at the profile-chain ends.

The plant's coupling protocol appends the native unknowns to the ordinary
problem, preserves the ordinary sparse graph pattern, and accepts both state
families from the same Newton iterate. Adaptive stepping compares coarse and
two-half solutions, with synchronized restart data and combined conservation
ledgers. This extends the ordinary graph through the bounded profile-chain
component; it does not introduce arbitrary profile-cell geometries or slip.
The plant runner and independent auditor are
`scripts/run_pr_layered_plant.py` and `scripts/check_pr_layered_plant.py`.

The underlying kernel method
`joint_vessel_problem(ordinary, ordinary_problem, state, dt_s,
 boundary_nodes=(left_name, right_name), ...)` returns initial/lower/upper and
one common residual. Supply the tuple returned by the existing ordinary
`PREquilibriumTransportNetwork._problem`, reserving the corresponding
`external_ports`. All ordinary unknowns are followed by native profile,
velocity, and trace unknowns. The ordinary evaluate's `external_fluxes`
callback evaluates the native block and returns the opposite transfer. This
method does not solve either subsystem sequentially.

Only existing finite HEM `vessel` nodes are allowed, with uniform P/T and a
zero-velocity boundary packet. The default guard is `rho g D/P <= 1e-3`;
callers cannot relax it above 0.01. A vessel's elevation must match its disk
center. Vessel EOS and frozen-phase acoustic bounds are checked. Native end
wall forces are removed at connected ends, while the own-profile source is
retained. One HLLC fan is used per physical boundary patch. N/E transfer is
opposite after datum conversion. Vessels absorb momentum in supports; the
returned `vessel_support_force_n` is the signed momentum delivered to the
vessel support, and multiplying by dt gives its impulse. No vessel momentum
stock is introduced. Ordinary slip and pipe endpoints are rejected.

The kernel bridge has a common-residual finite-vessel driven/FD test. The
public plant wrapper publishes the combined state through ordinary acceptance
and native `accept_solved`; the kernel method itself returns a problem bundle.
`accept_solved(state, dt, native_y, joint_data["profile"], heat_w=None,
external_ends=(False, False), evaluations=0)` validates and publishes that same
native iterate without another solve. Its global N/E/momentum ledger subtracts
actual boundary transfers, and wall forces omit open ends. It reports native
entropy change minus boundary entropy influx, and checks adiabatic production.
For nonzero heat, `heat_entropy_validated` is false and entropy production is
null. The public plant wrapper accepts explicit
`profile_utility_temperature_k` with `profile_heat_w`, accounts for supplied
heat entropy, checks passive heat direction, and validates profile entropy
production before reporting heat entropy as validated. Boundary record `entropy_rate` is signed into the
ordinary vessel; native exchange `incoming_entropy` is its negative sum, and
the ordinary exchange already provides the opposite `incoming_entropy` scalar.

`predict_seeds(state, dt, native_data, heat_w=None)` returns balance-target
phase seeds for a new joint Newton. It rejects component-support changes and
does not modify accepted stocks.

The public plant driver wires this balance-target predictor into bounded
profile-topology retries. Native component support remains fixed; unsupported
new inlet components are rejected. Ordinary phase/support handling continues
through the existing graph driver. Release-case evidence is required for each
machine/topology configuration; the bridge's structural graph compatibility
is not a universal validation claim.

Native decoded data and accepted reports expose `pressure`, `wall_pressure`,
and the integrated `wall_force_n`; decoded `wall_faces` retain patch fans and
accepted `wall_fluxes` contain the closed-wall momentum-only transfers. The
independent native reference returns `wall_pressure`, and its auditor uses it
in momentum balances while continuing to check the own-profile gravity source.
Moving closed gas/two-phase tests cover both velocity signs and one/two cells,
N/E closure, momentum damping, positive entropy, internal-energy growth, and
agreement with the independent NumPy reflected fans.

The lower-level native `_problem(..., external_ends=(left_open,right_open))`
evaluate accepts `external_fluxes(decoded)`. `decoded` exposes live `bodies`,
`velocities`, and `boundary` patches (`cell`, `side`, `phase`, `heights`,
`weights`, and HLLC `packet`). Return `net` in integrated N/momentum/E rates
per native cell, `valid`, and any extra diagnostics. The packet energy uses
`model.datum`. The evaluator exposes `profile_slices`, `velocity_slice`,
`trace_slices`, `sparsity`, and `linear_blocks`. The joint bridge evaluator
exposes `ordinary_size` and `profile_evaluate`; its data contains `ordinary`,
`profile`, and vessel support forces. The bridge preserves `ordinary_evaluate.sparsity` when supplied and uses
ordinary problem offsets to restrict cross-edges to the connected vessel body
and corresponding native end profile/velocity and local boundary traces. A
conservative dense ordinary subblock is used only if that metadata is absent.
`maximum_update` preserves the ordinary prefix and appends unbounded native
updates; native primitive bounds still apply.

## CLI and independent audit

```sh
PYTHONPATH=src python scripts/run_pr_layered_transport.py \
  configs/surface_pr_layered_transport.json --output /tmp/layered.json
PYTHONPATH=src python scripts/check_pr_layered_transport.py \
  configs/surface_pr_layered_transport.json /tmp/layered.json
PYTHONPATH=src pytest -q tests/test_surface_pr_layered_transport.py
```

The config has `model`, `initial`, and `schedule`. Initial conditions can be
`equilibrium`, explicit per-cell P/T/x `cells` (plus `velocities_m_s`), or a
serialized `state`. Schedule entries accept duration, per-cell heat rates,
step bounds and adaptive tolerances. Every accepted half step includes its
state and conservation/source diagnostics. CLI execution hashes source files
and refuses a run if sources changed; `sources_unchanged_during_run` is
separate from the live-source `source_frozen` field.

`reference_pr_layered_transport.py` uses NumPy disk quadrature, the existing
independent NumPy PR reference, SciPy chemical traces, and independent scalar
HLLC. `check_pr_layered_transport.py` independently reconstructs profiles,
face fluxes, own end-pressure sources, and each cell's N/E/momentum balance
including heat and wall forces, and checks checkpoints and source provenance.
It audits the native closed-chain CLI family; it does not audit the plant
bridge. These checks are numerical verification, not experimental validation.
