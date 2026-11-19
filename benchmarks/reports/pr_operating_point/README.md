# Stationary single-phase PR vessel initialization

The new initializer uses the existing gravitational profile, finite aperture, PR properties and wall heat exchange equations. It solves zero component/energy accumulation directly, without advancing physical time. An independently solved transient step verifies the candidate. This is fixed-feed vessel initialization, not a coupled reservoir operating point or a multirate integrator.

The manufactured case has a 0.1 m diameter, 2 m tall vertical vessel, methane/ethane feed [3.5, 7.4] mol/s at 1.04 MPa and 330 K, and outlet conductance 3e-6 m3/(s Pa). The outlet's original 5% pressure-drop domain remains enforced. This differs from the earlier 20 m tall coupled pilot; it is not a rerun of that entire system.

`result.json` contains the full model, old guess, new initial state, rate balances, inventory changes, transient verification and full-step/two-half-step comparisons:

- Old initial state, dt=0.02 s: pressure difference 39.8953 Pa.
- Stationary state, dt=0.02 s: zero observed pressure difference; component difference approximately 1.1e-14 mol.
- Stationary state, dt=1 s: pressure difference 3.49e-9 Pa; temperature difference 1.93e-12 K; component difference 9.1e-13 mol.
- Scalar PR/32-node aperture audit accepted the verification step; energy residual approximately 4.07e-10 J.

Initialization took approximately 0.88 s on CPU. Timings are exploratory, not isolated comparative performance measurements. The initial conditions differ by design. This is evidence that stationary initialization removes this artificial startup, not evidence that real startups or valve closures can use 1 s steps.

Twelve CPU tests passed across operating-point initialization and existing gravitational flow tests. Tests check unchanged input state, zero-clock initialization, rate balance, a separate 1 s hold, rejection of live states and closed systems, and existing dynamic conservation/phase behavior.

Limitations: single vessel, one phase, identical active species in feed/guess, specified steady feeds, no solved well-pressure feedback, no experimental calibration. No automatic inventory reset in a running simulation. Multirate reservoir/surface integration remains unimplemented by this change.
