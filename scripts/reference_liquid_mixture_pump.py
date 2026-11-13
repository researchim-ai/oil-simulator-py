"""Independent NumPy PR allocation/PS/PH audit for dense liquid pump mixtures.

Checks serialized branch equilibrium and work. Global phase stability outside
that branch is a separate runtime multistart TPD check, not certified here.
"""

import numpy as np
from check_stratified_pr import Reference
from check_stratified_pr_flow import close


def mixture_trace(fluid, z, trace, pressure, temperature, minimum_density):
    close(trace["pressure_pa"], pressure, atol=1e-5, rtol=1e-10)
    close(trace["temperature_k"], temperature, atol=1e-7, rtol=1e-10)
    beta = np.asarray(trace["phase_fractions"])
    x = np.asarray(trace["phase_mole_fractions"])
    tags = trace["phase_roots"]
    if (
        beta.ndim != 1
        or not len(beta)
        or len(tags) != len(beta)
        or x.shape != (len(beta), len(z))
        or not np.isfinite(beta).all()
        or not np.isfinite(x).all()
        or np.any(beta <= 0)
        or np.any(x < 0)
        or any(tag not in ("stable", "liquid", "vapor") for tag in tags)
    ):
        raise ValueError("Invalid liquid-mixture pump phase registry")
    close(beta.sum(), 1.0, atol=1e-10, rtol=0.0)
    close(x.sum(-1), np.ones(len(beta)), atol=1e-10, rtol=0.0)
    close(beta @ x, z, atol=1e-10, rtol=1e-8)
    active = np.flatnonzero(z > 0)
    ref = Reference(fluid, active)
    qs = [ref.state(pressure, temperature, xx, tag) for xx, tag in zip(x, tags)]
    mu = np.asarray([np.log(xx[active]) + q["phi"] for xx, q in zip(x, qs)])
    close(mu - mu[-1], np.zeros_like(mu), atol=1e-7, rtol=0.0)
    if min(q["rho"] for q in qs) < minimum_density:
        raise ValueError("Liquid-mixture pump trace includes a non-dense phase")
    result = {key: sum(b * q[key] for b, q in zip(beta, qs)) for key in ("h", "s", "vm")}
    result["rho"] = (
        float(z @ np.asarray([c.molar_mass_kg_mol for c in fluid.components])) / result["vm"]
    )
    return result
