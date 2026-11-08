"""Scalar PR plus high-precision independent finite-inertia reference."""

from decimal import Decimal, localcontext

import numpy as np
from check_stratified_pr import Reference
from check_stratified_pr_flow import close


def decode(fluid, state, roots):
    if len(state["phases"]) != len(roots):
        raise ValueError("Reference phase registry differs")
    mw = np.array([c.molar_mass_kg_mol for c in fluid.components])
    output = []
    for stock, tag in zip(state["phases"], roots, strict=True):
        if set(stock) != {
            "component_moles",
            "volume_m3",
            "pressure_pa",
            "temperature_k",
            "momentum_kg_m_s",
            "total_energy_j",
        }:
            raise ValueError("Unknown/missing phase checkpoint fields")
        n = np.asarray(stock["component_moles"], dtype=float)
        if n.shape != mw.shape or not np.isfinite(n).all() or np.any(n < 0) or not n.sum() > 0:
            raise ValueError("Invalid phase component stocks")
        p, t = stock["pressure_pa"], stock["temperature_k"]
        if (
            not fluid.min_pressure_pa <= p <= fluid.max_pressure_pa
            or not fluid.min_temperature_k <= t <= fluid.max_temperature_k
        ):
            raise ValueError("Reference EOS domain exceeded")
        ref = Reference(fluid, np.flatnonzero(n > 0))
        q = ref.state(p, t, n / n.sum(), tag)
        mass = n @ mw
        velocity = stock["momentum_kg_m_s"] / mass
        volume, internal, entropy = [n.sum() * q[k] for k in ("vm", "u", "s")]
        close(volume, stock["volume_m3"], atol=1e-14, rtol=1e-9)
        close(internal + 0.5 * mass * velocity**2, stock["total_energy_j"], atol=2e-8, rtol=1e-10)
        output.append(
            {
                "n": n,
                "mass": mass,
                "velocity": velocity,
                "volume": volume,
                "internal": internal,
                "entropy": entropy,
                "p": p,
                "t": t,
            }
        )
    return output


def momentum(masses, velocities, dt, drag, wall):
    """90-digit original phase equations, not the runtime mean/slip solve.

    Resolves finite impulse at vanishing slip without copying the runtime
    coordinate transformation. Gaussian elimination is bounded to 2/3 phases.
    """
    with localcontext() as ctx:
        ctx.prec = 90
        cast = lambda x: Decimal(str(float(x)))
        m, v, w = [list(map(cast, values)) for values in (masses, velocities, wall)]
        k = [list(map(cast, row)) for row in drag]
        h = cast(dt)
        n = len(m)
        matrix = [
            [-h * k[i][j] if i != j else m[i] + h * (sum(k[i]) + w[i]) for j in range(n)]
            + [m[i] * v[i]]
            for i in range(n)
        ]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(matrix[r][col]))
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            factor = matrix[col][col]
            if not factor:
                raise ValueError("Singular reference phase momentum matrix")
            matrix[col] = [q / factor for q in matrix[col]]
            for row in range(n):
                if row == col:
                    continue
                factor = matrix[row][col]
                matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[col])]
        end = [matrix[i][-1] for i in range(n)]
        pair = [[-h * k[i][j] * (end[i] - end[j]) for j in range(n)] for i in range(n)]
        wall_impulse = [-h * w[i] * end[i] for i in range(n)]
        physical = [
            h * (sum(k[i][j] * (end[i] - end[j]) ** 2 for j in range(n)) / 2 + w[i] * end[i] ** 2)
            for i in range(n)
        ]
        numerical = [m[i] * (end[i] - v[i]) ** 2 / 2 for i in range(n)]
        return {
            "velocity": np.array(list(map(float, end))),
            "coordinates": np.array(
                [
                    float(sum(a * b for a, b in zip(m, end)) / sum(m)),
                    *(float(q - end[0]) for q in end[1:]),
                ]
            ),
            "pair_impulses": np.array([[float(x) for x in row] for row in pair]),
            "wall_impulse": sum(map(float, wall_impulse)),
            "physical": np.array(list(map(float, physical))),
            "numerical": np.array(list(map(float, numerical))),
        }
