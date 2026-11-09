"""Batch the existing scalar differentiable PR kernel, preserving root semantics.

Only present physical phases are passed to this function. Validation uses two
host reductions per batch, not per phase; vmap executes the same EOS algebra.
"""

import torch

from .liquid import HydraulicConvergenceError, PropertyModelError


def homogeneous_batch(fluid, pressures, temperatures, compositions, roots, *, derivative_order=1):
    p, t, x = map(fluid.tensor, (pressures, temperatures, compositions))
    count = len(roots)
    if (
        count < 1
        or p.shape != (count,)
        or t.shape != (count,)
        or x.shape != (count, len(fluid.names))
    ):
        raise ValueError("One P/T/x and explicit root per present phase required")
    if any(r not in ("liquid", "vapor", "stable") for r in roots):
        raise ValueError("Unknown homogeneous batch root")
    domain = (
        torch.isfinite(p).all()
        & torch.isfinite(t).all()
        & torch.isfinite(x).all()
        & (p >= fluid.min_pressure_pa).all()
        & (p <= fluid.max_pressure_pa).all()
        & (t >= fluid.min_temperature_k).all()
        & (t <= fluid.max_temperature_k).all()
        & (x >= 0).all()
        & ((x.sum(-1) - 1).abs() <= 1e-10).all()
    )
    alpha = 1 + fluid.kappa * (1 - torch.sqrt(t[:, None] / fluid.tc))
    if not bool((domain & (alpha > 0).all()).detach().cpu()):
        raise PropertyModelError(
            "Equilibrium batch pressure, temperature or composition outside domain"
        )
    result = {}
    for root in ("liquid", "vapor", "stable"):
        positions = [i for i, r in enumerate(roots) if r == root]
        if not positions:
            continue
        index = torch.tensor(positions, dtype=torch.long, device=fluid.device)

        def kernel(pp, tt, xx, root=root):
            return fluid.tensor_homogeneous(pp, tt, xx, root, derivative_order=derivative_order)

        values = torch.vmap(kernel)(p[index], t[index], x[index])
        for name, value in values.items():
            if name not in result:
                result[name] = value.new_zeros((count, *value.shape[1:]))
            result[name] = result[name].index_copy(0, index, value)
    if not bool(result["valid"].all().detach().cpu()):
        raise HydraulicConvergenceError("No nonsingular mechanically stable PR root in batch")
    return result
