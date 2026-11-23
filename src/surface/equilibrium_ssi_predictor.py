"""Damped fugacity successive substitution before expensive TPD fallbacks.

The fixed-point update is v_new = mu_diff - ln(phi)_diff. No component,
original start or acceptance test is removed. Failed predictors leave the
original start and original fallback available. This is a predictor, not a
replacement for stationarity, curvature or full-EOS validation.
"""

import torch


def successive_points(fluid, p, t, chemical, active, point, *, iterations=6):
    from .equilibrium_minimization_batch import _first_acceptable
    from .equilibrium_stability_batch import stability_objective

    evaluate = stability_objective(fluid, active)
    scales = point.new_tensor([2.0**-i for i in range(6)])
    with torch.no_grad():
        for _ in range(iterations):
            value, grad, residual, _, q = evaluate(point, p, t, chemical)
            done = q["valid"] & (residual.abs().amax(-1) < 1e-9) & (grad.abs().amax(-1) < 2e-9)
            if bool(done.all().cpu()):
                break
            # In log-ratio coordinates, subtracting the chemical differences
            # is precisely the normalized fugacity fixed-point iteration.
            direction = -residual
            direction = torch.where(((direction * grad).sum(-1) <= 0)[:, None], direction, -grad)
            direction = direction * (20.0 / direction.abs().amax(-1).clamp_min(20.0))[:, None]
            candidate = _first_acceptable(
                evaluate,
                point,
                direction,
                p,
                t,
                chemical,
                value,
                grad,
                residual,
                done,
                scales,
                len(scales),
            )
            if torch.equal(candidate, point):
                break
            point = candidate
    return point


def refine_candidates(fluid, p, t, chemical, active, point, original, **policy):
    from .equilibrium_stability_batch import stability_candidates

    unresolved = (~original[-1]).nonzero(as_tuple=True)[0]
    if len(unresolved) == 0 or point.shape[-1] == 0:
        return original
    fluid._ssi_attempted_seeds = getattr(fluid, "_ssi_attempted_seeds", 0) + len(unresolved)
    pressures, temperatures, reference, start = (
        v.index_select(0, unresolved) for v in (p, t, chemical, point)
    )
    trial = successive_points(fluid, pressures, temperatures, reference, active, start)
    if torch.equal(trial, start):
        return original
    try:
        refined = stability_candidates(
            fluid, pressures, temperatures, reference, active, trial, _allow_ssi=False, **policy
        )
    except torch.linalg.LinAlgError:
        return original
    resolved = refined[-1].nonzero(as_tuple=True)[0]
    if len(resolved) == 0:
        return original
    fluid._ssi_resolved_seeds = getattr(fluid, "_ssi_resolved_seeds", 0) + len(resolved)
    rows = unresolved.index_select(0, resolved)
    value, composition, properties, accepted = original
    return (
        value.index_copy(0, rows, refined[0].index_select(0, resolved)),
        composition.index_copy(0, rows, refined[1].index_select(0, resolved)),
        {
            k: v.index_copy(0, rows, refined[2][k].index_select(0, resolved))
            for k, v in properties.items()
        },
        accepted.index_copy(0, rows, refined[3].index_select(0, resolved)),
    )
