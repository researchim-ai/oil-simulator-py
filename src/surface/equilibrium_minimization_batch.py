"""Independent batched minimization before unresolved scalar TPD fallbacks.

Every original start remains represented. This is a predictor: only the
existing Newton/stationarity/curvature/full-EOS audit can accept its result.
"""

import torch

from .equilibrium_stability_batch import stability_candidates, stability_objective


def _minimize(fluid, pressures, temperatures, chemical, active, point, *, _compact=True):
    evaluate = stability_objective(fluid, active)
    width = point.shape[-1]
    identity = torch.eye(width, dtype=point.dtype, device=point.device)
    scales = point.new_tensor([2.0 ** (-i) for i in range(12)])
    old_point = old_grad = inverse = None
    all_points = point.clone()
    rows = torch.arange(len(point), device=point.device)
    with torch.no_grad():
        for _ in range(50):
            value, grad, residual, _, q = evaluate(point, pressures, temperatures, chemical)
            done = q["valid"] & (residual.abs().amax(-1) < 1e-9) & (grad.abs().amax(-1) < 2e-9)
            if bool(done.all().cpu()):
                break
            if inverse is None:
                inverse = identity[None] / grad.abs().amax(-1).clamp_min(1e-8)[:, None, None]
            else:
                displacement, change = point - old_point, grad - old_grad
                sy = (displacement * change).sum(-1)
                update = sy > 1e-12 * displacement.norm(dim=-1) * change.norm(dim=-1)
                reciprocal = torch.where(update, sy, torch.ones_like(sy)).reciprocal()
                transform = (
                    identity[None]
                    - reciprocal[:, None, None] * displacement[:, :, None] * change[:, None, :]
                )
                candidate = (
                    transform @ inverse @ transform.transpose(-1, -2)
                    + reciprocal[:, None, None]
                    * displacement[:, :, None]
                    * displacement[:, None, :]
                )
                update = update & torch.isfinite(candidate).all(dim=(-1, -2))
                inverse = torch.where(update[:, None, None], candidate, inverse)
            direction = -(inverse @ grad[:, :, None]).squeeze(-1)
            usable = torch.isfinite(direction).all(-1) & ((direction * grad).sum(-1) < 0)
            direction = torch.where(usable[:, None], direction, -grad)
            direction *= (20.0 / direction.abs().amax(-1).clamp_min(20.0))[:, None]
            trials = (point[:, None] + scales[None, :, None] * direction[:, None]).clamp(-160, 160)
            cv, cg, cr, _, cq = evaluate(
                trials, pressures[:, None], temperatures[:, None], chemical[:, None]
            )
            decrease = (grad * direction).sum(-1)[:, None] * scales[None]
            allowed = (
                cq["valid"]
                & torch.isfinite(cv)
                & torch.isfinite(cg).all(-1)
                & (
                    (cv < value[:, None] + 1e-4 * decrease)
                    | (
                        (cv <= value[:, None] + 1e-14)
                        & (cr.abs().amax(-1) < residual.abs().amax(-1)[:, None])
                    )
                )
            )
            # Independent first acceptable step, never one global line search.
            selected = allowed.to(torch.int64).argmax(-1)
            chosen = trials.gather(1, selected[:, None, None].expand(-1, 1, width)).squeeze(1)
            next_point = torch.where((allowed.any(-1) & ~done)[:, None], chosen, point)
            if torch.equal(next_point, point):
                break
            if _compact:
                all_points.index_copy_(0, rows, next_point)
                keep = ((~done) & (next_point != point).any(-1)).nonzero(as_tuple=True)[0]
                rows = rows.index_select(0, keep)
                pressures, temperatures, chemical = (
                    value.index_select(0, keep) for value in (pressures, temperatures, chemical)
                )
                old_point, old_grad, inverse, point = (
                    value.index_select(0, keep) for value in (point, grad, inverse, next_point)
                )
                if not len(point):
                    break
            else:
                old_point, old_grad, point = point, grad, next_point
    if _compact:
        all_points.index_copy_(0, rows, point)
        return all_points
    return point


def minimize_stability_candidates(fluid, pressures, temperatures, chemical, active, point):
    original = stability_candidates(fluid, pressures, temperatures, chemical, active, point)
    unresolved = (~original[-1]).nonzero(as_tuple=True)[0]
    if len(unresolved) == 0:
        return original
    p, t, mu, start = (
        v.index_select(0, unresolved) for v in (pressures, temperatures, chemical, point)
    )
    trial = _minimize(fluid, p, t, mu, active, start)
    try:
        refined = stability_candidates(fluid, p, t, mu, active, trial)
    except torch.linalg.LinAlgError:
        return original
    resolved = refined[-1].nonzero(as_tuple=True)[0]
    if len(resolved) == 0:
        return original
    rows = unresolved.index_select(0, resolved)
    value, composition, properties, accepted = original
    return (
        value.index_copy(0, rows, refined[0].index_select(0, resolved)),
        composition.index_copy(0, rows, refined[1].index_select(0, resolved)),
        {
            key: v.index_copy(0, rows, refined[2][key].index_select(0, resolved))
            for key, v in properties.items()
        },
        accepted.index_copy(0, rows, refined[3].index_select(0, resolved)),
    )
