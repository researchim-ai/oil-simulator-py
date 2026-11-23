"""Independent batched minimization before unresolved scalar TPD fallbacks.

Every original start remains represented. This is a predictor: only the
existing Newton/stationarity/curvature/full-EOS audit can accept its result.
"""

import torch

from .equilibrium_stability_batch import stability_candidates, stability_objective


def _group_width(rows, remaining, preferred, kernel_width, small_width=None):
    width = min(preferred, remaining)
    # Small tails occupy the same fused kernel tile anyway. Resolve all
    # remaining scales in that tile instead of launching it repeatedly.
    small_width = kernel_width if small_width is None else small_width

    def work(count):
        full, tail = divmod(count, kernel_width)
        return full * kernel_width + (
            0 if not tail else small_width if tail <= small_width else kernel_width
        )

    budget = work(rows * width)
    while width < remaining and work(rows * (width + 1)) <= budget:
        width += 1
    return width


def _first_acceptable(
    evaluate,
    point,
    direction,
    pressures,
    temperatures,
    chemical,
    value,
    grad,
    residual,
    done,
    scales,
    line_width,
    *,
    kernel_width=1,
    small_width=None,
):
    """Same first admissible scale per row; later scales cannot change it."""
    if _group_width(len(point), len(scales), line_width, kernel_width, small_width) == len(scales):
        # Keep the original full-batch reference, including its evaluation
        # order, for independent equivalence and performance comparisons.
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
        selected = allowed.to(torch.int64).argmax(-1)
        picked = trials.gather(1, selected[:, None, None].expand(-1, 1, point.shape[-1])).squeeze(1)
        return torch.where((allowed.any(-1) & ~done)[:, None], picked, point)
    chosen = point.clone()
    pending = torch.arange(len(point), device=point.device)
    begin = 0
    while begin < len(scales):
        if not len(pending):
            break
        count = _group_width(
            len(pending), len(scales) - begin, line_width, kernel_width, small_width
        )
        local = scales[begin : begin + count]
        source = point.index_select(0, pending)
        step = direction.index_select(0, pending)
        trials = (source[:, None] + local[None, :, None] * step[:, None]).clamp(-160, 160)
        cv, cg, cr, _, cq = evaluate(
            trials, pressures[pending, None], temperatures[pending, None], chemical[pending, None]
        )
        decrease = (grad[pending] * step).sum(-1)[:, None] * local[None]
        previous = value[pending, None]
        allowed = (
            cq["valid"]
            & torch.isfinite(cv)
            & torch.isfinite(cg).all(-1)
            & (
                (cv < previous + 1e-4 * decrease)
                | (
                    (cv <= previous + 1e-14)
                    & (cr.abs().amax(-1) < residual[pending].abs().amax(-1)[:, None])
                )
            )
        )
        take = allowed.any(-1) & ~done[pending]
        selected = allowed.to(torch.int64).argmax(-1)
        picked = trials.gather(1, selected[:, None, None].expand(-1, 1, point.shape[-1])).squeeze(1)
        chosen.index_copy_(0, pending, torch.where(take[:, None], picked, chosen[pending]))
        begin += count
        if begin < len(scales):
            pending = pending[(~allowed.any(-1) & ~done[pending]).nonzero(as_tuple=True)[0]]
    return chosen


def _minimize(fluid, pressures, temperatures, chemical, active, point, *, _compact=True):
    evaluate = stability_objective(fluid, active)
    width = point.shape[-1]
    identity = torch.eye(width, dtype=point.dtype, device=point.device)
    scales = point.new_tensor([2.0 ** (-i) for i in range(12)])
    line_width = getattr(fluid, "_minimization_line_width", 12)
    if type(line_width) is not int or not 1 <= line_width <= len(scales):
        raise ValueError("Minimizer line-search width must be an integer in 1..12")
    old_point = old_grad = inverse = None
    all_points = point.clone()
    rows = torch.arange(len(point), device=point.device)
    with torch.no_grad():
        kernel_width = 1
        small_width = None
        if line_width < len(scales) and hasattr(fluid, "tc"):
            from .chemical_kernel import _SMALL_WIDTH, _WIDTH, eligible

            if eligible(fluid, point, pressures, temperatures, chemical):
                kernel_width = _WIDTH
                if fluid.device.type == "cpu" and fluid._compact_chemical_cpu:
                    small_width = _SMALL_WIDTH
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
            next_point = _first_acceptable(
                evaluate,
                point,
                direction,
                pressures,
                temperatures,
                chemical,
                value,
                grad,
                residual,
                done,
                scales,
                line_width,
                kernel_width=kernel_width,
                small_width=small_width,
            )
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
