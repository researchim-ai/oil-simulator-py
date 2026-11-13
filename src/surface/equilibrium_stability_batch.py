"""Device tensor kernel for independent TPD starts across spatial points.

No phase classification, host fallback or tolerance relaxation occurs here.
The owner retains every unresolved start for its reference minimizer.
"""

import torch


def stability_candidates(
    fluid, pressures, temperatures, chemical, active, point, *, _scaled_curvature=True
):
    n = len(active)
    chemical_eos = getattr(fluid, "tensor_chemical", fluid.tensor_homogeneous)
    chemical_batch = getattr(fluid, "tensor_chemical_batch", torch.vmap(chemical_eos))
    identity = torch.eye(n - 1, dtype=point.dtype, device=point.device)
    scales = getattr(fluid, "_stability_line_scales", None)
    if scales is None:
        scales = point.new_tensor([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])

    def evaluate(v, row_p, row_t, row_chemical):
        logits = torch.cat((v, torch.zeros_like(v[..., :1])), -1)
        logw = torch.log_softmax(logits, -1)
        wa = logw.exp()
        w = v.new_zeros((*v.shape[:-1], len(fluid.components))).index_copy(-1, active, wa)
        if v.ndim == 1:
            q = chemical_eos(row_p, row_t, w)
        else:
            shape = v.shape[:-1]
            pp, tt = torch.broadcast_to(row_p, shape), torch.broadcast_to(row_t, shape)
            q = chemical_batch(
                pp.reshape(-1), tt.reshape(-1), w.reshape(-1, len(fluid.components))
            )
            q = {k: x.reshape(*shape, *x.shape[1:]) for k, x in q.items()}
        delta = logw + q["log_fugacity_coefficients"][..., active] - row_chemical
        tpd = (wa * delta).sum(-1)
        gradient = (wa * (delta - tpd[..., None]))[..., :-1]
        return tpd, gradient, delta[..., :-1] - delta[..., -1:], w, q

    def gradient(v, pp, tt, mu):
        return evaluate(v, pp, tt, mu)[1]

    def stationarity(v, pp, tt, mu):
        return evaluate(v, pp, tt, mu)[2]

    # Chemical differences remain conditioned at trace fractions. Each
    # seed owns its tangent and line search; curvature is checked only
    # at the candidate minimum. No cross-seed Jacobian is assembled.
    with torch.no_grad():
        for _ in range(10):
            value, grad, residual, _, q = evaluate(point, pressures, temperatures, chemical)
            if hasattr(fluid, "tensor_chemical_jacobian"):
                jac = fluid.tensor_chemical_jacobian(
                    pressures, temperatures, chemical, active, point
                )
            else:
                jac = torch.vmap(torch.func.jacfwd(stationarity, argnums=0))(
                    point, pressures, temperatures, chemical
                )
            valid = (
                torch.isfinite(jac).all(dim=(-1, -2)) & torch.isfinite(grad).all(-1) & q["valid"]
            )
            jac = torch.where(valid[:, None, None], jac, identity)
            done = valid & (residual.abs().amax(-1) < 1e-9) & (grad.abs().amax(-1) < 2e-9)
            if bool(done.all().cpu()):
                break
            step, info = torch.linalg.solve_ex(jac, -residual, check_errors=False)
            usable = valid & (info == 0) & torch.isfinite(step).all(-1)
            descent = (grad * step).sum(-1) < 0
            # Match the reference stationarity polish when weighting by a
            # trace mole fraction loses the objective gradient. The existing
            # line search still demands Gibbs decrease, or flat Gibbs AND
            # a smaller unweighted chemical residual; curvature is checked.
            polish = grad.abs().amax(-1) < 2e-9
            step = torch.where((usable & (descent | polish))[:, None], step, -grad)
            limit = (20.0 / step.abs().amax(-1).clamp_min(20.0))[:, None]
            step = step * limit
            candidates = (point[:, None, :] + scales[None, :, None] * step[:, None, :]).clamp(
                -160, 160
            )
            cv, cg, cr, _, cq = evaluate(
                candidates, pressures[:, None], temperatures[:, None], chemical[:, None, :]
            )
            decrease = (grad * step).sum(-1)[:, None] * scales[None, :]
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
            merit = torch.where(allowed, cr.abs().amax(-1), torch.inf)
            indices = merit.argmin(-1)
            chosen = candidates.gather(1, indices[:, None, None].expand(-1, 1, n - 1)).squeeze(1)
            take = allowed.gather(1, indices[:, None]).squeeze(1) & ~done
            point = torch.where(take[:, None], chosen, point)
        value, grad, residual, w, q = evaluate(point, pressures, temperatures, chemical)
        if _scaled_curvature:
            # A positive Gibbs minimum near a pure phase has an almost zero
            # logit Hessian. Test the SAME constrained second variation in
            # square-root mole fractions, without deleting trace species.
            weights = w[:, active]
            root = weights.sqrt()
            pivot = weights.argmax(-1)
            direction = root + torch.nn.functional.one_hot(pivot, n).to(root.dtype)
            householder = torch.eye(n, dtype=point.dtype, device=point.device)[None] - (
                2
                * direction[:, :, None]
                * direction[:, None, :]
                / direction.square().sum(-1)[:, None, None]
            )
            columns = torch.arange(n - 1, device=point.device)[None]
            columns = columns + (columns >= pivot[:, None]).long()
            basis = householder.gather(-1, columns[:, None, :].expand(-1, n, -1))

            def tangent_gradient(v, pp, tt, mu, reference, tangent):
                rr = reference + tangent @ v
                total = rr.square().sum()
                composition = rr.square() / total
                x = v.new_zeros(len(fluid.components)).index_copy(0, active, composition)
                q = chemical_eos(pp, tt, x)
                delta = composition.log() + q["log_fugacity_coefficients"][active] - mu
                potential = (composition * delta).sum()
                return 2 / total * (tangent.T @ (rr * (delta - potential)))

            jac = torch.vmap(torch.func.jacfwd(tangent_gradient, argnums=0))(
                point.new_zeros(point.shape), pressures, temperatures, chemical, root, basis
            )
        else:
            jac = torch.vmap(torch.func.jacfwd(gradient, argnums=0))(
                point, pressures, temperatures, chemical
            )
        hessian = 0.5 * (jac + jac.transpose(-1, -2))
        finite = torch.isfinite(hessian).all(dim=(-1, -2))
        hessian = torch.where(finite[:, None, None], hessian, identity)
        eigenvalues = torch.linalg.eigvalsh(hessian)
        # A residual larger than a tiny positive eigenvalue must not certify
        # an almost critical saddle. Keep unresolved cases on the scalar path.
        curvature_floor = (
            torch.maximum(value.new_full(value.shape, 1e-12), 16 * residual.abs().amax(-1))
            if _scaled_curvature
            else value.new_full(value.shape, 1e-12)
        )
        accepted = (
            finite
            & q["valid"]
            & torch.isfinite(value)
            & (residual.abs().amax(-1) < 1e-9)
            & (grad.abs().amax(-1) < 2e-9)
            & (eigenvalues[:, 0] > curvature_floor)
        )
        # Callers cache physical properties of every candidate. Complete them
        # once here, instead of on every objective/Jacobian/line-search trial.
        if hasattr(fluid, "tensor_chemical"):
            q = torch.vmap(fluid.tensor_homogeneous)(pressures, temperatures, w)
            delta = w[:, active].log() + q["log_fugacity_coefficients"][:, active] - chemical
            value = (w[:, active] * delta).sum(-1)
            exact_residual = delta[:, :-1] - delta[:, -1:]
            exact_gradient = w[:, active] * (delta - value[:, None])
            accepted = (
                accepted
                & q["valid"]
                & torch.isfinite(value)
                & (exact_residual.abs().amax(-1) < 1e-9)
                & (exact_gradient.abs().amax(-1) < 2e-9)
            )
    return value, w, q, accepted
