"""Transaction-scoped initial guesses, never cached answers at nearby states.

Every accepted warm flash still requires equal fugacities, positive stocks,
full multistart TPD, material balance and Gibbs checks. Failure returns to the
original cold algorithm. Exact memoization remains a separate mechanism.
"""

import torch

from .equilibrium_cache import FlashCache, _signature


def _context(fluid, active):
    cache = getattr(fluid, "_active_flash_cache", None)
    if not fluid._warm_equilibrium or not isinstance(cache, FlashCache) or len(active) < 2:
        return None
    try:
        key = (_signature(fluid), tuple(active.detach().cpu().tolist()))
    except (TypeError, RuntimeError):
        return None
    return cache, key


def guess(fluid, p, t, z, active):
    context = _context(fluid, active)
    if context is None:
        return None
    cache, key = context
    hint = cache.warm_equilibria.get(key)
    if hint is None:
        return None
    oldp, oldt, oldz, fractions, _pins = hint
    change = torch.stack(((p / oldp).log().abs(), (t / oldt - 1).abs(), (z - oldz).abs().max()))
    if not bool((change < 0.05).all().cpu()):
        return None
    cache.warm_flash_attempts += 1
    return fractions * z[active]


def remember(fluid, p, t, z, active, beta, x):
    context = _context(fluid, active)
    if context is None or len(beta) < 2:
        return
    cache, key = context
    fractions = beta[:, None] * x[:, active] / z[active]
    cache.warm_equilibria[key] = (
        p.detach().clone(),
        t.detach().clone(),
        z.detach().clone(),
        fractions.detach().clone(),
        tuple(
            getattr(fluid, n) for n in ("tc", "pc", "cp", "kappa", "b", "a0", "kij", "molar_masses")
        ),
    )
    cache.warm_equilibria.move_to_end(key)
    if len(cache.warm_equilibria) > min(8, cache.maximum_entries):
        cache.warm_equilibria.popitem(last=False)


def admissible(fluid, z, active, beta, x, q, chemical, initial_g):
    if len(beta) < 2:
        return False
    balance = (beta[:, None] * x).sum(0) - z
    final_g = (
        beta
        * (x[:, active] * (x[:, active].log() + q["log_fugacity_coefficients"][:, active])).sum(-1)
    ).sum()
    n = len(beta)
    separation = (x[:, None, :] - x[None, :, :]).abs().amax(-1)
    volume = q["molar_volume_m3_mol"]
    difference = (volume[:, None] - volume[None, :]).abs() / volume[:, None].abs().clamp_min(1e-300)
    distinct = ((separation > 1e-7) | (difference > 1e-7)) | torch.eye(
        n, dtype=torch.bool, device=z.device
    )
    return bool(
        (
            (balance.abs().max() < 1e-10)
            & (final_g <= initial_g + 1e-10)
            & torch.isfinite(final_g)
            & distinct.all()
        ).cpu()
    )
