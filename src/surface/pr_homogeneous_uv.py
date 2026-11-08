"""Batched homogeneous PR N/V/E/P recovery, without equilibrium phase splitting.

The caloric equation at prescribed molar volume is monotone where Cv>0.
A safeguarded device Newton recovers temperature; the explicit PR pressure
then supplies the root/branch audit. Extensive stocks remain authoritative.
"""

import math

import torch

from .liquid import HydraulicConvergenceError, PropertyModelError
from .peng_robinson import R
from .pr_interphase_relaxation import PRRelaxationPhase


def caloric_tv(fluid, temperature, molar_volume, composition):
    """Caloric PR from T/v/x, differentiable and vectorized over phases."""
    t, v, x = temperature, molar_volume, composition
    rt = (t[..., None] / fluid.tc).sqrt()
    m = 1 + fluid.kappa * (1 - rt)
    ai = fluid.a0 * m.square()
    ait = -fluid.a0 * fluid.kappa * m / (fluid.tc * rt)
    aitt = fluid.a0 * fluid.kappa * (1 + fluid.kappa) / (2 * fluid.tc.square() * rt.pow(3))
    aij = (ai[..., :, None] * ai[..., None, :]).sqrt() * (1 - fluid.kij)
    ar = ait / ai
    aijt = 0.5 * aij * (ar[..., :, None] + ar[..., None, :])
    aijtt = aij * (
        0.5 * (aitt[..., :, None] / ai[..., :, None] + aitt[..., None, :] / ai[..., None, :])
        - 0.25 * (ar[..., :, None] - ar[..., None, :]).square()
    )
    a = ((x[..., :, None] * x[..., None, :]) * aij).sum((-2, -1))
    at = ((x[..., :, None] * x[..., None, :]) * aijt).sum((-2, -1))
    att = ((x[..., :, None] * x[..., None, :]) * aijtt).sum((-2, -1))
    b = (x * fluid.b).sum(-1)
    ratio = torch.log1p(2 * math.sqrt(2) * b / (v + (1 - math.sqrt(2)) * b))
    cp = (x * fluid.cp).sum(-1)
    u = cp * (t - fluid.tref) - R * t + (t * at - a) / (2 * math.sqrt(2) * b) * ratio
    cv = cp - R + t * att / (2 * math.sqrt(2) * b) * ratio
    den = v.square() + 2 * b * v - b.square()
    p = R * t / (v - b) - a / den
    dpdv = -R * t / (v - b).square() + a * (2 * v + 2 * b) / den.square()
    return {"u": u, "cv": cv, "p": p, "dpdv": dpdv, "b": b, "a": a}


def recover_homogeneous_phases(
    fluid, component_moles, volumes_m3, momenta, energies_j, roots, temperature_guesses_k
):
    """Return finite nonequilibrium phases from canonical N/V/M/E rows.

    No UV flash is used: a metastable but mechanically stable homogeneous
    branch is retained. An inadmissible/spinodal target is rejected atomically.
    """
    n = fluid.tensor(component_moles)
    volume = fluid.tensor(volumes_m3)
    momentum = fluid.tensor(momenta)
    energy = fluid.tensor(energies_j)
    temperature = fluid.tensor(temperature_guesses_k)
    count = len(roots)
    if (
        n.shape != (count, len(fluid.names))
        or any(v.shape != (count,) for v in (volume, momentum, energy, temperature))
        or count < 1
    ):
        raise ValueError("Homogeneous recovery phase/component dimensions differ")
    if any(r not in ("liquid", "vapor", "stable") for r in roots):
        raise ValueError("Explicit homogeneous roots required")
    amount = n.sum(-1)
    mass = (n * fluid.molar_masses).sum(-1)
    valid = (
        torch.isfinite(n).all()
        & (n >= 0).all()
        & (amount > 0).all()
        & (mass > 0).all()
        & torch.isfinite(volume).all()
        & (volume > 0).all()
        & torch.isfinite(momentum).all()
        & torch.isfinite(energy).all()
        & torch.isfinite(temperature).all()
    )
    if not bool(valid.cpu()):
        raise PropertyModelError("Finite positive phase stocks required for homogeneous UV")
    x = n / amount[:, None]
    v = volume / amount
    target = (energy - 0.5 * momentum * (momentum / mass)) / amount
    lo = torch.full_like(target, fluid.min_temperature_k)
    hi = torch.full_like(target, fluid.max_temperature_k)
    low = caloric_tv(fluid, lo, v, x)
    high = caloric_tv(fluid, hi, v, x)
    if not bool(
        (
            (v > low["b"])
            & (low["u"] <= target)
            & (target <= high["u"])
            & (low["cv"] > 0)
            & (high["cv"] > 0)
        )
        .all()
        .cpu()
    ):
        raise PropertyModelError("Homogeneous UV target outside caloric domain")
    temperature = temperature.clamp(lo, hi)
    valid = target.new_ones((), dtype=torch.bool)
    for _ in range(32):
        q = caloric_tv(fluid, temperature, v, x)
        error = q["u"] - target
        valid = valid & torch.isfinite(error).all() & (q["cv"] > 0).all()
        lo = torch.where(error < 0, temperature, lo)
        hi = torch.where(error > 0, temperature, hi)
        proposed = temperature - error / q["cv"]
        temperature = torch.where(
            (proposed >= lo) & (proposed <= hi) & torch.isfinite(proposed),
            proposed,
            0.5 * (lo + hi),
        )
    q = caloric_tv(fluid, temperature, v, x)
    p = q["p"]
    valid = (
        valid
        & (q["dpdv"] < 0).all()
        & (p >= fluid.min_pressure_pa).all()
        & (p <= fluid.max_pressure_pa).all()
        & ((q["u"] - target).abs() <= 1e-8 + 1e-11 * target.abs()).all()
    )
    if not bool(valid.cpu()):
        raise HydraulicConvergenceError("Homogeneous UV caloric or mechanical recovery failed")
    rows = []
    for i, root in enumerate(roots):
        fluid._validate(p[i], temperature[i], x[i])
        eos = fluid.tensor_homogeneous(p[i], temperature[i], x[i], root)
        if not bool(
            (
                eos["valid"] & ((eos["molar_volume_m3_mol"] - v[i]).abs() <= 1e-12 + 1e-9 * v[i])
            ).cpu()
        ):
            raise PropertyModelError("Homogeneous UV target leaves its declared PR root")
        rows.append(
            PRRelaxationPhase(
                tuple(n[i].cpu().tolist()),
                float(volume[i].cpu()),
                float(p[i].cpu()),
                float(temperature[i].cpu()),
                float(momentum[i].cpu()),
                float(energy[i].cpu()),
            )
        )
    return tuple(rows)
