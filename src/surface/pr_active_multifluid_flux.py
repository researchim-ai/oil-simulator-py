"""Entropy-compatible pairwise BN closure with genuinely absent phase slots.

For each pair gamma_ij=alpha_j*dalpha_i-alpha_i*dalpha_j,
Uij=(alpha_i*ui+alpha_j*uj)/(alpha_i+alpha_j), and
Pij=(alpha_j*pi/Ti+alpha_i*pj/Tj)/(alpha_j/Ti+alpha_i/Tj).
Pair pressure/work sources and volume paths are antisymmetric. Their summed
continuous entropy contribution vanishes; discrete entropy is audited by the
owning finite-volume solver. Ghost primitives carry no conserved inventory.
"""

import math

import torch

# Eight-point Gauss-Legendre on [0,1], constant-exact residual form.
_NODES = (
    0.019855071751231884,
    0.10166676129318663,
    0.2372337950418355,
    0.4082826787521751,
    0.5917173212478249,
    0.7627662049581645,
    0.8983332387068134,
    0.9801449282487681,
)
_WEIGHTS = (
    0.05061426814518813,
    0.11119051722668724,
    0.15685332293894365,
    0.181341891689181,
    0.181341891689181,
    0.15685332293894365,
    0.11119051722668724,
    0.05061426814518813,
)


def fluctuations(left, right, wave_speed):
    alpha_l, alpha_r = left["alpha"], right["alpha"]
    active_l, active_r = alpha_l > 0, alpha_r > 0
    any_active = active_l | active_r

    def ghosts(field, empty):
        ll = torch.where(active_l, left[field], right[field])
        rr = torch.where(active_r, right[field], left[field])
        return torch.where(any_active, ll, empty), torch.where(any_active, rr, empty)

    pl, pr = ghosts("pressure", 0.0)
    tl, tr = ghosts("temperature", 1.0)
    ul, ur = ghosts("velocity", 0.0)
    da = alpha_r - alpha_l
    count = alpha_l.shape[-1]
    momentum = [torch.zeros_like(alpha_l[..., 0]) for _ in range(count)]
    energy = [torch.zeros_like(alpha_l[..., 0]) for _ in range(count)]
    volume = [torch.zeros_like(alpha_l[..., 0]) for _ in range(count)]

    def pair(s, i, j):
        ai = alpha_l[..., i] + s * da[..., i]
        aj = alpha_l[..., j] + s * da[..., j]
        ui = ul[..., i] + s * (ur[..., i] - ul[..., i])
        uj = ul[..., j] + s * (ur[..., j] - ul[..., j])
        pi = pl[..., i] + s * (pr[..., i] - pl[..., i])
        pj = pl[..., j] + s * (pr[..., j] - pl[..., j])
        ti = tl[..., i] + s * (tr[..., i] - tl[..., i])
        tj = tl[..., j] + s * (tr[..., j] - tl[..., j])
        u = ui + (aj / (ai + aj).clamp_min(1e-300)) * (uj - ui)
        w = ai / tj
        p = pi + (w / (aj / ti + w).clamp_min(1e-300)) * (pj - pi)
        return u, p, p * u

    for i in range(count):
        for j in range(i + 1, count):
            gamma = alpha_l[..., j] * da[..., i] - alpha_l[..., i] * da[..., j]
            u, p, pu = pair(0.5, i, j)
            du, dp, dpu = torch.zeros_like(u), torch.zeros_like(p), torch.zeros_like(pu)
            for s, w in zip(_NODES, _WEIGHTS):
                uq, pq, puq = pair(s, i, j)
                du = du + w * (uq - u)
                dp = dp + w * (pq - p)
                dpu = dpu + w * (puq - pu)
            uv, pv, ev = (u + du) * gamma, (p + dp) * gamma, (pu + dpu) * gamma
            volume[i] = volume[i] + uv
            volume[j] = volume[j] - uv
            momentum[i] = momentum[i] - pv
            momentum[j] = momentum[j] + pv
            energy[i] = energy[i] - ev
            energy[j] = energy[j] + ev
    alpha_path = torch.stack(volume, -1)
    path = torch.cat(
        (
            torch.zeros_like(left["conserved"][..., :-2]),
            torch.stack(momentum, -1)[..., None],
            torch.stack(energy, -1)[..., None],
        ),
        -1,
    )
    jump = right["flux"] - left["flux"] + path
    # Material contacts need no acoustic viscosity. ULP-scale recognition
    # retains small resolved pressure/velocity perturbations as acoustic waves.
    pp = torch.cat((pl, pr), -1)
    uu = torch.cat((ul, ur), -1)
    mask = torch.cat((any_active, any_active), -1)
    pmin = torch.where(mask, pp, math.inf).min(-1).values
    pmax = torch.where(mask, pp, -math.inf).max(-1).values
    umin = torch.where(mask, uu, math.inf).min(-1).values
    umax = torch.where(mask, uu, -math.inf).max(-1).values
    eps = 32 * torch.finfo(pp.dtype).eps
    precision = (
        torch.cat((left["pressure_precision"], right["pressure_precision"]), -1).max(-1).values
    )
    contact = (pmax - pmin <= torch.maximum(eps * pmax.abs().clamp_min(1.0), 2 * precision)) & (
        umax - umin <= eps * torch.maximum(umax.abs(), umin.abs()).clamp_min(1.0)
    )
    speed = torch.where(
        contact, torch.maximum(umax.abs(), umin.abs()), torch.full_like(pmax, wave_speed)
    )
    difference = speed[..., None, None] * (right["conserved"] - left["conserved"])
    adifference = speed[..., None] * da
    flux = 0.5 * (left["flux"] + right["flux"]) - 0.5 * difference
    dm, dp = 0.5 * (jump - difference), 0.5 * (jump + difference)
    am, ap = 0.5 * (alpha_path - adifference), 0.5 * (alpha_path + adifference)
    # Evaluate a material contact as pure upwind advection. Subtracting
    # pressure work from two large energy fluxes would leave roundoff energy
    # in a genuinely absent phase, which cannot own energy without mass.
    common_u = 0.5 * (umin + umax)
    common_p = 0.5 * (pmin + pmax)
    dq = right["conserved"] - left["conserved"]
    negative = common_u.clamp_max(0.0)
    positive = common_u.clamp_min(0.0)
    dm = torch.where(contact[..., None, None], negative[..., None, None] * dq, dm)
    dp = torch.where(contact[..., None, None], positive[..., None, None] * dq, dp)
    am = torch.where(contact[..., None], negative[..., None] * da, am)
    ap = torch.where(contact[..., None], positive[..., None] * da, ap)
    contact_path = torch.cat(
        (
            torch.zeros_like(dq[..., :-2]),
            (-common_p[..., None] * da)[..., None],
            (-common_p[..., None] * common_u[..., None] * da)[..., None],
        ),
        -1,
    )
    path = torch.where(contact[..., None, None], contact_path, path)
    alpha_path = torch.where(contact[..., None], common_u[..., None] * da, alpha_path)
    upwind = torch.where((common_u >= 0)[..., None, None], left["conserved"], right["conserved"])
    advected = upwind * common_u[..., None, None]
    pressure = 0.5 * (alpha_l + alpha_r) * common_p[..., None]
    contact_flux = torch.cat(
        (
            advected[..., :-2],
            (advected[..., -2] + pressure)[..., None],
            (advected[..., -1] + pressure * common_u[..., None])[..., None],
        ),
        -1,
    )
    flux = torch.where(contact[..., None, None], contact_flux, flux)
    return (
        dm,
        dp,
        am,
        ap,
        {"flux": flux, "phase_path": path, "alpha_path": alpha_path, "contact": contact},
    )
