"""Torch HLLC with left/contact/right waves and supersonic upwinding.

Local Davis estimates are used for a single mechanically stable phase. An
invalid local star falls back explicitly to the configured fixed-bound fan.
Thermochemical equilibrium branches are validated by the owning time step.
"""

import torch

from .pr_momentum_vessels import hllc_port


def hllc_local(left, right, sound_left, sound_right, global_bound, *, estimate="davis"):
    ul, fl, pl, vl, rl, sl = left
    ur, fr, pr, vr, rr, sr = right
    if estimate == "davis":
        speed_l = torch.minimum(vl - sound_left, vr - sound_right)
        speed_r = torch.maximum(vl + sound_left, vr + sound_right)
    elif estimate == "pvrs":
        predictor = (
            0.5 * (pl + pr) - 0.125 * (vr - vl) * (rl + rr) * (sound_left + sound_right)
        ).clamp_min(0.0)
        gl = rl * sound_left.square() / pl
        gr = rr * sound_right.square() / pr
        ql = torch.sqrt(1 + (gl + 1) / (2 * gl) * (predictor / pl - 1).clamp_min(0.0))
        qr = torch.sqrt(1 + (gr + 1) / (2 * gr) * (predictor / pr - 1).clamp_min(0.0))
        speed_l = vl - sound_left * ql
        speed_r = vr + sound_right * qr
    else:
        raise ValueError("Unknown local acoustic estimate")

    def safe(x):
        return torch.where(
            x.abs() > 1e-25, x, torch.where(x < 0, x.new_tensor(-1e-25), x.new_tensor(1e-25))
        )

    al = rl * (speed_l - vl)
    ar = rr * (speed_r - vr)
    contact = (pr - pl + al * vl - ar * vr) / safe(al - ar)
    pressure = pl + al * (contact - vl)

    def star(u, flux, p, v, rho, entropy, speed):
        denominator = safe(speed - contact)
        ratio = (speed - v) / denominator
        energy = ((speed - v) * u[..., -1] - p * v + pressure * contact) / denominator
        us = torch.cat(
            (u[..., :-2] * ratio[..., None], (rho * ratio * contact)[..., None], energy[..., None]),
            -1,
        )
        return flux + speed[..., None] * (us - u), entropy * ratio * contact, ratio

    lf, ls, lr = star(ul, fl, pl, vl, rl, sl, speed_l)
    rf, rs, rrstar = star(ur, fr, pr, vr, rr, sr, speed_r)
    middle = torch.where((contact >= 0)[..., None], lf, rf)
    flux = torch.where(
        (speed_l >= 0)[..., None], fl, torch.where((speed_r <= 0)[..., None], fr, middle)
    )
    entropy = torch.where(
        speed_l >= 0, sl * vl, torch.where(speed_r <= 0, sr * vr, torch.where(contact >= 0, ls, rs))
    )
    local_ok = (pressure > 0) & (lr > 0) & (rrstar > 0) & (contact > speed_l) & (contact < speed_r)
    contact = torch.where(speed_l >= 0, vl, torch.where(speed_r <= 0, vr, contact))
    pressure = torch.where(speed_l >= 0, pl, torch.where(speed_r <= 0, pr, pressure))
    local_ok = local_ok & torch.isfinite(flux).all(-1) & torch.isfinite(entropy)
    # The legacy fixed fan is scalar; vmap adds batched evaluation without any
    # host decisions and without evaluating a thermodynamic flash here.
    if ul.ndim == 1:
        bounded = hllc_port(left, right, global_bound)
    else:
        bounded = torch.vmap(lambda *values: hllc_port(values[:6], values[6:], global_bound))(
            *left, *right
        )
    use_fixed = ~local_ok
    return (
        torch.where(local_ok[..., None], flux, bounded[0]),
        torch.where(local_ok, entropy, bounded[1]),
        torch.where(local_ok, contact, bounded[2]),
        torch.where(local_ok, pressure, bounded[3]),
        local_ok | bounded[4],
        use_fixed,
    )
