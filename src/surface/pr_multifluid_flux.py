"""First-order reference-phase BN path fluctuations, all Torch tensor math.

The reference interface velocity is u_0. For non-reference phase k the
interface pressure is p_k. Paths are linear in alpha, pressure and velocity;
the p_k*u_0 path integral is quadratic and evaluated exactly.
"""

import torch


def path_fluctuations(left, right, wave_speed):
    ql, qr = left["conserved"], right["conserved"]
    fl, fr = left["flux"], right["flux"]
    da = right["alpha"] - left["alpha"]
    pbar = 0.5 * (left["pressure"] + right["pressure"])
    ul, ur = left["velocity"][..., 0], right["velocity"][..., 0]
    ubar = 0.5 * (ul + ur)
    pu = (left["pressure"] * ul[..., None] + right["pressure"] * ur[..., None]) / 3
    pu = pu + (left["pressure"] * ur[..., None] + right["pressure"] * ul[..., None]) / 6
    momentum = -pbar[..., 1:] * da[..., 1:]
    energy = -pu[..., 1:] * da[..., 1:]
    momentum = torch.cat((-momentum.sum(-1, keepdim=True), momentum), dim=-1)
    energy = torch.cat((-energy.sum(-1, keepdim=True), energy), dim=-1)
    phase_path = torch.cat(
        (torch.zeros_like(ql[..., :-2]), momentum[..., None], energy[..., None]), dim=-1
    )
    phase_jump = fr - fl + phase_path
    phase_diff = wave_speed * (qr - ql)
    alpha_jump = ubar[..., None] * da[..., 1:]
    alpha_diff = wave_speed * da[..., 1:]
    minus = torch.cat(
        (0.5 * (phase_jump - phase_diff).flatten(-2), 0.5 * (alpha_jump - alpha_diff)), dim=-1
    )
    plus = torch.cat(
        (0.5 * (phase_jump + phase_diff).flatten(-2), 0.5 * (alpha_jump + alpha_diff)), dim=-1
    )
    flux = 0.5 * (fl + fr) - 0.5 * phase_diff
    return minus, plus, {"flux": flux, "phase_path": phase_path, "alpha_path": alpha_jump}
