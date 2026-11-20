"""Analytic PR composition tangents; identical root/caloric admissibility.

The cubic-root tangent differentiates the same detached-root Newton correction
as tensor_homogeneous. Directions are local component columns, never a global
spatial Jacobian. No phase-stability start or curvature criterion is omitted.
"""

import math

import torch

from .equilibrium_properties import R


def directional(fluid, p, t, x, dx):
    q = fluid.tensor_homogeneous(p, t, x, _chemical_tangent=True)
    a, b, ax, z, B = (q[k] for k in ("a", "b", "ax", "z", "B"))
    # Retain exact derivative even for deliberately perturbed asymmetric kij.
    other = (q["aij"] @ x[..., None]).squeeze(-1)
    da = ((ax + other)[..., None, :] @ dx).squeeze(-2)
    db = fluid.b @ dx
    dax = q["aij"].transpose(-1, -2) @ dx
    dA = da * (p / (R * t) ** 2)[..., None]
    dB = db * (p / (R * t))[..., None]
    dz = q["za"][..., None] * dA + q["zb"][..., None] * dB
    sqrt2 = math.sqrt(2.0)
    bi = fluid.b / b[..., None]
    dbb = db / b[..., None]
    coefficient = 1 / (2 * sqrt2 * b * R * t)
    D = (2 * ax - a[..., None] * bi) * coefficient[..., None]
    dD = (
        2 * dax
        - bi[..., :, None] * da[..., None, :]
        + (a[..., None] * bi)[..., :, None] * dbb[..., None, :]
    ) * coefficient[..., None, None] - D[..., :, None] * dbb[..., None, :]
    dL = (
        2
        * sqrt2
        * (z[..., None] * dB - B[..., None] * dz)
        / ((z + (1 - sqrt2) * B) * (z + (1 + sqrt2) * B))[..., None]
    )
    derivative = (
        bi[..., :, None] * dz[..., None, :]
        - bi[..., :, None] * dbb[..., None, :] * (z - 1)[..., None, None]
        - (dz - dB)[..., None, :] / (z - B)[..., None, None]
        - dD * q["ratio"][..., None, None]
        - D[..., :, None] * dL[..., None, :]
    )
    return derivative, q


def logit_jacobian(fluid, p, t, chemical, active, point):
    logits = torch.cat((point, point.new_zeros((*point.shape[:-1], 1))), -1)
    w = logits.softmax(-1)
    x = point.new_zeros((*point.shape[:-1], len(fluid.components))).index_copy(-1, active, w)
    eye = torch.eye(len(active), dtype=point.dtype, device=point.device)[..., :-1]
    dw = w[..., :, None] * (eye - w[..., None, :-1])
    dx = point.new_zeros((*point.shape[:-1], len(fluid.components), point.shape[-1])).index_copy(
        -2, active, dw
    )
    dphi, _ = directional(fluid, p, t, x, dx)
    dphi = dphi[..., active, :]
    return (
        torch.eye(point.shape[-1], dtype=point.dtype, device=point.device)
        + dphi[..., :-1, :]
        - dphi[..., -1:, :]
    )


def tangent_hessian(fluid, p, t, chemical, active, root, basis):
    total = root.square().sum(-1)
    w = root.square() / total[..., None]
    rb = (root[..., None, :] @ basis).squeeze(-2)
    dw = (
        2
        / total[..., None, None]
        * (root[..., :, None] * basis - w[..., :, None] * rb[..., None, :])
    )
    x = root.new_zeros((*root.shape[:-1], len(fluid.components))).index_copy(-1, active, w)
    dx = root.new_zeros((*root.shape[:-1], len(fluid.components), basis.shape[-1])).index_copy(
        -2, active, dw
    )
    dphi, q = directional(fluid, p, t, x, dx)
    delta = w.log() + q["log_fugacity_coefficients"][..., active] - chemical
    potential = (w * delta).sum(-1)
    ddelta = dw / w[..., :, None] + dphi[..., active, :]
    dpotential = (dw * delta[..., :, None] + w[..., :, None] * ddelta).sum(-2)
    centered = delta - potential[..., None]
    h = (basis.transpose(-1, -2) @ (root * centered)[..., None]).squeeze(-1)
    dh = basis.transpose(-1, -2) @ (
        basis * centered[..., :, None] + root[..., :, None] * (ddelta - dpotential[..., None, :])
    )
    return (
        2 / total[..., None, None] * dh
        - 4 / total.square()[..., None, None] * h[..., :, None] * rb[..., None, :]
    )
