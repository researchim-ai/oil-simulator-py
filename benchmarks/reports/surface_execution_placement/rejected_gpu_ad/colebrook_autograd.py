"""Implicit AD of the same six-update Colebrook root.

The forward iterate/order is unchanged. Saving the returned root makes its
backward differentiable again, without retaining all six Newton tapes. There
are no scalar reads, detached approximation parameters, or new tolerances.
"""

import math

import torch


class ColebrookRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reynolds, roughness):
        u = torch.full_like(reynolds, 7.0)
        for _ in range(6):
            arg = roughness / 3.7 + 2.51 * u / reynolds
            value = u + 2 * torch.log10(arg)
            derivative = 1 + 2 / math.log(10) * (2.51 / reynolds) / arg
            u = u - value / derivative
        ctx.tensor_roughness = isinstance(roughness, torch.Tensor)
        if ctx.tensor_roughness:
            ctx.save_for_backward(reynolds, u, roughness)
        else:
            ctx.save_for_backward(reynolds, u)
            ctx.roughness = roughness
        return u

    @staticmethod
    def backward(ctx, grad_output):
        reynolds, u, *rest = ctx.saved_tensors
        roughness = rest[0] if ctx.tensor_roughness else ctx.roughness
        arg = roughness / 3.7 + 2.51 * u / reynolds
        factor = 2 / math.log(10) / arg
        denominator = 1 + factor * 2.51 / reynolds
        grad_r = grad_output * factor * 2.51 * u / reynolds.square() / denominator
        grad_r = grad_r.sum_to_size(reynolds.shape)
        grad_e = None
        if ctx.tensor_roughness:
            grad_e = (-grad_output * factor / 3.7 / denominator).sum_to_size(roughness.shape)
        return grad_r, grad_e
