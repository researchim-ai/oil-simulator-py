"""Exact quadratic PR mixing without per-point component-pair matrices.

Only immutable model coefficients are prepared. Temperature, composition and
their derivatives are evaluated at every trial; this is not a property table.
"""

import torch


def _coefficients(fluid):
    parameters = (fluid.a0, fluid.kappa, fluid.tc, fluid.kij)
    cacheable = not torch.compiler.is_compiling() and all(
        type(v) is torch.Tensor
        and not v.requires_grad
        and not torch._C._functorch.is_functorch_wrapped_tensor(v)
        for v in parameters
    )
    key = tuple((id(v), v._version) for v in parameters) if cacheable else None
    saved = getattr(fluid, "_attraction_coefficients", None) if cacheable else None
    if saved is not None and saved[0] == key:
        return saved[2]
    root_a = fluid.a0.sqrt()
    intercept = root_a * (1 + fluid.kappa)
    slope = root_a * fluid.kappa / fluid.tc.sqrt()
    interaction = 1 - fluid.kij
    # Construction normalizes kij to symmetry. Preserve the original equations
    # for in-place asymmetric parameter trials as well (e.g. sensitivities).
    symmetric = (interaction + interaction.transpose(-1, -2)) * 0.5
    result = intercept, slope, interaction, symmetric
    ordinary = cacheable and all(
        type(v) is torch.Tensor
        and not v.requires_grad
        and not torch._C._functorch.is_functorch_wrapped_tensor(v)
        for v in result
    )
    if ordinary:
        if torch.equal(interaction, symmetric):
            symmetric = interaction
        result = intercept, slope, interaction, symmetric
        # Pin objects: replacing a parameter twice must not alias a reused id.
        fluid._attraction_coefficients = key, parameters, result
    return result


def attraction(fluid, temperature, composition, *, pair_matrix=False):
    intercept, slope, interaction, symmetric = _coefficients(fluid)
    root_t = temperature.sqrt()[..., None]
    signed = intercept - root_t * slope
    u = signed.abs()
    ut = -0.5 * slope / root_t * signed.sign()
    utt = -ut / (2 * temperature[..., None])
    weighted = composition * u
    product = weighted @ interaction
    symmetric_product = product if symmetric is interaction else weighted @ symmetric
    ax = u * product
    a = (composition * ax).sum(-1)
    wt = composition * ut
    at = 2 * (wt * symmetric_product).sum(-1)
    att = 2 * ((composition * utt * symmetric_product).sum(-1) + ((wt @ symmetric) * wt).sum(-1))
    pairs = u[..., :, None] * u[..., None, :] * interaction if pair_matrix else None
    return a, at, att, ax, pairs
