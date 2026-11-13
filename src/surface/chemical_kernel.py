"""Fused float64 CUDA EOS trials; eager AD and full acceptance stay separate."""

import warnings

import torch

_compiled = None
_compiled_jacobian = None
_WIDTH = 64


def _batch(fluid, pressures, temperatures, compositions):
    return torch.vmap(lambda p, t, x: fluid.tensor_homogeneous(p, t, x, _chemical_only=True))(
        pressures, temperatures, compositions
    )


def _jacobian(fluid, pressures, temperatures, chemical, active, point, *, reverse=False):
    def residual(v, p, t, mu):
        logits = torch.cat((v, v.new_zeros(1)))
        logw = torch.log_softmax(logits, -1)
        x = v.new_zeros(len(fluid.components)).index_copy(0, active, logw.exp())
        q = fluid.tensor_homogeneous(p, t, x, _chemical_only=True)
        delta = logw + q["log_fugacity_coefficients"][active] - mu
        return delta[:-1] - delta[-1:]

    transform = torch.func.jacrev if reverse else torch.func.jacfwd
    return torch.vmap(transform(residual))(point, pressures, temperatures, chemical)


def jacobian(fluid, pressures, temperatures, chemical, active, point):
    """Same local AD derivative; compiled reverse AD, reference forward AD."""
    global _compiled_jacobian
    if point.shape[0] == 0:
        return point.new_empty((0, point.shape[-1], point.shape[-1]))
    if not eligible(fluid, pressures, temperatures, chemical, point) or (
        fluid._chemical_jacobian_compile_failure is not None
    ):
        return _jacobian(fluid, pressures, temperatures, chemical, active, point)
    if _compiled_jacobian is None:
        _compiled_jacobian = torch.compile(
            _jacobian, fullgraph=True, dynamic=False, options={"use_fast_math": False}
        )
    count = pressures.numel()
    pieces = []
    try:
        for start in range(0, count, _WIDTH):
            rows = [v[start : start + _WIDTH] for v in (pressures, temperatures, chemical, point)]
            size = rows[0].numel()
            if size < _WIDTH:
                rows = [torch.cat((v, v[-1:].expand(_WIDTH - size, *v.shape[1:]))) for v in rows]
            p, t, mu, v = [row.contiguous() for row in rows]
            answer = _compiled_jacobian(fluid, p, t, mu, active, v, reverse=True)
            pieces.append(answer[:size])
            fluid._chemical_jacobian_compiled_batches += 1
            fluid._chemical_jacobian_compiled_points += size
    except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as exc:
        fluid._chemical_jacobian_compile_failure = (
            f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        )
        warnings.warn(
            "Compiled chemical derivative unavailable; eager CUDA retained: "
            + fluid._chemical_jacobian_compile_failure,
            RuntimeWarning,
            stacklevel=2,
        )
        return _jacobian(fluid, pressures, temperatures, chemical, active, point)
    return torch.cat(pieces)


def eligible(fluid, *values):
    if (
        fluid.device.type != "cuda"
        or not fluid._compiled_chemical
        or fluid._chemical_compile_failure is not None
        or torch.is_grad_enabled()
    ):
        return False
    parameters = (fluid.tc, fluid.kappa, fluid.a0, fluid.b, fluid.kij, fluid.cp)
    return all(
        v.dtype == torch.float64
        and v.device == fluid.tc.device
        and not v.requires_grad
        and not torch._C._functorch.is_functorch_wrapped_tensor(v)
        for v in (*values, *parameters)
    )


def evaluate(fluid, pressures, temperatures, compositions):
    """Plain tensors only. Pad independent rows to one reusable kernel shape."""
    global _compiled
    count = pressures.numel()
    if not count:
        return {
            "log_fugacity_coefficients": compositions.new_empty((0, compositions.shape[-1])),
            "valid": pressures.new_empty((0,), dtype=torch.bool),
        }
    if _compiled is None:
        _compiled = torch.compile(
            _batch, fullgraph=True, dynamic=False, options={"use_fast_math": False}
        )
    pieces = []
    try:
        for start in range(0, count, _WIDTH):
            p = pressures[start : start + _WIDTH]
            t = temperatures[start : start + _WIDTH]
            x = compositions[start : start + _WIDTH]
            size = p.numel()
            if size < _WIDTH:
                padding = _WIDTH - size
                p = torch.cat((p, p[-1:].expand(padding)))
                t = torch.cat((t, t[-1:].expand(padding)))
                x = torch.cat((x, x[-1:].expand(padding, -1)))
            answer = _compiled(fluid, p.contiguous(), t.contiguous(), x.contiguous())
            pieces.append({key: value[:size] for key, value in answer.items()})
            fluid._chemical_compiled_batches += 1
            fluid._chemical_compiled_points += size
    except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as exc:
        # Compiler availability is not a physical domain failure. Remain on
        # the SAME CUDA device, and expose the loss of acceleration explicitly.
        fluid._chemical_compile_failure = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        warnings.warn(
            f"Compiled chemical EOS unavailable; eager CUDA retained: {fluid._chemical_compile_failure}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _batch(fluid, pressures, temperatures, compositions)
    return {key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0]}
