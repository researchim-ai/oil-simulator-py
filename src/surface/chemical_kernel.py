"""Fused float64 CPU/CUDA EOS trials; eager AD and full acceptance stay separate."""

import types
import warnings

import torch

_compiled = None
_compiled_jacobian = None
_compiled_cpu = None
_compiled_cpu_jacobian = None
_compiled_curvature = {}
_WIDTH = 64
_SMALL_WIDTH = 8
_compiled_stability_values = {}
_STABILITY_CONTEXTS = 16


def _stability_values(fluid, pressures, temperatures, chemical, active, point):
    logits = torch.cat((point, torch.zeros_like(point[:, :1])), -1)
    logw = torch.log_softmax(logits, -1)
    weights = logw.exp()
    composition = point.new_zeros((len(point), len(fluid.components))).index_copy(
        -1, active, weights
    )
    properties = _native_batch(fluid, pressures, temperatures, composition)
    delta = logw + properties["log_fugacity_coefficients"][:, active] - chemical
    potential = (weights * delta).sum(-1)
    gradient = (weights * (delta - potential[:, None]))[:, :-1]
    residual = delta[:, :-1] - delta[:, -1:]
    return potential, gradient, residual, composition, properties


def stability_values(fluid, point, pressures, temperatures, chemical, active):
    """Fuse the existing objective and analytic gradient; never classify phases.

    None requests the reference objective. AD, CUDA and unsupported inputs
    retain it. Each 8/64-row specialization has its own bounded guard cache.
    Final stationarity/curvature/full-EOS acceptance remains with the caller.
    """
    if (
        fluid.device.type != "cpu"
        or not getattr(fluid, "_fused_stability_objective_cpu", False)
        or fluid._stability_objective_compile_failure is not None
        or not eligible(fluid, point, pressures, temperatures, chemical)
        or active.dtype != torch.long
        or active.device != point.device
        or torch._C._functorch.is_functorch_wrapped_tensor(active)
        or len(active) < 2
        or point.numel() == 0
    ):
        return None
    shape = point.shape[:-1]
    pp = torch.broadcast_to(pressures, shape).reshape(-1)
    tt = torch.broadcast_to(temperatures, shape).reshape(-1)
    mu = torch.broadcast_to(chemical, (*shape, len(active))).reshape(-1, len(active))
    vv = point.reshape(-1, point.shape[-1])
    count = len(vv)
    pieces = []
    try:
        for start in range(0, count, _WIDTH):
            values = [v[start : start + _WIDTH] for v in (pp, tt, mu, vv)]
            size = len(values[0])
            width = _SMALL_WIDTH if fluid._compact_chemical_cpu and size <= _SMALL_WIDTH else _WIDTH
            key = (len(fluid.components), len(active), width)
            if key not in _compiled_stability_values:
                if len(_compiled_stability_values) >= _STABILITY_CONTEXTS:
                    fluid._stability_objective_compile_failure = "operator_admission"
                    return None
                frame = types.FunctionType(
                    _stability_values.__code__.replace(),
                    _stability_values.__globals__,
                    _stability_values.__name__,
                )
                _compiled_stability_values[key] = torch.compile(
                    frame, fullgraph=True, dynamic=False, options={"use_fast_math": False}
                )
            if size < width:
                values = [torch.cat((v, v[-1:].expand(width - size, *v.shape[1:]))) for v in values]
            p, t, m, v = [value.contiguous() for value in values]
            answer = _compiled_stability_values[key](fluid, p, t, m, active, v)
            pieces.append(
                tuple(value[:size] for value in answer[:4])
                + ({name: value[:size] for name, value in answer[4].items()},)
            )
            fluid._stability_objective_compiled_batches += 1
            fluid._stability_objective_compiled_points += size
    except (
        torch._dynamo.exc.BackendCompilerFailed,
        torch._dynamo.exc.Unsupported,
        torch._dynamo.exc.TorchRuntimeError,
        torch._dynamo.exc.FailOnRecompileLimitHit,
    ) as exc:
        from .native_execution import _propagate_cancellation

        _propagate_cancellation(exc)
        fluid._stability_objective_compile_failure = (
            f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        )
        warnings.warn(
            "Fused TPD objective unavailable; original objective retained: "
            + fluid._stability_objective_compile_failure,
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    values = tuple(torch.cat([piece[i] for piece in pieces]) for i in range(4))
    properties = {name: torch.cat([piece[4][name] for piece in pieces]) for name in pieces[0][4]}
    return (
        values[0].reshape(shape),
        values[1].reshape(*shape, point.shape[-1]),
        values[2].reshape(*shape, point.shape[-1]),
        values[3].reshape(*shape, len(fluid.components)),
        {name: value.reshape((*shape, *value.shape[1:])) for name, value in properties.items()},
    )


def _batch(fluid, pressures, temperatures, compositions):
    return torch.vmap(lambda p, t, x: fluid.tensor_homogeneous(p, t, x, _chemical_only=True))(
        pressures, temperatures, compositions
    )


def _native_batch(fluid, pressures, temperatures, compositions):
    return fluid.tensor_homogeneous(pressures, temperatures, compositions, _chemical_only=True)


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


def _autodiff_native_jacobian(fluid, pressures, temperatures, chemical, active, point):
    """Independent small composition pullbacks, no global batch Jacobian.

    A shared cotangent in coordinate j differentiates each row's j-th residual:
    the native EOS has no cross-row coupling. Storage is B*(C-1)^2 rather
    than a dense (B*(C-1))^2 Jacobian.
    """

    if point.shape[-1] == 0:
        return point.new_empty((point.shape[0], 0, 0))

    def residual(v):
        logits = torch.cat((v, v.new_zeros((*v.shape[:-1], 1))), -1)
        logw = torch.log_softmax(logits, -1)
        x = v.new_zeros((*v.shape[:-1], len(fluid.components))).index_copy(-1, active, logw.exp())
        q = fluid.tensor_homogeneous(pressures, temperatures, x, _chemical_only=True)
        delta = logw + q["log_fugacity_coefficients"][..., active] - chemical
        return delta[..., :-1] - delta[..., -1:]

    basis = torch.eye(point.shape[-1], dtype=point.dtype, device=point.device)
    _, pullback = torch.func.vjp(residual, point)
    return torch.stack([pullback(direction.expand_as(point))[0] for direction in basis], -2)


def _native_jacobian(fluid, pressures, temperatures, chemical, active, point, *, reverse=False):
    parameters = (fluid.tc, fluid.kappa, fluid.a0, fluid.b, fluid.kij, fluid.cp)
    if fluid._analytic_chemical and (
        torch.compiler.is_compiling()
        or (
            not torch.is_grad_enabled()
            and all(
                not v.requires_grad and not torch._C._functorch.is_functorch_wrapped_tensor(v)
                for v in (pressures, temperatures, chemical, point, *parameters)
            )
        )
    ):
        from .pr_chemical_derivatives import logit_jacobian

        return logit_jacobian(fluid, pressures, temperatures, chemical, active, point)
    return _autodiff_native_jacobian(fluid, pressures, temperatures, chemical, active, point)


def jacobian(fluid, pressures, temperatures, chemical, active, point):
    """Native CPU/compiled CUDA reverse AD; scalar forward AD reference."""
    global _compiled_jacobian, _compiled_cpu_jacobian
    if point.shape[0] == 0:
        return point.new_empty((0, point.shape[-1], point.shape[-1]))
    cpu = fluid.device.type == "cpu"
    eager = _native_jacobian if cpu else _jacobian
    if point.shape[-1] == 0:
        return point.new_empty((point.shape[0], 0, 0))
    if not eligible(fluid, pressures, temperatures, chemical, point) or (
        fluid._chemical_jacobian_compile_failure is not None
    ):
        return eager(fluid, pressures, temperatures, chemical, active, point)
    if cpu:
        if _compiled_cpu_jacobian is None:
            _compiled_cpu_jacobian = torch.compile(
                _native_jacobian, fullgraph=True, dynamic=False, options={"use_fast_math": False}
            )
        kernel = _compiled_cpu_jacobian
    else:
        if _compiled_jacobian is None:
            _compiled_jacobian = torch.compile(
                _native_jacobian, fullgraph=True, dynamic=False, options={"use_fast_math": False}
            )
        kernel = _compiled_jacobian
    count = pressures.numel()
    pieces = []
    try:
        for start in range(0, count, _WIDTH):
            rows = [v[start : start + _WIDTH] for v in (pressures, temperatures, chemical, point)]
            size = rows[0].numel()
            if size < _WIDTH:
                rows = [torch.cat((v, v[-1:].expand(_WIDTH - size, *v.shape[1:]))) for v in rows]
            p, t, mu, v = [row.contiguous() for row in rows]
            answer = (
                kernel(fluid, p, t, mu, active, v)
                if cpu
                else kernel(fluid, p, t, mu, active, v, reverse=True)
            )
            pieces.append(answer[:size])
            fluid._chemical_jacobian_compiled_batches += 1
            fluid._chemical_jacobian_compiled_points += size
    except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as exc:
        fluid._chemical_jacobian_compile_failure = (
            f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        )
        warnings.warn(
            f"Compiled chemical derivative unavailable; eager {fluid.device.type.upper()} retained: "
            + fluid._chemical_jacobian_compile_failure,
            RuntimeWarning,
            stacklevel=2,
        )
        return eager(fluid, pressures, temperatures, chemical, active, point)
    return torch.cat(pieces)


def eligible(fluid, *values):
    if (
        (
            fluid.device.type != "cuda"
            and not (fluid.device.type == "cpu" and fluid._compiled_chemical_cpu)
        )
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
    """Plain tensors only; reusable 8/64-row CPU or 64-row CUDA kernels."""
    global _compiled, _compiled_cpu
    count = pressures.numel()
    if not count:
        return {
            "log_fugacity_coefficients": compositions.new_empty((0, compositions.shape[-1])),
            "valid": pressures.new_empty((0,), dtype=torch.bool),
        }
    cpu = fluid.device.type == "cpu"
    if cpu:
        if _compiled_cpu is None:
            _compiled_cpu = torch.compile(
                _native_batch, fullgraph=True, dynamic=False, options={"use_fast_math": False}
            )
        kernel = _compiled_cpu
    else:
        if _compiled is None:
            _compiled = torch.compile(
                _batch, fullgraph=True, dynamic=False, options={"use_fast_math": False}
            )
        kernel = _compiled
    pieces = []
    try:
        for start in range(0, count, _WIDTH):
            p = pressures[start : start + _WIDTH]
            t = temperatures[start : start + _WIDTH]
            x = compositions[start : start + _WIDTH]
            size = p.numel()
            width = (
                _SMALL_WIDTH
                if cpu and fluid._compact_chemical_cpu and size <= _SMALL_WIDTH
                else _WIDTH
            )
            if size < width:
                padding = width - size
                p = torch.cat((p, p[-1:].expand(padding)))
                t = torch.cat((t, t[-1:].expand(padding)))
                x = torch.cat((x, x[-1:].expand(padding, -1)))
            answer = kernel(fluid, p.contiguous(), t.contiguous(), x.contiguous())
            pieces.append({key: value[:size] for key, value in answer.items()})
            fluid._chemical_compiled_batches += 1
            fluid._chemical_compiled_points += size
            fluid._chemical_padded_points += width
    except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as exc:
        # Compiler availability is not a physical domain failure. Remain on
        # the SAME device, and expose the loss of acceleration explicitly.
        fluid._chemical_compile_failure = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        warnings.warn(
            f"Compiled chemical EOS unavailable; eager {fluid.device.type.upper()} retained: {fluid._chemical_compile_failure}",
            RuntimeWarning,
            stacklevel=2,
        )
        return (_native_batch if cpu else _batch)(fluid, pressures, temperatures, compositions)
    return {key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0]}


def curvature(fluid, pressures, temperatures, chemical, active, root, basis):
    """Exact local Gibbs curvature with opt-in CPU/default CUDA fusion."""
    from .pr_chemical_derivatives import tangent_hessian

    if (
        not eligible(fluid, pressures, temperatures, chemical, root, basis)
        or fluid._chemical_curvature_compile_failure is not None
        or not len(root)
    ):
        return tangent_hessian(fluid, pressures, temperatures, chemical, active, root, basis)
    device = fluid.device.type
    if device not in _compiled_curvature:
        _compiled_curvature[device] = torch.compile(
            tangent_hessian, fullgraph=True, dynamic=False, options={"use_fast_math": False}
        )
    pieces = []
    try:
        for start in range(0, len(root), _WIDTH):
            values = [
                v[start : start + _WIDTH] for v in (pressures, temperatures, chemical, root, basis)
            ]
            count = len(values[0])
            if count < _WIDTH:
                values = [
                    torch.cat((v, v[-1:].expand(_WIDTH - count, *v.shape[1:]))) for v in values
                ]
            p, t, mu, r, b = [v.contiguous() for v in values]
            pieces.append(_compiled_curvature[device](fluid, p, t, mu, active, r, b)[:count])
            fluid._chemical_curvature_compiled_batches += 1
    except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as exc:
        fluid._chemical_curvature_compile_failure = (
            f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        )
        warnings.warn(
            f"Compiled curvature unavailable; eager {device.upper()} retained: "
            + fluid._chemical_curvature_compile_failure,
            RuntimeWarning,
            stacklevel=2,
        )
        return tangent_hessian(fluid, pressures, temperatures, chemical, active, root, basis)
    return torch.cat(pieces)
