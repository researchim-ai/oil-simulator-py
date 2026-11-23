"""Fused full caloric PR properties; sensitivities use the original Torch path.

These are exact point evaluations, not a property lookup/interpolation table.
All mechanical/caloric root checks and all output properties remain present.
"""

import warnings

import torch

_COMPILED = {}
_WIDTH = 64
_PARAMETERS = ("tc", "kappa", "a0", "b", "kij", "cp", "molar_masses", "_root_angle_offsets")


def eligible(fluid, p, t, x):
    enabled = (fluid.device.type == "cpu" and fluid._compiled_full_eos_cpu) or (
        fluid.device.type == "cuda" and fluid._compiled_full_eos_cuda
    )
    if not enabled or fluid._full_eos_compile_failure is not None:
        return False
    return all(
        isinstance(v, torch.Tensor)
        and v.dtype == torch.float64
        # ``cuda`` without an index denotes the current device, while actual
        # tensors report e.g. ``cuda:0``. Compare concrete parameter placement.
        and v.device == fluid.tc.device
        and not v.requires_grad
        and not torch._C._functorch.is_functorch_wrapped_tensor(v)
        for v in (p, t, x, *(getattr(fluid, name) for name in _PARAMETERS))
    )


def _full(fluid, p, t, x, phase, derivative_order):
    return fluid._tensor_homogeneous_eager(p, t, x, phase, derivative_order=derivative_order)


def evaluate(fluid, p, t, x, phase, derivative_order):
    shape = torch.broadcast_shapes(p.shape, t.shape, x.shape[:-1])
    pressures, temperatures = p.expand(shape).reshape(-1), t.expand(shape).reshape(-1)
    compositions = x.expand(*shape, x.shape[-1]).reshape(-1, x.shape[-1])
    count = len(pressures)
    if count == 0:
        return _full(fluid, p, t, x, phase, derivative_order)
    if fluid.device.type not in _COMPILED:
        _COMPILED[fluid.device.type] = torch.compile(
            _full,
            fullgraph=True,
            dynamic=False,
            options={"use_fast_math": False, "inplace_buffers": False, "allow_buffer_reuse": False},
        )
    kernel = _COMPILED[fluid.device.type]
    pieces = []
    try:
        with torch.no_grad():
            for start in range(0, count, _WIDTH):
                values = [
                    v[start : start + _WIDTH] for v in (pressures, temperatures, compositions)
                ]
                size = len(values[0])
                if size < _WIDTH:
                    values = [
                        torch.cat((v, v[-1:].expand(_WIDTH - size, *v.shape[1:]))) for v in values
                    ]
                pp, tt, xx = [v.contiguous() for v in values]
                answer = kernel(fluid, pp, tt, xx, phase, derivative_order)
                pieces.append({k: v[:size] for k, v in answer.items()})
                fluid._full_eos_compiled_batches += 1
                fluid._full_eos_compiled_points += size
    except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as exc:
        fluid._full_eos_compile_failure = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        warnings.warn(
            f"Full EOS compiler unavailable; exact eager {fluid.device.type.upper()} retained: "
            + fluid._full_eos_compile_failure,
            RuntimeWarning,
            stacklevel=2,
        )
        return _full(fluid, p, t, x, phase, derivative_order)
    result = {k: torch.cat([piece[k] for piece in pieces]) for k in pieces[0]}
    restored = {
        k: v.reshape(*shape, *v.shape[1:]) if shape or v.ndim > 1 else v.reshape(())
        for k, v in result.items()
    }
    # Preserve the eager API for a composition broadcast over several P/T
    # points: these two fields retain the ORIGINAL composition's shape.
    restored["mole_fractions"] = x
    restored["molar_mass_kg_mol"] = (x * fluid.molar_masses).sum(-1)
    return restored
