"""Bounded, exact, transaction-scoped reuse of already validated flashes.

No rounded keys, phase-region extrapolation, detached AD or persistent physics
cache. Returned tensor trees own their storage. Any input/model change misses.
"""

from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from functools import wraps
from inspect import signature

import torch

_MISSING = object()
_MODEL_TENSORS = ("tc", "pc", "cp", "kappa", "b", "a0", "kij", "molar_masses")


def _copy(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy(v) for v in value)
    return value


def _key(value):
    if torch.is_tensor(value):
        if value.requires_grad or torch._C._functorch.is_functorch_wrapped_tensor(value):
            raise TypeError("Differentiable requests bypass flash memoization")
        # Public flashes convert all numeric inputs to fluid float64/device.
        # A scalar tensor and the identical Python float are the SAME request;
        # retain every bit of the value and every nested shape, without bins.
        return _key(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return tuple((k, _key(v)) for k, v in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_key(v) for v in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("Unsupported flash key")


def _signature(fluid):
    tensors = _MODEL_TENSORS
    if any(
        getattr(fluid, name).requires_grad
        or torch._C._functorch.is_functorch_wrapped_tensor(getattr(fluid, name))
        for name in tensors
    ):
        raise TypeError("Model-parameter sensitivities bypass flash memoization")
    return (
        fluid.model_id,
        fluid.components,
        fluid.binary_interactions,
        str(fluid.device),
        fluid.min_pressure_pa,
        fluid.max_pressure_pa,
        fluid.min_temperature_k,
        fluid.max_temperature_k,
        fluid.tref,
        fluid.pref,
        fluid.max_phases,
        tuple((id(getattr(fluid, n)), getattr(fluid, n)._version) for n in tensors),
    )


def validated_state_key(owner, state, *, scheduling_fields=()):
    """Key a physical audit by exact stocks and the current EOS parameters.

    Callers must validate scheduling metadata before omitting it. Keep model
    tensors alive so replacement cannot reuse an old tensor id; in-place
    changes invalidate through the version counters. AD requests never reuse
    detached validation decisions.
    """
    from .operation import _digest

    try:
        signature = _signature(owner.fluid)
    except (TypeError, RuntimeError):
        owner._verified.clear()
        owner._verified_signature = None
        owner._verified_model_tensors = ()
        return None
    if signature != getattr(owner, "_verified_signature", None):
        owner._verified.clear()
        owner._verified_signature = signature
        owner._verified_model_tensors = tuple(getattr(owner.fluid, name) for name in _MODEL_TENSORS)
    data = state.to_dict()
    for name in scheduling_fields:
        data.pop(name)
    return _digest(data)


class FlashCache:
    def __init__(self, maximum_entries):
        self.maximum_entries = maximum_entries
        self.entries = OrderedDict()
        self.warm_equilibria = OrderedDict()
        self.transport_prefixes = OrderedDict()
        self.pump_valve_seeds = OrderedDict()
        self.transport_prefix_hits = self.transport_prefix_steps = 0
        self.warm_flash_attempts = self.warm_flash_accepts = self.warm_flash_fallbacks = 0
        self.hits = self.misses = self.evictions = 0
        self.caloric_fast_accepts = self.caloric_bracket_fallbacks = 0
        self.stability_seeds = self.stability_batched_seeds = 0
        self.spatial_stability_batches = self.spatial_stability_points = 0
        self.spatial_stability_confirmed = 0

    def statistics(self):
        return {
            "enabled": True,
            "transport_prefix_hits": self.transport_prefix_hits,
            "transport_prefix_macrosteps": self.transport_prefix_steps,
            "warm_flash_attempts": self.warm_flash_attempts,
            "warm_flash_accepts": self.warm_flash_accepts,
            "warm_flash_fallbacks": self.warm_flash_fallbacks,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "entries": len(self.entries),
            "maximum_entries": self.maximum_entries,
            "caloric_fast_accepts": self.caloric_fast_accepts,
            "caloric_bracket_fallbacks": self.caloric_bracket_fallbacks,
            "stability_seeds": self.stability_seeds,
            "stability_batched_seeds": self.stability_batched_seeds,
            "stability_reference_seeds": self.stability_seeds - self.stability_batched_seeds,
            "spatial_stability_batches": self.spatial_stability_batches,
            "spatial_stability_points": self.spatial_stability_points,
            "spatial_stability_confirmed": self.spatial_stability_confirmed,
        }


class _Disabled:
    def statistics(self):
        return {"enabled": False}


@contextmanager
def cached_flashes(fluid, *, enabled=True, maximum_entries=256):
    if type(enabled) is not bool or type(maximum_entries) is not int or maximum_entries < 1:
        raise ValueError("Explicit flash-cache switch and positive entry budget required")
    previous = getattr(fluid, "_active_flash_cache", _MISSING)
    if previous is not _MISSING and enabled:
        yield previous
        return
    cache = FlashCache(maximum_entries) if enabled else _Disabled()
    fluid._active_flash_cache = cache
    try:
        yield cache
    finally:
        if previous is _MISSING:
            del fluid._active_flash_cache
        else:
            fluid._active_flash_cache = previous
        if isinstance(cache, FlashCache):
            cache.entries.clear()
            cache.warm_equilibria.clear()
            cache.transport_prefixes.clear()
            cache.pump_valve_seeds.clear()


def _request_key(fluid, name, arguments, *, model_signature=None):
    return (
        _signature(fluid) if model_signature is None else model_signature,
        name,
        _key(arguments),
    )


def _cached_request(function, attribute):
    call_signature = signature(function)

    @wraps(function)
    def wrapped(fluid, *args, **kwargs):
        cache = getattr(fluid, attribute, None)
        if not isinstance(cache, FlashCache):
            return function(fluid, *args, **kwargs)
        try:
            bound = call_signature.bind(fluid, *args, **kwargs)
            bound.apply_defaults()
            arguments = {k: v for k, v in bound.arguments.items() if k != "self"}
            key = _request_key(fluid, function.__name__, arguments)
        except (TypeError, RuntimeError):
            return function(fluid, *args, **kwargs)
        if key in cache.entries:
            cache.hits += 1
            cache.entries.move_to_end(key)
            return _copy(cache.entries[key][0])
        cache.misses += 1
        result = function(fluid, *args, **kwargs)

        # Public flash interfaces are detached physical solves. Conservatively
        # bypass storage if a future implementation returns an AD graph.
        def differentiable(v):
            if torch.is_tensor(v):
                return v.requires_grad
            if isinstance(v, dict):
                return any(differentiable(x) for x in v.values())
            if isinstance(v, (tuple, list)):
                return any(differentiable(x) for x in v)
            return False

        if not differentiable(result):
            # Pin parameter objects while their identity/version is in a key.
            # Otherwise replacing tensors twice can reuse an old Python id and
            # alias a different EOS to a still-live cache entry.
            cache.entries[key] = (
                _copy(result),
                tuple(getattr(fluid, name) for name in _MODEL_TENSORS),
            )
            if len(cache.entries) > cache.maximum_entries:
                cache.entries.popitem(last=False)
                cache.evictions += 1
        return result

    return wrapped


def cached_equilibrium(function):
    return _cached_request(function, "_active_flash_cache")


def cached_stability_result(function):
    """Exact reuse of numerical TPD requests in a separate bounded registry."""
    cached = _cached_request(function, "_active_stability_cache")

    @wraps(function)
    def wrapped(fluid, p, t, z, chemical, active, **kwargs):
        # Public flashes normalize numeric types before solving; this private
        # TPD interface does not. Never turn an invalid index dtype/device into
        # a cache hit, or alias lower-precision inputs to float64 requests.
        normalized = all(
            isinstance(value, torch.Tensor)
            and value.dtype == torch.float64
            and value.device == fluid.molar_masses.device
            for value in (p, t, z, chemical)
        ) and (
            isinstance(active, torch.Tensor)
            and active.dtype == torch.long
            and active.device == fluid.molar_masses.device
        )
        if not normalized:
            return function(fluid, p, t, z, chemical, active, **kwargs)
        return cached(fluid, p, t, z, chemical, active, **kwargs)

    return wrapped


@contextmanager
def cached_stability_checks(fluid, *, enabled=True, maximum_entries=256):
    if type(enabled) is not bool or type(maximum_entries) is not int or maximum_entries < 1:
        raise ValueError("Explicit stability-cache switch and positive entry budget required")
    previous = getattr(fluid, "_active_stability_cache", _MISSING)
    if previous is not _MISSING and enabled:
        yield previous
        return
    cache = FlashCache(maximum_entries) if enabled else _Disabled()
    fluid._active_stability_cache = cache
    try:
        yield cache
    finally:
        if previous is _MISSING:
            del fluid._active_stability_cache
        else:
            fluid._active_stability_cache = previous
        if isinstance(cache, FlashCache):
            cache.entries.clear()


def cached_fluid_transaction(fluids):
    """Bind exact flash/TPD scratch storage to one public network transaction."""

    def decorate(function):
        @wraps(function)
        def wrapped(model, *args, **kwargs):
            with ExitStack() as stack:
                seen = set()
                for fluid in fluids(model):
                    if id(fluid) not in seen:
                        stack.enter_context(cached_flashes(fluid))
                        stack.enter_context(cached_stability_checks(fluid))
                        seen.add(id(fluid))
                return function(model, *args, **kwargs)

        return wrapped

    return decorate
