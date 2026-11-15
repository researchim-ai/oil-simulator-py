"""Bounded exact preparation of immutable external PR boundary packets.

Only prescribed feed/reservoir thermodynamics are reusable across intervals;
finite vessel states, flows and operating checks are never cached here.
"""

from collections import OrderedDict

import torch

from .equilibrium_cache import _MODEL_TENSORS, _copy, _signature
from .liquid import PropertyModelError


def _has_ad(value):
    if torch.is_tensor(value):
        return value.requires_grad or torch._C._functorch.is_functorch_wrapped_tensor(value)
    if isinstance(value, dict):
        return any(_has_ad(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(_has_ad(v) for v in value)
    return False


class PreparedPRBoundaries:
    def __init__(self, fluid, maximum_entries):
        if type(maximum_entries) is not int or maximum_entries < 1:
            raise ValueError("Positive boundary-cache capacity required")
        self.fluid = fluid
        self.maximum_entries = maximum_entries
        self.entries = OrderedDict()
        self.signature = None
        self.parameter_refs = ()
        self.hits = self.misses = self.bypasses = self.invalidations = 0

    def clear(self):
        self.entries.clear()
        self.signature = None
        self.parameter_refs = ()

    def resolve(self, key, evaluate, *, enabled=True):
        if not enabled:
            self.bypasses += 1
            return evaluate()
        try:
            signature = _signature(self.fluid)
        except (TypeError, RuntimeError):
            self.clear()
            self.bypasses += 1
            return evaluate()
        if signature != self.signature:
            if self.signature is not None:
                self.invalidations += 1
            self.clear()
            self.signature = signature
            # Keep old parameter objects alive while ids/versions are keys.
            self.parameter_refs = tuple(getattr(self.fluid, n) for n in _MODEL_TENSORS)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return _copy(self.entries[key])
        self.misses += 1
        value = evaluate()
        try:
            unchanged = _signature(self.fluid) == signature
        except (TypeError, RuntimeError):
            unchanged = False
        if not unchanged:
            self.invalidations += 1
            self.clear()
            raise PropertyModelError("EOS changed while preparing external boundary")
        if not _has_ad(value):
            self.entries[key] = _copy(value)
            if len(self.entries) > self.maximum_entries:
                self.entries.popitem(last=False)
        else:
            self.bypasses += 1
        return value

    def statistics(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "bypasses": self.bypasses,
            "invalidations": self.invalidations,
            "entries": len(self.entries),
            "maximum_entries": self.maximum_entries,
        }
