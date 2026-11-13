"""Exact transaction reuse must keep every distinct TPD request distinct."""

import math
from contextlib import ExitStack

import pytest
import torch
from test_surface_equilibrium_thermodynamics import fluid
from test_surface_pr_surface_control import controlled

from surface.equilibrium_cache import cached_flashes, cached_stability_checks


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_tpd_exact_keys_owned_outputs_and_model_invalidation(monkeypatch, device):
    f = fluid(device)
    p, t, z = f.tensor(1e6), f.tensor(320.0), f.tensor([0.5, 0.5])
    q = f.tensor_homogeneous(p, t, z)
    mu, active = z.log() + q["log_fugacity_coefficients"], torch.nonzero(z > 0).flatten()
    calls = []
    original = f._stability_predictors

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(f, "_stability_predictors", counted)
    with cached_stability_checks(f, maximum_entries=2) as cache:
        first = f._stability(p, t, z, mu, active)
        first[1].zero_()
        second = f._stability(p.clone(), t.clone(), z.clone(), mu.clone(), active.clone())
        assert len(calls) == 1 and cache.hits == 1
        assert float(second[1].sum()) == pytest.approx(1.0)
        f._stability(f.tensor(math.nextafter(1e6, math.inf)), t, z, mu, active)
        assert len(calls) == 2
        changed = mu.clone()
        changed[0] = math.nextafter(float(mu[0]), math.inf)
        f._stability(p, t, z, changed, active)
        assert len(calls) == 3
        f.cp.add_(0.0)
        f._stability(p, t, z, changed, active)
        assert len(calls) == 4 and cache.evictions == 2
    assert not cache.entries and not hasattr(f, "_active_stability_cache")


def test_tpd_gradients_disable_and_exception_do_not_leave_scratch():
    f = fluid()
    p, t, z = f.tensor(1e6), f.tensor(320.0), f.tensor([0.5, 0.5])
    q = f.tensor_homogeneous(p, t, z)
    mu, active = z.log() + q["log_fugacity_coefficients"], torch.nonzero(z > 0).flatten()
    with cached_stability_checks(f) as cache:
        f._stability(p, t, z.clone().requires_grad_(), mu, active)
        assert cache.hits == cache.misses == 0
    with pytest.raises(RuntimeError, match="reject"), cached_stability_checks(f) as cache:
        f._stability(p, t, z, mu, active)
        raise RuntimeError("reject")
    assert not cache.entries and not hasattr(f, "_active_stability_cache")
    with cached_stability_checks(f, enabled=False), cached_stability_checks(f) as disabled:
        assert disabled.statistics() == {"enabled": False}
        f._stability(p, t, z, mu, active)
    assert not hasattr(f, "_active_stability_cache")


def test_controlled_transaction_exactly_matches_disabled_caches():
    model, initial = controlled(feedback=True)
    fluids = (model.plant.transport.fluid, *(c.fluid for c in model.plant.columns.values()))
    with ExitStack() as stack:
        for f in fluids:
            stack.enter_context(cached_flashes(f, enabled=False))
            stack.enter_context(cached_stability_checks(f, enabled=False))
        expected = model.advance(initial, 1e-7)
    actual = model.advance(initial, 1e-7)
    assert actual == expected
    for f in fluids:
        assert not hasattr(f, "_active_flash_cache")
        assert not hasattr(f, "_active_stability_cache")


def test_tpd_invalid_index_type_cannot_alias_a_valid_request():
    f = fluid()
    p, t, z = f.tensor(1e6), f.tensor(320.0), f.tensor([0.5, 0.5])
    q = f.tensor_homogeneous(p, t, z)
    mu, active = z.log() + q["log_fugacity_coefficients"], torch.nonzero(z > 0).flatten()
    with cached_stability_checks(f) as cache:
        f._stability(p, t, z, mu, active)
        before = (cache.hits, cache.misses)
        with pytest.raises((IndexError, RuntimeError, TypeError)):
            f._stability(p, t, z, mu, active.double())
        assert (cache.hits, cache.misses) == before
        # A different precision executes its own original numerical request.
        f._stability(p.float(), t.float(), z.float(), mu, active)
        assert (cache.hits, cache.misses) == before


def test_cached_parameter_identity_cannot_be_recycled_before_eviction():
    import gc
    import weakref

    f = fluid()
    with cached_flashes(f):
        first = f.flash_pt(1e6, 320.0, [0.5, 0.5])
        old = weakref.ref(f.cp)
        f.cp = f.cp + 1.0
        gc.collect()
        assert old() is not None  # The old identity cannot be recycled into another key.
        second = f.flash_pt(1e6, 320.0, [0.5, 0.5])
        assert not torch.equal(first["molar_enthalpy_j_mol"], second["molar_enthalpy_j_mol"])
    gc.collect()
    assert old() is None


def test_forward_ad_model_parameters_bypass_property_memoization():
    from surface.equilibrium_cache import _signature

    f = fluid()
    original = f.cp

    def probe(cp):
        f.cp = cp
        try:
            with pytest.raises(TypeError, match="sensitivities"):
                _signature(f)
        finally:
            f.cp = original
        return cp

    torch.func.jacfwd(probe)(original)
