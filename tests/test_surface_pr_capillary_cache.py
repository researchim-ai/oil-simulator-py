"""Exact transaction-local reuse must preserve every accepted physical result."""

import pickle

import pytest
from check_dynamic_receiver import audit_step
from test_fim_dynamic_pr_capillary_surface import build
from test_surface_pr_capillary_facility import setup


def test_cache_is_bounded_exact_and_scoped_and_does_not_reuse_changed_eos():
    model, old = setup(self_loop=True)
    with model.cached_inspections(maximum_entries=1) as cache:
        first = model.inspect(old)
        same = model.inspect(old)
        assert cache.hits == 1 and same["out"] is first["out"]
        same.clear()  # Callers own the mapping, never the cache registry.
        assert set(model.inspect(old)) == set(model.lines)
        # Same numeric tensor, new version: a real mutation cannot reuse the entry.
        model.columns["A"].fluid.kij.add_(0.0)
        before = cache.misses
        model.inspect(old)
        assert cache.misses == before + 1
        # Mutated returned physical models must be reconstructed from the real inputs.
        models = model.inspect(old)
        models["out"].boundaries[0].clear()
        rebuilt = model.inspect(old)
        assert rebuilt["out"].boundaries[0]
        hot, _ = model.step(old, 1e-7, heaters={"A": {"power_w": 1e11, "temperature_k": 450}})
        model.inspect(hot)
        assert len(cache.entries) <= 1 and cache.evictions > 0
    assert not cache.entries and not hasattr(model, "_active_inspection_cache")
    with pytest.raises(RuntimeError), model.cached_inspections() as failed:
        model.inspect(old)
        raise RuntimeError("transaction failed")
    assert not failed.entries and not hasattr(model, "_active_inspection_cache")


def test_outer_disabled_scope_is_respected_by_nested_scopes():
    model, old = setup(self_loop=True)
    with model.cached_inspections(enabled=False), model.cached_inspections() as cache:
        assert cache is None
        a, b = model.inspect(old), model.inspect(old)
        assert a["out"] is not b["out"]
    assert not hasattr(model, "_active_inspection_cache")


def test_cached_and_uncached_longer_fim_transaction_are_physically_identical():
    model = build()
    reference = pickle.loads(pickle.dumps(model))
    old = model.receiver_state
    assert model.step(1e-6), model.last_attempt
    with reference.receiver.capillary.cached_inspections(enabled=False):
        assert reference.step(1e-6), reference.last_attempt
    assert model.receiver_state == reference.receiver_state
    actual, expected = dict(model.report), dict(reference.report)
    stats = actual.pop("boundary_inspections")
    actual.pop("property_evaluations")
    expected.pop("property_evaluations")
    assert actual == expected
    assert stats["hits"] > 0 and stats["entries"] <= stats["maximum_entries"]
    assert model.report["temporal_error_ratio"] <= 1
    assert not hasattr(model.receiver.capillary, "_active_inspection_cache")
    assert audit_step({**model.to_dict(), "initial_state": old.to_dict()}, model.report)["accepted"]
