"""Immutable external thermodynamics reuse, invalidation and live operating checks."""

from dataclasses import replace

import pytest
import torch
from test_surface_compositional_separator import binary
from test_surface_pr_open_transport import build

from surface.liquid import PropertyModelError
from surface.pr_boundary_cache import PreparedPRBoundaries
from surface.pr_open_transport import _BoundaryProperties
from surface.stratified_pr_flow import ReceiverFeedPressureError


def test_exact_hits_own_storage_and_cached_none():
    fluid = binary()
    cache = PreparedPRBoundaries(fluid, 2)
    calls = []

    def evaluate():
        calls.append(1)
        return {"value": fluid.tensor([1.0, 2.0])}

    first = cache.resolve("a", evaluate)
    first["value"][0] = 99.0
    second = cache.resolve("a", evaluate)
    second["value"][1] = 88.0
    third = cache.resolve("a", evaluate)
    assert third["value"].tolist() == [1.0, 2.0]
    assert len(calls) == 1
    assert cache.resolve("zero", lambda: None) is None

    def forbidden():
        raise AssertionError("None was treated as a miss")

    assert cache.resolve("zero", forbidden) is None


def test_eos_mutation_replacement_domain_and_reference_invalidate():
    fluid = binary()
    cache = PreparedPRBoundaries(fluid, 2)
    calls = []

    def evaluate():
        calls.append(1)
        return {"q": fluid.kij.sum().detach().clone()}

    cache.resolve("same", evaluate)
    old = fluid.kij
    assert any(t is old for t in cache.parameter_refs)
    fluid.kij = fluid.kij.clone()
    cache.resolve("same", evaluate)
    with torch.no_grad():
        fluid.kij.add_(0.01)
    assert float(cache.resolve("same", evaluate)["q"]) == pytest.approx(0.04)
    fluid.max_phases = 2
    cache.resolve("same", evaluate)
    fluid.tref += 0.1
    cache.resolve("same", evaluate)
    assert len(calls) == 5 and cache.invalidations == 4


def test_failed_solve_not_cached_and_eos_mutation_during_solve_rejected():
    fluid = binary()
    cache = PreparedPRBoundaries(fluid, 2)

    def failed():
        raise RuntimeError("failed PH")

    with pytest.raises(RuntimeError):
        cache.resolve("a", failed)
    assert not cache.entries

    def changed():
        with torch.no_grad():
            fluid.kij.add_(0.01)
        return fluid.tensor(1.0)

    with pytest.raises(PropertyModelError, match="EOS changed"):
        cache.resolve("a", changed)
    assert not cache.entries


def test_bounded_lru_and_disabled_path():
    cache = PreparedPRBoundaries(binary(), 2)
    calls = []

    def value():
        calls.append(1)
        return 1.0

    for key in ("a", "b", "c"):
        cache.resolve(key, value)
    assert list(cache.entries) == ["b", "c"]
    cache.resolve("b", value)
    cache.resolve("a", value)
    assert len(calls) == 4 and len(cache.entries) == 2
    cache.resolve("a", value, enabled=False)
    cache.resolve("a", value, enabled=False)
    assert len(calls) == 6 and cache.bypasses == 2


def test_parameter_sensitivities_and_differentiable_results_bypass():
    fluid = binary()
    cache = PreparedPRBoundaries(fluid, 2)
    calls = []

    def evaluate():
        calls.append(1)
        return fluid.tensor(2.0)

    cache.resolve("x", evaluate)
    fluid.kij.requires_grad_(True)
    cache.resolve("x", evaluate)
    cache.resolve("x", evaluate)
    assert len(calls) == 3 and not cache.entries
    fluid.kij.requires_grad_(False)

    def differentiable():
        return torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

    cache.resolve("ad", differentiable)
    assert not cache.entries


def test_feed_and_reverse_reservoir_preparation_reused_across_commands(monkeypatch):
    model, _state = build(reverse=True)
    first = _BoundaryProperties(model, {"boundary": 1.0})
    entropy = first.feeds[0][-1].clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Immutable boundary was solved again")

    monkeypatch.setattr(model.fluid, "flash_ph", forbidden)
    monkeypatch.setattr(model.fluid, "flash_pt", forbidden)
    first.feeds[0][-1].add_(123.0)
    closed = _BoundaryProperties(model, {"boundary": 0.0})
    assert closed.reservoirs == [None]
    again = _BoundaryProperties(model, {"boundary": 1.0})
    assert torch.equal(again.feeds[0][-1], entropy)
    assert model._prepared_boundaries.statistics()["misses"] == 2
    assert model._prepared_boundaries.statistics()["hits"] == 3


def test_changed_feed_energy_or_pressure_has_new_key(monkeypatch):
    model, _ = build()
    _BoundaryProperties(model, {"boundary": 0.0})
    old = model.feeds[0]
    original = model.fluid.flash_ph
    calls = []

    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model.fluid, "flash_ph", observed)
    model.feeds = (
        replace(
            old,
            stream=replace(old.stream, total_energy_rate_w=old.stream.total_energy_rate_w + 0.01),
        ),
    )
    _BoundaryProperties(model, {"boundary": 0.0})
    model.feeds = (
        replace(old, stream=replace(old.stream, pressure_pa=old.stream.pressure_pa + 10.0)),
    )
    _BoundaryProperties(model, {"boundary": 0.0})
    assert len(calls) == 2
    assert len(model._prepared_boundaries.entries) <= model._prepared_boundaries.maximum_entries


def test_reused_preparation_does_not_bypass_receiving_pressure_check():
    model, state = build()
    model.advance(state, 1e-5, integrator="radau3")
    high = model.transport.initialize(
        {
            node.name: {"pressure_pa": 1.02e6, "temperature_k": 320.0, "mole_fractions": [0.5, 0.5]}
            for node in model.transport.nodes
        }
    )
    before = high.to_dict()
    with pytest.raises(ReceiverFeedPressureError):
        model.advance(high, 1e-5, integrator="radau3")
    assert high.to_dict() == before


def test_cache_not_serialized_or_part_of_physical_identity():
    model, _ = build()
    snapshot = model.to_dict()
    identity = model.model_id
    _BoundaryProperties(model, {"boundary": 1.0})
    assert model.to_dict() == snapshot and model.model_id == identity
    restored = type(model).from_dict(snapshot)
    assert not restored._prepared_boundaries.entries


def test_identical_intensive_states_share_solve_without_sharing_flow(monkeypatch):
    from surface.pr_open_transport import PRTransportFeed

    model, _ = build()
    feed = model.feeds[0]
    twice = replace(
        feed.stream,
        component_molar_rates_mol_s=tuple(2 * n for n in feed.stream.component_molar_rates_mol_s),
        total_energy_rate_w=2 * feed.stream.total_energy_rate_w,
    )
    model.feeds = (feed, PRTransportFeed("double", feed.node, twice))
    original = model.fluid.flash_ph
    calls = []

    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model.fluid, "flash_ph", observed)
    prepared = _BoundaryProperties(model, {"boundary": 0.0})
    assert len(calls) == 1
    assert torch.equal(prepared.feeds[1][2], 2 * prepared.feeds[0][2])
    assert torch.equal(prepared.feeds[1][3], 2 * prepared.feeds[0][3])
    assert torch.equal(prepared.feeds[1][4], 2 * prepared.feeds[0][4])


def test_parameter_gradient_switch_during_preparation_clears_cache():
    fluid = binary()
    cache = PreparedPRBoundaries(fluid, 2)

    def changed():
        fluid.kij.requires_grad_(True)
        return fluid.tensor(1.0)

    with pytest.raises(PropertyModelError, match="EOS changed"):
        cache.resolve("x", changed)
    assert not cache.entries and not cache.parameter_refs
