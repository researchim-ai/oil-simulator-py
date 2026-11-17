"""Affine well starts are guesses, not substituted reservoir/network solutions."""
import copy

import pytest
import torch

from test_fim_blackoil_surface import config, solver


@pytest.mark.parametrize("target", [.1, 1.])
def test_frozen_cell_start_matches_requested_standard_rate(target):
    c = config()
    c["simulation"]["physical_surface_network"]["bhp_initialization"] = "rate_network"
    for well in c["wells"]:
        well["control_value"] = target
    s = solver(c)
    p, sw = s.fluid.pressure.flatten(), s.fluid.s_w.flatten()
    v, d = s.cell_state(p, sw, s.x, s.sat, s.ug)
    s.wells.begin_step(v, d, 0.)
    q, *_ = s.wells._production_connection_flows(v, d, s.wells.p_bh)
    rates = torch.zeros(s.wells.n_w, dtype=torch.float64).index_add_(
        0, s.wells.conn_well, (s.wells.conn_a * q).sum(1))
    targets = torch.tensor([t for _, t in s.wells.step_targets], dtype=torch.float64)
    torch.testing.assert_close(rates, targets, rtol=1e-8, atol=1e-15)
    assert bool((q >= 0).all())


def test_network_floor_is_only_a_seed_and_full_newton_still_solves():
    c = config()
    d = copy.deepcopy(c)
    d["simulation"]["physical_surface_network"]["bhp_initialization"] = "rate_network"
    previous, predictor = solver(c), solver(d)
    assert previous.step(3600.) and predictor.step(3600.)
    torch.testing.assert_close(previous.fluid.pressure, predictor.fluid.pressure, rtol=1e-8, atol=.01)
    for name in previous.wells.surface_report["wells"]:
        a, b = [s.wells.surface_report["wells"][name] for s in (previous, predictor)]
        assert a["mode"] == b["mode"]
        assert abs(a["bhp_pa"] - b["bhp_pa"]) < .1
        torch.testing.assert_close(torch.tensor(a["component_mass_rates_kg_s"]),
                                   torch.tensor(b["component_mass_rates_kg_s"]), rtol=1e-6, atol=1e-9)


def test_unknown_initialization_rejected():
    c = config()
    c["simulation"]["physical_surface_network"]["bhp_initialization"] = "unknown"
    with pytest.raises(ValueError, match="bhp_initialization"):
        solver(c)


def _initial_context(s):
    return s.cell_state(s.fluid.pressure.flatten(), s.fluid.s_w.flatten(),
                        s.x, s.sat, s.ug)


def test_validated_network_seed_is_reused_for_exact_sources(monkeypatch):
    c = config()
    c['simulation']['physical_surface_network']['bhp_initialization'] = 'rate_network'
    for w in c['wells']:
        w['control_value'] = .1
    s = solver(c)
    network = s.wells.surface_network
    original, calls = network.solve, []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(network, 'solve', counted)
    v, d = _initial_context(s)
    s.wells.begin_step(v, d, 0.)
    assert len(calls) == 1 and s.wells._initial_surface_result is not None
    p, sw = s.fluid.pressure.flatten(), s.fluid.s_w.flatten()
    old = s.masses(p, sw, s.x, s.sat, s.ug)
    s.assemble_bo(p, sw, s.x, s.sat, old, 3600., s.wells.p_bh, ug=s.ug)
    assert len(calls) == 1
    s.assemble_bo(p, sw, s.x, s.sat, old, 3600., s.wells.p_bh - 1., ug=s.ug)
    assert len(calls) == 2


def test_predictor_domain_failure_falls_back_but_bad_properties_propagate(monkeypatch):
    from simulator.surface_seed import initial_network_seed
    from surface import PropertyModelError
    from surface.blackoil_pipe import ChokedFlowError
    s = solver()
    v, d = _initial_context(s)
    s.wells.begin_step(v, d, 0.)
    fallback = s.wells.p_bh.clone()
    proposed = fallback - 1.
    def fail(*args, **kwargs):
        raise ChokedFlowError('guessed flow chokes')
    monkeypatch.setattr(s.wells.surface_network, 'solve', fail)
    selected, cache = initial_network_seed(s.wells, v, d, proposed, fallback)
    torch.testing.assert_close(selected, fallback)
    assert cache is None
    def corrupt(*args, **kwargs):
        raise PropertyModelError('invalid PVT')
    monkeypatch.setattr(s.wells.surface_network, 'solve', corrupt)
    with pytest.raises(PropertyModelError, match='invalid PVT'):
        initial_network_seed(s.wells, v, d, proposed, fallback)


@pytest.mark.parametrize('change', ['source', 'network'])
def test_seed_cache_cannot_be_reused_for_changed_sources_or_network(monkeypatch, change):
    c = config()
    c['simulation']['physical_surface_network']['bhp_initialization'] = 'rate_network'
    for w in c['wells']:
        w['control_value'] = .1
    s = solver(c)
    v, d = _initial_context(s)
    s.wells.begin_step(v, d, 0.)
    assert s.wells._initial_surface_result is not None
    if change == 'network':
        s.wells.surface_network = copy.copy(s.wells.surface_network)
    network = s.wells.surface_network
    original, calls = network.solve, []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(network, 'solve', counted)
    p, sw = s.fluid.pressure.flatten(), s.fluid.s_w.flatten()
    old = s.masses(p, sw, s.x, s.sat, s.ug)
    pb = s.wells.p_bh - (1. if change == 'source' else 0.)
    s.assemble_bo(p, sw, s.x, s.sat, old, 3600., pb, ug=s.ug)
    assert len(calls) == 1 and s.wells._initial_surface_result is None
