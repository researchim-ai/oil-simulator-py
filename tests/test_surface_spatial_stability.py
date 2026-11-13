"""Spatial TPD shares predictors, never drops unresolved minima or phases."""

import pytest
import torch
from test_surface_compositional_separator import ternary
from test_surface_equilibrium_thermodynamics import fluid
from test_surface_liquid_mixture_pump import mixture_setup

from surface import equilibrium_properties as properties
from surface.equilibrium_cache import cached_stability_checks

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def request(f, p, t, z, *, equilibrium=False):
    p, t, z = map(f.tensor, (p, t, z))
    q = f.flash_pt(p, t, z)["phases"][0] if equilibrium else f.tensor_homogeneous(p, t, z)
    x = q["mole_fractions"]
    mu = x.clamp_min(1e-300).log() + q["log_fugacity_coefficients"]
    return p, t, z, mu, torch.nonzero(z > 0).flatten()


def assert_same_search(actual, expected):
    assert not isinstance(actual, Exception)
    assert float(actual[0].cpu()) <= float(expected[0].cpu()) + 1e-8
    assert (float(actual[0].cpu()) < -1e-8) == (float(expected[0].cpu()) < -1e-8)
    assert float(actual[1].sum().cpu()) == pytest.approx(1.0, abs=1e-12)
    assert bool(actual[2]["valid"].cpu())


@pytest.mark.parametrize("device", DEVICES)
def test_spatial_search_keeps_stable_unstable_and_equilibrium_requests(device):
    f = fluid(device)
    requests = [
        request(f, 1e6, 320.0, [0.2, 0.8]),
        request(f, 1e6, 196.2044254, [0.5, 0.5]),
        request(f, 1e6, 196.2044254, [0.5, 0.5], equilibrium=True),
        request(f, 1e6, 320.0, [1 - 1e-12, 1e-12]),
    ]
    actual = f._stability_many(requests, maximum_seed_rows=14)
    for q, row in zip(actual, requests):
        expected = f._stability(*row, _accelerated=False)
        assert_same_search(q, expected)
    assert not hasattr(f, "_spatial_stability_predictor")


@pytest.mark.parametrize("device", DEVICES)
def test_support_groups_pure_points_and_owned_duplicate_results(device):
    f = ternary(device=device)
    rows = [
        request(f, 1e6, 350.0, [0.2, 0.8, 0.0]),
        request(f, 1.01e6, 351.0, [0.2, 0.8, 0.0]),
        request(f, 1e6, 350.0, [0.0, 0.5, 0.5]),
        request(f, 1.01e6, 351.0, [0.0, 0.5, 0.5]),
        request(f, 1e6, 350.0, [0.0, 1.0, 0.0]),
    ]
    rows.append(rows[0])
    output = f._stability_many(rows)
    for q, row in zip(output, rows):
        assert_same_search(q, f._stability(*row))
    output[0][1].zero_()
    assert float(output[-1][1].sum().cpu()) == pytest.approx(1.0)


def test_warm_exact_keys_skip_all_predictor_work(monkeypatch):
    f = fluid()
    rows = [request(f, 1e6, 320.0, [0.2, 0.8]), request(f, 1.1e6, 330.0, [0.3, 0.7])]
    with cached_stability_checks(f) as cache:
        expected = [f._stability(*row) for row in rows]

        def forbidden(*args, **kwargs):
            raise AssertionError("Warm exact requests reached predictor kernel")

        monkeypatch.setattr(properties, "stability_candidates", forbidden)
        actual = f._stability_many(rows)
        assert cache.hits == 2
        for q, ref in zip(actual, expected):
            torch.testing.assert_close(q[0], ref[0], rtol=0, atol=0)


def test_unresolved_spatial_seeds_all_use_reference_minimizers(monkeypatch):
    f = fluid()
    rows = [request(f, 1e6, 320.0, [0.2, 0.8]), request(f, 1.1e6, 330.0, [0.3, 0.7])]
    original = properties.stability_candidates
    calls = []
    step = torch.optim.LBFGS.step

    def unresolved(*args, **kwargs):
        value, x, q, mask = original(*args, **kwargs)
        return value, x, q, torch.zeros_like(mask)

    def counted(self, *args, **kwargs):
        calls.append(True)
        return step(self, *args, **kwargs)

    monkeypatch.setattr(properties, "stability_candidates", unresolved)
    monkeypatch.setattr(torch.optim.LBFGS, "step", counted)
    result = f._stability_many(rows)
    assert len(calls) == 2 * (3 + 2 * 2)
    assert all(not isinstance(q, Exception) for q in result)


def test_spatial_factorization_failure_restores_original_predictors(monkeypatch):
    f = fluid()
    rows = [request(f, 1e6, 320.0, [0.2, 0.8]), request(f, 1.1e6, 330.0, [0.3, 0.7])]
    original = properties.stability_candidates
    failures = []

    def broken(fluid, p, *args, **kwargs):
        if len(p) > 7:
            failures.append(True)
            raise torch.linalg.LinAlgError("injected spatial factorization")
        return original(fluid, p, *args, **kwargs)

    monkeypatch.setattr(properties, "stability_candidates", broken)
    actual = f._stability_many(rows)
    assert failures
    for q, row in zip(actual, rows):
        assert_same_search(q, f._stability(*row))
    assert not hasattr(f, "_spatial_stability_predictor")


def test_predictor_slot_is_restored_on_unexpected_failure(monkeypatch):
    f = fluid()
    rows = [request(f, 1e6, 320.0, [0.2, 0.8]), request(f, 1.1e6, 330.0, [0.3, 0.7])]
    marker = object()
    f._spatial_stability_predictor = marker

    def failed(*args, **kwargs):
        raise RuntimeError("injected owner failure")

    monkeypatch.setattr(f, "_stability", failed)
    with pytest.raises(RuntimeError, match="owner failure"):
        f._stability_many(rows)
    assert f._spatial_stability_predictor is marker


def test_complete_cell_audits_and_invalid_inventory_are_retained():
    from dataclasses import replace

    from surface.liquid import ConservationError

    m, state, _ = mixture_setup()
    data = [m._state_data(i, s) for i, (_, s) in enumerate(state.nodes)]
    assert m._stable_all(data) == [None, None]
    m._batch_stability = False
    assert m._stable_all(data) == [None, None]
    m._batch_stability = True
    name, stock = state.nodes[0]
    bad = replace(
        state,
        nodes=(
            (name, replace(stock, total_energy_j=stock.total_energy_j + 10.0)),
            *state.nodes[1:],
        ),
    )
    with pytest.raises(ConservationError):
        m.inspect(bad)
