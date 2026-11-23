"""Exact grouped wave tangents with full physical audits and scalar fallback."""

from dataclasses import replace

import pytest
import torch
from test_surface_pr_equilibrium_transport import pipe

from surface import PREquilibriumTransportNetwork
from surface.liquid import PropertyModelError


def entries(model, state):
    result = []
    for i, (_, s) in enumerate(state.nodes):
        d = model._state_data(i, s)
        active = torch.nonzero(d["n"] / d["n"].sum() > 0).flatten()
        result.append((i, d, active))
    return result


@pytest.mark.parametrize(
    "temperature,velocity", [(320.0, 0.0), (320.0, -2.0), (196.2, 0.0), (196.2, 2.0)]
)
def test_vmap_tangent_matches_scalar_without_scalar_fallback(monkeypatch, temperature, velocity):
    model, state = pipe(temperature, velocity, True)
    model._analytic_single_phase_characteristics = False
    batch = entries(model, state)
    expected = [model._characteristics(*entry) for entry in batch]

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected scalar fallback")

    monkeypatch.setattr(model, "_characteristics", forbidden)
    model._characteristics_many(batch)
    for (_, d, _), ref in zip(batch, expected):
        torch.testing.assert_close(
            d["characteristic_speeds"].sort().values, ref.sort().values, atol=1e-7, rtol=1e-10
        )


def test_different_geometry_keeps_separate_scalar_kernels(monkeypatch):
    model, _ = pipe(320.0)
    nodes = [
        model.nodes[0],
        replace(model.nodes[1], length_m=0.75, volume_m3=model.nodes[1].volume_m3 * 1.5),
    ]
    model = PREquilibriumTransportNetwork(model.fluid, nodes)
    state = model.initialize(
        {
            n.name: {"pressure_pa": 1e6, "temperature_k": 320.0, "mole_fractions": [0.5, 0.5]}
            for n in nodes
        }
    )
    batch = entries(model, state)
    original = model._characteristics
    calls = []

    def observed(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_characteristics", observed)
    model._characteristics_many(batch)
    assert calls == [0, 1]


def test_stage_groups_separate_support_roots_and_chunk_budget(monkeypatch):
    model, _ = pipe(196.2)
    model._analytic_single_phase_characteristics = False
    state = model.initialize(
        {
            "0": {"pressure_pa": 1e6, "temperature_k": 320.0, "mole_fractions": [1.0, 0.0]},
            "1": {"pressure_pa": 1e6, "temperature_k": 196.2, "mole_fractions": [0.5, 0.5]},
        }
    )
    first = entries(model, state)
    second = entries(model, state)
    expected = [model._characteristics(*row) for row in first + second]
    model._characteristic_batch_size = 2

    def forbidden(*args, **kwargs):
        raise AssertionError("Compatible cross-stage group was not batched")

    monkeypatch.setattr(model, "_characteristics", forbidden)
    model._characteristics_many(first + second)
    for (_, data, _), ref in zip(first + second, expected):
        torch.testing.assert_close(
            data["characteristic_speeds"].sort().values, ref.sort().values, atol=1e-7, rtol=1e-10
        )


def test_batch_backend_failure_uses_original_scalar_diagnosis(monkeypatch):
    model, state = pipe(196.2)
    batch = entries(model, state)
    original = torch.linalg.solve
    calls = []

    def failing(a, b, *args, **kwargs):
        calls.append(a.ndim)
        if a.ndim == 3:
            raise RuntimeError("Synthetic missing batch backend")
        return original(a, b, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve", failing)
    model._characteristics_many(batch)
    assert calls == [3, 2, 2]
    assert all(torch.isfinite(d["characteristic_speeds"]).all() for _, d, _ in batch)


def test_no_cache_hides_new_state_or_changed_spectral_bound():
    model, state = pipe(320.0)
    batch = entries(model, state)
    model._characteristics_many(batch)
    model.wave_speed_bound_m_s = 1.0
    with pytest.raises(PropertyModelError, match="wave bound"):
        model._characteristics_many(batch)


def test_all_original_stability_audits_are_called_for_every_stage(monkeypatch):
    model, state = pipe(196.2)
    first = [d for _, d, _ in entries(model, state)]
    second = [d for _, d, _ in entries(model, state)]
    original = model._stable
    seen = []

    def observed(index, data, *args, **kwargs):
        seen.append(index)
        return original(index, data, *args, **kwargs)

    monkeypatch.setattr(model, "_stable", observed)
    assert model._stable_all_many([first, second]) == [[None, None], [None, None]]
    assert seen == [0, 1, 0, 1]
    assert all("characteristic_speeds" in d for d in first + second)


def test_distinct_states_and_wall_capacities_use_their_own_tangents(monkeypatch):
    from check_pr_collocation_walls import wall_fixture

    model, state = wall_fixture()
    model._analytic_single_phase_characteristics = False
    batch = entries(model, state)
    expected = [model._characteristics(*row) for row in batch]
    assert not torch.equal(expected[0], expected[1])

    def forbidden(*args, **kwargs):
        raise AssertionError("Different numerical states must still batch")

    monkeypatch.setattr(model, "_characteristics", forbidden)
    model._characteristics_many(batch)
    for (_, data, _), ref in zip(batch, expected):
        torch.testing.assert_close(
            data["characteristic_speeds"].sort().values, ref.sort().values, atol=1e-7, rtol=1e-10
        )


@pytest.mark.parametrize("invalid", ["nan", "complex", "bound"])
def test_bad_member_of_batch_is_rejected_with_original_spectral_check(monkeypatch, invalid):
    model, state = pipe(320.0)
    model._analytic_single_phase_characteristics = False
    batch = entries(model, state)
    original = torch.linalg.eigvals

    def damaged(matrix):
        values = original(matrix)
        if matrix.ndim == 3:
            values = values.clone()
            values[1, 0] = (
                complex(float("nan"), 0.0)
                if invalid == "nan"
                else complex(0.0, 1000.0)
                if invalid == "complex"
                else complex(3000.0, 0.0)
            )
        return values

    monkeypatch.setattr(torch.linalg, "eigvals", damaged)
    with pytest.raises(PropertyModelError):
        model._characteristics_many(batch)


def test_nonuniform_two_phase_states_are_not_replaced_by_one_representative(monkeypatch):
    model, _ = pipe(196.2, 0.0, True)
    state = model.initialize(
        {
            str(i): {
                "pressure_pa": 1e6 + (-1) ** i * 500.0,
                "temperature_k": 196.0 + 0.4 * i,
                "mole_fractions": [0.5, 0.5],
                "velocity_m_s": -0.5 + 1.5 * i,
            }
            for i in range(2)
        }
    )
    batch = entries(model, state)
    expected = [model._characteristics(*row) for row in batch]
    assert not torch.equal(expected[0], expected[1])

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected scalar fallback")

    monkeypatch.setattr(model, "_characteristics", forbidden)
    model._characteristics_many(batch)
    for (_, data, _), ref in zip(batch, expected):
        torch.testing.assert_close(
            data["characteristic_speeds"].sort().values, ref.sort().values, atol=1e-7, rtol=1e-10
        )


def test_non_hem_closure_keeps_scalar_characteristic_kernel(monkeypatch):
    model, _ = pipe(320.0, 0.1, True)
    model = PREquilibriumTransportNetwork(
        model.fluid, [replace(n, distribution_strength=0.001) for n in model.nodes], model.links
    )
    state = model.initialize(
        {
            n.name: {
                "pressure_pa": 1e6,
                "temperature_k": 320.0,
                "mole_fractions": [0.5, 0.5],
                "velocity_m_s": 0.1,
            }
            for n in model.nodes
        }
    )
    original = model._characteristics
    seen = []

    def observed(*args, **kwargs):
        seen.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_characteristics", observed)
    model._characteristics_many(entries(model, state))
    assert seen == [0, 1]
