"""Reuse only independently accepted full local pairs, with exact input keys."""

import copy
from dataclasses import replace

import pytest
import torch
from test_surface_compositional_separator import binary

from surface.equilibrium_cache import cached_flashes
from surface.pr_equilibrium_transport import (
    PREquilibriumLink,
    PREquilibriumNode,
    PREquilibriumTransportNetwork,
    PREquilibriumTransportState,
)
from surface.pr_fast_prefix import advance_fast


@pytest.fixture
def closed():
    f = binary()
    area = torch.pi * 0.05**2 / 4
    model = PREquilibriumTransportNetwork(
        f,
        [
            PREquilibriumNode(n, area * 0.5, "pipe_cell", length_m=0.5, diameter_m=0.05)
            for n in ("a", "b")
        ],
        [PREquilibriumLink("ab", "a", "b", 0.05), PREquilibriumLink("ba", "b", "a", 0.05)],
        initial_step_s=0.001,
        maximum_step_s=0.1,
    )
    state = model.initialize(
        {
            n: {"pressure_pa": 1e6, "temperature_k": 350.0, "mole_fractions": [1.0, 0.0]}
            for n in model.names
        }
    )
    return model, state


@pytest.mark.parametrize("duration", [0.004, 0.01, 0.02])
def test_prefix_reuses_work_and_preserves_original_adaptive_path(closed, monkeypatch, duration):
    model, initial = closed
    original = model._step
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_step", counted)
    reference = model.advance(initial, duration)
    reference_calls = len(calls)
    with cached_flashes(model.fluid) as cache:
        advance_fast(model, initial, 0.01, {})
        calls.clear()
        report = advance_fast(model, initial, duration, {})
        assert cache.transport_prefix_hits == 1
        assert report["reused_macrosteps"] > 0
        assert len(calls) < reference_calls
        assert report["state"]["time_s"] == initial.time_s + duration
        final = PREquilibriumTransportState.from_dict(report["state"])
        expected = PREquilibriumTransportState.from_dict(reference["state"])
        assert model._error(final, expected) < 1e-8
        assert final.next_step_s == pytest.approx(expected.next_step_s, rel=1e-14)
        assert [s["dt_s"] for s in report["steps"]] == pytest.approx(
            [s["dt_s"] for s in reference["steps"]], abs=1e-17
        )
        assert max(map(abs, report["conservation_residuals"])) < 1e-8
    assert not cache.transport_prefixes


def test_returned_report_cannot_corrupt_cached_physical_states(closed):
    model, initial = closed
    with cached_flashes(model.fluid):
        first = advance_fast(model, initial, 0.01, {})
        expected = copy.deepcopy(first["state"])
        first["state"]["nodes"]["a"]["component_moles"] = (-1.0, 0.0)
        first["steps"][0]["state"]["nodes"]["a"]["total_energy_j"] = 1e100
        second = advance_fast(model, initial, 0.01, {})
        assert second["state"] == expected
        assert second["executed_attempts"] == 0


def test_changed_commands_and_eos_version_never_hit_old_prefix(closed):
    model, initial = closed
    with cached_flashes(model.fluid) as cache:
        advance_fast(model, initial, 0.01, {})
        changed = advance_fast(model, initial, 0.004, {"openings": {"ab": False}})
        assert changed["reused_macrosteps"] == 0
        with torch.no_grad():
            model.fluid.kij[1, 1] += 1e-6
        other = advance_fast(model, initial, 0.004, {})
        assert other["reused_macrosteps"] == 0
        assert cache.transport_prefix_hits == 0


def test_different_initial_proposal_never_reuses_other_adaptive_history(closed):
    model, initial = closed
    with cached_flashes(model.fluid) as cache:
        advance_fast(model, initial, 0.01, {})
        report = advance_fast(model, replace(initial, next_step_s=0.002), 0.004, {})
        assert report["reused_macrosteps"] == 0 and cache.transport_prefix_hits == 0


def test_no_public_transaction_means_no_retained_prefix(closed):
    model, initial = closed
    report = advance_fast(model, initial, 0.004, {})
    assert "reused_macrosteps" not in report
    assert not hasattr(model.fluid, "_active_flash_cache")
