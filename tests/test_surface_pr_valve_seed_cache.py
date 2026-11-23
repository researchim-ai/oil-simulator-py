"""Accepted active sets are numerical guesses, never cached physical decisions."""

from types import SimpleNamespace

import pytest
import torch
from test_surface_pr_shutoff_valve_predictor import fixture, step

from surface.equilibrium_cache import cached_flashes
from surface.pr_pump_connection import PumpConnectionBundle


def test_transaction_seed_copy_version_and_lifetime():
    model, _ = fixture()
    assert PumpConnectionBundle.previous_valves(model) is None
    with cached_flashes(model.fluid) as cache:
        PumpConnectionBundle.remember_valves(model, {"booster": False})
        guessed = PumpConnectionBundle.previous_valves(model)
        assert guessed == {"booster": False}
        guessed["booster"] = True
        assert PumpConnectionBundle.previous_valves(model) == {"booster": False}
        model.fluid.kij.add_(0.0)
        assert PumpConnectionBundle.previous_valves(model) is None
        assert not cache.pump_valve_seeds
        PumpConnectionBundle.remember_valves(model, {"booster": True})
    assert not cache.pump_valve_seeds
    assert PumpConnectionBundle.previous_valves(model) is None


def test_cache_is_bounded_and_parameter_sensitivity_bypasses_it():
    model, _ = fixture()
    with cached_flashes(model.fluid) as cache:
        for i in range(25):
            other = SimpleNamespace(fluid=model.fluid, model_id=str(i))
            PumpConnectionBundle.remember_valves(other, {"booster": False})
        assert len(cache.pump_valve_seeds) == 16
        assert "0" not in cache.pump_valve_seeds
        PumpConnectionBundle.remember_valves(model, {"booster": False})
        model.fluid.kij.requires_grad_(True)
        assert PumpConnectionBundle.previous_valves(model) is None
        PumpConnectionBundle.remember_valves(model, {"booster": False})
        assert model.model_id not in cache.pump_valve_seeds


def test_wrong_cached_branch_still_switches_and_certifies_physical_balances():
    model, state = fixture()
    before = state.to_dict()
    with cached_flashes(model.fluid):
        PumpConnectionBundle.remember_valves(model, {"booster": True})
        _, report = step(model, state)
        assert report["pump_check_valve_retries"] == 1
        assert not report["pump_connections"]["booster"]["valve_closed"]
        assert PumpConnectionBundle.previous_valves(model) == {"booster": False}
    assert state.to_dict() == before
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


def test_stale_open_seed_closes_under_new_backpressure():
    model, state = fixture(backpressure=1.1e6)
    with cached_flashes(model.fluid):
        PumpConnectionBundle.remember_valves(model, {"booster": False})
        _, report = step(model, state)
        assert report["pump_check_valve_warm_start"]
        assert report["pump_check_valve_retries"] >= 1 or report["pump_check_valve_warm_fallback"]
        row = report["pump_connections"]["booster"]
        assert row["valve_closed"]
        assert row["flow_m3_s"] == row["shaft_power_w"] == 0


def test_failed_warm_attempt_retries_original_predictor(monkeypatch):
    from surface.liquid import HydraulicConvergenceError

    model, state = fixture()
    original = model._problem
    branches = []

    def problem(*args, **kwargs):
        branch = kwargs.get("pump_valve_states")
        branches.append(branch)
        if branch == {"booster": False}:
            raise HydraulicConvergenceError("Injected warm branch failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_problem", problem)
    with cached_flashes(model.fluid):
        PumpConnectionBundle.remember_valves(model, {"booster": False})
        _, report = step(model, state)
        assert report["pump_check_valve_warm_start"]
        assert report["pump_check_valve_warm_fallback"]
        assert branches[:2] == [{"booster": False}, {}]
        assert not report["pump_connections"]["booster"]["valve_closed"]


def test_two_failed_attempts_propagate_and_evict_hint(monkeypatch):
    from surface.liquid import HydraulicConvergenceError

    model, state = fixture()
    before = state.to_dict()
    branches = []

    def failed(*args, **kwargs):
        branches.append(kwargs.get("pump_valve_states"))
        raise HydraulicConvergenceError("Injected real failure")

    monkeypatch.setattr(model, "_problem", failed)
    with cached_flashes(model.fluid):
        PumpConnectionBundle.remember_valves(model, {"booster": False})
        with pytest.raises(HydraulicConvergenceError, match="real failure"):
            step(model, state)
        assert branches == [{"booster": False}, {}]
        assert PumpConnectionBundle.previous_valves(model) is None
    assert state.to_dict() == before


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_device_warm_branch_retains_original_stocks_and_ledgers(device):
    model, state = fixture(device=device)
    with cached_flashes(model.fluid):
        cold, _ = step(model, state)
        warm, report = step(model, state)
        assert report["pump_check_valve_warm_start"]
        assert not report["pump_check_valve_warm_fallback"]
        assert model._error(cold, warm) < 1e-6
        assert max(map(abs, report["conservation_residuals"])) < 1e-7
