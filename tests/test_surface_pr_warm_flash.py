"""Near-state hints are guesses; material, Gibbs and all TPD starts still audit."""

import pytest
import torch
from test_surface_equilibrium_thermodynamics import fluid, symmetric

from surface.equilibrium_cache import cached_flashes
from surface.liquid import HydraulicConvergenceError


@pytest.mark.parametrize(
    "factory,p,t,z", [(fluid, 1e6, 196.2, [0.5, 0.5]), (symmetric, 1e6, 300.0, [1 / 3] * 3)]
)
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_warm_multiphase_matches_cold_and_runs_complete_stability(
    factory, p, t, z, monkeypatch, device
):
    f = factory(device)
    with cached_flashes(f) as cache:
        first = f.flash_pt(p, t, z)
        assert first["phase_count"] >= 2
        calls = []
        old = f._stability

        def checked(*args, **kwargs):
            calls.append(True)
            return old(*args, **kwargs)

        monkeypatch.setattr(f, "_stability", checked)
        actual = f.flash_pt(p * 1.0001, t + 0.001, z)
        assert calls and cache.warm_flash_accepts == 1
        assert cache.warm_flash_attempts == 1
    assert not cache.warm_equilibria
    with cached_flashes(f, enabled=False):
        expected = f.flash_pt(p * 1.0001, t + 0.001, z)
    assert actual["phase_count"] == expected["phase_count"]
    for key in ("molar_enthalpy_j_mol", "molar_volume_m3_mol", "density_kg_m3"):
        torch.testing.assert_close(actual[key], expected[key], rtol=2e-8, atol=2e-8)
    assert float(actual["component_balance_error"]) < 1e-10
    assert float(actual["minimum_tpd"]) >= -1e-8
    assert float(actual["log_fugacity_error"]) < 1e-8


def test_failed_warm_allocation_falls_back_and_parameter_change_invalidates(monkeypatch):
    f = fluid()
    old = f._allocation
    cold = []

    def checked(*args, **kwargs):
        if kwargs.get("_polish_only"):
            raise HydraulicConvergenceError("injected failed warm tangent")
        cold.append(True)
        return old(*args, **kwargs)

    with cached_flashes(f) as cache:
        f.flash_pt(1e6, 196.2, [0.5, 0.5])
        monkeypatch.setattr(f, "_allocation", checked)
        q = f.flash_pt(1e6, 196.201, [0.5, 0.5])
        assert cold and q["phase_count"] == 2
        assert cache.warm_flash_fallbacks == 1
        attempts = cache.warm_flash_attempts
        f.kij.add_(0.0)
        f.flash_pt(1e6, 196.202, [0.5, 0.5])
        assert cache.warm_flash_attempts == attempts


def test_support_change_and_single_phase_transition_keep_cold_classification():
    f = fluid()
    with cached_flashes(f) as cache:
        f.flash_pt(1e6, 196.2, [0.5, 0.5])
        q = f.flash_pt(1e6, 196.201, [1.0, 0.0])
        assert q["phase_count"] == 1 and cache.warm_flash_attempts == 0
        q = f.flash_pt(1e6, 320.0, [0.5, 0.5])
        assert q["phase_count"] == 1 and cache.warm_flash_attempts == 0
