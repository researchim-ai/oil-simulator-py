"""Caloric acceleration retains physical acceptance; exact cache is scratch."""

import math
import pickle

import pytest
import torch
from test_surface_equilibrium_thermodynamics import fluid

from surface.equilibrium_cache import cached_flashes
from surface.equilibrium_properties import _gibbs_logit_gradient
from surface.peng_robinson_mixture import PengRobinsonMixture

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("allocation", [False, True])
def test_closed_gibbs_gradient_equals_original_reverse_ad(device, allocation):
    gen = torch.Generator(device=device).manual_seed(83)
    for scale in (0.1, 5.0, 100.0):
        shape = (3, 5) if allocation else (5,)
        u = (
            scale * torch.randn(shape, generator=gen, device=device, dtype=torch.float64)
        ).requires_grad_()
        bound = 80 * torch.tanh(u / 80)
        potentials = torch.randn(shape, generator=gen, device=device, dtype=torch.float64)
        if allocation:
            logp = torch.log_softmax(bound, dim=0)
            probabilities = logp.exp()
            amounts = (logp + torch.linspace(-20, 2, shape[-1], device=device)).exp()
            dim = 0
        else:
            probabilities = torch.softmax(bound, dim=-1)
            amounts, dim = probabilities, -1
        exact = torch.autograd.grad((amounts * potentials).sum(), u)[0]
        closed = _gibbs_logit_gradient(
            u.detach(), (amounts * potentials).detach(), probabilities.detach(), dim
        )
        assert torch.allclose(exact, closed, atol=2e-14, rtol=2e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_single_phase_ph_ps_matches_scalar_eos_and_keeps_full_stability(device, monkeypatch):
    f = fluid(device)
    oracle = PengRobinsonMixture(f.components, f.binary_interactions)
    target = oracle.state(1.3e6, 327.4, [0.2, 0.8])
    calls = []
    original = f.flash_pt

    def checked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(f, "flash_pt", checked)
    for solve, key in ((f.flash_ph, "molar_enthalpy_j_mol"), (f.flash_ps, "molar_entropy_j_mol_k")):
        fast = solve(1.3e6, target[key], [0.2, 0.8], temperature_guess_k=320)
        reference = f._flash_property_bracketed(1.3e6, target[key], [0.2, 0.8], key, 320)
        assert fast["phase_count"] == reference["phase_count"] == 1
        assert float(fast["minimum_tpd"].cpu()) >= -1e-8
        assert float(fast["temperature_k"].cpu()) == pytest.approx(327.4, abs=2e-7)
        assert float(fast["temperature_k"].cpu()) == pytest.approx(
            float(reference["temperature_k"].cpu()), abs=2e-7
        )
        assert abs(float(fast["property_residual"].cpu())) <= 1e-7 + abs(target[key]) * 1e-10
    assert len(calls) >= 2


@pytest.mark.parametrize("device", DEVICES)
def test_two_phase_caloric_target_uses_original_multiphase_fallback(device, monkeypatch):
    f = fluid(device)
    q = f.flash_pt(1e6, 196.2044254, [0.5, 0.5])
    assert q["phase_count"] == 2
    original = f._flash_property_bracketed
    calls = []

    def checked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(f, "_flash_property_bracketed", checked)
    final = f.flash_ph(1e6, float(q["molar_enthalpy_j_mol"].cpu()), [0.5, 0.5], 200)
    assert calls and final["phase_count"] == 2
    assert float(final["temperature_k"].cpu()) == pytest.approx(196.2044254, abs=2e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_exact_cache_normalizes_tensor_inputs_and_owns_outputs(device, monkeypatch):
    f = fluid(device)
    calls = []
    original = f._stability

    def checked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(f, "_stability", checked)
    with cached_flashes(f, maximum_entries=2) as cache:
        first = f.flash_pt(1e6, 320, [0.5, 0.5])
        first["phases"][0]["mole_fractions"].zero_()
        first["molar_enthalpy_j_mol"].fill_(1e20)
        second = f.flash_pt(
            f.tensor(1e6), f.tensor(320), f.tensor([0.5, 0.5]), _allow_pure_coexistence=False
        )
        assert len(calls) == 1 and cache.hits == 1
        assert second["phases"][0]["mole_fractions"].cpu().tolist() == [0.5, 0.5]
        assert float(second["molar_enthalpy_j_mol"].cpu()) < 1e5
        f.flash_pt(math.nextafter(1e6, math.inf), 320, [0.5, 0.5])
        assert len(calls) == 2  # no rounding of nearby physical states
        f.cp.add_(0)  # even an in-place no-op invalidates the tensor version
        f.flash_pt(1e6, 320, [0.5, 0.5])
        assert len(calls) == 3 and cache.evictions == 1
        assert len(cache.entries) == 2
    assert not hasattr(f, "_active_flash_cache") and not cache.entries


def test_cache_scope_failure_nested_disable_and_restart_have_no_persistent_entries():
    f = fluid()
    with pytest.raises(RuntimeError, match="trial"), cached_flashes(f) as cache:
        f.flash_pt(1e6, 320, [0.5, 0.5])
        raise RuntimeError("rejected physical trial")
    assert not hasattr(f, "_active_flash_cache") and not cache.entries
    restored = pickle.loads(pickle.dumps(f))
    assert not hasattr(restored, "_active_flash_cache")
    with cached_flashes(f, enabled=False), cached_flashes(f) as disabled:
        assert disabled.statistics() == {"enabled": False}
        f.flash_pt(1e6, 320, [0.5, 0.5])
    assert not hasattr(f, "_active_flash_cache")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "p,t,z,kij",
    [
        (1e6, 320, [0.2, 0.8], 0.0),
        (1e6, 196.2044254, [0.5, 0.5], 0.0),
        (2e6, 290, [0.1, 0.9], 0.15),
        (1e5, 180, [0.9, 0.1], 0.05),
        (1e6, 320, [1 - 1e-12, 1e-12], 0.0),
    ],
)
def test_all_seed_batched_search_retains_reference_instability_detection(device, p, t, z, kij):
    f = fluid(device, kij)
    pp, tt, zz = map(f.tensor, (p, t, z))
    q = f.tensor_homogeneous(pp, tt, zz)
    mu = torch.log(zz) + q["log_fugacity_coefficients"]
    active = torch.nonzero(zz > 0).flatten()
    fast = f._stability(pp, tt, zz, mu, active)
    reference = f._stability(pp, tt, zz, mu, active, _accelerated=False)
    # A more negative valid minimum is allowed; a reference instability
    # must never disappear behind the batched predictor.
    assert float(fast[0].cpu()) <= float(reference[0].cpu()) + 1e-8
    assert bool(fast[2]["valid"].cpu())
    assert torch.allclose(fast[1].sum(), f.tensor(1.0), atol=1e-12, rtol=0)


def test_failed_batch_predictor_runs_original_seeds(monkeypatch):
    f = fluid()
    expected = f.flash_pt(1e6, 196.2044254, [0.5, 0.5])

    def failed(*args, **kwargs):
        raise torch.linalg.LinAlgError("injected batched factorization failure")

    monkeypatch.setattr(f, "_stability_predictors", failed)
    actual = f.flash_pt(1e6, 196.2044254, [0.5, 0.5])
    assert actual["phase_count"] == expected["phase_count"] == 2
    assert float(actual["molar_enthalpy_j_mol"].cpu()) == pytest.approx(
        float(expected["molar_enthalpy_j_mol"].cpu()), abs=1e-6
    )


def test_differentiable_inputs_bypass_transaction_cache():
    f = fluid()
    z = f.tensor([0.5, 0.5]).requires_grad_()
    with cached_flashes(f) as cache:
        first = f.flash_pt(1e6, 320, z)
        second = f.flash_pt(1e6, 320, z)
        assert cache.hits == cache.misses == 0
    assert torch.equal(first["molar_enthalpy_j_mol"], second["molar_enthalpy_j_mol"])


@pytest.mark.parametrize("device", DEVICES)
def test_model_parameter_sensitivities_bypass_a_previous_constant_cache(device):
    f = fluid(device)
    with cached_flashes(f) as cache:
        f.flash_pt(1e6, 320, [0.5, 0.5])
        f.cp.requires_grad_(True)  # requires_grad_ does not bump _version
        q = f.flash_pt(1e6, 320, [0.5, 0.5])
        assert cache.hits == 0
        derivative = torch.autograd.grad(q["molar_enthalpy_j_mol"], f.cp)[0]
        assert torch.allclose(
            derivative, f.tensor([0.5, 0.5]) * (320 - f.tref), atol=1e-12, rtol=1e-12
        )
