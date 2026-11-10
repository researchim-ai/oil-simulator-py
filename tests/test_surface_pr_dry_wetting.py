"""An explicit finite dry-wall model must move actual stock, never a floor."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from reference_pr_film_mist_exchange import audit
from reference_wetting_potential import pressure
from test_surface_pr_film_mist_exchange import case

from surface.film_wetting import LongWaveWettingPotential
from surface.pr_film_mist_exchange import PRFilmMistExchange


def material():
    return LongWaveWettingPotential(0.02, 1e-4, 5.0, "c2_half_adsorption")


def test_explicit_finite_dry_extension_is_c2_and_retains_the_existing_wet_law():
    law = material()
    plain = replace(law, dry_extension=None)
    length = law.adsorption_thickness_m / 2
    for order in (0, 1, 2):
        function = law.potential
        for _ in range(order):
            function = torch.func.jacrev(function)
        a = function(torch.tensor(length * (1 - 1e-9), dtype=torch.float64))
        b = function(torch.tensor(length * (1 + 1e-9), dtype=torch.float64))
        assert float(a) == pytest.approx(float(b), rel=1e-7)
    heights = torch.linspace(0, length, 101, dtype=torch.float64)
    assert (law.potential(heights) >= 0).all()
    assert float(law.potential(heights)[0]) == 0.0
    wet = torch.tensor([length, 1e-4, 1e-3], dtype=torch.float64)
    torch.testing.assert_close(law.potential(wet), plain.potential(wet), rtol=0, atol=0)


def test_dry_crossing_discrete_pressure_matches_energy_and_independent_quadrature():
    law = material()
    a = torch.tensor([0.0, 0.0, 0.25e-4, 0.8e-4, 0.5e-4], dtype=torch.float64)
    b = torch.tensor([0.0, 0.3e-4, 0.9e-4, 0.1e-4, 0.5e-4], dtype=torch.float64)
    mu = law.divided_difference(a, b)
    torch.testing.assert_close(
        mu * (b - a), law.potential(b) - law.potential(a), rtol=2e-12, atol=1e-17
    )
    np.testing.assert_allclose(
        mu.numpy(), pressure(law.to_dict(), 0.02, a.numpy(), b.numpy()), rtol=1e-11, atol=1e-10
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_film_disappears_and_returns_without_ghost_stock_and_with_surface_energy(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pipe, old, base = case(gas_u=100.0, efficiency=1.0, device=device)
    law = replace(
        base,
        film_surface_model="planar_long_wave_wetting",
        wetting={
            "film_width_m": 0.1,
            **material().to_dict(),
        },
    )
    law = PRFilmMistExchange.from_dict(law.to_dict())
    amount = dict(old.cells[0])["film"].volume_m3
    zero = [0.0, 0.0]
    entrainment = [amount, 0.0]
    dry, report = law.apply(pipe, old, 1.0, entrainment_m3_s=entrainment, deposition_m3_s=zero)
    assert "film" not in dict(dry.cells[0])
    assert audit(pipe, old.to_dict(), dry.to_dict(), report, law.to_dict(), entrainment, zero, 1.0)[
        "accepted"
    ]
    wet, report = law.apply(pipe, dry, 1.0, entrainment_m3_s=zero, deposition_m3_s=entrainment)
    assert "mist" not in dict(wet.cells[0])
    assert dict(wet.cells[0])["film"].volume_m3 == amount
    assert audit(pipe, dry.to_dict(), wet.to_dict(), report, law.to_dict(), zero, entrainment, 1.0)[
        "accepted"
    ]
    assert pipe._totals(wet)[:2] == pytest.approx(pipe._totals(old)[:2], rel=1e-12)
