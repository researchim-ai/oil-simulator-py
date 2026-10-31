"""Component-conserving flash, strict domains, and derivatives in both phase regimes."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.blackoil_properties import BlackOilMixture
from surface.liquid import PropertyModelError
from surface.process import BlackOilSeparator, ComponentStream, StockDensities

DENSITIES = StockDensities(850, 1000, 1.2)


def pvt_data():
    p = [0.1, 1.0, 5.0]
    rs = [0.0, 20.0, 80.0]
    data = {
        "pressure_MPa": p,
        "Bo": [1.0, 1.04, 1.16],
        "Bw": [1.0, 0.999, 0.995],
        "Bg": [1.0, 0.1, 0.02],
        "mu_o_cP": [5.0, 4.2, 1.8],
        "mu_w_cP": [1.0, 1.0, 1.0],
        "mu_g_cP": [0.01, 0.012, 0.015],
        "Rs_m3m3": rs,
        "Rv_m3m3": [0.0, 0.0, 0.0],
    }
    data["extended"] = {
        "oil": {
            "rs_values": rs,
            "tables": {
                str(r): {
                    "pressure_MPa": p,
                    "Bo": [1 + 0.002 * r - 0.001 * (pp - pb) for pp in p],
                    "mu_o_cP": [5 - 0.04 * r + 0.01 * (pp - pb) for pp in p],
                }
                for r, pb in zip(rs, p)
            },
        }
    }
    return data


def model():
    return BlackOilMixture(pvt_data(), DENSITIES)


@pytest.mark.parametrize("pressure,gas", [(0.6e6, 0.144), (3e6, 0.024), (3e6, 0.144)])
def test_flash_matches_existing_separator_and_conserves_mass(pressure, gas):
    m = model()
    rates = [0.85, 2.0, gas]
    s = m.state(pressure, rates)
    feed = ComponentStream.from_mass_rates(*rates, DENSITIES)
    r = BlackOilSeparator("reference", pressure, pvt_data(), DENSITIES).separate(feed)
    assert s["phase_volume_rates_m3_s"] == pytest.approx(
        list(r.phase_volume_rates_m3_s.values()), rel=1e-12
    )
    assert sum(s["phase_mass_rates_kg_s"]) == pytest.approx(sum(rates), abs=1e-15)
    assert min(s["phase_mass_rates_kg_s"]) >= 0
    assert s["specific_volume"] == pytest.approx(sum(s["phase_volume_rates_m3_s"]) / sum(rates))
    assert s["viscosity_pa_s"] > 0


@pytest.mark.parametrize("gas", [0.03, 0.144])
def test_flash_derivatives_against_finite_differences(gas):
    m = model()
    x = np.array([2.4e6, 0.85, 2.0, gas])
    s = m.state(x[0], x[1:], derivatives=True)
    for j, h in enumerate([1.0, 1e-5, 1e-5, 1e-7]):
        a = x.copy()
        b = x.copy()
        a[j] += h
        b[j] -= h
        plus = m.state(a[0], a[1:])
        minus = m.state(b[0], b[1:])
        for key in ("specific_volume", "viscosity_pa_s"):
            expected = (plus[key] - minus[key]) / (2 * h)
            assert s["derivatives"][key][j] == pytest.approx(expected, rel=2e-6, abs=1e-13)


def test_pressure_breakpoint_moves_with_gas_oil_ratio():
    import torch

    m = model()
    masses = torch.tensor([0.85, 2.0, 0.048], dtype=torch.float64, requires_grad=True)
    lower = torch.tensor(0.1e6)
    upper = torch.tensor(5e6)
    points = m.split_pressures(lower, upper, masses)
    moving = [p for p in points if p.requires_grad]
    assert len(moving) == 1
    assert float(moving[0].detach()) == pytest.approx(2.333333333e6)
    grad = torch.autograd.grad(moving[0], masses)[0]
    assert grad[0] < 0 and grad[1] == 0 and grad[2] > 0


def test_strict_domain_and_snapshot():
    data = pvt_data()
    m = BlackOilMixture(data, DENSITIES)
    data["Bo"][0] = 999
    assert m.state(0.1e6, [0.85, 0, 0.144])["phase_volume_rates_m3_s"][0] == pytest.approx(0.001)
    for p, rates in [(9e6, [1, 1, 1]), (1e6, [0, 0, 0]), (1e6, [-1, 0, 0])]:
        with pytest.raises(PropertyModelError):
            m.state(p, rates)
    data = pvt_data()
    data["Rv_m3m3"][1] = 0.1
    with pytest.raises(PropertyModelError, match="Rv"):
        BlackOilMixture(data, DENSITIES)
    data = pvt_data()
    data.pop("extended")
    with pytest.raises(PropertyModelError, match="PVTO"):
        BlackOilMixture(data, DENSITIES).state(3e6, [0.85, 2, 0.024])
    data = pvt_data()
    data["extended"]["oil"]["tables"]["0.0"]["pressure_MPa"] = [0.2, 1, 5]
    with pytest.raises(PropertyModelError, match="entire"):
        BlackOilMixture(data, DENSITIES)


def test_inconsistent_saturation_join_is_rejected():
    data = pvt_data()
    data["extended"]["oil"]["tables"]["20.0"]["Bo"][1] += 0.01
    with pytest.raises(PropertyModelError, match="continuous"):
        BlackOilMixture(data, DENSITIES).state(3e6, [0.85, 2, 0.06])


def test_pure_water_one_sided_composition_derivative():
    m = model()
    r = m.state(2.4e6, [0.0, 2.0, 0.0], derivatives=True)
    h = 1e-6
    more = m.state(2.4e6, [h, 2.0, 0.0])
    for key in ("specific_volume", "viscosity_pa_s"):
        fd = (more[key] - r[key]) / h
        assert r["derivatives"][key][1] == pytest.approx(fd, rel=2e-6, abs=1e-12)


def test_unavailable_oil_perturbation_at_pure_water_is_not_invented():
    data = pvt_data()
    data.pop("extended")
    m = BlackOilMixture(data, DENSITIES)
    assert m.state(2e6, [0, 1, 0])["phase_mass_rates_kg_s"] == [0, 1, 0]
    with pytest.raises(PropertyModelError, match="zero-GOR"):
        m.state(2e6, [0, 1, 0], derivatives=True)
