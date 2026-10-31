"""Pure PR references, thermodynamic identities and phase-aware PH/PS flash."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import PengRobinsonFluid, PropertyModelError


def fluid():
    return PengRobinsonFluid("propane", 369.89, 4251200.0, 0.1521, 0.04409562, 73.6, 200.0, 550.0)


@pytest.mark.parametrize(
    "pressure,temperature", [(1e5, 450.0), (5e6, 300.0), (1.2e6, 300.0), (5e6, 450.0)]
)
def test_derivatives_and_thermodynamic_identities(pressure, temperature):
    f = fluid()
    r = f.state(pressure, temperature)
    d = r["derivatives"]
    for key in ("enthalpy_j_kg", "entropy_j_kg_k", "specific_volume_m3_kg"):
        dp = (
            f.state(pressure + 10, temperature)[key] - f.state(pressure - 10, temperature)[key]
        ) / 20
        dt = (
            f.state(pressure, temperature + 0.001)[key]
            - f.state(pressure, temperature - 0.001)[key]
        ) / 0.002
        assert d[key]["pressure_pa"] == pytest.approx(dp, rel=2e-6, abs=1e-10)
        assert d[key]["temperature_k"] == pytest.approx(dt, rel=2e-6, abs=1e-10)
    assert d["enthalpy_j_kg"]["temperature_k"] == pytest.approx(
        temperature * d["entropy_j_kg_k"]["temperature_k"], rel=1e-10
    )
    assert d["entropy_j_kg_k"]["pressure_pa"] == pytest.approx(
        -d["specific_volume_m3_kg"]["temperature_k"], rel=1e-10
    )
    assert d["enthalpy_j_kg"]["pressure_pa"] == pytest.approx(
        r["specific_volume_m3_kg"] - temperature * d["specific_volume_m3_kg"]["temperature_k"],
        rel=1e-10,
        abs=1e-12,
    )
    assert r["cp_j_kg_k"] > r["cv_j_kg_k"] > 0 and r["sound_speed_m_s"] > 0


def test_published_coolprop_pr_saturation_references():
    f = fluid()
    sat = f.saturation_at_temperature(300.0)
    # CoolProp cubic backend example, not HEOS/experimental values.
    assert sat["pressure_pa"] == pytest.approx(997429.8169997118, rel=5e-7)
    assert f.saturation_at_pressure(101325.0)["temperature_k"] == pytest.approx(
        230.96204040305227, abs=1e-5
    )
    assert abs(sat["fugacity_residual"]) < 1e-9
    with pytest.raises(PropertyModelError, match="quality"):
        f.state(sat["pressure_pa"], 300.0)
    assert sat["liquid"]["density_kg_m3"] > sat["vapor"]["density_kg_m3"]


def test_clapeyron_latent_heat_and_phase_stability():
    f = fluid()
    sat = f.saturation_at_temperature(300.0)
    l, v = sat["liquid"], sat["vapor"]
    assert v["enthalpy_j_kg"] - l["enthalpy_j_kg"] == pytest.approx(
        300 * (v["entropy_j_kg_k"] - l["entropy_j_kg_k"]), rel=1e-10
    )
    clap = (v["entropy_j_kg_k"] - l["entropy_j_kg_k"]) / (
        v["specific_volume_m3_kg"] - l["specific_volume_m3_kg"]
    )
    fd = (
        f.saturation_at_temperature(300.001)["pressure_pa"]
        - f.saturation_at_temperature(299.999)["pressure_pa"]
    ) / 0.002
    assert clap == pytest.approx(fd, rel=1e-7)
    assert f.state(1.2e6, 300.0)["phase"] == "liquid"
    assert not f.state(1.2e6, 300.0, "vapor")["stable"]


@pytest.mark.parametrize(
    "pressure,temperature", [(1e5, 450.0), (5e6, 300.0), (1.2e6, 300.0), (5e6, 450.0)]
)
def test_ph_ps_roundtrip(pressure, temperature):
    f = fluid()
    r = f.state(pressure, temperature)
    a = f.flash_ph(pressure, r["enthalpy_j_kg"])
    b = f.flash_ps(pressure, r["entropy_j_kg_k"])
    assert a["temperature_k"] == pytest.approx(temperature, abs=1e-6)
    assert b["temperature_k"] == pytest.approx(temperature, abs=1e-6)
    assert abs(a["flash_residual"]) < 1e-4 and abs(b["flash_residual"]) < 1e-6


@pytest.mark.parametrize("quality", [0.0, 0.35, 1.0])
def test_two_phase_flash_conserves_h_s_and_specific_volume(quality):
    f = fluid()
    sat = f.saturation_at_temperature(300.0)
    l, v = sat["liquid"], sat["vapor"]
    h = (1 - quality) * l["enthalpy_j_kg"] + quality * v["enthalpy_j_kg"]
    s = (1 - quality) * l["entropy_j_kg_k"] + quality * v["entropy_j_kg_k"]
    a, b = f.flash_ph(sat["pressure_pa"], h), f.flash_ps(sat["pressure_pa"], s)
    assert a["vapor_quality"] == pytest.approx(quality, abs=1e-9)
    assert b["vapor_quality"] == pytest.approx(quality, abs=1e-9)
    assert a["density_kg_m3"] == pytest.approx(
        1 / ((1 - quality) * l["specific_volume_m3_kg"] + quality * v["specific_volume_m3_kg"]),
        rel=1e-9,
    )


def test_explicit_thermal_domain_is_not_extrapolated():
    f = fluid()
    with pytest.raises(PropertyModelError, match="range"):
        f.state(1e5, 100.0)
    with pytest.raises(PropertyModelError, match="range"):
        f.flash_ph(1e5, 1e9)
    with pytest.raises(PropertyModelError, match="0.999"):
        f.saturation_at_temperature(f.critical_temperature_k)
