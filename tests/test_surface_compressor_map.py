"""Corrected coordinates, bounded map interpolation and caloric consistency."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    CompressorEnvelopeError,
    CompressorMap,
    GasCompressor,
    GasState,
    IdealGas,
    MappedGasCompressor,
)


def mapping():
    return CompressorMap(
        (60, 100, 140),
        (0, 0.5, 1),
        ((0.12, 0.3, 0.48), (0.25, 0.5, 0.75), (0.4, 0.8, 1.2)),
        ((1.2, 1.12, 1.02), (1.5, 1.3, 1.05), (1.9, 1.6, 1.1)),
        ((0.65, 0.78, 0.65), (0.68, 0.8, 0.67), (0.65, 0.78, 0.65)),
    )


def test_bilinear_map_and_inverse_pressure_at_exact_and_intermediate_nodes():
    p = mapping()
    q = p.operating_point(100, 1e5, 300, 1.3e5)
    assert q["beta"] == pytest.approx(0.5)
    assert q["mass_rate_kg_s"] == pytest.approx(0.5)
    assert q["isentropic_efficiency"] == pytest.approx(0.8)
    q = p.operating_point(80, 1e5, 300, 1.21e5)
    assert q["beta"] == pytest.approx(0.5)
    assert q["mass_rate_kg_s"] == pytest.approx(0.4)
    assert q["isentropic_efficiency"] == pytest.approx(0.79)


def test_corrected_coordinates_preserve_same_operating_point():
    p = mapping()
    base = p.operating_point(100, 1e5, 300, 1.3e5)
    changed = p.operating_point(120, 2e5, 432, 2.6e5)
    assert changed["corrected_speed_rad_s"] == pytest.approx(100)
    assert changed["corrected_mass_flow_kg_s"] == pytest.approx(base["corrected_mass_flow_kg_s"])
    assert changed["mass_rate_kg_s"] == pytest.approx(base["mass_rate_kg_s"] * 2 / 1.2)


def test_surge_guard_choke_boundary_and_speed_domain_do_not_extrapolate():
    p = mapping()
    with pytest.raises(CompressorEnvelopeError) as error:
        p.operating_point(100, 1e5, 300, 1.6e5)
    assert error.value.boundary == "surge"
    for speed in (0, 59, 141):
        with pytest.raises(CompressorEnvelopeError):
            p.operating_point(speed, 1e5, 300, 1.2e5)
    choke = p.operating_point(100, 1e5, 300, 1.01e5)
    assert choke["choke_limited"] and choke["beta"] == 1
    assert choke["pressure_ratio"] == 1.05
    assert choke["mass_rate_kg_s"] == 0.75


def test_calorics_match_independent_isentropic_formula_and_existing_compressor():
    gas = IdealGas(500, 1.3)
    p = MappedGasCompressor(mapping(), gas)
    r = p.run(100, 1e5, 300, 1.3e5)
    dt = 300 * (1.3 ** (3 / 13) - 1) / 0.8
    assert r["discharge_temperature_k"] == pytest.approx(300 + dt)
    assert r["gas_power_w"] == pytest.approx(0.5 * gas.cp_j_kg_k * dt)
    assert r["shaft_power_w"] == pytest.approx(r["gas_power_w"] / 0.95)
    old = GasCompressor("reference", 1.3e5, isentropic_efficiency=0.8).run(
        GasState(0.5, 1e5, 300), gas
    )
    assert old.shaft_power_w == pytest.approx(r["shaft_power_w"])
    assert r["compression_entropy_generation_w_k"] > 0
    with pytest.raises(CompressorEnvelopeError):
        replace(p, max_shaft_power_w=1).run(100, 1e5, 300, 1.3e5)
    with pytest.raises(CompressorEnvelopeError):
        replace(p, max_discharge_temperature_k=301).run(100, 1e5, 300, 1.3e5)


@pytest.mark.parametrize(
    "changes",
    [
        {"beta": (1, 0.5, 0)},
        {"corrected_speed_rad_s": (60, 60, 140)},
        {"isentropic_efficiency": ((2, 0.8, 0.7),) * 3},
        {"pressure_ratio": ((1.1, 1.2, 1.3),) * 3},
        {"corrected_mass_flow_kg_s": ((0.1, 0.1, 0.3),) * 3},
        {"reference_pressure_pa": float("nan")},
    ],
)
def test_bad_maps_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(mapping(), **changes)


def test_trial_continuation_is_labelled_and_low_backpressure_has_explicit_trim():
    p = mapping()
    trial = p.operating_point(100, 1e5, 300, 1.6e5, strict=False)
    assert trial["trial_only"] and not trial["within_map"]
    r = MappedGasCompressor(p, IdealGas(500, 1.3)).run(100, 1e5, 300, 1.01e5)
    assert r["within_map"] and not r["trial_only"]
    assert r["discharge_trim_pressure_drop_pa"] == pytest.approx(4000)
    assert r["trim_entropy_generation_w_k"] > 0


def test_roundoff_at_speed_boundary_is_reported_consistently():
    import math

    r = mapping().operating_point(math.nextafter(140, math.inf), 1e5, 300, 1.6e5)
    assert r["within_map"] and not r["trial_only"]
