"""Spatial reference must explicitly cross PVT and gas-appearance events."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from check_surface_blackoil import spatial_reference

from surface import BlackOilMixture, BlackOilPipe, StockDensities


def test_spatial_reference_resolves_gas_appearance_event():
    case = json.loads((ROOT / "tests/data/surface/blackoil_phase_transition.json").read_text())
    model = BlackOilMixture(case["pvt"], StockDensities(**case["stock_densities"]))
    pipe = BlackOilPipe(**case["pipe"])
    r = pipe.run(case["inlet_pressure_pa"], case["component_mass_rates_kg_s"], model)
    ref = spatial_reference(
        pipe, case["inlet_pressure_pa"], case["component_mass_rates_kg_s"], case
    )
    assert r["inlet_state"]["phase_mass_rates_kg_s"][2] == 0
    assert r["outlet_state"]["phase_mass_rates_kg_s"][2] > 0
    assert ref == pytest.approx(case["reference_outlet_pa"], abs=0.0001)
    assert r["outlet_pressure_pa"] == pytest.approx(ref, abs=0.0001)


def test_reference_crosses_multiple_pvt_and_phase_events():
    cfg = json.loads((ROOT / "configs/surface_gathering_facility.json").read_text())["gathering"]
    model = BlackOilMixture(cfg["pvt"], StockDensities(**cfg["stock_densities"]))
    pipe = BlackOilPipe("multiple_events", 2000, 0.1, 600, 1e-5)
    masses = [0.85, 2.0, 0.06]
    r = pipe.run(4.8e6, masses, model)
    ref = spatial_reference(pipe, 4.8e6, masses, cfg)
    assert 0.1e6 < r["outlet_pressure_pa"] < 1e6
    assert r["outlet_pressure_pa"] == pytest.approx(ref, abs=0.001)
