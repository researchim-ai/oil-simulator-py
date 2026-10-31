"""Gas source pressure routing, PVT compatibility and facility utility budgets."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_surface_facility import load_facility

from surface import EquipmentOperatingError, GasExportSystem, StockDensities

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/surface_gas_facility.json").read_text())


def test_gas_facility_routes_all_sources_and_closes_mass_and_energy():
    facility = load_facility(config())
    r = facility.run()
    g = r["gas_export"]
    assert r["feasible"]
    assert g["mass_rate_kg_s"] == pytest.approx(0.144)
    assert abs(g["mass_residual_kg_s"]) < 1e-15
    assert abs(g["energy_residual_w"]) < 1e-7
    assert abs(g["drive_energy_residual_w"]) < 1e-7
    counts = []
    for route in g["routes"]:
        counts.append(
            sum(e["kind"] == "adiabatic_ideal_gas_compressor" for e in route["equipment"])
        )
        assert route["outlet"]["pressure_pa"] == pytest.approx(6e6)
        assert abs(route["mass_residual_kg_s"]) < 1e-15
        assert abs(route["energy_residual_w"]) < 1e-7
    assert counts == [1, 2, 3]
    assert g["outlet"]["pressure_pa"] < g["header"]["pressure_pa"]
    assert g["pipe"]["outlet_mach"] < 0.01
    assert g["shaft_power_w"] == pytest.approx(g["gas_power_w"] + g["drive_loss_w"])
    assert r["measurements"]["facility_drive_power_total_w"]["value"] == pytest.approx(
        g["drive_power_w"] + r["measurements"]["shaft_power_total_w"]["value"]
    )
    resumed = facility.run(initial=facility.network_initial(json.loads(json.dumps(r))))
    assert resumed["network"]["iterations"] == 0
    assert resumed["gas_export"] == g


def test_different_pressure_sources_are_conditioned_before_mixing():
    c = config()
    c["gas_export"]["header_pressure_pa"] = 2e6
    c["limits"] = [limit for limit in c["limits"] if limit["name"] != "gas_delivery"]
    r = load_facility(c).run()["gas_export"]
    assert r["routes"][0]["equipment"][0]["kind"] == "ideal_gas_isenthalpic_letdown"
    assert all(route["outlet"]["pressure_pa"] == pytest.approx(2e6) for route in r["routes"])
    assert abs(r["energy_residual_w"]) < 1e-7


def test_gas_capacity_violations_are_reported_not_relaxed():
    c = config()
    c["limits"].append(
        {"name": "tiny_gas_power", "metric": "gas_drive_power_w", "unit": "W", "maximum": 1}
    )
    r = load_facility(c).run()
    assert not r["feasible"]
    assert r["violations"][0]["name"] == "tiny_gas_power"
    assert r["violations"][0]["violation"] > 1000


def test_mismatched_pvt_and_ideal_gas_density_requires_real_gas_model():
    c = config()
    c["pvt"]["Bg"] = [1, 0.11, 0.025]
    with pytest.raises(EquipmentOperatingError, match="density mismatch"):
        load_facility(c).run()


def test_zero_free_gas_has_no_arbitrary_header_temperature():
    c = config()
    c["feeds"] = [{"water_m3_s": 0.002}]
    c["limits"] = [limit for limit in c["limits"] if limit["name"] != "gas_delivery"]
    r = load_facility(c).run()["gas_export"]
    assert r["header"] is None and r["outlet"] is None
    assert r["drive_power_w"] == 0
    assert all(route["status"] == "no_flow" for route in r["routes"])
    assert GasExportSystem(1e6).run([], StockDensities(850, 1000, 1.2))["mass_rate_kg_s"] == 0
