"""Stock-component conservation, black-oil phase split and facility handoff."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_surface_facility import load_facility

from surface import (
    BlackOilSeparator,
    ComponentStream,
    EquipmentOperatingError,
    FacilityLimit,
    StockDensities,
    component_balance,
    evaluate_limits,
    run_separator_train,
)

ROOT = Path(__file__).resolve().parents[1]
DENSITIES = StockDensities(850, 1000, 1.2)


def config():
    return json.loads((ROOT / "configs/surface_facility.json").read_text())


def stage(pressure=5e6, pvt=None):
    return BlackOilSeparator("test", pressure, pvt or config()["pvt"], DENSITIES)


def test_component_mixing_splitting_and_mass_conversions():
    a = ComponentStream(0.001, 0.002, 0.12)
    left, right = a.split(0.35)
    mixed = ComponentStream.mix([left, right])
    assert mixed.total_mass_rate(DENSITIES) == pytest.approx(2.994)
    assert component_balance([a], [left, right], DENSITIES) == pytest.approx(
        {"oil": 0, "water": 0, "gas": 0}, abs=1e-15
    )
    mass = a.mass_rates(DENSITIES)
    assert ComponentStream.from_mass_rates(mass["oil"], mass["water"], mass["gas"], DENSITIES) == a
    assert ComponentStream.mix([]) == ComponentStream()
    with pytest.raises(ValueError):
        a.split(1.1)
    with pytest.raises(ValueError):
        ComponentStream(gas_m3_s=-1)


def test_saturated_phase_split_against_component_equations():
    r = stage().separate(ComponentStream(0.001, 0.002, 0.12))
    assert r.dissolved_gas_ratio_m3_m3 == 80
    assert r.oil == ComponentStream(0.001, 0, 0.08)
    assert r.water == ComponentStream(0, 0.002, 0)
    assert r.gas.gas_m3_s == pytest.approx(0.04)
    assert r.phase_volume_rates_m3_s == pytest.approx(
        {"oil": 0.00115, "water": 0.002, "gas": 0.0008}
    )
    assert r.phase_density_kg_m3["oil"] == pytest.approx((0.85 + 0.08 * 1.2) / 0.00115)
    assert max(map(abs, r.component_mass_residuals_kg_s.values())) < 1e-15


def test_three_stages_release_gas_without_losing_dissolved_component():
    cfg = config()
    facility = load_facility(cfg)
    feed = ComponentStream.mix(facility.feeds)
    results, products, residual = run_separator_train(feed, facility.stages)
    assert [r.gas.gas_m3_s for r in results] == pytest.approx([0.04, 0.06, 0.02])
    assert products["oil"] == ComponentStream(0.001, 0, 0)
    assert products["gas"].gas_m3_s == pytest.approx(0.12)
    assert max(map(abs, residual.values())) < 1e-15
    with pytest.raises(ValueError, match="strictly decrease"):
        run_separator_train(feed, facility.stages[::-1])


def test_undersaturated_volume_requires_real_covering_pvto():
    feed = ComponentStream(0.001, 0, 0.04)
    with pytest.raises(EquipmentOperatingError, match="extended PVTO"):
        stage().separate(feed)
    pvt = config()["pvt"]
    pvt["extended"] = {
        "oil": {
            "rs_values": [0, 100],
            "tables": {
                str(rs): {"pressure_MPa": [0.1, 1, 5], "Bo": [bo] * 3, "mu_o_cP": [5] * 3}
                for rs, bo in [(0, 1.0), (100, 1.2)]
            },
        }
    }
    r = stage(pvt=pvt).separate(feed)
    assert r.gas.gas_m3_s == 0
    assert r.oil.gas_m3_s == 0.04
    assert r.phase_volume_rates_m3_s["oil"] == pytest.approx(0.00108)
    short = copy.deepcopy(pvt)
    for table in short["extended"]["oil"]["tables"].values():
        table["pressure_MPa"] = [0.1, 1, 4]
    with pytest.raises(EquipmentOperatingError, match="PVTO coverage"):
        stage(pvt=short).separate(feed)


@pytest.mark.parametrize(
    "feed", [ComponentStream(), ComponentStream(water_m3_s=0.01), ComponentStream(gas_m3_s=0.2)]
)
def test_single_phase_and_empty_feed(feed):
    r = stage().separate(feed)
    assert r.oil.oil_m3_s == 0
    assert max(map(abs, r.component_mass_residuals_kg_s.values())) == 0


def test_pressure_and_vaporized_oil_domain_are_explicit():
    with pytest.raises(ValueError, match="outside PVT"):
        stage(pressure=6e6)
    pvt = config()["pvt"]
    pvt["Rv_m3m3"] = [0, 0.001, 0.002]
    with pytest.raises(ValueError, match="Rv"):
        stage(pvt=pvt)


def test_limits_keep_values_units_and_all_violations():
    limits = [
        FacilityLimit("power", "power", "W", maximum=100),
        FacilityLimit("pressure", "pressure", "Pa", minimum=200),
    ]
    report = evaluate_limits(
        limits, {"power": {"value": 120, "unit": "W"}, "pressure": {"value": 180, "unit": "Pa"}}
    )
    assert not report["feasible"]
    assert [r["violation"] for r in report["violations"]] == [20, 20]
    with pytest.raises(ValueError, match="Unknown facility metric"):
        evaluate_limits(limits, {})
    with pytest.raises(ValueError, match="Unit mismatch"):
        evaluate_limits(limits, {"power": {"value": 120, "unit": "kW"}})


def test_full_facility_mass_handoff_limits_and_restart():
    facility = load_facility(config())
    before = facility.water_network.to_dict()
    result = facility.run()
    assert result["feasible"]
    assert result["water_handoff"]["mass_rate_kg_s"] == 2
    assert result["network"]["boundary_mass_rates_kg_s"]["disposal"] == pytest.approx(-2)
    assert result["water_handoff"]["dissipation_w"] > 0
    assert result["measurements"]["shaft_power_total_w"]["value"] == pytest.approx(
        1000 * 9.80665 * 0.002 * 16 / 0.8
    )
    assert max(map(abs, result["component_mass_residuals_kg_s"].values())) < 1e-15
    again = facility.run(initial=facility.network_initial(json.loads(json.dumps(result))))
    assert again["network"]["iterations"] == 0
    assert facility.water_network.to_dict() == before
    cfg = config()
    cfg["limits"][0]["maximum"] = 1
    violated = load_facility(cfg).run()
    assert not violated["feasible"]
    assert violated["violations"][0]["name"] == "platform_power"


def test_facility_cannot_silently_change_mass_or_density():
    cfg = config()
    cfg["water_network"]["fluid"]["density_kg_m3"] = 999
    with pytest.raises(ValueError, match="density"):
        load_facility(cfg).run()
    cfg = config()
    cfg["water_network"]["nodes"][0]["source_mass_rate_kg_s"] = 1
    with pytest.raises(ValueError, match="additional sources"):
        load_facility(cfg)


def test_extended_vaporized_oil_and_nonmonotone_rs_rejected():
    pvt = config()["pvt"]
    pvt["Rs_m3m3"] = [0, 90, 80]
    with pytest.raises(ValueError, match="Rs"):
        stage(pvt=pvt)
    pvt = config()["pvt"]
    pvt["extended"] = {
        "oil": {
            "rs_values": [0, 100],
            "tables": {
                str(rs): {
                    "pressure_MPa": [0.1, 1, 5],
                    "Bo": [1.1] * 3,
                    "mu_o_cP": [5] * 3,
                    "Rv_m3m3": [0.001] * 3,
                }
                for rs in [0, 100]
            },
        }
    }
    with pytest.raises(ValueError, match="Rv"):
        stage(pvt=pvt).separate(ComponentStream(0.001, 0, 0.04))


def test_facility_cli_preserves_failed_capacity_report(tmp_path):
    import os
    import subprocess

    cfg = config()
    cfg["limits"][0]["maximum"] = 1
    source = tmp_path / "facility.json"
    result = tmp_path / "report.json"
    source.write_text(json.dumps(cfg))
    command = [
        sys.executable,
        str(ROOT / "scripts/run_surface_facility.py"),
        str(source),
        "--output",
        str(result),
    ]
    completed = subprocess.run(
        command,
        check=False,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 3
    report = json.loads(result.read_text())
    assert not report["feasible"] and report["violations"][0]["violation"] > 0
