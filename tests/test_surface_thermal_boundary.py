"""Thermodynamic boundary integration without pretending FIM solves energy."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.thermal_surface_boundary import characterize_well_thermal_boundaries
from surface import PengRobinsonFluid, PengRobinsonMixture, ThermalStream
from surface.component_energy import ComponentEnergyStream, StockComponentAssay
from surface.equilibrium_inventory import EquilibriumInventory
from surface.equilibrium_process import run_equilibrium_process
from surface.equilibrium_properties import EquilibriumFluid
from surface.liquid import ConservationError, EquipmentOperatingError, PropertyModelError

DATA = json.loads((Path(__file__).parent / "data/surface/pr_mixture_coolprop.json").read_text())
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def fluid(device):
    return EquilibriumFluid(
        [PengRobinsonFluid(**c) for c in DATA["components"]], [[0, 0], [0, 0]], device=device
    )


@pytest.mark.parametrize("device", DEVICES)
def test_legacy_pr_bridge_preserves_composition_reference_and_energy(device):
    f = fluid(device)
    oldfluid = PengRobinsonMixture(f.components, f.binary_interactions)
    old = ThermalStream.from_pt(oldfluid, 1e6, 320, 3, [0.3, 0.7])
    new = ComponentEnergyStream.from_thermal_stream(old, f, elevation_m=-100, velocity_m_s=2)
    assert new.enthalpy_rate_w(f) == pytest.approx(old.enthalpy_rate_w, abs=1e-9)
    again = new.to_thermal_stream(f, temperature_guess_k=320)
    assert again.component_molar_rates_mol_s == old.component_molar_rates_mol_s
    assert again.model_id == old.model_id
    assert again.molar_enthalpy_j_mol == pytest.approx(old.molar_enthalpy_j_mol, abs=1e-8)
    wrong = EquilibriumFluid(
        [replace(c, reference_temperature_k=300) for c in f.components],
        f.binary_interactions,
        device=device,
    )
    with pytest.raises(PropertyModelError, match="mismatch"):
        ComponentEnergyStream.from_thermal_stream(old, wrong)


@pytest.mark.parametrize("device", DEVICES)
def test_well_report_boundary_to_equilibrium_preparation_process(device):
    f = fluid(device)
    assay = StockComponentAssay(f.names, ((0.2, 0.8), (0.1, 0.9), (0.9, 0.1)), f.model_id)
    reports = {
        "P1": {"component_mass_rates_kg_s": [1, 0.2, 0.1], "bhp_pa": 1e6},
        "P2": {"component_mass_rates_kg_s": [0.5, 0.2, 0.2], "bhp_pa": 1.2e6},
    }
    conditions = {
        "P1": {"temperature_k": 320, "elevation_m": -20},
        "P2": {"temperature_k": 330, "elevation_m": -10},
    }
    before = json.dumps(reports, sort_keys=True)
    boundary = characterize_well_thermal_boundaries(reports, conditions, f, assay)
    result = run_equilibrium_process(
        f,
        boundary["streams"],
        {
            name: {"pressure_pa": 5e5, "elevation_m": 0, "temperature_guess_k": 320}
            for name in reports
        },
        {"pressure_pa": 5e5, "temperature_guess_k": 320},
    )
    assert boundary["reservoir_energy_equation_solved"] is False
    assert boundary["pvt_eos_characterization_calibrated"] is False
    assert abs(result["balances"]["energy_residual_w"]) < 1e-5
    assert max(map(abs, result["balances"]["component_residuals_mol_s"])) < 1e-9
    assert result["balances"]["entropy_production_w_k"] > 0
    assert json.dumps(reports, sort_keys=True) == before
    bad = {"P1": {}, "P2": conditions["P2"]}
    with pytest.raises(ValueError, match="thermal boundary"):
        characterize_well_thermal_boundaries(reports, bad, f, assay)
    with pytest.raises(TypeError, match="Assay"):
        characterize_well_thermal_boundaries(reports, conditions, f, None)


@pytest.mark.parametrize("device", DEVICES)
def test_pure_latent_heat_uv_inventory_and_thermodynamic_inspection(device):
    f = fluid(device)
    p = f.components[1].saturation_at_temperature(200)["pressure_pa"]
    l = f.homogeneous(p, 200, [0, 1], "liquid")
    v = f.homogeneous(p, 200, [0, 1], "vapor")
    u = 10 * float(
        (0.7 * l["molar_internal_energy_j_mol"] + 0.3 * v["molar_internal_energy_j_mol"]).cpu()
    )
    volume = 10 * float((0.7 * l["molar_volume_m3_mol"] + 0.3 * v["molar_volume_m3_mol"]).cpu())
    q = f.flash_uv([0, 10], u, volume, pressure_guess_pa=p * 1.05, temperature_guess_k=202)
    assert q["phase_count"] == 2
    assert float(q["temperature_k"].cpu()) == pytest.approx(200, abs=2e-6)
    assert float(q["pressure_pa"].cpu()) == pytest.approx(p, rel=2e-7)
    assert float(q["phases"][1]["molar_fraction"].cpu()) == pytest.approx(0.3, abs=2e-7)
    assert abs(float(q["energy_residual_j"].cpu())) < 1e-5
    state = EquilibriumInventory((0, 10), u, volume, p, 200, f.model_id)
    assert state.inspect(f)["phase_count"] == 2
    with pytest.raises(ConservationError):
        replace(state, temperature_k=210).inspect(f)


@pytest.mark.parametrize("device", DEVICES)
def test_work_and_heat_cannot_violate_second_law(device):
    f = fluid(device)
    s = ComponentEnergyStream.from_pt(f, [1, 1], 1e5, 320)
    with pytest.raises(ConservationError, match="entropy"):
        s.transport(f, 2e5, shaft_power_w=1, temperature_guess_k=320)
    with pytest.raises(ValueError, match="utility"):
        s.transport(f, 1e5, heat_w=1, temperature_guess_k=320)
    with pytest.raises(EquipmentOperatingError, match="utility"):
        s.transport(f, 1e5, heat_w=1, utility_temperature_k=300, temperature_guess_k=320)
    proper = s.transport(f, 1e5, heat_w=1, utility_temperature_k=400, temperature_guess_k=320)
    assert proper.total_energy_rate_w == s.total_energy_rate_w + 1


@pytest.mark.parametrize("device", DEVICES)
def test_zero_flow_boundary_keeps_registry_without_fictitious_flash(device, monkeypatch):
    f = fluid(device)

    def fail(*args, **kwargs):
        raise AssertionError("No material at zero flow")

    monkeypatch.setattr(f, "flash_pt", fail)
    s = ComponentEnergyStream.from_pt(f, [0, 0], 1e5, 320)
    assert s.equilibrate(f) is None
    assert s.total_energy_rate_w == 0
