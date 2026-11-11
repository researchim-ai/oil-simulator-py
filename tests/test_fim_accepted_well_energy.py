"""Actual accepted reservoir material, declared caloric boundary and rollback."""

import copy
import pickle

import pytest
import torch
from test_fim_facility_coupling import build
from test_surface_thermal_boundary import fluid

from simulator.accepted_well_energy import AcceptedWellEnergyProcess
from surface.component_energy import StockComponentAssay
from surface.liquid import PropertyModelError


def case():
    coupled = build(False)
    f = fluid("cpu")
    names = [w.name for w in coupled.solver.wells.wells]
    assay = StockComponentAssay(f.names, ((0.2, 0.8), (0.1, 0.9), (0.9, 0.1)), f.model_id)
    coupled.energy_process = AcceptedWellEnergyProcess(
        f,
        assay,
        {name: {"temperature_k": 330.0, "elevation_m": 0.0} for name in names},
        {
            name: {"pressure_pa": 5e5, "elevation_m": 0.0, "temperature_guess_k": 330.0}
            for name in names
        },
        {"pressure_pa": 5e5, "temperature_guess_k": 330.0},
    )
    return coupled


def test_actual_fim_material_reaches_caloric_exports_and_survives_restart(tmp_path):
    model = case()
    assert model.step(1000.0)
    first = copy.deepcopy(model.report["energy_process"])
    assert first["reservoir_energy_equation_solved"] is False
    assert first["pressure_feedback_from_caloric_process"] is False
    assert max(map(abs, first["cumulative"]["component_residuals_mol"])) < 1e-8
    assert abs(first["cumulative"]["energy_residual_j"]) < 1e-4
    for name, well in model.solver.wells.surface_report["wells"].items():
        assert first["boundaries"][name]["mass_residual_kg_s"] == pytest.approx(0.0, abs=1e-12)
        assert first["boundaries"][name]["stream"]["total_energy_rate_w"] != 0.0
        assert sum(well["component_mass_rates_kg_s"]) > 0.0
    model.controls = {"gas_valve_max_opening": 0.01}
    model.energy_controls = {
        "thermal_conditions": {
            name: {"temperature_k": 335.0, "elevation_m": 0.0}
            for name in model.energy_process.thermal_conditions
        }
    }
    restored = pickle.loads(pickle.dumps(model))
    assert model.step(1000.0) and restored.step(1000.0)
    assert model.report == restored.report
    assert model.report["fraction"] < 1.0
    assert model.report["energy_process"]["cumulative"]["elapsed_s"] == 2000.0
    assert model.report["energy_process"]["applied_controls"] == model.energy_controls
    import json

    (tmp_path / "actual-fim-energy.json").write_text(
        json.dumps(
            {
                "energy_model": model.energy_process.to_dict(),
                "first": first,
                "final": model.report["energy_process"],
                "capacity_fraction": model.report["fraction"],
                "accepted_wells": model.solver.wells.surface_report["wells"],
                "material_account": model.report["material_account"],
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def test_failed_caloric_conversion_rolls_back_actual_reservoir_and_both_reports(monkeypatch):
    model = case()
    assert model.step(1000.0)
    previous = copy.deepcopy(model.report)
    pressure = model.solver.fluid.pressure.clone()

    def fail(*args, **kwargs):
        raise PropertyModelError("Caloric target outside physical domain")

    monkeypatch.setattr(AcceptedWellEnergyProcess, "run", fail)
    with pytest.raises(PropertyModelError, match="Caloric"):
        model.step(1000.0)
    assert model.report == previous
    assert torch.equal(model.solver.fluid.pressure, pressure)


def test_json_simulator_configuration_binds_caloric_process_to_the_accepted_step():
    import json
    from dataclasses import asdict

    from test_fim_blackoil_surface import config
    from test_fim_facility_coupling import facility_spec

    from simulator.fluid import Fluid
    from simulator.reservoir import Reservoir
    from simulator.simulation import Simulator
    from simulator.well import WellManager

    original = case()
    raw = json.loads(json.dumps(original.energy_process.to_dict()))
    restored = AcceptedWellEnergyProcess.from_dict(raw)
    assert restored.model_id == original.energy_process.model_id
    c = config()
    c["simulation"]["coupled_facility"] = {
        "facility": facility_spec(),
        "control": asdict(original.control),
        "energy_process": raw,
    }
    device = torch.device("cpu")
    reservoir = Reservoir(c["reservoir"], device=device)
    fluid_ = Fluid(c["fluid"], reservoir, device=device)
    sim = Simulator(
        reservoir, fluid_, WellManager(c["wells"], reservoir), c["simulation"], device=device
    )
    assert sim.run_step(1000.0)
    assert sim.current_time == 1000.0
    assert sim.facility_coupling.report["energy_process"]["cumulative"]["elapsed_s"] == 1000.0
    assert sim.facility_coupling.solver is sim.fim_solver
