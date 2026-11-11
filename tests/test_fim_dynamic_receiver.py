"""Actual FIM plus finite PR storage/withdrawals, pressure feedback and rollback.

The methane/ethane stock assay is a manufactured coupling test, not a field
water/oil characterization or experimental calibration.
"""

import copy
import json
import pickle
import sys

import pytest
import torch
from test_fim_blackoil_surface import ROOT, report_rates, solver

from simulator.dynamic_receiver_coupling import DynamicReceiverCoupledFIM
from surface.component_energy import StockComponentAssay
from surface.compositional_separator import EquilibriumOutlet
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState
from surface.stratified_pr_flow import StratifiedPRFlow
from surface.vessel_wall import VesselWall


def build(device="cpu", *, closed=False):
    if device == "cpu":
        s = solver()
    else:
        from test_fim_surface_cuda import build as device_solver

        s = device_solver(device, backend="torch_device")
    raw = json.loads((ROOT / "configs/surface_stratified_pr_gas.json").read_text())
    raw["system"]["geometry"]["length_m"] = 20.0
    raw["system"]["elevation_m"] = 0.0
    c = StratifiedPRColumn.from_dict(raw["system"], device=device)
    column = StratifiedPRState.from_dict(c.initialize(**raw["initial"])["state"])
    flow = StratifiedPRFlow(
        c,
        VesselWall(1000.0, 2.0, 1.0, 310.0),
        outlets=[
            EquilibriumOutlet(
                "export",
                990000.0,
                1e-10,
                height_m=1.0,
                diameter_m=0.05,
                opening=0.0 if closed else 1.0,
            )
        ],
        initial_step_s=0.02,
    )
    state = flow.initialize(column, column.temperature_k)
    assay = StockComponentAssay(
        c.fluid.names, ((0.2, 0.8), (0.1, 0.9), (0.9, 0.1)), c.fluid.model_id
    )
    conditions = {w.name: {"temperature_k": 330.0, "elevation_m": -40.0} for w in s.wells.wells}
    return DynamicReceiverCoupledFIM(s, flow, state, assay, conditions, 0.0)


def test_dynamic_receiver_end_pressure_balances_and_no_trial_production(tmp_path):
    model = build()
    before = model.receiver_state
    assert model.step(0.02), model.last_attempt
    report = model.report
    assert report["pressure_feedback_from_caloric_process"] is True
    assert report["reservoir_energy_equation_solved"] is False
    assert model.receiver_state.time_s == 0.02
    assert max(map(abs, report["component_residuals_mol"])) < 1e-8
    assert abs(report["energy_residual_j"]) < 1e-5
    assert report["cumulative"]["feed_component_moles"][0] > 0
    assert report["cumulative"]["export_component_moles"][0] > 0
    assert model.receiver_state.column.component_moles != before.column.component_moles
    for stage in report["stages"]:
        last = stage["pressure_trials"][-1]
        assert abs(last["pressure_defect_pa"]) < 0.012
        assert last["physical_trial"]
        assert len(stage["pressure_trials"]) > 1
        rates = [stream["component_molar_rates_mol_s"] for stream in stage["inlet_streams"]]
        for j in range(2):
            assert stage["flow"]["feed_component_moles"][j] == pytest.approx(
                stage["dt_s"] * sum(row[j] for row in rates), rel=1e-12
            )
    # Independent replay of only the two accepted boundary pressures, never
    # production from the rejected/full-step trials.
    direct = solver()
    for stage in report["stages"]:
        direct.wells.set_surface_boundary_pressures(
            {direct.wells.surface_network.root: stage["pressure_trials"][-1]["pressure_pa"]}
        )
        assert direct.step(stage["dt_s"])
    assert torch.equal(direct.fluid.pressure, model.solver.fluid.pressure)
    for a, b in zip(direct.wells.wells, model.solver.wells.wells):
        for key, value in a.__dict__.items():
            if key.startswith("cum_"):
                assert value == b.__dict__[key]
    # A frozen-pressure calculation must produce a different actual flux;
    # this distinguishes two-way coupling from downstream postprocessing.
    frozen = solver()
    frozen.wells.set_surface_boundary_pressures(
        {frozen.wells.surface_network.root: model._pressure(before)}
    )
    assert frozen.step(0.02)
    assert report_rates(model.solver).sum() < report_rates(frozen).sum()
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_dynamic_receiver import audit_step

    spec = {**model.to_dict(), "initial_state": before.to_dict()}
    assert audit_step(spec, report)["accepted"]
    bad = copy.deepcopy(report)
    bad["stages"][0]["accepted_wells"]["P1"]["component_mass_rates_kg_s"][0] *= 1.01
    with pytest.raises(ValueError):
        audit_step(spec, bad)
    bad = copy.deepcopy(report)
    bad["stages"][-1]["pressure_trials"][-1]["pressure_pa"] += 10
    with pytest.raises(ValueError, match="pressure"):
        audit_step(spec, bad)
    (tmp_path / "dynamic-receiver.json").write_text(json.dumps(report, indent=2, allow_nan=False))


def test_restart_and_commands_preserve_dynamic_storage_and_pressure_feedback():
    model = build()
    assert model.step(0.02), model.last_attempt
    model.outlet_openings = {"export": 0.0}
    other = pickle.loads(pickle.dumps(model))
    assert model.step(0.02, current_time=0.02), model.last_attempt
    assert other.step(0.02, current_time=0.02), other.last_attempt
    assert model.report == other.report
    assert model.report["cumulative"]["elapsed_s"] == 0.04
    assert model.report["stages"][-1]["flow"]["outlet_component_moles"] == [0.0, 0.0]
    assert torch.equal(model.solver.fluid.pressure, other.solver.fluid.pressure)


def test_failed_surface_or_temporal_solve_restores_every_accepted_state(monkeypatch):
    model = build()
    assert model.step(0.02), model.last_attempt
    state, report = model.receiver_state, copy.deepcopy(model.report)
    pressure = model.solver.fluid.pressure.clone()
    original_network = model.solver.wells.surface_network
    original_wells = copy.deepcopy(model.solver.wells.surface_report)
    monkeypatch.setattr(model, "_temporal_error", lambda *args: 2.0)
    assert not model.step(0.02)
    assert "temporal" in model.last_attempt["reason"]
    assert model.receiver_state == state and model.report == report
    assert torch.equal(model.solver.fluid.pressure, pressure)
    assert model.solver.wells.surface_network is original_network
    assert model.solver.wells.surface_report == original_wells
    assert model.outlet_openings == {}


def test_rejects_unbound_commands_and_clock_before_reservoir_mutation():
    model = build()
    before = model.solver.fluid.pressure.clone()
    with pytest.raises(ValueError, match="clocks"):
        model.step(0.02, current_time=1.0)
    model.outlet_openings = {"invented": 0.0}
    with pytest.raises(ValueError, match="outlet"):
        model.step(0.02)
    assert torch.equal(before, model.solver.fluid.pressure)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_actual_dynamic_receiver_cpu_cuda():
    a, b = build(), build("cuda")
    assert a.step(0.02), a.last_attempt
    assert b.step(0.02), b.last_attempt
    assert b.solver._graphs is not None
    assert torch.allclose(
        a.solver.fluid.pressure, b.solver.fluid.pressure.cpu(), rtol=1e-9, atol=0.001
    )
    assert torch.allclose(report_rates(a.solver), report_rates(b.solver), rtol=1e-6, atol=1e-9)
    assert a.receiver_state.column.component_moles == pytest.approx(
        b.receiver_state.column.component_moles, rel=1e-9, abs=1e-9
    )


def test_json_factory_rejects_two_drivers_and_binds_one_shared_solver():
    from test_fim_blackoil_surface import config

    from simulator.fluid import Fluid
    from simulator.reservoir import Reservoir
    from simulator.simulation import Simulator
    from simulator.well import WellManager

    original = build()
    cfg = config()
    cfg["simulation"]["dynamic_receiver"] = {
        **original.to_dict(),
        "initial_state": original.receiver_state.to_dict(),
    }
    cfg = json.loads(json.dumps(cfg))
    dev = torch.device("cpu")
    r = Reservoir(cfg["reservoir"], device=dev)
    f = Fluid(cfg["fluid"], r, device=dev)
    sim = Simulator(r, f, WellManager(cfg["wells"], r), cfg["simulation"], device=dev)
    assert sim.facility_coupling.solver is sim.fim_solver
    assert sim.facility_coupling.model_id == original.model_id
    assert sim.run_step(0.02)
    assert sim.current_time == sim.facility_coupling.receiver_state.time_s == 0.02
    restored = pickle.loads(pickle.dumps(sim))
    assert restored.facility_coupling.solver is restored.fim_solver
    cfg["simulation"]["coupled_facility"] = {}
    with pytest.raises(ValueError, match="two drivers"):
        Simulator(r, f, WellManager(cfg["wells"], r), cfg["simulation"], device=dev)
