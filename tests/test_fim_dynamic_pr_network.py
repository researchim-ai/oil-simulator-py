"""Actual FIM coupled to a finite PR vessel graph, no invented pipe inventory."""

import copy
import json
import pickle
import sys

import pytest
import torch
from test_fim_blackoil_surface import ROOT
from test_fim_dynamic_receiver import build as single

sys.path.insert(0, str(ROOT / "scripts"))
from check_dynamic_receiver import audit_step

from simulator.dynamic_receiver_coupling import DynamicReceiverCoupledFIM
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState
from surface.stratified_pr_flow import StratifiedPRFlow
from surface.stratified_pr_network import StratifiedConnection, StratifiedPRNetwork

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def build(device="cpu", *, cycle=False):
    source = single(device)
    v = source.receiver
    first = StratifiedPRFlow(v.column, v.wall)
    second_col = StratifiedPRColumn.from_dict(v.column.to_dict(), device=device)
    second = StratifiedPRFlow(
        second_col,
        v.wall,
        outlets=[type(v.outlets[0])("export", 980000.0, 1e-7, height_m=1.0, diameter_m=0.05)],
    )
    a = first.initialize(source.receiver_state.column, 320)
    bcol = StratifiedPRState.from_dict(second_col.initialize(990000.0, 320, [0.5, 0.5])["state"])
    b = second.initialize(bcol, 320)
    vessels = {"inlet": first, "separator": second}
    states = {"inlet": a, "separator": b}
    links = [StratifiedConnection("link", "inlet", "separator", 1, 1, 0.05, 1e-7)]
    if cycle:
        third_col = StratifiedPRColumn.from_dict(v.column.to_dict(), device=device)
        third = StratifiedPRFlow(third_col, v.wall)
        col = StratifiedPRState.from_dict(third_col.initialize(995000.0, 320, [0.5, 0.5])["state"])
        vessels["buffer"] = third
        states["buffer"] = third.initialize(col, 320)
        links += [
            StratifiedConnection("side", "separator", "buffer", 1, 1, 0.05, 1e-8),
            StratifiedConnection("return", "buffer", "inlet", 1, 1, 0.05, 1e-8),
        ]
    network = StratifiedPRNetwork(vessels, links, initial_step_s=0.01)
    return DynamicReceiverCoupledFIM(
        source.solver,
        network,
        network.initialize(states),
        source.assay,
        source.thermal_conditions,
        0.0,
        inlet_vessel="inlet",
    )


@pytest.mark.parametrize("device", DEVICES)
def test_real_fim_network_pressure_and_independent_open_node_balances(device, tmp_path):
    model = build(device)
    before = model.receiver_state
    assert model.step(0.01), model.last_attempt
    r = model.report
    assert r["model"] == "dynamic_pr_vessel_network_fim_v1"
    assert max(map(abs, r["component_residuals_mol"])) < 1e-8
    assert abs(r["energy_residual_j"]) < 1e-5
    assert r["cumulative"]["export_component_moles"][0] > 0
    assert model.receiver_state.time_s == 0.01
    audit = audit_step({**model.to_dict(), "initial_state": before.to_dict()}, r)
    assert audit["accepted"]
    bad = copy.deepcopy(r)
    bad["stages"][0]["flow"]["network_report"]["transfers"][0]["component_rates_mol_s"][0] *= 1.01
    with pytest.raises(ValueError):
        audit_step({**model.to_dict(), "initial_state": before.to_dict()}, bad)
    replay = single(device).solver
    for stage in r["stages"]:
        replay.wells.set_surface_boundary_pressures(
            {replay.wells.surface_network.root: stage["pressure_trials"][-1]["pressure_pa"]}
        )
        assert replay.step(stage["dt_s"])
    assert torch.equal(replay.fluid.pressure, model.solver.fluid.pressure)
    for a, b in zip(replay.wells.wells, model.solver.wells.wells):
        for name, value in a.__dict__.items():
            if name.startswith("cum_"):
                assert value == b.__dict__[name]

    for stage in r["stages"]:
        assert stage["flow"]["network_report"]["transfers"][0]["component_rates_mol_s"][0] > 0
    if device == "cuda":
        assert model.solver._graphs is not None
    (tmp_path / f"coupled-network-{device}.json").write_text(
        json.dumps(
            {"model": model.to_dict(), "report": r, "audit": audit}, indent=2, allow_nan=False
        )
        + "\n"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_graph_restart_isolation_and_atomic_temporal_failure(device, monkeypatch):
    model = build(device)
    assert model.step(0.01), model.last_attempt
    model.connection_openings = {"link": 0.0}
    restored = pickle.loads(pickle.dumps(model))
    assert model.step(0.01) and restored.step(0.01)
    assert model.report == restored.report
    assert all(
        s["flow"]["network_report"]["transfers"][0]["component_rates_mol_s"] == [0.0, 0.0]
        for s in model.report["stages"]
    )
    previous = copy.deepcopy(model.report)
    state = model.receiver_state
    pressure = model.solver.fluid.pressure.clone()
    net = model.solver.wells.surface_network
    monkeypatch.setattr(model, "_temporal_error", lambda *args: 2.0)
    assert not model.step(0.01)
    assert model.report == previous and model.receiver_state == state
    assert torch.equal(model.solver.fluid.pressure, pressure)
    assert model.solver.wells.surface_network is net
    assert model.connection_openings == {"link": 0.0}


def test_graph_commands_and_inlet_binding_validated_before_trials():
    model = build()
    p = model.solver.fluid.pressure.clone()
    model.outlet_openings = {"export": 0.0}
    with pytest.raises(ValueError, match="outlet"):
        model.step(0.01)
    assert torch.equal(p, model.solver.fluid.pressure)
    model.outlet_openings = {}
    model.connection_openings = {"imaginary": 0.0}
    with pytest.raises(ValueError, match="connection"):
        model.step(0.01)


def test_absent_species_activation_is_rejected_atomically():
    from surface.liquid import PropertyModelError

    model = build()
    states = {}
    for n, v in model.receiver.vessels.items():
        col = StratifiedPRState.from_dict(v.column.initialize(1e6, 320, [1, 0])["state"])
        states[n] = v.initialize(col, 320)
    model.receiver_state = model.receiver.initialize(states)
    old = model.receiver_state
    pressure = model.solver.fluid.pressure.clone()
    with pytest.raises(PropertyModelError, match="activates"):
        model.step(0.005)
    assert model.receiver_state == old and model.report is None
    assert torch.equal(pressure, model.solver.fluid.pressure)


@pytest.mark.parametrize("device", DEVICES)
def test_three_vessel_ring_with_signed_internal_flows(device):
    model = build(device, cycle=True)
    old = model.receiver_state
    assert model.step(0.005), model.last_attempt
    assert len(model.receiver_state.vessels) == 3
    raw = model.report["stages"][-1]["flow"]["network_report"]
    rates = {p["name"]: p["component_rates_mol_s"][0] for p in raw["transfers"]}
    assert rates["link"] > 0 and rates["side"] < 0 and rates["return"] < 0
    assert audit_step({**model.to_dict(), "initial_state": old.to_dict()}, model.report)["accepted"]


def test_simulator_factory_cuts_rejected_interval_and_advances_both_clocks():
    from test_fim_blackoil_surface import config

    from simulator.fluid import Fluid
    from simulator.reservoir import Reservoir
    from simulator.simulation import Simulator
    from simulator.well import WellManager

    model = build()
    cfg = config()
    cfg["simulation"].update(
        dynamic_receiver={**model.to_dict(), "initial_state": model.receiver_state.to_dict()},
        initial_time_step_days=0.02 / 86400,
        min_time_step_days=1e-9,
        max_time_step_days=0.02 / 86400,
    )
    cfg = json.loads(json.dumps(cfg))
    dev = torch.device("cpu")
    r = Reservoir(cfg["reservoir"], device=dev)
    f = Fluid(cfg["fluid"], r, device=dev)
    sim = Simulator(r, f, WellManager(cfg["wells"], r), cfg["simulation"], device=dev)
    records = []
    sim.substep_hook = lambda _: records.append(copy.deepcopy(sim.facility_coupling.report))
    attempted = []
    original = sim.facility_coupling.step

    def recorded(*args, **kwargs):
        result = original(*args, **kwargs)
        attempted.append(result)
        return result

    sim.facility_coupling.step = recorded
    assert sim.run_step(0.02)
    assert False in attempted
    assert len(records) >= 2
    assert sim.current_time == sim.facility_coupling.receiver_state.time_s == 0.02
    assert sum(s["dt_s"] for s in records) == pytest.approx(0.02, abs=1e-14)
    assert all(s["temporal_error_ratio"] <= 1 for s in records)
    assert sim.facility_coupling.solver is sim.fim_solver
