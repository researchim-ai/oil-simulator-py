"""Actual FIM/capillary shared-stock coupling, independent audit and rollback."""

import json
import pickle
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from check_dynamic_receiver import audit_step
from test_fim_dynamic_receiver import build as reservoir_case
from test_surface_pr_capillary_reuse import shared_case

from simulator.dynamic_receiver_coupling import DynamicReceiverCoupledFIM
from surface.blackoil_network import BlackOilProductionNetwork
from surface.component_energy import StockComponentAssay
from surface.pr_capillary_surface_plant import PRCapillarySurfacePlant
from surface.pr_open_gravity_transport import PROpenGravityTransportNetwork


def build(device="cpu"):
    source = reservoir_case(device)
    plant, state = shared_case(device)
    plant = PRCapillarySurfacePlant(
        PROpenGravityTransportNetwork(plant.graph, pressure_ports=plant.pressure_ports),
        plant.capillary,
    )
    state = replace(state, model_id=plant.model_id)
    network = source.solver.wells.surface_network.to_dict()
    elevation = plant.transport.nodes[plant.transport.names.index("inlet")].elevation_m
    for node in network["nodes"]:
        if node["name"] == source.solver.wells.surface_network.root:
            node["elevation_m"] = elevation
    tree = BlackOilProductionNetwork.from_dict(network)
    tree.model = tree.model.to(source.solver.device)
    source.solver.wells.surface_network = tree
    assay = StockComponentAssay(
        plant.fluid.names, source.assay.stock_mass_fractions, plant.fluid.model_id
    )
    return DynamicReceiverCoupledFIM(
        source.solver, plant, state, assay, source.thermal_conditions, 0.0, inlet_vessel="inlet"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actual_capillary_fim_audit_and_restart(device, tmp_path):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = build(device)
    initial = model.receiver_state
    assert model.step(1e-7), model.last_attempt
    first = deepcopy(model.report)
    spec = {**model.to_dict(), "initial_state": initial.to_dict()}
    assert audit_step(spec, first)["accepted"]
    assert first["temporal_error_ratio"] <= 1
    assert len(model.receiver_state.columns) == 1
    assert all(v > 0 for v in first["cumulative"]["feed_component_moles"])
    assert model.receiver_state.capillary_pipes != initial.capillary_pipes
    restored = pickle.loads(pickle.dumps(model))
    model.outlet_openings = restored.outlet_openings = {"boundary": 0.0}
    previous = model.receiver_state
    assert model.step(1e-7) and restored.step(1e-7)
    assert model.report == restored.report
    assert all(s["flow"]["outlet_component_moles"] == [0.0, 0.0] for s in model.report["stages"])
    assert audit_step({**model.to_dict(), "initial_state": previous.to_dict()}, model.report)[
        "accepted"
    ]
    raw = {**model.to_dict(), "initial_state": model.receiver_state.to_dict()}
    rebuilt = DynamicReceiverCoupledFIM.from_dict(
        model.solver, json.loads(json.dumps(raw)), device=device
    )
    assert rebuilt.model_id == model.model_id and rebuilt.receiver_state == model.receiver_state
    if device == "cuda":
        assert model.solver._graphs is not None
    (tmp_path / f"capillary-fim-{device}.json").write_text(
        json.dumps({"spec": spec, "first": first, "second": model.report}, indent=2) + "\n"
    )


def test_rejected_whole_system_error_restores_capillary_columns_and_reservoir(monkeypatch):
    model = build()
    old = model.receiver_state
    snapshot = old.to_dict()
    pressure = model.solver.fluid.pressure.clone()
    network = model.solver.wells.surface_network
    monkeypatch.setattr(model, "_temporal_error", lambda *args: 2.0)
    assert not model.step(1e-7)
    assert model.receiver_state is old and old.to_dict() == snapshot and model.report is None
    assert torch.equal(model.solver.fluid.pressure, pressure)
    assert model.solver.wells.surface_network is network


def test_independent_audit_rejects_changed_packets_shared_stocks_and_surface_work():
    model = build()
    initial = model.receiver_state
    assert model.step(1e-7), model.last_attempt
    spec = {**model.to_dict(), "initial_state": initial.to_dict()}
    for change in ("packet", "shared", "work", "pipe"):
        bad = json.loads(json.dumps(model.report))
        stage = bad["stages"][0]
        raw = stage["flow"]["plant_report"]
        if change == "packet":
            stage["accepted_wells"]["P1"]["component_mass_rates_kg_s"][0] *= 1.01
        elif change == "shared":
            raw["midpoint"]["columns"]["A"]["component_moles"][0] *= 1.01
        elif change == "work":
            raw["geometry_work_j"]["A"] += 1.0
        else:
            raw["state"]["capillary_pipes"] = initial.to_dict()["capillary_pipes"]
        with pytest.raises(ValueError):
            audit_step(spec, bad)
