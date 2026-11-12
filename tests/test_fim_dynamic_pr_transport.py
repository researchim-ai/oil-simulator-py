"""Real FIM with finite-volume pipe stocks and implicit export pressure."""

import copy
import json
import pickle
import sys
from pathlib import Path

import pytest
import torch
from test_fim_dynamic_receiver import build as single

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_dynamic_receiver import audit_step

from simulator.dynamic_receiver_coupling import DynamicReceiverCoupledFIM
from surface.pr_equilibrium_transport import PREquilibriumNode
from surface.pr_open_transport import PROpenTransportNetwork, PRTransportPressurePort
from surface.pr_transport_builder import (
    PREquilibriumPipe,
    build_equilibrium_transport,
    initialize_grouped,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def build(device="cpu"):
    original = single(device)
    fluid = original.receiver.column.fluid
    transport = build_equilibrium_transport(
        fluid,
        [PREquilibriumNode("inlet", 0.1), PREquilibriumNode("export", 0.1)],
        [PREquilibriumPipe("line", "inlet", "export", 10.0, 0.05, 2)],
        gravity_m_s2=9.80665,
        initial_step_s=0.001,
        maximum_step_s=0.1,
    )
    initial = {
        "pressure_pa": 1e6,
        "temperature_k": 320.0,
        "mole_fractions": [0.5, 0.5],
        "velocity_m_s": 0.0,
    }
    state = initialize_grouped(transport, {n: initial for n in ["inlet", "export", "line"]})
    boundary = PROpenTransportNetwork(
        transport,
        pressure_ports=[
            PRTransportPressurePort("export", "export", 990000.0, 320.0, (0.5, 0.5), 1e-7)
        ],
    )
    return DynamicReceiverCoupledFIM(
        original.solver,
        boundary,
        state,
        original.assay,
        original.thermal_conditions,
        0.0,
        inlet_vessel="inlet",
    )


@pytest.mark.parametrize("device", DEVICES)
def test_actual_fim_distributed_pipe_pressure_and_independent_audit(device, tmp_path):
    m = build(device)
    before = m.receiver_state
    assert m.step(0.001), m.last_attempt
    r = m.report
    assert audit_step({**m.to_dict(), "initial_state": before.to_dict()}, r)["accepted"]
    assert any(
        abs(s.momentum_kg_m_s) > 0
        for name, s in m.receiver_state.nodes
        if name.startswith("line__")
    )
    assert r["model"] == "dynamic_pr_transport_fim_v1"
    assert r["cumulative"]["export_component_moles"][0] > 0
    if device == "cuda":
        assert m.solver._graphs is not None
    saved = pickle.loads(pickle.dumps(m))
    m.outlet_openings = {"export": 0.0}
    saved.outlet_openings = {"export": 0.0}
    assert m.step(0.001) and saved.step(0.001)
    assert m.receiver_state.to_dict() == saved.receiver_state.to_dict()
    assert m.report == saved.report
    for stage in m.report["stages"]:
        assert stage["flow"]["outlet_component_moles"] == [0.0, 0.0]
    (tmp_path / f"fim-distributed-{device}.json").write_text(
        json.dumps({"model": m.to_dict(), "first_report": r, "second_report": m.report}, indent=2)
        + "\n"
    )


def test_distributed_temporal_rejection_restores_pipe_reservoir_and_production(monkeypatch):
    m = build()
    old = m.receiver_state
    fields = m.solver.fluid.pressure.clone()
    monkeypatch.setattr(m, "_temporal_error", lambda *a: 2.0)
    assert not m.step(0.001)
    assert m.receiver_state is old and m.report is None
    assert torch.equal(m.solver.fluid.pressure, fields)


def test_distributed_configuration_roundtrip():
    m = build()
    raw = {**m.to_dict(), "initial_state": m.receiver_state.to_dict()}
    other = DynamicReceiverCoupledFIM.from_dict(m.solver, json.loads(json.dumps(raw)))
    assert other.model_id == m.model_id
    assert other.to_dict() == m.to_dict()
    bad = copy.deepcopy(raw)
    bad["inlet_height_m"] = 1.0
    with pytest.raises(ValueError, match="height"):
        DynamicReceiverCoupledFIM.from_dict(m.solver, bad)
