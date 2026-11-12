"""Actual accepted FIM coupled to distributed pipes and a resolved gravity vessel."""

import json
import pickle
import sys
from pathlib import Path

import pytest
import torch
from test_fim_dynamic_receiver import build as single
from test_surface_pr_open_gravity_transport import build as surface

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_dynamic_receiver import audit_step

from simulator.dynamic_receiver_coupling import DynamicReceiverCoupledFIM
from surface.pr_open_gravity_transport import PROpenGravityTransportNetwork

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def build(device="cpu"):
    source = single(device)
    model, state = surface(device)
    # The FIM factory validates the actual physical elevation of its inlet.
    assert model.transport.nodes[model.transport.names.index("inlet")].elevation_m == 0.0
    receiver = PROpenGravityTransportNetwork(model.graph, pressure_ports=model.pressure_ports)
    return DynamicReceiverCoupledFIM(
        source.solver,
        receiver,
        state,
        source.assay,
        source.thermal_conditions,
        0.0,
        inlet_vessel="inlet",
    )


@pytest.mark.parametrize("device", DEVICES)
def test_actual_resolved_fim_pressure_audit_and_checkpoint(device, tmp_path):
    m = build(device)
    old = m.receiver_state
    assert m.step(1e-6), m.last_attempt
    report = m.report
    assert audit_step({**m.to_dict(), "initial_state": old.to_dict()}, report)["accepted"]
    assert len(m.receiver_state.columns) == 1
    assert any(
        abs(v.momentum_kg_m_s) > 0
        for n, v in m.receiver_state.transport.nodes
        if n.startswith("supply__")
    )
    other = pickle.loads(pickle.dumps(m))
    m.outlet_openings = {"boundary": 0.0}
    other.outlet_openings = {"boundary": 0.0}
    assert m.step(1e-6) and other.step(1e-6)
    assert m.report == other.report
    assert all(s["flow"]["outlet_component_moles"] == [0.0, 0.0] for s in m.report["stages"])
    raw = {**m.to_dict(), "initial_state": m.receiver_state.to_dict()}
    restored = DynamicReceiverCoupledFIM.from_dict(
        m.solver, json.loads(json.dumps(raw)), device=device
    )
    assert restored.model_id == m.model_id
    if device == "cuda":
        assert m.solver._graphs is not None
    (tmp_path / f"fim-gravity-{device}.json").write_text(
        json.dumps(
            {"model": m.to_dict(), "first_report": report, "second_report": m.report}, indent=2
        )
        + "\n"
    )


def test_failed_resolved_transaction_restores_columns_pipes_and_fim(monkeypatch):
    m = build()
    old = m.receiver_state
    p = m.solver.fluid.pressure.clone()
    monkeypatch.setattr(m, "_temporal_error", lambda *a: 2.0)
    assert not m.step(1e-6)
    assert m.receiver_state is old and m.report is None
    assert torch.equal(m.solver.fluid.pressure, p)
