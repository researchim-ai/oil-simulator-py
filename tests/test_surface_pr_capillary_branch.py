"""Actual finite junction stocks: material and calorics move into a chamber."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from check_pr_capillary_pressure import audit as normal_audit
from test_surface_pr_capillary_axial import reflecting_case

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
from surface.pr_capillary_branch import PRCapillaryBranchNetwork, PRCapillaryBranchState


def setup(device="cpu"):
    model, old = reflecting_case(device)
    raw = model.pipe.to_dict()
    raw["lengths_m"] = list(model.pipe.lengths[:2])
    pipe = PRActiveMultifluidPipe.from_dict(raw, device=device)
    axes = {"a": (1.0, 0.0, 0.0), "b": (0.0, 1.0, 0.0), "c": (-1.0, 0.0, 0.0)}
    pipes = {
        name: {"pipe": pipe, "boundary_nodes": (None, "J"), "axis": axis}
        for name, axis in axes.items()
    }
    params = {
        k: v for k, v in model.source.to_dict().items() if k not in ("model", "pipe_model_id")
    }
    network = PRCapillaryBranchNetwork(pipes, {"J": pipe.volumes[0] * 1.5}, params)
    chamber = {}
    for name, p in old.cells[0]:
        mass = sum(
            n * c.molar_mass_kg_mol for n, c in zip(p.component_moles, pipe.fluid.components)
        )
        chamber[name] = replace(
            p,
            component_moles=tuple(v * 1.5 for v in p.component_moles),
            volume_m3=p.volume_m3 * 1.5,
            momentum_kg_m_s=0.0,
            total_energy_j=1.5 * (p.total_energy_j - p.momentum_kg_m_s**2 / (2 * mass)),
        )
    state = network.initialize(
        {n: pipe.initialize([dict(c) for c in old.cells[:2]]) for n in pipes}, {"J": chamber}
    )
    return network, state


def test_real_branch_inventory_energy_and_stationary_chamber_reactions():
    model, old = setup()
    end, r = model.advance(old, 1e-12, initial_step_s=1e-12)
    assert end.time_s == 1e-12
    assert end.junctions != old.junctions
    assert all(p.momentum_kg_m_s == 0 for _, phases in end.junctions for _, p in phases)
    step = r["advance"]["steps"][0]
    axial = PRActiveMultifluidState.from_dict(step["axial"]["state"])
    final = PRActiveMultifluidState.from_dict(step["normal"]["state"])
    from check_pr_capillary_branch import audit

    assert audit(model, model._view(old), axial, step["axial"])["accepted"]
    assert normal_audit(model.source, axial, final, step["normal"])["accepted"]
    assert len(step["axial"]["conservation_residuals"]) == len(model.fluid.names) + 1
    assert (
        max(abs(v) for v in step["axial"]["momentum_balance"]["vector_momentum_residual_kg_m_s"])
        < 1e-22
    )
    assert any(
        abs(v) > 0
        for v in step["axial"]["momentum_balance"]["junction_support_impulses_kg_m_s"]["J"]
    )
    assert step["axial"]["entropy_production_j_k"] > 0
    a = model._view(old)
    b = model._view(end)
    initial = model.source.pipe._totals(a)
    finalstock = model.source.pipe._totals(b)
    initial[-2] += model.source.surface_energy(a)
    finalstock[-2] += model.source.surface_energy(b)
    assert initial[:-1] == pytest.approx(finalstock[:-1], rel=1e-10, abs=1e-22)


def test_finite_branch_model_and_checkpoint_restart():
    model, old = setup()
    restored = PRCapillaryBranchNetwork.from_dict(json.loads(json.dumps(model.to_dict())))
    checkpoint = PRCapillaryBranchState.from_dict(json.loads(json.dumps(old.to_dict())))
    a, ra = model.advance(old, 1e-12, initial_step_s=1e-12)
    b, rb = restored.advance(checkpoint, 1e-12, initial_step_s=1e-12)
    assert a == b and ra == rb
    assert model._view(old).cells == restored._view(checkpoint).cells


@pytest.mark.parametrize(
    "field", ["junction_support_impulses_kg_m_s", "surface_force_resultant_impulse_kg_m_s"]
)
def test_independent_audit_rejects_wrong_support_and_surface_resultant(field):
    from check_pr_capillary_branch import audit

    model, old = setup()
    _, r = model.advance(old, 1e-12, initial_step_s=1e-12)
    step = r["advance"]["steps"][0]["axial"]
    final = PRActiveMultifluidState.from_dict(step["state"])
    bad = copy.deepcopy(step)
    if field.startswith("junction"):
        bad["momentum_balance"][field]["J"][0] += 1e-15
    else:
        bad["momentum_balance"][field][0] += 1e-15
    with pytest.raises(ValueError):
        audit(model, model._view(old), final, bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_finite_chamber_graph_matches_cpu_and_independent_audits():
    from check_pr_capillary_branch import audit

    states = []
    for device in ("cpu", "cuda"):
        model, old = setup(device)
        end, r = model.advance(old, 1e-12, initial_step_s=1e-12)
        record = r["advance"]["steps"][0]
        axial = PRActiveMultifluidState.from_dict(record["axial"]["state"])
        final = PRActiveMultifluidState.from_dict(record["normal"]["state"])
        assert audit(model, model._view(old), axial, record["axial"])["accepted"]
        assert normal_audit(model.source, axial, final, record["normal"])["accepted"]
        states.append(model._view(end))
    for ca, cb in zip(states[0].cells, states[1].cells):
        for (_, a), (_, b) in zip(ca, cb):
            np.testing.assert_allclose(a.component_moles, b.component_moles, rtol=1e-10, atol=1e-22)
            assert a.temperature_k == pytest.approx(b.temperature_k, abs=1e-8)


def test_open_finite_junction_inlet_withdrawal_and_heater_are_in_actual_balances():
    from check_pr_capillary_branch import audit
    from test_surface_pr_active_multifluid_pipe import phase

    base, old = setup()
    f = base.fluid
    inlet = phase(f, 1.1e6, 200.0, "liquid", 1e-12, u=0.0, x=(0.05, 0.95))
    sources = {
        "J": {
            "inlets": [{"phase": "liquid", "molar_rate_mol_s": 0.001, "state": inlet}],
            "withdrawals": [{"phase": "gas", "molar_rate_mol_s": 0.0001}],
            "heaters": [{"phase": "liquid", "power_w": 0.001, "temperature_k": 400.0}],
        }
    }
    model = PRCapillaryBranchNetwork(
        base.pipes, base.junction_volumes, base.parameters, sources=sources
    )
    initial = model.initialize(
        dict(old.pipes), {name: dict(phases) for name, phases in old.junctions}
    )
    end, report = model.advance(initial, 1e-12, initial_step_s=1e-12)
    record = report["advance"]["steps"][0]
    axial = PRActiveMultifluidState.from_dict(record["axial"]["state"])
    assert audit(model, model._view(initial), axial, record["axial"])["accepted"]
    assert any(v != 0 for v in record["axial"]["external_transfer_n_m_e"])
    assert record["axial"]["incoming_entropy_j_k"] != 0
    restored = PRCapillaryBranchNetwork.from_dict(json.loads(json.dumps(model.to_dict())))
    state = PRCapillaryBranchState.from_dict(json.loads(json.dumps(initial.to_dict())))
    other, _ = restored.advance(state, 1e-12, initial_step_s=1e-12)
    assert other == end


def test_binary_isolation_accounts_for_surface_work_and_retains_all_stocks():
    from check_pr_capillary_branch import audit, audit_switch

    model, old = setup()
    isolated, state, event = model.switch_ports(old, {"a:1": False})
    assert audit_switch(model, old, isolated, state, event)["accepted"]
    assert event["external_actuator_work_j"] < 0
    assert old.pipes == state.pipes and old.junctions == state.junctions
    assert state.time_s == old.time_s
    restored = PRCapillaryBranchNetwork.from_dict(json.loads(json.dumps(isolated.to_dict())))
    checkpoint = PRCapillaryBranchState.from_dict(json.loads(json.dumps(state.to_dict())))
    _end, r = restored.advance(checkpoint, 1e-12, initial_step_s=1e-12)
    first = r["advance"]["steps"][0]["axial"]
    view = PRActiveMultifluidState.from_dict(first["state"])
    assert audit(restored, restored._view(checkpoint), view, first)["accepted"]
    reopened, same, back = isolated.switch_ports(state, {"a:1": True})
    assert audit_switch(isolated, state, reopened, same, back)["accepted"]
    assert back["external_actuator_work_j"] == -event["external_actuator_work_j"]
    assert reopened.model_id == model.model_id
    damaged = copy.deepcopy(event)
    damaged["external_actuator_work_j"] = 0.0
    with pytest.raises(ValueError, match="actuator work"):
        audit_switch(model, old, isolated, state, damaged)


def recycle_case(device="cpu"):
    """Two finite chambers, parallel return paths and a blind branch."""
    base, old = setup(device)
    p = base.pipes["a"]["pipe"]
    entries = {
        "forward": {"pipe": p, "boundary_nodes": ("J", "K"), "axis": (1.0, 0.0, 0.0)},
        "return": {"pipe": p, "boundary_nodes": ("K", "J"), "axis": (-1.0, 0.0, 0.0)},
        "blind": {"pipe": p, "boundary_nodes": ("J", None), "axis": (0.0, 1.0, 0.0)},
    }
    model = PRCapillaryBranchNetwork(entries, {"J": p.volumes[0] * 1.5, "K": p.volumes[0] * 1.5}, base.parameters)
    initial = model.initialize(
        {n: dict(old.pipes)["a"] for n in entries},
        {n: dict(dict(old.junctions)["J"]) for n in ("J", "K")},
    )
    return model, initial


def test_recycle_parallel_paths_isolation_and_restart_are_audited_end_to_end():
    from run_pr_capillary_branch import run

    model, old = recycle_case()
    period = {"duration_s": 1e-12, "initial_step_s": 1e-12, "normal_relaxation": True}
    request = {
        "model": model.to_dict(), "initial": old.to_dict(),
        "schedule": [period, {**period, "openings": {"return:0": False}},
                     {**period, "openings": {"return:0": True}}],
    }
    result = run(json.loads(json.dumps(request)))
    assert result["final"]["time_s"] == 3e-12
    assert all(v["axial"]["accepted"] and v["normal"]["accepted"]
               for interval in result["intervals"] for v in interval["independent_audits"])
    assert all(p["isolation_audit"]["accepted"] for p in result["intervals"][1:])
    restart = {
        **request, "model": result["intervals"][1]["model"],
        "initial": result["intervals"][1]["result"]["state"],
        "schedule": [request["schedule"][2]],
    }
    assert run(restart)["final"] == result["final"]
