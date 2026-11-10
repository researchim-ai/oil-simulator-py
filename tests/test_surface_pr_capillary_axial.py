"""Actual jointly advanced PR volumes, impulses, calorics and capillary energy."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from check_pr_capillary_axial import audit
from test_surface_pr_capillary_pressure import chemical_case

from surface.pr_active_multifluid_pipe import PRActiveMultifluidState
from surface.pr_capillary_axial import PRCapillaryAxialPipe
from surface.pr_capillary_pressure import PRCapillaryPressure


def setup(device="cpu", sigma=None):
    source, old = chemical_case(device)
    if sigma is not None:
        params = {k: v for k, v in source.to_dict().items() if k not in ("model", "pipe_model_id")}
        params["surface_tension_n_m"] = sigma
        source = PRCapillaryPressure(source.pipe, **params)
    return PRCapillaryAxialPipe(source), old


def test_actual_joint_step_passes_independent_scalar_equations():
    model, old = setup()
    new, r = model.step(old, 1e-11)
    assert new.time_s == 1e-11
    assert new.cells != old.cells
    assert r["newton_iterations"] > 0
    assert r["entropy_production_j_k"] >= 0
    assert all(x >= -1e-30 for x in r["capillary_diffusion_heat_j"])
    assert any(x != 0 for row in r["capillary_force_n"] for x in row)
    assert audit(model, old, new, r)["accepted"]
    assert not r["normal_sources_applied"]
    assert max(abs(x) for x in r["conservation_residuals"]) < 1e-18
    corrupted = copy.deepcopy(r)
    corrupted["capillary_force_n"][0][0] += 1e-10
    with pytest.raises(ValueError, match="adjoint force"):
        audit(model, old, new, corrupted)


def test_surface_tension_changes_actual_phase_momentum():
    a, old = setup(sigma=0.02)
    b, _ = setup(sigma=0.04)
    x, _ = a.step(old, 1e-11)
    y, _ = b.step(old, 1e-11)
    changes = [
        abs(dict(c)["liquid"].momentum_kg_m_s - dict(d)["liquid"].momentum_kg_m_s)
        for c, d in zip(x.cells, y.cells)
    ]
    assert max(changes) > 1e-23


def test_consistent_transport_removes_the_old_false_entropy_failure_and_covers_time():
    model, old = setup()
    checkpoint = old.to_dict()
    single, record = model.step(old, 1e-10)
    assert audit(model, old, single, record)["accepted"]
    end, report = model.advance(old, 1e-10, initial_step_s=1e-10)
    assert end.time_s == 1e-10
    assert report["steps"]
    last = old
    for step in report["steps"]:
        current = PRActiveMultifluidState.from_dict(step["state"])
        assert audit(model, last, current, step)["accepted"]
        last = current
    assert old.to_dict() == checkpoint


def test_restart_preserves_the_next_joint_step():
    model, old = setup()
    nextstate, _ = model.step(old, 1e-11)
    restarted = PRActiveMultifluidState.from_dict(json.loads(json.dumps(nextstate.to_dict())))
    a, ra = model.step(nextstate, 1e-11)
    restored_model = PRCapillaryAxialPipe.from_dict(json.loads(json.dumps(model.to_dict())))
    b, rb = restored_model.step(restarted, 1e-11)
    assert a == b and ra == rb
    with pytest.raises(ValueError, match="unknown budget"):
        PRCapillaryAxialPipe(model.source, maximum_unknowns=1).step(old, 1e-11)
    with pytest.raises(ValueError, match="clock resolution"):
        model.step(replace(old, time_s=1e20), 1e-11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("boundary_mode", ["periodic", "walls"])
def test_cuda_joint_pr_capillary_step_matches_cpu(boundary_mode):
    states = []
    for device in ("cpu", "cuda"):
        model, old = (setup if boundary_mode == "periodic" else reflecting_case)(device)
        end, r = model.step(old, 1e-11)
        assert audit(model, old, end, r)["accepted"]
        states.append(end)
    for ca, cb in zip(states[0].cells, states[1].cells):
        for (_, a), (_, b) in zip(ca, cb):
            np.testing.assert_allclose(a.component_moles, b.component_moles, rtol=1e-10, atol=1e-22)
            assert a.temperature_k == pytest.approx(b.temperature_k, abs=1e-8)
            assert a.momentum_kg_m_s == pytest.approx(b.momentum_kg_m_s, rel=1e-10, abs=1e-22)


def test_complete_loop_advances_actual_segment_stocks_with_normal_exchange():
    from check_pr_capillary_pressure import audit as normal_audit
    from test_surface_pr_continuation import capillary_case

    from surface.pr_capillary_chain import assemble

    graph, old, params, _, _ = capillary_case(loop=True)
    _, view, source = assemble(graph, old, ["a", "b"], params)
    whole = PRCapillaryAxialPipe(source)
    expected, _ = whole.advance(view, 1e-11, initial_step_s=1e-11, normal_relaxation=True)
    end, report = graph.advance_capillary_loop(old, 1e-11, ["a", "b"], params, initial_step_s=1e-11)
    carry = replace(old, next_step_s=5e-12)
    _, carried = graph.advance_capillary_loop(carry, 1e-11, ["a", "b"], params)
    assert carried["advance"]["steps"][0]["dt_s"] == 5e-12
    actual = tuple(c for name in ("a", "b") for c in dict(end.pipes)[name].cells)
    assert actual == expected.cells
    assert end.time_s == 1e-11
    assert all(state.time_s == end.time_s for _, state in end.pipes)
    previous = view
    for record in report["advance"]["steps"]:
        axial = PRActiveMultifluidState.from_dict(record["axial"]["state"])
        final = PRActiveMultifluidState.from_dict(record["normal"]["state"])
        assert audit(whole, previous, axial, record["axial"])["accepted"]
        assert normal_audit(source, axial, final, record["normal"])["accepted"]
        previous = final
    restored = type(graph).from_dict(json.loads(json.dumps(graph.to_dict())))
    checkpoint = type(old).from_dict(json.loads(json.dumps(end.to_dict())))
    a, _ = graph.advance_capillary_loop(end, 1e-11, ["a", "b"], params, initial_step_s=1e-11)
    b, _ = restored.advance_capillary_loop(
        checkpoint, 1e-11, ["a", "b"], params, initial_step_s=1e-11
    )
    assert a == b
    open_graph, open_state, _, _, _ = capillary_case(loop=False)
    with pytest.raises(ValueError, match="complete isolated graph"):
        open_graph.advance_capillary_loop(open_state, 1e-11, ["a", "b"], params)


def reflecting_case(device="cpu"):
    from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe

    model, old = setup(device)
    raw = model.pipe.to_dict()
    raw["boundary_mode"] = "walls"
    pipe = PRActiveMultifluidPipe.from_dict(raw, device=device)
    params = {
        k: v for k, v in model.source.to_dict().items() if k not in ("model", "pipe_model_id")
    }
    source = PRCapillaryPressure(pipe, **params)
    return PRCapillaryAxialPipe(source), pipe.initialize([dict(c) for c in old.cells])


def test_reflecting_ends_account_for_wall_impulse_and_do_not_cycle_inventory():
    model, old = reflecting_case()
    new, r = model.step(old, 1e-11)
    assert audit(model, old, new, r)["accepted"]
    assert r["wall_impulse_kg_m_s"] != 0
    assert r["entropy_production_j_k"] > 0
    assert abs(sum(sum(row) for row in r["capillary_force_n"])) < 1e-20
    assert abs(r["conservation_residuals"][-1]) < 1e-22
    corrupted = copy.deepcopy(r)
    corrupted["wall_impulse_kg_m_s"] *= 2
    with pytest.raises(ValueError, match="wall impulse"):
        audit(model, old, new, corrupted)


def test_isolated_graph_line_matches_unsplit_reflecting_pipe():
    from check_pr_capillary_pressure import audit as normal_audit
    from test_surface_pr_continuation import capillary_case

    from surface.pr_active_surface import PRActivePipeContinuation, PRActiveSurfaceNetwork
    from surface.pr_capillary_chain import assemble

    ring, old, params, _, _ = capillary_case(loop=True)
    line = PRActiveSurfaceNetwork(
        {
            "a": {**ring.pipes["a"], "boundary_nodes": (None, "join")},
            "b": {**ring.pipes["b"], "boundary_nodes": ("join", None)},
        },
        {},
        [],
        continuations={"join": PRActivePipeContinuation("a", "b")},
    )
    initial = line.initialize({}, dict(old.pipes))
    _, view, source = assemble(line, initial, ["a", "b"], params)
    whole = PRCapillaryAxialPipe(source)
    expected, _ = whole.advance(view, 1e-11, initial_step_s=1e-11, normal_relaxation=True)
    final, report = line.advance_capillary_line(
        initial, 1e-11, ["a", "b"], params, initial_step_s=1e-11
    )
    assert tuple(c for _, st in final.pipes for c in st.cells) == expected.cells
    previous = view
    for step in report["advance"]["steps"]:
        axial = PRActiveMultifluidState.from_dict(step["axial"]["state"])
        end = PRActiveMultifluidState.from_dict(step["normal"]["state"])
        assert audit(whole, previous, axial, step["axial"])["accepted"]
        assert normal_audit(source, axial, end, step["normal"])["accepted"]
        previous = end
    with pytest.raises(ValueError, match="actual closing"):
        line.advance_capillary_loop(initial, 1e-11, ["a", "b"], params)


def test_programming_error_in_jacobian_is_not_hidden_by_adaptive_retries(monkeypatch):
    model, old = setup()
    calls = []

    def broken_jacobian(function):
        calls.append(function)
        raise RuntimeError("intentional differentiation bug")

    monkeypatch.setattr(torch.func, "jacfwd", broken_jacobian)
    with pytest.raises(RuntimeError, match="intentional differentiation bug"):
        model.advance(old, 1e-11, initial_step_s=1e-11)
    assert len(calls) == 1
