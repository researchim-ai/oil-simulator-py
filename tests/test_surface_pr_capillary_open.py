"""Maintained-face PR flow with physical face geometry and explicit work."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from check_pr_capillary_axial import audit
from check_pr_capillary_pressure import audit as normal_audit
from test_surface_pr_capillary_axial import reflecting_case

from surface.pr_active_multifluid_pipe import PRActiveMultifluidState
from surface.pr_capillary_axial import PRCapillaryAxialPipe


def setup(device="cpu", reverse=False, side=None):
    base, old = reflecting_case(device)
    if reverse:
        old = replace(
            old,
            cells=tuple(
                tuple((name, replace(p, momentum_kg_m_s=-p.momentum_kg_m_s)) for name, p in c)
                for c in old.cells
            ),
        )
    rows = [dict(old.cells[0]), dict(old.cells[-1])]
    if side is not None:
        rows[1 - side] = None
    return PRCapillaryAxialPipe(base.source, boundaries=rows), old


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("side", [None, 0, 1])
def test_actual_open_step_includes_boundary_components_energy_momentum_and_surface_work(
    reverse, side
):
    model, old = setup(reverse=reverse, side=side)
    new, r = model.step(old, 1e-12)
    assert audit(model, old, new, r)["accepted"]
    assert any(x != 0 for x in r["external_transfer_n_m_e"])
    assert r["external_capillary_work_j"] != 0
    assert r["external_capillary_impulse_kg_m_s"] != 0
    assert r["wall_impulse_kg_m_s"] == 0 if side is None else r["wall_impulse_kg_m_s"] != 0
    final, nr = model.source.apply(new, 1e-12)
    assert normal_audit(model.source, new, final, nr)["accepted"]
    corrupted = copy.deepcopy(r)
    corrupted["external_capillary_work_j"] += 1e-15
    with pytest.raises(ValueError, match="external capillary work"):
        audit(model, old, new, corrupted)


def test_open_restart_and_fixed_face_geometry_roundtrip():
    model, old = setup()
    restored = PRCapillaryAxialPipe.from_dict(json.loads(json.dumps(model.to_dict())))
    a, ra = model.step(old, 1e-12)
    b, rb = restored.step(old, 1e-12)
    assert a == b and ra == rb
    assert all(h is not None for h in restored.source.boundary_heights)
    with pytest.raises(ValueError, match="requires maintained"):
        PRCapillaryAxialPipe(model.source)


def test_segmented_open_line_advances_actual_graph_stocks_with_audited_boundary_ports():
    from test_surface_pr_continuation import capillary_case

    from surface.pr_capillary_chain import assemble

    graph, old, params, _, _ = capillary_case(loop=False)
    _, view, source = assemble(graph, old, ["a", "b"], params)
    boundaries = []
    for pipe_name, node in (("a", "feed"), ("b", "export")):
        reservoir = graph.reservoirs[node]
        boundaries.append(
            graph._assign(
                graph.pipes[pipe_name]["pipe"], (), reservoir.phases, reservoir.phase_roots
            )
        )
    whole = PRCapillaryAxialPipe(source, boundaries=boundaries)
    expected, rr = whole.advance(view, 1e-12, initial_step_s=1e-12, normal_relaxation=True)
    final, report = graph.advance_capillary_open_line(
        old, 1e-12, ["a", "b"], params, initial_step_s=1e-12
    )
    assert tuple(c for _, st in final.pipes for c in st.cells) == expected.cells
    previous = view
    for record in report["advance"]["steps"]:
        axial = PRActiveMultifluidState.from_dict(record["axial"]["state"])
        end = PRActiveMultifluidState.from_dict(record["normal"]["state"])
        assert audit(whole, previous, axial, record["axial"])["accepted"]
        assert normal_audit(whole.source, axial, end, record["normal"])["accepted"]
        previous = end
    assert report["advance"] == rr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_open_cuda_states_match_cpu():
    states = []
    for device in ("cpu", "cuda"):
        model, old = setup(device)
        end, r = model.step(old, 1e-12)
        assert audit(model, old, end, r)["accepted"]
        states.append(end)
    for ca, cb in zip(states[0].cells, states[1].cells):
        for (_, a), (_, b) in zip(ca, cb):
            np.testing.assert_allclose(a.component_moles, b.component_moles, rtol=1e-10, atol=1e-22)
            assert a.temperature_k == pytest.approx(b.temperature_k, abs=1e-8)
