"""Pipe segmentation must not introduce a vessel or lose a surface work port."""

import json

import numpy as np
import pytest
import torch
from test_surface_equilibrium_thermodynamics import fluid
from test_surface_pr_active_multifluid_pipe import phase
from test_surface_pr_capillary_pressure import chemical_case

from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe
from surface.pr_active_surface import (
    PRActivePipeContinuation,
    PRActiveReservoir,
    PRActiveSurfaceNetwork,
    PRActiveSurfaceState,
)
from surface.pr_capillary_pressure import PRCapillaryPressure


def gas_case(device="cpu", reverse=False):
    f = EquilibriumFluid.from_dict(fluid().to_dict(), device=device)
    lengths = [0.1, 0.15, 0.12, 0.08]
    slots = [{"name": "liquid", "root": "liquid"}, {"name": "gas", "root": "vapor"}]
    whole = PRActiveMultifluidPipe(f, slots, lengths, 0.01, face_elevations_m=[0.0] * 5)
    velocity = -0.1 if reverse else 0.1
    cells = [
        {"gas": phase(f, p, 320.0, "vapor", v, velocity, x=(0.5, 0.5))}
        for p, v in zip((1e6, 1.0001e6, 0.9999e6, 1e6), whole.volumes)
    ]
    boundary = {"gas": phase(f, 1e6, 320.0, "vapor", whole.volumes[0], velocity, x=(0.5, 0.5))}
    pipes = [
        PRActiveMultifluidPipe(f, slots, lengths[i : i + 2], 0.01, face_elevations_m=[0.0] * 3)
        for i in (0, 2)
    ]
    r = PRActiveReservoir(tuple(boundary.values()), ("vapor",))
    graph = PRActiveSurfaceNetwork(
        {
            "a": {"pipe": pipes[0], "boundary_nodes": ("feed", "join")},
            "b": {"pipe": pipes[1], "boundary_nodes": ("join", "export")},
        },
        {},
        [],
        reservoirs={"feed": r, "export": r},
        continuations={"join": PRActivePipeContinuation("a", "b")},
    )
    state = graph.initialize(
        {}, {"a": pipes[0].initialize(cells[:2]), "b": pipes[1].initialize(cells[2:])}
    )
    return graph, state, whole, whole.initialize(cells), boundary


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_bulk_face_matches_unsplit_pipe_and_independent_audit(reverse):
    from check_pr_active_surface import audit

    m, old, p, reference, boundary = gas_case(reverse=reverse)
    dt = 0.1 * m.maximum_step_s
    end, r = m.step(old, dt)
    expected, _ = p.step(reference, dt, boundary_cells=(boundary, boundary))
    cells = tuple(c for _, state in end.pipes for c in state.cells)
    for a, b in zip(cells, expected.cells):
        for (_, u), (_, v) in zip(a, b):
            np.testing.assert_allclose(u.component_moles, v.component_moles, rtol=1e-12, atol=1e-20)
            assert u.momentum_kg_m_s == pytest.approx(v.momentum_kg_m_s, rel=1e-11, abs=1e-18)
            assert u.total_energy_j == pytest.approx(v.total_energy_j, rel=1e-12, abs=1e-14)
    assert r["continuation_fluxes"]["join"]["closed"] is False
    assert audit(m, old, end, r)["accepted"]
    assert end.equipment is None and not end.columns
    restored = PRActiveSurfaceNetwork.from_dict(json.loads(json.dumps(m.to_dict())))
    checkpoint = PRActiveSurfaceState.from_dict(json.loads(json.dumps(end.to_dict())))
    a, ra = m.step(end, dt)
    b, rb = restored.step(checkpoint, dt)
    assert a.to_dict() == b.to_dict() and ra == rb


def capillary_case(device="cpu", loop=False):
    base, old = chemical_case(device)
    raw = base.pipe.to_dict()
    raw["boundary_mode"] = "walls"
    raw["lengths_m"] = list(base.pipe.lengths[:2])
    raw["face_elevations_m"] = [0.0] * 3
    raw["gravity_m_s2"] = 0.0
    raw["hydrostatic_reference"] = None
    a = PRActiveMultifluidPipe.from_dict(raw, device=device)
    b = PRActiveMultifluidPipe.from_dict(raw, device=device)
    r = PRActiveReservoir(tuple(p for _, p in old.cells[0]), ("liquid", "vapor"))
    entries = {
        "a": {"pipe": a, "boundary_nodes": ("return" if loop else "feed", "join")},
        "b": {"pipe": b, "boundary_nodes": ("join", "return" if loop else "export")},
    }
    joins = {"join": PRActivePipeContinuation("a", "b")}
    if loop:
        joins["return"] = PRActivePipeContinuation("b", "a")
    m = PRActiveSurfaceNetwork(
        entries, {}, [], reservoirs={} if loop else {"feed": r, "export": r}, continuations=joins
    )
    st = m.initialize(
        {},
        {
            "a": a.initialize([dict(c) for c in old.cells[:2]]),
            "b": b.initialize([dict(c) for c in old.cells[2:]]),
        },
    )
    parameters = {k: v for k, v in base.to_dict().items() if k not in ("model", "pipe_model_id")}
    return m, st, parameters, base, old


@pytest.mark.parametrize("loop", [False, True])
def test_joint_capillary_work_ports_match_unsplit_geometry_and_newton(loop):
    from check_pr_capillary_pressure import audit

    from surface.pr_capillary_chain import assemble

    m, old, params, _, _ = capillary_case(loop=loop)
    _p, view, source = assemble(m, old, ["a", "b"], params)
    expected, rr = source.apply(view, 0.001)
    end, r = m.relax_capillary_chain(old, 0.001, ["a", "b"], params)
    cells = tuple(c for _, ps in end.pipes for c in ps.cells)
    assert cells == expected.cells
    assert r["source"]["newton_iterations"] == rr["newton_iterations"]
    assert audit(source, view, expected, r["source"])["accepted"]
    for port in r["capillary_ports"].values():
        assert port["work_into_left_j"] == -port["work_into_right_j"]
    assert any(abs(port["work_into_left_j"]) > 0 for port in r["capillary_ports"].values())
    assert max(abs(x) for x in r["segment_energy_residuals_j"].values()) < 1e-18
    before = m.capillary_chain_totals(old, ["a", "b"], params)
    after = m.capillary_chain_totals(end, ["a", "b"], params)
    np.testing.assert_allclose(before["stocks"], after["stocks"], atol=1e-20, rtol=1e-12)
    assert not r["clock_advanced"] and end.time_s == old.time_s
    # Cross-interface second derivative is present in the joint energy.
    a_only = PRCapillaryPressure(m.pipes["a"]["pipe"], **params)
    b_only = PRCapillaryPressure(m.pipes["b"]["pipe"], **params)
    isolated = a_only.surface_energy(dict(old.pipes)["a"]) + b_only.surface_energy(
        dict(old.pipes)["b"]
    )
    assert before["surface_energy_j"] > isolated
    with pytest.raises(ValueError, match="whole continuation component"):
        m.capillary_chain_totals(old, ["a"], params)


def test_isolation_and_invalid_continuation_are_atomic():
    m, old, _, _, _ = gas_case()
    end, r = m.step(old, 0.1 * m.maximum_step_s, port_openings={"join": False})
    assert r["continuation_fluxes"]["join"]["closed"]
    assert r["pipes"]["a"]["boundary_cells"][1] is None
    assert r["pipes"]["b"]["boundary_cells"][0] is None
    raw = m.to_dict()
    raw["continuations"]["join"]["right_pipe"] = "a"
    with pytest.raises(ValueError, match="Continuation"):
        PRActiveSurfaceNetwork.from_dict(raw)
    raw = m.to_dict()
    raw["pipes"]["b"]["pipe"]["diameter_m"] *= 1.01
    with pytest.raises(ValueError, match="matching diameter"):
        PRActiveSurfaceNetwork.from_dict(raw)
    assert end.time_s > old.time_s


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_shared_face_and_capillary_ports_match_cpu():
    values = []
    for device in ("cpu", "cuda"):
        m, st, params, _, _ = capillary_case(device)
        end, r = m.relax_capillary_chain(st, 0.001, ["a", "b"], params)
        values.append((end, r))
    for (_, a), (_, b) in zip(values[0][0].pipes, values[1][0].pipes):
        for ca, cb in zip(a.cells, b.cells):
            for (_, u), (_, v) in zip(ca, cb):
                np.testing.assert_allclose(
                    u.component_moles, v.component_moles, rtol=1e-10, atol=1e-22
                )
                assert u.temperature_k == pytest.approx(v.temperature_k, abs=1e-8)


def test_heat_and_phase_birth_cross_a_continuation_like_an_unsplit_line():
    from check_pr_active_surface import audit
    from test_surface_pr_open_surface import setup

    base, bs = setup(chemical=0.001, evaporation=True)
    p = base.pipes["line"]["pipe"]
    raw = p.to_dict()
    raw["lengths_m"] = [p.lengths[0]]
    raw["face_elevations_m"] = [0.0, 0.0]
    a = PRActiveMultifluidPipe.from_dict(raw)
    b = PRActiveMultifluidPipe.from_dict(raw)
    m = PRActiveSurfaceNetwork(
        {
            "a": {"pipe": a, "boundary_nodes": ("supply", "join")},
            "b": {"pipe": b, "boundary_nodes": ("join", "export")},
        },
        {},
        [],
        reservoirs=base.reservoirs,
        continuations={"join": PRActivePipeContinuation("a", "b")},
    )
    ps = bs.pipes[0][1]
    old = m.initialize(
        {}, {"a": a.initialize([dict(ps.cells[0])]), "b": b.initialize([dict(ps.cells[1])])}
    )
    dt = 0.2 * m.maximum_step_s
    end, r = m.step(
        old,
        dt,
        pipe_heat_w={"a": [{"liquid": 100.0}], "b": [{"liquid": 100.0}]},
        pipe_heat_temperatures_k={"a": [{"liquid": 400.0}], "b": [{"liquid": 400.0}]},
    )
    expected, _reference = base.step(
        bs,
        dt,
        pipe_heat_w={"line": [{"liquid": 100.0}] * 2},
        pipe_heat_temperatures_k={"line": [{"liquid": 400.0}] * 2},
    )
    cells = tuple(c for _, state in end.pipes for c in state.cells)
    for actual, target in zip(cells, expected.pipes[0][1].cells):
        assert len(actual) == 2
        for (_, u), (_, v) in zip(actual, target):
            np.testing.assert_allclose(u.component_moles, v.component_moles, rtol=1e-11, atol=1e-20)
            assert u.temperature_k == pytest.approx(v.temperature_k, abs=1e-8)
    assert audit(m, old, end, r)["accepted"]


def test_closed_bulk_loop_matches_periodic_reference_without_external_stocks():
    from check_pr_active_surface import audit

    m, st, p, ps, _ = gas_case()
    entries = {
        "a": {**m.pipes["a"], "boundary_nodes": ("return", "join")},
        "b": {**m.pipes["b"], "boundary_nodes": ("join", "return")},
    }
    loop = PRActiveSurfaceNetwork(
        entries,
        {},
        [],
        continuations={
            "join": PRActivePipeContinuation("a", "b"),
            "return": PRActivePipeContinuation("b", "a"),
        },
    )
    old = loop.initialize({}, dict(st.pipes))
    raw = p.to_dict()
    raw["boundary_mode"] = "periodic"
    periodic = PRActiveMultifluidPipe.from_dict(raw)
    seed = periodic.initialize([dict(c) for c in ps.cells])
    dt = 0.1 * loop.maximum_step_s
    end, r = loop.step(old, dt)
    target, _ = periodic.step(seed, dt)
    assert not r["reservoir_exchange"]
    assert all(x == 0 for x in r["external_component_moles"])
    assert r["external_energy_j"] == 0
    assert audit(loop, old, end, r)["accepted"]
    for a, b in zip(tuple(c for _, pstate in end.pipes for c in pstate.cells), target.cells):
        np.testing.assert_allclose(
            a[0][1].component_moles, b[0][1].component_moles, rtol=1e-12, atol=1e-20
        )
        assert a[0][1].total_energy_j == pytest.approx(b[0][1].total_energy_j, rel=1e-12)


def test_joint_surface_solve_changes_both_sides_of_the_shared_edge():
    m, old, params, _, _ = capillary_case()
    end, _ = m.relax_capillary_chain(old, 0.001, ["a", "b"], params)
    isolated = {}
    for name in ("a", "b"):
        source = PRCapillaryPressure(m.pipes[name]["pipe"], **params)
        isolated[name], _ = source.apply(dict(old.pipes)[name], 0.001)
    differences = []
    for name in ("a", "b"):
        for a, b in zip(dict(end.pipes)[name].cells, isolated[name].cells):
            differences.append(abs(dict(a)["liquid"].volume_m3 - dict(b)["liquid"].volume_m3))
    assert max(differences) > 1e-25


def test_missing_chemical_surface_support_does_not_create_a_residual_phase():
    m, old, _, _, _ = gas_case()
    params = {
        "liquid_slot": "liquid",
        "gas_slot": "gas",
        "width_m": 0.001,
        "surface_tension_n_m": 0.02,
        "volume_mobility_m3_pa_s": 1e-15,
    }
    snapshot = old.to_dict()
    with pytest.raises(ValueError, match="both declared phases"):
        m.relax_capillary_chain(old, 0.001, ["a", "b"], params)
    assert old.to_dict() == snapshot


@pytest.mark.parametrize("loop", [False, True])
def test_independent_chain_audit_rejects_corrupt_work_and_inventory(loop):
    import copy

    from check_pr_capillary_chain import audit

    m, old, params, _, _ = capillary_case(loop=loop)
    end, report = m.relax_capillary_chain(old, 0.001, ["a", "b"], params)
    assert audit(m, old, end, report, params)["accepted"]
    bad = copy.deepcopy(report)
    bad["capillary_ports"]["join"]["work_into_left_j"] *= -1
    with pytest.raises(ValueError, match="left port work"):
        audit(m, old, end, bad, params)
    bad = copy.deepcopy(report)
    bad["capillary_ports"].pop("join")
    with pytest.raises(ValueError, match="Missing or extra"):
        audit(m, old, end, bad, params)


def test_closed_capillary_join_rejects_instead_of_transferring_work():
    m, old, params, _, _ = capillary_case()
    snapshot = old.to_dict()
    for command, message in (
        ({"join": False}, "Closed continuation"),
        ({"join": 0}, "boolean"),
        ({"absent": True}, "Unknown"),
    ):
        with pytest.raises(ValueError, match=message):
            m.relax_capillary_chain(old, 0.001, ["a", "b"], params, port_openings=command)
    assert old.to_dict() == snapshot


def test_single_pipe_closed_continuation_and_source_restart():
    import json

    from check_pr_capillary_chain import audit

    from surface.pr_active_surface import PRActiveSurfaceState

    m, old, params, _, _ = capillary_case(loop=True)
    from surface.pr_capillary_chain import assemble

    pipe, view, _ = assemble(m, old, ["a", "b"], params)
    raw = pipe.to_dict()
    raw["boundary_mode"] = "walls"
    pipe = PRActiveMultifluidPipe.from_dict(raw)
    view = pipe.initialize([dict(c) for c in view.cells])
    one = PRActiveSurfaceNetwork(
        {"line": {"pipe": pipe, "boundary_nodes": ("join", "join")}},
        {},
        [],
        continuations={"join": PRActivePipeContinuation("line", "line")},
    )
    seed = one.initialize({}, {"line": view})
    restored = PRActiveSurfaceNetwork.from_dict(json.loads(json.dumps(one.to_dict())))
    restart = PRActiveSurfaceState.from_dict(json.loads(json.dumps(seed.to_dict())))
    end, report = restored.relax_capillary_chain(restart, 0.001, ["line"], params)
    assert audit(restored, restart, end, report, params)["shared_edges"] == 1
    expected, _ = one.relax_capillary_chain(seed, 0.001, ["line"], params)
    assert end == expected
