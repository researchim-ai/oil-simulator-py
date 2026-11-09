"""Actual flowing PR gas, open graph, heat and phase conversion together."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_surface_equilibrium_thermodynamics import fluid
from test_surface_pr_active_multifluid_pipe import phase

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe
from surface.pr_active_surface import (
    PRActiveReservoir,
    PRActiveSurfaceNetwork,
    PRActiveSurfaceState,
)


def setup(*, device="cpu", chemical=None, evaporation=False, reverse=False):
    from surface.equilibrium_properties import EquilibriumFluid

    f = EquilibriumFluid.from_dict(fluid().to_dict(), device=device)
    pipe = PRActiveMultifluidPipe(
        f,
        [{"name": "liquid", "root": "liquid"}, {"name": "gas", "root": "vapor"}],
        [0.1] * 2,
        0.01,
        face_elevations_m=[0.0, 0.0, 0.0],
        thermochemical_relaxation_time_s=chemical,
    )
    if evaporation:
        p = f.components[1].saturation_at_temperature(250.0)["pressure_pa"] * 0.9
        T = 250.0
        root = "liquid"
        name = "liquid"
        x = (0.0, 1.0)
    else:
        p = 1e6
        T = 320.0
        root = "vapor"
        name = "gas"
        x = (0.5, 0.5)
    speed = -0.1 if reverse else 0.1
    cell = {name: phase(f, p, T, root, pipe.volumes[0], speed, x)}
    reservoir = PRActiveReservoir(tuple(cell.values()), (root,))
    model = PRActiveSurfaceNetwork(
        {"line": {"pipe": pipe, "boundary_nodes": ("supply", "export")}},
        {},
        [],
        reservoirs={"supply": reservoir, "export": reservoir},
    )
    return model, model.initialize({}, {"line": pipe.initialize([cell, cell])})


@pytest.mark.parametrize("reverse", [False, True])
def test_uniform_actual_gas_throughflow_is_stationary_with_opposite_boundary_transfers(reverse):
    from check_pr_active_surface import audit

    m, st = setup(reverse=reverse)
    end, r = m.step(st, 0.2 * m.maximum_step_s)
    assert end.pipes[0][1].cells == st.pipes[0][1].cells
    assert not end.columns and end.equipment is None
    assert audit(m, st, end, r)["accepted"]
    sign = -1 if reverse else 1
    assert sign * sum(r["reservoir_exchange"]["supply"][:-1]) > 0
    assert sign * sum(r["reservoir_exchange"]["export"][:-1]) < 0
    assert max(abs(x) for x in r["conservation_residuals"]) < 1e-10


def test_open_heated_evaporation_changes_phase_inventory_and_temperature_and_audits():
    from check_pr_active_surface import audit

    m, st = setup(chemical=0.001, evaporation=True)
    dt = 0.2 * m.maximum_step_s
    end, r = m.step(
        st,
        dt,
        pipe_heat_w={"line": [{"liquid": 100.0}, {"liquid": 100.0}]},
        pipe_heat_temperatures_k={"line": [{"liquid": 400.0}, {"liquid": 400.0}]},
    )
    assert all(len(cell) == 2 for cell in dict(end.pipes)["line"].cells)
    assert any(
        abs(dict(cell)["gas"].temperature_k - 250.0) > 1e-5
        for cell in dict(end.pipes)["line"].cells
    )
    assert r["pipes"]["line"]["sources"]
    assert audit(m, st, end, r)["accepted"]
    assert sum(r["reservoir_exchange"]["supply"][:-1]) > 0
    assert r["external_energy_j"] != 0
    restored = PRActiveSurfaceNetwork.from_dict(json.loads(json.dumps(m.to_dict())))
    checkpoint = PRActiveSurfaceState.from_dict(json.loads(json.dumps(end.to_dict())))
    restored.inspect(checkpoint)
    a, ra = m.step(end, dt)
    b, rb = restored.step(checkpoint, dt)
    assert a.to_dict() == b.to_dict()
    assert ra == rb


def test_isolation_interval_and_bad_reservoir_are_atomic():
    m, st = setup()
    r = m.advance(st, 2.1 * m.maximum_step_s, port_openings={"supply": False, "export": False})
    assert r["state"]["time_s"] == 2.1 * m.maximum_step_s
    assert all(not any(v) for s in r["steps"] for v in s["reservoir_exchange"].values())
    bad = replace(m.reservoirs["supply"], elevation_m=1.0)
    with pytest.raises(ValueError, match="elevations"):
        PRActiveSurfaceNetwork(
            m.pipes, {}, [], reservoirs={"supply": bad, "export": m.reservoirs["export"]}
        )
    snapshot = st.to_dict()
    with pytest.raises(ValueError, match="isolation"):
        m.step(st, 0.1 * m.maximum_step_s, port_openings={"typo": False})
    assert st.to_dict() == snapshot


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_open_gas_heat_phase_exchange_matches_cpu():
    ends = []
    for device in ("cpu", "cuda"):
        m, st = setup(device=device, chemical=0.001, evaporation=True)
        end, _ = m.step(
            st,
            0.2 * m.maximum_step_s,
            pipe_heat_w={"line": [{"liquid": 100.0}, {"liquid": 100.0}]},
            pipe_heat_temperatures_k={"line": [{"liquid": 400.0}, {"liquid": 400.0}]},
        )
        ends.append(end)
    for ca, cb in zip(ends[0].pipes[0][1].cells, ends[1].pipes[0][1].cells):
        assert [n for n, _ in ca] == [n for n, _ in cb]
        for (_, a), (_, b) in zip(ca, cb):
            np.testing.assert_allclose(a.component_moles, b.component_moles, rtol=1e-9, atol=1e-18)
            assert a.temperature_k == pytest.approx(b.temperature_k, abs=1e-7)


def test_three_branch_graph_updates_real_finite_junction_without_fake_column():
    from check_pr_active_surface import audit

    from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork

    base, st = setup()
    pipe = base.pipes["line"]["pipe"]
    graph = PREquilibriumTransportNetwork(pipe.fluid, [PREquilibriumNode("junction", 0.001)])
    gs = graph.initialize(
        {"junction": {"pressure_pa": 1e6, "temperature_k": 320.0, "mole_fractions": [0.5, 0.5]}}
    )
    entries = {
        "in": {"pipe": pipe, "boundary_nodes": ("supply", "junction")},
        "out_a": {"pipe": pipe, "boundary_nodes": ("junction", "export")},
        "out_b": {"pipe": pipe, "boundary_nodes": ("junction", "export")},
    }
    m = PRActiveSurfaceNetwork(entries, {}, [], equipment=graph, reservoirs=base.reservoirs)
    old = m.initialize({}, {n: st.pipes[0][1] for n in entries}, gs)
    new, r = m.step(old, 0.05 * m.maximum_step_s)
    assert new.equipment.nodes != old.equipment.nodes
    assert not new.columns
    assert audit(m, old, new, r)["accepted"]
    assert max(abs(v) for v in r["conservation_residuals"]) < 1e-7


def test_unbalanced_boundary_exchange_changes_stocks_and_auditor_detects_tampering():
    from copy import deepcopy

    from check_pr_active_surface import audit

    m, old = setup()
    new, r = m.step(old, 0.2 * m.maximum_step_s, port_openings={"export": False})
    assert sum(r["external_component_moles"]) > 0
    before = m.totals(old)
    after = m.totals(new)
    np.testing.assert_allclose(
        np.array(after[:-1]) - before[:-1], r["external_component_moles"], atol=1e-17, rtol=1e-9
    )
    assert audit(m, old, new, r)["accepted"]
    bad = deepcopy(r)
    bad["reservoir_exchange"]["supply"][0] += 1e-6
    with pytest.raises(ValueError, match="audit"):
        audit(m, old, new, bad)


def test_open_heat_chemical_complete_interval_after_phase_birth():
    from check_pr_active_surface import audit

    m, old = setup(chemical=0.001, evaporation=True)
    commands = {
        "pipe_heat_w": {"line": [{"liquid": 100.0}, {"liquid": 100.0}]},
        "pipe_heat_temperatures_k": {"line": [{"liquid": 400.0}, {"liquid": 400.0}]},
    }
    duration = 0.6 * m.maximum_step_s
    state = replace(old, next_step_s=0.2 * m.maximum_step_s)
    result = m.advance(state, duration, **commands)
    assert result["state"]["time_s"] == duration
    assert len(result["steps"]) >= 3
    for r in result["steps"]:
        end = PRActiveSurfaceState.from_dict(r["state"])
        assert audit(m, state, end, r)["accepted"]
        state = end
    assert all(len(cell) == 2 for cell in state.pipes[0][1].cells)
