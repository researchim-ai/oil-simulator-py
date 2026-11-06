"""Rest over terrain, liquid traps and coupled reconstruction derivatives."""

import json
import math
from pathlib import Path

import pytest
import torch
from test_surface_compositional_separator import binary

from surface.peng_robinson import R
from surface.pr_equilibrium_transport import (
    PREquilibriumLink,
    PREquilibriumNode,
    PREquilibriumTransportNetwork,
    PREquilibriumTransportState,
)
from surface.pr_inclined_pipe import solve_chemical_nodes


def pipe(u_shape=False, dense_limit=128):
    f = binary()
    length, diameter, gravity = 0.05, 0.01, 9.80665
    heights = [0.075, 0.025, 0.025, 0.075] if u_shape else [0.025, 0.075, 0.125, 0.175]
    slopes = [-length, -length, length, length] if u_shape else [length] * 4
    nodes = [
        PREquilibriumNode(
            str(i),
            math.pi * diameter**2 / 4 * length,
            "pipe_cell",
            elevation_m=z,
            length_m=length,
            diameter_m=diameter,
            elevation_change_m=dz,
            hydrostatic_reconstruction=True,
        )
        for i, (z, dz) in enumerate(zip(heights, slopes))
    ]
    links = [PREquilibriumLink(f"face{i}", str(i), str(i + 1), diameter) for i in range(3)]
    model = PREquilibriumTransportNetwork(
        f, nodes, links, gravity_m_s2=gravity, dense_limit=dense_limit
    )
    temperature = 200.0 if u_shape else 350.0
    x = f.tensor([0.0, 1.0] if u_shape else [1.0, 0.0])
    pressure = (
        f.components[1].saturation_at_temperature(temperature)["pressure_pa"] if u_shape else 1e6
    )
    anchor = 0.05 if u_shape else 0.0
    active = torch.nonzero(x > 0).flatten()
    phase = f.tensor_homogeneous(f.tensor(pressure), f.tensor(temperature), x, "stable")
    mu = phase["log_fugacity_coefficients"][active] + math.log(pressure / f.pref)
    target = (
        mu[None, :]
        + (f.molar_masses[active] * gravity / (R * temperature))[None, :]
        * (anchor - f.tensor(heights))[:, None]
    )
    initial = f.tensor([[math.log(pressure)]] * 4)
    if u_shape:
        for i, z in enumerate(heights):
            phase = f.tensor_homogeneous(
                f.tensor(pressure), f.tensor(temperature), x, "liquid" if z < anchor else "vapor"
            )
            initial[i, 0] = (
                f.tensor(pressure) + phase["density_kg_m3"] * gravity * (anchor - z)
            ).log()
    solved = solve_chemical_nodes(f, initial, f.tensor([temperature] * 4), target, active)
    state = model.initialize(
        {
            str(i): {
                "pressure_pa": float(solved[i, 0].exp()),
                "temperature_k": temperature,
                "mole_fractions": x.tolist(),
            }
            for i in range(4)
        }
    )
    return model, state


@pytest.mark.parametrize("dense_limit", [1, 128])
@pytest.mark.parametrize("u_shape", [False, True])
def test_hydrostatic_terrain_and_liquid_trap_remain_at_rest(u_shape, dense_limit):
    m, old = pipe(u_shape, dense_limit)
    if u_shape:
        density = [float(m._state_data(i, s)["rho"]) for i, (_, s) in enumerate(old.nodes)]
        assert density[1] > 10 * density[0] and density[2] > 10 * density[3]
    new, report = m._step(
        old, 0.01, m.default_openings, [0.0] * 4, [200.0 if u_shape else 350.0] * 4
    )
    for i, ((_, a), (_, b)) in enumerate(zip(old.nodes, new.nodes)):
        assert abs(float(m._state_data(i, b)["u"])) < 1e-7
        assert b.component_moles == pytest.approx(a.component_moles, rel=1e-8, abs=1e-12)
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
    assert len(report["hydrostatic_faces"]) == 4
    assert len(old.nodes) == len(new.nodes)
    restored = type(m).from_dict(json.loads(json.dumps(m.to_dict())))
    restored.inspect(PREquilibriumTransportState.from_dict(json.loads(json.dumps(new.to_dict()))))


def test_face_hydrostatic_jacobian_and_sparse_support_are_live():
    m, state = pipe(dense_limit=1)
    seeds = [m._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, evaluate, offsets, *_ = m._problem(state, seeds, 1e-4, m.default_openings, [0.0] * 4)
    assert len(y) > offsets[-1]
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype)
    eps = 1e-4
    finite = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, finite, rtol=1e-5, atol=1e-7)
    support = m._sparsity(
        offsets, m.default_openings, hydrostatic_entries=evaluate.hydrostatic_faces.entries
    )
    mask = torch.zeros_like(jac, dtype=torch.bool)
    mask[support[0], support[1]] = True
    assert not bool((jac[~mask].abs() > 1e-12).any())


def test_hydrostatic_flag_is_explicit_and_legacy_graph_identity_is_preserved():
    f = binary()
    node = PREquilibriumNode("mixed", 0.001)
    m = PREquilibriumTransportNetwork(f, [node])
    assert "hydrostatic_reconstruction" not in m.to_dict()["nodes"][0]
    with pytest.raises(ValueError, match="finite pipe"):
        PREquilibriumNode("bad", 0.001, hydrostatic_reconstruction=True)


def test_independent_reconstruction_and_balances_reject_tampered_face(monkeypatch):
    import copy

    m, state = pipe(u_shape=True)
    initial = {
        name: {
            "pressure_pa": s.pressure_pa,
            "temperature_k": s.temperature_k,
            "mole_fractions": [n / sum(s.component_moles) for n in s.component_moles],
        }
        for name, s in state.nodes
    }
    config = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [{"duration_s": 0.01}],
            }
        )
    )
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    result = json.loads(json.dumps(run(config)))
    assert audit(config, result)["accepted"]
    bad = copy.deepcopy(result)
    bad["intervals"][0]["steps"][0]["hydrostatic_faces"]["0"]["left"]["pressure_pa"] *= 1.001
    with pytest.raises(ValueError):
        audit(config, bad)


def test_surveyed_route_has_exact_inventory_and_keeps_static_liquid_pool():
    from surface.pr_terrain_pipe import PRTerrainPipe, initialize_hydrostatic_transport
    from surface.pr_transport_builder import build_equilibrium_transport

    f = binary()
    d = 0.01
    vessels = [
        PREquilibriumNode("left", 1e-5, elevation_m=0.1),
        PREquilibriumNode("right", 1e-5, elevation_m=0.1),
    ]
    survey = PRTerrainPipe(
        "route",
        "left",
        "right",
        d,
        (0.1, 0.1, 0.2),
        (0.1, 0.0, 0.1, 0.1),
        (2, 2, 3),
        bend_loss_coefficients=(1.0, 2.0),
    )
    m = build_equilibrium_transport(
        f,
        vessels,
        [survey],
        gravity_m_s2=9.80665,
        initial_step_s=0.01,
        maximum_step_s=1.0,
        minimum_step_s=1e-6,
    )
    assert len(m.nodes) == 9
    assert sum(n.volume_m3 for n in m.nodes) == pytest.approx(2e-5 + math.pi * d * d / 4 * 0.4)
    assert sum(n.local_loss_coefficient for n in m.nodes) == 3.0
    p = f.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    state = initialize_hydrostatic_transport(m, p, 200.0, [0.0, 1.0], anchor_elevation_m=0.05)
    old = state.to_dict()
    result = m.advance(state, 0.1)
    end = PREquilibriumTransportState.from_dict(result["state"])
    assert end.time_s == 0.1
    assert state.to_dict() == old
    for i, ((_, a), (_, b)) in enumerate(zip(state.nodes, end.nodes)):
        assert abs(float(m._state_data(i, b)["u"])) < 1e-7
        assert b.component_moles == pytest.approx(a.component_moles, rel=1e-7, abs=1e-11)


def test_survey_rejects_impossible_geometry_and_mismatched_junctions():
    from surface.pr_terrain_pipe import PRTerrainPipe
    from surface.pr_transport_builder import build_equilibrium_transport

    with pytest.raises(ValueError, match="length must"):
        PRTerrainPipe("bad", "a", "b", 0.01, (1.0,), (0.0, 2.0), (2,))
    route = PRTerrainPipe("bad", "a", "b", 0.01, (1.0,), (0.0, 0.5), (2,))
    with pytest.raises(ValueError, match="terminal elevations"):
        build_equilibrium_transport(
            binary(), [PREquilibriumNode("a", 0.01), PREquilibriumNode("b", 0.01)], [route]
        )


def test_interface_aligned_survey_preserves_geometry_and_original_bend_losses():
    from surface.pr_terrain_pipe import PRTerrainPipe

    route = PRTerrainPipe(
        "trap",
        "a",
        "b",
        0.01,
        (2.0, 3.0),
        (1.0, 0.0, 1.0),
        (4, 6),
        bend_loss_coefficients=(2.5,),
    )
    split = route.split_at_levels([0.25, 0.75, 0.25])
    assert split.length_m == route.length_m
    assert sum(split.bend_loss_coefficients) == 2.5
    assert split.elevations_m == (1.0, 0.75, 0.25, 0.0, 0.25, 0.75, 1.0)
    for _, center, change, _ in split.cell_routes():
        assert not any(
            center - abs(change) / 2 + 1e-12 < h < center + abs(change) / 2 - 1e-12
            for h in (0.25, 0.75)
        )


def test_three_phase_terrain_rest_and_restart_use_consistent_algebraic_warm_starts():
    from test_surface_compositional_separator import ternary

    from surface.pr_terrain_pipe import PRTerrainPipe, initialize_from_gravity_profile
    from surface.pr_transport_builder import build_equilibrium_transport
    from surface.separator_geometry import SeparatorGeometry
    from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState

    fluid = ternary()
    column = StratifiedPRColumn(
        fluid,
        SeparatorGeometry(0.05, 0.1, "vertical"),
        quadrature_order=2,
        maximum_quadrature_order=8,
    )
    profile = StratifiedPRState.from_dict(column.initialize(1e6, 350.0, [0.4, 0.3, 0.3])["state"])
    route = PRTerrainPipe(
        "trap", "a", "b", 0.005, (0.1, 0.1), (0.1, 0.0, 0.1), (2, 2)
    ).split_at_levels(profile.interface_heights_m)
    model = build_equilibrium_transport(
        fluid,
        [
            PREquilibriumNode("a", 1e-6, elevation_m=0.1),
            PREquilibriumNode("b", 1e-6, elevation_m=0.1),
        ],
        [route],
        gravity_m_s2=column.gravity_m_s2,
        dense_limit=1,
        wave_speed_bound_m_s=5000.0,
    )
    state = initialize_from_gravity_profile(model, column, profile)
    assert any(s.hydrostatic_face_guesses for _, s in state.nodes)
    # Direct centre-only input must discover the same stationary profile;
    # callers do not need private column traces to initialise the graph.
    state = model.initialize(
        {
            name: {
                "pressure_pa": s.pressure_pa,
                "temperature_k": s.temperature_k,
                "mole_fractions": [n / sum(s.component_moles) for n in s.component_moles],
            }
            for name, s in state.nodes
        }
    )
    assert any(s.hydrostatic_face_guesses for _, s in state.nodes)
    legacy = json.loads(json.dumps(state.to_dict()))
    for node in legacy["nodes"].values():
        node.pop("hydrostatic_face_guesses", None)
    restored = PREquilibriumTransportState.from_dict(legacy)
    for _ in range(2):
        end, record = model._step(
            restored,
            0.001,
            model.default_openings,
            [0.0] * len(model.nodes),
            [350.0] * len(model.nodes),
        )
        assert max(map(abs, record["conservation_residuals"])) < 1e-8
        for i, ((_, before), (_, after)) in enumerate(zip(state.nodes, end.nodes)):
            assert after.component_moles == pytest.approx(
                before.component_moles, rel=1e-11, abs=1e-25
            )
            assert abs(float(model._state_data(i, after)["u"])) < 1e-12
        restored = PREquilibriumTransportState.from_dict(json.loads(json.dumps(end.to_dict())))


def test_hydrostatic_warm_start_is_not_a_fixed_pressure_boundary():
    model, state = pipe()
    warm, _ = model._step(state, 0.001, model.default_openings, [0.0] * 4, [350.0] * 4)
    assert all(s.hydrostatic_face_guesses for _, s in warm.nodes)
    end, report = model._step(
        warm, 0.001, model.default_openings, [1.0, 0.0, 0.0, 0.0], [500.0] * 4
    )
    assert any(abs(v["flux"][0]) > 1e-10 for v in report["links"].values())
    assert any(
        abs(p - dict(warm.nodes)[name].hydrostatic_face_guesses[j][1]) > 1e-3
        for name, s in end.nodes
        for j, (_, p, _) in enumerate(s.hydrostatic_face_guesses)
    )
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
