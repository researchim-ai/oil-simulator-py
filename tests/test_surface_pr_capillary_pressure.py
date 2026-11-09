"""Nonisothermal normal work changes PR volumes without inventing energy."""

import math

import numpy as np
import pytest
import torch
from test_surface_equilibrium_thermodynamics import fluid
from test_surface_pr_active_multifluid_pipe import phase

from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe
from surface.pr_capillary_pressure import PRCapillaryPressure


def setup(device="cpu", *, perturbation=0.05, pressure_jump=1e4, sigma=0.02):
    f = EquilibriumFluid.from_dict(fluid().to_dict(), device=device)
    width = 0.001
    dx = 1e-5
    depth = 20e-6
    p = PRActiveMultifluidPipe(
        f,
        [{"name": "liquid", "root": "liquid"}, {"name": "gas", "root": "vapor"}],
        [dx] * 4,
        math.sqrt(4 * width * depth / math.pi),
        boundary_mode="periodic",
    )
    cells = []
    for i in range(4):
        h = 3e-6 * (1 + perturbation * math.cos(2 * math.pi * i / 4))
        vl = dx * width * h
        vg = p.volumes[i] - vl
        cells.append(
            {
                "liquid": phase(f, 1e6 + pressure_jump, 200.0, "liquid", vl, u=0.1),
                "gas": phase(f, 1e6, 300.0, "vapor", vg, u=-0.2, x=(1.0, 0.0)),
            }
        )
    source = PRCapillaryPressure(p, "liquid", "gas", width, sigma, 1e-15)
    return source, p.initialize(cells)


def test_nonisothermal_normal_step_conserves_components_momentum_and_full_energy():
    s, old = setup()
    end, r = s.apply(old, 0.001)
    assert end.time_s == old.time_s
    assert not r["clock_advanced"]
    for before, after in zip(old.cells, end.cells):
        for (na, a), (nb, b) in zip(before, after):
            assert na == nb
            assert a.component_moles == b.component_moles
            assert a.momentum_kg_m_s == b.momentum_kg_m_s
        assert sum(p.volume_m3 for _, p in before) == pytest.approx(
            sum(p.volume_m3 for _, p in after), rel=1e-13
        )
    assert any(
        dict(a)["liquid"].temperature_k != dict(b)["liquid"].temperature_k
        for a, b in zip(old.cells, end.cells)
    )
    assert all(v >= 0 for v in r["normal_dissipation_j"])
    assert abs(r["energy_residual_j"]) <= r["energy_roundoff_allowance_j"]
    assert r["entropy_production_j_k"] >= -1e-12
    # Independent small-slope surface energy, no source energy helper.
    pipe = s.pipe

    def surface(st):
        h = np.array([dict(c)["liquid"].volume_m3 for c in st.cells]) / (0.001 * 1e-5)
        return 0.5 * 0.02 * 0.001 / 1e-5 * np.sum((np.roll(h, -1) - h) ** 2)

    defect = (
        math.fsum(
            b.total_energy_j - a.total_energy_j
            for ca, cb in zip(old.cells, end.cells)
            for (_, a), (_, b) in zip(ca, cb)
        )
        + surface(end)
        - surface(old)
    )
    assert abs(defect) <= r["energy_roundoff_allowance_j"]
    assert pipe.inspect(end)["entropy"] >= pipe.inspect(old)["entropy"] - 1e-12


def test_capillary_pressure_is_exact_discrete_surface_energy_gradient():
    s, old = setup(pressure_jump=0.0)
    end, r = s.apply(old, 0.001)
    area = 0.001 * 1e-5
    h0 = np.array([dict(c)["liquid"].volume_m3 for c in old.cells]) / area
    h1 = np.array([dict(c)["liquid"].volume_m3 for c in end.cells]) / area
    hm = (h0 + h1) / 2
    cap = 0.02 * (2 * hm - np.roll(hm, -1) - np.roll(hm, 1)) / 1e-10
    np.testing.assert_allclose(r["capillary_pressure_pa"], cap, rtol=1e-12, atol=1e-10)
    jump = np.array([dict(c)["liquid"].pressure_pa - dict(c)["gas"].pressure_pa for c in end.cells])
    dv = np.array(r["volume_transfer_m3"])
    np.testing.assert_allclose(jump - cap, dv / (0.001 * 1e-15), rtol=1e-6, atol=1e-6)


def test_flat_equal_pressure_interface_does_not_equilibrate_temperatures():
    s, old = setup(perturbation=0.0, pressure_jump=0.0)
    end, r = s.apply(old, 0.001)
    np.testing.assert_allclose(r["volume_transfer_m3"], 0.0, atol=1e-27)
    for cell in end.cells:
        assert dict(cell)["liquid"].temperature_k == pytest.approx(200.0, abs=1e-10)
        assert dict(cell)["gas"].temperature_k == pytest.approx(300.0, abs=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu():
    ends = []
    for device in ("cpu", "cuda"):
        s, old = setup(device)
        end, r = s.apply(old, 0.001)
        ends.append((end, r))
    for a, b in zip(ends[0][0].cells, ends[1][0].cells):
        for (_, p), (_, q) in zip(a, b):
            assert p.volume_m3 == pytest.approx(q.volume_m3, rel=1e-11, abs=1e-25)
            assert p.temperature_k == pytest.approx(q.temperature_k, abs=1e-8)


def test_nonuniform_nonperiodic_mesh_normal_source_and_restart():
    import json
    from dataclasses import replace

    s, old = setup()
    raw = s.pipe.to_dict()
    raw["boundary_mode"] = "walls"
    raw["lengths_m"] = [0.7e-5, 1.2e-5, 0.9e-5, 1.4e-5]
    pipe = PRActiveMultifluidPipe.from_dict(raw)
    cells = []
    for i, cell in enumerate(old.cells):
        scale = pipe.volumes[i] / s.pipe.volumes[i]
        cells.append(
            {
                name: replace(
                    p,
                    component_moles=tuple(n * scale for n in p.component_moles),
                    volume_m3=p.volume_m3 * scale,
                    momentum_kg_m_s=p.momentum_kg_m_s * scale,
                    total_energy_j=p.total_energy_j * scale,
                )
                for name, p in cell
            }
        )
    st = pipe.initialize(cells)
    source = PRCapillaryPressure(pipe, "liquid", "gas", s.width, s.sigma, s.mobility)
    end, r = source.apply(st, 0.001)
    assert abs(r["energy_residual_j"]) <= r["energy_roundoff_allowance_j"]
    rebuilt = PRCapillaryPressure.from_dict(pipe, json.loads(json.dumps(source.to_dict())))
    other, rr = rebuilt.apply(st, 0.001)
    assert other.to_dict() == end.to_dict()
    assert rr == r
    with pytest.raises(ValueError, match="identity"):
        PRCapillaryPressure.from_dict(s.pipe, source.to_dict())


def test_absent_phase_rejected_without_mutating_active_pipe_state():
    source, _old = setup()
    p = source.pipe
    f = p.fluid
    st = p.initialize([{"liquid": phase(f, 1e6, 200.0, "liquid", v)} for v in p.volumes])
    snapshot = st.to_dict()
    with pytest.raises(ValueError, match="both declared phases"):
        source.apply(st, 0.001)
    assert st.to_dict() == snapshot


@pytest.mark.parametrize("root,T,x", [("liquid", 200.0, (0.0, 1.0)), ("vapor", 320.0, (0.5, 0.5))])
def test_chemical_potentials_match_independent_scalar_gibbs_energy(root, T, x):
    from check_stratified_pr import Reference

    from surface.pr_capillary_pressure import chemical_potentials_tv

    f = fluid()
    q = Reference(f, np.flatnonzero(np.array(x) > 0)).state(1e6, T, x, root)
    mu = chemical_potentials_tv(f, f.tensor(T), f.tensor(q["vm"]), f.tensor(x)).cpu().numpy()
    assert np.dot(mu, x) == pytest.approx(q["u"] + 1e6 * q["vm"] - T * q["s"], rel=1e-12, abs=1e-9)


def chemical_case(device="cpu"):
    source, old = setup(device)
    pipe = source.pipe
    f = pipe.fluid
    cells = []
    for row in old.cells:
        cells.append(
            {
                name: phase(
                    f,
                    p.pressure_pa,
                    p.temperature_k,
                    "liquid" if name == "liquid" else "vapor",
                    p.volume_m3,
                    0.1 if name == "liquid" else -0.2,
                    x=(0.05, 0.95) if name == "liquid" else (0.95, 0.05),
                )
                for name, p in row
            }
        )
    model = PRCapillaryPressure(
        pipe,
        "liquid",
        "gas",
        source.width,
        source.sigma,
        source.mobility,
        component_mobility_mol2_k_j_s=[1e-12, 1e-12],
        transfer_energy_j_mol=[-3000.0, -8000.0],
        heat_conductance_w_k=1e-8,
    )
    return model, pipe.initialize(cells)


def test_surface_heat_and_component_exchange_are_solved_jointly_without_energy_projection():
    source, old = chemical_case()
    end, r = source.apply(old, 0.001)
    transfers = np.array(r["component_transfer_to_liquid_mol"])
    assert np.max(np.abs(transfers)) > 1e-15
    assert all(v > 0 for v in r["heat_transfer_to_liquid_j"])
    for i, (before, after) in enumerate(zip(old.cells, end.cells)):
        for j in range(2):
            assert sum(p.component_moles[j] for _, p in after) == pytest.approx(
                sum(p.component_moles[j] for _, p in before), rel=1e-14, abs=1e-24
            )
        dl = (
            np.array(dict(after)["liquid"].component_moles) - dict(before)["liquid"].component_moles
        )
        np.testing.assert_allclose(dl, transfers[i], rtol=1e-10, atol=1e-24)
        assert sum(p.momentum_kg_m_s for _, p in after) == pytest.approx(
            sum(p.momentum_kg_m_s for _, p in before), abs=1e-27
        )
    assert r["entropy_production_lower_bound_j_k"] > 0
    assert r["entropy_production_j_k"] >= r["entropy_production_lower_bound_j_k"] - 1e-14
    assert abs(r["energy_residual_j"]) <= r["energy_roundoff_allowance_j"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_joint_heat_mass_normal_source_matches_cpu():
    states = []
    for device in ("cpu", "cuda"):
        model, old = chemical_case(device)
        end, r = model.apply(old, 0.001)
        states.append((end, r))
    for a, b in zip(states[0][0].cells, states[1][0].cells):
        for (_, p), (_, q) in zip(a, b):
            np.testing.assert_allclose(p.component_moles, q.component_moles, rtol=1e-10, atol=1e-22)
            assert p.temperature_k == pytest.approx(q.temperature_k, abs=1e-8)


def test_partial_chemical_potentials_match_independent_helmholtz_derivatives():
    from check_stratified_pr import Reference

    from surface.peng_robinson import R
    from surface.pr_capillary_pressure import chemical_potentials_tv

    f = fluid()
    T = 320.0
    numbers = np.array([0.4, 0.6])
    x = numbers / numbers.sum()
    ref = Reference(f, np.array([0, 1]))
    volume = ref.state(1e6, T, x, "vapor")["vm"]

    def helmholtz(nn):
        xx = nn / nn.sum()
        v = volume / nn.sum()
        ai = (
            f.a0.cpu().numpy()
            * (1 + f.kappa.cpu().numpy() * (1 - np.sqrt(T / f.tc.cpu().numpy()))) ** 2
        )
        a = np.sum(
            xx[:, None]
            * xx[None, :]
            * np.sqrt(ai[:, None] * ai[None, :])
            * (1 - np.array(f.binary_interactions))
        )
        b = np.dot(xx, f.b.cpu().numpy())
        p = R * T / (v - b) - a / (v * v + 2 * b * v - b * b)
        q = ref.state(p, T, xx, "vapor")
        return nn.sum() * (q["u"] - T * q["s"])

    mu = chemical_potentials_tv(f, f.tensor(T), f.tensor(volume), f.tensor(x)).cpu().numpy()
    for i in range(2):
        delta = np.eye(2)[i] * 1e-5
        derivative = (helmholtz(numbers + delta) - helmholtz(numbers - delta)) / (2e-5)
        assert mu[i] == pytest.approx(derivative, rel=1e-8, abs=1e-6)


def test_inadmissible_slope_and_unsupported_mass_transfer_are_rejected_atomically():
    m, old = setup()
    tight = PRCapillaryPressure(
        m.pipe, m.liquid, m.gas, m.width, m.sigma, m.mobility, maximum_slope=0.001
    )
    with pytest.raises(ValueError, match="slope"):
        tight.apply(old, 0.001)
    with pytest.raises(ValueError, match="transfer energies"):
        PRCapillaryPressure(
            m.pipe,
            m.liquid,
            m.gas,
            m.width,
            m.sigma,
            m.mobility,
            component_mobility_mol2_k_j_s=[1e-12, 1e-12],
        )
    mass = PRCapillaryPressure(
        m.pipe,
        m.liquid,
        m.gas,
        m.width,
        m.sigma,
        m.mobility,
        component_mobility_mol2_k_j_s=[1e-12, 1e-12],
        transfer_energy_j_mol=[-3000.0, -8000.0],
    )
    snapshot = old.to_dict()
    with pytest.raises(ValueError, match="positive support"):
        mass.apply(old, 0.001)
    assert old.to_dict() == snapshot


@pytest.mark.parametrize("factory", [setup, chemical_case])
def test_independent_source_audit_and_corrupt_work_rejection(factory):
    from copy import deepcopy

    from check_pr_capillary_pressure import audit

    model, old = factory()
    end, report = model.apply(old, 0.001)
    assert audit(model, old, end, report)["checked_cells"] == 4
    for field, delta in (
        ("capillary_pressure_pa", 10.0),
        ("normal_dissipation_j", 1e-12),
        ("heat_transfer_to_liquid_j", 1e-12),
    ):
        corrupt = deepcopy(report)
        corrupt[field][0] += delta
        with pytest.raises(ValueError, match="audit"):
            audit(model, old, end, corrupt)


def test_joint_source_operates_on_live_open_graph_stocks_with_augmented_energy():
    from surface.pr_active_surface import PRActiveReservoir, PRActiveSurfaceNetwork

    source, old = chemical_case()
    raw = source.pipe.to_dict()
    raw["boundary_mode"] = "walls"
    raw["face_elevations_m"] = [0.0] * 5
    raw["gravity_m_s2"] = 0.0
    raw["hydrostatic_reference"] = None
    pipe = PRActiveMultifluidPipe.from_dict(raw)
    ps = pipe.initialize([dict(c) for c in old.cells])
    reservoir = PRActiveReservoir(tuple(p for _, p in ps.cells[0]), ("liquid", "vapor"))
    graph = PRActiveSurfaceNetwork(
        {"line": {"pipe": pipe, "boundary_nodes": ("feed", "export")}},
        {},
        [],
        reservoirs={"feed": reservoir, "export": reservoir},
    )
    model = PRCapillaryPressure.from_dict(
        pipe, {**source.to_dict(), "pipe_model_id": pipe.model_id}
    )
    state = graph.initialize({}, {"line": ps})
    before = state.to_dict()
    end, r = graph.relax_capillary(state, 0.001, {"line": model})
    assert state.to_dict() == before
    assert end.time_s == state.time_s
    assert end.pipes != state.pipes
    assert r["entropy_production_j_k"] > 0
    a = graph.capillary_totals(state, {"line": model})
    b = graph.capillary_totals(end, {"line": model})
    np.testing.assert_allclose(a["stocks"], b["stocks"], rtol=1e-12, atol=1e-20)
    assert max(abs(v) for v in r["conservation_residuals"]) < 1e-18
    with pytest.raises(ValueError, match="named graph pipes"):
        graph.relax_capillary(state, 0.001, {"typo": model})
    with pytest.raises(ValueError, match="differ"):
        graph.relax_capillary(state, 0.001, {"line": source})
    assert state.to_dict() == before
