"""Dynamic annular interface work, positive dissipation and independent audit."""

import math
from copy import deepcopy

import pytest
import torch
from test_surface_pr_active_multifluid_pipe import phase, setup

from surface.annular_film import annular_response
from surface.pr_active_annular_friction import MODEL
from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe


def model_state(*, gas_u=3.0, liquid_u=-0.1, alpha=0.02, single=False, device="cpu"):
    base, volume = setup(count=2)
    raw = base.to_dict()
    raw["model"] = "pr_active_multifluid_pipe_v3"
    length = raw["lengths_m"][0]
    raw["face_elevations_m"] = [0.0, length, 2 * length]
    raw["gravity_m_s2"] = 9.80665
    raw["wall_friction"] = {
        "model": MODEL,
        "gas_slot": "gas",
        "liquid_slot": "liquid",
        "viscosity_pa_s": {"gas": 1e-5, "liquid": 0.001},
        "roughness_m": 0.0,
    }
    pipe = PRActiveMultifluidPipe.from_dict(raw, device=device)
    f = pipe.fluid
    p = f.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    cell = {
        "liquid": phase(f, p, 200.0, "liquid", volume * alpha, liquid_u),
        "gas": phase(f, p, 200.0, "vapor", volume * (1 - alpha), gas_u),
    }
    other = {"gas": phase(f, p, 200.0, "vapor", volume, gas_u)} if single else cell
    return pipe, pipe.initialize([cell, other])


@pytest.mark.parametrize("gas_u,liquid_u", [(3.0, -0.1), (-3.0, 0.1), (0.0, 0.0), (2.0, 0.1)])
def test_interface_work_is_conservative_and_dissipation_heats_each_phase(gas_u, liquid_u):
    from check_pr_active_surface import friction_audit

    pipe, old = model_state(gas_u=gas_u, liquid_u=liquid_u, single=True)
    new, r = pipe.friction.apply(pipe, old, 0.001)
    friction_audit(pipe, old, new, r)
    assert abs(sum(r["interface_work_j"])) < 1e-14
    assert min(r["physical_dissipation_j"]) >= -1e-14
    assert min(r["temporal_dissipation_j"]) >= 0
    for old_cell, new_cell in zip(old.cells, new.cells):
        for name, a in old_cell:
            b = dict(new_cell)[name]
            assert b.temperature_k >= a.temperature_k - 1e-10
            assert b.volume_m3 == a.volume_m3
            assert b.component_moles == a.component_moles
    assert max(map(abs, r["conservation_residuals"])) < 1e-8
    assert len(new.cells[1]) == 1


def test_independent_audit_rejects_fake_interface_work_and_surface_velocity():
    from check_pr_active_surface import friction_audit

    pipe, old = model_state()
    new, r = pipe.friction.apply(pipe, old, 0.001)
    for mutate in ("work", "surface"):
        bad = deepcopy(r)
        if mutate == "work":
            bad["interface_work_j"][0] += 0.1
        else:
            bad["annular_cells"][0]["interface_velocity_m_s"] += 1.0
        with pytest.raises(ValueError, match="audit differs"):
            friction_audit(pipe, old, new, bad)


@pytest.mark.parametrize("alpha", [1e-12, 1e-8, 1e-4])
def test_true_vanishing_film_has_finite_thin_limit_without_mass_floor(alpha):
    a = torch.tensor(alpha, dtype=torch.float64)
    eta, beta, c, perimeter = annular_response(a, 0.1, 0.001)
    delta = 0.05 * alpha / (1 + math.sqrt(1 - alpha))
    assert float(eta) == pytest.approx(1.5, rel=1e-4)
    assert float(beta) == pytest.approx(delta / (4 * 0.001), rel=1e-4)
    assert float(c) == pytest.approx(3 * 0.001 * math.pi * 0.1 / delta, rel=1e-4)
    assert float(perimeter) > 0
    pipe, old = model_state(alpha=alpha, liquid_u=0.0)
    new, r = pipe.friction.apply(pipe, old, 0.1)
    assert (
        dict(new.cells[0])["liquid"].component_moles == dict(old.cells[0])["liquid"].component_moles
    )
    assert all(math.isfinite(v) for v in r["dissipation_j"])


@pytest.mark.parametrize("dt", [1e-8, 0.1, 1e6])
def test_stiff_source_remains_bounded_and_energy_conservative(dt):
    pipe, old = model_state()
    new, r = pipe.friction.apply(pipe, old, dt)
    assert max(map(abs, r["conservation_residuals"])) < 1e-8
    assert r["entropy_production_j_k"] >= -1e-9
    if dt == 1e6:
        assert max(abs(p.momentum_kg_m_s) for _, p in new.cells[0]) < 1e-8


def test_laminar_gas_core_tends_to_single_phase_poiseuille_when_film_vanishes():
    pipe, old = model_state(alpha=1e-12, gas_u=1e-4, liquid_u=0.0)
    dt = 0.1
    new, r = pipe.friction.apply(pipe, old, dt)
    a, b = dict(old.cells[0])["gas"], dict(new.cells[0])["gas"]
    mass = sum(n * c.molar_mass_kg_mol for n, c in zip(a.component_moles, pipe.fluid.components))
    expected = 1e-4 / (
        1 + dt * 32 * pipe.friction.viscosity["gas"] / ((mass / a.volume_m3) * pipe.diameter**2)
    )
    assert b.momentum_kg_m_s / mass == pytest.approx(expected, rel=1e-9, abs=1e-15)
    assert r["annular_cells"][0]["gas_driving_reynolds"] < 2000


def test_dynamic_momentum_and_interface_energy_converge_to_independent_ode():
    import numpy as np
    from reference_pr_annular_source import resistance
    from scipy.integrate import solve_ivp

    pipe, initial = model_state()
    phases = dict(initial.cells[0])
    names = ["gas", "liquid"]
    masses = np.array(
        [
            sum(
                n * c.molar_mass_kg_mol
                for n, c in zip(phases[name].component_moles, pipe.fluid.components)
            )
            for name in names
        ]
    )
    alpha = phases["liquid"].volume_m3 / pipe.volumes[0]
    eta, beta, c, perimeter = resistance(alpha, pipe.diameter, pipe.friction.viscosity["liquid"])
    rho = masses[0] / phases["gas"].volume_m3
    delta = pipe.diameter / 2 * (1 - math.sqrt(1 - alpha))
    k0 = 0.5 * 0.005 * (1 + 300 * delta / pipe.diameter) * rho
    length = pipe.lengths[0]
    y0 = np.array(
        [phases[name].momentum_kg_m_s / masses[i] for i, name in enumerate(names)]
        + [phases["gas"].total_energy_j]
    )

    def rhs(_, y):
        ug, ul, _energy = y
        drive = ug - eta * ul
        # This fixture remains in the fully turbulent gas branch throughout.
        assert (
            rho * abs(drive) * pipe.diameter * math.sqrt(1 - alpha) / pipe.friction.viscosity["gas"]
            > 4000
        )
        slip = 2 * drive / (1 + math.sqrt(1 + 4 * beta * k0 * abs(drive)))
        tau = k0 * slip * abs(slip)
        ui = eta * ul + beta * tau
        return [
            -length * perimeter * tau / masses[0],
            length * (perimeter * eta * tau - c * ul) / masses[1],
            -length * perimeter * tau * ui,
        ]

    horizon = 0.02
    solution = solve_ivp(rhs, (0.0, horizon), y0, method="DOP853", rtol=1e-12, atol=1e-13)
    assert solution.success
    target = solution.y[:, -1]
    errors = []
    for steps in (4, 8, 16):
        state = initial
        for _ in range(steps):
            state, _ = pipe.friction.apply(pipe, state, horizon / steps)
        cell = dict(state.cells[0])
        actual = np.array(
            [cell[name].momentum_kg_m_s / masses[i] for i, name in enumerate(names)]
            + [cell["gas"].total_energy_j]
        )
        errors.append(np.abs(actual - target))
    for coarse, fine in zip(errors, errors[1:]):
        assert np.all(coarse / fine > 1.7)
        assert np.all(coarse / fine < 2.3)


def test_pipe_transport_and_source_pass_full_independent_spatial_audit():
    from check_pr_active_surface import pipe_audit

    pipe, old = model_state()
    new, r = pipe.step(old, 0.01 * pipe.maximum_step_s)
    pipe_audit(pipe, old, new, r)
    restored = PRActiveMultifluidPipe.from_dict(pipe.to_dict())
    assert restored.model_id == pipe.model_id


def test_annular_domain_rejects_horizontal_and_thick_liquid_not_silent_regime_switch():
    pipe, old = model_state(alpha=0.5)
    with pytest.raises(ValueError, match="Annular state outside"):
        pipe.friction.apply(pipe, old, 0.001)
    raw = pipe.to_dict()
    raw["face_elevations_m"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="near-vertical"):
        PRActiveMultifluidPipe.from_dict(raw)


def test_connected_finite_aperture_stop_restart_and_checkpoint_are_auditable():
    import json
    from pathlib import Path

    from check_pr_active_surface import audit
    from run_pr_active_surface import run

    from surface.pr_active_surface import PRActiveSurfaceNetwork, PRActiveSurfaceState

    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/surface_pr_active_surface_annular.json"
        ).read_text()
    )
    result = run(config, device="cpu")
    model = PRActiveSurfaceNetwork.from_dict(config["system"])
    old = PRActiveSurfaceState.from_dict(result["initial"])
    positions = []
    for r in result["result"]["steps"]:
        new = PRActiveSurfaceState.from_dict(r["state"])
        audit(model, old, new, r)
        positions.append(r["port_positions"]["pickup"])
        assert all(p["model"] == MODEL for p in (row["friction"] for row in r["pipes"].values()))
        old = new
    assert set(positions) == {0.0, 0.5, 1.0}
    assert old.time_s == 3e-6
    assert max(map(abs, result["result"]["conservation_residuals"])) < 1e-7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_annular_source_cuda_matches_cpu():
    cpu, old = model_state()
    gpu, gold = model_state(device="cuda")
    a, ra = cpu.friction.apply(cpu, old, 0.001)
    b, rb = gpu.friction.apply(gpu, gold, 0.001)
    for ca, cb in zip(a.cells, b.cells):
        for (_, pa), (_, pb) in zip(ca, cb):
            assert pb.momentum_kg_m_s == pytest.approx(pa.momentum_kg_m_s, abs=1e-12, rel=1e-10)
            assert pb.total_energy_j == pytest.approx(pa.total_energy_j, abs=1e-9, rel=1e-10)
    assert rb["dissipation_j"] == pytest.approx(ra["dissipation_j"], abs=1e-9, rel=1e-9)
