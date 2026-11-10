"""Existing wall potential reused in nonisothermal PR energy and momentum."""

import json

import numpy as np
import pytest
import torch
from check_pr_capillary_axial import audit as axial_audit
from check_pr_capillary_pressure import audit as normal_audit
from test_surface_pr_capillary_pressure import chemical_case

from surface.film_wetting import LongWaveWettingPotential
from surface.pr_capillary_axial import PRCapillaryAxialPipe
from surface.pr_capillary_pressure import PRCapillaryPressure
from surface.thin_film_dynamics import ThinFilmDynamics


def setup(device="cpu", theta=5.0):
    base, old = chemical_case(device)
    parameters = {k: v for k, v in base.to_dict().items() if k not in ("model", "pipe_model_id")}
    parameters["wetting"] = {"adsorption_thickness_m": 1e-6, "equilibrium_contact_angle_deg": theta}
    return PRCapillaryPressure(base.pipe, **parameters), old


def test_shared_material_law_matches_existing_film_and_wall_energy_identity():
    potential = LongWaveWettingPotential(0.02, 1e-6, 5.0)
    film = ThinFilmDynamics((4,), (4e-5,), 0.02, 0.001, 600.0, 1e-6, 5.0, 1e-8)
    a = torch.tensor([0.9e-6, 1e-6, 2e-6, 3e-6], dtype=torch.float64)
    b = a * torch.tensor([1.03, 1.0, 0.98, 1.01])
    torch.testing.assert_close(film.potential(a), potential.potential(a), rtol=0, atol=0)
    change = potential.potential(b) - potential.potential(a)
    torch.testing.assert_close(
        change, potential.divided_difference(a, b) * (b - a), rtol=1e-12, atol=1e-20
    )
    assert float(potential.potential(a)[1]) == pytest.approx(-0.02 * (1 - np.cos(np.deg2rad(5.0))))
    neutral = LongWaveWettingPotential(0.02, 1e-6, 0.0)
    assert torch.isfinite(neutral.potential(a * 1e-290)).all()
    assert (neutral.potential(a * 1e-290) == 0).all()


def test_nonisothermal_normal_source_includes_real_wall_energy_and_pressure():
    model, old = setup()
    end, report = model.apply(old, 0.001)
    assert normal_audit(model, old, end, report)["accepted"]
    neutral, _ = chemical_case()
    assert model.surface_energy(old) != neutral.surface_energy(old)
    expected, _ = neutral.apply(old, 0.001)
    assert (
        max(
            abs(dict(c)["liquid"].volume_m3 - dict(d)["liquid"].volume_m3)
            for c, d in zip(end.cells, expected.cells)
        )
        > 1e-25
    )
    restored = PRCapillaryPressure.from_dict(model.pipe, json.loads(json.dumps(model.to_dict())))
    same, _ = restored.apply(old, 0.001)
    assert same == end


def test_joint_pr_wetting_step_updates_momentum_with_the_shared_potential():
    source, old = setup()
    with pytest.raises(ValueError, match="volume_diffusion"):
        PRCapillaryAxialPipe(source)
    model = PRCapillaryAxialPipe(source, volume_diffusion="none")
    end, record = model.step(old, 1e-12)
    assert axial_audit(model, old, end, record)["accepted"]
    assert all(v == 0 for v in record["capillary_diffusion_heat_j"])
    assert end.cells != old.cells
    restored = PRCapillaryAxialPipe.from_dict(json.loads(json.dumps(model.to_dict())))
    same, _ = restored.step(old, 1e-12)
    assert same == end


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_shared_wetting_coupling_matches_cpu():
    results = []
    for device in ("cpu", "cuda"):
        source, old = setup(device)
        model = PRCapillaryAxialPipe(source, volume_diffusion="none")
        end, _r = model.step_coupled(old, 1e-12)
        results.append(end)
    for a, b in zip(results[0].cells, results[1].cells):
        for (_, x), (_, y) in zip(a, b):
            np.testing.assert_allclose(x.component_moles, y.component_moles, rtol=1e-10, atol=1e-22)
            assert x.temperature_k == pytest.approx(y.temperature_k, abs=1e-8)


def test_wetting_throughflow_uses_material_entropy_flux_and_normal_energy():
    from test_surface_pr_capillary_open import setup as open_case

    base, old = open_case()
    parameters = {
        k: v for k, v in base.source.to_dict().items() if k not in ("model", "pipe_model_id")
    }
    parameters["wetting"] = {"adsorption_thickness_m": 1e-6, "equilibrium_contact_angle_deg": 5.0}
    source = PRCapillaryPressure(base.pipe, **parameters)
    model = PRCapillaryAxialPipe(source, boundaries=base.boundaries, volume_diffusion="none")
    transport, record = model.step(old, 1e-12)
    assert axial_audit(model, old, transport, record)["accepted"]
    final, normal = model.source.apply(transport, 1e-12)
    assert normal_audit(model.source, transport, final, normal)["accepted"]


def test_finite_junction_reuses_the_same_wall_energy_and_reactions():
    from check_pr_capillary_branch import audit as branch_audit
    from test_surface_pr_capillary_branch import setup as branch_case

    from surface.pr_active_multifluid_pipe import PRActiveMultifluidState
    from surface.pr_capillary_branch import PRCapillaryBranchNetwork

    base, old = branch_case()
    parameters = {
        **base.parameters,
        "wetting": {"adsorption_thickness_m": 1e-6, "equilibrium_contact_angle_deg": 5.0},
    }
    model = PRCapillaryBranchNetwork(
        base.pipes, base.junction_volumes, parameters, volume_diffusion="none"
    )
    initial = model.initialize(dict(old.pipes), {n: dict(v) for n, v in old.junctions})
    final, record = model.advance(initial, 1e-12, initial_step_s=1e-12)
    step = record["advance"]["steps"][0]
    transport = PRActiveMultifluidState.from_dict(step["axial"]["state"])
    end = PRActiveMultifluidState.from_dict(step["normal"]["state"])
    assert branch_audit(model, model._view(initial), transport, step["axial"])["accepted"]
    assert normal_audit(model.source, transport, end, step["normal"])["accepted"]
    restored = PRCapillaryBranchNetwork.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.to_dict() == model.to_dict()
    assert final.time_s == 1e-12


def test_periodic_material_force_translation_error_converges_under_mesh_refinement():
    """The discrete wall reaction is a truncation error, not calibrated pinning."""
    from reference_wetting_potential import pressure
    from test_surface_pr_active_multifluid_pipe import phase

    from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe

    seed, _ = setup()
    errors = []
    for count in (16, 32, 64, 128):
        raw = seed.pipe.to_dict()
        raw["lengths_m"] = [1e-3 / count] * count
        pipe = PRActiveMultifluidPipe.from_dict(raw)
        parameters = {k: v for k, v in seed.to_dict().items() if k not in ("model", "pipe_model_id")}
        source = PRCapillaryPressure(pipe, **{**parameters, "maximum_unknowns": 5 * count})
        x = (np.arange(count) + 0.5) * 2 * np.pi / count
        height = 1e-6 * (1.6 + 0.2 * np.cos(x) + 0.13 * np.sin(2 * x))
        cells = []
        for i, h in enumerate(height):
            vl = float(source.area[i]) * h
            cells.append({
                "liquid": phase(pipe.fluid, 1e6, 200.0, "liquid", vl, x=(0.05, 0.95)),
                "gas": phase(pipe.fluid, 1e6, 300.0, "vapor", pipe.volumes[i] - vl, x=(0.95, 0.05)),
            })
        old = pipe.initialize(cells)
        model = PRCapillaryAxialPipe(source, volume_diffusion="none")
        packet, h0, h1 = model.state_packet(old, old)
        wall_pressure = source.wetting.divided_difference(h0, h1)
        force = model._spatial(packet, h0, h1, potential_override=wall_pressure)[3]
        mu = pressure(parameters["wetting"], source.sigma, height, height)
        reference = source.width / 2 * np.sum(
            (mu + np.roll(mu, -1)) * (np.roll(height, -1) - height)
        )
        assert float(force.sum()) == pytest.approx(reference, rel=1e-8, abs=1e-24)
        errors.append(abs(float(force.sum())))
    assert all(a / b > 3.8 for a, b in zip(errors, errors[1:]))


def test_wall_energy_is_owned_once_by_the_axial_transport_ports():
    from check_pr_capillary_transport import audit
    from test_surface_pr_capillary_transport import case

    from surface.pr_capillary_transport import PRCapillaryTransportWork

    base, old, final, q, supply = case(periodic=True, sources=True)
    parameters = {k: v for k, v in base.to_dict().items() if k not in ("model", "pipe_model_id")}
    parameters["wetting"] = {"adsorption_thickness_m": 1e-6, "equilibrium_contact_angle_deg": 5.0}
    source = PRCapillaryPressure(base.pipe, **parameters)
    report = PRCapillaryTransportWork(source).evaluate(old, final, q, volume_source_m3=supply)
    assert audit(source, old, final, report)["accepted"]


@pytest.mark.parametrize("loop", [False, True])
def test_segmented_normal_relaxation_owns_material_energy_without_extra_port_work(loop):
    from check_pr_capillary_chain import audit
    from test_surface_pr_continuation import capillary_case

    model, initial, parameters, _base, _old = capillary_case(loop=loop)
    parameters["wetting"] = {"adsorption_thickness_m": 1e-6, "equilibrium_contact_angle_deg": 5.0}
    final, report = model.relax_capillary_chain(initial, 0.001, ("a", "b"), parameters)
    assert audit(model, initial, final, report, parameters)["accepted"]
