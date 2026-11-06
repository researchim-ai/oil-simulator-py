"""Sectional particle stocks, upper-tail conservation and energy coupling."""

import json
from dataclasses import replace

import pytest
import torch

from surface.pr_sectional_population import PRSectionalDropletPopulation


def spec(kernel=1e-15, rates=(0.0, 0.1, 0.2, 0.3), power=1.0):
    return PRSectionalDropletPopulation(
        family="oil_drops",
        coalescence_kernel_m3_s=kernel,
        agitation_power_w=power,
        reference_volumes_m3=(1e-15, 2e-15, 4e-15, 8e-15),
        initial_volume_fractions=(0.4, 0.4, 0.1999, 0.0001),
        section_breakup_rates_s=rates,
    )


def reference_moment(s, coordinates):
    return (
        coordinates[:-2] * coordinates.new_tensor(s.reference_volumes_m3[:-1])
    ).sum() + coordinates[-1]


def setup(kernel=1e-15, rates=(0.0, 0.1, 0.2, 0.3), power=1.0, vertical=False, physical=False):
    from test_surface_pr_droplet_population import closed

    from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork

    base, _, initial = closed(vertical=vertical)
    population = spec(0.0 if physical else kernel, rates, power)
    if physical:
        population = replace(
            population, coalescence_model="brownian_gravity", collision_efficiency=0.4
        )
    model = PREquilibriumTransportNetwork(
        base.fluid,
        [replace(base.nodes[0], droplet_population=population)],
        initial_step_s=0.001,
        gravity_m_s2=base.gravity_m_s2,
    )
    return model, model.initialize({"emulsion": initial}), initial


@pytest.mark.parametrize("tail_ratio", [1.0, 2.0, 100.0])
def test_aggregation_and_binary_breakup_conserve_reference_volume_and_event_count(tail_ratio):
    s = spec()
    c = torch.tensor([1e9, 2e9, 3e9, 4e8, 4e8 * 8e-15 * tail_ratio], dtype=torch.float64)
    coal, split = s.sources(c, 1e-4)
    assert abs(float(reference_moment(s, coal))) < 1e-18
    assert abs(float(reference_moment(s, split))) < 1e-18
    assert float(coal[:-1].sum()) == pytest.approx(
        -0.5 * s.coalescence_kernel_m3_s * float(c[:-1].sum()) ** 2 / 1e-4, rel=1e-12
    )
    assert float(split[:-1].sum()) == pytest.approx(
        float(c[:-1] @ c.new_tensor(s.section_breakup_rates_s)), rel=1e-12
    )


def test_tail_keeps_all_volume_when_two_largest_drops_coalesce():
    s = spec(rates=(0.0, 0.0, 0.0, 0.0))
    c = torch.tensor([0.0, 0.0, 0.0, 1e9, 1e9 * 8e-15 * 5], dtype=torch.float64)
    coal, _ = s.sources(c, 1e-4)
    assert coal[-2] < 0 and abs(float(coal[-1])) < 1e-18
    assert torch.equal(coal[:3], torch.zeros(3, dtype=torch.float64))


def test_closed_mixer_resolves_spectrum_and_interface_energy_in_common_newton():
    m, start, _ = setup()
    end, report = m._step(start, 0.001, m.default_openings, [0.0], [350.0])
    a = torch.tensor(start.nodes[0][1].droplet_sections, dtype=torch.float64)
    b = torch.tensor(end.nodes[0][1].droplet_sections, dtype=torch.float64)
    assert reference_moment(spec(), b) == pytest.approx(
        float(reference_moment(spec(), a)), rel=1e-10
    )
    coal, split = m.nodes[0].droplet_population.sources(b, m.nodes[0].volume_m3)
    torch.testing.assert_close(b, a + 0.001 * (coal + split), rtol=1e-8, atol=1e-14)
    assert end.nodes[0][1].total_energy_j - start.nodes[0][1].total_energy_j == pytest.approx(
        0.001, abs=1e-8
    )
    assert b[-2] != a[-2]
    assert (
        max(map(abs, report["droplet_populations"]["emulsion"]["section_balance_residuals"])) < 1e-8
    )
    restored = type(m).from_dict(json.loads(json.dumps(m.to_dict())))
    assert restored.model_id == m.model_id
    checkpoint = type(end).from_dict(json.loads(json.dumps(end.to_dict())))
    restored.inspect(checkpoint)


def test_spectrum_produces_size_dependent_settling_and_resolved_kinetic_variance():
    m, state, _ = setup(vertical=True)
    d = m._state_data(0, state.nodes[0][1])
    p = d["population"]
    v = p["section_velocities_m_s"]
    weights = p["section_volume_fractions"]
    assert v.max() > v.min()
    mean = (d["phase_velocity"] * p["drop_mask"]).sum()
    assert float(weights @ v) == pytest.approx(float(mean), abs=1e-12)
    masses = d["phase_n"] * (d["x"] * m.fluid.molar_masses).sum(-1)
    assert d["kinetic"] > 0.5 * (masses * d["phase_velocity"].square()).sum()


@pytest.mark.parametrize("physical", [False, True])
def test_common_jacobian_has_every_spectrum_column_and_matches_finite_difference(physical):
    m, s, _ = setup(vertical=True, physical=physical)
    seeds = [m._seed(i, x) for i, (_, x) in enumerate(s.nodes)]
    y, _, _, evaluate, offsets, *_ = m._problem(s, seeds, 1e-5, m.default_openings, [0.0])
    assert len(y) == offsets[-1] + spec().width
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-1e-5, 1e-5, len(y), dtype=y.dtype)
    eps = 1e-4
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("sectional", [False, True])
def test_closed_connection_stops_particle_flux_as_well_as_component_flux(sectional):
    from test_surface_oil_water_drift import three_phase_pipe

    from surface.pr_droplet_population import PRDropletPopulation
    from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork

    base, _, initial = three_phase_pipe(interface="immobile")
    population = (
        spec(kernel=0.0, rates=(0.0, 0.0, 0.0, 0.0), power=0.0)
        if sectional
        else PRDropletPopulation("oil_drops", 0.0)
    )
    m = PREquilibriumTransportNetwork(
        base.fluid,
        [replace(n, droplet_population=population) for n in base.nodes],
        base.links,
        gravity_m_s2=base.gravity_m_s2,
    )
    s = m.initialize(initial)
    data = [m._state_data(i, body) for i, (_, body) in enumerate(s.nodes)]
    _, opened, _ = m._fluxes(data, {"internal": True})
    assert bool((opened[0]["droplet_number_fluxes"]["oil_drops"].abs() > 0).any())
    _, closed, _ = m._fluxes(data, {"internal": False})
    assert bool((closed[0]["droplet_number_fluxes"]["oil_drops"] == 0).all())
    assert bool((closed[0]["flux"] == 0).all())


def test_independent_transport_audit_size_selective_flux_and_restart():
    from pathlib import Path

    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    cfg = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/surface_pr_sectional_transport.json"
        ).read_text()
    )
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]
    first = result["intervals"][0]["steps"][0]
    rates = torch.tensor(first["links"]["internal"]["droplet_section_fluxes"]["oil_drops"][:-1])
    counts = torch.tensor(result["initial"]["nodes"]["0"]["droplet_sections"][:-1])
    normalized = rates / counts
    assert float(normalized.max() - normalized.min()) > 1e-7
    for step in result["intervals"][1]["steps"]:
        assert all(x == 0 for x in step["links"]["internal"]["droplet_section_fluxes"]["oil_drops"])
    resumed = {
        **cfg,
        "initial": {"state": result["intervals"][-1]["state"]},
        "schedule": cfg["schedule"][:1],
    }
    continued = json.loads(json.dumps(run(resumed)))
    assert audit(resumed, continued)["accepted"]
    broken = json.loads(json.dumps(result))
    broken["intervals"][0]["steps"][0]["droplet_populations"]["0"]["section_numbers"][0] *= 1.01
    with pytest.raises(ValueError):
        audit(cfg, broken)


def test_power_limit_rejects_breakup_without_changing_checkpoint():
    m, state, _ = setup(rates=(0.0, 10.0, 10.0, 10.0), power=1e-15)
    before = state.to_dict()
    with pytest.raises(ValueError, match="mechanical"):
        m.advance(state, 0.001)
    assert state.to_dict() == before


def test_no_droplet_stock_survives_when_dispersed_pair_is_absent():
    m, _, initial = setup()
    state = m.initialize({"emulsion": {**initial, "pressure_pa": 1e5, "temperature_k": 450.0}})
    assert state.nodes[0][1].droplet_sections == (0.0,) * spec().width
    assert state.nodes[0][1].droplet_number == 0.0


def test_spectrum_can_discharge_into_finite_capture_volume_with_energy_audit():
    from dataclasses import asdict
    from pathlib import Path

    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    from surface.pr_equilibrium_transport import PREquilibriumLink, PREquilibriumNode

    cfg = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/surface_pr_sectional_transport.json"
        ).read_text()
    )
    cfg["model"]["nodes"].append(
        asdict(PREquilibriumNode("export", 0.001, elevation_m=0.2, droplet_capture=True))
    )
    cfg["model"]["links"].append(asdict(PREquilibriumLink("outlet", "1", "export", 0.05)))
    for row in cfg["initial"]["nodes"].values():
        row["velocity_m_s"] = 0.1
    cfg["initial"]["nodes"]["export"] = {
        k: v for k, v in cfg["initial"]["nodes"]["1"].items() if k != "velocity_m_s"
    }
    cfg["initial"]["nodes"]["export"]["pressure_pa"] = 9.99e5
    cfg["schedule"] = [{"duration_s": 1e-7}]
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]
    rate = result["intervals"][0]["steps"][0]["links"]["outlet"]["droplet_section_fluxes"][
        "oil_drops"
    ]
    assert all(x > 0 for x in rate)
    assert "droplet_sections" not in result["intervals"][-1]["state"]["nodes"]["export"]
    assert max(map(abs, result["conservation_residuals"])) < 1e-7


def test_constant_kernel_spectrum_matches_exact_smoluchowski_solution_with_large_tail():
    import numpy as np
    from scipy.integrate import solve_ivp

    s = replace(
        spec(kernel=2.0, rates=(0.0, 0.0, 0.0, 0.0), power=0.0),
        reference_volumes_m3=tuple(float(k) for k in range(1, 9)),
        initial_volume_fractions=(0.125,) * 8,
        section_breakup_rates_s=(0.0,) * 8,
    )
    initial = np.zeros(s.width)
    initial[0] = 1.0

    def rhs(_time, coords):
        return s.sources(torch.as_tensor(coords, dtype=torch.float64), 1.0)[0].numpy()

    solution = solve_ivp(rhs, (0.0, 5.0), initial, method="DOP853", rtol=1e-11, atol=1e-13)
    assert solution.success
    tau = 5.0
    q = tau / (1 + tau)
    exact = np.array([q ** (k - 1) / (1 + tau) ** 2 for k in range(1, 8)])
    tail_number = q**7 / (1 + tau)
    tail_volume = 1.0 - np.dot(np.arange(1, 8), exact)
    np.testing.assert_allclose(solution.y[:-2, -1], exact, rtol=2e-8, atol=1e-11)
    assert solution.y[-2, -1] == pytest.approx(tail_number, rel=2e-8)
    assert solution.y[-1, -1] == pytest.approx(tail_volume, rel=2e-8)
    assert tail_volume > 0.5
    assert solution.y[:-1, -1].sum() == pytest.approx(1 / 6, rel=1e-9)
    assert np.arange(1, 8) @ solution.y[:-2, -1] + solution.y[-1, -1] == pytest.approx(
        1.0, abs=1e-10
    )


def test_multifragment_distribution_preserves_volume_and_expected_particle_gain():
    s = replace(
        spec(),
        daughter_volume_fractions=(0.25, 0.25, 0.5),
        section_breakup_rates_s=(0.0, 0.0, 0.2, 0.3),
    )
    c = torch.tensor([1e9, 2e9, 3e9, 4e8, 4e8 * 8e-15 * 2], dtype=torch.float64)
    _, split = s.sources(c, 1e-4)
    assert abs(float(reference_moment(s, split))) < 1e-18
    assert float(split[:-1].sum()) == pytest.approx(
        2 * float(c[:-1] @ c.new_tensor(s.section_breakup_rates_s)), rel=1e-12
    )
    with pytest.raises(ValueError, match="Daughters"):
        replace(s, section_breakup_rates_s=(0.1, 0.1, 0.2, 0.3))


def test_live_brownian_and_gravitational_collision_kernel_matches_independent_pairs():
    from dataclasses import asdict

    from reference_pr_sections import sources as reference_sources

    from surface.pr_sectional_population import source_context

    m, state, _ = setup(kernel=0.0, rates=(0.0, 0.0, 0.0, 0.0), power=0.0, vertical=True)
    d = m._state_data(0, state.nodes[0][1])
    p = d["population"]
    s = replace(
        m.nodes[0].droplet_population,
        coalescence_model="brownian_gravity",
        collision_efficiency=0.4,
    )
    context = source_context(m.nodes[0].drift_flux, p, d["t"])
    coal, split = s.sources(p["coordinates"], m.nodes[0].volume_m3, **context)
    npop = {k: v.detach().numpy() if isinstance(v, torch.Tensor) else v for k, v in p.items()}
    expected, _ = reference_sources(
        asdict(s),
        p["coordinates"].numpy(),
        m.nodes[0].volume_m3,
        npop,
        float(d["t"]),
        asdict(m.nodes[0].drift_flux),
    )
    torch.testing.assert_close(
        coal, torch.tensor(expected, dtype=coal.dtype), rtol=1e-10, atol=1e-15
    )
    assert abs(float(reference_moment(s, coal))) < 1e-18
    assert torch.equal(split, torch.zeros_like(split))
    with pytest.raises(ValueError, match="live"):
        s.sources(p["coordinates"], m.nodes[0].volume_m3)
