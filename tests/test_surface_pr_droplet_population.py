"""Population count, coalescence heat and feedback on physical transport."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_surface_oil_water_drift import three_phase_pipe

from surface.pr_droplet_population import PRDropletPopulation
from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork


def closed(kernel=0.0, vertical=False):
    base, _, initial = three_phase_pipe(interface="immobile")
    node = replace(
        base.nodes[0],
        name="emulsion",
        elevation_m=0.0,
        elevation_change_m=0.1 if vertical else 0.0,
        drift_flux=replace(base.nodes[0].drift_flux, drift_velocity_m_s=0.0),
        droplet_population=PRDropletPopulation("oil_drops", kernel),
    )
    model = PREquilibriumTransportNetwork(
        base.fluid, [node], gravity_m_s2=9.80665 if vertical else 0.0
    )
    return model, model.initialize({"emulsion": initial["0"]}), initial["0"]


def test_coalescence_changes_number_and_surface_energy_in_shared_newton():
    zero, old, _ = closed()
    n0 = old.nodes[0][1].droplet_number
    kernel = 2 * zero.nodes[0].volume_m3 / n0
    m, state, _ = closed(kernel)
    before = m._state_data(0, state.nodes[0][1])
    dt = 0.01
    end, report = m._step(state, dt, m.default_openings, [0.0], [350.0])
    actual = end.nodes[0][1].droplet_number
    expected = 2 * n0 / (1 + math.sqrt(1 + 2 * kernel * n0 * dt / m.nodes[0].volume_m3))
    assert actual == pytest.approx(expected, rel=1e-10)
    after = m._state_data(0, end.nodes[0][1])
    assert float(after["population"]["diameter_m"]) > float(before["population"]["diameter_m"])
    assert float(after["surface"]) < float(before["surface"])
    assert end.nodes[0][1].temperature_k > state.nodes[0][1].temperature_k
    assert end.nodes[0][1].total_energy_j == pytest.approx(
        state.nodes[0][1].total_energy_j, abs=1e-8
    )
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
    restored = type(m).from_dict(json.loads(json.dumps(m.to_dict())))
    assert restored.model_id == m.model_id


def test_population_jacobian_includes_surface_energy_and_live_drop_velocity():
    m, state, _ = closed(kernel=1e-15, vertical=True)
    seeds = [m._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, ev, offsets, *_ = m._problem(state, seeds, 1e-6, m.default_openings, [0.0])
    assert len(y) == offsets[-1] + 1
    jac = torch.func.jacfwd(lambda z: ev(z)[0])(y)
    delta = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype)
    eps = 1e-4
    finite = (ev(y + eps * delta)[0] - ev(y - eps * delta)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ delta, finite, rtol=1e-5, atol=1e-7)
    assert bool((jac[: offsets[-1], -1].abs() > 1e-12).any())


def test_coalescence_converges_to_continuous_smoluchowski_solution():
    zero, old, _ = closed()
    n0 = old.nodes[0][1].droplet_number
    kernel = 2 * zero.nodes[0].volume_m3 / n0
    model, start, _ = closed(kernel)
    exact = n0 / 1.2  # dN/dt = -K N^2/(2 V), elapsed 0.2 s.
    errors = []
    for count in (4, 8):
        state = start
        for _ in range(count):
            state, record = model._step(state, 0.2 / count, model.default_openings, [0.0], [350.0])
            assert max(map(abs, record["conservation_residuals"])) < 1e-6
        errors.append(abs(state.nodes[0][1].droplet_number - exact))
        assert state.nodes[0][1].component_moles == pytest.approx(
            start.nodes[0][1].component_moles, abs=1e-10
        )
        assert state.nodes[0][1].total_energy_j == pytest.approx(
            start.nodes[0][1].total_energy_j, abs=1e-7
        )
    assert errors[0] / errors[1] > 1.8
    assert errors[1] / exact < 0.004


@pytest.mark.parametrize("temperature,pressure,phase_count", [(450.0, 1e6, 2), (450.0, 1e5, 1)])
def test_water_rich_vapor_without_oil_drops_has_no_fictitious_particle_stock(
    temperature, pressure, phase_count
):
    model, _, initial = closed()
    state = model.initialize(
        {
            "emulsion": {
                **initial,
                "pressure_pa": pressure,
                "temperature_k": temperature,
            }
        }
    )
    assert len(state.nodes[0][1].phase_roots) == phase_count
    assert state.nodes[0][1].droplet_number == 0.0
    data = model._state_data(0, state.nodes[0][1])
    assert not bool(data["population"]["present"])
    assert float(data["surface"]) == 0.0
    end, report = model._step(state, 1e-3, model.default_openings, [0.0], [temperature])
    assert end.nodes[0][1].droplet_number == 0.0
    assert max(map(abs, report["conservation_residuals"])) < 1e-6


def test_coalescence_and_surface_heat_have_independent_numpy_audit(monkeypatch):
    import copy

    zero, state, initial = closed()
    kernel = 2 * zero.nodes[0].volume_m3 / state.nodes[0][1].droplet_number
    model, _, initial = closed(kernel)
    request = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"nodes": {"emulsion": initial}},
                "schedule": [{"duration_s": 0.001}],
            }
        )
    )
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    report = json.loads(json.dumps(run(request)))
    assert audit(request, report)["accepted"]
    damaged = copy.deepcopy(report)
    damaged["intervals"][0]["steps"][0]["state"]["nodes"]["emulsion"]["droplet_number"] *= 1.01
    with pytest.raises(ValueError):
        audit(request, damaged)


def test_particle_transfer_between_cells_conserves_number_without_collisions():
    base, _, initial = three_phase_pipe(interface="immobile")
    nodes = [
        replace(node, droplet_population=PRDropletPopulation("oil", 0.0)) for node in base.nodes
    ]
    model = PREquilibriumTransportNetwork(
        base.fluid, nodes, base.links, gravity_m_s2=base.gravity_m_s2, dense_limit=1
    )
    state = model.initialize(initial)
    before = sum(s.droplet_number for _, s in state.nodes)
    end, report = model._step(state, 1e-4, model.default_openings, [0.0, 0.0], [350.0, 350.0])
    assert sum(s.droplet_number for _, s in end.nodes) == pytest.approx(before, rel=1e-11)
    rate = report["links"]["internal"]["droplet_number_fluxes"]["oil"]
    assert rate != 0.0
    assert end.nodes[0][1].droplet_number - state.nodes[0][1].droplet_number == pytest.approx(
        -1e-4 * rate, abs=0.1
    )
    assert max(map(abs, report["conservation_residuals"])) < 1e-6


@pytest.mark.parametrize("reverse", [False, True])
def test_finite_collector_captures_particles_or_injects_declared_size_without_losing_energy(
    reverse,
):
    from surface.pr_equilibrium_transport import PREquilibriumLink, PREquilibriumNode

    base, _, initial = closed()
    collector = PREquilibriumNode("collector", 0.001, droplet_capture=True)
    link = PREquilibriumLink("nozzle", "emulsion", "collector", 0.01, source_port="side:nozzle")
    model = PREquilibriumTransportNetwork(base.fluid, [base.nodes[0], collector], [link])
    state = model.initialize(
        {
            "emulsion": initial,
            "collector": {**initial, "pressure_pa": 1010000.0 if reverse else 990000.0},
        }
    )
    end, report = model._step(state, 1e-4, model.default_openings, [0.0, 0.0], [350.0, 350.0])
    rate = report["links"]["nozzle"]["droplet_number_fluxes"]["oil_drops"]
    assert (rate < 0) is reverse
    assert dict(end.nodes)["collector"].droplet_number is None
    assert max(map(abs, report["conservation_residuals"])) < 1e-6
    assert dict(end.nodes)["emulsion"].droplet_number - dict(state.nodes)[
        "emulsion"
    ].droplet_number == pytest.approx(-1e-4 * rate, abs=0.1)


def test_injected_oil_that_dissolves_does_not_create_fictitious_drops():
    from surface.pr_equilibrium_transport import PREquilibriumLink, PREquilibriumNode

    base, _, initial = closed()
    model = PREquilibriumTransportNetwork(
        base.fluid,
        [base.nodes[0], PREquilibriumNode("feed", 0.001, droplet_capture=True)],
        [PREquilibriumLink("feed", "feed", "emulsion", 0.01, target_port="side:feed")],
    )
    fractions = list(initial["mole_fractions"])
    fractions[2] += fractions[1]
    fractions[1] = 0.0
    state = model.initialize(
        {
            "emulsion": {**initial, "mole_fractions": fractions},
            "feed": {**initial, "pressure_pa": 1010000.0},
        }
    )
    assert dict(state.nodes)["emulsion"].droplet_number == 0.0
    end, record = model._step(state, 1e-4, model.default_openings, [0.0, 0.0], [350.0, 350.0])
    assert dict(end.nodes)["emulsion"].component_moles[1] > 0.0
    assert dict(end.nodes)["emulsion"].droplet_number == 0.0
    assert max(map(abs, record["conservation_residuals"])) < 1e-6


def test_oil_phase_birth_in_saturated_carrier_creates_particle_inventory():
    from surface.pr_equilibrium_transport import PREquilibriumLink, PREquilibriumNode

    base, old, initial = closed()
    data = base._state_data(0, old.nodes[0][1])
    gas, _oil, water = data["population"]["roles"][0]
    amount = (0.2 * gas + 0.8 * water) / data["q"]["molar_volume_m3_mol"]
    fractions = (data["x"] * amount[:, None]).sum(0) / amount.sum()
    model = PREquilibriumTransportNetwork(
        base.fluid,
        [base.nodes[0], PREquilibriumNode("feed", 0.001, droplet_capture=True)],
        [PREquilibriumLink("feed", "feed", "emulsion", 0.01, target_port="side:feed")],
    )
    state = model.initialize(
        {
            "emulsion": {**initial, "mole_fractions": fractions.tolist()},
            "feed": {**initial, "pressure_pa": 1010000.0},
        }
    )
    assert dict(state.nodes)["emulsion"].droplet_number == 0.0
    end, record = model._step(state, 1e-4, model.default_openings, [0.0, 0.0], [350.0, 350.0])
    assert len(dict(end.nodes)["emulsion"].phase_roots) == 3
    assert dict(end.nodes)["emulsion"].droplet_number > 0.0
    assert max(map(abs, record["conservation_residuals"])) < 1e-6


def test_particle_capture_and_gravitational_column_share_one_conservative_step():
    from surface.pr_equilibrium_transport import PREquilibriumLink, PREquilibriumNode
    from surface.pr_gravity_transport import PRColumnPort, PRGravityTransportNetwork
    from surface.separator_geometry import SeparatorGeometry
    from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState

    base, _, initial = closed()
    h, diameter, length = 0.05, 0.01, 0.1
    collector = PREquilibriumNode(
        "collector",
        math.pi * diameter**2 / 4 * length,
        "pipe_cell",
        elevation_m=h,
        length_m=length,
        diameter_m=diameter,
        droplet_capture=True,
    )
    transport = PREquilibriumTransportNetwork(
        base.fluid,
        [replace(base.nodes[0], elevation_m=h), collector],
        [
            PREquilibriumLink(
                "capture", "emulsion", "collector", diameter, source_port="side:capture"
            )
        ],
        gravity_m_s2=9.80665,
        dense_limit=1,
    )
    column = StratifiedPRColumn(
        base.fluid,
        SeparatorGeometry(0.05, 0.1, "vertical"),
        quadrature_order=2,
        maximum_quadrature_order=8,
    )
    cs = StratifiedPRState.from_dict(
        column.initialize(1e6, 350.0, initial["mole_fractions"])["state"]
    )
    model = PRGravityTransportNetwork(
        transport,
        {"separator": column},
        [PRColumnPort("outlet", "separator", "collector", h, diameter, pipe_port="right")],
    )
    ts = transport.initialize(
        {"emulsion": {**initial, "pressure_pa": 1010000.0}, "collector": initial}
    )
    state = model.initialize(ts, {"separator": cs})
    end, record = model._step(
        state, 1e-7, {**transport.default_openings, "outlet": True}, {}, {}, {}
    )
    assert record["transport"]["links"]["capture"]["droplet_number_fluxes"]["oil_drops"] > 0
    assert (
        dict(end.transport.nodes)["emulsion"].droplet_number
        < dict(ts.nodes)["emulsion"].droplet_number
    )
    assert max(map(abs, record["conservation_residuals"])) < 1e-6
