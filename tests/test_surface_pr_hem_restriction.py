"""Critical PR restriction: coupled PS throat, balances and real-gas limits."""

import json

import pytest
import torch
from test_surface_compositional_separator import binary

from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork
from surface.pr_hem_restriction import PRHEMRestriction


def setup(
    backpressure=6e5, temperature=320.0, dense_limit=128, reverse=False, composition=(0.5, 0.5)
):
    f = binary()
    model = PREquilibriumTransportNetwork(
        f,
        [PREquilibriumNode("a", 0.01), PREquilibriumNode("b", 0.01)],
        [],
        restrictions=[PRHEMRestriction("valve", "a", "b", 1e-7, 10.0, 1e5)],
        dense_limit=dense_limit,
    )
    initial = {
        n: {"pressure_pa": p, "temperature_k": temperature, "mole_fractions": list(composition)}
        for n, p in zip(model.names, [backpressure, 1e6] if reverse else [1e6, backpressure])
    }
    return model, model.initialize(initial), initial


@pytest.mark.parametrize("dense_limit", [1, 128])
@pytest.mark.parametrize("backpressure", [8e5, 2e5])
def test_subcritical_and_critical_coupled_flow(backpressure, dense_limit):
    m, s, _ = setup(backpressure=backpressure, dense_limit=dense_limit)
    end, r = m._step(s, 1e-5, m.default_openings, [0.0, 0.0], [320.0, 320.0])
    q = r["restrictions"]["valve"]
    assert q["molar_rate_mol_s"] > 0
    assert q["choked"] == (backpressure == 2e5)
    assert max(map(abs, r["conservation_residuals"])) < 1e-7
    assert q["throat_pressure_pa"] >= dict(end.nodes)["b"].pressure_pa - 1e-3
    assert abs(q["energy_residual_j_mol"]) < 1e-6
    if q["choked"]:
        assert abs(q["sonic_criterion"]) < 1e-8
    else:
        assert q["throat_pressure_pa"] == pytest.approx(dict(end.nodes)["b"].pressure_pa, abs=1e-3)
    m.inspect(end)


@pytest.mark.parametrize("temperature", [196.2, 320.0])
def test_critical_restriction_jacobian_matches_directional_difference(temperature):
    m, s, _ = setup(backpressure=2e5, temperature=temperature)
    seeds = [m._seed(i, v) for i, (_, v) in enumerate(s.nodes)]
    y, _, _, evaluate, offsets, *_ = m._problem(s, seeds, 1e-5, m.default_openings, [0.0, 0.0])
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype)
    eps = 1e-4
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=1e-6, atol=1e-9)
    support = set(zip(*m._sparsity(offsets, m.default_openings))) | evaluate.additional_sparsity
    mask = torch.zeros_like(jac, dtype=torch.bool)
    for row, col in support:
        mask[row, col] = True
    assert not bool((jac[~mask].abs() > 1e-12).any())


@pytest.mark.parametrize("temperature", [196.2, 320.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_two_phase_or_reversed_throat_conserves_components_and_energy(temperature, reverse):
    m, s, _ = setup(backpressure=6e5, temperature=temperature, reverse=reverse)
    _, r = m._step(s, 1e-5, m.default_openings, [0.0, 0.0], [temperature, temperature])
    row = r["restrictions"]["valve"]
    assert (row["molar_rate_mol_s"] < 0) == reverse
    assert max(map(abs, r["conservation_residuals"])) < 1e-7
    assert abs(row["energy_residual_j_mol"]) < 1e-6


def test_disabled_restriction_has_no_material_or_energy_transfer():
    m, s, _ = setup(backpressure=2e5)
    end, r = m._step(s, 1e-5, {"valve": False}, [0.0, 0.0], [320.0, 320.0])
    assert r["restrictions"]["valve"]["mass_rate_kg_s"] == 0.0
    for (_, old), (_, new) in zip(s.nodes, end.nodes):
        assert new.component_moles == pytest.approx(old.component_moles, abs=1e-12)
        assert new.total_energy_j == pytest.approx(old.total_energy_j, abs=1e-7)


@pytest.fixture(autouse=True)
def script_path(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.mark.parametrize("temperature", [196.2, 320.0])
@pytest.mark.parametrize("backpressure", [8e5, 2e5])
def test_cli_independent_ps_and_critical_derivative_audit(temperature, backpressure):
    import copy

    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    m, _, initial = setup(temperature=temperature, backpressure=backpressure)
    cfg = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [
                    {"duration_s": 1e-5},
                    {"duration_s": 1e-5, "restriction_openings": {"valve": 0.3}},
                    {"duration_s": 1e-5, "openings": {"valve": False}},
                ],
            }
        )
    )
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]
    damaged = copy.deepcopy(result)
    damaged["intervals"][0]["steps"][0]["restrictions"]["valve"][
        "isentropic_volume_derivative"
    ] *= 1.01
    with pytest.raises(ValueError):
        audit(cfg, damaged)


@pytest.mark.parametrize("temperature", [196.2, 320.0])
def test_critical_capacity_agrees_with_separate_numpy_pressure_search(temperature):
    from surface.mixture_nozzle import MixtureEquilibriumNozzle
    from surface.peng_robinson_mixture import PengRobinsonMixture

    m, s, _ = setup(temperature=temperature, backpressure=4e5)
    end, report = m._step(s, 1e-5, m.default_openings, [0.0, 0.0], [temperature, temperature])
    source = dict(end.nodes)["a"]
    receiver = dict(end.nodes)["b"]
    ref = PengRobinsonMixture(m.fluid.components, m.fluid.binary_interactions)
    flow = MixtureEquilibriumNozzle("independent", 1e-7, 3e5, scan_points=13).run(
        source.pressure_pa,
        source.temperature_k,
        receiver.pressure_pa,
        [v / sum(source.component_moles) for v in source.component_moles],
        ref,
        with_outlet=False,
    )
    actual = report["restrictions"]["valve"]
    assert actual["mass_rate_kg_s"] == pytest.approx(flow["mass_rate_kg_s"], rel=2e-6)
    assert actual["throat_pressure_pa"] == pytest.approx(flow["throat"]["pressure_pa"], rel=2e-5)


def test_critical_valve_actuator_and_controller_are_part_of_common_checkpoint():
    from check_pr_controlled_surface import audit
    from run_pr_controlled_surface import run

    from surface.actuator import ConnectionActuator
    from surface.pr_gravity_transport import PRGravityTransportNetwork
    from surface.pr_surface_control import PRControlledSurface, PRSurfacePI, PRThroatControlValve

    transport, _, initial = setup(backpressure=2e5)
    plant = PRGravityTransportNetwork(transport, {}, [])
    model = PRControlledSurface(
        plant,
        [
            PRThroatControlValve(
                "actuator", "valve", ConnectionActuator(1.0, 1.0, 0.0, time_constant_s=0.01)
            )
        ],
        [PRSurfacePI("actuator", 2e5, 1e-6, measurement="pressure", action=-1)],
        initial_step_s=1e-5,
        maximum_step_s=1e-5,
    )
    cfg = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {
                    "plant": {"transport": initial, "columns": {}, "wall_temperatures": {}},
                    "positions": {"actuator": 0.7},
                    "integrals": {},
                },
                "schedule": [
                    {"duration_s": 1e-5},
                    {"duration_s": 1e-5, "targets": {"actuator": 0.3}},
                    {"duration_s": 1e-5, "actuator_modes": {"actuator": "stuck"}},
                ],
            }
        )
    )
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]
    row = result["intervals"][0]["steps"][0]
    assert row["effective_valve_K"]["actuator"] is None
    assert row["effective_throat_openings"]["actuator"] > 0
    assert row["valve_envelope"]["actuator"]["choked_substeps"] > 0
    restart = {
        **cfg,
        "initial": {"state": result["intervals"][-1]["state"]},
        "schedule": [{"duration_s": 1e-5}],
    }
    resumed = json.loads(json.dumps(run(restart)))
    assert audit(restart, resumed)["accepted"]


def test_flashing_liquid_capacity_at_phase_onset_has_one_sided_critical_condition():
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    m, s, initial = setup(temperature=200.0, backpressure=1e5, composition=(0.0, 1.0))
    _, r = m._step(s, 1e-5, m.default_openings, [0.0, 0.0], [200.0, 200.0])
    point = r["restrictions"]["valve"]
    assert point["critical_phase_boundary"]
    assert point["phase_fractions"] == pytest.approx([1.0, 0.0], abs=1e-12)
    assert point["parent_sonic_criterion"] > 0 and point["sonic_criterion"] < 0
    assert 2e5 < point["throat_pressure_pa"] < 2.3e5
    cfg = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [{"duration_s": 1e-5}],
            }
        )
    )
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]


@pytest.mark.parametrize("reverse", [False, True])
def test_new_upstream_component_reaches_thermodynamic_throat(reverse):
    from dataclasses import asdict

    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    from surface.pr_equilibrium_transport import (
        PREquilibriumLink,
        PREquilibriumNode,
        PREquilibriumTransportNetwork,
    )

    base, _, initial = setup(
        backpressure=2e5, temperature=320.0, reverse=reverse, composition=(0.0, 1.0)
    )
    data = base.to_dict()
    data["nodes"].append(asdict(PREquilibriumNode("feed", 0.01)))
    donor = "b" if reverse else "a"
    data["links"].append(asdict(PREquilibriumLink("inlet", "feed", donor, 0.001)))
    initial["feed"] = {"pressure_pa": 1.2e6, "temperature_k": 320.0, "mole_fractions": [1.0, 0.0]}
    model = PREquilibriumTransportNetwork.from_dict(data)
    cfg = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [{"duration_s": 1e-5}],
            }
        )
    )
    result = json.loads(json.dumps(run(cfg)))
    throat = result["intervals"][0]["steps"][0]["restrictions"]["valve"]
    assert throat["phase_mole_fractions"][0][0] > 0
    assert audit(cfg, result)["accepted"]


@pytest.mark.parametrize("backpressure,choked", [(2.1e5, True), (2.2e5, False)])
def test_liquid_restriction_crosses_flashing_onset_and_recovers_subcritical_branch(
    backpressure, choked
):
    m, s, _ = setup(temperature=200.0, backpressure=backpressure, composition=(0.0, 1.0))
    _, r = m._step(s, 1e-6, m.default_openings, [0.0, 0.0], [200.0, 200.0])
    row = r["restrictions"]["valve"]
    assert row["choked"] is choked
    assert row["critical_phase_boundary"] is choked
    assert row["mass_rate_kg_s"] > 0


def test_three_phase_gas_oil_water_throat_and_independent_thermodynamic_tangent():
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run
    from test_surface_compositional_separator import ternary

    from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork

    f = ternary()
    model = PREquilibriumTransportNetwork(
        f,
        [PREquilibriumNode("a", 0.01), PREquilibriumNode("b", 0.01)],
        [],
        restrictions=[PRHEMRestriction("valve", "a", "b", 1e-7, 10.0, 1e5)],
    )
    initial = {
        n: {"pressure_pa": p, "temperature_k": 350.0, "mole_fractions": [0.4, 0.3, 0.3]}
        for n, p in (("a", 1e6), ("b", 2e5))
    }
    cfg = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [{"duration_s": 1e-5}],
            }
        )
    )
    result = json.loads(json.dumps(run(cfg)))
    throat = result["intervals"][0]["steps"][0]["restrictions"]["valve"]
    assert len(throat["phase_fractions"]) == 3
    assert throat["choked"] and throat["mass_rate_kg_s"] > 0
    assert audit(cfg, result)["accepted"]


def test_column_level_controls_downstream_critical_valve_through_finite_piping():
    import math
    from dataclasses import asdict

    from check_pr_controlled_surface import audit
    from run_pr_controlled_surface import run

    from surface.actuator import ConnectionActuator
    from surface.pr_equilibrium_transport import (
        PREquilibriumLink,
        PREquilibriumNode,
        PREquilibriumTransportNetwork,
    )
    from surface.pr_gravity_transport import PRColumnPort, PRGravityTransportNetwork
    from surface.pr_surface_control import PRControlledSurface, PRSurfacePI, PRThroatControlValve
    from surface.separator_geometry import SeparatorGeometry
    from surface.stratified_pr import StratifiedPRColumn

    base, _, initial = setup(temperature=196.2, backpressure=2e5)
    data = base.to_dict()
    data["gravity_m_s2"] = 9.80665
    data["nodes"] = [{**n, "elevation_m": 0.05} for n in data["nodes"]]
    data["nodes"].append(
        asdict(
            PREquilibriumNode(
                "draw",
                math.pi * 0.001**2 / 4 * 0.1,
                "pipe_cell",
                elevation_m=0.05,
                length_m=0.1,
                diameter_m=0.001,
            )
        )
    )
    data["links"].append(asdict(PREquilibriumLink("suction", "draw", "a", 0.001)))
    transport = PREquilibriumTransportNetwork.from_dict(data)
    initial["draw"] = {**initial["a"], "velocity_m_s": 0.0}
    column = StratifiedPRColumn(
        transport.fluid,
        SeparatorGeometry(0.1, 0.1, "vertical"),
        quadrature_order=2,
        maximum_quadrature_order=8,
    )
    plant = PRGravityTransportNetwork(
        transport,
        {"separator": column},
        [PRColumnPort("draw_port", "separator", "draw", 0.05, 0.001)],
    )
    controlled = PRControlledSurface(
        plant,
        [
            PRThroatControlValve(
                "actuator", "valve", ConnectionActuator(1.0, 1.0, 0.0, time_constant_s=0.01)
            )
        ],
        [
            PRSurfacePI(
                "actuator",
                0.01,
                10.0,
                measurement="level",
                component=transport.fluid.names[1],
                sensor="separator",
            )
        ],
        initial_step_s=1e-7,
        maximum_step_s=1e-5,
    )
    cfg = json.loads(
        json.dumps(
            {
                "model": controlled.to_dict(),
                "initial": {
                    "plant": {
                        "transport": initial,
                        "columns": {
                            "separator": {
                                "pressure_pa": 1e6,
                                "temperature_k": 196.2,
                                "mole_fractions": [0.5, 0.5],
                            }
                        },
                        "wall_temperatures": {},
                    },
                    "positions": {"actuator": 0.5},
                    "integrals": {},
                },
                "schedule": [{"duration_s": 1e-7}],
            }
        )
    )
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]
    step = result["intervals"][0]["steps"][0]
    assert step["measurements"]["actuator"] > 0
    assert any(
        abs(x) > 0 for x in step["plant"]["steps"][0]["columns"]["ports"]["draw_port"]["flux"][:-2]
    )
    assert step["plant"]["steps"][0]["transport"]["restrictions"]["valve"]["mass_rate_kg_s"] > 0


def test_compositional_phase_onset_maximum_uses_three_phase_and_two_phase_tangents():
    from pathlib import Path

    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    cfg = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/surface_pr_hem_phase_boundary.json"
        ).read_text()
    )
    result = json.loads(json.dumps(run(cfg)))
    throat = result["intervals"][0]["steps"][0]["restrictions"]["valve"]
    assert throat["critical_phase_boundary"] and throat["phase_boundary_birth"]
    assert len(throat["phase_fractions"]) == 3
    assert min(throat["phase_fractions"]) < 1e-9
    assert throat["parent_sonic_criterion"] > 0 and throat["sonic_criterion"] < 0
    assert throat["throat_pressure_pa"] == pytest.approx(6e5, rel=1e-4)
    assert audit(cfg, result)["accepted"]


def test_equal_pressure_open_throat_has_zero_flux_and_preserves_stocks():
    m, old, _ = setup(backpressure=1e6)
    new, report = m._step(old, 1e-5, m.default_openings, [0.0, 0.0], [320.0, 320.0])
    throat = report["restrictions"]["valve"]
    assert throat["enabled"] and not throat["choked"]
    assert throat["molar_rate_mol_s"] == pytest.approx(0.0, abs=1e-12)
    assert throat["throat_velocity_m_s"] == pytest.approx(0.0, abs=1e-10)
    for (_, before), (_, after) in zip(old.nodes, new.nodes):
        assert after.component_moles == pytest.approx(before.component_moles, abs=1e-12)
        assert after.total_energy_j == pytest.approx(before.total_energy_j, abs=1e-8)
