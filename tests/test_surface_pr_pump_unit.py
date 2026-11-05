"""Existing liquid pump cards coupled to the same finite PR Newton/ledger."""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_surface_compositional_separator import ternary

from surface.curves import EquipmentCurve
from surface.liquid import Pump
from surface.pr_equilibrium_transport import (
    PREquilibriumLink,
    PREquilibriumNode,
    PREquilibriumTransportNetwork,
)
from surface.pr_pump_unit import PRPumpUnit


def setup(tabulated=False, dense_limit=128, velocity=0.0):
    fluid = ternary()
    diameter, length = 0.02, 0.1
    nodes = [
        PREquilibriumNode("feed", 0.01),
        PREquilibriumNode("return", 0.01),
        PREquilibriumNode(
            "body",
            math.pi * diameter**2 / 4 * length,
            "pipe_cell",
            length_m=length,
            diameter_m=diameter,
        ),
    ]
    links = [
        PREquilibriumLink("inlet", "feed", "body", diameter),
        PREquilibriumLink("outlet", "body", "return", diameter),
    ]
    card = Pump(
        "booster",
        "feed",
        "return",
        5.0,
        1e8,
        1e-4,
        efficiency=0.8,
        head_curve=EquipmentCurve((0.0, 5e-5, 1e-4), (5.0, 4.8, 4.0)) if tabulated else None,
        efficiency_curve=EquipmentCurve((0.0, 5e-5, 1e-4), (0.7, 0.8, 0.75)) if tabulated else None,
    )
    model = PREquilibriumTransportNetwork(
        fluid, nodes, links, pumps=[PRPumpUnit("body", card)], dense_limit=dense_limit
    )
    initial = {
        n.name: {
            "pressure_pa": 1e6,
            "temperature_k": 350.0,
            "mole_fractions": [0.0, 1.0, 0.0],
            **({"velocity_m_s": velocity} if n.kind == "pipe_cell" else {}),
        }
        for n in nodes
    }
    return model, model.initialize(initial), initial


@pytest.mark.parametrize("tabulated", [False, True])
@pytest.mark.parametrize("dense_limit", [1, 128])
def test_pump_force_and_work_match_existing_card_and_conserve(tabulated, dense_limit):
    model, old, _ = setup(tabulated, dense_limit)
    new, report = model._step(old, 1e-4, model.default_openings, [0.0] * 3, [350.0] * 3)
    row = report["pumps"]["booster"]
    body = model._state_data(model.names.index("body"), dict(new.nodes)["body"])
    fluid = SimpleNamespace(density_kg_m3=float(body["rho"]))
    card = model.pumps[0].card
    loss, _ = card.loss_and_gradient(row["flow_m3_s"], fluid)
    assert row["force_n"] == pytest.approx(
        -loss * model.nodes[model.names.index("body")].face_area("left"), rel=1e-12
    )
    assert row["shaft_power_w"] == pytest.approx(
        -loss * row["flow_m3_s"] / card.efficiency_at(row["flow_m3_s"]), rel=1e-12
    )
    assert report["external_heat_j"] == 0
    assert report["external_work_j"] == pytest.approx(1e-4 * row["shaft_power_w"], rel=1e-12)
    assert report["external_work_j"] > 0
    assert report["entropy_production_j_k"] >= 0
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
    assert report["external_work_j"] / max(
        1e-30,
        abs(
            sum(v.total_energy_j for _, v in new.nodes)
            - sum(v.total_energy_j for _, v in old.nodes)
        ),
    ) == pytest.approx(1.0, rel=1e-3)
    restored = type(model).from_dict(json.loads(json.dumps(model.to_dict())))
    restored.inspect(new)


def test_disabled_pump_keeps_inventory_and_passes_reverse_flow_passively():
    model, old, _ = setup(velocity=-0.03)
    commands = {**model.default_openings, "booster": False}
    new, r = model._step(old, 1e-6, commands, [0.0] * 3, [350.0] * 3)
    row = r["pumps"]["booster"]
    assert row["enabled"] is False
    assert row["flow_m3_s"] < 0 and row["force_n"] > 0
    assert r["external_work_j"] == 0
    assert max(map(abs, r["conservation_residuals"])) < 1e-7
    assert dict(new.nodes)["body"].component_moles[0] == 0


def test_pump_map_is_in_full_automatic_jacobian_without_cpu_card_calls(monkeypatch):
    m, old, _ = setup(tabulated=True, velocity=0.05)
    seeds = [m._seed(i, body) for i, (_, body) in enumerate(old.nodes)]
    y, _, _, ev, _, _, _ = m._problem(old, seeds, 1e-5, m.default_openings, [0.0] * 3)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU equipment card called inside Torch Newton")

    monkeypatch.setattr(Pump, "loss_and_gradient", forbidden)
    jac = torch.func.jacfwd(lambda z: ev(z)[0])(y)
    direction = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype)
    eps = 1e-5
    fd = (ev(y + eps * direction)[0] - ev(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=1e-5, atol=1e-8)


def test_unsupported_check_valve_and_unqualified_npsh_are_explicit():
    m, _, _ = setup()
    unit = m.pumps[0]
    with pytest.raises(ValueError, match="check valve"):
        replace(unit, card=replace(unit.card, check_valve=True))
    card = replace(unit.card, npsh_required_m=1.0, inlet_diameter_m=0.02)
    with pytest.raises(ValueError, match="vapor pressure"):
        replace(unit, card=card)


def test_pump_operating_limits_are_checked_on_the_actual_pr_state():
    m, state, _ = setup()
    data = [m._state_data(i, v) for i, (_, v) in enumerate(state.nodes)]
    cell = m._pump_cells[0]
    for flow, valid in [(0.0, True), (-0.1, False), (1.0, False)]:
        trial = [dict(d) for d in data]
        trial[cell.index]["u"] = m.fluid.tensor(flow)
        assert bool(cell.evaluate(trial, True)[-1]) is valid
    unit = m.pumps[0]
    card = replace(unit.card, npsh_required_m=200.0, inlet_diameter_m=0.02)
    qualified = replace(unit, card=card, vapor_pressure_pa=1e4, vapor_pressure_temperature_k=350.0)
    from surface.pr_pump_unit import TorchPumpCell

    protected = TorchPumpCell(m, qualified)
    assert not bool(protected.evaluate(data, True)[-1])
    assert bool(protected.evaluate(data, False)[-1])
    # A finite plenum stores static (= total at u=0) pressure. Do not add
    # fictitious q/A kinetic head to that upstream reservoir a second time.
    for velocity in (0.0, 0.05):
        trial = [dict(d) for d in data]
        trial[cell.index]["u"] = m.fluid.tensor(velocity)
        npsh = protected.evaluate(trial, True)[2]["npsh_available_m"]
        expected = (trial[cell.inlet]["p"] - 1e4) / (trial[cell.inlet]["rho"] * 9.80665)
        torch.testing.assert_close(npsh, expected, rtol=1e-14, atol=1e-14)


def test_pump_work_has_an_independent_audit_and_is_not_relabelled_as_heat(monkeypatch):
    import copy
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    m, _, initial = setup(tabulated=True)
    config = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [
                    {"duration_s": 1e-5},
                    {"duration_s": 1e-5, "openings": {"booster": False}},
                    {"duration_s": 1e-5},
                ],
            }
        )
    )
    report = json.loads(json.dumps(run(config)))
    assert audit(config, report)["accepted"]
    assert report["total_work_j"] > 0
    assert report["total_heat_j"] == 0
    assert report["intervals"][1]["total_work_j"] == 0
    corrupted = copy.deepcopy(report)
    corrupted["intervals"][0]["steps"][0]["external_work_j"] += 0.01
    with pytest.raises(ValueError):
        audit(config, corrupted)


def test_pump_and_gravitational_column_share_the_same_energy_ledger(monkeypatch):
    from pathlib import Path

    from surface.pr_gravity_transport import PRGravityTransportNetwork
    from surface.pr_heat_link import PRHeatLink
    from surface.separator_geometry import SeparatorGeometry
    from surface.stratified_pr import StratifiedPRColumn

    base, _, initial = setup()
    config = base.to_dict()
    config["gravity_m_s2"] = 9.80665
    transport = type(base).from_dict(config)
    col = StratifiedPRColumn(
        transport.fluid,
        SeparatorGeometry(0.1, 0.1, "vertical"),
        quadrature_order=2,
        maximum_quadrature_order=8,
    )
    m = PRGravityTransportNetwork(
        transport,
        {"separator": col},
        [],
        heat_links=[PRHeatLink("cooler", "body", "separator", 1.0)],
    )
    config = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": {
                    "transport": initial,
                    "columns": {
                        "separator": {
                            "pressure_pa": 1e6,
                            "temperature_k": 349.0,
                            "mole_fractions": [0.0, 1.0, 0.0],
                        }
                    },
                    "wall_temperatures": {},
                },
                "schedule": [{"duration_s": 1e-5}],
            }
        )
    )
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_gravity_transport import audit
    from run_pr_gravity_transport import run

    report = json.loads(json.dumps(run(config)))
    assert audit(config, report)["accepted"]
    assert report["total_heat_j"] == 0
    assert report["total_work_j"] > 0
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


def test_control_wrapper_preserves_pump_work_and_motor_switching(monkeypatch):
    from pathlib import Path

    from surface.actuator import ConnectionActuator
    from surface.pr_gravity_transport import PRColumnPort, PRGravityTransportNetwork
    from surface.pr_surface_control import PRControlledSurface, PRControlValve
    from surface.separator_geometry import SeparatorGeometry
    from surface.stratified_pr import StratifiedPRColumn

    base, _, initial = setup()
    config = base.to_dict()
    config["gravity_m_s2"] = 9.80665
    node = PREquilibriumNode(
        "valve_pipe",
        math.pi * 0.0005**2 / 4 * 0.1,
        "pipe_cell",
        elevation_m=0.05,
        length_m=0.1,
        diameter_m=0.0005,
    )
    from dataclasses import asdict

    config["nodes"].append(asdict(node))
    transport = type(base).from_dict(config)
    column = StratifiedPRColumn(
        transport.fluid,
        SeparatorGeometry(0.1, 0.1, "vertical"),
        quadrature_order=2,
        maximum_quadrature_order=8,
    )
    plant = PRGravityTransportNetwork(
        transport,
        {"separator": column},
        [PRColumnPort("port", "separator", "valve_pipe", 0.05, 0.0005)],
    )
    valve = PRControlValve("v", "port", 1.0, ConnectionActuator(1.0, 1.0, 0.0, time_constant_s=0.1))
    model = PRControlledSurface(
        plant, [valve], initial_step_s=1e-5, minimum_step_s=1e-10, maximum_step_s=1e-5
    )
    initial["valve_pipe"] = {
        "pressure_pa": 1e6,
        "temperature_k": 350.0,
        "mole_fractions": [0.0, 1.0, 0.0],
    }
    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {
                    "plant": {
                        "transport": initial,
                        "columns": {
                            "separator": {
                                "pressure_pa": 1e6,
                                "temperature_k": 350.0,
                                "mole_fractions": [0.0, 1.0, 0.0],
                            }
                        },
                        "wall_temperatures": {},
                    },
                    "positions": {"v": 0.5},
                    "integrals": {},
                },
                "schedule": [
                    {"duration_s": 1e-5, "targets": {"v": 0.5}},
                    {"duration_s": 1e-5, "openings": {"booster": False}, "targets": {"v": 0.5}},
                ],
            }
        )
    )
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_controlled_surface import audit
    from run_pr_controlled_surface import run

    report = json.loads(json.dumps(run(config)))
    assert audit(config, report)["accepted"]
    assert report["total_work_j"] > 0
    assert report["intervals"][1]["total_work_j"] == 0
