"""Mapped gas compressor and PR PS/PH traces participate in common Newton."""

import json

import pytest
import torch
from test_surface_compositional_separator import binary

from surface.compressor_map import CompressorMap
from surface.pr_compressor_unit import PRCompressorUnit
from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork


def setup(pressure=1000.0, ratio=1.5, dense_limit=128, different_target=False):
    card = CompressorMap(
        (100.0, 200.0),
        (0.0, 0.5, 1.0),
        ((0.005, 0.01, 0.015), (0.015, 0.03, 0.045)),
        ((1.6, 1.4, 1.2), (2.2, 1.8, 1.4)),
        ((0.75, 0.8, 0.75), (0.75, 0.8, 0.75)),
    )
    unit = PRCompressorUnit("compressor", "suction", "delivery", card, 150.0, (1.0, 0.0))
    m = PREquilibriumTransportNetwork(
        binary(),
        [PREquilibriumNode("suction", 0.01), PREquilibriumNode("delivery", 0.01)],
        compressors=[unit],
        dense_limit=dense_limit,
    )
    initial = {
        "suction": {"pressure_pa": pressure, "temperature_k": 350.0, "mole_fractions": [1.0, 0.0]},
        "delivery": {
            "pressure_pa": pressure * ratio,
            "temperature_k": 350.0,
            "mole_fractions": [0.0, 1.0] if different_target else [1.0, 0.0],
        },
    }
    return m, m.initialize(initial), initial


@pytest.mark.parametrize("dense_limit", [1, 128])
@pytest.mark.parametrize("ratio", [1.1, 1.5])
def test_compressor_pr_energy_conservation_map_choke_and_ideal_limit(dense_limit, ratio):
    m, old, _ = setup(dense_limit=dense_limit, ratio=ratio)
    new, r = m._step(old, 1e-5, m.default_openings, [0.0] * 2, [350.0] * 2)
    point = r["compressors"]["compressor"]
    assert point["enabled"] and point["map_choked"] is (ratio == 1.1)
    assert point["mass_rate_kg_s"] > 0
    assert r["external_heat_j"] == 0
    assert r["external_work_j"] == pytest.approx(point["shaft_power_w"] * 1e-5, rel=1e-12)
    assert max(map(abs, r["conservation_residuals"])) < 1e-8
    assert point["entropy_production_w_k"] >= 0
    inlet = dict(new.nodes)["suction"]
    cp = m.fluid.components[0].ideal_cp_j_mol_k
    from surface.peng_robinson import R

    reference_t = inlet.temperature_k * point["pressure_ratio"] ** (R / cp)
    assert point["isentropic_temperature_k"] == pytest.approx(reference_t, rel=2e-5)
    expected = inlet.temperature_k + (reference_t - inlet.temperature_k) / point["efficiency"]
    assert point["delivery_temperature_k"] == pytest.approx(expected, rel=2e-5)
    assert type(m).from_dict(json.loads(json.dumps(m.to_dict()))).model_id == m.model_id


def test_compressor_jacobian_contains_map_and_real_pr_trace_derivatives(monkeypatch):
    m, old, _ = setup(pressure=1e6)
    seeds = [m._seed(i, s) for i, (_, s) in enumerate(old.nodes)]
    y, _, _, ev, offsets, _, _ = m._problem(old, seeds, 1e-6, m.default_openings, [0.0] * 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU compressor map called inside prepared Torch residual")

    monkeypatch.setattr(CompressorMap, "operating_point", forbidden)
    assert len(y) == offsets[-1] + 3
    jac = torch.func.jacfwd(lambda z: ev(z)[0])(y)
    direction = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype)
    eps = 1e-4
    fd = (ev(y + eps * direction)[0] - ev(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=1e-5, atol=1e-8)
    pattern = m._sparsity(offsets, m.default_openings, compressor_entries=ev.compressors.entries)
    mask = torch.zeros_like(jac, dtype=torch.bool)
    mask[pattern[0], pattern[1]] = True
    assert not bool((jac[~mask].abs() > 1e-12).any())


def test_compressor_isolation_has_no_flux_or_work_and_new_species_flow_only_forward():
    m, old, _ = setup(pressure=1e6, different_target=True)
    off, r = m._step(old, 1e-5, {"compressor": False}, [0.0] * 2, [350.0] * 2)
    assert r["external_work_j"] == 0
    assert dict(off.nodes)["delivery"].component_moles[0] == 0
    new, r = m._step(old, 1e-5, {"compressor": True}, [0.0] * 2, [350.0] * 2)
    assert dict(new.nodes)["delivery"].component_moles[0] > 0
    assert dict(new.nodes)["suction"].component_moles[1] == 0
    assert max(map(abs, r["conservation_residuals"])) < 1e-7


def test_independent_map_ps_ph_and_work_audit(monkeypatch):
    import copy
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    m, _, initial = setup(pressure=1e6)
    config = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [
                    {"duration_s": 1e-6},
                    {"duration_s": 1e-6, "openings": {"compressor": False}},
                ],
            }
        )
    )
    report = json.loads(json.dumps(run(config)))
    assert audit(config, report)["accepted"]
    assert report["total_heat_j"] == 0 and report["total_work_j"] > 0
    damaged = copy.deepcopy(report)
    damaged["intervals"][0]["steps"][0]["compressors"]["compressor"][
        "isentropic_temperature_k"
    ] += 1
    with pytest.raises(ValueError):
        audit(config, damaged)


def test_compressor_auxiliary_rows_are_in_the_sparse_gravity_system(monkeypatch):
    from pathlib import Path

    from surface.pr_gravity_transport import PRGravityTransportNetwork
    from surface.pr_heat_link import PRHeatLink
    from surface.separator_geometry import SeparatorGeometry
    from surface.stratified_pr import StratifiedPRColumn

    base, _, initial = setup(pressure=1e6, dense_limit=1, different_target=True)
    cfg = base.to_dict()
    cfg["gravity_m_s2"] = 9.80665
    transport = type(base).from_dict(cfg)
    column = StratifiedPRColumn(
        transport.fluid,
        SeparatorGeometry(0.1, 0.1, "vertical"),
        quadrature_order=2,
        maximum_quadrature_order=8,
    )
    m = PRGravityTransportNetwork(
        transport,
        {"separator": column},
        [],
        heat_links=[PRHeatLink("cooler", "delivery", "separator", 1.0)],
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
                            "mole_fractions": [0.0, 1.0],
                        }
                    },
                    "wall_temperatures": {},
                },
                "schedule": [{"duration_s": 1e-6}],
            }
        )
    )
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_gravity_transport import audit
    from run_pr_gravity_transport import run

    report = json.loads(json.dumps(run(config)))
    assert audit(config, report)["accepted"]
    assert report["total_heat_j"] == 0 and report["total_work_j"] > 0
    final = report["intervals"][-1]["state"]
    assert final["transport"]["nodes"]["suction"]["component_moles"][1] == 0
    assert final["transport"]["nodes"]["delivery"]["component_moles"][0] > 0
    assert final["columns"]["separator"]["component_moles"][0] == 0


def test_compressor_surge_composition_and_power_domains_are_not_extrapolated():
    from dataclasses import replace

    m, old, _ = setup(pressure=1e6, ratio=2.5)
    seeds = [m._seed(i, s) for i, (_, s) in enumerate(old.nodes)]
    y, _, _, ev, *_ = m._problem(old, seeds, 1e-6, m.default_openings, [0.0] * 2)
    assert not bool(ev(y)[1]["valid"])
    good, state, initial = setup(pressure=1e6)
    wrong_gas = replace(good.compressors[0], reference_mole_fractions=(0.0, 1.0))
    gas_model = PREquilibriumTransportNetwork(good.fluid, good.nodes, compressors=[wrong_gas])
    gas_state = gas_model.initialize(initial)
    gas_seeds = [gas_model._seed(i, s) for i, (_, s) in enumerate(gas_state.nodes)]
    z, _, _, equation, *_ = gas_model._problem(
        gas_state, gas_seeds, 1e-6, gas_model.default_openings, [0.0] * 2
    )
    assert not bool(equation(z)[1]["valid"])
    limited = replace(good.compressors[0], maximum_shaft_power_w=1e-6)
    model = PREquilibriumTransportNetwork(good.fluid, good.nodes, compressors=[limited])
    state = model.initialize(initial)
    seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, ev, *_ = model._problem(state, seeds, 1e-6, model.default_openings, [0.0] * 2)
    assert not bool(ev(y)[1]["valid"])


def test_parallel_compressors_share_plenums_and_sum_component_and_work_fluxes():
    from dataclasses import replace

    base, _, initial = setup(pressure=1e6, dense_limit=1)
    second = replace(base.compressors[0], name="second", speed_rad_s=170.0)
    m = PREquilibriumTransportNetwork(
        base.fluid, base.nodes, compressors=[base.compressors[0], second], dense_limit=1
    )
    old = m.initialize(initial)
    dt = 1e-6
    new, report = m._step(old, dt, m.default_openings, [0.0] * 2, [350.0] * 2)
    points = report["compressors"]
    assert set(points) == {"compressor", "second"}
    assert all(point["mass_rate_kg_s"] > 0 for point in points.values())
    mass = sum(point["mass_rate_kg_s"] for point in points.values()) * dt
    before, after = dict(old.nodes)["suction"], dict(new.nodes)["suction"]
    actual = (before.component_moles[0] - after.component_moles[0]) * base.fluid.components[
        0
    ].molar_mass_kg_mol
    assert actual == pytest.approx(mass, rel=1e-8, abs=1e-12)
    assert report["external_work_j"] == pytest.approx(
        dt * sum(point["shaft_power_w"] for point in points.values()), rel=1e-12
    )
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


def test_closed_compressor_recycle_parallel_return_paths_and_exact_restart():
    """A finite closed circuit gains shaft energy, without a material source."""
    from surface.pr_equilibrium_transport import PREquilibriumTransportState
    from surface.pr_transport_builder import (
        PREquilibriumPipe,
        build_equilibrium_transport,
        initialize_grouped,
    )

    base, _, initial = setup(pressure=1e6)
    model = build_equilibrium_transport(
        base.fluid,
        base.nodes,
        [
            PREquilibriumPipe("return_a", "delivery", "suction", 0.2, 0.002, 2),
            PREquilibriumPipe("return_b", "delivery", "suction", 0.2, 0.002, 2),
        ],
        compressors=base.compressors,
        dense_limit=1,
    )
    for name in model.pipe_paths:
        initial[name] = {
            "pressure_pa": 1.25e6,
            "temperature_k": 350.0,
            "mole_fractions": [1.0, 0.0],
        }
    old = initialize_grouped(model, initial)
    dt = 1e-7
    heat = [0.0] * len(model.nodes)
    ambient = [350.0] * len(model.nodes)
    new, report = model._step(old, dt, model.default_openings, heat, ambient)
    for name in model.pipe_paths:
        assert report["links"][name + "__face_0000"]["flux"][0] > 0
        assert report["links"][name + "__face_0002"]["flux"][0] > 0
    assert report["compressors"]["compressor"]["mass_rate_kg_s"] > 0
    energy_change = sum(s.total_energy_j for _, s in new.nodes) - sum(
        s.total_energy_j for _, s in old.nodes
    )
    assert energy_change == pytest.approx(report["external_work_j"], abs=1e-8)
    for component in range(len(base.fluid.components)):
        assert sum(s.component_moles[component] for _, s in new.nodes) == pytest.approx(
            sum(s.component_moles[component] for _, s in old.nodes), abs=1e-11
        )
    restored_model = type(model).from_dict(json.loads(json.dumps(model.to_dict())))
    restored_state = PREquilibriumTransportState.from_dict(json.loads(json.dumps(new.to_dict())))
    expected, _ = model._step(new, dt, model.default_openings, heat, ambient)
    actual, _ = restored_model._step(
        restored_state, dt, restored_model.default_openings, heat, ambient
    )
    assert actual.to_dict() == expected.to_dict()
