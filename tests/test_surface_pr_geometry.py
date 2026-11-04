"""Imposed PR layers: geometric area, AD, selective transfer and level feedback."""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.compositional_network import CompositionalConnection, CompositionalNetwork
from surface.compositional_separator import CompositionalSeparator, EquilibriumOutlet
from surface.equilibrium_properties import EquilibriumFluid
from surface.peng_robinson import PengRobinsonFluid
from surface.phase_geometry import EquilibriumLevelControl, phase_layout, pickup_fractions
from surface.separator_geometry import SeparatorGeometry

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def binary(device):
    c = json.loads((ROOT / "tests/data/surface/pr_mixture_coolprop.json").read_text())
    return EquilibriumFluid(
        [PengRobinsonFluid(**x) for x in c["components"]], [[0, 0], [0, 0]], device=device
    )


def geometry(orientation="vertical", volume=0.001):
    diameter = 0.1
    return SeparatorGeometry(diameter, 4 * volume / (math.pi * diameter**2), orientation)


def seed_layout(f, g, p=1e6, t=196.2):
    q = f.flash_pt(p, t, [0.19, 0.81])
    n = g.volume_m3 / float(q["molar_volume_m3_mol"].cpu())
    phase_n = torch.stack([x["molar_fraction"] for x in q["phases"]]) * n
    props = {k: torch.stack([x[k] for x in q["phases"]]) for k in q["phases"][0]}
    return q, phase_n, props, phase_layout(phase_n, props, g)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_ordered_layers_and_disk_aperture_against_independent_quadrature(device, orientation):
    g = SeparatorGeometry(1, 2, orientation)
    n = torch.tensor([0.5, 0.2, 0.3], dtype=torch.float64, device=device) * g.volume_m3
    props = {
        "molar_volume_m3_mol": torch.ones_like(n),
        "density_kg_m3": n.new_tensor([10, 1000, 700]),
    }
    layout = phase_layout(n, props, g)
    assert layout["phase_order"].cpu().tolist() == [1, 2, 0]
    expected_top = np.array(
        [g.height_m, g.level_m(0.2 * g.volume_m3), g.level_m(0.5 * g.volume_m3)]
    )
    np.testing.assert_allclose(layout["phase_tops_m"].cpu(), expected_top, atol=1e-11)
    height = expected_top[2]
    diameter = 0.1
    area = pickup_fractions(layout, height, diameter).cpu().numpy()
    bottom = layout["phase_bottoms_m"].cpu().numpy()
    reference = []
    radius = diameter / 2
    for a, b in zip(bottom, expected_top):
        lo = max(a, height - radius)
        hi = min(b, height + radius)
        reference.append(
            quad(
                lambda z: 2 * math.sqrt(max(0, radius**2 - (z - height) ** 2)), lo, hi, epsabs=1e-14
            )[0]
            / (math.pi * radius**2)
            if hi > lo
            else 0
        )
    np.testing.assert_allclose(area, reference, atol=1e-11)
    assert area.sum() == pytest.approx(1, abs=1e-12)
    # Every shape derivative in the coupled residual includes moving interfaces.
    fun = lambda nn: pickup_fractions(phase_layout(nn, props, g), height, diameter)
    ad = torch.func.jacfwd(fun)(n).cpu().numpy()
    fd = []
    for j in range(3):
        d = torch.zeros_like(n)
        d[j] = g.volume_m3 * 1e-7
        fd.append(((fun(n + d) - fun(n - d)) / (2 * d[j])).cpu().numpy())
    np.testing.assert_allclose(ad, np.array(fd).T, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_empty_full_aperture_and_permutation_are_finite(device):
    g = SeparatorGeometry(1, 1, "horizontal")
    for fraction in [1e-10, 1e-6, 0.1, 0.9, 1 - 1e-10]:
        n = torch.tensor([fraction, 1 - fraction], device=device, dtype=torch.float64)
        props = {
            "molar_volume_m3_mol": torch.ones_like(n),
            "density_kg_m3": n.new_tensor([1000, 10]),
        }
        fun = lambda nn, props=props: pickup_fractions(phase_layout(nn, props, g), 0.5, 0.05)
        out = fun(n)
        assert torch.isfinite(out).all() and torch.isfinite(torch.func.jacfwd(fun)(n)).all()
        assert float(out.sum().cpu()) == pytest.approx(1, abs=1e-12)
        flipped = {k: v.flip(0) for k, v in props.items()}
        assert fun(n).cpu().tolist() == pytest.approx(
            pickup_fractions(phase_layout(n.flip(0), flipped, g), 0.5, 0.05).flip(0).cpu().tolist(),
            abs=1e-12,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_real_two_phase_pickup_uses_local_aperture_not_global_holdup(device, monkeypatch):
    f = binary(device)
    g = geometry()
    _q, _pn, _props, layout = seed_layout(f, g)
    i = int(layout["phase_order"][0].cpu())
    top = float(layout["phase_tops_m"][i].cpu())
    port = EquilibriumOutlet(
        "liquid", 0.99e6, 1e-11, height_m=top / 2, diameter_m=min(0.002, top / 3)
    )
    m = CompositionalSeparator(f, g.volume_m3, [port], geometry=g, temporal_error_control=False)
    s = m.initialize(1e6, 196.2, [0.19, 0.81])

    def forbid(*args, **kwargs):
        raise AssertionError("SciPy inverse geometry in Torch runtime")

    monkeypatch.setattr(SeparatorGeometry, "level_m", forbid)
    result = m.advance(s, 0.005)
    row = result["steps"][0]
    picked = row["ports"]["liquid"]
    exposure = np.array(picked["pickup_area_fractions"])
    assert exposure.sum() == pytest.approx(1, abs=1e-12)
    assert exposure[i] == pytest.approx(1, abs=1e-12)
    x = np.array(row["phase_mole_fractions"])[i]
    rate = np.array(picked["component_molar_rates_mol_s"])
    np.testing.assert_allclose(rate / rate.sum(), x, atol=1e-10)
    expected = port.volume_conductance_m3_s_pa * (row["pressure_pa"] - port.back_pressure_pa)
    assert picked["volume_rate_m3_s"] == pytest.approx(expected, rel=1e-10)
    assert row["phase_volume_fractions"][i] < 0.99


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reverse", [False, True])
def test_finite_connected_pickups_switch_actual_donor_layer(device, reverse):
    f = binary(device)
    g = geometry()
    _, _, _, layout = seed_layout(f, g)
    dense = int(layout["phase_order"][0].cpu())
    light = int(layout["phase_order"][-1].cpu())
    low = float(layout["phase_tops_m"][dense].cpu()) / 2
    high = (float(layout["phase_bottoms_m"][light].cpu()) + g.height_m) / 2
    c = CompositionalConnection(
        "ab",
        "a",
        "b",
        1e-11,
        source_height_m=low,
        source_diameter_m=0.001,
        target_height_m=high,
        target_diameter_m=0.001,
    )
    m = CompositionalNetwork(
        {n: CompositionalSeparator(f, g.volume_m3, geometry=g) for n in ["a", "b"]},
        [c],
        temporal_error_control=False,
    )
    s = m.initialize(
        {
            "a": {
                "pressure_pa": 0.99e6 if reverse else 1e6,
                "temperature_k": 196.2,
                "mole_fractions": [0.19, 0.81],
            },
            "b": {
                "pressure_pa": 1e6 if reverse else 0.99e6,
                "temperature_k": 196.2,
                "mole_fractions": [0.19, 0.81],
            },
        }
    )
    r = m.advance(s, 0.005)
    step = r["steps"][0]
    link = step["connections"]["ab"]
    donor = "b" if reverse else "a"
    index = light if reverse else dense
    rate = np.array(link["component_molar_rates_mol_s"])
    assert rate.sum() < 0 if reverse else rate.sum() > 0
    np.testing.assert_allclose(
        rate / rate.sum(), step["vessels"][donor]["phase_mole_fractions"][index], atol=1e-9
    )
    assert max(map(abs, r["component_residuals_mol"])) < 1e-9 and abs(r["energy_residual_j"]) < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_implicit_level_controller_and_coupled_geometric_jacobian(device):
    f = binary(device)
    g = geometry()
    _, _, _, layout = seed_layout(f, g)
    component = f.names[1]
    height = float(layout["phase_tops_m"][0].cpu())
    ctl = EquilibriumLevelControl(component, height, 5, 0.4)
    c = CompositionalConnection(
        "ab",
        "a",
        "b",
        1e-10,
        source_height_m=height,
        source_diameter_m=0.002,
        target_height_m=height,
        target_diameter_m=0.002,
        level_control=ctl,
        control_side="target",
    )
    m = CompositionalNetwork(
        {n: CompositionalSeparator(f, g.volume_m3, geometry=g) for n in ["a", "b"]},
        [c],
        temporal_error_control=False,
    )
    s = m.initialize(
        {
            n: {"pressure_pa": p, "temperature_k": 196.2, "mole_fractions": [0.19, 0.81]}
            for n, p in [("a", 1e6), ("b", 0.99e6)]
        }
    )
    seeds = m.inspect(s)
    states = [v for _, v in s.vessels]
    cmd = m.prepare(s)
    active = m._active(states, [c[0] for c in cmd[0]], cmd[1])
    branches = tuple((q, tuple(range(q["phase_count"]))) for q in seeds)
    problems, _, evaluate, _ = m._problem(states, branches, 0.005, cmd, active)
    y = torch.cat([p["initial"] for p in problems])
    fun = lambda z: evaluate(z)[0]
    ad = torch.func.jacfwd(fun)(y).cpu().numpy()
    fd = []
    for j in range(len(y)):
        d = torch.zeros_like(y)
        d[j] = 2e-6
        fd.append(((fun(y + d) - fun(y - d)) / 4e-6).cpu().numpy())
    np.testing.assert_allclose(ad, np.array(fd).T, rtol=2e-4, atol=5e-7)
    r = m.advance(s, 0.005)
    link = r["steps"][0]["connections"]["ab"]
    assert link["opening"] == pytest.approx(
        np.clip(0.4 + 5 * (link["controlled_level_m"] - height), 0, 1), abs=1e-11
    )
    assert link["controlled_phase_present"]
    overridden = m.advance(s, 0.005, connection_openings={"ab": 0})
    assert overridden["steps"][0]["connections"]["ab"]["opening"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_interface_crosses_finite_aperture_during_drain(device):
    f = binary(device)
    g = geometry()
    _, _, _, layout = seed_layout(f, g)
    dense = int(layout["phase_order"][0].cpu())
    top = float(layout["phase_tops_m"][dense].cpu())
    port = EquilibriumOutlet(
        "draw",
        0.85e6,
        1e-9,
        maximum_pressure_drop_fraction=0.3,
        height_m=top - 0.0001,
        diameter_m=0.002,
    )
    m = CompositionalSeparator(
        f, g.volume_m3, [port], geometry=g, maximum_dt_s=0.002, temporal_error_control=False
    )
    s = m.initialize(1e6, 196.2, [0.19, 0.81])
    r = m.advance(s, 0.02)
    rows = r["steps"]
    fractions = [x["ports"]["draw"]["pickup_area_fractions"][dense] for x in rows]
    assert fractions[0] > 0.5 and fractions[-1] < 0.5
    assert all(0 <= x <= 1 for x in fractions)
    assert all(abs(row["fluid_energy_residual_j"]) < 1e-5 for row in rows)


def test_geometry_configuration_validation():
    f = binary("cpu")
    g = geometry()
    with pytest.raises(ValueError, match="volume"):
        CompositionalSeparator(f, 0.002, geometry=g)
    with pytest.raises(ValueError, match="gravity"):
        CompositionalSeparator(f, g.volume_m3, geometry=g, gravity_m_s2=9.8)
    with pytest.raises(TypeError, match="geometry"):
        CompositionalSeparator(
            f, g.volume_m3, [EquilibriumOutlet("p", 0.99e6, 1e-10, height_m=0.01, diameter_m=0.001)]
        )
    with pytest.raises(ValueError, match="outside"):
        CompositionalSeparator(
            f,
            g.volume_m3,
            [EquilibriumOutlet("p", 0.99e6, 1e-10, height_m=0, diameter_m=0.001)],
            geometry=g,
        )
    with pytest.raises(ValueError, match="pressure or level"):
        EquilibriumOutlet(
            "p",
            0.99e6,
            1e-10,
            pressure_setpoint_pa=1e6,
            level_control=EquilibriumLevelControl(f.names[1], 0.01, 1),
        )
    with pytest.raises(ValueError, match="component"):
        CompositionalSeparator(
            f,
            g.volume_m3,
            [
                EquilibriumOutlet(
                    "p", 0.99e6, 1e-10, level_control=EquilibriumLevelControl("typo", 0.01, 1)
                )
            ],
            geometry=g,
        )
    # Geometry is serialized into the model identity, not a visual-only label.
    m = CompositionalSeparator(f, g.volume_m3, geometry=g)
    assert CompositionalSeparator.from_dict(m.to_dict()).model_id == m.model_id
    m.geometry = replace(g, orientation="horizontal")
    with pytest.raises(ValueError, match="configuration changed"):
        m.inspect(m.initialize(1e6, 320, [0.5, 0.5]))


@pytest.mark.parametrize("device", DEVICES)
def test_absent_control_phase_fails_closed_and_manual_override(device):
    f = binary(device)
    g = geometry()
    control = EquilibriumLevelControl(
        f.names[1], 0.01, 1, opening_bias=1, minimum_mole_fraction=0.99
    )
    port = EquilibriumOutlet(
        "controlled", 0.99e6, 1e-10, height_m=0.05, diameter_m=0.002, level_control=control
    )
    m = CompositionalSeparator(f, g.volume_m3, [port], geometry=g, temporal_error_control=False)
    assert CompositionalSeparator.from_dict(m.to_dict(), device=device).model_id == m.model_id
    s = m.initialize(1e6, 320, [0.5, 0.5])
    r = m.advance(s, 0.005)
    logged = r["steps"][0]["ports"]["controlled"]
    assert not logged["controlled_phase_present"] and logged["opening"] == 0
    assert logged["volume_rate_m3_s"] == 0
    assert r["state"]["inventory"]["component_moles"] == pytest.approx(
        s.inventory.component_moles, abs=1e-12
    )
    opened = m.advance(s, 0.005, openings={"controlled": 1})
    assert opened["steps"][0]["ports"]["controlled"]["volume_rate_m3_s"] > 0


@pytest.mark.parametrize("device", DEVICES)
def test_three_phase_water_pickup_and_layered_checkpoint(device):
    from surface.compositional_network_schedule import run_compositional_network_schedule

    cfg = json.loads((ROOT / "configs/surface_equilibrium_process.json").read_text())
    f = EquilibriumFluid.from_dict(cfg["fluid"], device=device)
    g = geometry("horizontal")
    q = f.flash_pt(1e6, 350, [0.4, 0.3, 0.3])
    total = g.volume_m3 / float(q["molar_volume_m3_mol"].cpu())
    n = torch.stack([p["molar_fraction"] for p in q["phases"]]) * total
    props = {
        k: torch.stack([p[k] for p in q["phases"]])
        for k in ("molar_volume_m3_mol", "density_kg_m3")
    }
    layers = phase_layout(n, props, g)
    dense = int(layers["phase_order"][0].cpu())
    h = float(layers["phase_tops_m"][dense].cpu())
    outlet = EquilibriumOutlet(
        "water",
        0.99e6,
        1e-12,
        height_m=h / 2,
        diameter_m=min(0.0001, h / 3),
        selection="component_rich",
        component="water",
    )
    m = CompositionalNetwork(
        {"separator": CompositionalSeparator(f, g.volume_m3, [outlet], geometry=g)},
        temporal_error_control=False,
    )
    s = m.initialize(
        {"separator": {"pressure_pa": 1e6, "temperature_k": 350, "mole_fractions": [0.4, 0.3, 0.3]}}
    )
    periods = [
        {"duration_s": 0.005},
        {"duration_s": 0.005, "outlet_openings": {"separator": {"water": 0}}},
    ]
    full = run_compositional_network_schedule(m, s, periods)
    prefix = run_compositional_network_schedule(m, s, periods, stop_after=1)
    assert run_compositional_network_schedule(m, s, periods, checkpoint=prefix) == full
    step = full["records"][0]["steps"][0]["vessels"]["separator"]
    assert step["phase_count"] == 3
    exposure = step["ports"]["water"]["pickup_area_fractions"]
    assert exposure[dense] == pytest.approx(1, abs=1e-10)
    rate = step["ports"]["water"]["component_molar_rates_mol_s"]
    assert rate[2] > 1000 * rate[0]


@pytest.mark.parametrize("device", DEVICES)
def test_geometric_density_ties_are_not_silently_resolved(device):
    n = torch.tensor([1.0, 1.0], device=device, dtype=torch.float64)
    props = {"molar_volume_m3_mol": torch.ones_like(n), "density_kg_m3": n.new_tensor([500, 500])}
    assert not bool(phase_layout(n, props, geometry())["density_order_resolved"].cpu())
    control = EquilibriumLevelControl("absent", 0, 1, opening_bias=1, minimum_mole_fraction=0)
    one = n[:1]
    single = {k: v[:1] for k, v in props.items()}
    opening, _, present = control.value(
        phase_layout(one, single, geometry()), one.new_tensor([[1, 0]]), ["present", "absent"]
    )
    assert not bool(present.cpu()) and float(opening.cpu()) == 0
