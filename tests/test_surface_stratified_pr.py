"""Conservative gravity equilibrium: analytic limits, ODE, AD and replay."""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.equilibrium_properties import EquilibriumFluid
from surface.liquid import ConservationError, HydraulicConvergenceError, PropertyModelError
from surface.peng_robinson import PengRobinsonFluid, R
from surface.peng_robinson_mixture import PengRobinsonMixture
from surface.separator_geometry import SeparatorGeometry
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def binary(device, **kwargs):
    cfg = json.loads((ROOT / "tests/data/surface/pr_mixture_coolprop.json").read_text())
    return EquilibriumFluid(
        [PengRobinsonFluid(**c) for c in cfg["components"]],
        [[0, 0], [0, 0]],
        device=device,
        **kwargs,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_ideal_barometric_composition_and_energy_limit(device):
    f = binary(device, min_pressure_pa=1e-9)
    height = 10000.0
    area = math.pi * 0.1**2 / 4
    m = StratifiedPRColumn(
        f,
        SeparatorGeometry(0.1, height, "vertical"),
        relative_tolerance=1e-8,
        temperature_tolerance_k=1e-7,
    )
    r = m.initialize(0.01, 400, [0.5, 0.5])
    s = StratifiedPRState.from_dict(r["state"])
    n = np.array(s.component_moles)
    mw = np.array([c.molar_mass_kg_mol for c in f.components])
    cp = np.array([c.ideal_cp_j_mol_k for c in f.components])

    def energy(t):
        a = mw * m.gravity_m_s2 / (R * t)
        mean = 1 / a - height / np.expm1(a * height)
        return np.sum(n * (cp * (t - f.tref) - R * t + mw * m.gravity_m_s2 * mean))

    t = brentq(lambda temp: energy(temp) - s.total_energy_j, 300, 600, xtol=1e-10)
    assert s.temperature_k == pytest.approx(t, rel=1e-8)
    a = mw * m.gravity_m_s2 / (R * t)
    concentrations = n * a / (area * (-np.expm1(-a * height)))
    for h in [0, height / 2, height]:
        expected = concentrations * np.exp(-a * h)
        q = m.sample(s, h)
        assert q["pressure_pa"] == pytest.approx(R * t * expected.sum(), rel=1e-7)
        assert q["mole_fractions"] == pytest.approx(expected / expected.sum(), rel=1e-7)
    assert m.sample(s, 0)["mole_fractions"][0] < m.sample(s, height)["mole_fractions"][0]
    assert r["entropy_gain_j_k"] >= -1e-7


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("composition,temperature", [([1, 0], 320), ([0, 1], 200)])
def test_real_pr_hydrostatic_profile_against_independent_ode(device, composition, temperature):
    f = binary(device)
    height = 10.0
    m = StratifiedPRColumn(f, SeparatorGeometry(0.1, height, "vertical"))
    r = m.initialize(1e6, temperature, composition)
    s = StratifiedPRState.from_dict(r["state"])
    component = f.components[composition.index(1)]
    bottom = m.sample(s, 0)["pressure_pa"]
    ref = solve_ivp(
        lambda h, p: [
            -component.state(p[0], s.temperature_k, derivatives=False)["density_kg_m3"]
            * m.gravity_m_s2
        ],
        [0, height],
        [bottom],
        method="DOP853",
        rtol=1e-11,
        atol=1e-5,
        dense_output=True,
    )
    assert ref.success
    for h in [height * 0.25, height * 0.5, height]:
        q = m.sample(s, h)
        assert q["pressure_pa"] == pytest.approx(ref.sol(h)[0], rel=2e-8, abs=1e-3)
        assert q["mole_fractions"] == composition
    assert abs(r["energy_residual_j"]) < 1e-5 + 1e-9 * abs(s.total_energy_j)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_two_phase_interfaces_and_gibbs_duhem_hydrostatics(device, orientation):
    f = binary(device)
    m = StratifiedPRColumn(f, SeparatorGeometry(0.1, 0.2, orientation))
    r = m.initialize(1e6, 196.2, [0.19, 0.81])
    s = StratifiedPRState.from_dict(r["state"])
    assert len(s.interface_heights_m) == 1
    h = s.interface_heights_m[0]
    with pytest.raises(ValueError, match="side"):
        m.sample(s, h)
    a = m.sample(s, h, side="below")
    b = m.sample(s, h, side="above")
    assert a["pressure_pa"] == pytest.approx(b["pressure_pa"], abs=1e-3)
    assert a["density_kg_m3"] > b["density_kg_m3"]
    ref = PengRobinsonMixture(f.components, f.binary_interactions)
    constants = np.array(s.chemical_constants)
    mw = f.molar_masses.cpu().numpy()
    for z in [h / 2, (h + m.geometry.height_m) / 2]:
        point = m.sample(s, z)
        phase = "liquid" if z < h else "vapor"
        q = ref.state(point["pressure_pa"], s.temperature_k, point["mole_fractions"], phase)
        mu = (
            np.log(point["mole_fractions"])
            + q["log_fugacity_coefficients"]
            + np.log(point["pressure_pa"] / f.pref)
            + mw * m.gravity_m_s2 * z / (R * s.temperature_k)
        )
        np.testing.assert_allclose(mu, constants, atol=1e-8)
        dz = 1e-4 * min(h, m.geometry.height_m - h)
        slope = (m.sample(s, z + dz)["pressure_pa"] - m.sample(s, z - dz)["pressure_pa"]) / (2 * dz)
        assert slope == pytest.approx(-point["density_kg_m3"] * m.gravity_m_s2, rel=1e-3, abs=0.1)
    assert r["quadrature_error_ratio"] <= 1


@pytest.mark.parametrize("device", DEVICES)
def test_three_phase_distinct_interfaces_and_gravity_energy(device):
    cfg = json.loads((ROOT / "configs/surface_equilibrium_process.json").read_text())
    f = EquilibriumFluid.from_dict(cfg["fluid"], device=device)
    m = StratifiedPRColumn(f, SeparatorGeometry(0.1, 0.2, "vertical"))
    r = m.initialize(1e6, 350, [0.4, 0.3, 0.3])
    s = StratifiedPRState.from_dict(r["state"])
    assert len(s.phase_roots) == 3 and len(s.interface_heights_m) == 2
    pressures = []
    for h in s.interface_heights_m:
        a = m.sample(s, h, side="below")
        b = m.sample(s, h, side="above")
        assert a["pressure_pa"] == pytest.approx(b["pressure_pa"], abs=1e-3)
        pressures.append(a["pressure_pa"])
    assert pressures[0] > pressures[1]
    assert r["potential_energy_j"] > 0
    assert r["internal_energy_j"] + r["potential_energy_j"] == pytest.approx(
        s.total_energy_j, abs=1e-5
    )
    assert max(map(abs, r["component_residuals_mol"])) < 1e-8


@pytest.mark.parametrize("device", DEVICES)
def test_zero_gravity_recovers_uniform_flash_and_geometric_layers(device):
    f = binary(device)
    g = SeparatorGeometry(0.1, 0.2, "horizontal")
    q = f.flash_pt(1e6, 196.2, [0.19, 0.81])
    total = g.volume_m3 / float(q["molar_volume_m3_mol"].cpu())
    m = StratifiedPRColumn(f, g, gravity_m_s2=0)
    r = m.initialize(1e6, 196.2, [0.19, 0.81])
    s = StratifiedPRState.from_dict(r["state"])
    assert s.temperature_k == pytest.approx(196.2, abs=1e-7)
    for layer in r["layers"]:
        assert layer["pressure_pa"] == pytest.approx([1e6] * s.quadrature_order, rel=1e-8)
    volume = (
        total
        * float(q["phases"][0]["molar_fraction"].cpu())
        * float(q["phases"][0]["molar_volume_m3_mol"].cpu())
    )
    assert s.interface_heights_m[0] == pytest.approx(g.level_m(volume), abs=1e-8)
    assert r["potential_energy_j"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_monolithic_jacobian_and_no_flash_or_cpu_quadrature_in_trials(device, monkeypatch):
    f = binary(device)
    m = StratifiedPRColumn(f, SeparatorGeometry(0.1, 0.2, "horizontal"))
    r = m.initialize(1e6, 196.2, [0.19, 0.81])
    s = StratifiedPRState.from_dict(r["state"])
    n = f.tensor(s.component_moles)
    seed = (
        [f.tensor(x) for x in s.phase_seeds],
        list(s.phase_roots),
        f.tensor(s.interface_heights_m),
        r["layers"][0]["pressure_pa"][0],
        s.temperature_k,
        f.tensor(s.chemical_constants),
    )
    _, _, _, evaluate, _, _ = m._problem(n, s.total_energy_j, seed, s.quadrature_order)

    def forbidden(*args, **kwargs):
        raise AssertionError("Host flash/quadrature in monolithic residual")

    monkeypatch.setattr(f, "flash_pt", forbidden)
    monkeypatch.setattr(f, "flash_uv", forbidden)
    monkeypatch.setattr(np.polynomial.legendre, "leggauss", forbidden)
    y = f.tensor(s.solution_variables)
    fun = lambda v: evaluate(v)[0]
    ad = torch.func.jacfwd(fun)(y).cpu().numpy()
    fd = []
    for i in range(len(y)):
        d = torch.zeros_like(y)
        d[i] = 1e-6
        fd.append(((fun(y + d) - fun(y - d)) / 2e-6).cpu().numpy())
    np.testing.assert_allclose(ad, np.array(fd).T, rtol=3e-4, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_elevation_gauge_local_stream_and_serialized_state_validation(device):
    f = binary(device)
    g = SeparatorGeometry(0.1, 1, "vertical")
    a = StratifiedPRColumn(f, g, elevation_m=0)
    b = StratifiedPRColumn(f, g, elevation_m=1000)
    ra = a.initialize(1e6, 320, [0.5, 0.5])
    rb = b.initialize(1e6, 320, [0.5, 0.5])
    sa = StratifiedPRState.from_dict(ra["state"])
    sb = StratifiedPRState.from_dict(rb["state"])
    assert sa.temperature_k == pytest.approx(sb.temperature_k, abs=1e-7)
    qa = a.sample(sa, 0.3)
    qb = b.sample(sb, 0.3)
    assert qa["pressure_pa"] == pytest.approx(qb["pressure_pa"], abs=1e-3)
    qa["mole_fractions"][0] = 0
    assert a.sample(sa, 0.3)["mole_fractions"][0] > 0
    flow = b.stream_at(sb, 0.3, 0.1)
    mass = sum(
        n * c.molar_mass_kg_mol for n, c in zip(flow.component_molar_rates_mol_s, f.components)
    )
    assert flow.total_energy_rate_w == pytest.approx(
        0.1 * qb["molar_enthalpy_j_mol"] + mass * b.gravity_m_s2 * 1000.3, abs=1e-7
    )
    fresh = StratifiedPRColumn.from_dict(a.to_dict(), device=device)
    fresh.inspect(sa)
    forged = replace(sa, total_energy_j=sa.total_energy_j + 1)
    with pytest.raises(ConservationError):
        fresh.inspect(forged)
    forged = replace(sa, temperature_k=sa.temperature_k + 0.1)
    with pytest.raises(ConservationError):
        fresh.inspect(forged)
    with pytest.raises(ValueError, match="outside"):
        a.sample(sa, 2)
    assert b.stream_at(sb, 0.3, 0).total_energy_rate_w == 0


def test_invalid_configuration_budget_and_identity():
    f = binary("cpu")
    g = SeparatorGeometry(0.1, 1, "vertical")
    with pytest.raises(ValueError, match="higher maximum"):
        StratifiedPRColumn(f, g, quadrature_order=8, maximum_quadrature_order=8)
    m = StratifiedPRColumn(f, g)
    with pytest.raises(ValueError, match="inventory"):
        m.equilibrate([0, 0], 1, pressure_guess_pa=1e6, temperature_guess_k=320)
    r = m.initialize(1e6, 320, [0.5, 0.5])
    s = StratifiedPRState.from_dict(r["state"])
    with pytest.raises(PropertyModelError):
        m.sample(replace(s, model_id="0" * 64), 0)
    # Refinement may never silently publish an unqualified coarse result.
    strict = StratifiedPRColumn(
        f,
        SeparatorGeometry(0.1, 0.2, "horizontal"),
        quadrature_order=2,
        maximum_quadrature_order=4,
        relative_tolerance=1e-12,
        temperature_tolerance_k=1e-12,
    )
    with pytest.raises(HydraulicConvergenceError, match="quadrature"):
        strict.initialize(1e6, 196.2, [0.19, 0.81])
