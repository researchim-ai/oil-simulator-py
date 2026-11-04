"""Independent EOS/phase/caloric identities and conservative hand-off tests."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.component_energy import (
    ComponentEnergyStream,
    StockComponentAssay,
    audit_component_energy,
    mix_component_energy_streams,
    separate_equilibrium_stream,
)
from surface.equilibrium_inventory import EquilibriumInventory
from surface.equilibrium_properties import EquilibriumFluid
from surface.liquid import ConservationError, EquipmentOperatingError, PropertyModelError
from surface.peng_robinson import PengRobinsonFluid
from surface.peng_robinson_mixture import PengRobinsonMixture

DATA = json.loads((Path(__file__).parent / "data/surface/pr_mixture_coolprop.json").read_text())
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def fluid(device="cpu", kij=0):
    return EquilibriumFluid(
        [PengRobinsonFluid(**c) for c in DATA["components"]], [[0, kij], [kij, 0]], device=device
    )


def symmetric(device="cpu", max_phases=3):
    cs = [PengRobinsonFluid(n, 500, 5e6, 0.2, 0.1, 100, 250, 700) for n in ["a", "b", "c"]]
    return EquilibriumFluid(
        cs, [[0, 0.6, 0.6], [0.6, 0, 0.6], [0.6, 0.6, 0]], device=device, max_phases=max_phases
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pressure", [0.01, 100.0, 1e6])
def test_forward_ad_cubic_root_is_finite_in_dilute_limit(device, pressure):
    f = EquilibriumFluid(
        [PengRobinsonFluid(**c) for c in DATA["components"]],
        [[0, 0], [0, 0]],
        device=device,
        min_pressure_pa=1e-9,
    )

    def values(y):
        p = y[0].exp()
        t = y[1] * 300
        x = torch.softmax(torch.stack((y[2], y[2] * 0)), 0)
        q = f.tensor_homogeneous(p, t, x)
        return torch.cat(
            (
                q["log_fugacity_coefficients"],
                q["compressibility_factor"].reshape(1),
                q["molar_enthalpy_j_mol"].reshape(1) / 1e4,
            )
        )

    y = f.tensor([np.log(pressure), 400 / 300, 0.0])
    ad = torch.func.jacfwd(values)(y)
    assert torch.isfinite(ad).all()
    fd = []
    for j in range(3):
        d = torch.zeros_like(y)
        d[j] = 1e-5
        fd.append(((values(y + d) - values(y - d)) / 2e-5).cpu().numpy())
    np.testing.assert_allclose(ad.cpu().numpy(), np.array(fd).T, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", DATA["cases"])
def test_independent_coolprop_multiphase(case, device):
    f = fluid(device, case["kij"])
    q = f.flash_pt(case["pressure_pa"], case["temperature_k"], [0.5, 0.5])
    assert q["phase_count"] == 2
    assert float(q["phases"][-1]["molar_fraction"].cpu()) == pytest.approx(
        case["vapor_molar_fraction"], abs=2e-4
    )
    assert float(q["density_kg_m3"].cpu()) == pytest.approx(case["density_kg_m3"], rel=6e-4)
    for p, name in zip(q["phases"], ["liquid", "vapor"]):
        assert p["mole_fractions"].cpu().tolist() == pytest.approx(
            case[name + "_mole_fractions"], abs=2e-4
        )
    assert float(q["component_balance_error"].cpu()) < 1e-10
    assert float(q["log_fugacity_error"].cpu()) < 1e-8
    assert float(q["minimum_tpd"].cpu()) >= -1e-8
    assert q["density_kg_m3"].device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_three_liquid_equilibrium_against_independent_symmetric_root(device):
    f = symmetric(device)
    p, t = 1e6, 300.0
    q = f.flash_pt(p, t, [1 / 3] * 3)
    assert q["phase_count"] == 3
    oracle = PengRobinsonMixture(f.components, f.binary_interactions)

    def residual(a):
        x = np.array([a, (1 - a) / 2, (1 - a) / 2])
        state = oracle.state(p, t, x)
        mu = np.log(x) + state["log_fugacity_coefficients"]
        return mu[0] - mu[1]

    a = brentq(residual, 0.9, 0.99999999, xtol=1e-14)
    expected = sorted(
        [
            [a, (1 - a) / 2, (1 - a) / 2],
            [(1 - a) / 2, a, (1 - a) / 2],
            [(1 - a) / 2, (1 - a) / 2, a],
        ]
    )
    # Equal trace fractions differ by a few ulps between devices; compare
    # phase identity by its dominant component, not lexicographic trace order.
    expected.sort(key=lambda row: int(np.argmax(row)))
    actual = sorted(
        (v["mole_fractions"].cpu().tolist() for v in q["phases"]),
        key=lambda row: int(np.argmax(row)),
    )
    assert np.array(actual) == pytest.approx(np.array(expected), abs=2e-9)
    assert [float(v["molar_fraction"].cpu()) for v in q["phases"]] == pytest.approx(
        [1 / 3] * 3, abs=1e-9
    )
    assert float(q["minimum_tpd"].cpu()) >= -1e-8
    with pytest.raises(PropertyModelError, match="additional phases"):
        symmetric(device, max_phases=2).flash_pt(p, t, [1 / 3] * 3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("p,t", [(1e6, 320.0), (1e6, 196.2044254)])
def test_implicit_equilibrium_jacobian_and_caloric_maxwell(p, t, device):
    f = fluid(device)
    q = f.flash_pt(p, t, [0.5, 0.5])
    d = f.equilibrium_derivatives(p, t, [0.5, 0.5], equilibrium=q)
    jac = d["jacobian"].cpu().numpy()
    y = np.array([p, t, 0.5, 0.5])
    steps = [10.0, 0.0001, 1e-5, 1e-5]
    numerical = []
    for i, h in enumerate(steps):
        a, b = y.copy(), y.copy()
        a[i] += h
        b[i] -= h
        qa = f.flash_pt(a[0], a[1], a[2:] / a[2:].sum())
        qb = f.flash_pt(b[0], b[1], b[2:] / b[2:].sum())
        numerical.append([float((qa[k] - qb[k]).cpu()) / (2 * h) for k in d["property_names"]])
    assert jac == pytest.approx(np.array(numerical).T, rel=5e-5, abs=2e-8)
    assert jac[0, 1] == pytest.approx(t * jac[1, 1], rel=2e-8)
    assert jac[0, 0] == pytest.approx(
        float(q["molar_volume_m3_mol"].cpu()) - t * jac[2, 1], rel=2e-8
    )
    assert jac[1, 0] == pytest.approx(-jac[2, 1], rel=2e-8)
    assert np.max(np.abs(jac[:, 2:].sum(1))) < 1e-7


@pytest.mark.parametrize("device", DEVICES)
def test_ph_ps_uv_roundtrip_across_two_phase_equilibrium(device):
    f = fluid(device)
    p, t = 1e6, 196.2044254
    z = [0.5, 0.5]
    q = f.flash_pt(p, t, z)
    ph = f.flash_ph(p, float(q["molar_enthalpy_j_mol"].cpu()), z, temperature_guess_k=200)
    ps = f.flash_ps(p, float(q["molar_entropy_j_mol_k"].cpu()), z, temperature_guess_k=200)
    for r in [ph, ps]:
        assert float(r["temperature_k"].cpu()) == pytest.approx(t, abs=2e-7)
        assert r["phase_count"] == 2
    uv = f.flash_uv(
        [5, 5],
        10 * float(q["molar_internal_energy_j_mol"].cpu()),
        10 * float(q["molar_volume_m3_mol"].cpu()),
        pressure_guess_pa=1.1e6,
        temperature_guess_k=198,
    )
    assert float(uv["pressure_pa"].cpu()) == pytest.approx(p, rel=1e-8)
    assert float(uv["temperature_k"].cpu()) == pytest.approx(t, abs=1e-6)
    assert abs(float(uv["energy_residual_j"].cpu())) < 1e-5
    assert abs(float(uv["volume_residual_m3"].cpu())) < 1e-10


@pytest.mark.parametrize("device", DEVICES)
def test_absent_species_zero_flow_and_strict_model_identity(device):
    f = fluid(device)
    q = f.flash_pt(1e5, 300, [1.0, 0.0])
    assert q["phase_count"] == 1
    assert q["phases"][0]["mole_fractions"].cpu().tolist() == [1.0, 0.0]
    s = ComponentEnergyStream.from_pt(f, [0, 0], 1e5, 300)
    assert s.equilibrate(f) is None
    assert separate_equilibrium_stream(s, f)["outputs"] == ()
    with pytest.raises(ConservationError):
        replace(s, total_energy_rate_w=1)
    with pytest.raises(PropertyModelError, match="mismatch"):
        replace(s, model_id="0" * 64).equilibrate(f)
    with pytest.raises(PropertyModelError):
        f.flash_pt(1e5, 300, [-0.1, 1.1])
    assert f.to("cpu").model_id == f.model_id
    assert EquilibriumFluid.from_dict(f.to_dict(), device=device).model_id == f.model_id


@pytest.mark.parametrize("device", DEVICES)
def test_component_energy_transfer_mixing_and_three_phase_separation(device):
    f = symmetric(device)
    s = ComponentEnergyStream.from_pt(f, [1.0, 1.0, 1.0], 2e6, 310, elevation_m=100, velocity_m_s=2)
    out = s.transport(f, 1e6, elevation_m=50, velocity_m_s=3, temperature_guess_k=310)
    assert out.component_molar_rates_mol_s == s.component_molar_rates_mol_s
    assert out.total_energy_rate_w == s.total_energy_rate_w
    a, b = out.split(0.37)
    mixed = mix_component_energy_streams(
        [a, b], f, 1e6, elevation_m=50, velocity_m_s=3, temperature_guess_k=310
    )
    assert mixed.total_energy_rate_w == pytest.approx(out.total_energy_rate_w, abs=1e-9)
    result = separate_equilibrium_stream(mixed, f, temperature_guess_k=310)
    assert len(result["outputs"]) == 3
    assert max(map(abs, result["component_residuals_mol_s"])) < 1e-10
    assert abs(result["energy_residual_w"]) < 1e-5
    assert ComponentEnergyStream.from_dict(s.to_dict()) == s
    assert torch.equal(s.oriented_flux(f, -1), -s.oriented_flux(f))
    with pytest.raises(EquipmentOperatingError):
        s.transport(f, 3e6)
    with pytest.raises(ConservationError):
        audit_component_energy(
            [s], [replace(s, total_energy_rate_w=s.total_energy_rate_w + 100)], f
        )


@pytest.mark.parametrize("device", DEVICES)
def test_explicit_stock_assay_mass_and_declared_isothermal_energy_boundary(device):
    f = fluid(device)
    assay = StockComponentAssay(f.names, ((0.1, 0.9), (0.3, 0.7), (0.95, 0.05)), f.model_id)
    rates = assay.component_molar_rates([1.0, 2.0, 3.0], f)
    assert float((rates * f.molar_masses).sum().cpu()) == pytest.approx(6.0, abs=1e-12)
    source = assay.from_isothermal_boundary([1.0, 2.0, 3.0], f, 1e6, 320, elevation_m=10)
    assert source["reservoir_energy_balance_available"] is False
    assert source["boundary_type"] == "prescribed_temperature_external_energy"
    s = source["stream"]
    exact = assay.from_energy_boundary(
        [1.0, 2.0, 3.0], f, 1e6, s.total_energy_rate_w, elevation_m=10, temperature_guess_k=320
    )
    assert exact.component_molar_rates_mol_s == s.component_molar_rates_mol_s
    assert exact.total_energy_rate_w == s.total_energy_rate_w
    with pytest.raises(PropertyModelError):
        StockComponentAssay(f.names, ((0.2, 0.2), (0.3, 0.7), (0.95, 0.05)), f.model_id)
    with pytest.raises(PropertyModelError):
        replace(assay, component_names=f.names[::-1]).component_molar_rates([1, 2, 3], f)


@pytest.mark.parametrize("device", DEVICES)
def test_rigid_inventory_heat_and_internal_energy_exchange(device):
    f = fluid(device)
    a = EquilibriumInventory.from_pt(f, [50, 50], 1e6, 320, elevation_m=10)
    heated = a.advance_fluxes(f, 0.01, heat_w=1000, utility_temperature_k=400)
    b = heated["state"]
    assert b.component_moles == a.component_moles
    assert b.internal_energy_j - a.internal_energy_j == pytest.approx(10.0, abs=1e-8)
    assert b.volume_m3 == a.volume_m3
    assert b.temperature_k > a.temperature_k
    assert heated["entropy_production_j_k"] > 0
    assert EquilibriumInventory.from_dict(b.to_dict()) == b
    with pytest.raises(ConservationError):
        replace(a, internal_energy_j=a.internal_energy_j + 100).inspect(f)
    with pytest.raises(ValueError, match="utility"):
        a.advance_fluxes(f, 0.1, heat_w=1)
    assert a.time_s == 0


@pytest.mark.parametrize("device", DEVICES)
def test_vapor_oil_aqueous_equilibrium_transfers_water_and_dissolved_gas(device):
    # Illustrative PR parameterization, deliberately NOT calibrated water PVT.
    cs = [
        PengRobinsonFluid("methane", 190.564, 4599200, 0.01142, 0.0160428, 35.7, 150, 700, 350),
        PengRobinsonFluid("hexane", 507.6, 3025000, 0.301, 0.086178, 150, 260, 700, 350),
        PengRobinsonFluid("water", 647.096, 22064000, 0.344, 0.01801528, 33.6, 324, 700, 350),
    ]
    f = EquilibriumFluid(cs, [[0, 0.03, 0.48], [0.03, 0, 0.5], [0.48, 0.5, 0]], device=device)
    q = f.flash_pt(1e6, 350, [0.4, 0.3, 0.3])
    assert q["phase_count"] == 3
    aqueous, oil, gas = q["phases"]
    assert float(aqueous["mole_fractions"][2].cpu()) > 0.999
    assert float(oil["mole_fractions"][1].cpu()) > 0.9
    assert float(gas["mole_fractions"][0].cpu()) > 0.8
    assert float(gas["mole_fractions"][2].cpu()) > 0.01  # water evaporation
    assert float(oil["mole_fractions"][0].cpu()) > 0.01  # dissolved methane
    assert float(q["log_fugacity_error"].cpu()) < 1e-8
    for phase in q["phases"]:
        assert float(phase["molar_enthalpy_j_mol"].cpu()) - 1e6 * float(
            phase["molar_volume_m3_mol"].cpu()
        ) == pytest.approx(float(phase["molar_internal_energy_j_mol"].cpu()), abs=1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_pure_species_latent_heat_ph_ps_and_pt_degeneracy(device):
    f = fluid(device)
    c = f.components[1]
    saturation = c.saturation_at_temperature(200.0)
    p = saturation["pressure_pa"]
    z = [0.0, 1.0]
    with pytest.raises(PropertyModelError, match="coexistence"):
        f.flash_pt(p, 200, z)
    for method, key in [
        (f.flash_ph, "molar_enthalpy_j_mol"),
        (f.flash_ps, "molar_entropy_j_mol_k"),
    ]:
        l = f.homogeneous(p, 200, z, "liquid")
        v = f.homogeneous(p, 200, z, "vapor")
        target = float((0.7 * l[key] + 0.3 * v[key]).cpu())
        q = method(p, target, z, temperature_guess_k=202)
        assert q["phase_count"] == 2
        assert float(q["temperature_k"].cpu()) == pytest.approx(200, abs=1e-6)
        assert float(q["phases"][1]["molar_fraction"].cpu()) == pytest.approx(0.3, abs=1e-7)
        assert abs(float(q["property_residual"].cpu())) < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_runtime_never_calls_scipy_for_phase_or_property_solution(device, monkeypatch):
    import numpy.linalg
    import scipy.optimize

    def forbid(*args, **kwargs):
        raise AssertionError("CPU reference solver used by Torch equilibrium")

    for name in ["minimize", "least_squares", "brentq", "root"]:
        monkeypatch.setattr(scipy.optimize, name, forbid)
    monkeypatch.setattr(numpy.linalg, "solve", forbid)
    f = fluid(device)
    q = f.flash_pt(1e6, 196.2, [0.5, 0.5])
    assert q["phase_count"] == 2
    assert f.equilibrium_derivatives(1e6, 196.2, [0.5, 0.5])["jacobian"].device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_phase_appearance_small_fractions_and_homogeneous_heat_capacities(device):
    f = fluid(device)
    reference = PengRobinsonMixture(f.components, f.binary_interactions)
    eq = reference.flash_pt(1e6, 196.2, [0.5, 0.5])
    x = np.array(eq["phases"][0]["mole_fractions"])
    y = np.array(eq["phases"][1]["mole_fractions"])
    for beta in [1e-4, 1e-6, 1 - 1e-6]:
        q = f.flash_pt(1e6, 196.2, ((1 - beta) * x + beta * y).tolist())
        assert q["phase_count"] == 2
        assert float(q["phases"][1]["molar_fraction"].cpu()) == pytest.approx(beta, abs=2e-8)
    p, t = f.tensor(1e6), f.tensor(320.0).requires_grad_()
    z = f.tensor([0.4, 0.6])
    q = f.tensor_homogeneous(p, t, z)
    cp = torch.autograd.grad(q["molar_enthalpy_j_mol"], t)[0]
    assert float(q["homogeneous_molar_cp_j_mol_k"].detach().cpu()) == pytest.approx(
        float(cp.cpu()), rel=1e-10
    )
    assert float(q["homogeneous_molar_cv_j_mol_k"].detach().cpu()) > 0


@pytest.mark.parametrize("device", DEVICES)
def test_ambient_water_domain_and_thermal_flash_without_legacy_half_tc_limit(device):
    from surface.equilibrium_properties import EquilibriumComponent

    cs = [
        EquilibriumComponent("methane", 190.564, 4599200, 0.01142, 0.0160428, 35.7, 150, 700),
        EquilibriumComponent("hexane", 507.6, 3025000, 0.301, 0.086178, 150, 275, 700),
        EquilibriumComponent("water", 647.096, 22064000, 0.344, 0.01801528, 33.6, 275, 700),
    ]
    f = EquilibriumFluid(cs, [[0, 0.03, 0.48], [0.03, 0, 0.5], [0.48, 0.5, 0]], device=device)
    q = f.flash_pt(1e6, 298.15, [0.4, 0.3, 0.3])
    assert q["phase_count"] == 3
    assert float(q["phases"][0]["mole_fractions"][2].cpu()) > 0.999
    assert float(q["phases"][-1]["mole_fractions"][2].cpu()) > 0  # finite water vapor
    restored = EquilibriumFluid.from_dict(f.to_dict(), device=device)
    assert restored.model_id == f.model_id
    ph = restored.flash_ph(
        1e6, float(q["molar_enthalpy_j_mol"].cpu()), [0.4, 0.3, 0.3], temperature_guess_k=300
    )
    assert float(ph["temperature_k"].cpu()) == pytest.approx(298.15, abs=1e-6)
