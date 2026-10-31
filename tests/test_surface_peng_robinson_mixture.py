"""Independent PR VLE fixture, thermodynamic identities and balance checks."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import PengRobinsonFluid, PengRobinsonMixture, PropertyModelError

DATA = json.loads((Path(__file__).parent / "data/surface/pr_mixture_coolprop.json").read_text())


def mixture(kij=0):
    return PengRobinsonMixture(
        tuple(PengRobinsonFluid(**c) for c in DATA["components"]), ((0, kij), (kij, 0))
    )


@pytest.mark.parametrize("case", DATA["cases"])
@pytest.mark.parametrize("matched_coefficients", [False, True])
def test_independent_coolprop_vle(case, matched_coefficients):
    m = mixture(case["kij"])
    if matched_coefficients:
        # CoolProp 7.2 source uses rounded 0.45724/0.07780. Only this
        # test adapter matches that convention; production uses exact constants.
        from surface.peng_robinson import OMEGA_A, OMEGA_B

        class RoundedPR(PengRobinsonFluid):
            @property
            def b(self):
                return super().b * 0.07780 / OMEGA_B

            def _a(self, t):
                a, at = super()._a(t)
                return a * 0.45724 / OMEGA_A, at * 0.45724 / OMEGA_A

        m = PengRobinsonMixture(
            tuple(RoundedPR(**c) for c in DATA["components"]), m.binary_interactions
        )
    fraction_tol = 2e-7 if matched_coefficients else 2e-4
    density_tol = 2e-7 if matched_coefficients else 6e-4
    r = m.flash_pt(case["pressure_pa"], case["temperature_k"], [0.5, 0.5])
    assert r["phase_count"] == 2
    assert r["vapor_molar_fraction"] == pytest.approx(
        case["vapor_molar_fraction"], abs=fraction_tol
    )
    assert r["density_kg_m3"] == pytest.approx(case["density_kg_m3"], rel=density_tol)
    for state, name in zip(r["phases"], ("liquid", "vapor")):
        assert state["mole_fractions"] == pytest.approx(
            case[name + "_mole_fractions"], abs=fraction_tol
        )
    assert r["component_balance_error"] < 1e-9
    assert r["log_fugacity_error"] < 1e-8
    assert r["stability"]["minimum_tpd"] >= -1e-8


@pytest.mark.parametrize("p,t", [(1e5, 300), (8e6, 220)])
def test_stable_single_phase(p, t):
    m = mixture()
    r = m.flash_pt(p, t, [0.5, 0.5])
    assert r["phase_count"] == 1
    assert r["stability"]["stable"]


def test_homogeneous_root_is_not_phase_stability():
    m = mixture()
    root = m.state(3e6, 220, [0.5, 0.5])
    assert root["density_kg_m3"] > 300
    assert not m.stability(3e6, 220, [0.5, 0.5])["stable"]
    assert m.flash_pt(3e6, 220, [0.5, 0.5])["phase_count"] == 2


def test_permutation_and_component_mass_balance():
    m = mixture(-0.03)
    r = m.flash_pt(3e6, 220, [0.3, 0.7])
    reverse = PengRobinsonMixture(m.components[::-1], m.binary_interactions)
    rr = reverse.flash_pt(3e6, 220, [0.7, 0.3])
    assert rr["density_kg_m3"] == pytest.approx(r["density_kg_m3"], rel=1e-9)
    mw = np.array([c.molar_mass_kg_mol for c in m.components])
    total = sum(q["molar_fraction"] * np.array(q["mole_fractions"]) * mw for q in r["phases"])
    assert total == pytest.approx(np.array([0.3, 0.7]) * mw, abs=1e-10)


def test_caloric_mixing_and_maxwell_identity():
    m = mixture()
    p, t, z = 1e6, 320, [0.4, 0.6]
    state = m.state(p, t, z)
    dp, dt = 10.0, 0.002
    hp = (
        m.state(p + dp, t, z)["molar_enthalpy_j_mol"]
        - m.state(p - dp, t, z)["molar_enthalpy_j_mol"]
    ) / (2 * dp)
    vt = (
        m.state(p, t + dt, z)["molar_volume_m3_mol"] - m.state(p, t - dt, z)["molar_volume_m3_mol"]
    ) / (2 * dt)
    sp = (
        m.state(p + dp, t, z)["molar_entropy_j_mol_k"]
        - m.state(p - dp, t, z)["molar_entropy_j_mol_k"]
    ) / (2 * dp)
    assert hp == pytest.approx(state["molar_volume_m3_mol"] - t * vt, rel=1e-7)
    assert sp == pytest.approx(-vt, rel=1e-7)
    ht = (
        m.state(p, t + dt, z)["molar_enthalpy_j_mol"]
        - m.state(p, t - dt, z)["molar_enthalpy_j_mol"]
    ) / (2 * dt)
    st = (
        m.state(p, t + dt, z)["molar_entropy_j_mol_k"]
        - m.state(p, t - dt, z)["molar_entropy_j_mol_k"]
    ) / (2 * dt)
    assert ht == pytest.approx(t * st, rel=1e-8)


@pytest.mark.parametrize(
    "key,method", [("molar_enthalpy_j_mol", "flash_ph"), ("molar_entropy_j_mol_k", "flash_ps")]
)
def test_thermal_flash_roundtrip(key, method):
    m = mixture()
    r = m.flash_pt(3e6, 220, [0.5, 0.5])
    back = getattr(m, method)(3e6, r[key], [0.5, 0.5])
    assert back["temperature_k"] == pytest.approx(220, abs=1e-7)
    assert back["vapor_molar_fraction"] == pytest.approx(r["vapor_molar_fraction"], abs=1e-8)
    assert abs(back["property_residual"]) < 1e-6


@pytest.mark.parametrize("z", [[0.2, 0.2], [0, 1], [-1, 2], [float("nan"), 0.5]])
def test_invalid_compositions(z):
    with pytest.raises(PropertyModelError):
        mixture().flash_pt(1e6, 250, z)


def test_bad_interactions_and_thermal_domain():
    c = mixture().components
    for k in [None, [[0, 0.1], [0.2, 0]], [[1, 0], [0, 0]], [[0, float("nan")], [float("nan"), 0]]]:
        with pytest.raises((PropertyModelError, TypeError)):
            PengRobinsonMixture(c, k)
    with pytest.raises(PropertyModelError):
        mixture().flash_pt(1e6, 100, [0.5, 0.5])
    with pytest.raises(PropertyModelError):
        mixture().flash_ph(1e6, 1e12, [0.5, 0.5])


def test_identical_species_reduce_to_pure_fluid():
    c = mixture().components[1]
    m = PengRobinsonMixture((c, replace(c, name="ethane-copy")), ((0, 0), (0, 0)))
    r = m.flash_pt(1e6, 320, [0.2, 0.8])
    pure = c.state(1e6, 320, derivatives=False)
    assert r["density_kg_m3"] == pytest.approx(pure["density_kg_m3"], rel=1e-12)
    assert r["molar_enthalpy_j_mol"] == pytest.approx(pure["enthalpy_j_kg"] * c.molar_mass_kg_mol)


def test_ternary_flash_and_balance():
    m = mixture()
    propane = PengRobinsonFluid("propane", 369.89, 4251200, 0.1521, 0.04409562, 73.6, 200, 500)
    ternary = PengRobinsonMixture(m.components + (propane,), ((0, 0, 0), (0, 0, 0), (0, 0, 0)))
    r = ternary.flash_pt(2e6, 250, [0.4, 0.3, 0.3])
    assert r["phase_count"] == 2
    assert r["component_balance_error"] < 1e-9
    assert r["log_fugacity_error"] < 1e-8


def test_mixture_letdown_energy_entropy_and_phase_mass_balance():
    from surface import MixtureLetdown

    m = mixture()
    r = MixtureLetdown("letdown", 1e6).run(3e6, 260, 0.25, [0.5, 0.5], m)
    assert r["outlet"]["temperature_k"] < r["inlet"]["temperature_k"]
    assert abs(r["energy_error_w"]) < 1e-6
    assert r["entropy_generation_w_k"] > 0
    assert sum(r["component_mass_rates_kg_s"]) == pytest.approx(0.25)
    z = sum(np.array(q["mole_fractions"]) * q["molar_fraction"] for q in r["outlet"]["phases"])
    assert z == pytest.approx([0.5, 0.5], abs=1e-9)
    same = MixtureLetdown("same", 3e6).run(3e6, 260, 0.25, [0.5, 0.5], m)
    assert same["energy_error_w"] == same["entropy_generation_w_k"] == 0
    stopped = MixtureLetdown("closed", 1e6).run(3e6, 260, 0, [0.5, 0.5], m)
    assert stopped["outlet"] is None
    with pytest.raises(PropertyModelError):
        MixtureLetdown("wrong", 4e6).run(3e6, 260, 0.25, [0.5, 0.5], m)


def test_cli_mixture_and_unknown_operation():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/run_surface_thermo.py"
    spec = importlib.util.spec_from_file_location("surface_thermo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = {
        "fluid": {"components": DATA["components"], "binary_interactions": [[0, 0], [0, 0]]},
        "operations": [
            {"type": "pt", "pressure_pa": 3e6, "temperature_k": 220, "mole_fractions": [0.5, 0.5]}
        ],
    }
    assert module.run_case(case)["operations"][0]["result"]["phase_count"] == 2
    case["operations"][0]["type"] = "nozzle"
    with pytest.raises(ValueError, match="Unsupported mixture operation"):
        module.run_case(case)


def test_component_chemical_potentials_from_total_gibbs():
    m = mixture(-0.03)
    p, t = 2e6, 300
    amounts = np.array([0.4, 0.6])
    state = m.state(p, t, amounts)
    chemical = np.log(amounts) + state["log_fugacity_coefficients"]

    def total_gibbs(n):
        x = n / n.sum()
        s = m.state(p, t, x)
        return n @ (np.log(x) + s["log_fugacity_coefficients"])

    for i in range(2):
        dn = np.eye(2)[i] * 1e-5
        tangent = (total_gibbs(amounts + dn) - total_gibbs(amounts - dn)) / (2e-5)
        assert tangent == pytest.approx(chemical[i], abs=1e-9)


def test_degenerate_pure_coexistence_needs_quality():
    c = mixture().components[1]
    m = PengRobinsonMixture((c, replace(c, name="copy")), ((0, 0), (0, 0)))
    p = c.saturation_at_temperature(240)["pressure_pa"]
    with pytest.raises(PropertyModelError, match="Degenerate coexistence"):
        m.flash_pt(p, 240, [0.4, 0.6])
