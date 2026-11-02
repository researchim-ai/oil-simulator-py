"""Analytic mixture calorics/acoustics and independently checked sonic flow."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    DynamicVessel,
    DynamicVesselNetwork,
    IdealGas,
    MixtureEquilibriumNozzle,
    PengRobinsonFluid,
    PengRobinsonMixture,
    PRNozzleConnection,
    PropertyModelError,
    PRSinglePhaseNozzle,
    ideal_gas_nozzle_flow,
    run_network_schedule,
    single_phase_properties,
    vessel_connection_from_dict,
)
from surface.peng_robinson import R


def fluid():
    c = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/surface_thermal_mixture.json").read_text()
    )["fluid"]
    return PengRobinsonMixture(
        tuple(PengRobinsonFluid(**x) for x in c["components"]), c["binary_interactions"]
    )


@pytest.mark.parametrize("p,t", [(1e5, 400), (2e6, 350), (4e6, 300)])
def test_analytic_cp_and_sound_against_thermodynamic_finite_differences(p, t):
    f = fluid()
    z = [0.4, 0.6]
    q = single_phase_properties(f, p, t, z)
    dt = 0.002
    dp = 10
    cp = (
        f.state(p, t + dt, z)["molar_enthalpy_j_mol"]
        - f.state(p, t - dt, z)["molar_enthalpy_j_mol"]
    ) / (2 * dt)
    assert q["cp_j_mol_k"] == pytest.approx(cp, rel=1e-7)
    sm = q["molar_entropy_j_mol_k"]
    a = f.flash_ps(p + dp, sm, z)
    b = f.flash_ps(p - dp, sm, z)
    sound2 = 2 * dp / (a["density_kg_m3"] - b["density_kg_m3"])
    assert q["sound_speed_m_s"] ** 2 == pytest.approx(sound2, rel=2e-6)


def test_identical_components_reduce_to_independent_pure_fluid_tangents():
    c = fluid().components[1]
    f = PengRobinsonMixture((c, replace(c, name="copy")), ((0, 0), (0, 0)))
    a = single_phase_properties(f, 1e6, 350, [0.3, 0.7])
    b = c.state(1e6, 350)
    assert a["cp_j_mol_k"] == pytest.approx(b["cp_j_kg_k"] * c.molar_mass_kg_mol, rel=1e-10)
    assert a["cv_j_mol_k"] == pytest.approx(b["cv_j_kg_k"] * c.molar_mass_kg_mol, rel=1e-10)
    assert a["sound_speed_m_s"] == pytest.approx(b["sound_speed_m_s"], rel=1e-10)


def test_sonic_nozzle_matches_ideal_gas_limit():
    f = fluid()
    z = [0.3, 0.7]
    cp = sum(x * c.ideal_cp_j_mol_k for x, c in zip(z, f.components))
    mw = sum(x * c.molar_mass_kg_mol for x, c in zip(z, f.components))
    q = PRSinglePhaseNozzle("gas", 1e-4, pressure_tolerance_pa=1e-7).run(100, 400, 10, z, f)
    ideal = ideal_gas_nozzle_flow(100, 400, 10, IdealGas(R / mw, cp / (cp - R)), 1e-4)
    assert q["mass_rate_kg_s"] == pytest.approx(ideal["mass_rate_kg_s"], rel=2e-5)
    assert q["critical_pressure_pa"] / 100 == pytest.approx(
        ideal["critical_pressure_ratio"], rel=2e-5
    )
    assert q["throat_mach"] == pytest.approx(1, abs=1e-7)


def test_real_fluid_flow_matches_full_equilibrium_reference_and_critical_plateau():
    f = fluid()
    z = [0.5, 0.5]
    nozzle = PRSinglePhaseNozzle("gas", 1e-5, discharge_coefficient=0.8)
    a = nozzle.run(2e6, 350, 5e5, z, f)
    b = nozzle.run(2e6, 350, 8e5, z, f)
    reference = MixtureEquilibriumNozzle("hem", 1e-5, 8e5, 0.8, scan_points=17).run(
        2e6, 350, 8e5, z, f
    )
    assert a["regime"] == b["regime"] == "choked"
    assert a["mass_rate_kg_s"] == pytest.approx(b["mass_rate_kg_s"], rel=1e-12)
    assert a["mass_rate_kg_s"] == pytest.approx(reference["mass_rate_kg_s"], rel=2e-6)
    assert a["critical_pressure_pa"] == pytest.approx(reference["critical_pressure_pa"], rel=2e-5)
    sub = nozzle.run(2e6, 350, 1.9e6, z, f)
    assert sub["regime"] == "subcritical" and sub["throat_mach"] < 1
    assert sub["mass_rate_kg_s"] < a["mass_rate_kg_s"]
    assert abs(a["outlet_energy_residual_w"]) < 1e-6


def test_phase_change_is_rejected_and_closed_or_unresolved_flow_is_explicit():
    f = fluid()
    n = PRSinglePhaseNozzle("gas", 1e-5)
    with pytest.raises(PropertyModelError):
        n.run(3e6, 220, 1e6, [0.5, 0.5], f)
    assert n.run(2e6, 350, 1e6, [0.5, 0.5], f, opening=0)["mass_rate_kg_s"] == 0
    narrow = PRSinglePhaseNozzle("narrow", 1e-5, minimum_pressure_ratio=0.95)
    q = narrow.run(2e6, 350, 1.98e6, [0.5, 0.5], f)
    assert q["critical_pressure_pa"] is None
    with pytest.raises(PropertyModelError, match="unresolved"):
        narrow.run(2e6, 350, 1e6, [0.5, 0.5], f)


def network_case():
    f = fluid()
    v = {n: DynamicVessel(n, 0.01, 1e4, 5e6) for n in ("a", "b")}
    s = {
        "a": v["a"].initialize(f, 2e6, 350, [0.5, 0.5]),
        "b": v["b"].initialize(f, 5e5, 350, [0.5, 0.5]),
    }
    c = PRNozzleConnection(
        "sonic", "a", "b", 1e-7, 3e6, 1, discharge_coefficient=0.8, allow_reverse=True
    )
    return f, v, s, c


def test_nonlinear_connection_is_current_state_coupled_and_conservative():
    f, v, s, c = network_case()
    r = DynamicVesselNetwork(v, (c,)).advance(s, f, 0.1, initial_dt_s=0.1)
    a, b = r["states"]["a"], r["states"]["b"]
    link = r["steps"][0]["connections"][0]
    z = np.array(a["component_moles"]) / sum(a["component_moles"])
    direct = PRSinglePhaseNozzle("check", c.throat_area_m2, discharge_coefficient=0.8).run(
        a["pressure_pa"], a["temperature_k"], b["pressure_pa"], z, f, with_outlet=False
    )
    mw = sum(x * component.molar_mass_kg_mol for x, component in zip(z, f.components))
    assert link["molar_rate_mol_s"] == pytest.approx(direct["mass_rate_kg_s"] / mw, rel=1e-9)
    assert link["regime"] == "choked" and link["throat_mach"] == pytest.approx(1, abs=1e-7)
    assert abs(r["steps"][0]["energy_residual_j"]) < 1e-5
    assert max(abs(x) for x in r["steps"][0]["component_residuals_mol"]) < 1e-8


def test_nozzle_network_reverse_closed_and_json_resume():
    f, v, s, c = network_case()
    c = replace(c, source="b", target="a")
    net = DynamicVesselNetwork(v, (c,))
    periods = [{"duration_s": 0.1}, {"duration_s": 0.1, "connection_factors": {"sonic": 0}}]
    full = run_network_schedule(f, net, s, periods, stepper={"initial_dt_s": 0.1})
    partial = run_network_schedule(f, net, s, periods, stepper={"initial_dt_s": 0.1}, stop_after=1)
    saved = copy.deepcopy(partial)
    resumed = run_network_schedule(
        f,
        net,
        s,
        periods,
        stepper={"initial_dt_s": 0.1},
        checkpoint=json.loads(json.dumps(partial)),
    )
    assert full == resumed and partial == saved
    assert full["records"][0]["steps"][0]["connections"][0]["molar_rate_mol_s"] < 0
    assert full["records"][1]["steps"][0]["connections"][0]["molar_rate_mol_s"] == 0
    assert (
        vessel_connection_from_dict(
            {"model": "pr_single_phase_nozzle", **{k: val for k, val in c.__dict__.items()}}
        )
        == c
    )
    with pytest.raises(ValueError):
        vessel_connection_from_dict({"model": "unknown"})
