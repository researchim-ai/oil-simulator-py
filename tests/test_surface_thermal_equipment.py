"""Thermal phase change, HEM choking and independent perfect-gas limit."""

import pytest
from test_surface_peng_robinson import fluid

from surface import (
    EquilibriumNozzle,
    IdealGas,
    PropertyModelError,
    ThermalLetdown,
    ideal_gas_nozzle_flow,
)
from surface.peng_robinson import R


def test_real_fluid_letdown_flash_preserves_enthalpy_and_produces_entropy():
    f = fluid()
    r = ThermalLetdown("trim", 101325).run(2e6, 300.0, 0.25, f)
    assert r["outlet"]["phase"] == "two_phase"
    assert 0 < r["outlet"]["vapor_quality"] < 1
    assert r["outlet"]["temperature_k"] == pytest.approx(230.96204040305227, abs=1e-5)
    assert abs(r["energy_residual_w"]) < 1e-6 and r["entropy_generation_w_k"] > 0
    zero = ThermalLetdown("trim", 101325).run(2e6, 300.0, 0, f)
    assert zero["outlet"] is None
    equal = ThermalLetdown("equal", 2e6).run(2e6, 300.0, 0.25, f)
    assert equal["inlet"] == equal["outlet"]
    assert equal["energy_residual_w"] == equal["entropy_generation_w_k"] == 0
    with pytest.raises(PropertyModelError, match="boost"):
        ThermalLetdown("trim", 3e6).run(2e6, 300, 0.1, f)


def test_equilibrium_choking_plateau_and_phase_boundary():
    f = fluid()
    n = EquilibriumNozzle("trim", 1e-5, 101325, 0.8, scan_points=17)
    a = n.run(2e6, 300, 101325, f)
    b = n.run(2e6, 300, 3e5, f)
    c = n.run(2e6, 300, 1.5e6, f)
    assert a["regime"] == b["regime"] == "choked" and c["regime"] == "subcritical"
    assert a["mass_rate_kg_s"] == pytest.approx(b["mass_rate_kg_s"], rel=1e-10)
    assert c["mass_rate_kg_s"] < a["mass_rate_kg_s"]
    assert a["critical_pressure_pa"] == pytest.approx(974356.2524, rel=1e-5)
    assert a["throat"].get("vapor_quality", 0) < 1e-6
    assert a["outlet"]["vapor_quality"] > 0.1
    assert a["entropy_generation_w_k"] > 0
    for r in (a, b, c):
        assert abs(r["isentropic_energy_residual_w"]) < 1e-6
        assert abs(r["outlet_energy_residual_w"]) < 1e-6
    dense = EquilibriumNozzle("trim", 1e-5, 101325, 0.8, scan_points=49).run(2e6, 300, 101325, f)
    assert dense["critical_pressure_pa"] == pytest.approx(a["critical_pressure_pa"], abs=0.1)
    assert dense["mass_rate_kg_s"] == pytest.approx(a["mass_rate_kg_s"], rel=1e-6)


def test_equilibrium_nozzle_approaches_independent_ideal_gas_limit():
    f = fluid()
    n = EquilibriumNozzle("gas", 1e-4, 1.0, 0.8, scan_points=17, pressure_tolerance_pa=1e-7)
    r = n.run(100.0, 400.0, 10.0, f)
    g = IdealGas(R / f.molar_mass_kg_mol, f.ideal_cp_j_mol_k / (f.ideal_cp_j_mol_k - R))
    ideal = ideal_gas_nozzle_flow(100.0, 400.0, 10.0, g, 1e-4, 0.8)
    assert r["regime"] == "choked"
    assert r["mass_rate_kg_s"] == pytest.approx(ideal["mass_rate_kg_s"], rel=2e-5)
    assert r["critical_pressure_pa"] / 100 == pytest.approx(
        ideal["critical_pressure_ratio"], rel=2e-5
    )


def test_thermal_nozzle_closed_and_invalid_domains():
    f = fluid()
    n = EquilibriumNozzle("trim", 1e-5, 101325, scan_points=17)
    assert n.run(2e6, 300.0, 101325, f, opening=0)["throat"] is None
    narrow = EquilibriumNozzle("narrow", 1e-4, 95.0, scan_points=17).run(100.0, 400.0, 95.0, f)
    assert narrow["regime"] == "subcritical"
    assert not narrow["critical_pressure_resolved"] and narrow["critical_pressure_pa"] is None
    with pytest.raises(PropertyModelError, match="domain"):
        n.run(2e6, 300, 1e4, f)
    with pytest.raises(PropertyModelError, match="domain"):
        n.run(2e6, 300, 3e6, f)
