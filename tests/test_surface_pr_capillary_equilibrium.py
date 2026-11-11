"""Heat-driven phase appearance with actual capillary free energy."""

import copy
import json

import pytest
import torch
from check_pr_capillary_equilibrium import audit
from test_surface_pr_active_capillary_axial import setup
from test_surface_pr_active_multifluid_pipe import phase

from surface.pr_capillary_equilibrium import PRCapillaryEquilibrium, thermal_entropy
from surface.pr_capillary_pressure import chemical_potentials_tv
from surface.pr_homogeneous_uv import caloric_tv


def case(device="cpu", wetting=False):
    model, _, _, _ = setup(device, wetting=wetting)
    f = model.pipe.fluid
    pressure = f.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    v = model.pipe.volumes[0]
    old = model.pipe.initialize(
        [
            {
                "liquid": phase(f, pressure, 200.0, "liquid", v * 0.05, x=(0.0, 1.0)),
                "gas": phase(f, pressure, 200.0, "vapor", v * 0.95, x=(0.0, 1.0)),
            }
        ]
        * 2
    )
    return PRCapillaryEquilibrium(model.source, maximum_iterations=24), old


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_heat_evaporates_all_liquid_and_cooling_recreates_it_without_a_material_inlet(
    device, tmp_path
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model, old = case(device)
    snapshot = old.to_dict()
    records = []
    for heat, bath, count in [(2.0, 450.0, 1), (-2.0, 150.0, 2)]:
        end, report = model.project(old, heat_j=[heat] * 2, heat_temperature_k=bath)
        assert audit(model, old, end, report)["accepted"]
        assert all(len(c) == count for c in end.cells)
        assert all(
            abs(v) <= tol
            for v, tol in zip(report["conservation_residuals"], report["roundoff_allowances"])
        )
        records.append({"before": old.to_dict(), "result": report})
        old = end
    assert snapshot != old.to_dict()
    (tmp_path / ("thermal-capillary-" + device + ".json")).write_text(
        json.dumps({"model": model.to_dict(), "records": records}, indent=2, allow_nan=False) + "\n"
    )


def test_wall_energy_and_nonuniform_neighbours_are_part_of_uv_and_pressure_jump():
    model, old = case(wetting=True)
    hot, first = model.project(old, heat_j=[2.0, 2.0], heat_temperature_k=450.0)
    assert audit(model, old, hot, first)["accepted"]
    cold, second = model.project(hot, heat_j=[-2.0, -2.0], heat_temperature_k=150.0)
    assert audit(model, hot, cold, second)["accepted"]
    assert all(len(c) == 2 for c in cold.cells)
    assert first["surface_energy_before_j"] != first["surface_energy_after_j"]
    altered = copy.deepcopy(second)
    altered["heat_j"][0] += 0.01
    with pytest.raises(ValueError):
        audit(model, hot, cold, altered)


def test_tv_entropy_has_scalar_pr_values_and_helmholtz_conjugates():
    model, _ = case()
    f = model.fluid
    n = f.tensor([0.01, 0.02])
    temperature = f.tensor(250.0)
    properties = f.homogeneous(1e6, 250.0, n / n.sum(), "vapor")
    volume = n.sum() * properties["molar_volume_m3_mol"]

    def helmholtz(z):
        numbers = z[:-1]
        v = z[-1] / numbers.sum()
        x = numbers / numbers.sum()
        q = caloric_tv(f, temperature, v, x)
        return numbers.sum() * (q["u"] - temperature * thermal_entropy(f, temperature, v, x, q))

    point = torch.cat((n, volume.reshape(1)))
    derivative = torch.func.jacfwd(helmholtz)(point)
    q = caloric_tv(f, temperature, volume / n.sum(), n / n.sum())
    mu = chemical_potentials_tv(f, temperature, volume / n.sum(), n / n.sum())
    torch.testing.assert_close(derivative[:-1], mu, atol=1e-8, rtol=1e-10)
    torch.testing.assert_close(derivative[-1], -q["p"], atol=1e-4, rtol=1e-10)
    torch.testing.assert_close(
        thermal_entropy(f, temperature, volume / n.sum(), n / n.sum(), q),
        properties["molar_entropy_j_mol_k"],
        atol=1e-10,
        rtol=1e-10,
    )


def test_running_axial_pipe_condenses_metastable_gas_through_declared_equilibrium_source():
    from check_pr_capillary_pressure import audit as normal_audit

    from surface.pr_capillary_axial import PRCapillaryAxialPipe
    from surface.pr_capillary_pressure import PRCapillaryPressure

    base, _, _, _ = setup(wetting=False)
    f = base.pipe.fluid
    pressure = f.components[1].saturation_at_temperature(225.0)["pressure_pa"] * 1.05
    old = base.pipe.initialize(
        [{"gas": phase(f, pressure, 225.0, "vapor", v, x=(0.0, 1.0))} for v in base.pipe.volumes]
    )
    raw = base.source.to_dict()
    raw["phase_equilibrium"] = True
    source = PRCapillaryPressure.from_dict(base.pipe, raw)
    model = PRCapillaryAxialPipe(source, volume_diffusion="none")
    end, report = model.step_coupled(old, 1e-6)
    transported = type(old).from_dict(report["axial"]["state"])
    assert all(len(c) == 2 for c in end.cells)
    assert normal_audit(source, transported, end, report["normal"])["accepted"]
    forged = copy.deepcopy(report["normal"])
    forged.pop("equilibrium_sweeps")
    with pytest.raises(ValueError, match="dispatch"):
        normal_audit(source, transported, end, forged)
    assert old.time_s == 0.0 and end.time_s == 1e-6
    restored = PRCapillaryAxialPipe.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.model_id == model.model_id


@pytest.mark.parametrize("coupling", ["imex", "newton"])
def test_equilibrium_source_runs_on_the_actual_shared_gravity_graph_and_conserves_its_stocks(
    coupling,
):
    from check_pr_capillary_facility import audit as graph_audit
    from test_surface_pr_capillary_facility import setup as gravity_case

    from surface.pr_capillary_facility import PRCapillaryFacility

    base, old = gravity_case(self_loop=True)
    raw = base.to_dict()
    raw["coupling"] = coupling
    raw["lines"] = {"out": raw["lines"]["out"]}
    raw["lines"]["out"]["parameters"].update(phase_support="active", phase_equilibrium=True)
    model = PRCapillaryFacility.from_dict(raw)
    old = model.initialize(
        {"out": dict(old.pipes)["out"]}, dict(old.columns), layer_slots=dict(old.layer_slots)
    )
    snapshot = old.to_dict()
    end, report = model.step(old, 1e-7)
    assert report["coupling"] == coupling and report["coupling_converged"]
    assert report["pipes"]["out"]["normal"]["thermochemical_equilibrium"]
    assert graph_audit(model.to_dict(), old, end, report)["accepted"]
    assert old.to_dict() == snapshot


def test_binary_metastable_gas_condenses_and_impossible_heat_bath_is_atomic():
    from surface.liquid import ConservationError

    base, old, _, _ = setup(wetting=False)
    f = base.pipe.fluid
    equilibrium = f.flash_pt(1e6, 196.2, [0.5, 0.5])
    gas = max(equilibrium["phases"], key=lambda p: float(p["molar_volume_m3_mol"].cpu()))
    x = tuple(gas["mole_fractions"].cpu().tolist())
    gas_only = base.pipe.initialize(
        [{"gas": phase(f, 1.02e6, 196.2, "vapor", v, x=x)} for v in base.pipe.volumes]
    )
    model = PRCapillaryEquilibrium(base.source)
    end, report = model.project(gas_only)
    assert all(len(c) == 2 for c in end.cells)
    assert audit(model, gas_only, end, report)["accepted"]
    model, old = case()
    snapshot = copy.deepcopy(old.to_dict())
    with pytest.raises(ConservationError, match="entropy"):
        model.project(old, heat_j=[2.0, 2.0], heat_temperature_k=100.0)
    assert old.to_dict() == snapshot


def test_nonlocal_capillary_equilibrium_converges_and_rejects_an_insufficient_sweep_budget():
    from surface.liquid import HydraulicConvergenceError
    from surface.pr_capillary_pressure import PRCapillaryPressure

    model, old = case()
    raw = model.source.to_dict()
    raw["surface_tension_n_m"] = 20.0
    raw["boundary_heights_m"] = [1e-4, 2e-4]
    source = PRCapillaryPressure.from_dict(model.pipe, raw)
    model = PRCapillaryEquilibrium(source)
    snapshot = copy.deepcopy(old.to_dict())
    hot, first = model.equilibrate(old, heat_j=[2.0, 2.0], heat_temperature_k=450.0)
    assert audit(model, old, hot, first)["accepted"]
    with pytest.raises(HydraulicConvergenceError, match="sweep budget"):
        model.equilibrate(hot, heat_j=[-2.0, -1.9], heat_temperature_k=150.0, maximum_sweeps=1)
    cold, second = model.equilibrate(hot, heat_j=[-2.0, -1.9], heat_temperature_k=150.0)
    assert second["converged"] and len(second["equilibrium_sweeps"]) >= 2
    assert second["maximum_equilibrium_residual"] <= 2e-11
    assert audit(model, hot, cold, second)["accepted"]
    assert old.to_dict() == snapshot


def test_counterflow_relaxation_uses_actual_momentum_operands_and_retains_energy():
    from dataclasses import replace

    model, old = case()
    f = model.fluid
    cells = []
    kinetic0 = 0.0
    for i, cell in enumerate(old.cells):
        row = []
        for name, ph in cell:
            mass = sum(n * c.molar_mass_kg_mol for n, c in zip(ph.component_moles, f.components))
            momentum = (1.0 if i == 0 else -1.0) * (1e-6 if name == "liquid" else -0.3e-6)
            energy = 0.5 * momentum**2 / mass
            kinetic0 += energy
            row.append(
                (
                    name,
                    replace(
                        ph, momentum_kg_m_s=momentum, total_energy_j=ph.total_energy_j + energy
                    ),
                )
            )
        cells.append(tuple(row))
    old = replace(old, cells=tuple(cells))
    assert model.pipe._totals(old)[-1] == 0.0
    end, report = model.equilibrate(old)
    assert audit(model, old, end, report)["accepted"]
    kinetic1 = sum(
        0.5
        * ph.momentum_kg_m_s**2
        / sum(n * c.molar_mass_kg_mol for n, c in zip(ph.component_moles, f.components))
        for cell in end.cells
        for _, ph in cell
    )
    assert kinetic1 < kinetic0
    assert abs(report["conservation_residuals"][-1]) <= report["roundoff_allowances"][-1]


def test_unsupported_flash_topology_is_rejected_instead_of_dropping_a_liquid_phase(monkeypatch):
    from types import SimpleNamespace

    from surface.pr_phase_equilibration import PRPhaseEquilibration

    base, old, _, _ = setup(wetting=False)
    model = PRCapillaryEquilibrium(base.source)
    phases = tuple(ph for _, ph in old.cells[0])
    snapshot = copy.deepcopy(old.to_dict())

    def unsupported(*args, **kwargs):
        return SimpleNamespace(
            phases=(*phases, phases[0]), phase_roots=("liquid", "liquid", "vapor")
        )

    monkeypatch.setattr(PRPhaseEquilibration, "recover", unsupported)
    with pytest.raises(ValueError, match="phase topology"):
        model.equilibrate(old)
    assert old.to_dict() == snapshot
