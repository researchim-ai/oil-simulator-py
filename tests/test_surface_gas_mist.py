"""Dilute gas-carried drops: force balance, moments, energy and capture."""

import json
import math
from pathlib import Path

import pytest
import torch

from surface.dispersed_gas_liquid_drift import DispersedGasLiquidClosure
from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_droplet_population import PRDropletPopulation
from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork


def law(**kw):
    data = {
        "gas_component_index": 0,
        "oil_component_index": 1,
        "water_component_index": 2,
        "droplet_diameter_m": 1e-5,
        "oil_viscosity_pa_s": 0.001,
        "water_viscosity_pa_s": 0.001,
        "gas_viscosity_pa_s": 1e-5,
        "surface_tension_n_m": 0.02,
    }
    return DispersedGasLiquidClosure(**{**data, **kw})


def closed(*, kernel=0.0, vertical=False, device="cpu"):
    raw = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/surface_pr_equilibrium_euler_plant_pump.json"
        ).read_text()
    )["model"]["transport"]["fluid"]
    f = EquilibriumFluid.from_dict(raw, device=device)
    node = PREquilibriumNode(
        "mist",
        math.pi * 0.01**2 / 4 * 0.1,
        kind="pipe_cell",
        length_m=0.1,
        diameter_m=0.01,
        elevation_change_m=0.1 if vertical else 0.0,
        drift_flux=law(),
        droplet_population=PRDropletPopulation("condensate", kernel),
    )
    m = PREquilibriumTransportNetwork(f, [node], gravity_m_s2=9.80665 if vertical else 0.0)
    s = m.initialize(
        {"mist": {"pressure_pa": 1e6, "temperature_k": 350.0, "mole_fractions": [0.8, 0.2, 0.0]}}
    )
    return m, s


@pytest.mark.parametrize("bulk", [-0.2, 0.0, 0.2])
@pytest.mark.parametrize("axial", [-9.80665, 0.0, 9.80665])
@pytest.mark.parametrize("interface", ["immobile", "clean"])
def test_droplet_force_balance_and_mass_barycenter(bulk, axial, interface):
    c = law(interface=interface)
    a = torch.tensor([0.995, 0.005], dtype=torch.float64)
    rho = a.new_tensor([10.0, 600.0])
    v, ok = c.velocities(
        a,
        rho,
        a.new_tensor(bulk),
        diameter_m=0.01,
        axial_gravity_m_s2=axial,
        gravity_m_s2=9.80665,
        phase_mole_fractions=torch.eye(3, dtype=a.dtype)[:2],
    )
    assert bool(ok)
    expected = -(600 - 10) * axial * 1e-10 / (18 * 1e-5)
    if interface == "clean":
        expected *= 3 * (0.001 + 1e-5) / (3 * 0.001 + 2 * 1e-5)
    assert float(v[1] - v[0]) == pytest.approx(expected, abs=1e-15)
    assert float((a * rho * v).sum() / (a * rho).sum()) == pytest.approx(bulk, abs=1e-15)


def test_no_gas_carrier_or_dense_flooding_is_rejected():
    c = law()
    for a, rho, x in [
        ([1.0], [600.0], [[0.0, 1.0, 0.0]]),
        ([0.5, 0.5], [10.0, 600.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    ]:
        aa = torch.tensor(a, dtype=torch.float64)
        _, ok = c.velocities(
            aa,
            aa.new_tensor(rho),
            aa.new_tensor(0.0),
            diameter_m=0.01,
            axial_gravity_m_s2=0.0,
            phase_mole_fractions=aa.new_tensor(x),
        )
        assert not bool(ok)


def test_closed_mist_has_actual_drop_stock_surface_energy_and_coalescence_feedback():
    m, s = closed()
    before = m._state_data(0, s.nodes[0][1])
    assert bool(before["population"]["present"])
    assert float(before["surface"]) > 0
    assert s.nodes[0][1].droplet_number > 0
    n0 = s.nodes[0][1].droplet_number
    kernel = 2 * m.nodes[0].volume_m3 / n0
    m, s = closed(kernel=kernel)
    new, r = m._step(s, 0.01, m.default_openings, [0.0], [350.0])
    assert new.nodes[0][1].droplet_number < n0
    assert new.nodes[0][1].temperature_k > s.nodes[0][1].temperature_k
    assert new.nodes[0][1].total_energy_j == pytest.approx(s.nodes[0][1].total_energy_j, abs=1e-8)
    assert max(map(abs, r["conservation_residuals"])) < 1e-7
    restored = PREquilibriumTransportNetwork.from_dict(json.loads(json.dumps(m.to_dict())))
    restored.inspect(new)


def test_mist_population_jacobian_includes_gas_settling_and_interface_energy():
    m, s = closed(kernel=1e-15, vertical=True)
    seeds = [m._seed(i, p) for i, (_, p) in enumerate(s.nodes)]
    y, _, _, ev, *_ = m._problem(s, seeds, 1e-6, m.default_openings, [0.0])
    j = torch.func.jacfwd(lambda z: ev(z)[0])(y)
    direction = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype) * torch.maximum(
        torch.ones_like(y), y.abs()
    )
    h = 1e-4
    finite = (ev(y + h * direction)[0] - ev(y - h * direction)[0]) / (2 * h)
    torch.testing.assert_close(j @ direction, finite, rtol=1e-5, atol=1e-7)


def test_independent_numpy_audit_checks_surface_energy_and_coalescence():
    from copy import deepcopy

    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    m, s = closed(kernel=1e-15)
    request = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": {"state": s.to_dict()},
                "schedule": [{"duration_s": 1e-4}],
            }
        )
    )
    report = json.loads(json.dumps(run(request)))
    assert audit(request, report)["accepted"]
    bad = deepcopy(report)
    bad["intervals"][0]["steps"][0]["state"]["nodes"]["mist"]["droplet_number"] *= 1.01
    with pytest.raises(ValueError):
        audit(request, bad)


@pytest.mark.parametrize("reverse", [False, True])
def test_finite_capture_and_reinjection_conserve_components_and_surface_energy(reverse):
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    from surface.pr_equilibrium_transport import PREquilibriumLink

    base, _ = closed()
    m = PREquilibriumTransportNetwork(
        base.fluid,
        [base.nodes[0], PREquilibriumNode("collector", 0.001, droplet_capture=True)],
        [PREquilibriumLink("capture", "mist", "collector", 0.002, source_port="side:capture")],
    )
    initial = {
        n: {
            "pressure_pa": 1.01e6
            if n == "collector" and reverse
            else (9.9e5 if n == "collector" else 1e6),
            "temperature_k": 350.0,
            "mole_fractions": [0.8, 0.2, 0.0],
        }
        for n in m.names
    }
    request = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [{"duration_s": 1e-5}],
            }
        )
    )
    report = json.loads(json.dumps(run(request)))
    assert audit(request, report)["accepted"]
    r = report["intervals"][0]["steps"][0]
    flux = r["links"]["capture"]["droplet_number_fluxes"]["condensate"]
    assert (flux < 0) is reverse
    assert report["intervals"][-1]["state"]["nodes"]["collector"].get("droplet_number") is None
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


def test_sectional_gas_mist_uses_gas_viscosity_in_physical_collision_kernel_and_slip():
    from dataclasses import replace

    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run
    from test_surface_pr_sectional_population import spec

    base, _ = closed(vertical=True)
    population = replace(
        spec(kernel=0.0, rates=(0.0, 0.0, 0.0, 0.0), power=0.0),
        family="condensate",
        coalescence_model="brownian_gravity",
        collision_efficiency=0.4,
        reference_volumes_m3=tuple(v * 0.125 for v in spec().reference_volumes_m3),
    )
    m = PREquilibriumTransportNetwork(
        base.fluid, [replace(base.nodes[0], droplet_population=population)], gravity_m_s2=9.80665
    )
    request = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": {
                    "nodes": {
                        "mist": {
                            "pressure_pa": 1e6,
                            "temperature_k": 350.0,
                            "mole_fractions": [0.8, 0.2, 0.0],
                        }
                    }
                },
                "schedule": [{"duration_s": 1e-6}],
            }
        )
    )
    report = json.loads(json.dumps(run(request)))
    assert audit(request, report)["accepted"]
    from surface.pr_equilibrium_transport import PREquilibriumTransportState

    d = m._state_data(0, PREquilibriumTransportState.from_dict(report["initial"]).nodes[0][1])
    p = d["population"]
    assert float(p["section_velocities_m_s"].max() - p["section_velocities_m_s"].min()) > 0


def test_absent_drops_stay_zero_but_liquid_flooding_cannot_hide_behind_zero_population():
    from surface.liquid import PropertyModelError

    m, _ = closed()
    state = m.initialize(
        {"mist": {"pressure_pa": 1e6, "temperature_k": 350.0, "mole_fractions": [0.9, 0.1, 0.0]}}
    )
    assert len(state.nodes[0][1].phase_roots) == 1
    assert state.nodes[0][1].droplet_number == 0.0
    with pytest.raises(PropertyModelError, match="regime"):
        m.initialize(
            {
                "mist": {
                    "pressure_pa": 1e6,
                    "temperature_k": 350.0,
                    "mole_fractions": [0.0, 1.0, 0.0],
                }
            }
        )


def test_continuum_guard_uses_gas_density_temperature_and_actual_drop_size():
    c = law()
    x = torch.eye(3, dtype=torch.float64)[:2]
    m = x.new_tensor([0.016, 0.086, 0.018])
    assert bool(c.continuum_valid(x, x.new_tensor([10.0, 600.0]), x.new_tensor(350.0), m, 1e-5))
    assert not bool(c.continuum_valid(x, x.new_tensor([1e-4, 600.0]), x.new_tensor(350.0), m, 1e-5))


def test_population_cannot_silently_change_gas_carrier_into_water_carrier():
    from dataclasses import replace

    from test_surface_oil_water_drift import closure

    from surface.pr_equilibrium_transport import PREquilibriumLink

    base, _ = closed()
    other = replace(base.nodes[0], name="emulsion", drift_flux=closure())
    with pytest.raises(ValueError, match="family"):
        PREquilibriumTransportNetwork(
            base.fluid,
            [base.nodes[0], other],
            [PREquilibriumLink("pipe", "mist", "emulsion", 0.01)],
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_direct_mist_transport_conserves_drop_count_and_energy_without_capture(reverse):
    from dataclasses import replace

    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    from surface.pr_equilibrium_transport import PREquilibriumLink

    base, _ = closed()
    m = PREquilibriumTransportNetwork(
        base.fluid,
        [base.nodes[0], replace(base.nodes[0], name="downstream")],
        [PREquilibriumLink("transfer", "mist", "downstream", 0.01)],
    )
    request = {
        "model": m.to_dict(),
        "initial": {
            "nodes": {
                n: {
                    "pressure_pa": (1.01e6 if reverse else 9.9e5) if n == "downstream" else 1e6,
                    "temperature_k": 350.0,
                    "mole_fractions": [0.8, 0.2, 0.0],
                }
                for n in m.names
            }
        },
        "schedule": [{"duration_s": 1e-5}],
    }
    report = json.loads(json.dumps(run(request)))
    assert audit(request, report)["accepted"]
    initial = sum(n["droplet_number"] for n in report["initial"]["nodes"].values())
    final = sum(n["droplet_number"] for n in report["intervals"][-1]["state"]["nodes"].values())
    assert final == pytest.approx(initial, rel=1e-10)
    flux = report["intervals"][0]["steps"][0]["links"]["transfer"]["droplet_number_fluxes"][
        "condensate"
    ]
    assert (flux < 0) is reverse
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_monodisperse_and_sectional_mist_cuda_state_matches_cpu():
    for sectional in (False, True):
        m, s = closed(kernel=1e-15)
        if sectional:
            from dataclasses import replace

            from test_surface_pr_sectional_population import spec

            population = replace(
                spec(kernel=1e-15, rates=(0.0, 0.0, 0.0, 0.0), power=0.0),
                family="condensate",
                reference_volumes_m3=tuple(v * 0.125 for v in spec().reference_volumes_m3),
            )
            m = PREquilibriumTransportNetwork(
                m.fluid, [replace(m.nodes[0], droplet_population=population)]
            )
            s = m.initialize(
                {
                    "mist": {
                        "pressure_pa": 1e6,
                        "temperature_k": 350.0,
                        "mole_fractions": [0.8, 0.2, 0.0],
                    }
                }
            )
        gpu = PREquilibriumTransportNetwork.from_dict(m.to_dict(), device="cuda")
        a, _ = m._step(s, 1e-5, m.default_openings, [0.0], [350.0])
        b, rb = gpu._step(s, 1e-5, gpu.default_openings, [0.0], [350.0])
        aa, bb = a.nodes[0][1], b.nodes[0][1]
        assert bb.droplet_number == pytest.approx(aa.droplet_number, rel=1e-9)
        assert bb.temperature_k == pytest.approx(aa.temperature_k, abs=1e-9)
        assert max(map(abs, rb["conservation_residuals"])) < 1e-7
