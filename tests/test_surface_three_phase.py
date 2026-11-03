"""Independent geometry/nozzle/energy oracles and coupled three-phase dynamics."""

import copy
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from surface import (
    EquipmentOperatingError,
    HydraulicConvergenceError,
    ThreePhaseFluid,
    ThreePhasePort,
    ThreePhaseSeparatorNetwork,
    ThreePhaseState,
    ThreePhaseStream,
    ThreePhaseVessel,
)
from surface.separator_geometry import SeparatorGeometry
from surface.three_phase_properties import disk_area_moment, frozen_nozzle, level_tensor
from surface.vessel_wall import VesselWall

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def fluid():
    return ThreePhaseFluid(850, 1000, 2000, 4180, 1700, 518)


def model(
    device="cpu", ports=(), names=("s",), gravity=0.0, wall=None, orientation="vertical", **kw
):
    vessels = [
        ThreePhaseVessel(n, SeparatorGeometry(1, 2, orientation), 1e4, 1e6, wall=wall)
        for n in names
    ]
    return ThreePhaseSeparatorNetwork(
        fluid(), vessels, ports, device=device, gravity_m_s2=gravity, maximum_dt_s=0.02, **kw
    )


def initial(m, pressures=None, water=0.4):
    return m.initialize(
        {
            n: {
                "pressure_pa": (pressures or {}).get(n, 3e5),
                "temperature_k": 300.0,
                "oil_volume_m3": 0.4,
                "water_volume_m3": water,
            }
            for n in m.names
        }
    )


def audit(result):
    for step in result["steps"]:
        assert max(map(abs, step["phase_mass_residuals_kg"])) < 1e-8
        assert abs(step["energy_residual_j"]) < 1e-4
        assert step["entropy_production_j_k"] >= -1e-5
        for q in step["vessels"].values():
            assert max(map(abs, q["phase_mass_residuals_kg"])) < 1e-8
            assert abs(q["energy_residual_j"]) < 1e-4
            assert 0 <= q["interface_level_m"] <= q["liquid_level_m"]


@pytest.mark.parametrize("device", DEVICES)
def test_horizontal_geometry_and_implicit_inverse_derivative(device):
    geometry = SeparatorGeometry(2.0, 3.0)
    values = np.r_[1e-12, 1e-8, 0.001, 0.05, 0.3, 0.5, 0.8, 0.999, 1 - 1e-9]
    for fraction in values:
        volume = torch.tensor(
            geometry.volume_m3 * fraction, dtype=torch.float64, device=device, requires_grad=True
        )
        h = level_tensor(volume, geometry)
        reference = geometry.level_m(float(volume.detach().cpu()))
        assert float(h.detach().cpu()) == pytest.approx(reference, abs=2e-11)
        derivative = torch.autograd.grad(h, volume)[0]
        expected = 1 / (
            2
            * geometry.length_m
            * math.sqrt(float(h.detach().cpu()) * (2 - float(h.detach().cpu())))
        )
        assert float(derivative.cpu()) == pytest.approx(expected, rel=1e-10)
    area, moment = disk_area_moment(torch.tensor(1.0, dtype=torch.float64, device=device), 2.0)
    assert float(area.cpu()) == pytest.approx(math.pi / 2)
    assert float(moment.cpu()) == pytest.approx(math.pi / 2 - 2 / 3, abs=1e-14)


@pytest.mark.parametrize("device", DEVICES)
def test_nozzle_reduces_to_nasa_gas_choke_and_liquid_orifice(device):
    f = fluid()
    t = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
    gamma = (f.gas_cv_j_kg_k + f.gas_constant_j_kg_k) / f.gas_cv_j_kg_k
    critical = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    expected = (
        3e5
        * math.sqrt(gamma / (f.gas_constant_j_kg_k * 300))
        * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    )
    for pb in (1e4, 1e5):
        flux, p, temperature, choked = frozen_nozzle(
            t(3e5), t(300.0), t(pb), t([0, 0, 1]), f, t(0), 0.0
        )
        assert bool(choked)
        assert float(flux.cpu()) == pytest.approx(expected, rel=2e-14)
        assert float(p.cpu()) == pytest.approx(3e5 * critical, rel=2e-14)
        assert float(temperature.cpu()) == pytest.approx(600 / (gamma + 1))
    flux, _, _, choked = frozen_nozzle(t(3e5), t(300.0), t(1e5), t([0, 1, 0]), f, t(2.0), 9.80665)
    assert not bool(choked)
    assert float(flux.cpu()) == pytest.approx(
        math.sqrt(2 * 1000 * (2e5 - 1000 * 9.80665 * 2)), rel=2e-14
    )


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_frozen_nozzle_matches_independent_scalar_maximum_and_tangent(device):
    f = fluid()
    weights = np.array([0.5, 0.1, 0.4])
    r = weights[2] * f.gas_constant_j_kg_k
    cp = weights[0] * 2000 + weights[1] * 4180 + weights[2] * (1700 + 518)
    vl = weights[0] / 850 + weights[1] / 1000

    def reference(p0):
        def negative(p):
            temp = 300 * (p / p0) ** (r / cp)
            dh = cp * (300 - temp) + vl * (p0 - p)
            return -math.sqrt(max(0, 2 * dh)) / (vl + r * temp / p)

        optimum = minimize_scalar(
            negative, bounds=(1000, p0), method="bounded", options={"xatol": 1e-5}
        )
        return -optimum.fun

    p = torch.tensor(3e5, dtype=torch.float64, device=device, requires_grad=True)
    flux, *_ = frozen_nozzle(
        p, p * 0 + 300, p * 0 + 1000, p.new_tensor(weights), f, p * 0, 0.0, mixed=True
    )
    assert float(flux.detach().cpu()) == pytest.approx(reference(3e5), rel=1e-10)
    derivative = torch.autograd.grad(flux, p)[0]
    finite = (reference(3e5 + 1) - reference(3e5 - 1)) / 2
    assert float(derivative.cpu()) == pytest.approx(finite, rel=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_closed_vessel_keeps_all_inventories_temperature_and_energy(device, orientation):
    m = model(device, orientation=orientation, gravity=9.80665)
    state = initial(m)
    before = m.inspect(state)
    result = m.advance(state, 0.04)
    assert result["state"] == {**state.to_dict(), "time_s": 0.04}
    assert result["properties"] == before
    audit(result)


@pytest.mark.parametrize("device", DEVICES)
def test_external_heat_matches_caloric_closed_form_and_utility_entropy(device):
    m = model(device)
    state = initial(m)
    masses = dict(state.vessels)["s"][:3]
    capacity = sum(x * c for x, c in zip(masses, m.fluid.heat_capacities))
    result = m.advance(state, 0.04, heat_duties_w={"s": 2000}, utility_temperatures_k={"s": 350})
    assert result["properties"]["s"]["temperature"] == pytest.approx(300 + 80 / capacity, abs=1e-10)
    audit(result)
    with pytest.raises(ValueError, match="utility"):
        m.advance(state, 0.01, heat_duties_w={"s": 100})
    with pytest.raises(EquipmentOperatingError, match="direction"):
        m.advance(state, 0.01, heat_duties_w={"s": 100}, utility_temperatures_k={"s": 280})


@pytest.mark.parametrize("device", DEVICES)
def test_three_phase_outlets_and_geometric_water_oil_interface(device):
    ports = [
        ThreePhasePort(k, "s", k, h, 0.01, 0.7, back_pressure_pa=1e5)
        for k, h in [("oil", 0.75), ("water", 0.2), ("gas", 1.7)]
    ]
    m = model(device, ports, gravity=9.80665)
    state = initial(m)
    result = m.advance(state, 0.04)
    audit(result)
    for i, phase in enumerate(("oil", "water", "gas")):
        removed = math.fsum(
            s["dt_s"] * s["ports"][phase]["mass_rate_kg_s"] for s in result["steps"]
        )
        assert dict(state.vessels)["s"][i] - result["state"]["vessels"]["s"][i] == pytest.approx(
            removed, abs=1e-10
        )
    assert result["steps"][0]["ports"]["gas"]["choked"] == 1
    assert (
        result["properties"]["s"]["interface_level_m"] < m.inspect(state)["s"]["interface_level_m"]
    )


@pytest.mark.parametrize("device", DEVICES)
def test_two_vessels_share_identical_component_and_total_energy_flux(device):
    port = ThreePhasePort(
        "link", "a", "gas", 1.7, 0.01, 0.7, target="b", target_height_m=1.7, allow_reverse=True
    )
    wall = VesselWall(2e4, 100, 10, 290)
    m = model(device, [port], names=("a", "b"), wall=wall, gravity=9.80665)
    state = initial(m, {"a": 3e5, "b": 2e5})
    before = m.inspect(state)
    result = m.advance(state, 0.04)
    audit(result)
    for component in range(3):
        assert sum(v[component] for _, v in state.vessels) == pytest.approx(
            sum(v[component] for v in result["state"]["vessels"].values()), abs=1e-10
        )
    energy0 = sum(q["energy_j"] + q["wall_energy_j"] for q in before.values())
    energy1 = sum(q["energy_j"] + q["wall_energy_j"] for q in result["properties"].values())
    exchange = math.fsum(s["external_energy_in_j"] for s in result["steps"])
    assert energy1 - energy0 == pytest.approx(exchange, abs=1e-5)
    reverse = initial(m, {"a": 2e5, "b": 3e5})
    rr = m.advance(reverse, 0.04)
    audit(rr)
    assert all(s["ports"]["link"]["mass_rate_kg_s"] < 0 for s in rr["steps"])


@pytest.mark.parametrize("device", DEVICES)
def test_dry_pickup_cannot_remove_absent_water_and_feed_creates_interface(device):
    port = ThreePhasePort("water", "s", "water", 0.1, 0.01, 0.7, back_pressure_pa=1e5)
    m = model(device, [port])
    state = initial(m, water=0)
    closed = m.advance(state, 0.04)
    assert closed["state"]["vessels"]["s"][1] == 0
    assert all(s["ports"]["water"]["mass_rate_kg_s"] == 0 for s in closed["steps"])
    feed = ThreePhaseStream.create(m.fluid, [0, 0.1, 0], 4e5, 300)
    result = m.advance(state, 0.04, feeds={"s": [feed]})
    audit(result)
    assert result["state"]["vessels"]["s"][1] == pytest.approx(0.004, abs=1e-10)
    assert result["properties"]["s"]["interface_level_m"] > 0


@pytest.mark.parametrize("device", DEVICES)
def test_entrained_phase_is_taken_from_inventory_not_created(device):
    port = ThreePhasePort(
        "oil", "s", "oil", 0.75, 0.005, 0.7, back_pressure_pa=2e5, pickup_affinities=(1, 0.01, 2)
    )
    m = model(device, [port])
    state = initial(m)
    result = m.advance(state, 0.02)
    audit(result)
    rates = result["steps"][0]["ports"]["oil"]
    assert rates["water_mass_rate_kg_s"] > 0 and rates["gas_mass_rate_kg_s"] > 0
    for i, k in enumerate(("oil", "water", "gas")):
        removed = math.fsum(
            s["dt_s"] * s["ports"]["oil"][k + "_mass_rate_kg_s"] for s in result["steps"]
        )
        assert dict(state.vessels)["s"][i] - result["state"]["vessels"]["s"][i] == pytest.approx(
            removed, abs=1e-10
        )


def test_domains_identity_and_failed_report_do_not_modify_initial_state():
    m = model()
    state = initial(m)
    saved = copy.deepcopy(state.to_dict())
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        m.advance(state, 0.1, max_substeps=1)
    assert state.to_dict() == saved
    with pytest.raises(EquipmentOperatingError, match="Passive"):
        m.advance(state, 0.01, feeds={"s": [ThreePhaseStream.create(m.fluid, [1, 0, 0], 1e5, 300)]})
    with pytest.raises(TypeError, match="PHASE stream"):
        m.advance(state, 0.01, feeds={"s": [object()]})
    assert m.to_dict() == ThreePhaseSeparatorNetwork.from_dict(m.to_dict()).to_dict()
    assert ThreePhaseState.from_dict(state.to_dict()) == state
    with pytest.raises(ValueError, match="model mismatch"):
        m.advance(replace(state, model_id="wrong"), 0.01)
    with pytest.raises(ValueError, match="initial field"):
        m.initialize(
            {
                "s": {
                    "pressure_pa": 3e5,
                    "temperature_k": 300,
                    "oil_volume_m3": 0.4,
                    "water_volume_m3": 0.4,
                    "oops": 1,
                }
            }
        )


@pytest.mark.parametrize("device", DEVICES)
def test_adiabatic_choked_blowdown_converges_to_closed_form(device):
    f = fluid()
    gamma = 1 + f.gas_constant_j_kg_k / f.gas_cv_j_kg_k
    area = math.pi * 0.03**2 / 4
    volume = math.pi / 4 * 2
    k = (
        0.7
        * area
        / volume
        * math.sqrt(gamma * f.gas_constant_j_kg_k)
        * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    )
    a = f.gas_constant_j_kg_k / f.gas_cv_j_kg_k
    factor = 1 + a / 2 * k * math.sqrt(300) * 0.1
    exact_t = 300 / factor**2
    exact_p = 3e5 / factor ** (2 * (1 + a) / a)
    errors = []
    for dt in (0.02, 0.01):
        port = ThreePhasePort("gas", "s", "gas", 1.0, 0.03, 0.7, back_pressure_pa=1e4)
        m = model(device, [port], temporal_error_control=False)
        data = m.to_dict()
        data["maximum_dt_s"] = dt
        m = ThreePhaseSeparatorNetwork.from_dict(data)
        state = m.initialize(
            {
                "s": {
                    "pressure_pa": 3e5,
                    "temperature_k": 300,
                    "oil_volume_m3": 0.0,
                    "water_volume_m3": 0.0,
                }
            }
        )
        result = m.advance(state, 0.1)
        audit(result)
        errors.append(abs(result["properties"]["s"]["pressure_pa"] - exact_p))
        assert result["properties"]["s"]["temperature"] == pytest.approx(exact_t, abs=0.02)
    assert errors[1] < 0.6 * errors[0]


@pytest.mark.parametrize("device", DEVICES)
def test_finite_wall_exchange_converges_to_independent_linear_ode(device):
    from scipy.linalg import expm

    wall = VesselWall(2e4, 100, 10, 290)
    results = []
    for dt in (0.02, 0.01):
        m = model(device, wall=wall, temporal_error_control=False)
        data = m.to_dict()
        data["maximum_dt_s"] = dt
        m = ThreePhaseSeparatorNetwork.from_dict(data)
        state = m.initialize(
            {
                "s": {
                    "pressure_pa": 3e5,
                    "temperature_k": 300,
                    "wall_temperature_k": 330,
                    "oil_volume_m3": 0.4,
                    "water_volume_m3": 0.4,
                }
            }
        )
        capacity = sum(a * b for a, b in zip(dict(state.vessels)["s"][:3], m.fluid.heat_capacities))
        matrix = np.array(
            [
                [-100 / capacity, 100 / capacity, 0],
                [100 / 2e4, -110 / 2e4, 10 * 290 / 2e4],
                [0, 0, 0],
            ]
        )
        exact = expm(matrix * 0.1) @ np.array([300, 330, 1.0])
        result = m.advance(state, 0.1)
        audit(result)
        actual = result["state"]["vessels"]["s"][3:]
        results.append(max(abs(np.array(actual) - exact[:2])))
    assert results[1] < 0.6 * results[0]


@pytest.mark.parametrize("device", DEVICES)
def test_implicit_level_and_interface_valves_act_on_new_state(device):
    ports = [
        ThreePhasePort(
            "water",
            "s",
            "water",
            0.2,
            0.01,
            0.7,
            back_pressure_pa=1e5,
            control_metric="interface_level_m",
            setpoint=0.5,
            opening_gain=5,
        ),
        ThreePhasePort(
            "oil",
            "s",
            "oil",
            0.75,
            0.01,
            0.7,
            back_pressure_pa=1e5,
            control_metric="liquid_level_m",
            setpoint=1.0,
            opening_gain=5,
        ),
    ]
    m = model(device, ports)
    state = initial(m)
    result = m.advance(state, 0.04)
    audit(result)
    for step in result["steps"]:
        for port, metric, setpoint in [
            ("oil", "liquid_level_m", 1.0),
            ("water", "interface_level_m", 0.5),
        ]:
            assert step["ports"][port]["opening"] == pytest.approx(
                min(1, max(0, 5 * (step["vessels"]["s"][metric] - setpoint))), abs=1e-12
            )
    stopped = m.advance(state, 0.04, openings={"oil": 0, "water": 0})
    assert stopped["state"] == {**state.to_dict(), "time_s": 0.04}


@pytest.mark.parametrize("device", DEVICES)
def test_three_phase_schedule_restart_replays_and_rejects_forgery(device):
    from surface import run_three_phase_schedule
    from surface.operation import _digest

    port = ThreePhasePort("water", "s", "water", 0.2, 0.01, 0.7, back_pressure_pa=1e5)
    m = model(device, [port])
    state = initial(m)
    periods = [
        {"duration_s": 0.02},
        {"duration_s": 0.02, "openings": {"water": 0}},
        {"duration_s": 0.02},
    ]
    full = run_three_phase_schedule(m, state, periods)
    prefix = run_three_phase_schedule(m, state, periods, stop_after=1)
    assert run_three_phase_schedule(m, state, periods, checkpoint=prefix) == full
    bad = copy.deepcopy(prefix)
    bad["records"][0]["steps"][0]["energy_residual_j"] += 1
    bad.pop("checksum")
    bad["checksum"] = _digest(bad)
    from surface import ConservationError

    with pytest.raises(ConservationError, match="replay"):
        run_three_phase_schedule(m, state, periods, checkpoint=bad)


@pytest.mark.parametrize("device", DEVICES)
def test_network_autodiff_jacobian_matches_independent_finite_differences(monkeypatch, device):
    ports = [
        ThreePhasePort(
            "mixed",
            "a",
            "oil",
            0.75,
            0.005,
            0.7,
            target="b",
            target_height_m=0.75,
            pickup_affinities=(1, 0.01, 1),
            allow_reverse=True,
        ),
        ThreePhasePort(
            "water",
            "a",
            "water",
            0.2,
            0.005,
            0.7,
            back_pressure_pa=1e5,
            control_metric="interface_level_m",
            setpoint=0.5,
            opening_gain=2,
        ),
    ]
    m = model(device, ports, names=("a", "b"), gravity=9.80665)
    state = initial(m, {"a": 3e5, "b": 2e5})
    solve = m.newton.solve
    checks = []

    def checked(fn, x, *args, **kwargs):
        jac = torch.func.jacfwd(fn)(x)
        columns = []
        for j in range(len(x)):
            h = 1e-6 * max(1, float(x[j].detach().cpu()))
            shift = torch.zeros_like(x)
            shift[j] = h
            columns.append((fn(x + shift) - fn(x - shift)) / (2 * h))
        finite = torch.stack(columns, dim=1)
        torch.testing.assert_close(jac, finite, rtol=2e-6, atol=2e-8)
        checks.append(1)
        return solve(fn, x, *args, **kwargs)

    monkeypatch.setattr(m.newton, "solve", checked)
    result = m.advance(state, 0.001)
    audit(result)
    assert checks


@pytest.mark.parametrize("device", DEVICES)
def test_real_sparse_separator_chain_without_scipy_or_full_dense_fallback(monkeypatch, device):
    import scipy.optimize
    import scipy.sparse.linalg

    names = tuple(f"v{i:02}" for i in range(27))
    ports = [
        ThreePhasePort(
            f"link{i}", names[i], "gas", 1.7, 0.002, 0.7, target=names[i + 1], target_height_m=1.7
        )
        for i in range(26)
    ]
    m = model(device, ports, names=names, temporal_error_control=False)
    state = initial(m, {n: 3e5 - i * 1000 for i, n in enumerate(names)})

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected CPU/SciPy solver")

    for module in (scipy.optimize, scipy.sparse.linalg):
        for name in ("least_squares", "root", "minimize", "spsolve", "splu"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    original = torch.linalg.solve_ex

    def bounded_dense(matrix, *args, **kwargs):
        assert matrix.shape[-1] <= 128
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve_ex", bounded_dense)
    result = m.advance(state, 0.001)
    audit(result)
    assert m.newton.matrix is not None and m.newton.matrix.size == 135
    assert len(m.newton.matrix.rows) < 135 * 135
    assert all(s["ports"]["link0"]["mass_rate_kg_s"] > 0 for s in result["steps"])


@pytest.mark.parametrize("device", DEVICES)
def test_vertical_hydrostatic_gas_mass_and_two_liquid_heads(device):
    m = model(device, gravity=9.80665)
    state = initial(m)
    q = m.inspect(state)["s"]
    area = math.pi / 4
    hw = 0.4 / area
    hl = 0.8 / area
    gap = 2 - hl
    k = 9.80665 * gap / (518 * 300)
    expected = 3e5 * area / 9.80665 * math.expm1(k)
    assert q["phase_masses_kg"][2] == pytest.approx(expected, rel=1e-12)
    expected_bottom = 3e5 * math.exp(k) + 9.80665 * (850 * (hl - hw) + 1000 * hw)
    assert q["bottom_pressure_pa"] == pytest.approx(expected_bottom, rel=1e-12)
