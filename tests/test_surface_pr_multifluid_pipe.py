"""Spatial multifluid transport: exact contacts, inertia, walls and sources."""

import cmath
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_interphase_relaxation import PRInterphaseRelaxation
from surface.pr_multifluid_pipe import PRMultifluidPipe, PRMultifluidPipeState

ROOT = Path(__file__).resolve().parents[1]


def setup(*, count=4, three=False, mode="periodic", dense=128, exchange=False):
    fluid = EquilibriumFluid.from_dict(
        json.loads((ROOT / "configs/surface_pr_signed_pump_reversal.json").read_text())["model"][
            "fluid"
        ]
    )
    n = 3 if three else 2
    pair = lambda c: tuple(tuple(0.0 if i == j else c for j in range(n)) for i in range(n))
    source = PRInterphaseRelaxation(
        fluid,
        ("vapor", "liquid", "liquid")[:n],
        pair(0.05 if exchange else 0.0),
        pressure_conductance_m3_pa_s=pair(1e-10 if exchange else 0.0),
        heat_conductance_w_k=pair(0.1 if exchange else 0.0),
    )
    pipe = PRMultifluidPipe(
        [source] * count, [0.4 / count] * count, 0.1, boundary_mode=mode, dense_limit=dense
    )
    return pipe


def initial(pipe, *, contact=False, velocity=0.02, pulse=False, composition=False):
    cells = []
    n = len(pipe.phase_roots)
    for i, vol in enumerate(pipe.volumes):
        a = 0.3 + 0.08 * math.sin(2 * math.pi * i / len(pipe.volumes)) if contact else 0.4
        fractions = [a, 1 - a] if n == 2 else [a, 0.4, 0.6 - a]
        phases = []
        for k in range(n):
            x = [0.0, 0.0, 0.0]
            x[k] = 1.0
            if composition and k == 0 and i == 0:
                x = [0.95, 0.05, 0.0]
            phases.append(
                {
                    "pressure_pa": 1e6
                    + (1000.0 * math.cos(2 * math.pi * i / len(pipe.volumes)) if pulse else 0.0),
                    "temperature_k": 350.0,
                    "mole_fractions": x,
                    "volume_m3": fractions[k] * vol,
                    "velocity_m_s": velocity,
                }
            )
        cells.append(phases)
    return pipe.initialize(cells)


@pytest.mark.parametrize("three", [False, True])
def test_moving_contact_preserves_pressure_temperature_and_matches_circulant(three):
    pipe = setup(count=8, three=three)
    state = initial(pipe, contact=True)
    before = state.to_dict()
    dt = 1e-6
    end, r = pipe.transport_step(state, dt)
    assert state.to_dict() == before
    theta = 2 * math.pi / len(pipe.sources)
    g = 1 / (
        1
        + dt * pipe.wave_speed / pipe.lengths[0] * (1 - math.cos(theta))
        + 1j * dt * 0.02 / pipe.lengths[0] * math.sin(theta)
    )
    for i, cell in enumerate(end.cells):
        expected = 0.3 + 0.08 * (g * cmath.exp(1j * theta * i)).imag
        assert cell.phases[0].volume_m3 / pipe.volumes[i] == pytest.approx(expected, abs=1e-9)
        for p in cell.phases:
            assert p.pressure_pa == pytest.approx(1e6, abs=2e-5)
            assert p.temperature_k == pytest.approx(350.0, abs=1e-8)
    assert max(map(abs, r["conservation_residuals"])) < 1e-7


def test_pressure_gradient_creates_distinct_phase_inertia_and_keeps_balances():
    pipe = setup()
    state = initial(pipe, pulse=True, velocity=0.0)
    end, r = pipe.transport_step(state, 1e-6)
    velocity = []
    for cell in end.cells:
        velocity.append(
            [
                p.momentum_kg_m_s
                / sum(
                    n * c.molar_mass_kg_mol
                    for n, c in zip(p.component_moles, pipe.fluid.components)
                )
                for p in cell.phases
            ]
        )
    assert max(abs(a - b) for a, b in velocity) > 1e-5
    assert max(map(abs, r["conservation_residuals"])) < 1e-7
    assert r["entropy_change_j_k"] >= -1e-7


@pytest.mark.parametrize("direction", [-1, 1])
def test_rigid_walls_reflect_phases_without_mass_or_energy_leak(direction):
    pipe = setup(count=2, mode="walls")
    state = initial(pipe, velocity=direction * 0.02)
    end, r = pipe.transport_step(state, 1e-6)
    before = sum(p.momentum_kg_m_s for c in state.cells for p in c.phases)
    after = sum(p.momentum_kg_m_s for c in end.cells for p in c.phases)
    assert abs(after) < abs(before)
    assert r["wall_impulse_kg_m_s"] == pytest.approx(after - before, abs=1e-10)
    for flux in r["wall_fluxes"]:
        for phase in flux:
            assert max(map(abs, phase[:-2])) < 1e-12
            assert abs(phase[-1]) < 1e-8
    assert max(map(abs, r["conservation_residuals"])) < 1e-7


def test_component_arrives_from_exact_zero_and_local_sources_are_same_operator():
    pipe = setup(count=3)
    state = initial(pipe, composition=True)
    end, r = pipe.transport_step(state, 1e-6)
    assert state.cells[1].phases[0].component_moles[1] == 0.0
    assert end.cells[1].phases[0].component_moles[1] > 0.0
    assert max(map(abs, r["conservation_residuals"])) < 1e-7
    source_pipe = setup(count=2, three=True, exchange=True)
    start = initial(source_pipe)
    one = [
        {
            "pressure_pa": (1.2e6, 0.8e6, 1e6)[k],
            "temperature_k": (340.0, 330.0, 335.0)[k],
            "mole_fractions": [float(j == k) for j in range(3)],
            "volume_m3": p.volume_m3,
            "velocity_m_s": (10.0, -1.0, 0.5)[k],
        }
        for k, p in enumerate(start.cells[0].phases)
    ]
    start = source_pipe.initialize([one, one])
    # Uniform but out of equilibrium: no spatial jump, nonzero local exchange.
    moved, rr = source_pipe.step(start, 1e-5)
    expected, _ = source_pipe.sources[0].step(start.cells[0], 1e-5)
    for a, b in zip(moved.cells[0].phases, expected.phases):
        assert a.pressure_pa == pytest.approx(b.pressure_pa, rel=1e-10)
        assert a.temperature_k == pytest.approx(b.temperature_k, abs=1e-8)
    assert moved.time_s == 1e-5
    assert all(c.time_s == moved.time_s for c in moved.cells)
    assert rr["source_inputs"][0]["time_s"] == 0.0


def test_sparse_pattern_dense_solution_and_derivatives(monkeypatch):
    dense = setup(count=4, dense=256)
    state = initial(dense, pulse=True)
    raw = dense.to_dict()
    raw["dense_limit"] = 1
    sparse = PRMultifluidPipe.from_dict(raw)
    ss = replace(state, model_id=sparse.model_id)
    a, _ = dense.transport_step(state, 1e-6)
    b, _ = sparse.transport_step(ss, 1e-6)
    for ca, cb in zip(a.cells, b.cells):
        for pa, pb in zip(ca.phases, cb.phases):
            assert pa.pressure_pa == pytest.approx(pb.pressure_pa, rel=1e-9)
    y, _, _, evaluate = sparse._problem(ss, 1e-6)

    def forbidden(*args, **kwargs):
        raise AssertionError("Nested source or equilibrium solve")

    monkeypatch.setattr(PRInterphaseRelaxation, "step", forbidden)
    monkeypatch.setattr(sparse.fluid, "flash_pt", forbidden)
    jac = torch.func.jacrev(lambda z: evaluate(z)[0])(y)
    mask = torch.zeros_like(jac, dtype=torch.bool)
    rows, cols = evaluate.sparsity
    mask[rows, cols] = True
    assert (~mask).any()
    assert float(jac[~mask].abs().max()) < 1e-12
    for col in (0, 2, 3, len(y) // 2, len(y) - 1):
        delta = torch.zeros_like(y)
        h = 1e-6 * max(1.0, abs(float(y[col])))
        delta[col] = h
        # Component coordinates at zero use the admissible one-sided derivative.
        if y[col] == 0 and col % (len(sparse.fluid.names) + 3) >= 3:
            fd = (evaluate(y + delta)[0] - evaluate(y)[0]) / h
            assert torch.allclose(jac[:, col], fd, atol=1e-4, rtol=2e-4)
        else:
            fd = (evaluate(y + delta)[0] - evaluate(y - delta)[0]) / (2 * h)
            assert torch.allclose(jac[:, col], fd, atol=1e-4, rtol=2e-4)


def test_full_interval_and_restart_are_atomic():
    pipe = setup(count=2, exchange=True)
    start = initial(pipe, contact=True, pulse=True)
    report = pipe.advance(start, 1e-7, maximum_attempts=1)
    end = PRMultifluidPipeState.from_dict(json.loads(json.dumps(report["state"])))
    assert end.time_s == 1e-7
    restored = PRMultifluidPipe.from_dict(json.loads(json.dumps(pipe.to_dict())))
    restored.inspect(end)
    continued, _ = restored.step(end, 1e-7)
    assert continued.time_s == 2e-7
    before = end.to_dict()
    with pytest.raises(ValueError):
        restored.step(end, 0.0)
    assert end.to_dict() == before


def test_quadratic_interface_work_is_not_product_of_average_pressure_and_speed():
    from surface.pr_multifluid_flux import path_fluctuations

    left = {
        "conserved": torch.zeros((2, 5), dtype=torch.float64),
        "flux": torch.zeros((2, 5), dtype=torch.float64),
        "alpha": torch.tensor([0.3, 0.7], dtype=torch.float64),
        "pressure": torch.tensor([1e6, 0.8e6], dtype=torch.float64),
        "velocity": torch.tensor([0.1, 0.0], dtype=torch.float64),
    }
    right = {
        **left,
        "alpha": torch.tensor([0.5, 0.5], dtype=torch.float64),
        "pressure": torch.tensor([1e6, 1.2e6], dtype=torch.float64),
        "velocity": torch.tensor([-0.1, 0.0], dtype=torch.float64),
    }
    _a, _b, details = path_fluctuations(left, right, 4000.0)
    # Exact integral of (.8e6+.4e6*x)*(.1-.2*x) over x=0..1.
    integral = 0.8e6 * 0.1 + (0.4e6 * 0.1 - 0.8e6 * 0.2) / 2 + (0.4e6 * (-0.2)) / 3
    expected = -integral * (0.5 - 0.7)
    assert float(details["phase_path"][1, -1]) == pytest.approx(expected, abs=1e-9)
    assert float(details["phase_path"][:, -1].sum()) == pytest.approx(0.0, abs=1e-12)


def test_second_source_failure_does_not_publish_partial_transport(monkeypatch):
    from surface.liquid import HydraulicConvergenceError

    pipe = setup(count=2, exchange=True)
    state = initial(pipe, contact=True, pulse=True)
    expected, _ = pipe.step(state, 1e-7)
    before = state.to_dict()
    original = pipe.sources[0].step
    calls = []

    def rejected(cell, dt):
        calls.append(1)
        if len(calls) == 2:
            raise HydraulicConvergenceError("forced source failure")
        return original(cell, dt)

    monkeypatch.setattr(pipe.sources[0], "step", rejected)
    with pytest.raises(HydraulicConvergenceError):
        pipe.step(state, 1e-7)
    assert state.to_dict() == before
    monkeypatch.undo()
    end, _ = pipe.step(state, 1e-7)
    assert end.to_dict() == expected.to_dict()


def test_wave_and_vanishing_phase_domains_reject_without_fallback():
    from surface.liquid import PropertyModelError

    pipe = setup(count=2)
    state = initial(pipe)
    raw = pipe.to_dict()
    raw["wave_speed_bound_m_s"] = 10.0
    invalid = PRMultifluidPipe.from_dict(raw)
    with pytest.raises(PropertyModelError):
        invalid.inspect(replace(state, model_id=invalid.model_id))

    raw = pipe.to_dict()
    raw["minimum_phase_fraction"] = 0.45
    invalid = PRMultifluidPipe.from_dict(raw)
    with pytest.raises(PropertyModelError):
        invalid.inspect(replace(state, model_id=invalid.model_id))


def test_contact_grid_refinement_converges_to_exact_advected_profile():
    errors = []
    dt = 1e-6
    for count in (8, 16, 32):
        pipe = setup(count=count)
        state = initial(pipe, contact=True)
        end, _ = pipe.transport_step(state, dt)
        error = 0.0
        for i, c in enumerate(end.cells):
            target = 0.3 + 0.08 * math.sin(2 * math.pi * i / count - 2 * math.pi * 0.02 * dt / 0.4)
            error += (c.phases[0].volume_m3 / pipe.volumes[i] - target) ** 2
        errors.append(math.sqrt(error / count))
    assert 1.7 < errors[0] / errors[1] < 2.2
    assert 1.7 < errors[1] / errors[2] < 2.2


@pytest.mark.parametrize(
    "name", ["multifluid_counterflow", "multifluid_three_phase", "multifluid_walls"]
)
def test_independent_spatial_source_audit_and_tamper_rejection(name, monkeypatch):
    from check_pr_multifluid_pipe import audit
    from run_pr_multifluid_pipe import run

    cfg = json.loads((ROOT / f"configs/surface_pr_{name}.json").read_text())
    report = json.loads(json.dumps(run(cfg)))

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent audit invoked runtime physics")

    monkeypatch.setattr(PRMultifluidPipe, "inspect", forbidden)
    monkeypatch.setattr(PRMultifluidPipe, "transport_step", forbidden)
    monkeypatch.setattr(PRInterphaseRelaxation, "step", forbidden)
    monkeypatch.setattr(EquilibriumFluid, "tensor_homogeneous", forbidden)
    assert audit(cfg, report)["accepted"]
    for field in ("face_fluxes", "phase_path_integrals"):
        damaged = json.loads(json.dumps(report))
        damaged["intervals"][0]["steps"][0]["transport"][field][0][0][-1] += 1000.0
        with pytest.raises(ValueError):
            audit(cfg, damaged)
    damaged = json.loads(json.dumps(report))
    damaged["intervals"][0]["steps"][0]["source_inputs"][0]["phases"][0]["component_moles"][0] += (
        0.01
    )
    with pytest.raises(ValueError):
        audit(cfg, damaged)
    changed = json.loads(json.dumps(cfg))
    changed["initial"]["cells"][0][0]["pressure_pa"] += 1000.0
    damaged = json.loads(json.dumps(report))
    damaged["config"] = changed
    with pytest.raises(ValueError):
        audit(changed, damaged)


def test_full_runner_restart_and_epoch_clock():
    from check_pr_multifluid_pipe import audit
    from run_pr_multifluid_pipe import run

    cfg = json.loads((ROOT / "configs/surface_pr_multifluid_counterflow.json").read_text())
    report = json.loads(json.dumps(run(cfg)))
    request = {**cfg, "initial": {"state": report["state"]}, "schedule": [{"duration_s": 1e-8}]}
    continued = json.loads(json.dumps(run(request)))
    assert continued["initial"] == report["state"]
    assert audit(request, continued)["accepted"]
    pipe = PRMultifluidPipe.from_dict(cfg["model"])
    state = PRMultifluidPipeState.from_dict(report["initial"])
    epoch = 1e9
    state = replace(state, time_s=epoch, cells=tuple(replace(c, time_s=epoch) for c in state.cells))
    duration = 12 * math.ulp(epoch)
    end = pipe.advance(state, duration, relative_tolerance=0.1, velocity_tolerance_m_s=0.1)
    assert end["state"]["time_s"] == epoch + duration
    assert all(c["time_s"] == end["state"]["time_s"] for c in end["state"]["cells"])
