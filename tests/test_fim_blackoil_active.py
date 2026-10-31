"""Regulated multiphase tree enters all FIM well blocks, not lagged sources."""

import copy
import pickle

import pytest
import torch
from test_fim_blackoil_surface import config, context, report_rates, solver

from surface import BlackOilPressureControl, ComponentStream, HydraulicConvergenceError


def active_config():
    c = config()
    pipes = c["simulation"]["physical_surface_network"]["network"]["pipes"]
    for pipe in pipes:
        if pipe["name"] == "collector":
            pipe.update(loss_coefficient=80, valve_opening=0.8)
    return c


def regulate(s, target=None):
    # Choose an interior operating point at the same linearization state.
    *_, assemble = context(s)
    assemble()
    sources = {
        w["node"]: ComponentStream.from_mass_rates(
            *w["component_mass_rates_kg_s"], s.wells.surface_network.model.densities
        )
        for w in s.wells.surface_report["wells"].values()
    }
    n = s.wells.surface_network
    if target is None:
        target = n.with_controls({"collector": 0.7}).solve(sources)["node_pressures_pa"]["header"]
    s.wells.set_surface_pressure_control(
        BlackOilPressureControl("collector", "header", target, 0.5, 1.0)
    )


@pytest.mark.parametrize("mode", ["manual", "target", "min", "max"])
@pytest.mark.parametrize("initial_rs", [None, 25.0])
def test_full_well_blocks_with_active_restriction(mode, initial_rs):
    s = solver(active_config(), initial_rs=initial_rs)
    if mode != "manual":
        regulate(s, {"min": 4.5e6, "max": 0.9e6}.get(mode))
    p, sw, x, pb, assemble = context(s)
    _, _, ws = assemble()
    B, D = ws["B_full"].clone(), ws["D_full"].clone()
    if mode != "manual":
        assert (
            s.wells.surface_report["network"]["pressure_control"]["status"]
            == {"target": "target", "min": "min_opening", "max": "max_opening"}[mode]
        )
    for j in range(2):
        a, b = pb.clone(), pb.clone()
        a[j] += 2
        b[j] -= 2
        *_, plus = assemble(pw=a)
        *_, minus = assemble(pw=b)
        assert torch.allclose((plus["r_w"] - minus["r_w"]) / 4, D[:, j], rtol=3e-5, atol=1e-5)
    for cell in (0, 2):
        for k, h in enumerate((2.0, 1e-6, 1e-6 if initial_rs is None else 0.001)):
            a, b = [t.clone() for t in (p, sw, x)], [t.clone() for t in (p, sw, x)]
            a[k][cell] += h
            b[k][cell] -= h
            *_, plus = assemble(*a)
            *_, minus = assemble(*b)
            expected = B[:, s.wells.conn_cell == cell, k].sum(1)
            assert torch.allclose(
                (plus["r_w"] - minus["r_w"]) / (2 * h), expected, rtol=3e-5, atol=0.003
            )


def test_coupled_mass_feedback_restart_and_atomic_control_change():
    results = []
    for opening in (1.0, 0.5):
        s = solver(active_config())
        s.wells.set_surface_controls({"collector": opening})
        old = s.masses(s.fluid.pressure.flatten(), s.fluid.s_w.flatten(), s.x, s.sat, s.ug).sum(1)
        assert s.step(86400)
        new = s.masses(s.fluid.pressure.flatten(), s.fluid.s_w.flatten(), s.x, s.sat, s.ug).sum(1)
        error = old - new - report_rates(s).sum(0) * 86400
        assert error.abs().max() < 0.001
        results.append(report_rates(s).sum())
    assert results[0] > results[1]
    before = s.wells.surface_network.to_dict()
    with pytest.raises(ValueError):
        s.wells.set_surface_controls({"collector": 0})
    assert s.wells.surface_network.to_dict() == before
    s.wells.set_surface_boundary_pressures({"outlet": 1.1e6})
    assert (
        next(p for p in s.wells.surface_network.pipes if p.name == "collector").valve_opening == 0.5
    )
    restored = pickle.loads(pickle.dumps(s))
    assert s.step(86400) and restored.step(86400)
    assert torch.equal(s.fluid.pressure, restored.fluid.pressure)


def test_regulator_failure_rolls_back_and_manual_conflict(monkeypatch):
    s = solver(active_config())
    regulate(s)
    with pytest.raises(ValueError, match="conflicts"):
        s.wells.set_surface_controls({"collector": 0.9})
    old = s.fluid.pressure.clone()
    report = copy.deepcopy(s.wells.surface_report)

    def fail(*args):
        raise HydraulicConvergenceError("regulator failed")

    monkeypatch.setattr(BlackOilPressureControl, "solve", fail)
    assert not s.step(86400)
    assert torch.equal(s.fluid.pressure, old)
    assert s.wells.surface_report == report


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("backend", ["cpu_reference", "torch_device"])
def test_regulated_blackoil_cuda_steps_and_well_blocks(backend):
    from test_fim_surface_cuda import build

    c = active_config()
    # A fixed, attainable header target for the beginning of the interval.
    c["simulation"]["physical_surface_network"]["pressure_control"] = {
        "valve": "collector",
        "node": "header",
        "target_pressure_pa": 2.18e6,
        "min_opening": 0.5,
        "max_opening": 1.0,
    }
    a, b = build("cpu", case=c), build("cuda", backend, case=c)
    assert a.step(86400) and b.step(86400)
    assert torch.allclose(a.fluid.pressure, b.fluid.pressure.cpu(), rtol=1e-10, atol=0.001)
    assert torch.allclose(a.fluid.s_w, b.fluid.s_w.cpu(), rtol=1e-9, atol=1e-10)
    assert torch.allclose(a.fluid.s_g, b.fluid.s_g.cpu(), rtol=1e-9, atol=1e-10)
    assert torch.allclose(report_rates(a), report_rates(b), rtol=1e-8, atol=1e-9)
    ca = a.wells.surface_report["network"]["pressure_control"]
    cb = b.wells.surface_report["network"]["pressure_control"]
    assert ca["status"] == cb["status"] == "target"
    assert ca["opening"] == pytest.approx(cb["opening"], abs=1e-7)
    assert b._graphs is not None
