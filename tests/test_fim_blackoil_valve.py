"""Local liquid trim in a black-oil FIM well tree, with full cross-well tangents."""

import copy
from dataclasses import asdict

import pytest
import torch
from test_fim_blackoil_surface import config, context, report_rates, solver

from surface import (
    BlackOilPressureControl,
    ComponentStream,
    EquipmentOperatingError,
    PropertyModelError,
)


def active_config():
    c = config()
    n = c["simulation"]["physical_surface_network"]["network"]
    n["valves"] = []
    for index in (1, 2):
        start = f"bottom_{index}"
        end = f"trim_out_{index}"
        n["nodes"].append({"name": end, "elevation_m": -40})
        next(p for p in n["pipes"] if p["start"] == start)["start"] = end
        n["valves"].append(
            {
                "name": f"trim{index}",
                "start": start,
                "end": end,
                "diameter_m": 0.012,
                "loss_coefficient": 50,
                "valve_opening": 0.8,
                "vapor_pressure_pa": 3000,
                "pressure_recovery_factor": 0.8,
            }
        )
    return c


def regulator(s):
    *_, assemble = context(s)
    assemble()
    sources = {
        w["node"]: ComponentStream.from_mass_rates(
            *w["component_mass_rates_kg_s"], s.wells.surface_network.model.densities
        )
        for w in s.wells.surface_report["wells"].values()
    }
    target = s.wells.surface_network.with_controls({"trim1": 0.7}).solve(sources)[
        "node_pressures_pa"
    ]["bottom_1"]
    return BlackOilPressureControl("trim1", "bottom_1", target, 0.5, 1.0)


@pytest.mark.parametrize("regulated", [False, True])
def test_fim_local_valve_full_jacobian(regulated):
    s = solver(active_config(), initial_rs=25)
    if regulated:
        s.wells.set_surface_pressure_control(regulator(s))
    p, sw, x, pb, assemble = context(s)
    _, _, ws = assemble()
    B, D = ws["B_full"].clone(), ws["D_full"].clone()
    for j in range(2):
        a, b = pb.clone(), pb.clone()
        a[j] += 2
        b[j] -= 2
        *_, ra = assemble(pw=a)
        *_, rb = assemble(pw=b)
        assert torch.allclose((ra["r_w"] - rb["r_w"]) / 4, D[:, j], rtol=3e-5, atol=1e-5)
    for cell in (0, 2):
        for k, h in enumerate((2.0, 1e-6, 0.001)):
            a, b = [t.clone() for t in (p, sw, x)], [t.clone() for t in (p, sw, x)]
            a[k][cell] += h
            b[k][cell] -= h
            *_, ra = assemble(*a)
            *_, rb = assemble(*b)
            assert torch.allclose(
                (ra["r_w"] - rb["r_w"]) / (2 * h),
                B[:, s.wells.conn_cell == cell, k].sum(1),
                rtol=3e-5,
                atol=0.003,
            )


def test_actual_local_valve_feedback_and_three_component_balance():
    rates = []
    for opening in (1.0, 0.5):
        s = solver(active_config(), initial_rs=25)
        s.wells.set_surface_controls({"trim1": opening, "trim2": opening})
        old = s.masses(s.fluid.pressure.flatten(), s.fluid.s_w.flatten(), s.x, s.sat, s.ug).sum(1)
        assert s.step(86400)
        new = s.masses(s.fluid.pressure.flatten(), s.fluid.s_w.flatten(), s.x, s.sat, s.ug).sum(1)
        assert (old - new - report_rates(s).sum(0) * 86400).abs().max() < 0.001
        rates.append(report_rates(s).sum())
        assert s.wells.surface_report["network"]["valves"]["trim1"]["vapor_pressure_margin_pa"] > 0
    assert rates[0] > rates[1]


def test_free_gas_and_unsafe_equipment_restore_state():
    s = solver(active_config())
    old = s.fluid.pressure.clone()
    with pytest.raises(PropertyModelError, match="free gas|flashing"):
        s.step(86400)
    assert torch.equal(s.fluid.pressure, old)
    c = active_config()
    for v in c["simulation"]["physical_surface_network"]["network"]["valves"]:
        v["pressure_recovery_factor"] = 0.02
    s = solver(c, initial_rs=25)
    old = s.fluid.pressure.clone()
    report = copy.deepcopy(s.wells.surface_report)
    with pytest.raises(EquipmentOperatingError, match="vena-contracta"):
        s.step(86400)
    assert torch.equal(s.fluid.pressure, old) and s.wells.surface_report == report


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("backend", ["cpu_reference", "torch_device"])
@pytest.mark.parametrize("regulated", [False, True])
def test_local_valve_cuda_parity_and_graphs(backend, regulated):
    from test_fim_surface_cuda import build

    c = active_config()
    if regulated:
        probe = solver(c, initial_rs=25)
        c["simulation"]["physical_surface_network"]["pressure_control"] = asdict(regulator(probe))
    a = build("cpu", rs=25, case=c)
    b = build("cuda", backend, rs=25, case=c)
    assert a.step(86400) and b.step(86400)
    assert torch.allclose(a.fluid.pressure, b.fluid.pressure.cpu(), rtol=1e-10, atol=0.001)
    assert torch.allclose(a.fluid.s_w, b.fluid.s_w.cpu(), rtol=1e-9, atol=1e-10)
    assert torch.allclose(report_rates(a), report_rates(b), rtol=1e-8, atol=1e-9)
    assert b._graphs is not None
