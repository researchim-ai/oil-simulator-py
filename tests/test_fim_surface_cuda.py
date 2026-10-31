"""CPU/CUDA parity of the physical coupling, including AD and graph replay.

Select the intended CUDA device externally with CUDA_VISIBLE_DEVICES.
"""

import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from test_fim_blackoil_surface import config, context

from simulator.blackoil_fim import BlackOilFIMSolver
from simulator.fluid import Fluid
from simulator.reservoir import Reservoir
from simulator.well import WellManager
from surface import BlackOilMixture, BlackOilPipe, HydraulicConvergenceError, StockDensities

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")


def build(device, backend="cpu_reference", rs=None, case=None):
    c = config() if case is None else copy.deepcopy(case)
    c["simulation"]["physical_surface_network"]["network_backend"] = backend
    dev = torch.device(device)
    r = Reservoir(c["reservoir"], device=dev)
    f = Fluid(c["fluid"], r, device=dev)
    if rs is not None:
        f.s_g.zero_()
        f.s_o = 1 - f.s_w
        f.rs = torch.full_like(f.pressure, rs)
    return BlackOilFIMSolver(r, f, WellManager(c["wells"], r), c["simulation"])


@pytest.mark.parametrize("pin", [3.1e6, 1.15e6])
def test_pipe_cuda_values_and_component_derivatives(pin):
    c = config()["simulation"]["physical_surface_network"]["network"]
    cpu = BlackOilMixture(c["pvt"], StockDensities(**c["stock_densities"]))
    gpu = cpu.to("cuda")
    assert gpu.rho.is_cuda and gpu.pressures.is_cuda and not cpu.rho.is_cuda
    pipe = BlackOilPipe("gpu", 300, 0.08, 30, 1e-5)
    m = [0.85, 2.0, 0.06]
    a = pipe.run(pin, m, cpu)
    b = pipe.run(pin, m, gpu)
    assert b["compute_device"].startswith("cuda")
    assert b["outlet_pressure_pa"] == pytest.approx(a["outlet_pressure_pa"], abs=0.0001)
    for k in a["pressure_derivatives"]:
        assert b["pressure_derivatives"][k] == pytest.approx(
            a["pressure_derivatives"][k], rel=1e-8, abs=0.0001
        )


@pytest.mark.parametrize("backend", ["cpu_reference", "torch_device"])
def test_coupled_cuda_jacobian_blocks_match_cpu(backend):
    cpu = build("cpu")
    gpu = build("cuda", backend)
    p, sw, x, pb, _ = context(cpu)

    def assemble(s):
        dev = s.device
        pp = p.to(dev)
        ww = sw.to(dev)
        xx = x.to(dev)
        s.wells.begin_step(*s.cell_state(pp, ww, xx, s.sat, s.ug), 0)
        return s.assemble_bo(
            pp, ww, xx, s.sat, s.masses(pp, ww, xx, s.sat, s.ug), 86400, pb.to(dev), ug=s.ug
        )

    ra, ja, a = assemble(cpu)
    rb, jb, b = assemble(gpu)
    assert b["B_full"].is_cuda and b["D_full"].is_cuda and b["Cv"].is_cuda
    for key in ["B_full", "D_full", "Cv", "r_w"]:
        assert torch.allclose(a[key], b[key].cpu(), rtol=1e-8, atol=1e-5), key
    assert torch.allclose(ra, rb.cpu(), rtol=1e-8, atol=1e-8)
    assert torch.allclose(ja, jb.cpu(), rtol=1e-8, atol=1e-8)
    dx = torch.linspace(0.1, 1, cpu.n_cells * 3, dtype=torch.float64).view(-1, 3)
    cpu.wells.add_local_schur(a, ja)
    gpu.wells.add_local_schur(b, jb)
    assert torch.allclose(
        cpu.wells.schur_correction(a, dx),
        gpu.wells.schur_correction(b, dx.cuda()).cpu(),
        rtol=1e-8,
        atol=1e-8,
    )
    assert torch.allclose(
        cpu.wells.recover_dpw(a, dx),
        gpu.wells.recover_dpw(b, dx.cuda()).cpu(),
        rtol=1e-8,
        atol=0.0001,
    )


@pytest.mark.parametrize("backend", ["cpu_reference", "torch_device"])
@pytest.mark.parametrize("rs", [None, 25.0])
def test_coupled_steps_match_cpu_and_keep_field_graphs(backend, rs):
    a = build("cpu", rs=rs)
    b = build("cuda", backend, rs=rs)
    assert a.step(86400) and b.step(86400)
    assert b._graphs is not None and len(b._graphs) > 0
    for key, tol in [("pressure", 0.0001), ("s_w", 1e-10), ("s_g", 1e-10), ("rs", 1e-8)]:
        assert torch.allclose(
            getattr(a.fluid, key), getattr(b.fluid, key).cpu(), rtol=0, atol=tol
        ), key
    for well in a.wells.surface_report["wells"]:
        aa = a.wells.surface_report["wells"][well]
        bb = b.wells.surface_report["wells"][well]
        assert aa["mode"] == bb["mode"]
        assert bb["component_mass_rates_kg_s"] == pytest.approx(
            aa["component_mass_rates_kg_s"], rel=1e-8, abs=1e-10
        )
    assert b.wells.surface_report["execution"]["network_device"].startswith(
        "cuda" if backend == "torch_device" else "cpu"
    )


def test_cuda_hydraulic_failure_rolls_back(monkeypatch):
    s = build("cuda")
    assert s.step(86400)
    old = s.fluid.pressure.clone()
    x = s.x.clone()
    report = s.wells.surface_report

    def failure(_sources):
        raise HydraulicConvergenceError("injected GPU surface failure")

    monkeypatch.setattr(s.wells.surface_network, "solve", failure)
    assert not s.step(86400, 86400)
    assert torch.equal(old, s.fluid.pressure) and torch.equal(x, s.x)
    assert s.wells.surface_report is report


@pytest.mark.parametrize("backend", ["cpu_reference", "torch_device"])
@pytest.mark.parametrize("mode", ["rate", "bhp_limit"])
def test_cuda_active_control_rows_match_cpu(backend, mode):
    c = config()
    for w in c["wells"]:
        if mode == "rate":
            w["control_value"] = 1.0
        else:
            w["bhp_min"] = 2.6
    a = build("cpu", case=c)
    b = build("cuda", backend, case=c)
    assert a.step(86400) and b.step(86400)
    assert b._graphs is not None and b._graphs
    for name in a.wells.surface_report["wells"]:
        aa = a.wells.surface_report["wells"][name]
        bb = b.wells.surface_report["wells"][name]
        assert aa["mode"] == bb["mode"] == mode
        assert aa["bhp_pa"] == pytest.approx(bb["bhp_pa"], abs=0.0001)
        assert bb["component_mass_rates_kg_s"] == pytest.approx(
            aa["component_mass_rates_kg_s"], rel=1e-8, abs=1e-10
        )


def test_async_return_packet_preserves_previous_values_and_pickle():
    import pickle

    import numpy as np

    from simulator.device_exchange import HostReturnPacket

    packet = HostReturnPacket()
    ref = torch.empty(0, dtype=torch.float64, device="cuda")
    a = np.arange(8192, dtype=np.float64)
    first = packet.upload(a, ref)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        second = packet.upload(a + 7, ref)
    stream.synchronize()
    torch.cuda.synchronize()
    assert torch.equal(first.cpu(), torch.from_numpy(a))
    assert torch.equal(second.cpu(), torch.from_numpy(a + 7))
    restored = pickle.loads(pickle.dumps(packet))
    assert restored._host is None and restored._event is None
    assert torch.equal(restored.upload(a + 9, ref).cpu(), torch.from_numpy(a + 9))
