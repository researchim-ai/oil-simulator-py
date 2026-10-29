"""CUDA-графы свойств ячеек и CPR дают те же числа, что и обычное исполнение."""
import contextlib
import io
import os
import sys

import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from simulator.cpr_gpu import CPRPreconditionerGPU  # noqa: E402
from simulator.eclipse_deck import EclipseCase  # noqa: E402
from simulator.simulation import Simulator  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна CUDA")
DECK = os.path.join(ROOT, "tests", "data", "spe9", "SPE9_CP.DATA")


def _solver(device):
    with contextlib.redirect_stdout(io.StringIO()):
        res, fluid, wm, sp = EclipseCase(DECK).build(device=device)
        return Simulator(res, fluid, wm, sp, device=device).fim_solver


def test_cell_state_graph_matches_eager():
    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    f = _solver(dev)
    p = f.fluid.pressure.reshape(-1).double()
    sw = f.fluid.s_w.reshape(-1).double()
    for scale in (1.0, 1.01, 0.97):
        pp = p * scale
        v, d = f.cell_state(pp, sw, f.x, f.sat, f.ug)
        ve, de = f._cell_state_eager(pp, sw, f.x, f.sat, f.ug)
        assert torch.equal(v, ve) and torch.equal(d, de)
    assert f._graphs and len(f._graphs) == 1


def test_cpr_graph_matches_eager():
    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    f = _solver(dev)
    seen = []
    orig = CPRPreconditionerGPU.matvec

    def matvec(self, r):
        y = orig(self, r)
        if isinstance(self._graph, tuple):
            seen.append(torch.equal(y, self._matvec_eager(r)))
        return y

    CPRPreconditionerGPU.matvec = matvec
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for day in range(3):
                assert f.step(86400.0, current_time=day * 86400.0)
            assert f._cpr_graph_pool["anchor"] is not None
            assert f.__getstate__()["_cpr_graph_pool"] is None
    finally:
        CPRPreconditionerGPU.matvec = orig
    assert seen and all(seen)


def test_stopped_well_mixing_on_cuda_conserves_components():
    from test_stopped_wells import _model
    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = _model(device=dev)
        s, f = sim.fim_solver, sim.fluid
        def mass():
            return s.masses(f.pressure.reshape(-1), f.s_w.reshape(-1), s.x, s.sat, s.ug).sum(1)
        before = mass().clone()
        assert sim.run_step(.01 * 86400)
    torch.testing.assert_close(mass(), before, rtol=1e-9, atol=1e-5)
    assert s.wells._producer_mix
    assert max(abs(x) for x in s.wells.wells[0].last_net_connection_rates.values()) < 1e-5
