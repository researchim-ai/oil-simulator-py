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
            assert f.step(86400.0, current_time=0.0)
    finally:
        CPRPreconditionerGPU.matvec = orig
    assert seen and all(seen)
