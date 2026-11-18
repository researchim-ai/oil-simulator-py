"""Short solves avoid capture; longer solves still replay current CPR data."""
from types import SimpleNamespace

import pytest
import torch

from simulator import cpr_gpu


@pytest.mark.parametrize('warmups', [1, 3, 5])
def test_delayed_capture_is_per_instance_and_refreshes_rhs(monkeypatch, warmups):
    def capture(fn, device, pool=None, stream=None):
        output = fn()
        return SimpleNamespace(replay=lambda: output.copy_(fn())), output
    monkeypatch.setattr(cpr_gpu, 'capture', capture)
    for _ in range(2):
        c = cpr_gpu.CPRPreconditionerGPU.__new__(cpr_gpu.CPRPreconditionerGPU)
        c._graph = None
        c._graph_pool = None
        c.graph_warmup_calls = warmups
        c.graph_eager_calls = c.graph_captures = c.graph_replays = 0
        c._matvec_eager = lambda r: 2*r
        for i in range(warmups+3):
            r = torch.tensor([float(i), -float(i)])
            torch.testing.assert_close(c._matvec_graph(r), 2*r, rtol=0, atol=0)
            assert c.graph_captures == int(i >= warmups)
        assert c.graph_eager_calls == warmups
        assert c.graph_replays == 3


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_capture_policy_rejected_before_matrix_work(value):
    with pytest.raises(ValueError, match='warmup'):
        cpr_gpu.CPRPreconditionerGPU(None, None, None, 0, 'cpu', graph_warmup_calls=value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_gpu_delayed_graph_matches_eager_for_changing_rhs():
    dev = torch.device('cuda')
    eye = torch.eye(3, dtype=torch.float64, device=dev)
    crow = torch.tensor([0, 2, 4], device=dev)
    col = torch.tensor([0, 1, 0, 1], device=dev)
    for factor in (1., 2.):
        values = torch.stack([4*factor*eye, -eye, -eye, 3*factor*eye])
        c = cpr_gpu.CPRPreconditionerGPU(values, crow, col, 2, dev, graph_warmup_calls=3)
        assert c.amg is not None
        for i in range(6):
            r = torch.arange(1., 7., dtype=torch.float64, device=dev)*(i+1)
            actual = c.matvec(r)
            torch.testing.assert_close(actual, c._matvec_eager(r), rtol=0, atol=0)
        assert isinstance(c._graph, tuple)
        assert c.graph_eager_calls == 3 and c.graph_captures == 1 and c.graph_replays == 3
