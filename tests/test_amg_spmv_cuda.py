"""Native CSR SpMV on all AMG levels, including cached restriction."""
from pathlib import Path
import sys
import copy

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from solver.classical_amg import ClassicalAMG
from simulator.cuda_graphs import capture
from simulator.cpr_gpu import CPRPreconditionerGPU
from solver.csr_spmv import csr_spmv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


def _matrix(dtype):
    # Rectangular interpolation with an empty row and an unused column.
    crow = torch.tensor([0, 2, 2, 3, 5, 6], device='cuda')
    col = torch.tensor([0, 2, 1, 0, 2, 1], device='cuda')
    values = torch.tensor([.2, .8, 1., -.5, 1.5, .3], dtype=dtype, device='cuda')
    return torch.sparse_csr_tensor(crow, col, values, size=(5, 4))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('transpose', [False, True])
def test_csr_product_matches_cpu_and_repeats_deterministically(dtype, transpose):
    A = _matrix(dtype)
    amg = ClassicalAMG.__new__(ClassicalAMG)
    amg.levels = [{'P': A}]
    x = torch.arange(5 if transpose else 4, device='cuda', dtype=dtype) + .2
    reference = A.cpu().to_dense()
    if transpose:
        reference = reference.T
    expected = reference.double() @ x.cpu().double()
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        first = amg._spmv_csr(A, x, transpose)
        for _ in range(3):
            assert torch.equal(first, amg._spmv_csr(A, x, transpose))
        tol = 1e-6 if dtype == torch.float32 else 1e-14
        torch.testing.assert_close(first.cpu().double(), expected, rtol=tol, atol=tol)
        if transpose:
            cache = amg._transpose_csr(A)
            assert amg._transpose_csr(A) is cache
            A.values().mul_(2.)
            changed = amg._spmv_csr(A, x, True)
            assert amg._transpose_csr(A) is not cache
            torch.testing.assert_close(changed, first*2, rtol=tol, atol=tol)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def test_restriction_cache_replacement_and_graph_replay():
    amg = ClassicalAMG.__new__(ClassicalAMG)
    A = _matrix(torch.float64)
    amg.levels = [{'P': A}]
    x = torch.ones(5, dtype=torch.float64, device='cuda')
    first = amg._spmv_csr(A, x, True)
    graph, result = capture(lambda: amg._spmv_csr(A, x, True), x.device)
    x.mul_(3.)
    graph.replay()
    torch.testing.assert_close(result, first*3, rtol=1e-14, atol=1e-14)
    B = _matrix(torch.float64)
    amg.levels = [{'P': B}]
    amg._spmv_csr(B, x, True)
    assert id(A) not in amg._transpose_csr_cache
    assert amg._transpose_csr_cache[id(B)][0] is B


@pytest.mark.parametrize('nb', [2, 3])
def test_cpr_block_product_matches_dense_and_keeps_current_values(nb):
    crow = torch.tensor([0, 2, 4], device='cuda')
    col = torch.tensor([0, 1, 0, 1], device='cuda')
    values = torch.arange(4*nb*nb, device='cuda', dtype=torch.float64).reshape(4,nb,nb)/100.
    values[0] += 5*torch.eye(nb, device='cuda')
    values[3] += 6*torch.eye(nb, device='cuda')
    cpr = CPRPreconditionerGPU(values, crow, col, 2, torch.device('cuda'))
    assert cpr._J_bsr.crow_indices().dtype == torch.int32
    assert cpr._J_bsr.col_indices().dtype == torch.int32
    assert crow.dtype == col.dtype == torch.int64
    dense = torch.cat((torch.cat((values[0],values[1]),1),
                       torch.cat((values[2],values[3]),1)),0)
    x = torch.arange(2*nb, device='cuda', dtype=torch.float64) + .7
    expected = dense @ x
    torch.testing.assert_close(cpr.apply_A(x), expected, rtol=1e-14, atol=1e-14)
    # The native sparse wrapper must alias current block values, not a stale copy.
    values.mul_(2.)
    torch.testing.assert_close(cpr.apply_A(x), expected*2, rtol=1e-14, atol=1e-14)




@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_fused_long_rows_and_strided_input(dtype):
    n = 137  # crosses several fixed-size chunks in the GPU row reduction
    crow = torch.tensor([0, 0, n, n+1], device='cuda')
    col = torch.cat((torch.arange(n, device='cuda'), torch.tensor([3],device='cuda')))
    storage = torch.linspace(-1., 2., 2*(n+1), device='cuda', dtype=dtype)
    A = torch.sparse_csr_tensor(crow, col, storage[::2], size=(3,n))
    x = torch.linspace(.1, 1.3, 2*n, device='cuda', dtype=dtype)[::2]
    expected = A.cpu().to_dense().double() @ x.cpu().double()
    actual = csr_spmv(A,x)
    tol = 2e-6 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=tol, atol=tol)


def test_optional_triton_fallback(monkeypatch):
    import solver.csr_spmv as implementation
    A = _matrix(torch.float64)
    x = torch.arange(4,device='cuda',dtype=torch.float64)
    fused = csr_spmv(A,x)
    monkeypatch.setattr(implementation,'triton',None)
    torch.testing.assert_close(csr_spmv(A,x),fused,rtol=1e-14,atol=1e-14)


@pytest.mark.parametrize('model',['oil_water','blackoil'])
def test_native_linear_backend_preserves_coupled_fim_step(model):
    from simulator.simulation import Simulator
    if model == 'oil_water':
        from test_fim_wells import BASE, _objects
    else:
        from test_blackoil_fim import BASE, _objects
    cfg=copy.deepcopy(BASE)
    cfg['simulation'].update(fim_model=model,newton_tol_cnv=1e-6,newton_tol_mb=1e-10,
                             newton_relaxed_max_pv_fraction=0.,linear_tol_min=1e-8,
                             linear_tol_max=1e-5)
    sims=[]
    for device in ['cpu','cuda']:
        dev=torch.device(device)
        res,fluid,wells=_objects(cfg,dev)
        sim=Simulator(res,fluid,wells,cfg['simulation'],device=dev)
        assert sim.run_step(.01*86400.)
        sims.append(sim)
    for field in ['pressure','s_w','s_o','s_g','rs','rv']:
        cpu=getattr(sims[0].fluid,field,None)
        gpu=getattr(sims[1].fluid,field,None)
        if torch.is_tensor(cpu):
            torch.testing.assert_close(gpu.cpu(),cpu,rtol=1e-7,atol=1e-8)
    torch.testing.assert_close(sims[1].fim_solver.wells.p_bh.cpu(),
                               sims[0].fim_solver.wells.p_bh,rtol=1e-7,atol=1e-6)
