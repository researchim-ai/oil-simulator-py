import pytest
import torch
from solver.csr_spmv import csr_diagonal
from solver.classical_amg import ClassicalAMG


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('index_dtype', [torch.int32, torch.int64])
def test_csr_diagonal_reads_unsorted_long_empty_and_missing_rows(dtype, index_dtype):
    # Last duplicate wins; one long row traverses more than one kernel tile.
    columns = [0, 0, 1] + [0]*39 + [2, 1, 0] + [0, 2]
    crow = torch.tensor([0, 3, 3, 45, 47, 47], device='cuda', dtype=index_dtype)
    col = torch.tensor(columns, device='cuda', dtype=index_dtype)
    values = torch.arange(1, 48, device='cuda', dtype=dtype)
    a = torch.sparse_csr_tensor(crow, col, values, size=(5, 5))
    expected = torch.tensor([2, 0, 43, 0, 0], device='cuda', dtype=dtype)
    torch.testing.assert_close(csr_diagonal(a), expected, rtol=0, atol=0)
    # No cache may retain stale numerical values.
    a.values().mul_(-2)
    torch.testing.assert_close(csr_diagonal(a), expected*-2, rtol=0, atol=0)
    amg = object.__new__(ClassicalAMG)
    torch.testing.assert_close(amg._extract_diag(a), expected*-2, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_empty_csr_and_capture():
    a = torch.sparse_csr_tensor(torch.zeros(1,device='cuda',dtype=torch.int64),
        torch.empty(0,device='cuda',dtype=torch.int64),torch.empty(0,device='cuda',dtype=torch.float64),size=(0,0))
    assert csr_diagonal(a).numel()==0
    a=torch.eye(8,device='cuda',dtype=torch.float64).to_sparse_csr()
    csr_diagonal(a)
    from simulator.cuda_graphs import capture
    graph, out=capture(lambda: csr_diagonal(a), a.device)
    a.values().fill_(3)
    graph.replay()
    torch.testing.assert_close(out,torch.full((8,),3.,device='cuda',dtype=torch.float64),rtol=0,atol=0)
