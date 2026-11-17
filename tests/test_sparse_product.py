"""Sparse CPU memory stays bounded; RAP and CUDA algebra remain equivalent."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from solver.classical_amg import rap_torch
from solver.sparse_product import sparse_product


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["csr", "coo", "csc"])
def test_rectangular_sparse_product_and_rap_match_dense_reference(dtype, layout):
    a = torch.tensor([[2.0, 0.0, -0.5], [0.0, 3.0, 1.0]], dtype=dtype)
    b = torch.tensor([[1.0, 0.0], [2.0, -1.0], [0.0, 4.0]], dtype=dtype)
    convert = lambda x: getattr(x, "to_sparse_" + layout)()
    result = sparse_product(convert(a), convert(b))
    assert result.layout == torch.sparse_csr
    torch.testing.assert_close(
        result.to_dense(), a @ b, rtol=1e-6 if dtype == torch.float32 else 1e-13, atol=1e-7
    )
    A = torch.tensor([[3.0, -1.0, 0.0], [-1.0, 4.0, -0.5], [0.0, -0.5, 2.0]], dtype=dtype)
    P = b
    ac = rap_torch(A.to_sparse_csr(), P.to_sparse_csr(), torch.device("cpu"))
    torch.testing.assert_close(
        ac.to_dense(), P.T @ A @ P, rtol=1e-6 if dtype == torch.float32 else 1e-13, atol=1e-7
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process RSS regression")
@pytest.mark.parametrize("operation", ["product", "amg"])
def test_repeated_products_do_not_accumulate_native_csr_allocations(operation):
    code = """
import torch,gc
from solver.sparse_product import sparse_product
from solver.classical_amg import ClassicalAMG
n=1024;i=torch.arange(n)
rows=[i];cols=[i];vals=[torch.full((n,),4.,dtype=torch.float64)]
for offset in [1,32]:
 a=i[:-offset];b=i[offset:]
 if offset==1:
  mask=a//32==b//32;a,b=a[mask],b[mask]
 rows.extend([a,b]);cols.extend([b,a]);vals.extend([-torch.ones(len(a),dtype=torch.float64)]*2)
A=torch.sparse_coo_tensor(torch.stack([torch.cat(rows),torch.cat(cols)]),torch.cat(vals),(n,n)).coalesce().to_sparse_csr()
def operation():
 if __import__('sys').argv[1]=='product':
  sparse_product(A,A)
 else:
  mg=ClassicalAMG(A,max_levels=15,theta=.25,mixed_precision=True,verbose=False)
  mg.solve(torch.ones(n,dtype=torch.float64))
for _ in range(20):operation()
gc.collect()
def rss():return int(open('/proc/self/statm').read().split()[1])*__import__('os').sysconf('SC_PAGE_SIZE')
before=rss()
for _ in range(150 if __import__('sys').argv[1]=='product' else 30):operation()
gc.collect();growth=rss()-before
assert growth<8*1024**2, f'Native sparse-product growth {growth} bytes'
"""
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    subprocess.run(
        [sys.executable, "-c", code, operation],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_keeps_native_sparse_product():
    a = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float64, device="cuda").to_sparse_csr()
    x = sparse_product(a, a)
    torch.testing.assert_close(x.to_dense(), a.to_dense() @ a.to_dense(), rtol=1e-12, atol=1e-12)
    assert x.is_cuda
