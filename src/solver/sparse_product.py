"""Sparse-only products with bounded native CPU memory on our supported runtime.

The CPU CSR x CSR path in PyTorch 2.7.0+cu126 exhibits native allocation growth
in a repeated-product reproducer. COO SpGEMM avoids that path. No dense matrix
is formed; CUDA keeps its native CSR implementation.
"""

import torch

_SPARSE_LAYOUTS = (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc)


def sparse_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if (
        left.device.type == "cpu"
        and right.device.type == "cpu"
        and left.layout in _SPARSE_LAYOUTS
        and right.layout in _SPARSE_LAYOUTS
    ):
        a = left.to_sparse_coo().coalesce()
        b = right.to_sparse_coo().coalesce()
        return torch.sparse.mm(a, b).to_sparse_csr()
    return torch.matmul(left, right)
