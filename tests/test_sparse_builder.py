import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.sparse_utils import SparseMatrixBuilder  # noqa: E402


def test_explicit_pairs_reproduce_structured_stencil():
    b = SparseMatrixBuilder(3, 4, 2)
    L, R = b._structured_pairs()
    e = SparseMatrixBuilder(3, 4, 2, pairs=(torch.as_tensor(R), torch.as_tensor(L)))
    assert torch.equal(b.indptr, e.indptr) and torch.equal(b.indices, e.indices)
    assert b.n_nonzero == 24 + 2 * (2 * 4 * 2 + 3 * 3 * 2 + 3 * 4 * 1)
    assert torch.equal(b.indices[b.diag_map], torch.arange(24))
    assert torch.equal(b.indices[b.x_pos_map[:16]], torch.arange(8, 24))
    assert (b.x_pos_map[16:] == -1).all()


def test_nnc_pairs_and_position_lookup():
    b = SparseMatrixBuilder(2, 1, 3, pairs=(np.array([0, 1, 0]), np.array([1, 2, 5])))
    assert b.n_nonzero == 6 + 6
    pos = b.position(np.array([0, 5, 3]), np.array([5, 0, 4]))
    assert pos[0] >= 0 and pos[1] >= 0 and pos[2] == -1
    assert b.indices[pos[0]] == 5 and b.indices[pos[1]] == 0
    rows = torch.repeat_interleave(torch.arange(6), b.indptr[1:] - b.indptr[:-1])
    assert rows[pos[1]] == 5
