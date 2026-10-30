"""Fixed-order CSR matrix-vector product without sorting or atomic additions."""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit(do_not_specialize=[5])
    def _csr_rows(crow, col, values, x, out, n_rows, BLOCK: tl.constexpr, ROWS: tl.constexpr):
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        starts = tl.load(crow + rows, rows < n_rows, other=0)
        ends = tl.load(crow + rows + 1, rows < n_rows, other=0)
        width = tl.max(ends - starts, axis=0)
        dtype: tl.constexpr = tl.float64 if values.dtype.element_ty == tl.float64 else tl.float32
        total = tl.full((ROWS,), 0., dtype)
        offset = 0
        while offset < width:
            pos = starts[:, None] + offset + tl.arange(0, BLOCK)[None, :]
            valid = (rows[:, None] < n_rows) & (pos < ends[:, None])
            columns = tl.load(col + pos, valid, other=0)
            a = tl.load(values + pos, valid, other=0)
            b = tl.load(x + columns, valid, other=0)
            total += tl.sum(a * b, axis=1)
            offset += BLOCK
        tl.store(out + rows, total, rows < n_rows)


def csr_spmv(matrix, x):
    """One warp per row, fixed reduction tree; preserve float32/float64 levels.

    Triton is optional. The fallback also uses CSR row boundaries, avoiding
    deterministic scatter's repeated sorting. Inputs are validated CSR storage.
    """
    if (triton is None or not x.is_cuda or matrix.dtype not in (torch.float32, torch.float64)
            or torch.version.hip is not None):
        products = matrix.values() * x[matrix.col_indices()]
        return torch.segment_reduce(products, reduce='sum', offsets=matrix.crow_indices(), unsafe=True)
    n_rows = matrix.size(0)
    out = x.new_empty(n_rows)
    if n_rows:
        _csr_rows[(triton.cdiv(n_rows, 4),)](
            matrix.crow_indices().contiguous(), matrix.col_indices().contiguous(),
            matrix.values().contiguous(), x.contiguous(), out, n_rows,
            BLOCK=32, ROWS=4, num_warps=4, enable_fp_fusion=False)
    return out
