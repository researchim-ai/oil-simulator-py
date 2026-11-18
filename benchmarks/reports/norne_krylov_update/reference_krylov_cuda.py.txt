"""Fixed-order Givens rotations without launching a kernel per scalar op."""
import torch

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice
except ImportError:
    triton = None


if triton is not None:
    @triton.jit(do_not_specialize=['j'])
    def _givens_column(H, cs, sn, g, j, HS0: tl.constexpr, HS1: tl.constexpr):
        for i in range(j):
            a = tl.load(H + i*HS0 + j*HS1)
            b = tl.load(H + (i+1)*HS0 + j*HS1)
            c, s = tl.load(cs+i), tl.load(sn+i)
            tmp = c*a + s*b
            tl.store(H + (i+1)*HS0 + j*HS1, -s*a + c*b)
            tl.store(H + i*HS0 + j*HS1, tmp)
        a = tl.load(H + j*HS0 + j*HS1)
        b = tl.load(H + (j+1)*HS0 + j*HS1)
        norm = libdevice.sqrt_rn(a*a + b*b)
        c, s = libdevice.div_rn(a, norm), libdevice.div_rn(b, norm)
        tl.store(cs+j, c)
        tl.store(sn+j, s)
        tl.store(H + j*HS0 + j*HS1, c*a + s*b)
        tl.store(H + (j+1)*HS0 + j*HS1, 0.)
        rhs = tl.load(g+j)
        tl.store(g+j+1, -s*rhs)
        tl.store(g+j, c*rhs)


def rotate_givens(H, cs, sn, g, j):
    """Apply previous rotations, form the next, and rotate the RHS in place.

    Preserve the eager operation order and precision. No atomics, FMA,
    changes to orthogonalization, or relaxed convergence are involved.
    """
    if (triton is not None and H.is_cuda and torch.version.hip is None
            and H.dtype in (torch.float32, torch.float64)
            and cs.is_contiguous() and sn.is_contiguous() and g.is_contiguous()):
        _givens_column[(1,)](H, cs, sn, g, j, H.stride(0), H.stride(1),
                            num_warps=1, enable_fp_fusion=False)
        return
    for i in range(j):
        temp = cs[i]*H[i, j] + sn[i]*H[i+1, j]
        H[i+1, j] = -sn[i]*H[i, j] + cs[i]*H[i+1, j]
        H[i, j] = temp
    a, b = H[j, j], H[j+1, j]
    norm = torch.sqrt(a**2 + b**2)
    cs[j], sn[j] = a/norm, b/norm
    H[j, j] = cs[j]*a + sn[j]*b
    H[j+1, j] = 0.
    g[j+1] = -sn[j]*g[j]
    g[j] = cs[j]*g[j]
