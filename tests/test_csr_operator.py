import numpy as np
import torch

from linear_gpu.csr import build_7pt_csr
from solver.geom_amg import GeoSolver


def explicit_apply(Tx, Ty, Tz, nx, ny, nz, x_np):
    """Нативное применение оператора к вектору x (numpy)."""
    y = np.zeros_like(x_np)
    idx = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                diag = 0.0
                center = idx
                if i > 0:
                    t = Tx[k, j, i-1]
                    y[center] -= t * x_np[center - 1]
                    diag += t
                if i < nx - 1:
                    t = Tx[k, j, i]
                    y[center] -= t * x_np[center + 1]
                    diag += t
                if j > 0:
                    t = Ty[k, j-1, i]
                    y[center] -= t * x_np[center - nx]
                    diag += t
                if j < ny - 1:
                    t = Ty[k, j, i]
                    y[center] -= t * x_np[center + nx]
                    diag += t
                if nz > 1:
                    if k > 0:
                        t = Tz[k-1, j, i]
                        y[center] -= t * x_np[center - nx * ny]
                        diag += t
                    if k < nz - 1:
                        t = Tz[k, j, i]
                        y[center] -= t * x_np[center + nx * ny]
                        diag += t
                y[center] += diag * x_np[center]
                idx += 1
    return y


def test_csr_vs_explicit():
    torch.manual_seed(0)
    np.random.seed(0)
    nx, ny, nz = 6, 5, 4
    # random permeability positive
    kx = np.random.rand(nz, ny, nx) + 0.1
    ky = np.random.rand(nz, ny, nx) + 0.1
    kz = np.random.rand(nz, ny, nx) + 0.1

    # transmissibilities (harmonic) – копируем из GeoSolver._harmonic
    def harm(a, b, eps=1e-12):
        return 2 * a * b / (a + b + eps)

    Tx = harm(kx[..., :-1], kx[..., 1:])
    Ty = harm(ky[:, :-1, :], ky[:, 1:, :])
    Tz = harm(kz[:-1, :, :], kz[1:, :, :]) if nz > 1 else None

    indptr, indices, data = build_7pt_csr(Tx, Ty, Tz, nx, ny, nz)
    A = torch.sparse_csr_tensor(indptr, indices, data, dtype=torch.float64)

    x = np.random.randn(nx * ny * nz)
    x_t = torch.from_numpy(x).double()

    y_csr = torch.sparse.mm(A, x_t.unsqueeze(1)).squeeze(1).numpy()
    y_exp = explicit_apply(Tx, Ty, Tz, nx, ny, nz, x)

    assert np.allclose(y_csr, y_exp, rtol=1e-10, atol=1e-12) 