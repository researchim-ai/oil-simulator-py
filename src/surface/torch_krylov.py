"""Device-resident compact Krylov kernels for the surface Newton solver."""

import torch


def gmres_device(operator, b, diagonal, restart=24, cycles=2, *, preconditioner=None):
    """Fixed-work GMRES with compact MGS2 and ordered Givens recurrence.

    Same restart/cycle budget, no host convergence reads or CPU solve.
    """
    n = b.numel()
    k = min(restart, n)
    safe = torch.where(diagonal.abs() > 1e-12, diagonal, torch.where(diagonal < 0, -1e-12, 1e-12))
    precondition = (lambda v: v / safe) if preconditioner is None else preconditioner
    x = torch.zeros_like(b)
    for _ in range(cycles):
        r = precondition(b - operator(x))
        beta = torch.linalg.vector_norm(r)
        basis = [r / beta.clamp_min(1e-30)]
        H = torch.zeros((k + 1, k), dtype=b.dtype, device=b.device)
        g = torch.zeros(k + 1, dtype=b.dtype, device=b.device)
        g[0] = beta
        cs = []
        sn = []
        for j in range(k):
            w = precondition(operator(basis[j]))
            # Compact modified Gram--Schmidt: the sequential coefficients
            # satisfy (I + strictly_lower(Q^T Q)) h = Q^T w. Retain both
            # reorthogonalization passes without one launch per scalar dot.
            Q = torch.stack(basis[: j + 1], dim=1)
            lower = torch.eye(j + 1, dtype=b.dtype, device=b.device) + torch.tril(
                Q.T @ Q, diagonal=-1
            )
            for _ in range(2):
                h = torch.linalg.solve_triangular(
                    lower, (Q.T @ w)[:, None], upper=False, unitriangular=True
                ).squeeze(-1)
                H[: j + 1, j] += h
                w = w - Q @ h
            H[j + 1, j] = torch.linalg.vector_norm(w)
            basis.append(w / H[j + 1, j].clamp_min(1e-30))
            if j:
                # Apply the same ordered Givens rotations by solving their
                # bidiagonal recurrence a[i] + sn[i-1] a[i-1] = cs[i-1] h[i].
                cosine_old, sine_old = torch.stack(cs), torch.stack(sn)
                column = H[: j + 1, j].clone()
                rotation = torch.eye(j + 1, dtype=b.dtype, device=b.device) + torch.diag(
                    sine_old, diagonal=-1
                )
                rhs = torch.cat((column[:1], cosine_old * column[1:]))
                propagated = torch.linalg.solve_triangular(
                    rotation, rhs[:, None], upper=False, unitriangular=True
                ).squeeze(-1)
                H[:j, j] = cosine_old * propagated[:-1] + sine_old * column[1:]
                H[j, j] = propagated[-1]
            radius = torch.hypot(H[j, j], H[j + 1, j])
            cosine = torch.where(radius > 1e-30, H[j, j] / radius.clamp_min(1e-30), 1.0)
            sine = H[j + 1, j] / radius.clamp_min(1e-30)
            cs.append(cosine)
            sn.append(sine)
            H[j, j] = radius
            H[j + 1, j].zero_()
            old = g[j].clone()
            g[j] = cosine * old
            g[j + 1] = -sine * old
        upper = H[:k].clone()
        d = upper.diagonal()
        upper = (
            upper - torch.diag(d) + torch.diag(torch.where(d.abs() > 1e-25, d, torch.ones_like(d)))
        )
        y = torch.linalg.solve_triangular(upper, g[:k, None], upper=True).squeeze(-1)
        x = x + torch.stack(basis[:k], dim=1) @ y
    return x
