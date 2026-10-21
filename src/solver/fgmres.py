import torch
import time

def fgmres(A_mv, b, M_mv=None, max_iter=100, tol=1e-5, restart=20, verbose=False, stats=None):
    """
    Flexible GMRES (FGMRES) solver implemented in PyTorch.
    Solves Ax = b using a flexible preconditioner M.
    
    Args:
        A_mv: Function that computes A @ x. Returns tensor on same device.
        b: RHS tensor (n,).
        M_mv: Function that computes M^{-1} @ x (Preconditioner application).
              Can be a non-linear operator. Returns tensor on same device.
        max_iter: Maximum number of outer iterations (restarts).
        tol: Relative tolerance for convergence.
        restart: Number of inner iterations (Krylov subspace size) before restart.
        verbose: Print convergence info.
        stats: Optional dict, filled with iters, restarts, converged, history (rel. residual per iteration).
        
    Returns:
        x: Solution tensor.
        info: 0 if converged, 1 if not.
    """
    n = b.shape[0]
    device = b.device
    dtype = b.dtype
    
    history = []

    def _done(x, info, iters, restarts):
        if stats is not None:
            stats.update(iters=iters, restarts=restarts, converged=(info == 0), history=history)
        return x, info

    # Initial guess x0 = 0
    x = torch.zeros_like(b)
    
    # Initial residual r0 = b - A x0 = b
    r = b.clone()
    
    # Norm of RHS for relative tolerance
    b_norm = torch.norm(b).item()
    if b_norm < 1e-15:
        return _done(x, 0, 0, 0) # RHS is zero
        
    # Current residual norm
    r_norm = torch.norm(r).item()
    
    if r_norm / b_norm < tol:
        return _done(x, 0, 0, 0)
        
    # Allocate memory for Krylov basis
    # V: Basis for Krylov subspace (orthogonal)
    # Z: Preconditioned basis vectors
    # H: Hessenberg matrix
    V = torch.zeros((restart + 1, n), device=device, dtype=dtype)
    Z = torch.zeros((restart, n), device=device, dtype=dtype)
    H = torch.zeros((restart + 1, restart), device=device, dtype=dtype)
    
    # Givens rotation holders
    cs = torch.zeros(restart, device=device, dtype=dtype)
    sn = torch.zeros(restart, device=device, dtype=dtype)
    
    # RHS for the least squares problem
    g = torch.zeros(restart + 1, device=device, dtype=dtype)
    
    total_iters = 0
    
    for outer_iter in range(max_iter):
        # Arnoldi Iteration
        
        # Normalize first basis vector
        r_norm = torch.norm(r).item()
        V[0] = r / r_norm
        g[0] = r_norm
        g[1:] = 0
        
        for j in range(restart):
            total_iters += 1
            
            # 1. Preconditioning: z_j = M^{-1} v_j
            if M_mv is not None:
                Z[j] = M_mv(V[j])
            else:
                Z[j] = V[j]
                
            # 2. Matrix-vector: w = A z_j
            w = A_mv(Z[j])
            
            # 3. Gram-Schmidt Orthogonalization
            for i in range(j + 1):
                H[i, j] = torch.dot(V[i], w)
                w = w - H[i, j] * V[i]
                
            H[j + 1, j] = torch.norm(w)
            
            # Check for breakdown
            if H[j + 1, j] < 1e-14:
                # Happy breakdown or exact solution found?
                pass
                
            V[j + 1] = w / (H[j + 1, j] + 1e-15)
            
            # 4. Apply Givens Rotations to H column
            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp
                
            # 5. Compute new Givens rotation
            # eliminate H[j+1, j]
            h_jj = H[j, j]
            h_j1j = H[j + 1, j]
            
            sq = torch.sqrt(h_jj**2 + h_j1j**2)
            cs[j] = h_jj / sq
            sn[j] = h_j1j / sq
            
            H[j, j] = cs[j] * h_jj + sn[j] * h_j1j
            H[j + 1, j] = 0.0
            
            # 6. Apply rotation to RHS g
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            
            # Check convergence
            rel_error = abs(g[j + 1].item()) / b_norm
            history.append(rel_error)
            if verbose:
                print(f"    FGMRES Iter {total_iters}: Rel Res = {rel_error:.2e}")
                
            if rel_error < tol:
                # Reconstruct solution
                y = torch.linalg.solve_triangular(H[:j+1, :j+1], g[:j+1].unsqueeze(1), upper=True).reshape(-1)
                # x = x + Z[:, :j+1] @ y
                # Optimized accumulation
                x = x + (Z[:j+1].T @ y)
                return _done(x, 0, total_iters, outer_iter)
                
        # End of Restart cycle
        # Update solution
        y = torch.linalg.solve_triangular(H[:restart, :restart], g[:restart].unsqueeze(1), upper=True).reshape(-1)
        x = x + (Z.T @ y)
        
        # Recompute residual exactly to avoid drift
        # r = b - A x
        if outer_iter < max_iter - 1:
            Ax = A_mv(x)
            r = b - Ax
            r_norm = torch.norm(r).item()
            if r_norm / b_norm < tol:
                return _done(x, 0, total_iters, outer_iter + 1)
                
    return _done(x, 1, total_iters, max_iter) # Not converged

