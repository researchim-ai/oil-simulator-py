import torch
from typing import Optional, Dict, Any, Tuple

# Local import of the vendored ClassicalAMG implementation
from .classical_amg import ClassicalAMG


_AMG_CACHE: Dict[Tuple[Any, ...], ClassicalAMG] = {}


def _to_csr_double(A: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Ensure a torch sparse CSR matrix in float64 on the target device.
    Accepts COO or CSR torch sparse tensors.
    """
    if A.layout == torch.sparse_coo:
        A = A.coalesce().to_sparse_csr()
    elif A.layout != torch.sparse_csr:
        # Convert dense or other layouts to CSR via COO
        A = A.to_sparse().coalesce().to_sparse_csr()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if A.dtype != torch.float64 or A.device != device:
        A = torch.sparse_csr_tensor(
            A.crow_indices().to(device=device),
            A.col_indices().to(device=device),
            A.values().to(device=device, dtype=torch.float64),
            size=A.size(),
            device=device,
            dtype=torch.float64,
        )
    return A


def amg_solve(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    tol: float = 1e-6,
    max_cycles: int = 20,
    theta: float = 0.25,
    max_levels: int = 10,
    coarsest_size: int = 200,
    device: Optional[str] = "auto",
    enable_equilibration: Optional[bool] = None,
    near_nullspace: Optional[torch.Tensor] = None,
    node_coords: Optional[torch.Tensor] = None,
    mixed_precision: bool = False,
    mixed_start_level: int = 2,
    cpu_offload: bool = False,
    offload_level: int = 3,
) -> torch.Tensor:
    """
    Solve A x = b using Classical RS-AMG V-cycles.
    - A: torch sparse (COO or CSR), preferably SPD M-matrix from IMPES pressure discretization
    - b: dense RHS (1-D)
    Returns x on the same device/dtype as b (float32 by default in simulator).
    """
    # Target device selection
    if device == "auto" or device is None:
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        target_device = torch.device(device)

    # Prepare optional geometry / near-nullspace
    basis_tensor: Optional[torch.Tensor] = None
    if near_nullspace is not None:
        basis_tensor = near_nullspace
        if basis_tensor.dim() == 1:
            basis_tensor = basis_tensor.unsqueeze(1)
        basis_tensor = basis_tensor.to(dtype=torch.float64, device='cpu')

    coords_tensor: Optional[torch.Tensor] = None
    if node_coords is not None:
        coords_tensor = node_coords
        if coords_tensor.dim() == 1:
            coords_tensor = coords_tensor.unsqueeze(1)
        coords_tensor = coords_tensor.to(dtype=torch.float64, device='cpu')

    # Prepare matrix and RHS in double on target device
    A_csr64 = _to_csr_double(A, device=target_device)
    b64 = b.to(target_device, dtype=torch.float64)

    n_total = A_csr64.size(0)
    if basis_tensor is not None and basis_tensor.size(0) != n_total:
        raise ValueError(
            f"near_nullspace size mismatch: expected {n_total}, got {basis_tensor.size(0)}"
        )
    if coords_tensor is not None and coords_tensor.size(0) != n_total:
        raise ValueError(
            f"node_coords size mismatch: expected {n_total}, got {coords_tensor.size(0)}"
        )

    # Build AMG hierarchy (equilibration is auto-detected inside ClassicalAMG)
    cache_key = (
        A_csr64.size(),
        float(theta),
        int(max_levels),
        int(coarsest_size),
        str(target_device),
        int(basis_tensor.size(1)) if basis_tensor is not None else 0,
        int(coords_tensor.size(1)) if coords_tensor is not None else 0,
        bool(mixed_precision),
        int(max(0, mixed_start_level)),
        bool(cpu_offload),
        int(max(0, offload_level)),
    )

    amg = _AMG_CACHE.get(cache_key)
    if amg is None:
        print(f"[AMG solve] building hierarchy (cache miss) key={cache_key}")
        amg = ClassicalAMG(
            A_csr64,
            theta=theta,
            max_levels=max_levels,
            coarsest_size=coarsest_size,
            near_nullspace=basis_tensor,
            node_coords=coords_tensor,
            mixed_precision=mixed_precision,
            mixed_start_level=mixed_start_level,
            cpu_offload=cpu_offload,
            offload_level=offload_level,
        )
        _AMG_CACHE[cache_key] = amg
    else:
        print(f"[AMG solve] reusing hierarchy (cache hit) key={cache_key}")
        amg.update_matrix(A_csr64, near_nullspace=basis_tensor, node_coords=coords_tensor)
    # Solve with V-cycles until tol
    x64 = amg.solve(b64, x0=None, tol=tol, max_iter=max_cycles)

    # Return to original b's device/dtype (float32 in our pipeline)
    return x64.to(b.device, dtype=b.dtype)


def amg_preconditioner(
    A: torch.Tensor,
    *,
    theta: float = 0.25,
    max_levels: int = 10,
    coarsest_size: int = 200,
    device: Optional[str] = "auto",
) -> Dict[str, Any]:
    """
    Build a right-preconditioner M(r) ≈ A^{-1} via AMG V-cycles.
    """
    if device == "auto" or device is None:
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        target_device = torch.device(device)
    A_csr64 = _to_csr_double(A, device=target_device)

    def _assess_quality(amg_obj: ClassicalAMG, A_mat: torch.Tensor, cycles: int = 2) -> float:
        tests = []
        n = A_mat.size(0)
        v_const = torch.ones(n, dtype=torch.float64, device=amg_obj.device)
        tests.append(v_const / (v_const.norm() + 1e-30))
        v_rand = torch.randn(n, dtype=torch.float64, device=amg_obj.device)
        tests.append(v_rand / (v_rand.norm() + 1e-30))
        v_grad = torch.linspace(0, 1, n, dtype=torch.float64, device=amg_obj.device)
        tests.append(v_grad / (v_grad.norm() + 1e-30))
        rels = []
        for idx, v in enumerate(tests):
            z = amg_obj.apply(v, cycles=cycles)
            Az = torch.sparse.mm(A_mat, z.unsqueeze(1)).squeeze(1)
            rel = (Az - v).norm() / (v.norm() + 1e-30)
            rels.append(rel.item())
            print(f"[AMG prec] test vec {idx}: quality={rel.item():.3e}")
        return max(rels)

    quality_tol = 0.05
    theta_candidates = [theta]
    if theta > 0.2:
        theta_candidates.append(0.2)
    theta_candidates.extend([0.15, 0.1])

    best_amg = None
    best_score = float('inf')
    amg = None
    for theta_try in theta_candidates:
        print(f"[AMG prec] building AMG with theta={theta_try:.3f}")
        amg_try = ClassicalAMG(
            A_csr64,
            theta=theta_try,
            max_levels=max_levels,
            coarsest_size=coarsest_size,
        )
        score = _assess_quality(amg_try, A_csr64, cycles=2)
        print(f"[AMG prec] quality score={score:.3e}")
        if score < best_score:
            best_score = score
            best_amg = amg_try
        if score <= quality_tol:
            amg = amg_try
            break
    if amg is None:
        raise RuntimeError(f"AMG preconditioner quality too low (best score={best_score:.3e})")

    def _apply(r: torch.Tensor) -> torch.Tensor:
        r64 = r.to(amg.device, dtype=torch.float64)
        z64 = amg.apply(r64, cycles=1)
        return z64.to(r.device, dtype=r.dtype)

    return {"apply": _apply, "device": amg.device}


