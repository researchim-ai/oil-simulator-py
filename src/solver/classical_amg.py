"""
Classical Algebraic Multigrid (Ruge-Stuben) solver.

TRUE RS-AMG с:
- Strong connections по отрицательным внедиагоналям с диагональным масштабированием
- MIS coarsening по графу сильных связей
- 1-point interpolation к сильным C-соседям
- Эффективный RAP через scatter (без SpGEMM/плотнения!)
"""

import torch
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional


def _torch_csr_to_scipy(A: torch.Tensor) -> sp.csr_matrix:
    if A.layout != torch.sparse_csr:
        raise TypeError("Expected torch.sparse_csr tensor")
    crow = A.crow_indices().cpu().numpy()
    col = A.col_indices().cpu().numpy()
    val = A.values().cpu().numpy()
    return sp.csr_matrix((val, col, crow), shape=A.size())


def _scipy_csr_to_torch(A_sp: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    crow = torch.from_numpy(A_sp.indptr).to(device=device, dtype=torch.int64)
    col = torch.from_numpy(A_sp.indices).to(device=device, dtype=torch.int64)
    val = torch.from_numpy(A_sp.data).to(device=device, dtype=torch.float64)
    return torch.sparse_csr_tensor(crow, col, val, size=A_sp.shape, device=device, dtype=torch.float64)


def find_strong_connections(A_csr: torch.Tensor, theta: float = 0.25) -> torch.Tensor:
    """Находит сильные связи по отрицательным внедиагоналям.
    
    Для M-матриц (диффузионные операторы): связь i→j сильная если:
    -a_ij / |a_ii| >= theta * max_k(-a_ik / |a_ii|)
    
    Args:
        A_csr: Sparse CSR матрица
        theta: Порог для сильных связей (обычно 0.25)
    
    Returns:
        Boolean mask размера nnz, True = сильная связь
    """
    crow, col, vals = A_csr.crow_indices(), A_csr.col_indices(), A_csr.values()
    n = crow.numel() - 1
    device, dtype = A_csr.device, A_csr.dtype
    
    # Row indices
    row_len = crow[1:] - crow[:-1]
    
    # Санитизация: row_len должен быть неотрицательным
    if (row_len < 0).any():
        raise RuntimeError(f"find_strong_connections: некорректный CSR - row_len имеет отрицательные значения")
    
    row_idx = torch.repeat_interleave(torch.arange(n, device=device), row_len)
    
    # Диагональ (для масштабирования)
    diag = torch.zeros(n, device=device, dtype=dtype)
    diag_mask = row_idx == col
    diag.scatter_(0, row_idx[diag_mask], vals[diag_mask].abs())
    
    # Масштабированные значения: a_ij / |a_ii|
    svals = vals / (diag[row_idx].abs() + 1e-30)
    
    # Только отрицательные внедиагонали
    off = col != row_idx
    neg = svals < 0
    cand = off & neg
    
    # Max |svals| среди отрицательных по строке
    neg_abs = (-svals).where(cand, torch.zeros_like(svals))
    max_per_row = torch.zeros(n, device=device, dtype=dtype)
    max_per_row.scatter_reduce_(0, row_idx, neg_abs, reduce='amax', include_self=False)
    
    # Порог: theta * max
    thr = theta * max_per_row[row_idx]
    strong = cand & (neg_abs >= thr)
    
    return strong


def classical_coarsening(A_csr: torch.Tensor, theta: float = 0.25) -> Tuple[torch.Tensor, int]:
    """Classical RS coarsening через MIS по графу сильных связей.
    
    Args:
        A_csr: Sparse CSR матрица
        theta: Порог для сильных связей
    
    Returns:
        cf_marker: 0=C-point, 1=F-point
        n_coarse: Число C-points
    """
    crow, col = A_csr.crow_indices(), A_csr.col_indices()
    n = crow.numel() - 1
    device = A_csr.device
    
    strong = find_strong_connections(A_csr, theta)
    
    # Делаем граф неориентированным: i<->j
    # Уникализируем рёбра {min(i,j), max(i,j)} чтобы избежать дублей
    row_len = crow[1:] - crow[:-1]
    row_idx = torch.repeat_interleave(torch.arange(n, device=device), row_len)
    i = row_idx[strong]
    j = col[strong]
    
    # Уникализация: ребро {min,max}
    e0 = torch.minimum(i, j)
    e1 = torch.maximum(i, j)
    E = torch.stack([e0, e1], 0)  # 2 x m
    G = torch.sparse_coo_tensor(E, torch.ones_like(e0, dtype=torch.float32, device=device),
                                size=(n, n)).coalesce()
    
    # Извлекаем уникальные рёбра
    ii, jj = G.indices()
    
    # Степень (число сильных соседей) - считаем для обоих направлений
    deg = torch.zeros(n, device=device, dtype=torch.float32)
    deg.scatter_add_(0, ii, torch.ones_like(ii, dtype=torch.float32))
    deg.scatter_add_(0, jj, torch.ones_like(jj, dtype=torch.float32))
    
    # Случайный tie-breaker
    w = torch.rand(n, device=device)
    score = deg + 1e-3 * w  # лексикографическое сравнение
    
    # Максимум score среди соседей (для обоих направлений)
    neigh_max = torch.zeros(n, device=device, dtype=score.dtype)
    neigh_max.scatter_reduce_(0, ii, score[jj], reduce='amax', include_self=False)
    neigh_max.scatter_reduce_(0, jj, score[ii], reduce='amax', include_self=False)
    
    # Местные максимумы -> C
    C = score > neigh_max
    cf_marker = torch.where(C, torch.tensor(0, device=device), torch.tensor(1, device=device))
    
    # Гарантия покрытия: F без C-соседей -> поднять в C
    has_C = torch.zeros(n, device=device, dtype=torch.int8)
    has_C.scatter_add_(0, ii, C[jj].to(torch.int8))
    has_C.scatter_add_(0, jj, C[ii].to(torch.int8))
    lift = (cf_marker == 1) & (has_C == 0)
    cf_marker[lift] = 0
    
    n_coarse = int((cf_marker == 0).sum().item())
    
    return cf_marker, n_coarse


def build_prolongation_rs_full(
    A_csr: torch.Tensor,
    cf_marker: torch.Tensor,
    strong_mask: torch.Tensor,
    basis: torch.Tensor | None,
    blend: float = 0.0,
    ls_reg: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_device = A_csr.device
    A_cpu = A_csr.to('cpu')
    cf = cf_marker.to('cpu').clone()
    strong = strong_mask.to('cpu')

    basis_cpu = None
    if basis is not None and basis.numel() > 0:
        basis_cpu = basis.to(torch.float64)
        if basis_cpu.dim() == 1:
            basis_cpu = basis_cpu.unsqueeze(1)
        basis_cpu = orthonormalize_columns(basis_cpu)

    crow = A_cpu.crow_indices()
    col = A_cpu.col_indices()
    val = A_cpu.values().to(torch.float64)
    n = crow.numel() - 1

    row_neigh = []
    row_vals = []
    row_strong = []
    row_val_dict = []
    for i in range(n):
        start = crow[i].item()
        end = crow[i + 1].item()
        neigh_i = col[start:end].tolist()
        vals_i = val[start:end].tolist()
        strong_i = strong[start:end].tolist()
        row_neigh.append(neigh_i)
        row_vals.append(vals_i)
        row_strong.append(strong_i)
        row_val_dict.append({int(c): float(v) for c, v in zip(neigh_i, vals_i)})

    diag = torch.zeros(n, dtype=torch.float64)
    for i in range(n):
        for c, v in zip(row_neigh[i], row_vals[i]):
            if c == i:
                diag[i] = v
                break
        if diag[i].abs() < 1e-14:
            diag[i] = 1.0

    promoted = True
    while promoted:
        promoted = False
        F_nodes = torch.where(cf == 1)[0]
        for i in F_nodes.tolist():
            neigh_i = row_neigh[i]
            strong_i = row_strong[i]
            has_c = False
            for c, s in zip(neigh_i, strong_i):
                if c != i and s and cf[c] == 0:
                    has_c = True
                    break
            if not has_c:
                cf[i] = 0
                promoted = True

    coarse_nodes = torch.where(cf == 0)[0]
    fine2coarse = -torch.ones(n, dtype=torch.int64)
    fine2coarse[coarse_nodes] = torch.arange(coarse_nodes.numel(), dtype=torch.int64)
    F_nodes_cpu = torch.where(cf == 1)[0]

    strong_C = []
    for i in range(n):
        neigh_i = row_neigh[i]
        strong_i = row_strong[i]
        C_list = []
        for c, s in zip(neigh_i, strong_i):
            if c != i and s and cf[c] == 0:
                C_list.append(int(c))
        strong_C.append(C_list)

    rows, cols_out, data = [], [], []

    # C-точки -> единичная строка
    for c in coarse_nodes.tolist():
        rows.append(c)
        cols_out.append(int(fine2coarse[c]))
        data.append(1.0)

    for i in F_nodes_cpu.tolist():
        C_i = strong_C[i]
        if not C_i:
            rows.append(i)
            cols_out.append(int(fine2coarse[i]))
            data.append(1.0)
            continue

        diag_i = diag[i].item()
        neigh_i = row_neigh[i]
        vals_i = row_vals[i]
        strong_i = row_strong[i]

        weights_dict: dict[int, float] = {}
        sum_strong = 0.0
        for c in C_i:
            a_ic = row_val_dict[i].get(c, 0.0)
            weights_dict[c] = weights_dict.get(c, 0.0) - a_ic / diag_i
            sum_strong += -a_ic

        weak_C_sum = 0.0
        F_strong: list[tuple[int, float]] = []
        F_weak_sum = 0.0
        for nbr, a_in, s in zip(neigh_i, vals_i, strong_i):
            if nbr == i:
                continue
            if cf[nbr] == 0:
                if not s:
                    weak_C_sum += float(-a_in)
            else:
                if s:
                    F_strong.append((int(nbr), float(a_in)))
                else:
                    F_weak_sum += float(-a_in)

        for nbr, a_in in F_strong:
            C_k = strong_C[nbr]
            if not C_k:
                F_weak_sum += float(-a_in)
                continue
            denom = 0.0
            contributions = []
            for c_k in C_k:
                a_kc = row_val_dict[nbr].get(c_k, 0.0)
                denom += -a_kc
                contributions.append((c_k, a_kc))
            if denom <= 1e-14:
                F_weak_sum += float(-a_in)
                continue
            factor = -a_in / diag_i
            for c_k, a_kc in contributions:
                weights_dict[c_k] = weights_dict.get(c_k, 0.0) + factor * (-a_kc / denom)
                if c_k not in C_i:
                    C_i.append(c_k)

        total_weak = weak_C_sum + F_weak_sum
        if total_weak > 0.0 and sum_strong > 1e-14:
            correction = total_weak / sum_strong
            for c in C_i:
                a_ic = row_val_dict[i].get(c, 0.0)
                weights_dict[c] = weights_dict.get(c, 0.0) + correction * (-a_ic / diag_i)

        weights_items = [(c_node, weights_dict.get(c_node, 0.0)) for c_node in C_i if abs(weights_dict.get(c_node, 0.0)) > 1e-16]
        if not weights_items:
            rows.append(i)
            cols_out.append(int(fine2coarse[i]))
            data.append(1.0)
            continue

        weights_vec = torch.tensor([weights_dict.get(c_node, 0.0) for c_node, _ in weights_items], dtype=torch.float64)
        weights_vec = torch.clamp(weights_vec, min=0.0)
        sum_weights = weights_vec.sum().item()
        if sum_weights < 1e-12:
            rows.append(i)
            cols_out.append(int(fine2coarse[i]))
            data.append(1.0)
            continue
        w_rs = weights_vec / sum_weights
        w_final = w_rs.clone()

        if basis_cpu is not None and blend > 0.0:
            dim_basis = basis_cpu.shape[1]
            if dim_basis > 0 and len(weights_items) >= dim_basis:
                B_c = torch.stack([basis_cpu[c] for c, _ in weights_items], dim=1)
                target = basis_cpu[i]
                w_final = solve_energy_weights(B_c, target, w_rs, blend, ls_reg)

        w_final = torch.clamp(w_final, min=0.0)
        sum_final = w_final.sum().item()
        if sum_final <= 1e-12:
            rows.append(i)
            cols_out.append(int(fine2coarse[i]))
            data.append(1.0)
            continue
        w_final = w_final / sum_final

        for (c_node, _), w_val in zip(weights_items, w_final.tolist()):
            rows.append(i)
            cols_out.append(int(fine2coarse[c_node]))
            data.append(float(w_val))

    n_coarse_actual = int(fine2coarse.max().item() + 1) if coarse_nodes.numel() > 0 else 0
    if n_coarse_actual == 0:
        n_coarse_actual = 1

    rows_tensor = torch.tensor(rows, dtype=torch.int64)
    cols_tensor = torch.tensor(cols_out, dtype=torch.int64)
    data_tensor = torch.tensor(data, dtype=torch.float64)

    P_coo = torch.sparse_coo_tensor(
        torch.stack([rows_tensor, cols_tensor]),
        data_tensor,
        size=(n, n_coarse_actual),
    ).coalesce()
    P_csr = P_coo.to_sparse_csr().to(orig_device)
    return (
        P_csr,
        cf.to(orig_device),
        fine2coarse.to(orig_device),
        coarse_nodes.to(orig_device),
        F_nodes_cpu.to(orig_device, dtype=torch.int64),
    )


def rap_onepoint_gpu(Af_csr: torch.Tensor,
                     parent_idx: torch.Tensor,
                     *,
                     weights: torch.Tensor | None = None,
                     n_coarse: int | None = None) -> torch.Tensor:
    """
    Galerkin Ac = P^T Af P при 1-point интерполяции.
    
    Без spspmm, полностью на GPU, O(nnz(Af)).
    
    Args:
        Af_csr: Fine-level матрица (CSR)
        parent_idx: parent_idx[i] = номер coarse-родителя узла i
        weights: P[i, parent[i]] (опционально, по умолчанию 1)
        n_coarse: Число coarse-узлов (если None, вычисляется)
    
    Returns:
        A_coarse: Coarse-level матрица (CSR)
    """
    device = Af_csr.device
    crow = Af_csr.crow_indices()
    col  = Af_csr.col_indices()
    val  = Af_csr.values()
    n_f  = Af_csr.size(0)
    
    if n_coarse is None:
        n_coarse = int(parent_idx.max().item()) + 1
    
    # Индексы строк для каждого nnz в Af
    row_counts = crow[1:] - crow[:-1]
    i = torch.repeat_interleave(torch.arange(n_f, device=device, dtype=col.dtype), row_counts)
    j = col
    
    # Coarse-индексы: I = parent[i], J = parent[j]
    I = parent_idx[i.long()]
    J = parent_idx[j.long()]
    
    # Веса: w = weights[i] * a_ij * weights[j]
    if weights is None:
        w = val
    else:
        w = (weights[i.long()].to(val.dtype) * val) * weights[j.long()].to(val.dtype)
    
    # Аккумулируем в COO и коалесим
    idx = torch.stack([I, J], dim=0)
    Ac_coo = torch.sparse_coo_tensor(idx, w, (n_coarse, n_coarse),
                                     device=device, dtype=val.dtype).coalesce()
    
    # Санитация NaN/Inf в значениях
    v = Ac_coo.values()
    if not torch.isfinite(v).all():
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        idx_clean = Ac_coo.indices()
        Ac_coo = torch.sparse_coo_tensor(idx_clean, v,
                                         size=(n_coarse, n_coarse),
                                         device=device, dtype=val.dtype)
    
    # Конвертируем в CSR
    Ac = Ac_coo.to_sparse_csr()
    
    # Проверяем корректность CSR (crow должен быть монотонно возрастающим)
    crow = Ac.crow_indices()
    if (crow[1:] < crow[:-1]).any():
        print(f"[WARNING] rap_onepoint_gpu: некорректный CSR после coalesce, используем SciPy")
        import scipy.sparse as sp
        
        # COO -> SciPy
        indices_np = Ac_coo.indices().cpu().numpy()
        values_np = Ac_coo.values().cpu().numpy()
        Ac_sp = sp.coo_matrix((values_np, (indices_np[0], indices_np[1])), 
                              shape=(n_coarse, n_coarse)).tocsr()
        
        # SciPy -> PyTorch CSR
        crow_fixed = torch.from_numpy(Ac_sp.indptr).to(device).to(torch.int64)
        col_fixed = torch.from_numpy(Ac_sp.indices).to(device).to(torch.int64)
        val_fixed = torch.from_numpy(Ac_sp.data).to(device).to(val.dtype)
        
        Ac = torch.sparse_csr_tensor(crow_fixed, col_fixed, val_fixed,
                                     size=(n_coarse, n_coarse),
                                     device=device, dtype=val.dtype)
    
    # Проверка финального размера
    if Ac.size(0) != n_coarse or Ac.size(1) != n_coarse:
        print(f"[ERROR] rap_onepoint_gpu: A_coarse size mismatch! Expected ({n_coarse},{n_coarse}), got ({Ac.size(0)},{Ac.size(1)})")
        raise RuntimeError(f"A_coarse size mismatch: expected {n_coarse}, got {Ac.size(0)}x{Ac.size(1)}")
    
    return Ac


def rap_torch(A_csr: torch.Tensor, P_csr: torch.Tensor, device: torch.device) -> torch.Tensor:
    A_cpu = A_csr.to('cpu')
    P_coo = P_csr.to('cpu').to_sparse_coo().coalesce()

    A_crow = A_cpu.crow_indices()
    A_col = A_cpu.col_indices()
    A_val = A_cpu.values()
    n_fine = A_cpu.size(0)
    n_coarse = P_csr.size(1)

    # Строим списки значений P для каждой строки
    P_rows_cols = [[] for _ in range(n_fine)]
    P_rows_vals = [[] for _ in range(n_fine)]
    rows_idx, cols_idx = P_coo.indices()
    vals = P_coo.values()
    for r, c, w in zip(rows_idx.tolist(), cols_idx.tolist(), vals.tolist()):
        P_rows_cols[r].append(c)
        P_rows_vals[r].append(w)

    from collections import defaultdict
    accum = defaultdict(float)

    for i in range(n_fine):
        row_cols_i = P_rows_cols[i]
        row_vals_i = P_rows_vals[i]
        if not row_cols_i:
            continue
        start = A_crow[i].item()
        end = A_crow[i + 1].item()
        for idx in range(start, end):
            j = A_col[idx].item()
            a_val = A_val[idx].item()
            row_cols_j = P_rows_cols[j]
            row_vals_j = P_rows_vals[j]
            if not row_cols_j:
                continue
            for c1, w1 in zip(row_cols_i, row_vals_i):
                for c2, w2 in zip(row_cols_j, row_vals_j):
                    accum[(c1, c2)] += w1 * a_val * w2

    if not accum:
        return torch.sparse_csr_tensor(
            torch.arange(n_coarse + 1, dtype=torch.int64, device=device),
            torch.zeros(0, dtype=torch.int64, device=device),
            torch.zeros(0, dtype=torch.float64, device=device),
            size=(n_coarse, n_coarse),
        )

    rows_out = []
    cols_out = []
    vals_out = []
    for (c1, c2), v in accum.items():
        if abs(v) > 0.0:
            rows_out.append(c1)
            cols_out.append(c2)
            vals_out.append(v)

    P_coarse = torch.sparse_coo_tensor(
        torch.tensor([rows_out, cols_out], dtype=torch.int64),
        torch.tensor(vals_out, dtype=torch.float64),
        size=(n_coarse, n_coarse),
    ).coalesce()
    return P_coarse.to_sparse_csr().to(device)


def _apply_reference_fix(A_csr: torch.Tensor, anchor_idx: int, anchor_value: float = 0.0) -> torch.Tensor:
    """Фиксирует одну степень свободы (anchor_idx), ставя Dirichlet-условие A[ii]=1."""
    device = A_csr.device
    dtype = A_csr.dtype
    A_coo = A_csr.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    val = A_coo.values().clone()

    mask_row = (row == anchor_idx)
    mask_col = (col == anchor_idx)
    keep_mask = ~(mask_row | mask_col)
    keep_mask = keep_mask | (mask_row & mask_col)

    row_new = row[keep_mask]
    col_new = col[keep_mask]
    val_new = val[keep_mask]

    diag_mask = (row_new == anchor_idx) & (col_new == anchor_idx)
    if diag_mask.any():
        val_new[diag_mask] = torch.tensor(1.0, dtype=dtype, device=device)
    else:
        row_new = torch.cat([row_new, torch.tensor([anchor_idx], device=device)])
        col_new = torch.cat([col_new, torch.tensor([anchor_idx], device=device)])
        val_new = torch.cat([val_new, torch.tensor([1.0], dtype=dtype, device=device)])

    A_fixed = torch.sparse_coo_tensor(torch.stack([row_new, col_new]), val_new, A_csr.size(), device=device, dtype=dtype).coalesce().to_sparse_csr()
    return A_fixed


class ClassicalAMG:
    """Classical Ruge-Stuben Algebraic Multigrid.
    
    TRUE RS-AMG с:
    - Algebraic coarsening через MIS по графу сильных связей
    - 1-point interpolation к сильным C-соседям
    - Эффективный RAP через scatter (O(nnz), полностью на GPU!)
    - Damped Jacobi smoother
    """
    
    def __init__(self, A_csr_np: torch.Tensor, theta: float = 0.25,
                 max_levels: int = 10, coarsest_size: int = 100,
                 anchor_idx: int | None = None,
                 nullspace_dim: int = 3,
                 nullspace_blend: float = 0.6,
                 nullspace_reg: float = 1e-6):
        """Инициализация AMG иерархии.

        Args:
            A_csr_np: Numpy CSR матрица (будет сконвертирована в torch CSR на GPU)
            theta: Порог для сильных связей (0.25 = классический RS)
            max_levels: Максимум уровней
            coarsest_size: Минимальный размер coarsest level
            anchor_idx: Индекс ячейки для Dirichlet-фикса (опционально). Если None – матрица используется как есть.
            nullspace_dim: Количество near-nullspace мод (0 отключает energy-minimizing корректировку)
            nullspace_blend: Вес смешивания RS и energy-minimizing весов (0..1)
            nullspace_reg: Tikhonov-регуляризация в energy-minimizing LS
        """
        self.theta = theta
        self.max_levels = max_levels
        self.coarsest_size = coarsest_size
        self.anchor_idx = anchor_idx
        self.nullspace_dim = max(0, nullspace_dim)
        self.nullspace_blend = float(max(0.0, min(1.0, nullspace_blend)))
        self.nullspace_reg = float(max(1e-10, nullspace_reg))
        self.apply_tol = 5e-2
        self.apply_max_cycles = 8
        self.debug_cycles = False
        
        # Конвертируем numpy CSR -> torch CSR на CUDA
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if hasattr(A_csr_np, 'indptr'):  # scipy sparse
            import scipy.sparse as sp
            if not sp.isspmatrix_csr(A_csr_np):
                A_csr_np = A_csr_np.tocsr()
            
            crow = torch.from_numpy(A_csr_np.indptr).to(device, dtype=torch.int64)
            col = torch.from_numpy(A_csr_np.indices).to(device, dtype=torch.int64)
            val = torch.from_numpy(A_csr_np.data).to(device, dtype=torch.float64)
            
            A_csr = torch.sparse_csr_tensor(crow, col, val,
                                           size=A_csr_np.shape,
                                           device=device, dtype=torch.float64)
        else:
            # Уже torch tensor
            A_csr = A_csr_np.to(device)
        
        self.device = device
        self.levels = []
        
        print(f"[ClassicalAMG] Строим algebraic AMG с theta={theta}...")
        
        node_coords_current = torch.arange(A_csr.size(0), dtype=torch.float64, device=self.device)

        # ═══════════════════════════════════════════════════════════════
        # АДАПТИВНОЕ EQUILIBRATION
        # ═══════════════════════════════════════════════════════════════
        # Проверяем нужно ли equilibration:
        # - Если median(diag) >> 1e-6: матрица хорошо масштабирована, equilibration НЕ НУЖЕН
        # - Если median(diag) << 1e-6: матрица плохая, equilibration КРИТИЧЕН
        # ═══════════════════════════════════════════════════════════════
        diag_orig = self._extract_diag(A_csr).abs().clamp_min(1e-30)
        diag_median = diag_orig.median().item()
        
        # Порог для equilibration: если диагональ слишком мала
        EQUILIBRATION_THRESHOLD = 1e-6
        use_equilibration = (diag_median < EQUILIBRATION_THRESHOLD)
        
        print(f"🔍 [ClassicalAMG] diag: min={diag_orig.min():.3e}, median={diag_median:.3e}, max={diag_orig.max():.3e}")
        print(f"🔍 [ClassicalAMG] Equilibration: {'ON' if use_equilibration else 'OFF'} (threshold={EQUILIBRATION_THRESHOLD:.0e})")
        
        if use_equilibration:
            # Применяем equilibration: A → D^(-1/2) A D^(-1/2)
            Dhalf_inv = 1.0 / torch.sqrt(diag_orig)  # D^(-1/2)
            crow = A_csr.crow_indices()
            col = A_csr.col_indices()
            vals = A_csr.values().clone()
            row_len = crow[1:] - crow[:-1]
            row_idx = torch.repeat_interleave(torch.arange(crow.numel()-1, device=device), row_len)
            vals = vals * Dhalf_inv[row_idx] * Dhalf_inv[col]
            A_csr = torch.sparse_csr_tensor(crow, col, vals, size=A_csr.size(),
                                            device=device, dtype=torch.float64)
            self.Dhalf_inv = Dhalf_inv
        else:
            # Equilibration не нужен - матрица уже хорошо масштабирована
            self.Dhalf_inv = None

        if self.anchor_idx is not None:
            A_csr = _apply_reference_fix(A_csr, int(self.anchor_idx))
        
        diag_scaled = self._extract_diag(A_csr).abs()
        print(f"[ClassicalAMG] Equilibration: diag {diag_orig.min():.2e}..{diag_orig.max():.2e} → {diag_scaled.min():.2e}..{diag_scaled.max():.2e}")
        
        # Строим иерархию
        A_current = A_csr
        basis_current = generate_nullspace(A_current, self.nullspace_dim)
        for lvl in range(max_levels):
            n = A_current.size(0)
            
            if n <= coarsest_size:
                # Coarsest level: прямое решение
                diag_abs = self._extract_diag(A_current).abs()
                row_abs = self._row_abs_sum(A_current)
                beta = 0.3  # L1-Jacobi параметр
                denom = torch.maximum(diag_abs, beta * row_abs).clamp_min(1e-30)
                inv_relax = (1.0 / denom).clamp(max=1e2)  # Мягкая страховка
                
                self.levels.append({
                    'A': A_current,
                    'n': n,
                    'P': None,
                    'diag': diag_abs,
                    'inv_relax': inv_relax,
                    'basis': basis_current,
                    'is_coarsest': True
                })
                print(f"[ClassicalAMG] L{lvl}: n={n} ≤ {coarsest_size}, coarsest level")
                break
            
            # C/F splitting
            strong_mask = find_strong_connections(A_current, theta)
            cf_marker, n_coarse = classical_coarsening(A_current, theta)
            if self.anchor_idx is not None and self.anchor_idx < n:
                cf_marker[self.anchor_idx] = 0
            
            if n_coarse >= n * 0.9:
                # Слишком мало coarsening
                diag_abs = self._extract_diag(A_current).abs()
                row_abs = self._row_abs_sum(A_current)
                beta = 0.3
                denom = torch.maximum(diag_abs, beta * row_abs).clamp_min(1e-30)
                inv_relax = (1.0 / denom).clamp(max=1e2)
                
                self.levels.append({
                    'A': A_current,
                    'n': n,
                    'P': None,
                    'fine2coarse': None,
                    'coarse_nodes': None,
                    'F_nodes': None,
                    'node_coords': node_coords_current,
                    'basis': basis_current,
                    'interp_metrics': None,
                    'diag': diag_abs,
                    'inv_relax': inv_relax,
                    'is_coarsest': True
                })
                print(f"[ClassicalAMG] L{lvl}: n={n}, coarsening failed (ratio={n/n_coarse:.1f}x), stopping")
                break
            
            # Prolongation: RS-интерполяция с F–F коррекцией (P·1=1)
            blend = self.nullspace_blend if basis_current is not None else 0.0
            P, cf_marker, fine2coarse_map, coarse_nodes, F_nodes = build_prolongation_rs_full(
                A_current, cf_marker, strong_mask, basis_current, blend=blend, ls_reg=self.nullspace_reg
            )
            if P.device != self.device:
                P = P.to(self.device)
            fine2coarse_map = fine2coarse_map.to(self.device)
            coarse_nodes = coarse_nodes.to(self.device, dtype=torch.int64)
            F_nodes = F_nodes.to(self.device, dtype=torch.int64)

            interp_metrics = self._analyze_prolongation(lvl, P, coarse_nodes, F_nodes, node_coords_current)

            A_coarse = rap_torch(A_current, P, self.device)

            n_coarse_actual = P.size(1)
            num_coarse_marked = int((cf_marker == 0).sum().item())
            ratio = n / max(n_coarse_actual, 1)
            c_pct = 100.0 * n_coarse_actual / n
            orphan_count = n_coarse_actual - num_coarse_marked

            print(f"[ClassicalAMG] L{lvl}: n={n} → n_c={n_coarse_actual} (ratio={ratio:.1f}x), C-points={num_coarse_marked}+{orphan_count} orphans/{n} ({c_pct:.1f}%)")
            
            diag_abs = self._extract_diag(A_current).abs()
            row_abs = self._row_abs_sum(A_current)
            beta = 0.3
            denom = torch.maximum(diag_abs, beta * row_abs).clamp_min(1e-30)
            inv_relax = (1.0 / denom).clamp(max=1e2)
            
            self.levels.append({
                'A': A_current,
                'n': n,
                'P': P,
                'fine2coarse': fine2coarse_map,
                'coarse_nodes': coarse_nodes,
                'F_nodes': F_nodes,
                'node_coords': node_coords_current,
                'basis': basis_current,
                'interp_metrics': interp_metrics,
                'diag': diag_abs,
                'inv_relax': inv_relax,
                'is_coarsest': False
            })
            
            node_coords_current = node_coords_current[coarse_nodes]
            if basis_current is not None:
                basis_gpu = basis_current.to(P.device)
                basis_coarse = torch.sparse.mm(P.transpose(0, 1), basis_gpu)
                basis_coarse = basis_coarse.to('cpu')
                basis_coarse = orthonormalize_columns(basis_coarse)
                basis_current = basis_coarse
            else:
                basis_current = None
            A_current = A_coarse
        
        # Если цикл завершился по max_levels, добавляем A_current как coarsest
        if len(self.levels) == 0 or not self.levels[-1]['is_coarsest']:
            n_c = A_current.size(0)
            diag_abs = self._extract_diag(A_current).abs()
            row_abs = self._row_abs_sum(A_current)
            beta = 0.3
            denom = torch.maximum(diag_abs, beta * row_abs).clamp_min(1e-30)
            inv_relax = (1.0 / denom).clamp(max=1e2)
            
            self.levels.append({
                'A': A_current,
                'n': n_c,
                'P': None,
                'diag': diag_abs,
                'inv_relax': inv_relax,
                'basis': basis_current,
                'is_coarsest': True
            })
            print(f"[ClassicalAMG] L{len(self.levels)-1}: reached max_levels → coarsest, n={n_c}")
        
        print(f"✅ ClassicalAMG: построено {len(self.levels)} уровней")
    
    def _extract_diag(self, A_csr: torch.Tensor) -> torch.Tensor:
        """Извлекает диагональ из CSR матрицы."""
        crow = A_csr.crow_indices()
        col = A_csr.col_indices()
        val = A_csr.values()
        n = crow.numel() - 1
        
        diag = torch.zeros(n, device=A_csr.device, dtype=A_csr.dtype)
        
        row_len = crow[1:] - crow[:-1]
        row_idx = torch.repeat_interleave(torch.arange(n, device=A_csr.device), row_len)
        diag_mask = row_idx == col
        
        diag.scatter_(0, row_idx[diag_mask], val[diag_mask])
        
        return diag
    
    def _row_abs_sum(self, A_csr: torch.Tensor) -> torch.Tensor:
        """Сумма |A_ij| по строкам для L1-Jacobi.
        
        Используется для вычисления робастного знаменателя:
        denom = max(|a_ii|, β·∑|a_ij|)
        """
        crow = A_csr.crow_indices()
        col = A_csr.col_indices()
        val = A_csr.values().abs()
        n = crow.numel() - 1
        
        row_len = crow[1:] - crow[:-1]
        row_idx = torch.repeat_interleave(torch.arange(n, device=A_csr.device), row_len)
        
        row_abs = torch.zeros(n, device=A_csr.device, dtype=val.dtype)
        row_abs.scatter_add_(0, row_idx, val)
        
        return row_abs
    
    def _spmv_csr(self, A_csr: torch.Tensor, x: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        """Sparse matrix-vector product через CSR индексы.
        
        Args:
            A_csr: Sparse CSR matrix
            x: Dense vector
            transpose: If True, compute A^T * x
        
        Returns:
            y = A*x или y = A^T*x
        """
        crow = A_csr.crow_indices()
        col = A_csr.col_indices()
        val = A_csr.values()
        n_rows = crow.numel() - 1
        
        if not transpose:
            # y = A * x: стандартный SpMV
            row_len = crow[1:] - crow[:-1]
            row_idx = torch.repeat_interleave(torch.arange(n_rows, device=A_csr.device, dtype=torch.int64), row_len)
            prod = val * x[col]
            y = torch.zeros(n_rows, device=A_csr.device, dtype=A_csr.dtype)
            y.scatter_add_(0, row_idx, prod)
        else:
            # y = A^T * x: transpose SpMV
            # A^T[j,i] = A[i,j], поэтому y[j] += A[i,j] * x[i]
            row_len = crow[1:] - crow[:-1]
            row_idx = torch.repeat_interleave(torch.arange(n_rows, device=A_csr.device, dtype=torch.int64), row_len)
            prod = val * x[row_idx]
            n_cols = A_csr.size(1)
            y = torch.zeros(n_cols, device=A_csr.device, dtype=A_csr.dtype)
            y.scatter_add_(0, col, prod)
        
        return y
    
    def _matvec(self, lvl: int, x: torch.Tensor) -> torch.Tensor:
        """Matrix-vector product: y = A_lvl × x"""
        A = self.levels[lvl]['A']
        return self._spmv_csr(A, x, transpose=False)
    
    def _smooth(self, lvl: int, x: torch.Tensor, b: torch.Tensor, nu: int = 1, debug: bool = False) -> torch.Tensor:
        """L1-Jacobi smoother: denom = max(|a_ii|, β·∑|a_ij|).
        
        Использует предвычисленный inv_relax из уровня, что обеспечивает
        локальную адаптивность и устойчивость без глобальных clamp.
        """
        omega = 0.7
        inv_relax = self.levels[lvl]['inv_relax']
        
        if debug:
            print(f"    [SMOOTH L{lvl}] inv_relax: min={inv_relax.min():.3e}, med={inv_relax.median():.3e}, max={inv_relax.max():.3e}")
        
        for it in range(nu):
            r = b - self._matvec(lvl, x)
            delta = omega * inv_relax * r
            x = x + delta
            if debug:
                print(f"    [SMOOTH L{lvl} iter{it+1}] ||r||={r.norm():.3e}, ||δ||={delta.norm():.3e}, ||x||={x.norm():.3e}, max|δ|={delta.abs().max():.3e}")
        
        return x
    
    def _v_cycle(self, lvl: int, x: torch.Tensor, b: torch.Tensor,
                 pre_smooth: int = 3, post_smooth: int = 2, debug: bool = False, cycle_num: int = 0) -> torch.Tensor:
        """V-cycle"""
        level = self.levels[lvl]
        
        if debug:
            print(f"  [V-CYCLE L{lvl}] ВХОД: ||x||={x.norm():.3e}, ||b||={b.norm():.3e}, n={level['n']}")
        
        if level['is_coarsest']:
            # Прямое решение на coarsest уровне (фиксируем первую степень свободы)
            n = level['n']
            A_dense = level['A'].to_dense().clone()
            b_local = b.clone()
            x = torch.zeros_like(b_local)
            if n > 0:
                A_dense[0, :] = 0.0
                A_dense[:, 0] = 0.0
                A_dense[0, 0] = 1.0
                b_local[0] = 0.0

            if n <= 500:
                try:
                    x = torch.linalg.solve(A_dense, b_local)
                except RuntimeError:
                    x = torch.linalg.lstsq(A_dense, b_local.unsqueeze(1)).solution.squeeze(1)
            else:
                inv_relax = level['inv_relax']
                omega = 0.7
                for it in range(50):
                    r = b_local - self._spmv_csr(level['A'], x)
                    delta = omega * inv_relax * r
                    x = x + delta
            return x
        
        # Pre-smoothing
        x = self._smooth(lvl, x, b, pre_smooth, debug=debug)
        if debug:
            print(f"  [V-CYCLE L{lvl}] ПОСЛЕ pre-smooth: ||x||={x.norm():.3e}")
        
        # Residual
        r = b - self._matvec(lvl, x)
        if debug:
            print(f"  [V-CYCLE L{lvl}] residual: ||r||={r.norm():.3e}")
        
        # Restrict: r_c = P^T * r
        P = level['P']
        r_c = self._spmv_csr(P, r, transpose=True)
        if debug:
            print(f"  [V-CYCLE L{lvl}] RESTRICT: ||r||={r.norm():.3e} → ||r_c||={r_c.norm():.3e}")
        
        # Coarse solve
        e_c = torch.zeros_like(r_c)
        e_c = self._v_cycle(lvl + 1, e_c, r_c, pre_smooth, post_smooth, debug=debug, cycle_num=cycle_num)
        if debug:
            print(f"  [V-CYCLE L{lvl}] COARSE возвращает: ||e_c||={e_c.norm():.3e}, max|e_c|={e_c.abs().max():.3e}")
        
        # Prolongate: e_f = P * e_c
        e_f = self._spmv_csr(P, e_c, transpose=False)
        if debug:
            print(f"  [V-CYCLE L{lvl}] PROLONGATE: ||e_c||={e_c.norm():.3e} → ||e_f||={e_f.norm():.3e}, max|e_f|={e_f.abs().max():.3e}")
        x = x + e_f
        if debug:
            print(f"  [V-CYCLE L{lvl}] ПОСЛЕ коррекции: ||x||={x.norm():.3e}")
        
        # Post-smoothing
        x = self._smooth(lvl, x, b, post_smooth, debug=debug)
        if debug:
            print(f"  [V-CYCLE L{lvl}] ВЫХОД: ||x||={x.norm():.3e}")
        
        return x
    
    def solve(self, b: torch.Tensor, x0: Optional[torch.Tensor] = None,
              tol: float = 1e-6, max_iter: int = 10) -> torch.Tensor:
        """Решает A·x = b через V-cycles с equilibration.
        
        Внутренне решается масштабированная система:
        D^(-1/2) A D^(-1/2) · (D^(1/2) x) = D^(-1/2) b
        
        Args:
            b: RHS вектор (физический)
            x0: Начальное приближение (физическое, если None → 0)
            tol: Относительная tolerance для ||r||/||b||
            max_iter: Максимум V-cycles
        
        Returns:
            x: Решение (физическое)
        """
        b = b.to(self.device)
        
        # Проверяем нужно ли equilibration
        if self.Dhalf_inv is not None:
            # С equilibration
            b_scaled = self.Dhalf_inv * b
            x_scaled = torch.zeros_like(b_scaled) if x0 is None else x0.to(self.device) / self.Dhalf_inv
        else:
            # Без equilibration - работаем напрямую
            b_scaled = b.clone()
            x_scaled = torch.zeros_like(b) if x0 is None else x0.to(self.device)
        
        b_norm = b_scaled.norm()
        
        for cycle in range(max_iter):
            x_scaled = self._v_cycle(0, x_scaled, b_scaled, pre_smooth=3, post_smooth=2,
                                     debug=self.debug_cycles, cycle_num=cycle)
            r = b_scaled - self._matvec(0, x_scaled)
            rel_res = r.norm() / (b_norm + 1e-30)
            x_ratio = x_scaled.norm() / (b_norm + 1e-30)
            print(f"  [ClassicalAMG cycle {cycle+1}] rel_res={rel_res:.3e}, ||x||/||b||={x_ratio:.2e}")
            if rel_res < tol:
                break
        
        # Обратное масштабирование (если было)
        # Мы решали: (D^{-1/2} A D^{-1/2}) · y = D^{-1/2} b, где y = D^{1/2} x
        # Следовательно, x = D^{-1/2} · y = Dhalf_inv * x_scaled
        if self.Dhalf_inv is not None:
            x = self.Dhalf_inv * x_scaled
        else:
            x = x_scaled
        return x

    def apply(self, b: torch.Tensor, cycles: int = 1, pre_smooth: int = 3, post_smooth: int = 2) -> torch.Tensor:
        """Неполный solve: N V-циклов"""
        b = b.to(self.device)
        if self.Dhalf_inv is not None:
            b_scaled = self.Dhalf_inv * b
        else:
            b_scaled = b.clone()
        x_scaled = torch.zeros_like(b_scaled)

        tol = getattr(self, "apply_tol", 5e-2)
        max_cycles = getattr(self, "apply_max_cycles", max(3, cycles))
        cycle = 0
        rel_res = float('inf')
        while cycle < max_cycles:
            x_scaled = self._v_cycle(0, x_scaled, b_scaled, pre_smooth=3, post_smooth=2, debug=False, cycle_num=cycle)
            cycle += 1
            r = b_scaled - self._matvec(0, x_scaled)
            rel_res = r.norm().item() / (b_scaled.norm().item() + 1e-30)
            if cycle >= cycles and rel_res < tol:
                break

        if self.Dhalf_inv is not None:
            return self.Dhalf_inv * x_scaled
        return x_scaled

    def _analyze_prolongation(self, lvl: int, P: torch.Tensor, coarse_nodes: torch.Tensor, F_nodes: torch.Tensor, node_coords_fine: torch.Tensor) -> dict:
        """Вычисляет диагностические метрики для оператора продолбжения."""
        P_coo = P.to_sparse_coo().coalesce()
        row_idx = P_coo.indices()[0]
        vals = P_coo.values()
        device = P.device
        dtype = torch.float64

        row_sum = torch.zeros(P.size(0), dtype=dtype, device=device)
        row_sum.scatter_add_(0, row_idx, vals)
        ones_f = torch.ones_like(row_sum)
        const_err = row_sum - ones_f
        row_err_abs = const_err.abs()
        row_err_max = row_err_abs.max().item()
        const_rel = const_err.norm() / (ones_f.norm() + 1e-30)

        coords_fine = node_coords_fine.to(device, dtype=dtype)
        coords_coarse = coords_fine[coarse_nodes.long()]
        grad_recon = torch.sparse.mm(P, coords_coarse.unsqueeze(1)).squeeze(1)
        grad_err = grad_recon - coords_fine
        grad_err_abs = grad_err.abs()
        grad_rel = grad_err.norm() / (coords_fine.norm() + 1e-30)

        neg_vals = vals[vals < -1e-12]
        pos_vals = vals[vals > 1e-12]
        min_w = vals.min().item() if vals.numel() > 0 else 0.0
        max_w = vals.max().item() if vals.numel() > 0 else 0.0

        worst_const_rows = []
        worst_grad_rows = []
        if F_nodes.numel() > 0:
            topk = min(5, F_nodes.numel())
            f_const_vals, f_const_idx = torch.topk(row_err_abs[F_nodes], k=topk)
            f_grad_vals, f_grad_idx = torch.topk(grad_err_abs[F_nodes], k=topk)
            worst_const_rows = [(int(F_nodes[f_const_idx[j]].item()), f_const_vals[j].item()) for j in range(topk)]
            worst_grad_rows = [(int(F_nodes[f_grad_idx[j]].item()), f_grad_vals[j].item()) for j in range(topk)]
            print(f"[ClassicalAMG] L{lvl} worst const rows: {worst_const_rows}")
            print(f"[ClassicalAMG] L{lvl} worst grad  rows: {worst_grad_rows}")

        print(
            f"[ClassicalAMG] L{lvl} prolong diag: row_err_max={row_err_max:.2e}, "
            f"const_rel={const_rel:.2e}, grad_rel={grad_rel:.2e}, "
            f"neg_w={neg_vals.numel()}, pos_w={pos_vals.numel()}, min_w={min_w:.2e}, max_w={max_w:.2e}"
        )

        return {
            'row_err_max': row_err_max,
            'const_rel': const_rel.item(),
            'grad_rel': grad_rel.item(),
            'row_sum_min': row_sum.min().item(),
            'row_sum_max': row_sum.max().item(),
            'neg_count': int(neg_vals.numel()),
            'pos_count': int(pos_vals.numel()),
            'min_w': min_w,
            'max_w': max_w,
            'worst_const_rows': worst_const_rows,
            'worst_grad_rows': worst_grad_rows,
        }

    def _orthonormalize_columns(self, B: torch.Tensor) -> torch.Tensor:
        cols = []
        for j in range(B.shape[1]):
            v = B[:, j].clone()
            for q in cols:
                v = v - torch.dot(v, q) * q
            norm = v.norm()
            if norm > 1e-10:
                cols.append(v / norm)
        if not cols:
            v = torch.ones(B.shape[0], dtype=B.dtype, device=B.device)
            cols.append(v / (v.norm() + 1e-30))
        return torch.stack(cols, dim=1)

    def _generate_nullspace(self, A_csr: torch.Tensor, dim: int = 3) -> torch.Tensor | None:
        if dim <= 0:
            return None
        n = A_csr.size(0)
        device = A_csr.device
        basis = []
        const = torch.ones(n, dtype=torch.float64, device=device)
        basis.append(const / (const.norm() + 1e-30))
        rng = torch.Generator(device=device)
        diag = A_csr.diag().to(torch.float64)
        inv_diag = torch.where(diag.abs() > 1e-12, 1.0 / diag, torch.zeros_like(diag))
        for _ in range(max(0, dim - 1)):
            v = torch.randn(n, generator=rng, dtype=torch.float64, device=device)
            x = torch.zeros_like(v)
            for _ in range(4):
                r = v - torch.sparse.mm(A_csr, x.unsqueeze(1)).squeeze(1)
                x = x + inv_diag * r
            v = x
            for b in basis:
                v = v - torch.dot(v, b) * b
            norm = v.norm()
            if norm > 1e-8:
                basis.append(v / norm)
        B = torch.stack(basis, dim=1)
        B = B.to('cpu')
        return orthonormalize_columns(B)

    def _smooth_vector(self, A_csr: torch.Tensor, v: torch.Tensor, sweeps: int = 4) -> torch.Tensor:
        diag = self._extract_diag(A_csr).to(v.dtype)
        inv_diag = torch.where(diag.abs() > 1e-12, 1.0 / diag, torch.zeros_like(diag))
        x = torch.zeros_like(v)
        for _ in range(sweeps):
            r = v - self._spmv_csr(A_csr, x)
            x = x + inv_diag * r
        return x

    def _solve_least_squares(self, M: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        try:
            sol = torch.linalg.lstsq(M, target).solution
        except RuntimeError:
            sol = torch.linalg.pinv(M) @ target
        return sol


def orthonormalize_columns(B: torch.Tensor) -> torch.Tensor:
    cols: list[torch.Tensor] = []
    for j in range(B.shape[1]):
        v = B[:, j].clone()
        for q in cols:
            v = v - torch.dot(v, q) * q
        norm = v.norm()
        if norm > 1e-10:
            cols.append(v / norm)
    if not cols:
        v = torch.ones(B.shape[0], dtype=B.dtype)
        cols.append(v / (v.norm() + 1e-30))
    return torch.stack(cols, dim=1)


def _extract_diag_cpu(A_csr: torch.Tensor) -> torch.Tensor:
    crow = A_csr.crow_indices()
    col = A_csr.col_indices()
    val = A_csr.values()
    n = crow.numel() - 1
    diag = torch.zeros(n, dtype=val.dtype)
    for i in range(n):
        start = crow[i].item()
        end = crow[i + 1].item()
        for idx in range(start, end):
            if col[idx].item() == i:
                diag[i] = val[idx]
                break
    return diag


def generate_nullspace(A_csr: torch.Tensor, dim: int = 3) -> torch.Tensor | None:
    if dim <= 0:
        return None
    A_cpu = A_csr.to('cpu')
    n = A_cpu.size(0)
    basis: list[torch.Tensor] = []
    const = torch.ones(n, dtype=torch.float64)
    basis.append(const / (const.norm() + 1e-30))

    coords = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    centered = coords - coords.mean()
    if centered.norm() > 1e-8 and len(basis) < dim:
        basis.append(centered / centered.norm())
    poly = coords ** 2 - (coords ** 2).mean()
    if poly.norm() > 1e-8 and len(basis) < dim:
        poly = poly - sum(torch.dot(poly, b) * b for b in basis)
        if poly.norm() > 1e-8:
            basis.append(poly / poly.norm())

    diag = _extract_diag_cpu(A_cpu).to(torch.float64)
    inv_diag = torch.where(diag.abs() > 1e-12, 1.0 / diag, torch.zeros_like(diag))
    rng = torch.Generator(device='cpu')
    rng.manual_seed(0)
    while len(basis) < dim:
        v = torch.randn(n, generator=rng, dtype=torch.float64)
        x = torch.zeros_like(v)
        for _ in range(6):
            r = torch.sparse.mm(A_cpu, x.unsqueeze(1)).squeeze(1)
            r = v - r
            x = x + inv_diag * r
        v = x
        for b in basis:
            v = v - torch.dot(v, b) * b
        norm = v.norm()
        if norm > 1e-8:
            basis.append(v / norm)
        else:
            break

    B = torch.stack(basis, dim=1)
    return orthonormalize_columns(B)


def solve_energy_weights(
    B_c: torch.Tensor,
    target: torch.Tensor,
    w_rs: torch.Tensor,
    blend: float,
    ls_reg: float,
) -> torch.Tensor:
    if B_c is None or B_c.numel() == 0 or blend <= 0.0:
        return w_rs
    m, k = B_c.shape
    if k == 0:
        return w_rs
    ones_row = torch.ones(1, k, dtype=torch.float64)
    A_ls = torch.vstack([B_c, ones_row])
    y = torch.cat([target, torch.tensor([1.0], dtype=torch.float64)])
    At = A_ls.T
    ATA = At @ A_ls + ls_reg * torch.eye(k, dtype=torch.float64)
    RHS = At @ y
    try:
        w_ls = torch.linalg.solve(ATA, RHS)
    except RuntimeError:
        try:
            w_ls = torch.linalg.lstsq(A_ls, y).solution
        except RuntimeError:
            return w_rs
    w_ls = torch.clamp(w_ls, min=0.0)
    sum_ls = w_ls.sum().item()
    if sum_ls <= 1e-12:
        return w_rs
    w_ls = w_ls / sum_ls
    return (1.0 - blend) * w_rs + blend * w_ls
