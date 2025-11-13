"""
Classical Algebraic Multigrid (Ruge-Stuben) solver.

TRUE RS-AMG с:
- Strong connections по отрицательным внедиагоналям с диагональным масштабированием
- MIS coarsening по графу сильных связей
- 1-point interpolation к сильным C-соседям
- Эффективный RAP через scatter (без SpGEMM/плотнения!)
"""

import math
import torch
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional
import time


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


def pmis_coarsening(A_csr: torch.Tensor, strong_mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
    crow = A_csr.crow_indices()
    col = A_csr.col_indices()
    n = crow.numel() - 1
    device = A_csr.device

    row_len = crow[1:] - crow[:-1]
    row_idx = torch.repeat_interleave(torch.arange(n, device=device), row_len)

    edges_i = row_idx[strong_mask]
    edges_j = col[strong_mask]

    if edges_i.numel() == 0:
        cf_marker = torch.zeros(n, dtype=torch.int8, device=device)
        return cf_marker, n

    edges_i_sym = torch.cat([edges_i, edges_j])
    edges_j_sym = torch.cat([edges_j, edges_i])

    deg = torch.zeros(n, dtype=torch.float32, device=device)
    deg.scatter_add_(0, edges_i_sym, torch.ones_like(edges_i_sym, dtype=torch.float32))

    states = torch.zeros(n, dtype=torch.int8, device=device)
    eps = 1e-9
    iteration = 0
    max_iter = 10
    while (states == 0).any() and iteration < max_iter:
        undecided = states == 0
        if not undecided.any():
            break

        priorities = torch.rand(n, device=device)
        priorities = priorities + deg * 1e-3 + iteration * 1e-6
        priorities[~undecided] = -1e9

        mask_edges = undecided[edges_i_sym] & undecided[edges_j_sym]
        neigh_max = torch.full((n,), -1e9, device=device)
        if mask_edges.any():
            neigh_max.scatter_reduce_(0, edges_i_sym[mask_edges], priorities[edges_j_sym[mask_edges]],
                                      reduce='amax', include_self=False)

        new_C = undecided & (priorities >= neigh_max - eps)
        if not new_C.any():
            idx = torch.nonzero(undecided, as_tuple=False)[0, 0]
            new_C[idx] = True

        states[new_C] = 1

        if new_C.any():
            mask_edges_c = new_C[edges_j_sym]
            if mask_edges_c.any():
                targets = edges_i_sym[mask_edges_c]
                current = states[targets]
                states[targets] = torch.where(current == 0, torch.full_like(current, -1), current)

        iteration += 1

    cf_marker = torch.ones(n, dtype=torch.int8, device=device)
    cf_marker[states == 1] = 0
    n_coarse = int((cf_marker == 0).sum().item())
    return cf_marker, n_coarse


def build_prolongation_rs_full(
    A_csr: torch.Tensor,
    cf_marker: torch.Tensor,
    strong_mask: torch.Tensor,
    basis: torch.Tensor | None,
    blend: float = 0.0,
    ls_reg: float = 1e-7,
    node_coords: torch.Tensor | None = None,
    drop_tol: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = A_csr.device
    dtype = A_csr.dtype
    coords = None
    coord_dim = 0
    if node_coords is not None:
        coords = node_coords
        if coords.dim() == 1:
            coords = coords.unsqueeze(1)
        if coords.size(0) != A_csr.size(0):
            raise ValueError("node_coords размер не соответствует матрице")
        coords = coords.to(device=device, dtype=dtype)
        coord_dim = coords.size(1)

    crow = A_csr.crow_indices()
    col = A_csr.col_indices()
    val = A_csr.values()
    n = crow.numel() - 1

    cf = cf_marker.to(device=device, dtype=torch.int8).clone()
    strong = strong_mask.to(device=device)

    row_counts = crow[1:] - crow[:-1]
    row_idx = torch.repeat_interleave(torch.arange(n, device=device, dtype=torch.int64), row_counts)

    diag = torch.zeros(n, device=device, dtype=dtype)
    diag_mask = (row_idx == col)
    if diag_mask.any():
        diag.scatter_add_(0, row_idx[diag_mask], val[diag_mask])
    diag = torch.where(diag.abs() > 1e-14, diag, torch.ones_like(diag))

    # Promote orphan F-points that have no strong C-neighbours
    promoted = True
    off_mask_global = (col != row_idx)
    while promoted:
        coarse_mask_iter = (cf == 0)
        strong_C_iter = strong & off_mask_global & coarse_mask_iter[col]
        strong_C_counts = torch.zeros(n, device=device, dtype=torch.int64)
        if strong_C_iter.any():
            strong_C_counts.scatter_add_(0, row_idx[strong_C_iter], torch.ones_like(row_idx[strong_C_iter]))
        orphan_mask = (cf == 1) & (strong_C_counts == 0)
        promoted = bool(orphan_mask.any().item())
        if promoted:
            cf[orphan_mask] = 0

    coarse_mask = (cf == 0)
    F_mask = ~coarse_mask

    coarse_nodes = torch.nonzero(coarse_mask, as_tuple=False).view(-1)
    fine2coarse = torch.full((n,), -1, device=device, dtype=torch.int64)
    if coarse_nodes.numel() > 0:
        fine2coarse[coarse_nodes] = torch.arange(coarse_nodes.numel(), device=device, dtype=torch.int64)

    strong_C_mask = strong & off_mask_global & coarse_mask[col] & F_mask[row_idx]
    weak_C_mask = (~strong) & off_mask_global & coarse_mask[col] & F_mask[row_idx]
    strong_F_mask = strong & off_mask_global & F_mask[col] & F_mask[row_idx]
    weak_F_mask = (~strong) & off_mask_global & F_mask[col] & F_mask[row_idx]

    sum_strong = torch.zeros(n, device=device, dtype=dtype)
    if strong_C_mask.any():
        sum_strong.scatter_add_(0, row_idx[strong_C_mask], (-val[strong_C_mask]).to(dtype))

    weak_C_sum = torch.zeros(n, device=device, dtype=dtype)
    if weak_C_mask.any():
        weak_C_sum.scatter_add_(0, row_idx[weak_C_mask], (-val[weak_C_mask]).to(dtype))

    F_weak_sum = torch.zeros(n, device=device, dtype=dtype)
    if weak_F_mask.any():
        F_weak_sum.scatter_add_(0, row_idx[weak_F_mask], (-val[weak_F_mask]).to(dtype))

    strong_C_counts_final = torch.zeros(n, device=device, dtype=torch.int64)
    if strong_C_mask.any():
        strong_C_counts_final.scatter_add_(0, row_idx[strong_C_mask], torch.ones_like(row_idx[strong_C_mask]))

    # CSR-like structure for strong C edges
    strong_C_row_ptr = torch.zeros(n + 1, device=device, dtype=torch.int64)
    if strong_C_counts_final.numel() > 0:
        strong_C_row_ptr[1:] = torch.cumsum(strong_C_counts_final, dim=0)

    jc_indices = torch.nonzero(strong_C_mask, as_tuple=False).view(-1)
    jc_coarse = fine2coarse[col[jc_indices]]
    denom_rows = sum_strong[row_idx[jc_indices]]
    jc_weights = torch.zeros_like(denom_rows, dtype=dtype)
    denom_positive = denom_rows > 1e-14
    if denom_positive.any():
        jc_weights[denom_positive] = (-val[jc_indices][denom_positive]) / denom_rows[denom_positive]

    # Direct RS weights
    rows_direct = row_idx[strong_C_mask]
    cols_direct = jc_coarse
    vals_direct = torch.zeros_like(rows_direct, dtype=dtype)
    if rows_direct.numel() > 0:
        vals_direct = (-val[strong_C_mask]) / diag[rows_direct]
        vals_direct = torch.nan_to_num(vals_direct, nan=0.0, posinf=0.0, neginf=0.0)

    # F-F contributions
    ff_rows = row_idx[strong_F_mask]
    ff_cols = col[strong_F_mask]
    ff_vals = val[strong_F_mask]
    ff_factor = torch.zeros_like(ff_vals, dtype=dtype)
    if ff_rows.numel() > 0:
        ff_factor = (-ff_vals) / diag[ff_rows]
        ff_factor = torch.nan_to_num(ff_factor, nan=0.0, posinf=0.0, neginf=0.0)

    counts_j = strong_C_counts_final[ff_cols] if ff_cols.numel() > 0 else torch.empty(0, device=device, dtype=torch.int64)
    denom_j = sum_strong[ff_cols] if ff_cols.numel() > 0 else torch.empty(0, device=device, dtype=dtype)
    valid_ff = (
        (ff_rows.numel() > 0)
        and (counts_j.numel() > 0)
    )
    rows_contrib = torch.empty(0, device=device, dtype=torch.int64)
    cols_contrib = torch.empty(0, device=device, dtype=torch.int64)
    vals_contrib = torch.empty(0, device=device, dtype=dtype)

    if valid_ff:
        valid_mask = (counts_j > 0) & (denom_j > 1e-14)
        if valid_mask.any():
            ff_rows_valid = ff_rows[valid_mask]
            ff_cols_valid = ff_cols[valid_mask]
            ff_factor_valid = ff_factor[valid_mask]
            counts_valid = counts_j[valid_mask]
            starts_valid = strong_C_row_ptr[ff_cols_valid]
            total = int(counts_valid.sum().item())
            if total > 0:
                cum_counts = torch.cumsum(counts_valid, dim=0)
                base = cum_counts - counts_valid
                start_rep = torch.repeat_interleave(starts_valid, counts_valid)
                base_rep = torch.repeat_interleave(base, counts_valid)
                intra = torch.arange(total, device=device, dtype=torch.int64) - base_rep
                jc_pos = start_rep + intra
                rows_contrib = torch.repeat_interleave(ff_rows_valid, counts_valid)
                cols_contrib = jc_coarse[jc_pos]
                weights_contrib = jc_weights[jc_pos]
                factor_rep = torch.repeat_interleave(ff_factor_valid, counts_valid)
                vals_contrib = factor_rep * weights_contrib

        invalid_mask = ~valid_mask if counts_j.numel() > 0 else torch.empty(0, device=device, dtype=torch.bool)
        if invalid_mask.numel() > 0 and invalid_mask.any():
            F_weak_sum.scatter_add_(0, ff_rows[invalid_mask], (-ff_vals[invalid_mask]).to(dtype))

    # Weak connection correction
    total_weak = weak_C_sum + F_weak_sum
    corr = torch.zeros(n, device=device, dtype=dtype)
    mask_corr = (sum_strong > 1e-14) & F_mask
    if mask_corr.any():
        corr[mask_corr] = total_weak[mask_corr] / sum_strong[mask_corr]

    if vals_direct.numel() > 0:
        vals_direct = vals_direct * (1.0 + corr[rows_direct])

    vals_direct = torch.clamp(vals_direct, min=0.0)
    vals_contrib = torch.clamp(vals_contrib, min=0.0)

    rows_F = rows_direct
    cols_F = cols_direct
    vals_F = vals_direct
    if vals_contrib.numel() > 0:
        rows_F = torch.cat([rows_F, rows_contrib])
        cols_F = torch.cat([cols_F, cols_contrib])
        vals_F = torch.cat([vals_F, vals_contrib])

    coarse_rows = coarse_nodes
    coarse_cols = fine2coarse[coarse_nodes] if coarse_nodes.numel() > 0 else torch.empty(0, device=device, dtype=torch.int64)
    coarse_vals = torch.ones(coarse_nodes.numel(), device=device, dtype=dtype)

    rows_total = torch.cat([coarse_rows, rows_F]) if rows_F.numel() > 0 else coarse_rows
    cols_total = torch.cat([coarse_cols, cols_F]) if cols_F.numel() > 0 else coarse_cols
    vals_total = torch.cat([coarse_vals, vals_F]) if vals_F.numel() > 0 else coarse_vals

    if rows_total.numel() == 0:
        P_csr = torch.sparse_csr_tensor(
            torch.arange(n + 1, device=device, dtype=torch.int64),
            torch.zeros(0, device=device, dtype=torch.int64),
            torch.zeros(0, device=device, dtype=dtype),
            size=(n, 1),
            device=device,
            dtype=dtype,
        )
    else:
        P_coo = torch.sparse_coo_tensor(
            torch.stack([rows_total, cols_total], dim=0),
            vals_total,
            size=(n, max(int(fine2coarse.max().item()) + 1 if coarse_nodes.numel() > 0 else 1, 1)),
            device=device,
            dtype=dtype,
        ).coalesce()
        P_csr = P_coo.to_sparse_csr()

    # Row normalization for F-points
    crow_P = P_csr.crow_indices()
    col_P = P_csr.col_indices()
    val_P = P_csr.values()
    if val_P.numel() > 0:
        row_lengths = crow_P[1:] - crow_P[:-1]
        row_idx_P = torch.repeat_interleave(torch.arange(n, device=device, dtype=torch.int64), row_lengths)
        row_sums = torch.zeros(n, device=device, dtype=dtype)
        if row_idx_P.numel() > 0:
            row_sums.scatter_add_(0, row_idx_P, val_P)
        scale = torch.ones(n, device=device, dtype=dtype)
        mask_F_rows = F_mask & (row_sums > 1e-14)
        if mask_F_rows.any():
            scale[mask_F_rows] = 1.0 / row_sums[mask_F_rows]
        zero_rows = F_mask & (row_sums <= 1e-14)
        if zero_rows.any():
            # fallback: inject to first coarse neighbour (strong or weak)
            fallback_nodes = torch.nonzero(zero_rows, as_tuple=False).view(-1)
            for node in fallback_nodes.tolist():
                start = crow[node].item()
                end = crow[node + 1].item()
                neigh = col[start:end]
                coarse_neigh = neigh[coarse_mask[neigh]]
                if coarse_neigh.numel() == 0:
                    continue
                target_c = fine2coarse[coarse_neigh[0]]
                crow_insert = torch.empty(n + 1, device=device, dtype=torch.int64)
                col_insert = torch.empty(val_P.numel() + 1, device=device, dtype=torch.int64)
                val_insert = torch.empty(val_P.numel() + 1, device=device, dtype=dtype)
                crow_insert[: node + 1] = crow_P[: node + 1]
                crow_insert[node + 1 :] = crow_P[node + 1 :] + 1
                if crow_P[node] > 0:
                    col_insert[: crow_P[node]] = col_P[: crow_P[node]]
                    val_insert[: crow_P[node]] = val_P[: crow_P[node]]
                col_insert[crow_P[node]] = target_c
                val_insert[crow_P[node]] = 1.0
                if crow_P[node] < val_P.numel():
                    col_insert[crow_P[node] + 1 :] = col_P[crow_P[node] :]
                    val_insert[crow_P[node] + 1 :] = val_P[crow_P[node] :]
                crow_P = crow_insert
                col_P = col_insert
                val_P = val_insert
                row_lengths = crow_P[1:] - crow_P[:-1]
                row_idx_P = torch.repeat_interleave(torch.arange(n, device=device, dtype=torch.int64), row_lengths)
                row_sums = torch.zeros(n, device=device, dtype=dtype)
                if row_idx_P.numel() > 0:
                    row_sums.scatter_add_(0, row_idx_P, val_P)
                scale = torch.ones(n, device=device, dtype=dtype)
                mask_F_rows = F_mask & (row_sums > 1e-14)
                if mask_F_rows.any():
                    scale[mask_F_rows] = 1.0 / row_sums[mask_F_rows]

        val_P = val_P * scale[row_idx_P]
        if drop_tol > 0.0:
            drop_mask = val_P.abs() < drop_tol
            if drop_mask.any():
                val_P = torch.where(drop_mask, torch.zeros_like(val_P), val_P)
                row_sums = torch.zeros(n, device=device, dtype=dtype)
                row_sums.scatter_add_(0, row_idx_P, val_P)
                rescale = torch.ones(n, device=device, dtype=dtype)
                mask_rows = F_mask & (row_sums > 1e-14)
                if mask_rows.any():
                    rescale[mask_rows] = 1.0 / row_sums[mask_rows]
                val_P = val_P * rescale[row_idx_P]

        val_P = torch.clamp(val_P, min=0.0)
        P_csr = torch.sparse_csr_tensor(crow_P, col_P, val_P, size=P_csr.size(), device=device, dtype=dtype)

    # Energy-minimizing blending (optional)
    basis_dev = None
    if basis is not None and basis.numel() > 0 and blend > 0.0:
        basis_dev = basis
        if basis_dev.dim() == 1:
            basis_dev = basis_dev.unsqueeze(1)
        if basis_dev.size(0) != n:
            raise ValueError("basis размер не соответствует числу узлов")
        basis_dev = basis_dev.to(device=device, dtype=dtype)
        extra_terms: list[torch.Tensor] = []
        if coords is not None and coords.size(1) > 0:
            for d in range(coords.size(1)):
                term = coords[:, d] * coords[:, d]
                term = term - term.mean()
                if term.norm() > 1e-8:
                    extra_terms.append(term.unsqueeze(1))
            if coords.size(1) > 1:
                for d0 in range(coords.size(1)):
                    for d1 in range(d0 + 1, coords.size(1)):
                        term = coords[:, d0] * coords[:, d1]
                        term = term - term.mean()
                        if term.norm() > 1e-8:
                            extra_terms.append(term.unsqueeze(1))
        if extra_terms:
            extra_mat = torch.cat(extra_terms, dim=1)
            basis_dev = torch.cat([basis_dev, extra_mat], dim=1)
        # обеспечиваем ортонормированность столбцов
        basis_dev = orthonormalize_columns(basis_dev.to('cpu')).to(device=device, dtype=dtype)

    if basis_dev is not None and blend > 0.0:
        crow_P = P_csr.crow_indices()
        col_P = P_csr.col_indices()
        val_P = P_csr.values()
        row_lengths = crow_P[1:] - crow_P[:-1]
        F_rows = torch.nonzero(F_mask, as_tuple=False).view(-1)
        if F_rows.numel() > 0:
            F_lengths = row_lengths[F_rows]
            unique_k = torch.unique(F_lengths)
            for k in unique_k.tolist():
                if k <= 0:
                    continue
                rows_mask = F_lengths == k
                if not rows_mask.any():
                    continue
                rows_sel = F_rows[rows_mask]
                starts = crow_P[rows_sel]
                offsets = torch.arange(k, device=device, dtype=torch.int64)
                idx_matrix = starts.unsqueeze(1) + offsets.unsqueeze(0)
                cols_matrix = col_P[idx_matrix]
                rs_weights = val_P[idx_matrix]

                coarse_global = coarse_nodes[cols_matrix]
                B_c = basis_dev[coarse_global].transpose(1, 2)  # (num_rows, m, k)
                target = basis_dev[rows_sel].unsqueeze(2)  # (num_rows, m, 1)

                ones = torch.ones(B_c.size(0), 1, k, dtype=dtype, device=device)
                M = torch.cat([B_c, ones], dim=1)  # (num_rows, m+1, k)
                rhs = torch.cat(
                    [target, torch.ones(B_c.size(0), 1, 1, dtype=dtype, device=device)],
                    dim=1
                )  # (num_rows, m+1, 1)

                M_t = M.transpose(1, 2)  # (num_rows, k, m+1)
                ATA = torch.matmul(M_t, M)
                if ls_reg > 0:
                    eye = torch.eye(k, dtype=dtype, device=device).unsqueeze(0).expand_as(ATA)
                    ATA = ATA + ls_reg * eye
                ATy = torch.matmul(M_t, rhs)
                try:
                    w_ls = torch.linalg.solve(ATA, ATy).squeeze(2)
                except RuntimeError:
                    # fallback к RS весам, если система вырождена
                    continue

                w_ls = torch.clamp(w_ls, min=0.0)
                w_sum = w_ls.sum(dim=1, keepdim=True).clamp_min(1e-12)
                w_ls = w_ls / w_sum

                w_blend = (1.0 - blend) * rs_weights + blend * w_ls
                w_blend = torch.clamp(w_blend, min=0.0)
                w_blend = w_blend / w_blend.sum(dim=1, keepdim=True).clamp_min(1e-12)

                val_P[idx_matrix] = w_blend
        P_csr = torch.sparse_csr_tensor(crow_P, col_P, val_P, size=P_csr.size(), device=device, dtype=dtype)

    return (
        P_csr,
        cf,
        fine2coarse,
        coarse_nodes.to(dtype=torch.int64),
        torch.nonzero(F_mask, as_tuple=False).view(-1).to(dtype=torch.int64),
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


def _csr_spmm(left: torch.Tensor, right: torch.Tensor, block_rows: int = 64) -> torch.Tensor:
    """Sparse CSR x CSR → CSR using blockwise gather/scatter на устройстве с минимальной пиковой памятью."""
    device = left.device
    dtype = left.dtype

    left_crow = left.crow_indices()
    left_col = left.col_indices()
    left_val = left.values()
    right_crow = right.crow_indices()
    right_col = right.col_indices()
    right_val = right.values()

    n_rows = left.size(0)
    n_cols = right.size(1)

    if left_val.numel() == 0 or right_val.numel() == 0:
        crow_zero = torch.zeros(n_rows + 1, device=device, dtype=torch.int64)
        return torch.sparse_csr_tensor(
            crow_zero,
            torch.zeros(0, device=device, dtype=torch.int64),
            torch.zeros(0, device=device, dtype=dtype),
            size=(n_rows, n_cols),
            device=device,
            dtype=dtype,
        )

    right_row_counts = right_crow[1:] - right_crow[:-1]

    crow_out = torch.zeros(n_rows + 1, device=device, dtype=torch.int64)
    col_blocks: list[torch.Tensor] = []
    val_blocks: list[torch.Tensor] = []
    nnz_total = 0

    for row_start in range(0, n_rows, block_rows):
        row_end = min(row_start + block_rows, n_rows)
        if row_start >= row_end:
            continue

        nz_start = int(left_crow[row_start].item())
        nz_end = int(left_crow[row_end].item())
        if nz_start == nz_end:
            crow_out[row_start:row_end] = nnz_total
            continue

        block_cols = left_col[nz_start:nz_end]
        block_vals = left_val[nz_start:nz_end]
        block_row_counts = left_crow[row_start + 1 : row_end + 1] - left_crow[row_start:row_end]
        local_range = row_end - row_start
        local_rows = torch.arange(local_range, device=device, dtype=torch.int64)
        block_row_idx = torch.repeat_interleave(local_rows, block_row_counts)

        counts = right_row_counts[block_cols]
        mask = counts > 0
        if not mask.any():
            crow_out[row_start:row_end] = nnz_total
            continue

        counts = counts[mask]
        block_rows_nz = block_row_idx[mask]
        block_cols_nz = block_cols[mask]
        block_vals_nz = block_vals[mask]
        starts = right_crow[block_cols_nz]

        total = int(counts.sum().item())
        if total == 0:
            crow_out[row_start:row_end] = nnz_total
            continue

        cum_counts = torch.cumsum(counts, dim=0)
        base = cum_counts - counts
        start_rep = torch.repeat_interleave(starts, counts)
        base_rep = torch.repeat_interleave(base, counts)
        intra = torch.arange(total, device=device, dtype=torch.int64) - base_rep
        right_positions = start_rep + intra

        rows_chunk = torch.repeat_interleave(block_rows_nz, counts)
        cols_chunk = right_col[right_positions]
        right_vals_flat = right_val[right_positions]
        left_vals_rep = torch.repeat_interleave(block_vals_nz, counts)
        vals_chunk = left_vals_rep * right_vals_flat

        if rows_chunk.numel() == 0:
            crow_out[row_start:row_end] = nnz_total
            continue

        keys = rows_chunk.to(torch.int64) * n_cols + cols_chunk
        keys_sorted, order = torch.sort(keys)
        vals_sorted = vals_chunk[order]
        unique_keys, inverse = torch.unique_consecutive(keys_sorted, return_inverse=True)
        vals_reduced = torch.zeros(unique_keys.size(0), device=device, dtype=dtype)
        vals_reduced.scatter_add_(0, inverse, vals_sorted)

        rows_unique = torch.div(unique_keys, n_cols, rounding_mode='floor')
        cols_unique = unique_keys - rows_unique * n_cols

        row_counts = torch.bincount(rows_unique, minlength=local_range)
        block_crow = torch.zeros(local_range + 1, device=device, dtype=torch.int64)
        block_crow[1:] = torch.cumsum(row_counts, dim=0)

        crow_out[row_start:row_end] = nnz_total + block_crow[:-1]
        nnz_block = int(block_crow[-1].item())
        nnz_total += nnz_block

        if nnz_block > 0:
            block_cols_out = cols_unique
            block_vals_out = vals_reduced
            col_blocks.append(block_cols_out)
            val_blocks.append(block_vals_out)

    crow_out[n_rows] = nnz_total

    if nnz_total == 0 or not col_blocks:
        crow_zero = torch.zeros(n_rows + 1, device=device, dtype=torch.int64)
        return torch.sparse_csr_tensor(
            crow_zero,
            torch.zeros(0, device=device, dtype=torch.int64),
            torch.zeros(0, device=device, dtype=dtype),
            size=(n_rows, n_cols),
            device=device,
            dtype=dtype,
        )

    col_final = torch.cat(col_blocks)
    val_final = torch.cat(val_blocks)
    return torch.sparse_csr_tensor(
        crow_out,
        col_final,
        val_final,
        size=(n_rows, n_cols),
        device=device,
        dtype=dtype,
    )


def _drop_small_entries_csr(A_csr: torch.Tensor, tol: float) -> torch.Tensor:
    if tol <= 0.0 or A_csr.values().numel() == 0:
        return A_csr

    coo = A_csr.to_sparse_coo().coalesce()
    idx = coo.indices()
    val = coo.values()
    diag_mask = idx[0] == idx[1]
    keep_mask = (val.abs() >= tol) | diag_mask

    if keep_mask.all():
        return A_csr

    idx_keep = idx[:, keep_mask]
    val_keep = val[keep_mask]

    if idx_keep.numel() == 0:
        return A_csr

    A_keep = torch.sparse_coo_tensor(idx_keep, val_keep, A_csr.size(), device=A_csr.device, dtype=A_csr.dtype).coalesce()
    A_csr_new = A_keep.to_sparse_csr()
    crow = A_csr_new.crow_indices()
    if (crow[1:] == crow[:-1]).any():
        return A_csr
    return A_csr_new


def rap_torch(A_csr: torch.Tensor, P_csr: torch.Tensor, device: torch.device, drop_tol: float = 0.0) -> torch.Tensor:
    AP = _csr_spmm(A_csr, P_csr)
    PT = P_csr.transpose(0, 1).to_sparse_csr()
    Ac = _csr_spmm(PT, AP)
    if drop_tol > 0.0:
        Ac = _drop_small_entries_csr(Ac, drop_tol)
    return Ac.to(device=device)


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
                 nullspace_reg: float = 1e-6,
                 near_nullspace: torch.Tensor | None = None,
                 node_coords: torch.Tensor | None = None,
                 coarsening_method: str = "pmis",
                 mixed_precision: bool = False,
                 mixed_start_level: int = 2,
                 cpu_offload: bool = False,
                 offload_level: int = 3):
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
            near_nullspace: Пользовательский near-nullspace (n x m), перекрывает generate_nullspace
            node_coords: Геометрия узлов (n x d), используется для диагностики и energy-minimizing интерполяции
            coarsening_method: 'rs' или 'pmis' (по умолчанию pmis для более агрессивного сжатия)
            mixed_precision: Переводить ли глубинные уровни (>= mixed_start_level) в float32
            mixed_start_level: Уровень, начиная с которого применяется mixed precision
            cpu_offload: Переносить ли уровни (>= offload_level) на CPU после сборки
            offload_level: Первый уровень, который переводится на CPU (при cpu_offload=True)
        """
        self.theta = theta
        self.max_levels = max_levels
        self.coarsest_size = coarsest_size
        self.anchor_idx = anchor_idx
        self.nullspace_dim = max(0, nullspace_dim)
        self.base_nullspace_blend = float(max(0.0, min(1.0, nullspace_blend)))
        self.nullspace_blend = 0.18
        self.nullspace_reg = float(max(1e-10, nullspace_reg))
        self.coarsening_method = coarsening_method.lower()
        self.mixed_precision = bool(mixed_precision)
        self.mixed_start_level = max(0, int(mixed_start_level))
        self.cpu_offload = bool(cpu_offload)
        self.offload_level = max(0, int(offload_level))
        self.theta_decay = 0.90
        self.theta_min = max(0.05, min(theta, 0.12))
        self.blend_growth = 1.15
        self.blend_cap = 0.85
        self.coarsest_fraction = 1e-3
        self.coarsening_retry_factor = 0.7
        self.coarsening_max_retries = 3
        self.coarsening_target_cf = 0.35
        self.coarsening_max_fraction = 0.55
        self.drop_tolerance = 1e-6
        self.rap_drop_tol = 1e-7
        self._external_nullspace = near_nullspace
        self._external_coords = node_coords
        self.apply_tol = 5e-2
        self.apply_max_cycles = 8
        self.debug_cycles = False
        self.equilibration_threshold = 1e-6
        self.use_equilibration = False
        self.root_dtype = torch.float64
        if not torch.cuda.is_available():
            self.cpu_offload = False
        
        # Конвертируем numpy CSR -> torch CSR на CUDA
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.primary_device = device
        
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
        
        n_total = A_csr.size(0)
        self.n_initial = n_total
        dynamic_target = int(max(1, n_total * self.coarsest_fraction))
        self.coarsest_target = max(self.coarsest_size, dynamic_target)
        print(f"[ClassicalAMG] Целевой размер coarsest уровня: {self.coarsest_target} (из {self.coarsest_size})")
        if self._external_coords is not None:
            coords_ext = self._external_coords
            if coords_ext.dim() == 1:
                coords_ext = coords_ext.unsqueeze(1)
            if coords_ext.size(0) != n_total:
                raise ValueError(
                    f"node_coords size mismatch: expected {n_total}, got {coords_ext.size(0)}"
                )
            coords_ext = coords_ext.to(device=self.device, dtype=torch.float64)
            node_coords_current = coords_ext
        else:
            node_coords_current = torch.arange(n_total, dtype=torch.float64, device=self.device).unsqueeze(1)

        # ═══════════════════════════════════════════════════════════════
        # АДАПТИВНОЕ EQUILIBRATION
        # ═══════════════════════════════════════════════════════════════
        # Проверяем нужно ли equilibration:
        # - Если median(diag) >> 1e-6: матрица хорошо масштабирована, equilibration НЕ НУЖЕН
        # - Если median(diag) << 1e-6: матрица плохая, equilibration КРИТИЧЕН
        # ═══════════════════════════════════════════════════════════════
        if near_nullspace is not None:
            self._external_nullspace = near_nullspace
        if node_coords is not None:
            self._external_coords = node_coords

        n_total = A_csr.size(0)

        diag_orig = self._extract_diag(A_csr).abs().clamp_min(1e-30)
        diag_median = diag_orig.median().item()
        
        # Порог для equilibration: если диагональ слишком мала
        EQUILIBRATION_THRESHOLD = 1e-6
        use_equilibration = (diag_median < EQUILIBRATION_THRESHOLD)
        self.use_equilibration = use_equilibration
        
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
        if self._external_nullspace is not None:
            basis_ext = self._external_nullspace
            if basis_ext.dim() == 1:
                basis_ext = basis_ext.unsqueeze(1)
            if basis_ext.size(0) != n_total:
                raise ValueError(
                    f"near_nullspace size mismatch: expected {n_total}, got {basis_ext.size(0)}"
                )
            basis_current = orthonormalize_columns(basis_ext.to(dtype=torch.float64, device='cpu'))
        else:
            basis_current = generate_nullspace(A_current, self.nullspace_dim)
        build_start = time.perf_counter()
        for lvl in range(max_levels):
            level_start = time.perf_counter()
            n = A_current.size(0)
            
            theta_lvl = max(self.theta_min, self.theta * (self.theta_decay ** lvl))
            dynam_blend = self.base_nullspace_blend * (self.blend_growth ** max(0, lvl - 4))
            blend_lvl = min(self.blend_cap, self.nullspace_blend + dynam_blend)
            if n <= self.coarsest_target:
                diag_abs = self._extract_diag(A_current).abs()
                row_abs = self._row_abs_sum(A_current)
                beta = 0.3
                denom = torch.maximum(diag_abs, beta * row_abs).clamp_min(1e-30)
                inv_relax = (1.0 / denom).clamp(max=3e1)
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
                self._update_level_metrics(len(self.levels) - 1)
                self._apply_memory_policy(len(self.levels) - 1)
                break
            
            theta_eff = theta_lvl
            retries = 0
            strong_mask = None
            cf_marker = None
            n_coarse = n
            c_fraction = 1.0
            theta_lower = self.theta_min
            theta_upper = max(0.95, theta_eff)
            while True:
                strong_mask = find_strong_connections(A_current, theta_eff)
                if (self.coarsening_method == "pmis" and lvl >= 2):
                    cf_marker, n_coarse = pmis_coarsening(A_current, strong_mask)
                else:
                    cf_marker, n_coarse = classical_coarsening(A_current, theta_eff)
                c_fraction = n_coarse / max(n, 1)
                if (
                    self.coarsening_target_cf <= c_fraction <= self.coarsening_max_fraction
                    or retries >= self.coarsening_max_retries
                ):
                    break
                if c_fraction < self.coarsening_target_cf:
                    theta_eff = min(theta_upper, theta_eff * 1.4 + 1e-3)
                else:
                    theta_eff = max(theta_lower, theta_eff * self.coarsening_retry_factor)
                retries += 1

            stage_after_cf = time.perf_counter()
            print(
                f"[ClassicalAMG] L{lvl} timings: strong+coarsen={stage_after_cf - level_start:.3f}s "
                f"(theta={theta_eff:.3f}, c_frac={c_fraction:.3f}, retries={retries})"
            )
            if self.anchor_idx is not None and self.anchor_idx < n:
                cf_marker[self.anchor_idx] = 0

            fallback_used = False
            if c_fraction < self.coarsening_target_cf:
                fallback_used = True
                stride = max(1, int(1.0 / max(self.coarsening_target_cf, 1e-3)))
                coarse_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
                coarse_mask[::stride] = True
                coarse_mask[-1] = True
                cf_marker = torch.ones(n, dtype=torch.int8, device=self.device)
                cf_marker[coarse_mask] = 0
                if self.anchor_idx is not None and self.anchor_idx < n:
                    cf_marker[self.anchor_idx] = 0
                n_coarse = int((cf_marker == 0).sum().item())
                c_fraction = n_coarse / max(n, 1)
                print(
                    f"[ClassicalAMG] L{lvl}: stride fallback (stride={stride}) → c_frac={c_fraction:.3f}"
                )
            elif n_coarse >= n * 0.95:
                fallback_used = True
                coarse_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
                coarse_mask[::2] = True
                coarse_mask[-1] = True
                cf_marker = torch.ones(n, dtype=torch.int8, device=self.device)
                cf_marker[coarse_mask] = 0
                if self.anchor_idx is not None and self.anchor_idx < n:
                    cf_marker[self.anchor_idx] = 0
                n_coarse = int((cf_marker == 0).sum().item())
                c_fraction = n_coarse / max(n, 1)
                print(
                    f"[ClassicalAMG] L{lvl}: fallback coarsening applied (c_frac={c_fraction:.3f})"
                )
            else:
                coarse_mask = (cf_marker == 0)
                c_fraction_actual = coarse_mask.to(torch.float64).mean().item()
                if c_fraction_actual > self.coarsening_max_fraction:
                    fallback_used = True
                    stride = max(2, int(1.0 / max(self.coarsening_target_cf, 1e-3)))
                    coarse_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
                    coarse_mask[::stride] = True
                    coarse_mask[-1] = True
                    cf_marker = torch.ones(n, dtype=torch.int8, device=self.device)
                    cf_marker[coarse_mask] = 0
                    if self.anchor_idx is not None and self.anchor_idx < n:
                        cf_marker[self.anchor_idx] = 0
                    n_coarse = int((cf_marker == 0).sum().item())
                    c_fraction = n_coarse / max(n, 1)
                    print(
                        f"[ClassicalAMG] L{lvl}: uniform stride fallback (stride={stride}) → c_frac={c_fraction:.3f}"
                    )
            
            # Prolongation: RS-интерполяция с F–F коррекцией (P·1=1)
            P, cf_marker, fine2coarse_map, coarse_nodes, F_nodes = build_prolongation_rs_full(
                A_current,
                cf_marker,
                strong_mask,
                basis_current,
                blend=blend_lvl if basis_current is not None else 0.0,
                ls_reg=self.nullspace_reg,
                node_coords=node_coords_current,
                drop_tol=self.drop_tolerance,
            )
            stage_after_prol = time.perf_counter()
            print(
                f"[ClassicalAMG] L{lvl} timings: prolongation={stage_after_prol - stage_after_cf:.3f}s "
                f"(blend={blend_lvl:.2f}, fallback={fallback_used})"
            )
            if P.device != self.device:
                P = P.to(self.device)
            fine2coarse_map = fine2coarse_map.to(self.device)
            coarse_nodes = coarse_nodes.to(self.device, dtype=torch.int64)
            F_nodes = F_nodes.to(self.device, dtype=torch.int64)

            interp_metrics = self._analyze_prolongation(lvl, P, coarse_nodes, F_nodes, node_coords_current)

            A_coarse = rap_torch(A_current, P, self.device, drop_tol=self.rap_drop_tol)
            stage_after_rap = time.perf_counter()
            print(f"[ClassicalAMG] L{lvl} timings: RAP={stage_after_rap - stage_after_prol:.3f}s")
            print(f"[ClassicalAMG] L{lvl} total build time={stage_after_rap - level_start:.3f}s")

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
            inv_relax = (1.0 / denom).clamp(max=3e1)
            
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
            self._update_level_metrics(len(self.levels) - 1)
            self._apply_memory_policy(len(self.levels) - 1)
            
            node_coords_current = node_coords_current[coarse_nodes]
            if basis_current is not None:
                basis_gpu = basis_current.to(device=P.device, dtype=P.dtype)
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
            self._update_level_metrics(len(self.levels) - 1)
            self._apply_memory_policy(len(self.levels) - 1)
        
        total_build = time.perf_counter() - build_start
        print(f"✅ ClassicalAMG: построено {len(self.levels)} уровней за {total_build:.2f} c")
    
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
        level = self.levels[lvl]
        if level.get('use_chebyshev', False):
            return self._chebyshev_smooth(lvl, x, b, nu=nu, debug=debug)
        return self._jacobi_smooth(lvl, x, b, nu=nu, debug=debug)

    def _jacobi_smooth(self, lvl: int, x: torch.Tensor, b: torch.Tensor, nu: int = 1, debug: bool = False) -> torch.Tensor:
        omega = 0.7
        inv_relax = self.levels[lvl]['inv_relax']
        for it in range(nu):
            r = b - self._matvec(lvl, x)
            delta = omega * inv_relax * r
            x = x + delta
            if debug:
                print(f"    [Jacobi L{lvl} iter{it+1}] ||r||={r.norm():.3e}, ||δ||={delta.norm():.3e}")
        return x

    def _chebyshev_smooth(self, lvl: int, x: torch.Tensor, b: torch.Tensor, nu: int = 2, debug: bool = False) -> torch.Tensor:
        level = self.levels[lvl]
        lambda_max = level.get('lambda_max', None)
        lambda_min = level.get('lambda_min', None)
        if lambda_max is None or lambda_min is None or lambda_max <= lambda_min or lambda_max <= 1e-8:
            return self._jacobi_smooth(lvl, x, b, nu=nu, debug=debug)

        d = 0.5 * (lambda_max + lambda_min)
        c = 0.5 * (lambda_max - lambda_min)
        if d <= 0 or c <= 0:
            return self._jacobi_smooth(lvl, x, b, nu=nu, debug=debug)

        inv_diag = level['inv_diag_cheb']
        p = torch.zeros_like(x)
        for it in range(nu):
            r = b - self._matvec(lvl, x)
            if it == 0:
                beta = 0.0
            else:
                beta = (c / (2.0 * d)) ** 2
            p = inv_diag * r + beta * p
            x = x + (1.0 / d) * p
            if debug:
                print(f"    [Chebyshev L{lvl} iter{it+1}] ||r||={r.norm():.3e}")
        return x
    
    def _v_cycle(self, lvl: int, x: torch.Tensor, b: torch.Tensor,
                 pre_smooth: int = 3, post_smooth: int = 2, debug: bool = False, cycle_num: int = 0) -> torch.Tensor:
        """V-cycle"""
        level = self.levels[lvl]
        level_device = level.get('device', self.device)
        level_dtype = level.get('dtype', torch.float64)
        if x.device != level_device or x.dtype != level_dtype:
            x = x.to(device=level_device, dtype=level_dtype)
        if b.device != level_device or b.dtype != level_dtype:
            b = b.to(device=level_device, dtype=level_dtype)
        
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

            if n <= 2000:
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
        child = self.levels[lvl + 1]
        child_device = child.get('device', self.device)
        child_dtype = child.get('dtype', torch.float64)
        if r_c.device != child_device or r_c.dtype != child_dtype:
            r_c = r_c.to(device=child_device, dtype=child_dtype)
        if debug:
            print(f"  [V-CYCLE L{lvl}] RESTRICT: ||r||={r.norm():.3e} → ||r_c||={r_c.norm():.3e}")
        
        # Coarse solve
        e_c = torch.zeros_like(r_c)
        e_c = self._v_cycle(lvl + 1, e_c, r_c, pre_smooth, post_smooth, debug=debug, cycle_num=cycle_num)
        if debug:
            print(f"  [V-CYCLE L{lvl}] COARSE возвращает: ||e_c||={e_c.norm():.3e}, max|e_c|={e_c.abs().max():.3e}")
        
        # Prolongate: e_f = P * e_c
        if e_c.device != level_device or e_c.dtype != level_dtype:
            e_c = e_c.to(device=level_device, dtype=level_dtype)
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
        root_device = self.levels[0].get('device', self.device)
        root_dtype = self.levels[0].get('dtype', self.root_dtype if hasattr(self, "root_dtype") else torch.float64)
        b = b.to(root_device, dtype=root_dtype)
        self.device = root_device
        self.root_dtype = root_dtype
        
        # Проверяем нужно ли equilibration
        if self.Dhalf_inv is not None:
            # С equilibration
            b_scaled = self.Dhalf_inv * b
            x_scaled = torch.zeros_like(b_scaled) if x0 is None else x0.to(root_device, dtype=root_dtype) / self.Dhalf_inv
        else:
            # Без equilibration - работаем напрямую
            b_scaled = b.clone()
            x_scaled = torch.zeros_like(b) if x0 is None else x0.to(root_device, dtype=root_dtype)
        
        b_norm = b_scaled.norm()
        
        for cycle in range(max_iter):
            x_scaled = self._v_cycle(0, x_scaled, b_scaled, pre_smooth=3, post_smooth=2,
                                     debug=self.debug_cycles, cycle_num=cycle)
            if x_scaled.device != root_device or x_scaled.dtype != root_dtype:
                x_scaled = x_scaled.to(device=root_device, dtype=root_dtype)
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
        root_device = self.levels[0].get('device', self.device)
        root_dtype = self.levels[0].get('dtype', self.root_dtype if hasattr(self, "root_dtype") else torch.float64)
        b = b.to(root_device, dtype=root_dtype)
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
        return x_scaled.to(device=root_device, dtype=root_dtype)

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
        if coords_fine.dim() == 1 or coords_fine.size(-1) == 1:
            coords_vec = coords_fine.view(-1)
            coords_coarse = coords_vec[coarse_nodes.long()]
            grad_recon = torch.sparse.mm(P, coords_coarse.unsqueeze(1)).squeeze(1)
            grad_err = grad_recon - coords_vec
            grad_err_abs = grad_err.abs()
            grad_rel = grad_err.norm() / (coords_vec.norm() + 1e-30)
        else:
            coords_mat = coords_fine
            coords_coarse = coords_mat[coarse_nodes.long()]
            grad_components = []
            for d in range(coords_mat.size(1)):
                recon = torch.sparse.mm(P, coords_coarse[:, d].unsqueeze(1)).squeeze(1)
                grad_components.append(recon - coords_mat[:, d])
            grad_err_stack = torch.stack(grad_components, dim=1)
            grad_err_abs = torch.linalg.norm(grad_err_stack, dim=1)
            grad_rel = grad_err_stack.norm() / (coords_mat.norm() + 1e-30)

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

    def _update_level_metrics(self, lvl: int) -> None:
        A_lvl = self.levels[lvl]['A']
        diag_abs = self._extract_diag(A_lvl).abs()
        row_abs = self._row_abs_sum(A_lvl)
        beta = 0.3
        denom = torch.maximum(diag_abs, beta * row_abs).clamp_min(1e-30)
        inv_relax = (1.0 / denom).clamp(max=3e1)
        self.levels[lvl]['diag'] = diag_abs
        self.levels[lvl]['inv_relax'] = inv_relax
        ratio = row_abs / diag_abs.clamp_min(1e-30)
        lambda_max = float(torch.max(ratio).item())
        if not math.isfinite(lambda_max) or lambda_max <= 1e-8:
            lambda_max = 1.0
        lambda_min = max(0.05 * lambda_max, 1e-6)
        self.levels[lvl]['lambda_max'] = lambda_max
        self.levels[lvl]['lambda_min'] = lambda_min
        self.levels[lvl]['inv_diag_cheb'] = 1.0 / diag_abs.clamp_min(1e-30)
        self.levels[lvl]['use_chebyshev'] = self.levels[lvl]['n'] > 5000

    @staticmethod
    def _convert_sparse_tensor(tensor: torch.Tensor | None,
                               device: torch.device,
                               dtype: torch.dtype) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.device == device and tensor.dtype == dtype:
            return tensor
        return torch.sparse_csr_tensor(
            tensor.crow_indices().to(device),
            tensor.col_indices().to(device),
            tensor.values().to(device=device, dtype=dtype),
            size=tensor.size(),
            device=device,
            dtype=dtype,
        )

    def _policy_device_dtype(self, lvl: int) -> tuple[torch.device, torch.dtype]:
        target_device = self.primary_device
        if self.cpu_offload and self.primary_device.type == 'cuda' and lvl >= self.offload_level:
            target_device = torch.device('cpu')
        target_dtype = torch.float64
        if self.mixed_precision and lvl >= self.mixed_start_level:
            target_dtype = torch.float32
        return target_device, target_dtype

    def _convert_level_storage(self, lvl: int,
                               target_device: torch.device,
                               target_dtype: torch.dtype) -> None:
        level = self.levels[lvl]
        level['A'] = self._convert_sparse_tensor(level['A'], target_device, target_dtype)
        if level.get('P') is not None:
            level['P'] = self._convert_sparse_tensor(level['P'], target_device, target_dtype)
        if 'diag' in level:
            level['diag'] = level['diag'].to(device=target_device, dtype=target_dtype)
        if 'inv_relax' in level:
            level['inv_relax'] = level['inv_relax'].to(device=target_device, dtype=target_dtype)
        if 'inv_diag_cheb' in level:
            level['inv_diag_cheb'] = level['inv_diag_cheb'].to(device=target_device, dtype=target_dtype)
        if 'basis' in level and level['basis'] is not None:
            level['basis'] = level['basis'].to(device=target_device, dtype=target_dtype)
        if 'node_coords' in level and level['node_coords'] is not None:
            level['node_coords'] = level['node_coords'].to(device=target_device, dtype=target_dtype)
        if 'fine2coarse' in level:
            level['fine2coarse'] = level['fine2coarse'].to(device=target_device, dtype=torch.int64)
        if 'coarse_nodes' in level:
            level['coarse_nodes'] = level['coarse_nodes'].to(device=target_device, dtype=torch.int64)
        if 'F_nodes' in level:
            level['F_nodes'] = level['F_nodes'].to(device=target_device, dtype=torch.int64)

    def _apply_memory_policy(self, lvl: int) -> None:
        target_device, target_dtype = self._policy_device_dtype(lvl)
        self._convert_level_storage(lvl, target_device, target_dtype)
        level = self.levels[lvl]
        level['device'] = target_device
        level['dtype'] = target_dtype
        if lvl == 0:
            self.device = target_device
            self.root_dtype = target_dtype
            if self.Dhalf_inv is not None:
                self.Dhalf_inv = self.Dhalf_inv.to(device=target_device, dtype=target_dtype)

    def update_matrix(self, A_csr_np: torch.Tensor,
                      near_nullspace: torch.Tensor | None = None,
                      node_coords: torch.Tensor | None = None) -> None:
        """Обновляет значения матрицы, сохраняя построенную иерархию."""
        if near_nullspace is not None:
            self._external_nullspace = near_nullspace
        if node_coords is not None:
            self._external_coords = node_coords

        root_device = self.levels[0].get('device', self.device)
        root_dtype = self.levels[0].get('dtype', self.root_dtype if hasattr(self, "root_dtype") else torch.float64)
        self.device = root_device
        self.root_dtype = root_dtype

        if hasattr(A_csr_np, 'indptr'):  # scipy sparse
            import scipy.sparse as sp
            if not sp.isspmatrix_csr(A_csr_np):
                A_csr_np = A_csr_np.tocsr()
            crow = torch.from_numpy(A_csr_np.indptr).to(root_device, dtype=torch.int64)
            col = torch.from_numpy(A_csr_np.indices).to(root_device, dtype=torch.int64)
            val = torch.from_numpy(A_csr_np.data).to(root_device, dtype=root_dtype)
            A_csr = torch.sparse_csr_tensor(crow, col, val, size=A_csr_np.shape, device=root_device, dtype=root_dtype)
        else:
            A_csr = A_csr_np.to(root_device)
            if A_csr.layout == torch.sparse_coo:
                A_csr = A_csr.coalesce().to_sparse_csr()
            elif A_csr.layout != torch.sparse_csr:
                A_csr = A_csr.to_sparse().coalesce().to_sparse_csr()
            if A_csr.dtype != root_dtype:
                A_csr = torch.sparse_csr_tensor(
                    A_csr.crow_indices(),
                    A_csr.col_indices(),
                    A_csr.values().to(dtype=root_dtype),
                    size=A_csr.size(),
                    device=root_device,
                    dtype=root_dtype,
                )

        n_total = A_csr.size(0)

        diag_orig = self._extract_diag(A_csr).abs().clamp_min(1e-30)
        diag_median = diag_orig.median().item()
        use_equilibration = (diag_median < self.equilibration_threshold)
        self.use_equilibration = use_equilibration
        print(
            f"[ClassicalAMG] update: diag min={diag_orig.min():.3e}, "
            f"median={diag_median:.3e}, max={diag_orig.max():.3e}, "
            f"equil={'ON' if use_equilibration else 'OFF'}"
        )

        if use_equilibration:
            Dhalf_inv = 1.0 / torch.sqrt(diag_orig)
            crow = A_csr.crow_indices()
            col = A_csr.col_indices()
            vals = A_csr.values().clone()
            row_len = crow[1:] - crow[:-1]
            row_idx = torch.repeat_interleave(torch.arange(crow.numel() - 1, device=root_device), row_len)
            vals = vals * Dhalf_inv[row_idx] * Dhalf_inv[col]
            A_csr = torch.sparse_csr_tensor(crow, col, vals, size=A_csr.size(), device=root_device, dtype=root_dtype)
            self.Dhalf_inv = Dhalf_inv
        else:
            self.Dhalf_inv = None

        if self.anchor_idx is not None:
            A_csr = _apply_reference_fix(A_csr, int(self.anchor_idx))

        diag_scaled = self._extract_diag(A_csr).abs()
        print(
            f"[ClassicalAMG] update: diag scaled range {diag_orig.min():.2e}..{diag_orig.max():.2e} → "
            f"{diag_scaled.min():.2e}..{diag_scaled.max():.2e}"
        )

        if A_csr.size(0) != self.levels[0]['n']:
            raise ValueError(f"AMG hierarchy built for n={self.levels[0]['n']}, получена матрица n={A_csr.size(0)}")

        self.levels[0]['A'] = A_csr

        if self._external_coords is not None:
            coords_root = self._external_coords
            if coords_root.dim() == 1:
                coords_root = coords_root.unsqueeze(1)
            if coords_root.size(0) != n_total:
                raise ValueError(
                    f"node_coords size mismatch: expected {n_total}, got {coords_root.size(0)}"
                )
            coords_dev = coords_root.to(device=root_device, dtype=root_dtype)
            self.levels[0]['node_coords'] = coords_dev
        elif 'node_coords' in self.levels[0]:
            del self.levels[0]['node_coords']

        if self._external_nullspace is not None:
            basis_ext = self._external_nullspace
            if basis_ext.dim() == 1:
                basis_ext = basis_ext.unsqueeze(1)
            if basis_ext.size(0) != n_total:
                raise ValueError(
                    f"near_nullspace size mismatch: expected {n_total}, got {basis_ext.size(0)}"
                )
            basis_current = orthonormalize_columns(basis_ext.to(dtype=torch.float64, device='cpu'))
        else:
            basis_current = generate_nullspace(A_csr, self.nullspace_dim)
        self.levels[0]['basis'] = basis_current
        self._update_level_metrics(0)
        self._apply_memory_policy(0)
        basis_current = self.levels[0]['basis']
        A_current = self.levels[0]['A']

        for lvl in range(len(self.levels) - 1):
            P = self.levels[lvl].get('P')
            if P is None:
                self._update_level_metrics(lvl + 1)
                self._apply_memory_policy(lvl + 1)
                continue
            level_device = self.levels[lvl].get('device', root_device)
            level_dtype = self.levels[lvl].get('dtype', root_dtype)
            if A_current.device != level_device or A_current.dtype != level_dtype:
                A_current = self._convert_sparse_tensor(A_current, level_device, level_dtype)
            if basis_current is not None:
                basis_gpu = basis_current.to(device=P.device, dtype=P.dtype)
                basis_coarse = torch.sparse.mm(P.transpose(0, 1), basis_gpu).to('cpu')
                basis_coarse = orthonormalize_columns(basis_coarse)
            else:
                basis_coarse = None
            A_current = rap_torch(A_current, P, level_device, drop_tol=self.rap_drop_tol)
            self.levels[lvl + 1]['A'] = A_current
            self.levels[lvl + 1]['basis'] = basis_coarse
            self._update_level_metrics(lvl + 1)
            self._apply_memory_policy(lvl + 1)
            A_current = self.levels[lvl + 1]['A']
            basis_current = self.levels[lvl + 1]['basis']

        print(f"[ClassicalAMG] Иерархия обновлена без перестроения структуры (diag median={diag_median:.3e})")


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
    device = B_c.device
    dtype = B_c.dtype
    ones_row = torch.ones(1, k, dtype=dtype, device=device)
    A_ls = torch.vstack([B_c, ones_row])
    y = torch.cat([target, torch.tensor([1.0], dtype=dtype, device=device)])
    At = A_ls.T
    ATA = At @ A_ls + ls_reg * torch.eye(k, dtype=dtype, device=device)
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
