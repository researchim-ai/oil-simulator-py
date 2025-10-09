"""Classical Algebraic Multigrid (Ruge-Stuben) с адаптивным coarsening.

Это фундаментальное решение проблемы low-energy мод от wells:
- Geometric coarsening (2x2x2) не может захватить algebraic smooth modes от wells
- Algebraic coarsening строит уровни на основе strong connections в матрице
- Это даёт эффективное гашение всех мод, включая от wells!
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


def find_strong_connections(A_csr: torch.Tensor, theta: float = 0.25) -> torch.Tensor:
    """Находит strong connections в матрице.
    
    Connection i→j сильная если |a_ij| >= theta * max_k≠i(|a_ik|)
    
    Args:
        A_csr: Sparse CSR матрица
        theta: Порог для strong connections (0.25 стандартный)
    
    Returns:
        Strong connections mask (bool tensor)
    """
    crow = A_csr.crow_indices()
    col = A_csr.col_indices()
    vals = A_csr.values()
    n = crow.numel() - 1
    
    # Для каждой строки найдём max off-diagonal
    strong_mask = torch.zeros_like(vals, dtype=torch.bool)
    
    for i in range(n):
        row_start = crow[i]
        row_end = crow[i+1]
        
        if row_end <= row_start:
            continue
        
        # Найти max off-diagonal в строке
        row_vals = vals[row_start:row_end].abs()
        row_cols = col[row_start:row_end]
        
        # Исключить диагональ
        off_diag_mask = row_cols != i
        if off_diag_mask.any():
            max_off_diag = row_vals[off_diag_mask].max()
            threshold = theta * max_off_diag
            
            # Пометить strong connections
            strong_mask[row_start:row_end] = row_vals >= threshold
            # Диагональ не считаем strong connection
            diag_pos = (row_cols == i).nonzero(as_tuple=True)[0]
            if diag_pos.numel() > 0:
                strong_mask[row_start + diag_pos[0]] = False
    
    return strong_mask


def classical_coarsening(A_csr: torch.Tensor, theta: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classical Ruge-Stuben C/F splitting.
    
    Returns:
        cf_marker: 0=C-point (coarse), 1=F-point (fine)
        n_coarse: количество C-points
    """
    crow = A_csr.crow_indices()
    col = A_csr.col_indices()
    vals = A_csr.values()
    n = crow.numel() - 1
    
    # Найти strong connections
    strong_mask = find_strong_connections(A_csr, theta)
    
    # Построить граф strong connections
    # lambda_i = количество точек, сильно зависящих от i
    lambda_i = torch.zeros(n, dtype=torch.long, device=A_csr.device)
    
    for i in range(n):
        row_start = crow[i]
        row_end = crow[i+1]
        strong_in_row = strong_mask[row_start:row_end]
        lambda_i[i] = strong_in_row.sum()
    
    # Также нужно посчитать сколько точек для которых i - strong neighbor
    lambda_transpose = torch.zeros(n, dtype=torch.long, device=A_csr.device)
    for i in range(n):
        row_start = crow[i]
        row_end = crow[i+1]
        for pos in range(row_start, row_end):
            if strong_mask[pos]:
                j = col[pos]
                lambda_transpose[j] += 1
    
    # C/F splitting: greedy algorithm
    # -1 = undecided, 0 = C-point, 1 = F-point
    cf_marker = torch.full((n,), -1, dtype=torch.long, device=A_csr.device)
    
    # Priority: точки с максимальным lambda_transpose (нужны многим)
    remaining = torch.arange(n, device=A_csr.device)
    
    while (cf_marker == -1).any():
        undecided = (cf_marker == -1).nonzero(as_tuple=True)[0]
        if undecided.numel() == 0:
            break
        
        # Выбираем точку с максимальным lambda_transpose среди undecided
        lambda_undecided = lambda_transpose[undecided]
        max_idx = lambda_undecided.argmax()
        i = undecided[max_idx]
        
        # i становится C-point
        cf_marker[i] = 0
        
        # Все strong neighbors i становятся F-points
        row_start = crow[i]
        row_end = crow[i+1]
        for pos in range(row_start, row_end):
            if strong_mask[pos]:
                j = col[pos]
                if cf_marker[j] == -1:
                    cf_marker[j] = 1
        
        # Пересчитываем lambda_transpose для оставшихся
        # (упрощенно: не пересчитываем, просто обнуляем выбранные)
        lambda_transpose[i] = -1
        for pos in range(row_start, row_end):
            if strong_mask[pos]:
                j = col[pos]
                lambda_transpose[j] = -1
    
    # Все оставшиеся undecided становятся C-points
    cf_marker[cf_marker == -1] = 0
    
    n_coarse = (cf_marker == 0).sum().item()
    
    return cf_marker, n_coarse


def build_prolongation(A_csr: torch.Tensor, cf_marker: torch.Tensor, 
                       strong_mask: torch.Tensor, n_coarse: int) -> torch.Tensor:
    """Строит prolongation оператор P: coarse → fine.
    
    Classical interpolation:
    - C-points: P_ii = 1 (identity)
    - F-points: P_ij = -a_ij / (sum_k∈C_i a_ik) где C_i - strong C-neighbors
    """
    crow = A_csr.crow_indices()
    col = A_csr.col_indices()
    vals = A_csr.values()
    n_fine = cf_marker.numel()
    
    # Создаём mapping fine→coarse
    c_indices = (cf_marker == 0).nonzero(as_tuple=True)[0]
    fine_to_coarse = torch.full((n_fine,), -1, dtype=torch.long, device=A_csr.device)
    fine_to_coarse[c_indices] = torch.arange(n_coarse, device=A_csr.device)
    
    # Строим P в COO формате
    P_rows = []
    P_cols = []
    P_vals = []
    
    for i in range(n_fine):
        if cf_marker[i] == 0:
            # C-point: P_ii = 1
            P_rows.append(i)
            P_cols.append(fine_to_coarse[i])
            P_vals.append(1.0)
        else:
            # F-point: интерполируем от strong C-neighbors
            row_start = crow[i]
            row_end = crow[i+1]
            
            # Найдём strong C-neighbors и их веса
            strong_C_sum = 0.0
            strong_C_neighbors = []
            
            for pos in range(row_start, row_end):
                j = col[pos]
                if j != i and strong_mask[pos] and cf_marker[j] == 0:
                    a_ij = vals[pos].item()
                    strong_C_neighbors.append((j, a_ij))
                    strong_C_sum += a_ij
            
            if len(strong_C_neighbors) == 0:
                # Нет strong C-neighbors → делаем этуточку C-point
                cf_marker[i] = 0
                c_idx = fine_to_coarse.max() + 1
                fine_to_coarse[i] = c_idx
                P_rows.append(i)
                P_cols.append(c_idx.item())
                P_vals.append(1.0)
            else:
                # Классическая интерполяция
                for j, a_ij in strong_C_neighbors:
                    c_j = fine_to_coarse[j]
                    if c_j >= 0:
                        P_rows.append(i)
                        P_cols.append(c_j.item())
                        P_vals.append(-a_ij / (strong_C_sum + 1e-30))
    
    # Конвертируем в sparse COO
    P_indices = torch.tensor([P_rows, P_cols], dtype=torch.long, device=A_csr.device)
    P_values = torch.tensor(P_vals, dtype=torch.float64, device=A_csr.device)
    
    P_coo = torch.sparse_coo_tensor(
        P_indices, P_values,
        size=(n_fine, n_coarse),
        device=A_csr.device, dtype=torch.float64
    ).coalesce()
    
    return P_coo.to_sparse_csr()


def build_galerkin_coarse(A_csr: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Galerkin RAP: A_c = P^T·A·P."""
    # P^T
    PT = P.transpose(0, 1)
    
    # A·P
    AP = torch.sparse.mm(A_csr, P.to_sparse_coo().to_dense().unsqueeze(-1) if P.is_sparse else P)
    
    # P^T·(A·P)
    if AP.dim() == 1:
        AP = AP.unsqueeze(1)
    A_coarse_dense = torch.sparse.mm(PT, AP)
    
    # Конвертируем обратно в sparse CSR
    A_coarse_coo = A_coarse_dense.to_sparse_coo()
    A_coarse_csr = A_coarse_coo.to_sparse_csr()
    
    return A_coarse_csr


class ClassicalAMG:
    """Classical Algebraic Multigrid (Ruge-Stuben).
    
    Фундаментальное решение для wells:
    - Algebraic coarsening на основе strong connections
    - Адаптируется к структуре матрицы (wells автоматически учитываются)
    - Эффективно гасит все моды, включая low-energy от wells!
    """
    
    def __init__(self, A_csr: torch.Tensor, max_coarse: int = 100, 
                 theta: float = 0.25, max_levels: int = 10):
        """
        Args:
            A_csr: Fine матрица (sparse CSR)
            max_coarse: Минимальный размер coarse grid
            theta: Strong connection threshold
            max_levels: Максимум уровней
        """
        self.device = A_csr.device
        self.dtype = A_csr.dtype
        self.theta = theta
        self.max_levels = max_levels
        
        print(f"[ClassicalAMG] Строим algebraic AMG с theta={theta:.2f}...")
        
        # Строим иерархию
        self.levels = []
        A_current = A_csr
        
        for level in range(max_levels):
            n_current = A_current.size(0)
            
            if n_current <= max_coarse:
                print(f"[ClassicalAMG] L{level}: n={n_current} ≤ {max_coarse}, coarsest level")
                self.levels.append({'A': A_current, 'P': None, 'R': None})
                break
            
            # C/F splitting
            cf_marker, n_coarse = classical_coarsening(A_current, theta)
            
            if n_coarse == 0 or n_coarse >= n_current * 0.8:
                print(f"[ClassicalAMG] L{level}: n={n_current}, не удалось coarsen, останавливаем")
                self.levels.append({'A': A_current, 'P': None, 'R': None})
                break
            
            # Строим prolongation
            strong_mask = find_strong_connections(A_current, theta)
            P = build_prolongation(A_current, cf_marker, strong_mask, n_coarse)
            R = P.transpose(0, 1)  # Galerkin restriction
            
            # RAP
            A_coarse = build_galerkin_coarse(A_current, P)
            
            ratio = n_current / n_coarse
            print(f"[ClassicalAMG] L{level}: n={n_current} → n_c={n_coarse} (ratio={ratio:.1f}x), "
                  f"C-points={n_coarse}/{n_current} ({100*n_coarse/n_current:.1f}%)")
            
            self.levels.append({'A': A_current, 'P': P, 'R': R})
            A_current = A_coarse
        
        print(f"✅ ClassicalAMG: построено {len(self.levels)} уровней")
    
    def solve(self, b: torch.Tensor, x0: torch.Tensor = None, 
              tol: float = 1e-6, max_iter: int = 1, 
              pre_smooth: int = 2, post_smooth: int = 2) -> torch.Tensor:
        """V-cycle solve."""
        # Конвертируем в torch tensor если нужно
        if not isinstance(b, torch.Tensor):
            b = torch.from_numpy(b).to(self.device).to(self.dtype)
        if x0 is not None and not isinstance(x0, torch.Tensor):
            x0 = torch.from_numpy(x0).to(self.device).to(self.dtype)
        
        n = b.numel()
        x = x0.clone() if x0 is not None else torch.zeros_like(b)
        
        for cycle in range(max_iter):
            x = self._v_cycle(0, x, b, pre_smooth, post_smooth)
            
            # Проверка residual
            if cycle == max_iter - 1:
                r = b - self._apply_A(0, x)
                rel_res = r.norm() / (b.norm() + 1e-30)
                print(f"  [ClassicalAMG cycle {cycle+1}] rel_res={rel_res:.3e}")
        
        return x
    
    def _apply_A(self, level_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Применить матрицу к вектору."""
        A = self.levels[level_idx]['A']
        return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    
    def _smooth(self, level_idx: int, x: torch.Tensor, b: torch.Tensor, 
                iters: int = 2, omega: float = 0.67) -> torch.Tensor:
        """Damped Jacobi smoother."""
        A = self.levels[level_idx]['A']
        diag = self._extract_diagonal(A)
        inv_diag = 1.0 / (diag + 1e-30)
        
        for _ in range(iters):
            r = b - self._apply_A(level_idx, x)
            x = x + omega * inv_diag * r
        
        return x
    
    def _extract_diagonal(self, A_csr: torch.Tensor) -> torch.Tensor:
        """Извлечь диагональ из CSR матрицы."""
        crow = A_csr.crow_indices()
        col = A_csr.col_indices()
        vals = A_csr.values()
        n = crow.numel() - 1
        
        diag = torch.zeros(n, device=A_csr.device, dtype=A_csr.dtype)
        
        for i in range(n):
            for pos in range(crow[i], crow[i+1]):
                if col[pos] == i:
                    diag[i] = vals[pos]
                    break
        
        return diag
    
    def _v_cycle(self, level_idx: int, x: torch.Tensor, b: torch.Tensor,
                 pre_smooth: int, post_smooth: int) -> torch.Tensor:
        """V-cycle рекурсивно."""
        # Coarsest level
        if level_idx == len(self.levels) - 1 or self.levels[level_idx]['P'] is None:
            # Direct solve (Jacobi)
            return self._smooth(level_idx, x, b, iters=10, omega=0.67)
        
        # Pre-smoothing
        x = self._smooth(level_idx, x, b, iters=pre_smooth)
        
        # Residual
        r = b - self._apply_A(level_idx, x)
        
        # Restrict
        R = self.levels[level_idx]['R']
        r_coarse = torch.sparse.mm(R, r.unsqueeze(1)).squeeze(1)
        
        # Coarse solve
        e_coarse = torch.zeros(r_coarse.numel(), device=self.device, dtype=self.dtype)
        e_coarse = self._v_cycle(level_idx + 1, e_coarse, r_coarse, pre_smooth, post_smooth)
        
        # Prolong and correct
        P = self.levels[level_idx]['P']
        e_fine = torch.sparse.mm(P, e_coarse.unsqueeze(1)).squeeze(1)
        x = x + e_fine
        
        # Post-smoothing
        x = self._smooth(level_idx, x, b, iters=post_smooth)
        
        return x
