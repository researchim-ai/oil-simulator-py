"""
Classical Algebraic Multigrid (Ruge-Stuben) solver.

TRUE RS-AMG с:
- Strong connections по отрицательным внедиагоналям с диагональным масштабированием
- MIS coarsening по графу сильных связей
- 1-point interpolation к сильным C-соседям
- Эффективный RAP через scatter (без SpGEMM/плотнения!)
"""

import torch
from typing import Tuple, Optional


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


def build_prolongation(A_csr: torch.Tensor, cf_marker: torch.Tensor,
                       strong_mask: torch.Tensor, n_coarse: int, 
                       deterministic: bool = False, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """1-point interpolation: F привязывается к сильному C-соседу.
    
    Для каждого F-узла берём сильного C-соседа с максимальным весом -a_ij.
    Это дёшево и даёт алгебраическую привязку.
    
    Args:
        A_csr: Sparse CSR матрица
        cf_marker: 0=C, 1=F
        strong_mask: Boolean mask сильных связей
        n_coarse: Число C-points
        deterministic: Если True, используем медленный но детерминированный выбор родителя
        normalize: Если True, нормируем P: w_i = 1/sqrt(child_count) для энергетики
    
    Returns:
        P: Prolongation оператор (CSR), размер (n_fine, n_coarse_actual)
        parent_idx: parent_idx[i] = coarse-родитель узла i
        weights: weights[i] = 1/sqrt(child_count) для RAP (или None если normalize=False)
        n_coarse_actual: Фактическое число coarse-узлов (может быть больше если были orphans)
    """
    crow, col, vals = A_csr.crow_indices(), A_csr.col_indices(), A_csr.values()
    n = crow.numel() - 1
    device = A_csr.device
    
    # Mapping fine->coarse
    c_idx = (cf_marker == 0).nonzero(as_tuple=True)[0]
    fine2coarse = torch.full((n,), -1, dtype=torch.long, device=device)
    fine2coarse[c_idx] = torch.arange(n_coarse, device=device)
    
    # Row indices
    row_len = crow[1:] - crow[:-1]
    row = torch.repeat_interleave(torch.arange(n, device=device), row_len)
    
    # Кандидаты: сильные C-соседи с отрицательными a_ij
    cand = strong_mask & (cf_marker[col] == 0)
    w = (-vals).where(cand, torch.tensor(0., device=device, dtype=vals.dtype))
    
    parent = torch.full((n,), -1, dtype=torch.long, device=device)
    
    if deterministic:
        # ДЕТЕРМИНИРОВАННЫЙ выбор ОДНОГО родителя (argmax по строке)
        # Медленнее, но гарантирует ровно 1 родителя на строку
        for i in range(n):
            s, e = int(crow[i].item()), int(crow[i+1].item())
            if s == e:
                continue
            cols_i = col[s:e]
            w_i = w[s:e]
            cand_i = cand[s:e]
            if cand_i.any():
                k = torch.argmax(w_i * cand_i.to(w_i.dtype))
                parent[i] = fine2coarse[cols_i[int(k)]]
    else:
        # ВЕКТОРИЗОВАННЫЙ выбор (быстро, но недетерминированный при tie-break)
        best_w = torch.zeros(n, device=device, dtype=vals.dtype)
        best_w.scatter_reduce_(0, row, w, reduce='amax', include_self=False)
        winners = cand & (w >= best_w[row] - 1e-30)
        parent[row[winners]] = fine2coarse[col[winners]]
    
    # C-узлы указывают на себя
    parent[c_idx] = fine2coarse[c_idx]
    
    # Safety: orphan F -> поднять в C
    orphan = (parent < 0)
    if orphan.any():
        extra = orphan.nonzero(as_tuple=True)[0]
        add = torch.arange(n_coarse, n_coarse + extra.numel(), device=device)
        fine2coarse[extra] = add
        parent[extra] = add
        n_coarse = int(n_coarse + extra.numel())
    
    # Нормировка P (опционально)
    if normalize:
        # w_i = 1 / sqrt(child_count) для энергетической устойчивости
        # Улучшает сходимость на гетерогенных задачах
        child_cnt = torch.bincount(parent, minlength=n_coarse)
        weights = 1.0 / torch.sqrt(child_cnt[parent].to(torch.float64).clamp_min(1.0))
        valsP = weights
    else:
        # Единичная интерполяция (по умолчанию)
        # Проще, не уменьшает диагональ A_coarse
        weights = None
        valsP = torch.ones(n, device=device, dtype=torch.float64)
    
    # P: интерполяция (нормированная или единичная)
    rows = torch.arange(n, device=device)
    cols = parent
    P = torch.sparse_coo_tensor(torch.stack([rows, cols]), valsP,
                                size=(n, n_coarse), device=device, dtype=torch.float64).coalesce()
    
    return P.to_sparse_csr(), parent, weights, n_coarse


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
        # PyTorch CSR bug на больших матрицах - используем SciPy
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


class ClassicalAMG:
    """Classical Ruge-Stuben Algebraic Multigrid.
    
    TRUE RS-AMG с:
    - Algebraic coarsening через MIS по графу сильных связей
    - 1-point interpolation к сильным C-соседям
    - Эффективный RAP через scatter (O(nnz), полностью на GPU!)
    - Damped Jacobi smoother
    """
    
    def __init__(self, A_csr_np: torch.Tensor, theta: float = 0.25,
                 max_levels: int = 10, coarsest_size: int = 100):
        """Инициализация AMG иерархии.
        
        Args:
            A_csr_np: Numpy CSR матрица (будет сконвертирована в torch CSR на GPU)
            theta: Порог для сильных связей (0.25 = классический RS)
            max_levels: Максимум уровней
            coarsest_size: Минимальный размер coarsest level
        """
        self.theta = theta
        self.max_levels = max_levels
        self.coarsest_size = coarsest_size
        
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
        
        # ═══════════════════════════════════════════════════════════════
        # EQUILIBRATION: Симметричное диагональное масштабирование
        # ═══════════════════════════════════════════════════════════════
        # Преобразуем A → D^(-1/2) A D^(-1/2), где D = diag(A)
        # Результат: diag ≈ 1, что критично для устойчивости Jacobi
        # ═══════════════════════════════════════════════════════════════
        diag_orig = self._extract_diag(A_csr).abs().clamp_min(1e-30)
        Dhalf_inv = 1.0 / torch.sqrt(diag_orig)  # D^(-1/2)
        
        # Применяем масштабирование: vals[i,j] *= Dhalf_inv[i] * Dhalf_inv[j]
        crow = A_csr.crow_indices()
        col = A_csr.col_indices()
        vals = A_csr.values().clone()  # clone для изменения
        
        row_len = crow[1:] - crow[:-1]
        row_idx = torch.repeat_interleave(torch.arange(crow.numel()-1, device=device), row_len)
        vals = vals * Dhalf_inv[row_idx] * Dhalf_inv[col]
        
        A_csr = torch.sparse_csr_tensor(crow, col, vals, size=A_csr.size(), 
                                        device=device, dtype=torch.float64)
        
        # Сохраняем для обратного масштабирования решения
        self.Dhalf_inv = Dhalf_inv
        
        diag_scaled = self._extract_diag(A_csr).abs()
        print(f"[ClassicalAMG] Equilibration: diag {diag_orig.min():.2e}..{diag_orig.max():.2e} → {diag_scaled.min():.2e}..{diag_scaled.max():.2e}")
        
        # Строим иерархию
        A_current = A_csr
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
                    'is_coarsest': True
                })
                print(f"[ClassicalAMG] L{lvl}: n={n} ≤ {coarsest_size}, coarsest level")
                break
            
            # C/F splitting
            strong_mask = find_strong_connections(A_current, theta)
            cf_marker, n_coarse = classical_coarsening(A_current, theta)
            
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
                    'diag': diag_abs,
                    'inv_relax': inv_relax,
                    'is_coarsest': True
                })
                print(f"[ClassicalAMG] L{lvl}: n={n}, coarsening failed (ratio={n/n_coarse:.1f}x), stopping")
                break
            
            # Prolongation (1-point interpolation)
            # normalize=False по умолчанию: проще, не раздувает решение при плохо обусловленных матрицах
            # deterministic=False: быстрый векторизованный выбор родителя
            P, parent_idx, weights, n_coarse_actual = build_prolongation(
                A_current, cf_marker, strong_mask, n_coarse, 
                deterministic=False, normalize=False
            )
            
            # Galerkin RAP через эффективный scatter (без SpGEMM!)
            A_coarse = rap_onepoint_gpu(A_current, parent_idx, weights=weights, n_coarse=n_coarse_actual)
            
            ratio = n / n_coarse_actual
            c_pct = 100.0 * n_coarse_actual / n
            orphan_count = n_coarse_actual - n_coarse
            if orphan_count > 0:
                print(f"[ClassicalAMG] L{lvl}: n={n} → n_c={n_coarse_actual} (ratio={ratio:.1f}x), C-points={n_coarse}+{orphan_count} orphans/{n} ({c_pct:.1f}%)")
            else:
                print(f"[ClassicalAMG] L{lvl}: n={n} → n_c={n_coarse_actual} (ratio={ratio:.1f}x), C-points={n_coarse_actual}/{n} ({c_pct:.1f}%)")
            
            diag_abs = self._extract_diag(A_current).abs()
            row_abs = self._row_abs_sum(A_current)
            beta = 0.3
            denom = torch.maximum(diag_abs, beta * row_abs).clamp_min(1e-30)
            inv_relax = (1.0 / denom).clamp(max=1e2)
            
            self.levels.append({
                'A': A_current,
                'n': n,
                'P': P,
                'diag': diag_abs,
                'inv_relax': inv_relax,
                'is_coarsest': False
            })
            
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
                 pre_smooth: int = 1, post_smooth: int = 1, debug: bool = False, cycle_num: int = 0) -> torch.Tensor:
        """V-cycle"""
        level = self.levels[lvl]
        
        if debug:
            print(f"  [V-CYCLE L{lvl}] ВХОД: ||x||={x.norm():.3e}, ||b||={b.norm():.3e}, n={level['n']}")
        
        if level['is_coarsest']:
            # Прямое решение на coarsest уровне
            n = level['n']
            
            if n <= 500:
                # Точное решение через плотную алгебру (быстро для малых систем)
                A_dense = level['A'].to_dense()
                try:
                    x = torch.linalg.solve(A_dense, b)
                    if debug:
                        print(f"  [COARSEST L{lvl}] ТОЧНОЕ решение (n={n}), ||x||={x.norm():.3e}")
                except RuntimeError:
                    # Если система плохо обусловлена, используем lstsq
                    x = torch.linalg.lstsq(A_dense, b.unsqueeze(1)).solution.squeeze(1)
                    if debug:
                        print(f"  [COARSEST L{lvl}] LSTSQ решение (n={n}), ||x||={x.norm():.3e}")
            else:
                # Для больших систем: L1-Jacobi iterations
                inv_relax = level['inv_relax']
                omega = 0.7
                if debug:
                    print(f"  [COARSEST L{lvl}] L1-Jacobi (n={n})")
                for it in range(50):
                    r = b - self._matvec(lvl, x)
                    delta = omega * inv_relax * r
                    x = x + delta
                    if debug and it < 2:
                        print(f"  [COARSEST L{lvl} iter{it+1}] ||δ||={delta.norm():.3e}, ||x||={x.norm():.3e}")
            
            if debug:
                r_final = b - self._matvec(lvl, x)
                print(f"  [COARSEST L{lvl}] ВЫХОД: ||x||={x.norm():.3e}, ||r||={r_final.norm():.3e}")
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
        
        # Масштабируем RHS: b̃ = D^(-1/2) b
        b_scaled = self.Dhalf_inv * b
        
        # Начальное приближение
        if x0 is None:
            x_scaled = torch.zeros_like(b_scaled)
        else:
            # Масштабируем начальное приближение: x̃ = D^(1/2) x
            x_scaled = x0.to(self.device) / self.Dhalf_inv
        
        b_norm = b_scaled.norm()
        
        for cycle in range(max_iter):
            # Диагностика ТОЛЬКО для первого цикла
            debug = (cycle == 0)
            x_scaled = self._v_cycle(0, x_scaled, b_scaled, pre_smooth=1, post_smooth=1, 
                                    debug=debug, cycle_num=cycle+1)
            r = b_scaled - self._matvec(0, x_scaled)
            rel_res = r.norm() / (b_norm + 1e-30)
            x_ratio = x_scaled.norm() / (b_norm + 1e-30)
            print(f"  [ClassicalAMG cycle {cycle+1}] rel_res={rel_res:.3e}, ||x||/||b||={x_ratio:.2e}")
            if rel_res < tol:
                break
        
        # Обратное масштабирование: x = D^(-1/2) x̃
        x = self.Dhalf_inv * x_scaled
        
        return x
