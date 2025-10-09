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
    row_len = crow[1:] - crow[:-1]
    row_idx = torch.repeat_interleave(torch.arange(n, device=device), row_len)
    i = row_idx[strong]
    j = col[strong]
    ii = torch.cat([i, j])
    jj = torch.cat([j, i])
    
    # Степень (число сильных соседей)
    one = torch.ones_like(ii, dtype=torch.float32)
    deg = torch.zeros(n, device=device, dtype=torch.float32)
    deg.scatter_add_(0, ii, one)
    
    # Случайный tie-breaker
    w = torch.rand(n, device=device)
    score = deg + 1e-3 * w  # лексикографическое сравнение
    
    # Максимум score среди соседей
    neigh_max = torch.zeros(n, device=device, dtype=score.dtype)
    neigh_max.scatter_reduce_(0, ii, score[jj], reduce='amax', include_self=False)
    
    # Местные максимумы -> C
    C = score > neigh_max
    cf_marker = torch.where(C, torch.tensor(0, device=device), torch.tensor(1, device=device))
    
    # Гарантия покрытия: F без C-соседей -> поднять в C
    has_C = torch.zeros(n, device=device, dtype=torch.int8)
    has_C.scatter_add_(0, ii, C[jj].to(torch.int8))
    lift = (cf_marker == 1) & (has_C == 0)
    cf_marker[lift] = 0
    
    n_coarse = int((cf_marker == 0).sum().item())
    
    return cf_marker, n_coarse


def build_prolongation(A_csr: torch.Tensor, cf_marker: torch.Tensor,
                       strong_mask: torch.Tensor, n_coarse: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """1-point interpolation: F привязывается к сильному C-соседу.
    
    Для каждого F-узла берём сильного C-соседа с максимальным весом -a_ij.
    Это дёшево и даёт алгебраическую привязку.
    
    Args:
        A_csr: Sparse CSR матрица
        cf_marker: 0=C, 1=F
        strong_mask: Boolean mask сильных связей
        n_coarse: Число C-points
    
    Returns:
        P: Prolongation оператор (CSR), размер (n_fine, n_coarse)
        parent_idx: parent_idx[i] = coarse-родитель узла i
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
    
    # Лучший сосед по весу на каждую строку
    best_w = torch.zeros(n, device=device, dtype=vals.dtype)
    best_w.scatter_reduce_(0, row, w, reduce='amax', include_self=False)
    winners = cand & (w >= best_w[row] - 1e-30)
    
    # Выбранный coarse-столбец для каждой fine-строки
    parent = torch.full((n,), -1, dtype=torch.long, device=device)
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
    
    # P: одна 1 на строку
    rows = torch.arange(n, device=device)
    cols = parent
    valsP = torch.ones(n, device=device, dtype=torch.float64)
    P = torch.sparse_coo_tensor(torch.stack([rows, cols]), valsP,
                                size=(n, n_coarse), device=device, dtype=torch.float64).coalesce()
    
    return P.to_sparse_csr(), parent


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
        print(f"[WARNING] rap_onepoint_gpu: некорректный CSR после coalesce, пересоздаем")
        # Пересоздаем через dense (медленно, но надёжно для маленьких матриц)
        if n_coarse <= 5000:
            Ac_dense = Ac_coo.to_dense()
            Ac = Ac_dense.to_sparse_coo().to_sparse_csr()
        else:
            raise RuntimeError(f"Некорректный CSR формат после RAP на n={n_coarse}")
    
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
        
        # Строим иерархию
        A_current = A_csr
        for lvl in range(max_levels):
            n = A_current.size(0)
            
            if n <= coarsest_size:
                # Coarsest level: прямое решение
                self.levels.append({
                    'A': A_current,
                    'n': n,
                    'P': None,
                    'diag': self._extract_diag(A_current),
                    'is_coarsest': True
                })
                print(f"[ClassicalAMG] L{lvl}: n={n} ≤ {coarsest_size}, coarsest level")
                break
            
            # C/F splitting
            strong_mask = find_strong_connections(A_current, theta)
            cf_marker, n_coarse = classical_coarsening(A_current, theta)
            
            if n_coarse >= n * 0.9:
                # Слишком мало coarsening
                self.levels.append({
                    'A': A_current,
                    'n': n,
                    'P': None,
                    'diag': self._extract_diag(A_current),
                    'is_coarsest': True
                })
                print(f"[ClassicalAMG] L{lvl}: n={n}, coarsening failed (ratio={n/n_coarse:.1f}x), stopping")
                break
            
            # Prolongation (1-point interpolation)
            P, parent_idx = build_prolongation(A_current, cf_marker, strong_mask, n_coarse)
            
            # Galerkin RAP через эффективный scatter (без SpGEMM!)
            A_coarse = rap_onepoint_gpu(A_current, parent_idx, n_coarse=n_coarse)
            
            ratio = n / n_coarse
            c_pct = 100.0 * n_coarse / n
            print(f"[ClassicalAMG] L{lvl}: n={n} → n_c={n_coarse} (ratio={ratio:.1f}x), C-points={n_coarse}/{n} ({c_pct:.1f}%)")
            
            self.levels.append({
                'A': A_current,
                'n': n,
                'P': P,
                'diag': self._extract_diag(A_current),
                'is_coarsest': False
            })
            
            A_current = A_coarse
        
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
    
    def _matvec(self, lvl: int, x: torch.Tensor) -> torch.Tensor:
        """Matrix-vector product: y = A_lvl × x"""
        A = self.levels[lvl]['A']
        # torch.sparse.mm требует 2D
        y = torch.sparse.mm(A, x.view(-1, 1)).squeeze(1)
        return y
    
    def _smooth(self, lvl: int, x: torch.Tensor, b: torch.Tensor, nu: int = 1) -> torch.Tensor:
        """Damped Jacobi smoother"""
        omega = 0.67
        diag_inv = 1.0 / (self.levels[lvl]['diag'] + 1e-30)
        
        for _ in range(nu):
            r = b - self._matvec(lvl, x)
            x = x + omega * diag_inv * r
        
        return x
    
    def _v_cycle(self, lvl: int, x: torch.Tensor, b: torch.Tensor,
                 pre_smooth: int = 1, post_smooth: int = 1) -> torch.Tensor:
        """V-cycle"""
        level = self.levels[lvl]
        
        if level['is_coarsest']:
            # Прямое решение: x = D^{-1} b (несколько итераций Jacobi)
            diag_inv = 1.0 / (level['diag'] + 1e-30)
            for _ in range(10):
                r = b - self._matvec(lvl, x)
                x = x + diag_inv * r
            return x
        
        # Pre-smoothing
        x = self._smooth(lvl, x, b, pre_smooth)
        
        # Residual
        r = b - self._matvec(lvl, x)
        
        # Restrict
        P = level['P']
        PT = P.transpose(0, 1)
        r_c = torch.sparse.mm(PT, r.view(-1, 1)).squeeze(1)
        
        # Coarse solve
        e_c = torch.zeros_like(r_c)
        e_c = self._v_cycle(lvl + 1, e_c, r_c, pre_smooth, post_smooth)
        
        # Prolongate and correct
        e_f = torch.sparse.mm(P, e_c.view(-1, 1)).squeeze(1)
        x = x + e_f
        
        # Post-smoothing
        x = self._smooth(lvl, x, b, post_smooth)
        
        return x
    
    def solve(self, b: torch.Tensor, x0: Optional[torch.Tensor] = None,
              tol: float = 1e-6, max_iter: int = 10) -> torch.Tensor:
        """Решает A·x = b через V-cycles.
        
        Args:
            b: RHS вектор
            x0: Начальное приближение (если None, используется 0)
            tol: Относительная tolerance для ||r||/||b||
            max_iter: Максимум V-cycles
        
        Returns:
            x: Решение
        """
        b = b.to(self.device)
        
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.to(self.device)
        
        b_norm = b.norm()
        
        for cycle in range(max_iter):
            x = self._v_cycle(0, x, b, pre_smooth=1, post_smooth=1)
            r = b - self._matvec(0, x)
            rel_res = r.norm() / (b_norm + 1e-30)
            print(f"  [ClassicalAMG cycle {cycle+1}] rel_res={rel_res:.3e}")
            if rel_res < tol:
                break
        
        return x
