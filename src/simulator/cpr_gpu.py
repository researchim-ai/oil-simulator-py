import torch
import sys
import os

class CPRPreconditionerGPU:
    """
    GPU-версия CPR (Constrained Pressure Residual) предобуславливателя.
    Реализует двухступенчатую схему:
    1. Глобальная коррекция давления через AMG.
    2. Локальное сглаживание (Block Jacobi) для полной системы.
    """
    def __init__(self, J_vals, indptr, indices, n_cells, device):
        self.n_cells = n_cells
        self.device = device
        
        # --- 1. Построение матрицы давления A_p для AMG ---
        # A_p аппроксимируется как сумма уравнений сохранения массы (total compressibility).
        # J_vals: [NNZ, 2, 2]. Индекс [:, 0, 0] - dRo/dP, [:, 1, 0] - dRw/dP.
        # Суммируем производные по давлению.
        data_p = J_vals[:, 0, 0] + J_vals[:, 1, 0]
        
        # Масштабирование для численной устойчивости AMG
        max_val = torch.max(torch.abs(data_p))
        if max_val > 0:
            data_p = data_p / max_val
            self.p_scale = max_val
        else:
            self.p_scale = torch.tensor(1.0, device=device)
            
        # Создаем CSR матрицу давления
        crow = indptr.long()
        col = indices.long()
        
        try:
            self.A_p = torch.sparse_csr_tensor(
                crow, col, data_p, size=(n_cells, n_cells)
            )
        except Exception as e:
            raise RuntimeError(f"CPR: Failed to create A_p tensor: {e}")
        
        # Инициализация AMG
        self.amg = None
        try:
            # Пытаемся импортировать ClassicalAMG
            try:
                from solver.classical_amg import ClassicalAMG
            except ImportError:
                sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                from solver.classical_amg import ClassicalAMG

            self.amg = ClassicalAMG(
                self.A_p, 
                max_levels=15,  # Глубокая иерархия для больших сеток
                theta=0.25,
                mixed_precision=True,
                nullspace_dim=0 
            )
        except Exception as e:
            print(f"CPR Warning: AMG init failed ({e}). Pressure correction disabled.")

        # --- 2. Подготовка сглаживателя (Block Jacobi) ---
        # Нам нужны диагональные блоки D_ii (размер 2x2) для каждой ячейки.
        # J_vals хранит ВСЕ блоки. Нам нужно найти те, где col == row.
        # Это дорогая операция поиска, но indices (col) отсортированы.
        # Оптимизация: Мы знаем, что диагональный элемент обычно первый или последний?
        # Нет, в общем случае он где-то посередине строки.
        
        # Чтобы сделать это быстро на GPU без цикла:
        # Развернем row indices:
        row_indices = torch.repeat_interleave(torch.arange(n_cells, device=device), crow[1:] - crow[:-1])
        
        # Маска диагональных элементов: row == col
        diag_mask = (row_indices == col)
        
        # Извлекаем диагональные блоки: [N, 2, 2]
        # Внимание: мы предполагаем, что каждая строка имеет диагональный элемент.
        # В FIM это всегда так.
        self.diag_blocks = J_vals[diag_mask] 
        
        # Проверка: если диагональных блоков меньше N, значит где-то нет диагонали (баг сборки)
        if self.diag_blocks.shape[0] != n_cells:
            # Fallback: если структура кривая, заполняем Identity
            # print(f"CPR Warning: Found {self.diag_blocks.shape[0]} diag blocks, expected {n_cells}")
            self.inv_diag = None
        else:
            # Инвертируем блоки 2x2 аналитически или через torch.linalg.inv
            # Блоки [N, 2, 2]
            try:
                # Добавляем eps для устойчивости
                # self.inv_diag = torch.linalg.inv(self.diag_blocks + 1e-12 * torch.eye(2, device=device))
                
                # Аналитическая инверсия 2x2 быстрее и надежнее
                a = self.diag_blocks[:, 0, 0]
                b = self.diag_blocks[:, 0, 1]
                c = self.diag_blocks[:, 1, 0]
                d = self.diag_blocks[:, 1, 1]
                
                det = a*d - b*c
                # Избегаем деления на 0
                det = torch.where(torch.abs(det) < 1e-15, torch.ones_like(det)*1e-15, det)
                
                inv_det = 1.0 / det
                
                self.inv_diag = torch.zeros_like(self.diag_blocks)
                self.inv_diag[:, 0, 0] = d * inv_det
                self.inv_diag[:, 0, 1] = -b * inv_det
                self.inv_diag[:, 1, 0] = -c * inv_det
                self.inv_diag[:, 1, 1] = a * inv_det
                
            except Exception as e:
                print(f"CPR Warning: Diag inversion failed ({e})")
                self.inv_diag = None
        
        # Сохраняем ссылки для matvec полной системы
        self.J_vals = J_vals
        self.crow = crow
        self.col = col
        
    def apply_A(self, x):
        """
        Вычисляет y = A * x для полной блочной матрицы на GPU.
        x: [2*N]
        return: y [2*N]
        """
        x_blk = x.view(-1, 2) # [N, 2]
        
        # Gather x values corresponding to column indices
        # indices (col) длина NNZ
        x_j = x_blk[self.col] # [NNZ, 2]
        
        # Блочное умножение: J_vals [NNZ, 2, 2] @ x_j [NNZ, 2, 1] -> [NNZ, 2, 1]
        # bmm требует [B, N, M] @ [B, M, P]
        y_vals = torch.bmm(self.J_vals, x_j.unsqueeze(-1)).squeeze(-1) # [NNZ, 2]
        
        # Scatter add results to rows
        y = torch.zeros_like(x_blk)
        y.index_add_(0, torch.repeat_interleave(torch.arange(self.n_cells, device=self.device), 
                                              self.crow[1:] - self.crow[:-1]), y_vals)
        
        return y.view(-1)

    def matvec(self, r):
        """
        Применяет CPR предобуславливатель: x = M^{-1} r
        """
        # r: [2*N]
        x = torch.zeros_like(r)
        
        # --- Этап 1: Коррекция Давления (AMG) ---
        if self.amg:
            # Проекция невязки на давление (сумма уравнений для простоты или первая компонента)
            # r_p = r_oil + r_water
            r_p = r[0::2] + r[1::2]
            
            # Масштабирование
            r_p_scaled = r_p / self.p_scale
            
            # Решение AMG (1 V-cycle)
            delta_p = self.amg.solve(r_p_scaled, tol=0.1, max_iter=1)
            
            # Обновление полного вектора (только давление меняется)
            x[0::2] = delta_p
            # Насыщенность пока не меняем x[1::2] = 0
            
            # --- Этап 2: Сглаживание (Smoothing) ---
            # Нам нужно вычислить НОВУЮ невязку относительно найденного давления:
            # r_new = r - A * x
            # Это требует дорогого умножения матрицы.
            # True CPR делает это.
            
            Ax = self.apply_A(x)
            r_new = r - Ax
        else:
            r_new = r
            
        # --- Этап 3: Применение Block Jacobi к остаточной невязке ---
        # delta_x = omega * D^{-1} * r_new
        # Damping factor 0.85 helps stability (reduces oscillations)
        omega = 0.85
        
        if self.inv_diag is not None:
            r_blk = r_new.view(-1, 2) # [N, 2]
            
            # [N, 2, 2] @ [N, 2, 1] -> [N, 2]
            delta_x_blk = torch.bmm(self.inv_diag, r_blk.unsqueeze(-1)).squeeze(-1)
            
            # Добавляем коррекцию c демпфированием
            x = x + omega * delta_x_blk.view(-1)
            
        return x
