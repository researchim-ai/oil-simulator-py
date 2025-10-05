import numpy as np
import torch
from typing import Tuple


def assemble_full_csr(indptr_p: np.ndarray,
                       indices_p: np.ndarray,
                       data_p: np.ndarray,
                       vars_per_cell: int = 2,
                       diag_sat: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Упрощённо расширяет CSR блока давления до полной системы p+S.

    При vars_per_cell=2 порядок переменных (на ячейку): [p, S].
    Мы создаём блок-диагональную матрицу:
        [ A_pp     0 ]
        [   0   diag_sat*I ]
    Этого достаточно, чтобы Chebyshev-tail мог выполнять сглаживание
    и не требовалось хранить полные межфазовые связи.
    """
    if vars_per_cell not in (2, 3):
        raise ValueError("vars_per_cell must be 2 or 3")

    n = len(indptr_p) - 1  # число ячеек
    block = vars_per_cell
    N_full = n * block

    # Давление блок переносим без изменений; места достаточно под диагонали S/T
    indptr_full = np.zeros(N_full + 1, dtype=np.int64)
    nnz_est = indices_p.size + n * (block - 1)
    indices_full = np.empty(nnz_est, dtype=np.int64)
    data_full = np.empty(nnz_est, dtype=np.float64)

    pos = 0
    for cell in range(n):
        # p-строка
        row_p = cell * block
        start_p, end_p = indptr_p[cell], indptr_p[cell + 1]
        indptr_full[row_p] = pos
        for j in range(start_p, end_p):
            col_p = indices_p[j] * block  # p-колонка давления
            indices_full[pos] = col_p
            data_full[pos] = data_p[j]
            pos += 1
        indptr_full[row_p + 1] = pos

        # S-строка: только диагональный элемент (упрощённо)
        row_s = row_p + 1
        indptr_full[row_s] = pos
        indices_full[pos] = row_s
        data_full[pos] = diag_sat
        pos += 1
        indptr_full[row_s + 1] = pos

        if block == 3:
            # T-строка (третья переменная): только диагональный элемент
            row_t = row_p + 2
            indptr_full[row_t] = pos
            indices_full[pos] = row_t
            data_full[pos] = diag_sat
            pos += 1
            indptr_full[row_t + 1] = pos

    # обрезаем массивы
    indices_full = indices_full[:pos]
    data_full = data_full[:pos]

    return indptr_full, indices_full, data_full 