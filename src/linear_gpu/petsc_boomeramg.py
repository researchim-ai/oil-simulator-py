import numpy as np
from petsc4py import PETSc
import sys


def solve_boomeramg(indptr, indices, data, b, tol=1e-8, max_iter=1000, atol=1e-50):
    """Решает CSR-систему Ax=b с предобуславливателем BoomerAMG (Hypre) через PETSc.

    Parameters
    ----------
    indptr, indices, data : 1-D массивы numpy
        CSR-память матрицы A.
    b : 1-D numpy array
        Правая часть.
    tol : float, default 1e-8
        Относительная невязка.
    max_iter : int, default 1000
        Максимум итераций KSP.
    atol : float, default 1e-50
        Абсолютная невязка.

    Returns
    -------
    x : numpy.ndarray
        Вектор решения.
    its : int
        Количество итераций.
    res : float
        Итоговая относительная невязка.
    """
    n = b.shape[0]

    # Создаём параллельный коммуникатор – пока SINGLE-MPI (rank 0)
    comm = PETSc.COMM_SELF

    # PETSc ждёт CSR-индексы int32; если пришли int64, конвертируем
    if indptr.dtype != np.int32:
        indptr = indptr.astype(np.int32)
    if indices.dtype != np.int32:
        indices = indices.astype(np.int32)

    # ------------------------------------------------------------
    #  Ситуация «size(I) is 2N+1, expected N+1»
    # ------------------------------------------------------------
    # Иногда в CPR передают полный 2N×2N Якобиан, но правая часть
    # содержит только давление (N элементов). В этом случае rowptr
    # (indptr) имеет длину 2N+1. Чтобы избежать ValueError при
    # создании Mat, обрезаем CSR до верхнего левого N×N блока.
    if indptr.shape[0] == 2 * n + 1:
        # Последний валидный offset для строки N (указатель на начало строки N)
        last_offset = indptr[n]

        # Создаём новые списки для отфильтрованных строк
        new_indptr = np.zeros(n + 1, dtype=np.int32)
        new_indices_chunks = []
        new_data_chunks = []

        for row in range(n):
            start = indptr[row]
            end = indptr[row + 1]
            row_indices = indices[start:end]
            row_data = data[start:end]

            mask = row_indices < n  # оставляем только давление-колонки
            row_indices = row_indices[mask]
            row_data = row_data[mask]

            new_indices_chunks.append(row_indices)
            new_data_chunks.append(row_data)
            new_indptr[row + 1] = new_indptr[row] + row_indices.size

        indices = np.concatenate(new_indices_chunks).astype(np.int32)
        data = np.concatenate(new_data_chunks)
        indptr = new_indptr

        # Убедимся, что indptr[-1] совпадает с indices.size
        assert indptr[-1] == indices.size, "CSR truncation size mismatch"

    # --- Конец исправления размера CSR ---

    # Проверяем корректность матрицы
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Матрица содержит NaN или Inf значения")
    
    if np.any(np.isnan(b)) or np.any(np.isinf(b)):
        raise ValueError("Правая часть содержит NaN или Inf значения")

    A = PETSc.Mat().createAIJ(size=(n, n), csr=(indptr, indices, data), comm=comm)
    A.setUp()

    # Векторы RHS и решения
    bb = PETSc.Vec().createWithArray(b, comm=comm)
    xx = PETSc.Vec().create(comm=comm)
    xx.setSizes(n)
    xx.setUp()

    # Настройка BoomerAMG через PETSc опции
    opts = PETSc.Options()
    opts.setValue("pc_type", "hypre")
    opts.setValue("pc_hypre_type", "boomeramg")
    
    # Профессиональные настройки BoomerAMG для резервуарных симуляций
    opts.setValue("pc_hypre_boomeramg_coarsen_type", "falgout")        # Falgout coarsening (надежный)
    opts.setValue("pc_hypre_boomeramg_agg_nl", "1")                   # Aggressive coarsening levels
    opts.setValue("pc_hypre_boomeramg_relax_type_all", "symmetric-sor/jacobi")  # Симметричный релакс
    opts.setValue("pc_hypre_boomeramg_strong_threshold", "0.5")       # Strong threshold
    opts.setValue("pc_hypre_boomeramg_max_levels", "10")              # Max levels
    opts.setValue("pc_hypre_boomeramg_tol", "0.0")                   # Exact solve on coarse grid
    opts.setValue("pc_hypre_boomeramg_max_iter", "1")                # V-cycle once
    opts.setValue("pc_hypre_boomeramg_interp_type", "classical")     # Классическая интерполяция
    opts.setValue("pc_hypre_boomeramg_truncfactor", "0.3")           # Truncation factor
    opts.setValue("pc_hypre_boomeramg_P_max", "4")                   # Max elements per row in P
    
    # KSP настройки - используем простые, но надёжные
    opts.setValue("ksp_type", "gmres")         # Обычный GMRES
    opts.setValue("ksp_gmres_restart", "30")   # Restart every 30 iterations
    opts.setValue("ksp_max_it", str(max_iter))
    opts.setValue("ksp_rtol", str(tol))
    opts.setValue("ksp_atol", str(atol))

    # KSP с правильными настройками
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A)
    ksp.setFromOptions()  # Применяем все опции
    
    try:
        ksp.solve(bb, xx)
        its = ksp.getIterationNumber()
        res = ksp.getResidualNorm()
        
        # Проверяем результат на NaN
        x_result = xx.getArray().copy()
        if np.any(np.isnan(x_result)) or np.any(np.isinf(x_result)):
            print(f"ПРЕДУПРЕЖДЕНИЕ: BoomerAMG вернул некорректное решение (NaN/Inf)")
            # Возвращаем нулевое решение как fallback
            x_result = np.zeros_like(b)
            res = float('nan')
        
        if np.isnan(res) or np.isinf(res):
            print(f"ПРЕДУПРЕЖДЕНИЕ: BoomerAMG вернул некорректную невязку: {res}")
            res = float('nan')
            
        return x_result, its, res
        
    except Exception as e:
        print(f"ОШИБКА BoomerAMG: {e}")
        # Возвращаем нулевое решение при полном провале
        return np.zeros_like(b), 0, float('nan')
    
    finally:
        # Очищаем опции, чтобы не влиять на другие части кода
        opts.clear() 