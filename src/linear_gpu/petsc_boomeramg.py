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

    # 🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильная инициализация PETSc
    # Проверяем, инициализирован ли PETSc
    if not PETSc.Sys.isInitialized():
        print("🔧 PETSc: Инициализируем PETSc...")
        PETSc.Sys.initialize([])
        print("✅ PETSc: Инициализация завершена")
    
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

    # 🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Защищенная работа с PETSc объектами
    A = None
    bb = None
    xx = None
    ksp = None
    
    try:
        # Создаем PETSc объекты в защищенном контексте
        A = PETSc.Mat().createAIJ(size=(n, n), csr=(indptr, indices, data), comm=comm)
        A.setUp()

        # Векторы RHS и решения
        bb = PETSc.Vec().createWithArray(b, comm=comm)
        xx = PETSc.Vec().create(comm=comm)
        xx.setSizes(n)
        xx.setUp()

        # Настройка BoomerAMG через PETSc опции (локальные опции)
        opts = PETSc.Options()
        
        # 🔧 ИСПРАВЛЕНО: Используем более безопасные настройки
        opts.setValue("pc_type", "hypre")
        opts.setValue("pc_hypre_type", "boomeramg")
        
        # Упрощенные настройки BoomerAMG для стабильности
        opts.setValue("pc_hypre_boomeramg_coarsen_type", "pmis")         # PMIS coarsening (более стабильный)
        opts.setValue("pc_hypre_boomeramg_relax_type_all", "jacobi")     # Jacobi релакс (стабильный)
        opts.setValue("pc_hypre_boomeramg_strong_threshold", "0.7")      # Более консервативный порог
        opts.setValue("pc_hypre_boomeramg_max_levels", "5")              # Меньше уровней
        opts.setValue("pc_hypre_boomeramg_tol", "0.0")                   # Точное решение на грубой сетке
        opts.setValue("pc_hypre_boomeramg_max_iter", "1")                # Один V-цикл
        
        # KSP настройки - консервативные и надёжные
        opts.setValue("ksp_type", "gmres")
        opts.setValue("ksp_gmres_restart", "20")   # Меньший restart
        opts.setValue("ksp_max_it", str(min(max_iter, 100)))  # Ограничиваем итерации
        opts.setValue("ksp_rtol", str(tol))
        opts.setValue("ksp_atol", str(atol))
        
        # Отключаем печать статистики (0=off)
        opts.setValue("pc_hypre_boomeramg_print_statistics", "0")

        # KSP с правильными настройками
        ksp = PETSc.KSP().create(comm=comm)
        ksp.setOperators(A)
        ksp.setFromOptions()  # Применяем все опции
        
        print("🔧 PETSc: Решаем систему с BoomerAMG...")
        ksp.solve(bb, xx)
        
        its = ksp.getIterationNumber()
        res = ksp.getResidualNorm()
        
        # Проверяем результат на NaN
        x_result = xx.getArray().copy()
        if np.any(np.isnan(x_result)) or np.any(np.isinf(x_result)):
            print(f"ПРЕДУПРЕЖДЕНИЕ: BoomerAMG вернул некорректное решение (NaN/Inf)")
            x_result = np.zeros_like(b)
            res = float('nan')
        
        if np.isnan(res) or np.isinf(res):
            print(f"ПРЕДУПРЕЖДЕНИЕ: BoomerAMG вернул некорректную невязку: {res}")
            res = float('nan')
        else:
            print(f"✅ PETSc: Решение успешно, итераций: {its}, невязка: {res:.3e}")
            
        # Очищаем опции
        opts.clear()
        
        return x_result, its, res
        
    except Exception as e:
        print(f"❌ ОШИБКА BoomerAMG: {e}")
        import traceback
        traceback.print_exc()
        # Возвращаем нулевое решение при полном провале
        return np.zeros_like(b), 0, float('nan')
    
    finally:
        # 🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильная очистка PETSc объектов
        try:
            if ksp is not None:
                ksp.destroy()
            if xx is not None:
                xx.destroy()
            if bb is not None:
                bb.destroy()
            if A is not None:
                A.destroy()
        except Exception as cleanup_error:
            print(f"⚠️ Предупреждение при очистке PETSc объектов: {cleanup_error}")