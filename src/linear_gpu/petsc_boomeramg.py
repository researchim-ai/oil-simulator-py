import numpy as np
from petsc4py import PETSc


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

    A = PETSc.Mat().createAIJ(size=(n, n), csr=(indptr, indices, data), comm=comm)
    A.setUp()

    # Векторы RHS и решения
    bb = PETSc.Vec().createWithArray(b, comm=comm)
    xx = PETSc.Vec().create(comm=comm)
    xx.setSizes(n)
    xx.setUp()

    # KSP
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.getPC().setType("hypre")  # BoomerAMG внутри PETSc
    ksp.getPC().setHYPREType("boomeramg")

    ksp.setTolerances(rtol=tol, atol=atol, max_it=max_iter)
    ksp.setFromOptions()

    ksp.solve(bb, xx)

    its = ksp.getIterationNumber()
    res = ksp.getResidualNorm()

    return xx.getArray().copy(), its, res 