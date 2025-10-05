import torch
from typing import Callable

def gmres(*args, **kwargs):
    """Заглушка: используйте linear_gpu.fgmres.fgmres вместо solver.krylov.gmres.

    Этот модуль оставлен для обратной совместимости, но базовая реализация
    не поддерживается. Импортируйте и вызывайте `linear_gpu.fgmres.fgmres`.
    """
    raise NotImplementedError("Use linear_gpu.fgmres.fgmres instead of solver.krylov.gmres")