import torch

def dense_to_csr(mat: torch.Tensor) -> torch.Tensor:
    """Быстро переводит dense-тензор в `torch.sparse_csr_tensor`.
    Для работы на GPU требуется PyTorch ≥1.12.
    Если вход уже разреженный – возвращает как есть.
    """
    if mat.is_sparse_csr:
        return mat
    if mat.is_sparse:
        return mat.to_sparse_csr()
    # PyTorch позволяет to_sparse_csr начиная с 1.12
    try:
        return mat.to_sparse_csr()
    except RuntimeError:
        # На старых версиях fallback: через COO → CSR
        coo = mat.to_sparse()
        return coo.to_sparse_csr() 