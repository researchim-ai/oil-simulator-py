"""Geo-AMG v2: отдельный уровень сетки с CSR-матрицей и кешированной диагональю.

На одном уровне храним:
• kx, ky, kz            – тензоры проницаемостей (nz, ny, nx)
• hx, hy, hz            – размеры ячейки
• A_csr (torch)         – оператор давления в формате CSR (float64)
• diag (torch)          – |diag(A_csr)| (float64)

CSR строится через уже существующий helper build_7pt_csr из
`linear_gpu.csr`, поэтому код очень лёгкий.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from linear_gpu.csr import build_7pt_csr

__all__ = ["build_level_csr", "GeoLevel"]

def _harmonic(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:  # noqa: D401
    """Harmonic average used for transmissibilities."""
    return 2.0 * a * b / (a + b + eps)

def build_level_csr(kx: torch.Tensor, ky: torch.Tensor, kz: torch.Tensor | None,
                     hx: float, hy: float, hz: float, *, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Возвращает indptr, indices, data (CPU numpy) CSR оператора.

    kx, ky, kz – тензоры (nz, ny, nx), float64.  kz может быть None для 2-D.
    hx, hy, hz – размеры ячейки.
    """
    # Transmissibilities on cell faces -------------------------------------
    Tx = _harmonic(kx[..., :-1], kx[..., 1:]) * (hy * hz) / hx
    Ty = _harmonic(ky[:, :-1, :], ky[:, 1:, :]) * (hx * hz) / hy
    if kz is not None:
        Tz = _harmonic(kz[:-1, :, :], kz[1:, :, :]) * (hx * hy) / hz
    else:
        Tz = None

    indptr, indices, data = build_7pt_csr(Tx.cpu().numpy(), Ty.cpu().numpy(),
                                          Tz.cpu().numpy() if Tz is not None else None,
                                          kx.shape[2], kx.shape[1], kx.shape[0])
    # build_7pt_csr уже возвращает torch tensors; передаём их дальше
    return indptr, indices, data

class GeoLevel:  # noqa: D101
    def __init__(self, kx: torch.Tensor, ky: torch.Tensor, kz: torch.Tensor | None,
                 hx: float, hy: float, hz: float, *, device: str = "cuda"):
        self.kx, self.ky, self.kz = kx, ky, kz
        self.hx, self.hy, self.hz = float(hx), float(hy), float(hz)
        self.device = device

        indptr, indices, data = build_level_csr(kx, ky, kz, hx, hy, hz)
        indptr = indptr.to(device)
        indices = indices.to(device)
        data = data.to(device)
        self.A_csr = torch.sparse_csr_tensor(indptr, indices, data, dtype=torch.float64, device=device)
        # diag и inv-sqrt(diag) для эквилибрации
        vals = self.A_csr.values()
        crow = self.A_csr.crow_indices()
        # Индексы диагональных элементов (последние в строке)
        diag_idx = crow[1:] - 1

        # --- фиксация нулевых строк (неактивные ячейки) -----------------
        diag_vals = vals[diag_idx].abs()
        zero_mask = diag_vals < 1e-12
        if zero_mask.any():
            # Задаём A_ii = 1, off-diag оставляем как есть (они уже ~0)
            vals[diag_idx[zero_mask]] = 1.0
            diag_vals = vals[diag_idx].abs()  # обновляем

        self.diag = diag_vals.to(dtype=torch.float64)  # уже на device

        # --- Red/Black маски -------------------------------------------
        nz, ny, nx = kx.shape
        # Создаём шаблон (z+y+x) % 2 == 0 → red
        z_idx = torch.arange(nz, device=device)[:, None, None]
        y_idx = torch.arange(ny, device=device)[None, :, None]
        x_idx = torch.arange(nx, device=device)[None, None, :]
        colors = (z_idx + y_idx + x_idx) % 2 == 0
        self.is_red = colors.reshape(-1)
        self.is_black = ~self.is_red

        # --- коэффициенты вдоль оси z для line-GS ---------------------
        nx, ny, nz = nx, ny, nz  # локал
        stride_z = nx * ny
        total = self.diag.numel()
        self.a_up = torch.zeros(total, dtype=torch.float64, device=device)
        self.a_dn = torch.zeros_like(self.a_up)

        # заполняем a_up / a_dn из CSR (рассматриваем только соседей ±stride_z)
        row_idx = torch.repeat_interleave(torch.arange(total, device=device), crow[1:] - crow[:-1])
        diff = indices - row_idx
        mask_up = diff == stride_z
        mask_dn = diff == -stride_z
        if mask_up.any():
            self.a_up.index_copy_(0, row_idx[mask_up], vals[mask_up])
        if mask_dn.any():
            self.a_dn.index_copy_(0, row_idx[mask_dn], vals[mask_dn])

    @property
    def n_cells(self) -> int:  # noqa: D401
        return self.kx.numel() 