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
import os

__all__ = ["build_level_csr", "GeoLevel"]

def _harmonic(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:  # noqa: D401
    """Harmonic average used for transmissibilities."""
    return 2.0 * a * b / (a + b + eps)

def build_level_csr(kx: torch.Tensor, ky: torch.Tensor, kz: torch.Tensor | None,
                     hx: float, hy: float, hz: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Возвращает indptr, indices, data как torch‑тензоры (CPU); перенос на device делается снаружи.

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

    indptr, indices, data = build_7pt_csr(
        Tx.cpu().numpy(), Ty.cpu().numpy(),
        Tz.cpu().numpy() if Tz is not None else None,
        kx.shape[2], kx.shape[1], kx.shape[0]
    )
    if not torch.is_tensor(indptr):
        indptr  = torch.as_tensor(indptr,  dtype=torch.int64)
        indices = torch.as_tensor(indices, dtype=torch.int64)
        data    = torch.as_tensor(data,    dtype=torch.float64)
    return indptr, indices, data

def build_level_from_csr(A_csr: torch.Tensor,
                         diag: torch.Tensor,
                         inv_l1: torch.Tensor,
                         shape: tuple[int, int, int],
                         hx: float, hy: float, hz: float,
                         *, device: str = "cuda") -> "GeoLevel":
    """Создаёт GeoLevel из уже готовых A_csr/diag/inv_l1 (Galerkin уровень)."""
    nz, ny, nx = shape

    lvl = object.__new__(GeoLevel)   # обходим __init__
    lvl.kx = torch.zeros(nz, ny, nx, dtype=torch.float64, device=device)
    lvl.ky = torch.zeros_like(lvl.kx)
    lvl.kz = torch.zeros_like(lvl.kx)
    lvl.hx, lvl.hy, lvl.hz = float(hx), float(hy), float(hz)
    lvl.device = device

    lvl.A_csr = A_csr.to(device)
    lvl.diag  = diag.to(device)
    lvl.W_rows = torch.ones(lvl.A_csr.size(0), dtype=torch.float64, device=device)

    # --- Red/Black маски ---
    z = torch.arange(nz, device=device)[:, None, None]
    y = torch.arange(ny, device=device)[None, :, None]
    x = torch.arange(nx, device=device)[None, None, :]
    colors = (z + y + x) % 2 == 0
    lvl.is_red   = colors.reshape(-1)
    lvl.is_black = ~lvl.is_red

    # --- коэффициенты вдоль z для line-GS ---
    crow = lvl.A_csr.crow_indices()
    col  = lvl.A_csr.col_indices()
    vals = lvl.A_csr.values()

    n_rows = crow.numel() - 1           # фактическое число строк
    total  = n_rows                     # должно совпасть с nz*ny*nx, но берём из CSR
    stride_z = nx * ny

    row_idx = torch.repeat_interleave(torch.arange(n_rows, device=device),
                                      crow[1:] - crow[:-1])

    lvl.a_up = torch.zeros(total, dtype=torch.float64, device=device)
    lvl.a_dn = torch.zeros_like(lvl.a_up)

    diff = col - row_idx
    mask_up = diff == stride_z
    mask_dn = diff == -stride_z
    if mask_up.any():
        lvl.a_up.index_copy_(0, row_idx[mask_up], vals[mask_up])
    if mask_dn.any():
        lvl.a_dn.index_copy_(0, row_idx[mask_dn], vals[mask_dn])

    return lvl



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
        crow = self.A_csr.crow_indices()
        col  = self.A_csr.col_indices()
        vals = self.A_csr.values()


        # diag и inv-sqrt(diag) для эквилибрации
        vals = self.A_csr.values()
        crow = self.A_csr.crow_indices()
        # Безопасный поиск диагонали: если строки без диагонали – аккуратно чиним
        n_rows = int(crow.numel() - 1)
        row_idx = torch.repeat_interleave(torch.arange(n_rows, device=col.device), crow[1:] - crow[:-1])
        pos_all = torch.nonzero(col == row_idx, as_tuple=False).squeeze(1)
        diag_idx = torch.full((n_rows,), -1, dtype=torch.int64, device=col.device)
        if pos_all.numel() > 0:
            rows = row_idx[pos_all]
            diag_idx[rows] = pos_all

        if (diag_idx < 0).any():
            miss = torch.nonzero(diag_idx < 0, as_tuple=False).squeeze(1)
            for i in miss.tolist():
                s = int(crow[i].item()); e = int(crow[i+1].item())
                if e == s:
                    raise RuntimeError(f"Empty CSR row {i} — add diagonal before building GeoLevel")
                # Мягкая починка диагонали без зануления строки
                row_slice = slice(s, e)
                row_abs = vals[row_slice].abs()
                rel = torch.nonzero(col[row_slice] == i, as_tuple=False)
                if rel.numel():
                    j = s + int(rel[0])
                    # поднимем до безопасного уровня, но не обнуляя связи
                    safe_min = torch.clamp(row_abs.sum() * 1e-12, min=torch.tensor(1e-30, device=vals.device, dtype=vals.dtype))
                    vals[j] = torch.sign(vals[j]) * torch.clamp(vals[j].abs(), min=safe_min)
                    diag_idx[i] = j
                else:
                    # перепрофилируем наименее значимый элемент под диагональ
                    k_rel = int(torch.argmin(row_abs).item())
                    j = s + k_rel
                    col[j] = i
                    vals[j] = torch.clamp(row_abs.sum(), min=torch.tensor(1e-30, device=vals.device, dtype=vals.dtype))
                    diag_idx[i] = j



        # --- фиксация нулевых строк (неактивные ячейки) -----------------
        diag_vals = vals[diag_idx].abs()
        # 🔧 КРИТИЧЕСКИЙ ФИКС: порог 1e-12 оказался слишком высоким —
        #  при типичных transmissibility ~1e-13 все активные ячейки считались
        #  «пустыми», и их диагонали затирались до 1.  В итоге вся матрица
        #  превращалась в почти единичную и Geo-AMG «взрывался».  Снижаем
        #  порог до 1e-20 (≈ машинный эпсилон для float64) либо, что лучше,
        #  используем относительный: <1e-12 * median(|diag|).
        dmed = diag_vals.median()
        # Аккуратный порог: минимум из 1e-6*median и 1‑го перцентиля,
        # и только для действительно «тонких» строк по L1‑норме
        p1 = torch.quantile(diag_vals, 0.01)
        thr = torch.clamp(torch.minimum(1e-6 * dmed, p1), min=torch.tensor(1e-30, device=diag_vals.device, dtype=diag_vals.dtype))
        # Требуем одновременно малую диагональ и малую L1‑сумму строки
        row_abs_sum = self.row_abs_sum if hasattr(self, 'row_abs_sum') else None
        if row_abs_sum is None:
            row_counts = crow[1:] - crow[:-1]
            row_idx = torch.repeat_interleave(torch.arange(int(crow.numel()-1), device=vals.device), row_counts)
            row_abs_sum = torch.zeros_like(diag_vals)
            row_abs_sum.index_add_(0, row_idx, vals.abs())
        l1med = row_abs_sum.median().clamp_min(torch.tensor(1e-30, device=row_abs_sum.device, dtype=row_abs_sum.dtype))
        zero_mask = (diag_vals < thr) & (row_abs_sum < 1e-12 * l1med)

        if zero_mask.any():
            # Задаём A_ii = 1, off-diag оставляем как есть (они уже ~0)
            vals[diag_idx[zero_mask]] = 1.0
            diag_vals = vals[diag_idx].abs()  # обновляем

        self.diag = diag_vals.to(dtype=torch.float64)  # уже на device
        
        # row-scale (по умолчанию единичный, если не делали row-equil)
        self.W_rows = torch.ones(self.diag.numel(), dtype=torch.float64, device=device)


        # -------- L1-диагональ: 1 / Σ_j |A_ij| -------------------------
        row_counts = crow[1:] - crow[:-1]
        row_idx = torch.repeat_interleave(torch.arange(self.diag.numel(), device=device), row_counts)
        row_abs_sum = torch.zeros_like(self.diag)
        row_abs_sum.index_add_(0, row_idx, vals.abs())
        self.row_abs_sum = row_abs_sum

        # ---- Изолированные строки: Σ|A_ij| < 1e-8 -----------------------

        med = row_abs_sum.median()
        iso_thr = torch.clamp(1e-6 * med, min=torch.tensor(1e-30, device=med.device))

        iso_mask = row_abs_sum < iso_thr


        safe_sum = row_abs_sum.clone()
        safe_sum[iso_mask] = 1.0  # чтобы 1/sum не дал Inf
        self.inv_l1 = 1.0 / safe_sum
        # Jacobi не должен менять изолированные ячейки
        self.inv_l1[iso_mask] = 0.0

        # -------- Гибридная диагональ релаксации (L1-Jacobi / diag) ----
        # Если диагональ доминирует: invD = 1/|A_ii|, иначе 1/Σ|A_ij|
        tau = 0.2
        diag_abs = self.diag
        off_sum = (row_abs_sum - diag_abs).clamp_min(0.0)
        use_diag = diag_abs >= tau * off_sum
        invD = torch.empty_like(diag_abs)
        invD[use_diag] = 1.0 / diag_abs[use_diag].clamp_min(1e-30)
        invD[~use_diag] = 1.0 / row_abs_sum[~use_diag].clamp_min(1e-30)
        invD[iso_mask] = 0.0
        # Позволим релаксации быть сильнее единицы для слабозаселённых строк
        self.inv_relax = invD.clamp_max(4.0)

        if os.environ.get("OIL_DEBUG", "0") == "1":
            n_iso = iso_mask.sum().item()
            print(f"[GeoLevel] iso_thr={iso_thr.item():.3e}; isolated rows={n_iso}/{self.inv_l1.numel()}")


        # ----------------- DEBUG: статистика строк L1-нормы -----------------
        if os.environ.get("OIL_DEBUG", "0") == "1":
            print(
                f"[GeoLevel] row_abs_sum: min={row_abs_sum.min().item():.3e}, "
                f"median={row_abs_sum.median().item():.3e}, max={row_abs_sum.max().item():.3e}"
            )
            print(f"[GeoLevel] inv_l1 max={self.inv_l1.max().item():.3e}")
            if torch.isnan(self.inv_l1).any() or torch.isinf(self.inv_l1).any():
                nan_cnt = torch.isnan(self.inv_l1).sum().item()
                inf_cnt = torch.isinf(self.inv_l1).sum().item()
                print(f"[GeoLevel] ⚠️  inv_l1 has nan={nan_cnt}, inf={inf_cnt}")

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
        diff = col - row_idx
        mask_up = diff == stride_z
        mask_dn = diff == -stride_z
        if mask_up.any():
            self.a_up.index_copy_(0, row_idx[mask_up], vals[mask_up])
        if mask_dn.any():
            self.a_dn.index_copy_(0, row_idx[mask_dn], vals[mask_dn])

    def matvec_hat(self, x: torch.Tensor) -> torch.Tensor:
        """
        y = Â * x  (всё уже в hat-пространстве)
        """
        return torch.sparse.mm(self.A_csr, x.unsqueeze(1)).squeeze(1)

    @property
    def n_cells(self) -> int:  # noqa: D401
        return self.kx.numel() 