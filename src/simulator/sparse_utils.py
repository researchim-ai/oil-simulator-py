import torch
import numpy as np


class SparseMatrixBuilder:
    """
    Структура блочной CSR-матрицы для ячеечной схемы.

    По умолчанию — 7-точечный шаблон сетки nx×ny×nz (нумерация как tensor(nx, ny, nz).view(-1), k быстрее всех).
    Если заданы pairs = (L, R) — произвольные связи ячеек (NNC, разломы, выклинивания): шаблон строится по ним.

    Маппинги для векторизованной сборки (позиции блоков в values[]):
    - diag_map[idx] — блок (idx, idx);
    - x_pos_map[idx] — (idx, idx+ny·nz), x_neg_map — (idx, idx−ny·nz), аналогично y и z (−1, если связи нет);
    - position(rows, cols) — для произвольных пар.
    """

    def __init__(self, nx, ny, nz, block_size=1, device='cpu', pairs=None):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.n_cells = nx * ny * nz
        self.block_size = block_size
        self.device = device
        self._build_structure(pairs)

    def _structured_pairs(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        idx = np.arange(self.n_cells).reshape(nx, ny, nz)
        L, R = [idx[:-1].ravel(), idx[:, :-1].ravel(), idx[:, :, :-1].ravel()], \
               [idx[1:].ravel(), idx[:, 1:].ravel(), idx[:, :, 1:].ravel()]
        return np.concatenate(L), np.concatenate(R)

    def _build_structure(self, pairs):
        n = self.n_cells
        if pairs is None:
            L, R = self._structured_pairs()
        else:
            L, R = (np.asarray(v.detach().cpu() if torch.is_tensor(v) else v, dtype=np.int64) for v in pairs)
        diag = np.arange(n, dtype=np.int64)
        rows = np.concatenate([diag, L, R])
        cols = np.concatenate([diag, R, L])
        key = np.unique(rows * n + cols)
        rows, cols = key // n, key % n
        self._keys = key
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(indptr, rows + 1, 1)
        indptr = np.cumsum(indptr)
        dev = self.device
        self.indices = torch.as_tensor(cols, dtype=torch.long, device=dev)
        self.indptr = torch.as_tensor(indptr, dtype=torch.long, device=dev)
        self.n_nonzero = int(key.size)
        self.diag_map = self.position(diag, diag)

        sx, sy = self.ny * self.nz, self.nz
        for name, step, axis in (("x", sx, 0), ("y", sy, 1), ("z", 1, 2)):
            ijk = np.stack(np.unravel_index(diag, (self.nx, self.ny, self.nz)))
            for sign, suffix in ((1, "pos"), (-1, "neg")):
                nb = ijk[axis] + sign
                ok = (nb >= 0) & (nb < (self.nx, self.ny, self.nz)[axis])
                m = np.full(n, -1, dtype=np.int64)
                m[ok] = self._find(diag[ok], diag[ok] + sign * step)
                setattr(self, f"{name}_{suffix}_map", torch.as_tensor(m, dtype=torch.long, device=dev))

    def _find(self, rows, cols):
        key = np.asarray(rows, dtype=np.int64) * self.n_cells + np.asarray(cols, dtype=np.int64)
        pos = np.searchsorted(self._keys, key)
        pos = np.minimum(pos, self._keys.size - 1)
        return np.where(self._keys[pos] == key, pos, -1)

    def position(self, rows, cols):
        """Позиции блоков (rows[i], cols[i]) в values[]; −1, если блока нет в шаблоне."""
        r = rows.detach().cpu().numpy() if torch.is_tensor(rows) else rows
        c = cols.detach().cpu().numpy() if torch.is_tensor(cols) else cols
        return torch.as_tensor(self._find(r, c), dtype=torch.long, device=self.device)

    def get_bsr_tensor(self, block_values):
        """BSR-тензор PyTorch; block_values: [n_nonzero, B, B]."""
        return torch.sparse_bsr_tensor(
            self.indptr,
            self.indices,
            block_values,
            size=(self.n_cells * self.block_size, self.n_cells * self.block_size),
            dtype=torch.float64
        )
