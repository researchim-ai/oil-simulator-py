import contextlib
import io
import os
import sys
import time

import torch

try:
    from solver.classical_amg import ClassicalAMG
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from solver.classical_amg import ClassicalAMG


class CPRPreconditionerGPU:
    """
    Двухступенчатый CPR (Constrained Pressure Residual) для блочной системы nb×nb (первая переменная — P).

    1. Давление: квази-IMPES развязка — веса w_i = D_ii^{-T} e_p, A_p[i,j] = w_i · J_ij[:, p],
       r_p = w_i · r_i; решается одним V-циклом AMG.
    2. Полная система: блочный Якоби (обращение диагональных блоков 2x2) по остатку после шага 1.

    Иерархию AMG можно переиспользовать (``amg=``): тогда пересчитываются только значения
    (``ClassicalAMG.update_matrix``), что на порядок дешевле полного построения.
    """

    def __init__(self, J_vals, indptr, indices, n_cells, device, amg=None, smoother_sweeps=1,
                 amg_verbose=False):
        self.n_cells = n_cells
        self.device = device
        self.smoother_sweeps = max(1, int(smoother_sweeps))
        crow = indptr.long()
        col = indices.long()
        self.rows = torch.repeat_interleave(torch.arange(n_cells, device=J_vals.device), crow[1:] - crow[:-1])
        self.col = col
        self.J_vals = J_vals

        self.nb = J_vals.shape[1]
        diag_mask = self.rows == col
        diag = J_vals[diag_mask]
        if diag.shape[0] != n_cells:
            raise RuntimeError(f"CPR: найдено {diag.shape[0]} диагональных блоков, ожидалось {n_cells}")
        if self.nb == 2:
            a, b, c, d = diag[:, 0, 0], diag[:, 0, 1], diag[:, 1, 0], diag[:, 1, 1]
            det = a * d - b * c
            tiny = 1e-300
            det = torch.where(det.abs() < tiny, torch.full_like(det, tiny), det)
            self.inv_diag = torch.stack([torch.stack([d, -b], -1), torch.stack([-c, a], -1)], -2) / det.view(-1, 1, 1)
        else:
            inv, info = torch.linalg.inv_ex(diag)
            bad = (info != 0) | ~torch.isfinite(inv).all(dim=(1, 2))
            if bool(bad.any()):
                inv[bad] = torch.linalg.pinv(diag[bad])
            self.inv_diag = inv

        # квази-IMPES веса: w = D^{-T} e_p  -> первая строка D^{-1} как столбец
        self.w = self.inv_diag[:, 0, :].clone()  # [N, nb]
        data_p = (self.w[self.rows] * J_vals[:, :, 0]).sum(-1)
        max_val = data_p.abs().max()
        self.p_scale = max_val if max_val > 0 else torch.tensor(1.0, device=J_vals.device, dtype=J_vals.dtype)
        data_p = data_p / self.p_scale
        A_p = torch.sparse_csr_tensor(crow, col, data_p, size=(n_cells, n_cells))

        self.amg = None
        self.amg_error = None
        self.amg_rebuilt = False
        self.amg_levels = []
        self.amg_log = ""
        t0 = time.perf_counter()
        buf = io.StringIO()
        redirect = contextlib.nullcontext() if amg_verbose else contextlib.redirect_stdout(buf)
        try:
            with redirect:
                if amg is not None:
                    amg.update_matrix(A_p)
                    self.amg = amg
                else:
                    self.amg = ClassicalAMG(A_p, max_levels=15, theta=0.25, mixed_precision=True, nullspace_dim=0)
                    self.amg_rebuilt = True
            self.amg_levels = [int(lvl['n']) for lvl in self.amg.levels]
        except Exception as e:
            self.amg_error = str(e)
            self.amg = None
        self.amg_log = buf.getvalue()
        self.amg_build_s = time.perf_counter() - t0

    def apply_A(self, x):
        x_blk = x.view(-1, self.nb)
        y_vals = torch.bmm(self.J_vals, x_blk[self.col].unsqueeze(-1)).squeeze(-1)
        y = torch.zeros_like(x_blk)
        y.index_add_(0, self.rows, y_vals)
        return y.view(-1)

    def _block_jacobi(self, r):
        return torch.bmm(self.inv_diag, r.view(-1, self.nb).unsqueeze(-1)).view(-1)

    def matvec(self, r):
        x = torch.zeros_like(r)
        r_cur = r
        if self.amg is not None:
            r_p = (self.w * r.view(-1, self.nb)).sum(-1) / self.p_scale
            with torch.no_grad():
                dp = self.amg.solve(r_p, tol=0.1, max_iter=1)
            x.view(-1, self.nb)[:, 0] = dp.to(device=r.device, dtype=r.dtype)
            r_cur = r - self.apply_A(x)
        for _ in range(self.smoother_sweeps):
            x = x + self._block_jacobi(r_cur)
            if self.smoother_sweeps > 1:
                r_cur = r - self.apply_A(x)
        return x
