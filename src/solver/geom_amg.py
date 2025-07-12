import torch
import torch.nn.functional as F
from typing import Tuple


def _apply_poisson(x: torch.Tensor, hx: float, hy: float, hz: float) -> torch.Tensor:
    """Applies 7-point Poisson stencil to 3-D field x with grid steps h*."""
    # periodic = False, homogeneous Neumann at boundary (zero flux): we simply skip outside points
    scale_x = 1.0 / (hx * hx)
    scale_y = 1.0 / (hy * hy)
    scale_z = 1.0 / (hz * hz)

    r = (
        -2.0 * (scale_x + scale_y + scale_z) * x
        + scale_x * (torch.roll(x, shifts=1, dims=2) + torch.roll(x, shifts=-1, dims=2))
        + scale_y * (torch.roll(x, shifts=1, dims=1) + torch.roll(x, shifts=-1, dims=1))
        + scale_z * (torch.roll(x, shifts=1, dims=0) + torch.roll(x, shifts=-1, dims=0))
    )
    return -r  # so that A = -∇² is symmetric positive


def _jacobi_relax(x: torch.Tensor, b: torch.Tensor, hx: float, hy: float, hz: float,
                  omega: float = 0.8, iters: int = 2) -> torch.Tensor:
    diag = 2.0 * (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz))
    inv_diag = 1.0 / diag
    for _ in range(iters):
        r = b - _apply_poisson(x, hx, hy, hz)
        x = x + omega * inv_diag * r
    return x


def _restrict(vol: torch.Tensor) -> torch.Tensor:
    vol5d = vol[None, None, ...]  # add N,C dims
    coarse = F.avg_pool3d(vol5d, kernel_size=2, stride=2, padding=0)
    return coarse[0, 0]


def _prolong(coarse: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    vol5d = coarse[None, None, ...]
    fine = F.interpolate(vol5d, size=target_shape, mode="trilinear", align_corners=False)
    return fine[0, 0]


def _v_cycle(level: int, x: torch.Tensor, b: torch.Tensor, hx: float, hy: float, hz: float,
             max_levels: int, omega: float, pre: int, post: int) -> torch.Tensor:
    if level == max_levels or min(x.shape) <= 4:
        # Solve on coarsest grid by a few Jacobi sweeps
        return _jacobi_relax(x, b, hx, hy, hz, omega=omega, iters=20)

    # pre-smoothing
    x = _jacobi_relax(x, b, hx, hy, hz, omega=omega, iters=pre)

    # compute residual and restrict
    r = b - _apply_poisson(x, hx, hy, hz)
    r_c = _restrict(r)

    # init coarse correction
    x_c = torch.zeros_like(r_c)

    # grid spacing doubles
    x_c = _v_cycle(level + 1, x_c, r_c, 2 * hx, 2 * hy, 2 * hz,
                   max_levels, omega, pre, post)

    # prolongate and correct
    e = _prolong(x_c, x.shape)
    x = x + e

    # post-smoothing
    x = _jacobi_relax(x, b, hx, hy, hz, omega=omega, iters=post)
    return x


def mg_solve(b: torch.Tensor, hx: float = 1.0, hy: float = 1.0, hz: float = 1.0,
             cycles: int = 4, max_levels: int = 10, omega: float = 0.8,
             pre: int = 2, post: int = 2, device: str = "cuda") -> Tuple[torch.Tensor, float]:
    """Solves Poisson equation A x = b with homogeneous Neumann BC using geometric multigrid.

    Returns (x, final_residual_norm)."""
    x = torch.zeros_like(b, device=device)
    b = b.to(device)
    hx = float(hx); hy = float(hy); hz = float(hz)

    for _ in range(cycles):
        x = _v_cycle(0, x, b, hx, hy, hz, max_levels, omega, pre, post)
    res = torch.norm(b - _apply_poisson(x, hx, hy, hz)).item()
    return x, res 


class GeoSolver:
    """Простой геометрический мультигрид-решатель блока давления.

    Интерфейс повторяет BoomerSolver/AmgXSolver: метод solve(rhs, tol, max_iter)
    принимает rhs в виде numpy-массива (vector) и возвращает numpy-массив решения.
    """
    def __init__(self, reservoir):
        self.nx, self.ny, self.nz = reservoir.dimensions
        self.hx, self.hy, self.hz = map(float, reservoir.grid_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- копируем тензоры проницаемости (м^2) на выбранное устройство ---
        kx = reservoir.permeability_x.to(self.device, dtype=torch.float32).clone()
        ky = reservoir.permeability_y.to(self.device, dtype=torch.float32).clone()
        kz = reservoir.permeability_z.to(self.device, dtype=torch.float32).clone()

        # --- выравниваем размер до чётного, чтобы avg_pool3d 2× работал без ошибок
        def _pad_even(t: torch.Tensor):
            dz = t.shape[0] % 2
            dy = t.shape[1] % 2
            dx = t.shape[2] % 2
            if dx or dy or dz:
                pad = (0, dx, 0, dy, 0, dz)  # (W_left, W_right, H_left, H_right, D_left, D_right)
                t = F.pad(t[None, None, ...], pad, mode="replicate")[0, 0]
            return t, (dz, dy, dx)

        kx, self._pad = _pad_even(kx)
        ky, _ = _pad_even(ky)
        kz, _ = _pad_even(kz)

        self.kx, self.ky, self.kz = kx, ky, kz

        # --- строим иерархию сеток (geometric 2× coarsening) -------------
        self.levels = []  # каждый элемент: dict {"kx", "ky", "kz", "hx", ...}
        kx, ky, kz = self.kx, self.ky, self.kz
        hx, hy, hz = self.hx, self.hy, self.hz

        while min(kx.shape[-1], kx.shape[-2], kx.shape[-3]) >= 4 and len(self.levels) < 6:
            self.levels.append({
                "kx": kx,
                "ky": ky,
                "kz": kz,
                "hx": hx,
                "hy": hy,
                "hz": hz,
            })

            # coarsen permeability with volume-weighted average (avg_pool3d)
            def pool(t):
                return F.avg_pool3d(t[None, None, ...], kernel_size=2, stride=2, padding=0)[0, 0]

            kx = pool(kx)
            ky = pool(ky)
            kz = pool(kz)
            hx *= 2.0; hy *= 2.0; hz *= 2.0

        # добавляем последний (самый грубый) уровень
        self.levels.append({
            "kx": kx,
            "ky": ky,
            "kz": kz,
            "hx": hx,
            "hy": hy,
            "hz": hz,
        })

    # ------------------------------------------------------------------
    # Вспомогательные функции переменного коэффициентного оператора
    # ------------------------------------------------------------------
    @staticmethod
    def _harmonic(a, b, eps=1e-12):
        return 2 * a * b / (a + b + eps)

    def _apply_A(self, x: torch.Tensor, lvl: int) -> torch.Tensor:
        """A x с переменными коэффициентами на уровне lvl."""
        data = self.levels[lvl]
        kx, ky, kz = data["kx"], data["ky"], data["kz"]
        hx, hy, hz = data["hx"], data["hy"], data["hz"]

        # --- transmissibilities --------------------------------------
        # Right faces (i+1/2)
        Tx_r = self._harmonic(kx[..., :-1], kx[..., 1:]) * (hy * hz) / hx
        # Left faces – shift Tx_r to the left
        Tx_l = F.pad(Tx_r, (1, 0, 0, 0, 0, 0))[:, :, :-1]

        # Front/back (y-direction)
        Ty_f = self._harmonic(ky[:, :-1, :], ky[:, 1:, :]) * (hx * hz) / hy
        Ty_b = F.pad(Ty_f, (0, 0, 1, 0, 0, 0))[:, :-1, :]

        # Up/down (z-direction)
        if kz.shape[0] > 1:
            Tz_u = self._harmonic(kz[:-1, :, :], kz[1:, :, :]) * (hx * hy) / hz
            # сдвигаем Tz_u на один слой вниз без обрезки, чтобы форма совпадала
            Tz_d = F.pad(Tz_u, (0, 0, 0, 0, 1, 0))
        else:
            Tz_u = torch.zeros_like(kx)
            Tz_d = torch.zeros_like(kx)

        # --- диагональ ------------------------------------------------
        diag = Tx_r + Tx_l + Ty_f + Ty_b + Tz_u + Tz_d + 1e-12  # избегаем деления на 0

        # --- вычисляем A x = sum T*(x - x_nb) ------------------------
        div = torch.zeros_like(x)

        # x-right / left
        div[..., :-1] += Tx_r * (x[..., :-1] - x[..., 1:])
        div[..., 1:]  += Tx_l[..., 1:] * (x[..., 1:] - x[..., :-1])

        # y-front/back (dim=1)
        div[:, :-1, :] += Ty_f * (x[:, :-1, :] - x[:, 1:, :])
        div[:, 1:, :]  += Ty_b[:, 1:, :] * (x[:, 1:, :] - x[:, :-1, :])

        # z-up/down (dim=0)
        if x.shape[0] > 1:
            div[:-1, :, :] += Tz_u * (x[:-1, :, :] - x[1:, :, :])
            div[1:, :, :]  += Tz_d[1:, :, :] * (x[1:, :, :] - x[:-1, :, :])

        return div  # SPD (positive definite)

    # ------------------------------------------------------------------
    def _jacobi_relax_var(self, x, b, lvl_data, omega=0.8, iters=2):
        kx, ky, kz = lvl_data["kx"], lvl_data["ky"], lvl_data["kz"]
        hx, hy, hz = lvl_data["hx"], lvl_data["hy"], lvl_data["hz"]

        # вычислим диагональ один раз
        Tx_r = self._harmonic(kx[..., :-1], kx[..., 1:]) * (hy * hz) / hx
        Tx_l = F.pad(Tx_r, (1, 0, 0, 0, 0, 0))[:, :, :-1]
        Ty_f = self._harmonic(ky[:, :-1, :], ky[:, 1:, :]) * (hx * hz) / hy
        Ty_b = F.pad(Ty_f, (0, 0, 1, 0, 0, 0))[:, :-1, :]
        if kz.shape[0] > 1:
            Tz_u = self._harmonic(kz[:-1, :, :], kz[1:, :, :]) * (hx * hy) / hz
            Tz_d = F.pad(Tz_u, (0, 0, 0, 0, 1, 0))
        else:
            Tz_u = Tz_d = torch.zeros_like(kx)
        # Формируем диагональ так же, как в _apply_A, чтобы избежать ошибок
        diag = torch.zeros_like(x)
        diag[..., :-1] += Tx_r
        diag[..., 1:]  += Tx_l[..., 1:]
        diag[:, :-1, :] += Ty_f
        diag[:, 1:, :]  += Ty_b[:, 1:, :]
        if x.shape[0] > 1:
            diag[:-1, :, :] += Tz_u
            diag[1:, :, :]  += Tz_d[1:, :, :]
        diag = diag + 1e-12

        inv_diag = 1.0 / diag
        for _ in range(iters):
            r = b - self._apply_A(x, lvl=self.current_level)
            x = x + omega * inv_diag * r
        return x

    # ------------------------------------------------------------------
    def _restrict_vol(self, vol):
        vol5d = vol[None, None, ...]
        coarse = F.avg_pool3d(vol5d, kernel_size=2, stride=2, padding=0)
        return coarse[0, 0]

    def _prolong_vol(self, coarse, target_shape):
        fine = F.interpolate(coarse[None, None, ...], size=target_shape, mode="trilinear", align_corners=False)
        return fine[0, 0]

    def _v_cycle_var(self, lvl, x, b, omega=0.8, pre=2, post=2):
        self.current_level = lvl
        lvl_data = self.levels[lvl]

        # критерий грубой сетки
        if lvl == len(self.levels) - 1 or min(x.shape) <= 4:
            return self._jacobi_relax_var(x, b, lvl_data, omega=omega, iters=20)

        # pre-smooth
        x = self._jacobi_relax_var(x, b, lvl_data, omega=omega, iters=pre)

        # residual
        r = b - self._apply_A(x, lvl)

        # restrict r to coarse grid
        r_c = self._restrict_vol(r)

        # zero initial correction on coarse grid
        x_c = torch.zeros_like(r_c)

        # recurse one level deeper
        x_c = self._v_cycle_var(lvl + 1, x_c, r_c, omega, pre, post)

        # prolongate correction
        e = self._prolong_vol(x_c, x.shape)
        x = x + e

        # post-smooth
        x = self._jacobi_relax_var(x, b, lvl_data, omega=omega, iters=post)
        return x

    # ------------------------------------------------------------------
    def solve(self, rhs, tol=1e-6, max_iter=10):
        nz0, ny0, nx0 = self.nz, self.ny, self.nx
        # учтём возможный паддинг
        dz, dy, dx = self._pad
        b = torch.as_tensor(rhs.reshape(nz0, ny0, nx0), dtype=torch.float32, device=self.device)
        if dx or dy or dz:
            pad = (0, dx, 0, dy, 0, dz)
            b = F.pad(b, pad, mode="constant", value=0.0)
        x = torch.zeros_like(b)

        for itr in range(max_iter):
            x = self._v_cycle_var(0, x, b, omega=0.8)
            res = torch.norm(b - self._apply_A(x, lvl=0)) / (torch.norm(b) + 1e-12)
            if res < tol:
                break

        # обрезаем паддинг
        if dx or dy or dz:
            x = x[:nz0, :ny0, :nx0]

        return x.cpu().numpy().ravel() 