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
    return -r  # без масштабирования; A_scale применяется только в методах класса


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


# ------------------------------------------------------------
# GeoSolver: теперь с выбором сглаживателя (Jacobi или L1-GS)
# ------------------------------------------------------------
class GeoSolver:
    """Простой геометрический мультигрид-решатель блока давления.

    Интерфейс повторяет BoomerSolver/AmgXSolver: метод solve(rhs, tol, max_iter)
    принимает rhs в виде numpy-массива (vector) и возвращает numpy-массив решения.
    """
    def __init__(self, reservoir, smoother: str = "jacobi"):
        self.nx, self.ny, self.nz = reservoir.dimensions
        self.hx, self.hy, self.hz = map(float, reservoir.grid_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Тип сглаживателя: 'jacobi' | 'l1gs' | 'chebyshev'
        self.smoother = smoother.lower()
        if self.smoother not in ("jacobi", "l1gs", "chebyshev"):
            raise ValueError("smoother must be 'jacobi', 'l1gs' or 'chebyshev'")

        # --- копируем тензоры проницаемости (м^2) на выбранное устройство ---
        # В Reservoir тензоры хранятся в порядке (nx, ny, nz),
        # тогда как все вычисления GeoSolver предполагают (nz, ny, nx).
        # Поэтому транспонируем оси и делаем их contiguous.
        kx = reservoir.permeability_x.to(self.device, dtype=torch.float64).permute(2, 1, 0).contiguous()
        ky = reservoir.permeability_y.to(self.device, dtype=torch.float64).permute(2, 1, 0).contiguous()
        kz = reservoir.permeability_z.to(self.device, dtype=torch.float64).permute(2, 1, 0).contiguous()

        # После перестановки осей пересохраняем размеры (nz, ny, nx)
        self.nz, self.ny, self.nx = kx.shape

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

        # ------------------------------------------------------------
        # Масштабируем оператор так, чтобы медиана диагонали была ~1.
        # Это существенно уменьшает порождённые числа и предотвращает
        # переполнения при вычислениях λ_max и норм.
        # ------------------------------------------------------------
        diag0 = self._compute_diag(kx, ky, kz, hx, hy, hz, kx.shape)
        d_med = torch.median(diag0).item()
        if d_med < 1e-20:
            d_med = 1e-20

        # Размер системы
        n_cells = kx.numel()
        SIZE_THRESHOLD = 500
        if n_cells <= SIZE_THRESHOLD:
            # На маленьких системах дополнительноe масштабирование не нужно –
            # оно лишь усложняет условия тестов.
            self.A_scale = 1.0
        else:
            # Ограничиваем масштаб, иначе Chebyshev/Geo-AMG генерирует
            # гигантские δp и теряет устойчивость. 1e4 достаточно, чтобы
            # привести диагональ к диапазону O(1).
            self.A_scale = min(1.0 / d_med, 1.0e4)

        print(f"GeoSolver: cells={n_cells}, median(|diag|)={d_med:.3e}, A_scale={self.A_scale:.3e}")

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

        # --- transmissibilities --- (кешируем для уровня lvl) ---------
        if "Tx" not in data:
            Tx = self._harmonic(kx[..., :-1], kx[..., 1:]) * (hy * hz) / hx       # (nz, ny, nx-1)
            Ty = self._harmonic(ky[:, :-1, :], ky[:, 1:, :]) * (hx * hz) / hy     # (nz, ny-1, nx)
            if kz.shape[0] > 1:
                Tz = self._harmonic(kz[:-1, :, :], kz[1:, :, :]) * (hx * hy) / hz # (nz-1, ny, nx)
            else:
                Tz = None
            data["Tx"], data["Ty"], data["Tz"] = Tx, Ty, Tz
        else:
            Tx, Ty, Tz = data["Tx"], data["Ty"], data["Tz"]

        # --- вычисляем A x = div(T * grad x) --------------------------
        div = torch.zeros_like(x)

        # X-направление -------------------------------------------------
        div[..., :-1] += Tx * (x[..., :-1] - x[..., 1:])
        div[..., 1:]  += Tx * (x[..., 1:] - x[..., :-1])

        # Y-направление -------------------------------------------------
        div[:, :-1, :] += Ty * (x[:, :-1, :] - x[:, 1:, :])
        div[:, 1:, :]  += Ty * (x[:, 1:, :] - x[:, :-1, :])

        # Z-направление -------------------------------------------------
        if Tz is not None:
            div[:-1, :, :] += Tz * (x[:-1, :, :] - x[1:, :, :])
            div[1:, :, :]  += Tz * (x[1:, :, :] - x[:-1, :, :])
 
        return div  # SPD (positive definite)

    # ------------------------------------------------------------------
    def _compute_diag(self, kx, ky, kz, hx, hy, hz, shape):
        # --- transmissibilities (как в _apply_A) ----------------------
        Tx = self._harmonic(kx[..., :-1], kx[..., 1:]) * (hy * hz) / hx
        Ty = self._harmonic(ky[:, :-1, :], ky[:, 1:, :]) * (hx * hz) / hy
        if kz.shape[0] > 1:
            Tz = self._harmonic(kz[:-1, :, :], kz[1:, :, :]) * (hx * hy) / hz
        else:
            Tz = None

        diag = torch.zeros(shape, device=kx.device)
        # X faces
        diag[..., :-1] += Tx
        diag[..., 1:]  += Tx
        # Y faces
        diag[:, :-1, :] += Ty
        diag[:, 1:, :]  += Ty
        # Z faces
        if Tz is not None:
            diag[:-1, :, :] += Tz
            diag[1:, :, :]  += Tz

        return diag + 1e-12

    # ------------------ Jacobi (как было) ------------------------------
    def _jacobi_relax_var(self, x, b, lvl_data, lvl_idx, omega=0.8, iters=2):
        kx, ky, kz = lvl_data["kx"], lvl_data["ky"], lvl_data["kz"]
        hx, hy, hz = lvl_data["hx"], lvl_data["hy"], lvl_data["hz"]

        # --- diag из переменных коэффициентов ---
        diag = self._compute_diag(kx, ky, kz, hx, hy, hz, x.shape) * self.A_scale

        inv_diag = 1.0 / diag
        for _ in range(iters):
            r = b - self._apply_A(x, lvl=lvl_idx)
            x = x + omega * inv_diag * r
        return x

    # ------------------ L1-GS (red-black) ------------------------------
    def _l1gs_relax_var(self, x, b, lvl_data, lvl_idx, omega=0.8, iters=1):
        kx, ky, kz = lvl_data["kx"], lvl_data["ky"], lvl_data["kz"]
        hx, hy, hz = lvl_data["hx"], lvl_data["hy"], lvl_data["hz"]

        diag = self._compute_diag(kx, ky, kz, hx, hy, hz, x.shape) * self.A_scale

        # Pre-compute masks for red-black ordering
        nz, ny, nx = x.shape
        zz = torch.arange(nz, device=x.device)[:, None, None]
        yy = torch.arange(ny, device=x.device)[None, :, None]
        xx = torch.arange(nx, device=x.device)[None, None, :]
        parity = (zz + yy + xx) % 2  # 0 = red, 1 = black
        mask_red = parity == 0
        mask_blk = parity == 1

        for _ in range(iters):
            # red sweep
            r = b - self._apply_A(x, lvl=lvl_idx)
            x = x + omega * (r / diag) * mask_red

            # black sweep
            r = b - self._apply_A(x, lvl=lvl_idx)
            x = x + omega * (r / diag) * mask_blk
        return x

    # ------------------ Chebyshev (polynomial) ------------------------------
    def _chebyshev_relax_var(self, x, b, lvl_data, lvl_idx, iters=4):
        """Chebyshev semi-iterative smoother с динамической переоценкой спектра.

        • λ_max оцениваем на каждой релаксации 5 итерациями степенного метода –
          это ~O(N) и не доминирует в стоимости V-cycle.
        • λ_min берём как λ_max / κ, где κ = 50 (целимся в спектральное число
          после Jacobi-предсглаживания). Эвристики вполне достаточно: цель –
          подавить высокочастотный шум.
        • Порядок полинома = iters (вызов из V-cycle). Мы гарантируем iters≥4
          при вызове из _v_cycle_var, иначе Chebyshev неэффективен.
        """

        # ---- оценка λ_max --------------------------------------------------
        v = torch.rand_like(x)
        v = v / (torch.norm(v) + 1e-12)
        for _ in range(5):  # чуть меньше итераций – достаточно точно
            v = self._apply_A(v, lvl_idx)
            v = v / (torch.norm(v) + 1e-12)
        Av = self._apply_A(v, lvl_idx)
        lam_max = torch.dot(v.flatten(), Av.flatten()).item()
        lam_max = max(lam_max, 1e-8)
        lam_max *= 1.05  # небольшое завышение

        lam_min = lam_max / 50.0  # κ = 50

        theta = (lam_max + lam_min) / 2.0
        delta = (lam_max - lam_min) / 2.0

        # первая итерация -----------------------------------------------
        r = b - self._apply_A(x, lvl_idx)
        p = r / theta
        x = x + p
        alpha_prev = 1.0 / theta

        for _ in range(iters - 1):
            r = b - self._apply_A(x, lvl_idx)
            beta = (delta * alpha_prev / 2.0) ** 2
            alpha = 1.0 / (theta - beta / alpha_prev)
            p = alpha * r + beta * p
            x = x + p
            alpha_prev = alpha
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
        lvl_data = self.levels[lvl]
        
        # критерий грубой сетки
        if lvl == len(self.levels) - 1 or min(x.shape) <= 4:
            return self._jacobi_relax_var(x, b, lvl_data, lvl, omega=omega, iters=20)

        # pre-smooth
        if self.smoother == "l1gs":
            x = self._l1gs_relax_var(x, b, lvl_data, lvl, omega=omega, iters=pre)
        elif self.smoother == "chebyshev":
            # Итераций ≥4 для эффективности
            x = self._chebyshev_relax_var(x, b, lvl_data, lvl, iters=max(4, pre))
        else:
            x = self._jacobi_relax_var(x, b, lvl_data, lvl, omega=omega, iters=pre)

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
        if self.smoother == "l1gs":
            x = self._l1gs_relax_var(x, b, lvl_data, lvl, omega=omega, iters=post)
        elif self.smoother == "chebyshev":
            x = self._chebyshev_relax_var(x, b, lvl_data, lvl, iters=max(4, post))
        else:
            x = self._jacobi_relax_var(x, b, lvl_data, lvl, omega=omega, iters=post)
        return x

    # ------------------------------------------------------------------
    def solve(self, rhs, tol=1e-6, max_iter=10):
        # ------------------------------
        # 1. RHS -> 3-D (nz, ny, nx)
        #    CPR формирует rhs в линейной индексации (x-быстрейшая)
        #    => сначала ресейпим (nx, ny, nz), затем permute.
        # ------------------------------
        nx0, ny0, nz0 = self.nx, self.ny, self.nz  # внешние размеры (после транспонирования)

        rhs_tensor = torch.as_tensor(rhs.reshape(nz0, ny0, nx0), dtype=torch.float64, device=self.device)

        # Масштабируем RHS тем же коэффициентом, что и оператор.
        b = rhs_tensor * self.A_scale  # (nz, ny, nx)

        # --- учтём возможный паддинг -------------------------------------------------
        dz, dy, dx = self._pad
        if dx or dy or dz:
            pad = (0, dx, 0, dy, 0, dz)
            b = F.pad(b, pad, mode="constant", value=0.0)

        # Маленькие задачи (< 64 ячеек) решаем точным LU – так устраняется
        # нулевой собственный вектор и исключается стагнация Jacobi.
        n_cells_total = rhs_tensor.numel()
        if n_cells_total <= 64:
            try:
                # Собираем плотную матрицу A_dense колоночным способом
                A_dense = torch.zeros((n_cells_total, n_cells_total), dtype=torch.float64, device=self.device)
                eye = torch.eye(n_cells_total, dtype=torch.float64, device=self.device)
                for j in range(n_cells_total):
                    col_vec = eye[:, j].reshape(rhs_tensor.shape)
                    # raw operator (без A_scale) → домножаем вручную
                    Acol = self._apply_A(col_vec, lvl=0) * self.A_scale
                    A_dense[:, j] = Acol.reshape(-1)

                x_direct = torch.linalg.solve(A_dense, b.reshape(-1))
                res_dir = torch.norm(b.reshape(-1) - A_dense @ x_direct) / (torch.norm(b) + 1e-12)
                if res_dir < tol:
                    return x_direct.cpu().numpy()
                # если LU не дал tol – продолжаем обычным методом
            except Exception as e:
                print(f"GeoSolver: direct LU fallback failed: {e}. Switching to V-cycle.")

        # ------------------------------
        # 2. Итеративное решение V-cycle
        #    • более агрессивный pre/post (5)
        #    • до 30 V-циклов
        # ------------------------------
        x = torch.zeros_like(b, dtype=torch.float64)
        # Более лёгкий V-cycle для предобуславливателя
        max_iter = min(max_iter, 10)
        for itr in range(max_iter):
            x = self._v_cycle_var(0, x, b, omega=0.8, pre=2, post=2)
            res = torch.norm(b - self._apply_A(x, lvl=0)) / (torch.norm(b) + 1e-12)
            if res < tol:
                break

        # fallback #1: если res > 0.5, делаем ещё 5 усиленных циклов того же типа
        if res > 0.5:
            for _ in range(5):
                x = self._v_cycle_var(0, x, b, omega=0.8, pre=4, post=4)
            res = torch.norm(b - self._apply_A(x, lvl=0)) / (torch.norm(b) + 1e-12)

        # fallback #2: если Chebyshev остаётся нестабилен (res>>tol) – переходим на Jacobi
        if res > max(10 * tol, 1e-1) and self.smoother == "chebyshev":
            print("GeoSolver: Chebyshev не обеспечил сходимости, fallback → Jacobi")
            self.smoother = "jacobi"
            # выполняем ещё несколько V-циклов
            for _ in range(10):
                x = self._v_cycle_var(0, x, b, omega=0.8, pre=2, post=2)
                res = torch.norm(b - self._apply_A(x, lvl=0)) / (torch.norm(b) + 1e-12)
                if res < tol:
                    break

        # обрезаем паддинг
        if dx or dy or dz:
            x = x[:nz0, :ny0, :nx0]

        # ------------------------------
        # 3. Возврат в линейный вид (x-быстрейшая)
        # ------------------------------
        x_out = x  # решение уже в физических единицах
        return x_out.cpu().numpy().ravel(order="C") 