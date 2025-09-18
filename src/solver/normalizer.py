import math
import torch
import os

class Normalizer:
    """Единый слой безразмеризации всех переменных пласта.

    Сейчас поддерживает давление и до двух насыщенностей (Sw, опц. Sg).
    Алгоритм масштабов:
      • давление масштабируется на p_scale (по умолчанию 1 МПа);
      • насыщенности уже безразмерны, scale = 1;
    В будущем можно расширить `self.s_scales` для T, концентраций и пр.
    """

    def __init__(self, reservoir, fluid):
        # ---------------- давление -----------------------------------
        # Оценка характерного давления: max(ρ g H, 1/ct, 1 бар)
        g = 9.80665  # м/с²
        rho_w = getattr(fluid, "rho_water_ref", 1000.0)
        _, _, nz = reservoir.dimensions
        _, _, dz = reservoir.grid_size
        H = nz * dz
        hydro_head = rho_w * g * H  # Па

        ct_w = float(getattr(fluid, "water_compressibility", 1e-9))
        ct_o = float(getattr(fluid, "oil_compressibility", 1e-9))
        ct = max(ct_w, ct_o, 1e-9)
        compress_based = 1.0 / ct

        p_scale = max(hydro_head, compress_based, 1e5)  # не менее 1 бар
        decade = 10 ** int(math.floor(math.log10(p_scale)))
        p_scale = round(p_scale / decade) * decade

        # Позволяем переопределить шкалу давления через переменную окружения
        #   OIL_P_SCALE (в паскалях) или через аргумент конфигурации
        p_scale_env = os.environ.get("OIL_P_SCALE")
        if p_scale_env is not None:
            try:
                p_scale = float(p_scale_env)
            except ValueError:
                print(f"[Normalizer] Некорректное OIL_P_SCALE='{p_scale_env}', игнорирую")
        else:
            # РЕКОМЕНДУЕМЫЙ ДИАПАЗОН для пластовых задач: 5–20 МПа
            # Слишком большой p_scale (например, 1/ct=1e9 Па) убивает численку.
            lo, hi = 5e6, 2e7  # Па
            if p_scale < lo or p_scale > hi:
                print(f"[Normalizer] p_scale={p_scale:.3e} Па → приводим к [{lo:.1e}, {hi:.1e}]")
            p_scale = min(max(p_scale, lo), hi)
        # Если пользователь явно не задал шкалу – используем динамическое значение
        self.p_scale = p_scale
        self.inv_p_scale = 1.0 / self.p_scale

        # ---------------- насыщенности --------------------------------
        has_gas = hasattr(fluid, "s_g")
        # Базово насыщенности безразмерны, но для параметризации y требуется согласованный масштаб:
        # s_scale_y ≈ 1 / median(ds/dy) для воды, где ds/dy=(1−Swc−Sor)·σ(y)(1−σ(y))
        s_scale_sw = 1.0
        try:
            if hasattr(fluid, "s_w") and fluid.s_w is not None:
                sw = fluid.s_w.view(-1).detach()
                swc = float(getattr(fluid, "sw_cr", 0.0))
                sor = float(getattr(fluid, "so_r", 0.0))
                denom = max(1e-12, 1.0 - swc - sor)
                sigma = ((sw - swc) / denom).clamp(0.0, 1.0)
                ds_dy = denom * (sigma * (1.0 - sigma))
                med = float(ds_dy.median().item()) if ds_dy.numel() > 0 else 0.0
                if med > 1e-8:
                    s_scale_sw = 1.0 / med
        except Exception:
            s_scale_sw = 1.0
        self.s_scales = ([s_scale_sw, 1.0] if has_gas else [s_scale_sw])
        self.inv_s_scales = [1.0 / s for s in self.s_scales]

        # Обобщённые массивы коэффициентов (phys → hat, hat → phys)
        self.scale = [self.inv_p_scale] + self.inv_s_scales  # умножать, чтобы получить hat
        self.inv_scale = [self.p_scale] + self.s_scales      # умножать, чтобы вернуться

        # Кэшируем размер ячейки
        nx, ny, nz = reservoir.dimensions
        self.n_cells = nx * ny * nz

        print(f"Normalizer: p_scale={self.p_scale:.3e} Па (~{self.p_scale/1e6:.2f} МПа); s_scales={self.s_scales}")
    # ------------------------------------------------------------------
    # Scalar helpers
    # ------------------------------------------------------------------
    def p_to_hat(self, p_phys: torch.Tensor) -> torch.Tensor:
        return p_phys * self.inv_p_scale

    def p_from_hat(self, p_hat: torch.Tensor) -> torch.Tensor:
        return p_hat * self.p_scale

    # ------------------------------------------------------------------
    # Vector helpers
    # ------------------------------------------------------------------
    def scale_vec(self, vec_phys: torch.Tensor) -> torch.Tensor:
        """Преобразует вектор физических переменных → безразмерный.
        Поддерживает вектор размера 2N или 3N (P + S).
        """
        n = self.n_cells
        v_hat = vec_phys.clone()
        vars_per_cell = v_hat.numel() // n

        # Давление
        v_hat[:n] *= self.inv_p_scale

        # Насыщенности (кол-во = vars_per_cell-1)
        n_sats = max(vars_per_cell - 1, 0)
        for idx in range(n_sats):
            start = n + idx * n
            end = start + n
            v_hat[start:end] *= self.inv_s_scales[idx]

        return v_hat

    def unscale_vec(self, vec_hat: torch.Tensor) -> torch.Tensor:
        """Обратное преобразование: безразмерный вектор → физический."""
        n = self.n_cells
        v = vec_hat.clone()
        vars_per_cell = v.numel() // n

        # Давление
        v[:n] *= self.p_scale

        # Насыщенности
        n_sats = max(vars_per_cell - 1, 0)
        for idx in range(n_sats):
            start = n + idx * n
            end = start + n
            v[start:end] *= self.s_scales[idx]

        return v

    # ------------------------------------------------------------------
    # Утилита: перевод давления в МПа для логов/постпроцессинга
    # ------------------------------------------------------------------
    def to_mpa_vec(self, vec_pa: torch.Tensor) -> torch.Tensor:
        """Возвращает копию вектора, где давление (первые N элементов)
        переведено из Па в МПа (×1e-6). Полезно для вывода и пост-анализа.
        """
        n = self.n_cells
        out = vec_pa.clone()
        out[:n] = vec_pa[:n] / 1e6
        return out

    # Aliases для обратной совместимости
    scale = scale_vec
    unscale = unscale_vec

# ----------------------------------------------------------------------
# Обратная совместимость: старое имя VariableScaler
# ----------------------------------------------------------------------
VariableScaler = Normalizer 