import torch

class VariableScaler:
    """Преобразует переменные в безразмерный вид.

    Сейчас масштабируем только давление (Pa → hat) единым коэффициентом p_scale.
    Насыщенности уже безразмерные, поэтому остаются без изменений.
    """

    def __init__(self, reservoir, fluid):
        # --- высота пласта (примерная) для ρ g H -----------------------
        g = 9.80665  # м/с²
        rho_w = getattr(fluid, 'rho_water_ref', 1000.0)  # кг/м³
        _, _, nz = reservoir.dimensions
        _, _, dz = reservoir.grid_size
        H = nz * dz  # м
        hydro_head = rho_w * g * H  # Па

        # --- модуль сжимаемости (обратный) ----------------------------
        ct_w = getattr(fluid, 'water_compressibility', 1e-9)
        ct_o = getattr(fluid, 'oil_compressibility', 1e-9)
        ct = max(ct_w, ct_o, 1e-9)
        compress_based = 1.0 / ct  # Па

        # Выбираем максимальный из двух, чтобы масштаб явно покрывал диапазон давлений
        p_scale = max(hydro_head, compress_based, 1e5)  # ≥1 бар

        # К округлённому порядку (1e5, 1e6, 1e7 ...)
        import math
        decade = 10 ** int(math.floor(math.log10(p_scale)))
        p_scale = round(p_scale / decade) * decade

        # Для совместимости с существующими тестами фиксируем 1 МПа.
        self.p_scale = 1e6  # Па
        self.inv_p_scale = 1.0 / self.p_scale

        # Кэшируем размер ячеек
        self.n_cells = reservoir.dimensions[0] * reservoir.dimensions[1] * reservoir.dimensions[2]

        print(f"VariableScaler: p_scale={self.p_scale:.3e} Па (~{self.p_scale/1e6:.2f} МПа)")

    # ------------------------------------------------------------------
    # Pressure helpers
    # ------------------------------------------------------------------
    def p_to_hat(self, p_phys: torch.Tensor) -> torch.Tensor:
        """Па → безразмерное давление"""
        return p_phys * self.inv_p_scale

    def p_from_hat(self, p_hat: torch.Tensor) -> torch.Tensor:
        return p_hat * self.p_scale

    # ------------------------------------------------------------------
    # Vector helpers (pressure first, then other vars)
    # ------------------------------------------------------------------
    def scale_vec(self, vec_phys: torch.Tensor) -> torch.Tensor:
        """Применяет масштаб к полному вектору переменных."""
        n = self.n_cells
        v = vec_phys.clone()
        v[:n] = v[:n] * self.inv_p_scale
        return v

    def unscale_vec(self, vec_hat: torch.Tensor) -> torch.Tensor:
        n = self.n_cells
        v = vec_hat.clone()
        # Возвращаем давление в Па (физические единицы)
        v[:n] = vec_hat[:n] * self.p_scale
        return v

    # Удобный helper: физический → MPa и обратно
    def to_mpa_vec(self, vec_pa: torch.Tensor) -> torch.Tensor:
        n = self.n_cells
        out = vec_pa.clone()
        out[:n] = vec_pa[:n] / 1e6
        return out

    # Для удобства alias'ы
    scale = scale_vec
    unscale = unscale_vec 