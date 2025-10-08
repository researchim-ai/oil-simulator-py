import torch
from typing import Dict

def compute_cell_props(sim, x_hat: torch.Tensor, dt_sec: float) -> Dict[str, torch.Tensor]:
    """Return basic per-cell properties required by CPR-tail.

    Currently returns only porosity phi (with rock compressibility) and cell volume.
    Дополнено: возвращаем также фазовые мобильности λ_w/o/g, сжимаемости
    отдельных фаз и суммарную (c_t), а также базовые параметры, которые
    потребуются CPR-MS.

    Parameters
    ----------
    sim : Simulator
        Экземпляр симулятора, из которого извлекаем `reservoir`, `fluid`
        и потенциально `scaler`.
    x_hat : torch.Tensor
        Текущий вектор неизвестных *в физических единицах* (давление – Па,
        насыщенности – доли).  Внутри JFNK это уже "hat"-пространство, но
        сюда он передаётся без масштабирования, поэтому повторять
        нормализацию не нужно.
    dt_sec : float
        Текущий шаг времени в секундах.
    """
    reservoir = sim.reservoir
    fluid     = sim.fluid

    # ------------------------------------------------------------------
    # Распаковка переменных из x_hat
    # ------------------------------------------------------------------
    nx, ny, nz = reservoir.dimensions
    n_cells = nx * ny * nz

    # Давление – уже в Паскалях (VariableScaler применён заранее).
    p_vec = x_hat[:n_cells]

    # Насыщенности воды и, опционально, газа
    if x_hat.numel() == 3 * n_cells:
        sw_vec = x_hat[n_cells:2*n_cells]
        sg_vec = x_hat[2*n_cells:]
    else:
        sw_vec = x_hat[n_cells:]
        sg_vec = None

    s_w = sw_vec.view(nx, ny, nz)
    if sg_vec is not None:
        s_g = sg_vec.view(nx, ny, nz)
        s_o = 1.0 - s_w - s_g
    else:
        s_g = None
        s_o = 1.0 - s_w

    # ------------------------------------------------------------------
    # Пористость с учётом сжимаемости породы
    # ------------------------------------------------------------------
    phi0 = reservoir.porosity_ref
    c_r  = getattr(reservoir, 'rock_compressibility', 0.0)
    p_ref = getattr(reservoir, 'pressure_ref', 1e5)
    p     = p_vec.view_as(phi0)
    phi   = phi0 * (1.0 + c_r * (p - p_ref))

    # ------------------------------------------------------------------
    # Фазовые свойства
    # ------------------------------------------------------------------
    rho_w = fluid.calc_water_density(p)
    rho_o = fluid.calc_oil_density(p)
    rho_g = fluid.calc_gas_density(p) if s_g is not None else None

    mu_w = fluid.calc_water_viscosity(p)
    mu_o = fluid.calc_oil_viscosity(p)
    mu_g = fluid.calc_gas_viscosity(p) if s_g is not None else None

    # Относительные проницаемости
    if s_g is not None:
        kro, krw, krg = fluid.get_rel_perms_three(s_w, s_g)
    else:
        kro, krw = fluid.get_rel_perms(s_w)
        krg = None

    lam_w = krw / (mu_w + 1e-30)
    lam_o = kro / (mu_o + 1e-30)
    lam_g = (krg / (mu_g + 1e-30)) if s_g is not None else None

    # ------------------------------------------------------------------
    # Сжимаемости
    # ------------------------------------------------------------------
    c_w = torch.full((n_cells,), float(getattr(fluid, 'water_compressibility', 0.0)),
                     device=p_vec.device, dtype=p_vec.dtype)
    c_o = torch.full((n_cells,), float(getattr(fluid, 'oil_compressibility', 0.0)),
                     device=p_vec.device, dtype=p_vec.dtype)
    if s_g is not None:
        c_g_val = float(getattr(fluid, 'gas_compressibility', 0.0))
        c_g = torch.full((n_cells,), c_g_val, device=p_vec.device, dtype=p_vec.dtype)
    else:
        c_g = None

    # Взвешенная насыщенностями суммарная сжимаемость (для CPR coarse)
    sw_flat = s_w.reshape(-1)
    so_flat = s_o.reshape(-1)
    if s_g is not None:
        sg_flat = s_g.reshape(-1)
        c_t = sw_flat * c_w + so_flat * c_o + sg_flat * c_g + c_r
    else:
        c_t = sw_flat * c_w + so_flat * c_o + c_r

    # ------------------------------------------------------------------
    # Сборка выходного словаря
    # ------------------------------------------------------------------
    # cell_volume может быть 0-D тензором; берём скаляр float
    cell_vol = float(reservoir.cell_volume.item()) if torch.is_tensor(reservoir.cell_volume) else float(reservoir.cell_volume)

    # ------------------------------------------------------------------
    # Безопасный нижний порог шагу времени: если dt слишком мал (<dt_floor),
    # используем dt_floor, чтобы избежать деления на крошечный dt в CPR.
    # Значение можно переопределить через sim.sim_params['dt_floor_sec'].
    # ------------------------------------------------------------------
    dt_floor = float(sim.sim_params.get('dt_floor_sec', 300.0))  # 5 минут по умолчанию
    dt_eff   = max(float(dt_sec), dt_floor)

    # Суммарная мобильность (для TRUE-IMPES CPR)
    lam_t = lam_w + lam_o
    if lam_g is not None:
        lam_t = lam_t + lam_g

    props = {
        # Геометрия
        'phi': phi.reshape(-1),                               # (N,)
        'V': torch.full((n_cells,), cell_vol, device=p_vec.device, dtype=p_vec.dtype),
        'dt': torch.tensor(dt_eff, device=p_vec.device, dtype=p_vec.dtype),

        # Мобильности
        'lam_w': lam_w.reshape(-1),
        'lam_o': lam_o.reshape(-1),
        'lam_g': lam_g.reshape(-1) if lam_g is not None else None,
        'lam_t': lam_t.reshape(-1),  # NEW: total mobility для TRUE-IMPES

        # Вязкости (для вычисления ∂λ/∂S в TRUE-IMPES)
        'mu_w': mu_w.reshape(-1),
        'mu_o': mu_o.reshape(-1),
        'mu_g': mu_g.reshape(-1) if mu_g is not None else None,

        # Насыщенности (для вычисления производных ∂k_r/∂S)
        'sw': sw_flat,
        'so': so_flat,
        'sg': sg_flat if s_g is not None else None,

        # Сжимаемости
        'c_w': c_w,
        'c_o': c_o,
        'c_g': c_g,
        'c_t': c_t,

        # Плотности (пригодятся для гравитации)
        'rho_w': rho_w.reshape(-1),
        'rho_o': rho_o.reshape(-1),
        'rho_g': rho_g.reshape(-1) if rho_g is not None else None,
    }

    return props 