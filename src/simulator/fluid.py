import torch
import numpy as np

class Fluid:
    """
    Класс для моделирования свойств флюидов (нефть и вода).
    """
    def __init__(self, config, reservoir, device=None):
        """
        Инициализация флюидов по конфигурации.
        
        Args:
            config: Словарь с параметрами флюидов
            reservoir: Объект пласта
            device: Устройство для вычислений (CPU/GPU)
        """
        self.device = device if device is not None else torch.device('cpu')
        
        # Размеры и форма тензоров
        self.dimensions = reservoir.dimensions
        nx, ny, nz = self.dimensions
        
        # Начальные значения
        initial_pressure = config.get('pressure', 20.0) * 1e6  # МПа -> Па
        initial_sw = config.get('s_w', 0.2)
        initial_sg = config.get('s_g', 0.0)
        if initial_sg < 0 or initial_sg > 1 - initial_sw:
            raise ValueError("s_g должна быть в диапазоне [0, 1 - s_w]")
        
        # ------------------------------------------------------------------
        # 1. Свойства флюидов – постоянные по умолчанию
        # ------------------------------------------------------------------
        self.mu_oil   = float(config.get('mu_oil', 1.0))   * 1e-3  # сП → Па·с
        self.mu_water = float(config.get('mu_water', 0.5)) * 1e-3  # сП → Па·с
        self.mu_gas   = float(config.get('mu_gas', 0.05)) * 1e-3  # сП → Па·с
        
        # ------------------------------------------------------------------
        # 2. PVT-таблицы (опционально)
        # ------------------------------------------------------------------
        self._use_pvt = False
        pvt_cfg = config.get('pvt', None)
        if pvt_cfg is not None:
            try:
                # Давление в МПа → Па
                self._p_grid = torch.tensor(pvt_cfg['pressure'], dtype=torch.float32) * 1e6
                # Таблицы свойств (приводим единицы):
                # Плотности (кг/м3)
                self._rho_o_table = torch.tensor(pvt_cfg.get('rho_oil', []), dtype=torch.float32)
                self._rho_w_table = torch.tensor(pvt_cfg.get('rho_water', []), dtype=torch.float32)
                self._rho_g_table = torch.tensor(pvt_cfg.get('rho_gas', []), dtype=torch.float32)
                # Вязкости (cP → Pa·s)
                self._mu_o_table  = torch.tensor(pvt_cfg.get('mu_oil', []), dtype=torch.float32)  * 1e-3
                self._mu_w_table  = torch.tensor(pvt_cfg.get('mu_water', []), dtype=torch.float32) * 1e-3
                self._mu_g_table  = torch.tensor(pvt_cfg.get('mu_gas', []), dtype=torch.float32)  * 1e-3
                # Таблицы PVT – новые поля ---------------------------------
                self._bo_table   = torch.tensor(pvt_cfg.get('bo', []), dtype=torch.float32)
                self._bg_table   = torch.tensor(pvt_cfg.get('bg', []), dtype=torch.float32)
                self._bw_table   = torch.tensor(pvt_cfg.get('bw', []), dtype=torch.float32)
                self._rs_table   = torch.tensor(pvt_cfg.get('rs', []), dtype=torch.float32)
                self._rv_table   = torch.tensor(pvt_cfg.get('rv', []), dtype=torch.float32)

                # Проверка длины
                n_p = self._p_grid.numel()
                assert all(tbl.numel() == n_p for tbl in (
                    self._rho_o_table, self._rho_w_table, self._rho_g_table,
                    self._mu_o_table,  self._mu_w_table,  self._mu_g_table,
                    self._bo_table,    self._bg_table,    self._bw_table,
                    self._rs_table,    self._rv_table)), "Все PVT-таблицы должны иметь одинаковую длину"

                # Убедимся, что сетка давления отсортирована по возрастанию
                if not torch.all(self._p_grid[1:] >= self._p_grid[:-1]):
                    raise ValueError("pvt.pressure должен быть отсортирован по возрастанию")

                # По умолчанию храним таблицы на CPU; при вызове перенесём на нужное устройство
                self._use_pvt = True
                print("[Fluid] PVT-таблицы загружены (", n_p, "точек)")
            except Exception as e:
                print(f"[WARN] Ошибка при чтении PVT-таблиц: {e}. Используем константы.")
        
        # Плотности при стандартных условиях (surface) используем как ref
        self.rho_o_sc = float(config.get('rho_o_sc', 850.0))
        self.rho_w_sc = float(config.get('rho_w_sc', 1000.0))
        self.rho_g_sc = float(config.get('rho_g_sc', 150.0))
        
        # Алиасы для обратной совместимости со старым кодом
        self.rho_oil_ref   = self.rho_o_sc
        self.rho_water_ref = self.rho_w_sc
        self.rho_gas_ref   = self.rho_g_sc
        
        # Сжимаемость (1/Па)
        self.oil_compressibility   = float(config.get('c_oil', 1e-5))   / 1e6  # 1/Па
        self.water_compressibility = float(config.get('c_water', 1e-5)) / 1e6
        self.gas_compressibility   = float(config.get('c_gas', 3e-4)) / 1e6
        self.rock_compressibility  = float(config.get('c_rock', 1e-5))  / 1e6
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: правильное опорное давление для сжимаемости
        self.pressure_ref = getattr(reservoir, 'pressure_ref', 1e5)
        print(f"🔧 Опорное давление для плотности: {self.pressure_ref:.0f} Па ({self.pressure_ref/1e6:.1f} МПа)")
        
        # Совокупная сжимаемость флюида (используется в IMPES)
        total_c = (self.oil_compressibility + self.water_compressibility + self.gas_compressibility + self.rock_compressibility) / 2
        self.cf = torch.full(self.dimensions, total_c, device=self.device)
        
        # Параметры модели относительной проницаемости
        rp_cfg = config.get('relative_permeability', {})
        self.nw    = rp_cfg.get('nw', 2)           # Показатель Кори для воды
        self.no    = rp_cfg.get('no', 2)           # Показатель Кори для нефти
        self.ng    = rp_cfg.get('ng', 2)           # Показатель Кори для газа
        self.sw_cr = rp_cfg.get('sw_cr', 0.2)      # Связанная водонасыщенность
        self.so_r  = rp_cfg.get('so_r', 0.2)       # Остаточная нефтенасыщенность
        
        # Инициализация полей
        self.pressure = torch.full(self.dimensions, initial_pressure, device=self.device)
        self.s_w = torch.full(self.dimensions, initial_sw, device=self.device)
        self.s_g = torch.full(self.dimensions, initial_sg, device=self.device)
        self.s_o = 1.0 - self.s_w - self.s_g
        self.prev_pressure = self.pressure.clone()
        self.prev_sw = self.s_w.clone()
        self.prev_sg = self.s_g.clone()
        
        # Сохраняем предыдущее состояние для неявных расчетов
        self.prev_water_mass = None
        self.prev_oil_mass = None
        
        # Поддержка как старого, так и нового формата
        if 'capillary_pressure' in config:
            pc_params = config['capillary_pressure']
            self.pc_scale = pc_params.get('pc_scale', 0.0)
            self.pc_exponent = pc_params.get('pc_exponent', 1.5)
            self.pc_threshold = pc_params.get('pc_threshold', 0.01)
        else:
            self.pc_scale = config.get('pc_scale', 0.0)
            self.pc_exponent = config.get('pc_exponent', 1.5)
            self.pc_threshold = config.get('pc_threshold', 0.01)
        
        # Выводим информацию об инициализации
        print("Инициализация флюидов и начальных условий...")
        print(f"  Начальное давление: {initial_pressure/1e6:.2f} МПа")
        print(f"  Начальная водонасыщенность: {initial_sw}")
        print(f"  Начальная газонасыщенность:  {initial_sg}")
        print(f"  Вязкость нефти/воды: {self.mu_oil*1e3:.1f}/{self.mu_water*1e3:.1f} сП")
        print(f"  Вязкость газа:       {self.mu_gas*1e3:.2f} сП")
        print(f"  Плотность нефти/воды: {self.rho_oil_ref}/{self.rho_water_ref} кг/м^3")
        print(f"  Плотность газа:        {self.rho_gas_ref} кг/m^3")
        print(f"  Сжимаемость: {self.oil_compressibility*1e6:.1e} 1/Па")
        print(f"  Капиллярное давление: {self.pc_scale/1e6:.2e} МПа, показатель {self.pc_exponent}")
        print(f"  Связанная водонасыщенность: {self.sw_cr}, остаточная нефтенасыщенность: {self.so_r}")
        print(f"  Тензоры флюидов размещены на: {self.device}")

        self.pbubble = float(config.get('pbubble', 20.0)) * 1e6  # МПа → Па
        self.rs_bubble = float(config.get('rs_bubble', 100.0))   # (m³ газа)|(m³ нефти) – условные ед.

    # Свойства для совместимости со старым кодом IMPES
    @property
    def rho_w(self):
        """Плотность воды при текущем давлении"""
        return self.calc_water_density(self.pressure)
        
    @property
    def rho_o(self):
        """Плотность нефти при текущем давлении"""
        return self.calc_oil_density(self.pressure)
        
    @property
    def mu_w(self):
        """Вязкость воды (альтернативное имя)"""
        return self.mu_water
        
    @property
    def mu_o(self):
        """Вязкость нефти (альтернативное имя)"""
        return self.mu_oil

    @property
    def rho_g(self):
        """Плотность газа при текущем давлении"""
        return self.calc_gas_density(self.pressure)

    def _get_normalized_saturation(self, s_w):
        """
        Вычисляет нормализованную водонасыщенность с мягкими градиентами.
        """
        # ИСПРАВЛЕНО: более мягкий переход для стабильных градиентов
        eps = 0.02  # более мягкий переход чем 1e-1

        # Нормализуем в исходный диапазон [0,1]
        s_norm_raw = (s_w - self.sw_cr) / (1 - self.sw_cr - self.so_r + 1e-10)

        # ИСПРАВЛЕНО: используем более стабильную сигмоидальную функцию
        # Ограничиваем входные значения для избежания overflow
        sigmoid_input = torch.clamp((s_norm_raw - 0.5) / eps, -10.0, 10.0)
        s_norm = torch.sigmoid(sigmoid_input)

        return s_norm

    def get_rel_perms(self, s_w):
        """
        Вычисляет относительные фазовые проницаемости для воды и нефти по модели Кори.
        :param s_w: Тензор текущей водонасыщенности.
        :return: (kro, krw) - кортеж с тензорами ОФП.
        """
        kro = self.calc_oil_kr(s_w)
        krw = self.calc_water_kr(s_w)
        
        return kro, krw

    def get_rel_perms_derivatives(self, s_w):
        """
        Вычисляет производные ОФП по водонасыщенности.
        :param s_w: Тензор текущей водонасыщенности.
        :return: (dkrw_dsw, dkro_dsw) - кортеж с производными.
        """
        s_norm = self._get_normalized_saturation(s_w)
        dsw_norm_dsw = 1 / (1 - self.sw_cr - self.so_r)
        
        # d(krw)/d(sw) = d(krw)/d(s_norm) * d(s_norm)/d(sw)
        # d(krw)/d(s_norm) = nw * s_norm^(nw-1)
        dkrw_dsw = self.nw * (s_norm ** (self.nw - 1)) * dsw_norm_dsw
        
        # d(kro)/d(sw) = d(kro)/d(s_norm) * d(s_norm)/d(sw)
        # d(kro)/d(s_norm) = -no * (1-s_norm)^(no-1)
        dkro_dsw = -self.no * ((1 - s_norm) ** (self.no - 1)) * dsw_norm_dsw
        
        # Обработка особых случаев на границах
        dkrw_dsw = torch.where(s_norm <= 0, torch.zeros_like(dkrw_dsw), dkrw_dsw)
        dkro_dsw = torch.where(s_norm >= 1, torch.zeros_like(dkro_dsw), dkro_dsw)
        
        return dkro_dsw, dkrw_dsw

    def get_capillary_pressure(self, s_w):
        """
        Вычисляет капиллярное давление по простой степенной модели.
        :param s_w: Тензор текущей водонасыщенности.
        :return: Тензор капиллярного давления (в Па).
        """
        if self.pc_scale == 0.0:
            return torch.zeros_like(s_w)
            
        s_norm = self._get_normalized_saturation(s_w)
        
        # Простая степенная модель Pc = scale * (1-s_norm)^-exponent
        # Добавляем эпсилон для стабильности, если s_norm = 1
        pc = self.pc_scale * (1.0 - s_norm + 1e-6) ** (-self.pc_exponent)
        return pc

    def get_capillary_pressure_derivative(self, s_w):
        """
        Вычисляет производную капиллярного давления по водонасыщенности.
        :param s_w: Тензор текущей водонасыщенности.
        :return: Тензор d(Pc)/d(Sw).
        """
        if self.pc_scale == 0.0:
            return torch.zeros_like(s_w)
            
        s_norm = self._get_normalized_saturation(s_w)
        dsw_norm_dsw = 1 / (1 - self.sw_cr - self.so_r)

        # d(Pc)/d(sw) = d(Pc)/d(s_norm) * d(s_norm)/d(sw)
        # d(Pc)/d(s_norm) = pc_scale * (-exponent) * (1-s_norm)^(-exponent-1) * (-1)
        dpc_dsn = self.pc_scale * self.pc_exponent * (1.0 - s_norm + 1e-6) ** (-self.pc_exponent - 1)
        
        dpc_dsw = dpc_dsn * dsw_norm_dsw
        dpc_dsw = torch.where(s_norm >= 1, torch.zeros_like(dpc_dsw), dpc_dsw)
        return dpc_dsw

    def calc_water_density(self, pressure):
        """Плотность воды ρw(P)."""
        if self._use_pvt and self._bw_table.numel() > 0:
            Bw = self.calc_bw(pressure)
            return self.rho_w_sc / (Bw + 1e-12)
        return self.rho_water_ref * (1.0 + self.water_compressibility * (pressure - self.pressure_ref))

    def calc_oil_density(self, pressure):
        """Плотность нефти ρo(P)."""
        if self._use_pvt and self._bo_table.numel() > 0:
            Bo = self.calc_bo(pressure)
            return self.rho_o_sc / (Bo + 1e-12)
        return self.rho_oil_ref * (1.0 + self.oil_compressibility * (pressure - self.pressure_ref))

    def calc_water_kr(self, s_w):
        """
        Вычисляет относительную проницаемость воды по модели Кори.
        
        Args:
            s_w: Тензор водонасыщенности
            
        Returns:
            Тензор относительной проницаемости воды
        """
        s_norm = self._get_normalized_saturation(s_w)
        return s_norm**self.nw

    def calc_oil_kr(self, s_w):
        """
        Вычисляет относительную проницаемость нефти по модели Кори.
        
        Args:
            s_w: Тензор водонасыщенности
            
        Returns:
            Тензор относительной проницаемости нефти
        """
        s_norm = self._get_normalized_saturation(s_w)
        return (1 - s_norm)**self.no

    def calc_dkrw_dsw(self, s_w):
        """
        Вычисляет производную относительной проницаемости воды по водонасыщенности.
        
        Args:
            s_w: Тензор водонасыщенности
            
        Returns:
            Тензор производной относительной проницаемости воды
        """
        s_norm = self._get_normalized_saturation(s_w)
        normalized_range = 1.0 - self.sw_cr - self.so_r + 1e-10
        
        # ИСПРАВЛЕНО: используем torch.where вместо маскирования для сохранения градиентов
        # Проверяем, находится ли насыщенность в допустимом диапазоне
        in_range = (s_w >= self.sw_cr) & (s_w <= 1.0 - self.so_r)
        
        # Производная dkrw/dsw = dkrw/ds_norm * ds_norm/dsw
        # Производная сигмоидальной нормализации
        eps = 0.02  # должно совпадать с _get_normalized_saturation
        s_norm_raw = (s_w - self.sw_cr) / normalized_range
        sigmoid_input = torch.clamp((s_norm_raw - 0.5) / eps, -10.0, 10.0)
        dsigmoid_dx = torch.sigmoid(sigmoid_input) * (1 - torch.sigmoid(sigmoid_input)) / eps
        ds_norm_dsw = dsigmoid_dx / normalized_range
        
        # Полная производная
        dkrw_ds_norm = self.nw * torch.clamp(s_norm, 1e-8, 1-1e-8)**(self.nw - 1)
        result_full = dkrw_ds_norm * ds_norm_dsw
        
        # Применяем ограничение области без нарушения градиентов
        result = torch.where(in_range, result_full, torch.zeros_like(result_full))
        
        return result

    def calc_dkro_dsw(self, s_w):
        """
        Вычисляет производную относительной проницаемости нефти по водонасыщенности.
        
        Args:
            s_w: Тензор водонасыщенности
            
        Returns:
            Тензор производной относительной проницаемости нефти
        """
        s_norm = self._get_normalized_saturation(s_w)
        normalized_range = 1.0 - self.sw_cr - self.so_r + 1e-10
        
        # ИСПРАВЛЕНО: используем torch.where вместо маскирования для сохранения градиентов
        # Проверяем, находится ли насыщенность в допустимом диапазоне
        in_range = (s_w >= self.sw_cr) & (s_w <= 1.0 - self.so_r)
        
        # Производная dkro/dsw = dkro/ds_norm * ds_norm/dsw
        # Производная сигмоидальной нормализации
        eps = 0.02  # должно совпадать с _get_normalized_saturation
        s_norm_raw = (s_w - self.sw_cr) / normalized_range
        sigmoid_input = torch.clamp((s_norm_raw - 0.5) / eps, -10.0, 10.0)
        dsigmoid_dx = torch.sigmoid(sigmoid_input) * (1 - torch.sigmoid(sigmoid_input)) / eps
        ds_norm_dsw = dsigmoid_dx / normalized_range
        
        # Полная производная
        dkro_ds_norm = -self.no * torch.clamp(1 - s_norm, 1e-8, 1-1e-8)**(self.no - 1)
        result_full = dkro_ds_norm * ds_norm_dsw
        
        # Применяем ограничение области без нарушения градиентов
        result = torch.where(in_range, result_full, torch.zeros_like(result_full))
        
        return result

    # ---- Вязкости (Pa·s) ----
    def calc_water_viscosity(self, pressure):
        if self._use_pvt:
            return self._interp(pressure, self._p_grid, self._mu_w_table)
        return torch.full_like(pressure, self.mu_water)

    def calc_oil_viscosity(self, pressure):
        if self._use_pvt:
            return self._interp(pressure, self._p_grid, self._mu_o_table)
        return torch.full_like(pressure, self.mu_oil)

    # ---- Газовая фаза ----
    def calc_gas_density(self, pressure):
        """Плотность газа ρg(P)."""
        if self._use_pvt and self._bg_table.numel() > 0:
            Bg = self.calc_bg(pressure)
            return self.rho_g_sc / (Bg + 1e-12)
        return self.rho_gas_ref * (1.0 + self.gas_compressibility * (pressure - self.pressure_ref))

    def calc_gas_viscosity(self, pressure):
        """Вязкость газа μg(P)."""
        if self._use_pvt and hasattr(self, '_mu_g_table') and self._mu_g_table.numel() > 0:
            return self._interp(pressure, self._p_grid, self._mu_g_table)
        return torch.full_like(pressure, self.mu_gas)

    # ---- Алиасы для обратной совместимости со старым кодом ----
    # (симулятор обращается к этим именам)
    calc_capillary_pressure = get_capillary_pressure
    calc_dpc_dsw            = get_capillary_pressure_derivative

    # ------------------------------------------------------------------
    # Вспомогательная 1-D интерполяция (linear) на GPU/CPU
    # ------------------------------------------------------------------
    def _interp(self, p, p_grid, prop_grid):
        """Линейная интерполяция prop(p). p и сетки – torch.Tensor."""
        # Гарантируем одинаковое устройство
        p_grid = p_grid.to(p.device)
        prop_grid = prop_grid.to(p.device)

        p_flat = p.view(-1)
        idx_hi = torch.searchsorted(p_grid, p_flat, right=True)
        idx_hi = idx_hi.clamp(1, p_grid.numel() - 1)
        idx_lo = idx_hi - 1

        p_lo = p_grid[idx_lo]
        p_hi = p_grid[idx_hi]
        w = (p_flat - p_lo) / (p_hi - p_lo + 1e-12)
        prop = prop_grid[idx_lo] + w * (prop_grid[idx_hi] - prop_grid[idx_lo])
        return prop.view_as(p)

    # Для трёхфазного случая возвращаем krg дополнительно
    def get_rel_perms_three(self, s_w, s_g):
        """Возвращает (kro, krw, krg)."""
        kro = self.calc_oil_kr(s_w)
        krw = self.calc_water_kr(s_w)
        krg = self.calc_gas_kr(s_g)
        return kro, krw, krg

    def calc_gas_kr(self, s_g):
        """Относительная проницаемость газа (Corey)."""
        # Простая Corey: krg = Sg^ng
        return s_g ** self.ng

    def calc_rs(self, pressure):
        """Растворённый газовый фактор Rs(P).

        Простая линейная зависимость:
            P >= Pbubble   → Rs = Rs_bubble (насыщенная нефть)
            P <  Pbubble   → Rs линейно падает до 0 при P→0.
        Возвращает безразмерное отношение (объём газа при стандартных усл. / объём нефти).
        """
        pb = self.pbubble
        rs_b = self.rs_bubble
        return torch.where(pressure >= pb,
                           torch.full_like(pressure, rs_b),
                           rs_b * pressure / pb)

    # ---- PVT ------------------------------------------------------------
    def calc_bo(self, pressure):
        if self._use_pvt and self._bo_table.numel() > 0:
            return self._interp(pressure, self._p_grid, self._bo_table)
        return torch.ones_like(pressure)

    def calc_bg(self, pressure):
        if self._use_pvt and self._bg_table.numel() > 0:
            return self._interp(pressure, self._p_grid, self._bg_table)
        return torch.ones_like(pressure)

    def calc_bw(self, pressure):
        if self._use_pvt and self._bw_table.numel() > 0:
            return self._interp(pressure, self._p_grid, self._bw_table)
        return torch.ones_like(pressure)

    def calc_rs(self, pressure):
        # сначала попробуем табличное значение, иначе линейная модель ниже
        if self._use_pvt and self._rs_table.numel() > 0:
            return self._interp(pressure, self._p_grid, self._rs_table)
        return super().calc_rs(pressure)  # линейная базовая реализация

    def calc_rv(self, pressure):
        if self._use_pvt and self._rv_table.numel() > 0:
            return self._interp(pressure, self._p_grid, self._rv_table)
        return torch.zeros_like(pressure)

    # ---- плотности с учётом Bo/Bg/Bw -----------------------------------
    def calc_oil_density(self, pressure):
        if self._use_pvt and self._bo_table.numel() > 0:
            Bo = self.calc_bo(pressure)
            return self.rho_o_sc / (Bo + 1e-12)
        return super().calc_oil_density(pressure)

    def calc_water_density(self, pressure):
        if self._use_pvt and self._bw_table.numel() > 0:
            Bw = self.calc_bw(pressure)
            return self.rho_w_sc / (Bw + 1e-12)
        return super().calc_water_density(pressure)

    def calc_gas_density(self, pressure):
        if self._use_pvt and self._bg_table.numel() > 0:
            Bg = self.calc_bg(pressure)
            return self.rho_g_sc / (Bg + 1e-12)
        return super().calc_gas_density(pressure)
