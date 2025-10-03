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
        # Поддерживаем оба варианта: 'initial_pressure' (Па) и устаревший 'pressure' (МПа)
        if 'initial_pressure' in config:
            initial_pressure = float(config['initial_pressure'])  # уже в Паскалях
        else:
            initial_pressure = float(config.get('pressure', 20.0)) * 1e6  # МПа → Па
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
        self._use_temp = False  # температурная сетка по умолчанию отключена
        pvt_cfg = config.get('pvt', None)
        if pvt_cfg is not None:
            try:
                # --- Сетка давления (МПа → Па) ---
                self._p_grid = torch.tensor(pvt_cfg['pressure'], dtype=torch.float32) * 1e6

                # --- Необязательная температурная сетка (°C) ---
                if 'temperature' in pvt_cfg:
                    self._t_grid = torch.tensor(pvt_cfg['temperature'], dtype=torch.float32)  # °C
                    if not torch.all(self._t_grid[1:] >= self._t_grid[:-1]):
                        raise ValueError("pvt.temperature должен быть отсортирован по возрастанию")
                    self._use_temp = True
                else:
                    self._t_grid = torch.tensor([float(config.get('temperature', 60.0))], dtype=torch.float32)
                    self._use_temp = False
                # Таблицы свойств (приводим единицы):
                # Плотности (кг/м3)
                self._rho_o_table = torch.tensor(pvt_cfg.get('rho_oil', []), dtype=torch.float32)
                self._rho_w_table = torch.tensor(pvt_cfg.get('rho_water', []), dtype=torch.float32)
                self._rho_g_table = torch.tensor(pvt_cfg.get('rho_gas', []), dtype=torch.float32)
                # Вязкости (cP → Pa·s)
                self._mu_o_table  = torch.tensor(pvt_cfg.get('mu_oil', []), dtype=torch.float32)  * 1e-3
                self._mu_w_table  = torch.tensor(pvt_cfg.get('mu_water', []), dtype=torch.float32) * 1e-3
                self._mu_g_table  = torch.tensor(pvt_cfg.get('mu_gas', []), dtype=torch.float32)  * 1e-3
                # Таблицы PVT (может быть 1-D или 2-D T×P)
                def to_tensor(name):
                    arr = pvt_cfg.get(name, [])
                    return torch.tensor(arr, dtype=torch.float32)

                self._bo_table = to_tensor('bo')
                self._bg_table = to_tensor('bg')
                self._bw_table = to_tensor('bw')
                self._rs_table = to_tensor('rs')
                self._rv_table = to_tensor('rv')

                # Проверка длины
                n_p = self._p_grid.numel()
                if self._use_temp:
                    n_t = self._t_grid.numel()
                    # Таблицы 2-D должны иметь форму (n_t, n_p)
                    def check_shape(t):
                        # Разрешаем: (n_t, n_p) или (n_p,) или пустой
                        return t.numel()==0 or (t.dim()==2 and t.shape==(n_t,n_p)) or (t.numel()==n_p)
                    assert all(check_shape(tbl) for tbl in (
                        self._bo_table, self._bg_table, self._bw_table,
                        self._rs_table, self._rv_table,
                        self._rho_o_table, self._rho_w_table, self._rho_g_table,
                        self._mu_o_table,  self._mu_w_table,  self._mu_g_table)), "PVT-таблица имеет неверную форму"
                else:
                    def _ok(tbl):
                        return tbl.numel() == 0 or tbl.numel() == n_p
                    assert all(_ok(tbl) for tbl in (
                        self._rho_o_table, self._rho_w_table, self._rho_g_table,
                        self._mu_o_table,  self._mu_w_table,  self._mu_g_table,
                        self._bo_table,    self._bg_table,    self._bw_table,
                        self._rs_table,    self._rv_table)), "Все непустые PVT-таблицы должны иметь длину pvt.pressure"

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
        
        # Единый опорный уровень давления (из пласта)
        self.pressure_ref = float(getattr(reservoir, 'pressure_ref', 1e5))
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
        # Ширина активной зоны у границ для устойчивых производных Corey
        self.rp_delta = float(rp_cfg.get('delta', 0.01))
        # Минимальная водная мобильность для выхода из «мертвой зоны» (конфигурируемая)
        self.krw_min = float(rp_cfg.get('krw_min', 1e-8))
        
        # Инициализация полей
        self.pressure = torch.full(self.dimensions, initial_pressure, device=self.device)
        self.s_w = torch.full(self.dimensions, initial_sw, device=self.device)
        self.s_g = torch.full(self.dimensions, initial_sg, device=self.device)
        self.s_o = 1.0 - self.s_w - self.s_g
        # --- Hysteresis state: максимальная достигнутая Sw/Sg (Land) ---
        self.sw_max = self.s_w.clone()
        self.sg_max = self.s_g.clone()
        self.prev_pressure = self.pressure.clone()
        self.prev_sw = self.s_w.clone()
        self.prev_sg = self.s_g.clone()
        
        # Сохраняем предыдущее состояние для неявных расчетов
        self.prev_water_mass = None
        self.prev_oil_mass = None
        
        if 'capillary_pressure' in config:
            pc_params = config['capillary_pressure']
            # --- oil–water ---
            self.pc_ow_scale    = pc_params.get('pc_ow_scale', pc_params.get('pc_scale', 0.0))
            self.pc_ow_exponent = pc_params.get('pc_ow_exponent', pc_params.get('pc_exponent', 1.5))
            # --- oil–gas (по умолчанию те же, что и для ow) ---
            self.pc_og_scale    = pc_params.get('pc_og_scale', self.pc_ow_scale)
            self.pc_og_exponent = pc_params.get('pc_og_exponent', self.pc_ow_exponent)
            self.pc_threshold   = pc_params.get('pc_threshold', 0.01)
        else:
            self.pc_ow_scale    = config.get('pc_scale', 0.0)
            self.pc_ow_exponent = config.get('pc_exponent', 1.5)
            self.pc_og_scale    = self.pc_ow_scale
            self.pc_og_exponent = self.pc_ow_exponent
            self.pc_threshold   = config.get('pc_threshold', 0.01)

        # Для обратной совместимости оставляем старые поля
        self.pc_scale    = self.pc_ow_scale
        self.pc_exponent = self.pc_ow_exponent
        
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

        # --- Температура пласта (°C) ---
        self.temperature = float(config.get('temperature', 60.0))
        self.rs_bubble = float(config.get('rs_bubble', 100.0))   # (m³ газа)|(m³ нефти) – условные ед.

    # ------------------------------------------------------------------
    # Hysteresis helper
    # ------------------------------------------------------------------
    def update_hysteresis(self):
        """Обновляет максимальные значения насыщенностей для Land-коррекции."""
        self.sw_max = torch.maximum(self.sw_max, self.s_w)
        if hasattr(self, 's_g'):
            self.sg_max = torch.maximum(self.sg_max, self.s_g)

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
        :return: (dkro_dsw, dkrw_dsw) - производные в порядке (нефть, вода).
        """
        dkro_dsw = self.calc_dkro_dsw(s_w)
        dkrw_dsw = self.calc_dkrw_dsw(s_w)
        return dkro_dsw, dkrw_dsw

    # ------------------------------------------------------------------
    # Capillary pressure – oil-water (existing)
    # ------------------------------------------------------------------
    def get_capillary_pressure(self, s_w):
        """
        Вычисляет капиллярное давление по простой степенной модели.
        :param s_w: Тензор текущей водонасыщенности.
        :return: Тензор капиллярного давления (в Па).
        """
        if self.pc_scale == 0.0:
            return torch.zeros_like(s_w)

        s_norm = self._get_normalized_saturation(s_w)

        # --- Drainage curve (baseline) ---------------------------------
        pc_drain = self.pc_scale * (1.0 - s_norm + 1e-6) ** (-self.pc_exponent)

        # --- Land hysteresis correction --------------------------------
        #   Pc_imb = Pc_drain * (1 - Sw_max)/(1 - Sw)
        land_factor = torch.clamp((1.0 - self.sw_max) / (1.0 - s_w + 1e-6), 0.0, 1.0)
        pc = pc_drain * land_factor
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

        # Drainage derivative (with negative sign)
        dpc_dsn = -self.pc_scale * self.pc_exponent * (1.0 - s_norm + 1e-6) ** (-self.pc_exponent - 1)
        dpc_dsw_drain = dpc_dsn * dsw_norm_dsw

        # Land factor and its derivative
        land_factor = torch.clamp((1.0 - self.sw_max) / (1.0 - s_w + 1e-6), 0.0, 1.0)
        dland_dsw = (1.0 - self.sw_max) / (1.0 - s_w + 1e-6) ** 2

        dpc_dsw = dpc_dsw_drain * land_factor + (self.pc_scale * (1.0 - s_norm + 1e-6) ** (-self.pc_exponent)) * dland_dsw
        dpc_dsw = torch.where(s_norm >= 1, torch.zeros_like(dpc_dsw), dpc_dsw)
        return dpc_dsw

    # ------------------------------------------------------------------
    # Capillary pressure – oil–gas (новое)
    # ------------------------------------------------------------------
    def get_capillary_pressure_og(self, s_g):
        """Pcₒᵍ(Sg) с Land-Killough гистерезисом."""
        if self.pc_og_scale == 0.0:
            return torch.zeros_like(s_g)

        # Нормализованная Sg (принимаем sg_cr=0)
        denom = 1.0 - self.sw_cr - self.so_r
        s_norm = torch.clamp(s_g / (denom + 1e-12), 0.0, 1.0)

        # Drainage
        pc_drain = self.pc_og_scale * (1.0 - s_norm + 1e-6) ** (-self.pc_og_exponent)

        # Land hysteresis (по газу)
        land_factor = torch.clamp((1.0 - self.sg_max) / (1.0 - s_g + 1e-6), 0.0, 1.0)
        pc = pc_drain * land_factor
        return pc

    def get_capillary_pressure_og_derivative(self, s_g):
        """dPcₒᵍ/dSg (≤0)."""
        if self.pc_og_scale == 0.0:
            return torch.zeros_like(s_g)

        denom = 1.0 - self.sw_cr - self.so_r
        s_norm = torch.clamp(s_g / (denom + 1e-12), 0.0, 1.0)
        dsg_norm_dsg = 1.0 / denom

        dpc_dsn = -self.pc_og_scale * self.pc_og_exponent * (1.0 - s_norm + 1e-6) ** (-self.pc_og_exponent - 1)
        dpc_dsg_drain = dpc_dsn * dsg_norm_dsg

        land_factor = torch.clamp((1.0 - self.sg_max) / (1.0 - s_g + 1e-6), 0.0, 1.0)
        dland_dsg = (1.0 - self.sg_max) / (1.0 - s_g + 1e-6) ** 2

        dpc_dsg = dpc_dsg_drain * land_factor + (self.pc_og_scale * (1.0 - s_norm + 1e-6) ** (-self.pc_og_exponent)) * dland_dsg
        dpc_dsg = torch.where(s_norm >= 1, torch.zeros_like(dpc_dsg), dpc_dsg)
        return dpc_dsg

    # Алиасы для совместимости
    calc_pc_ow  = get_capillary_pressure
    calc_pc_og  = get_capillary_pressure_og
    calc_dpc_dsw = get_capillary_pressure_derivative
    calc_dpc_dsg = get_capillary_pressure_og_derivative

    def calc_water_density(self, pressure):
        """Плотность воды ρw(P) с учётом Bw(P,T)."""
        if self._use_pvt and self._bw_table.numel() > 0:
            Bw = self.calc_bw(pressure)
            return self.rho_w_sc / (Bw + 1e-12)
        return self.rho_water_ref * (1.0 + self.water_compressibility * (pressure - self.pressure_ref))

    def calc_oil_density(self, pressure):
        """Плотность нефти ρo(P) с учётом Bo(P,T)."""
        if self._use_pvt and self._bo_table.numel() > 0:
            Bo = self.calc_bo(pressure)
            return self.rho_o_sc / (Bo + 1e-12)
        return self.rho_oil_ref * (1.0 + self.oil_compressibility * (pressure - self.pressure_ref))

    def calc_water_kr(self, s_w):
        """
        Corey с устойчивым хвостом у Sw=Swc: в зоне [0, δ] используем
        линейную аппроксимацию кривой (согласованную по значению в точке δ),
        дающую ненулевую производную при Sw=Swc.
        """
        denom = (1.0 - self.sw_cr - self.so_r + 1e-12)
        s_raw = ((s_w - self.sw_cr) / denom).clamp(0.0, 1.0)
        delta = float(self.rp_delta)
        # Corey в основной области
        kr_core = s_raw ** self.nw
        # Линейный хвост на [0, δ]: kr = s_raw * δ^{nw-1}
        slope_tail = (delta ** max(self.nw - 1, 0))
        kr_tail = s_raw * slope_tail
        use_tail = (s_raw <= delta)
        kr = torch.where(use_tail, kr_tail, kr_core)
        # Минимальный пол: позволяет изменить чувствительность J по S при старте на границе
        return torch.clamp(kr, min=self.krw_min)

    def calc_oil_kr(self, s_w):
        """
        Corey с устойчивым хвостом у So=Sor: для s_raw∈[1-δ,1] используем
        линейный хвост kro = (1 - s_raw) · δ^{no-1}, обеспечивающий
        ненулевую производную при So=Sor и согласование в точке 1-δ.
        """
        denom = (1.0 - self.sw_cr - self.so_r + 1e-12)
        s_raw = ((s_w - self.sw_cr) / denom).clamp(0.0, 1.0)
        delta = float(self.rp_delta)
        kro_core = (1.0 - s_raw).clamp(0.0, 1.0) ** self.no
        slope_tail = (delta ** max(self.no - 1, 0))
        kro_tail = (1.0 - s_raw) * slope_tail
        use_tail = (s_raw >= (1.0 - delta))
        return torch.where(use_tail, kro_tail, kro_core)

    def calc_dkrw_dsw(self, s_w):
        """Производная d(krw)/d(Sw) для piecewise-Corey с линейным хвостом."""
        denom = (1.0 - self.sw_cr - self.so_r + 1e-12)
        s_raw = ((s_w - self.sw_cr) / denom).clamp(0.0, 1.0)
        delta = float(self.rp_delta)
        dsraw_dsw = 1.0 / denom
        # Хвост: d/dsw [s_raw * δ^{nw-1}] = δ^{nw-1} / denom
        tail_der = (delta ** max(self.nw - 1, 0)) * dsraw_dsw
        # Corey область: d/dsw [s_raw^{nw}] = nw * s_raw^{nw-1} / denom
        core_der = self.nw * torch.clamp(s_raw, 1e-12, 1.0) ** (self.nw - 1) * dsraw_dsw
        use_tail = (s_raw <= delta)
        return torch.where(use_tail, tail_der, core_der)

    def calc_dkro_dsw(self, s_w):
        """Производная d(kro)/d(Sw) для piecewise-Corey с линейным хвостом."""
        denom = (1.0 - self.sw_cr - self.so_r + 1e-12)
        s_raw = ((s_w - self.sw_cr) / denom).clamp(0.0, 1.0)
        delta = float(self.rp_delta)
        dsraw_dsw = 1.0 / denom
        # Хвост (вблизи 1): kro = (1 - s_raw) * δ^{no-1}
        tail_der = - (delta ** max(self.no - 1, 0)) * dsraw_dsw
        # Corey область: d/dsw [(1-s_raw)^{no}] = -no * (1-s_raw)^{no-1} / denom
        core_der = -self.no * torch.clamp(1.0 - s_raw, 1e-12, 1.0) ** (self.no - 1) * dsraw_dsw
        use_tail = (s_raw >= (1.0 - delta))
        return torch.where(use_tail, tail_der, core_der)

    # ---- Вязкости (Pa·s) ----
    def calc_water_viscosity(self, pressure):
        if self._use_pvt and self._mu_w_table.numel() > 0:
            if self._mu_w_table.dim()==2 and self._use_temp:
                return self._interp2d(pressure, self.temperature, self._p_grid, self._t_grid, self._mu_w_table)
            else:
                return self._interp(pressure, self._p_grid, self._mu_w_table)
        return torch.full_like(pressure, self.mu_water)

    def calc_oil_viscosity(self, pressure):
        if self._use_pvt and self._mu_o_table.numel() > 0:
            if self._mu_o_table.dim()==2 and self._use_temp:
                return self._interp2d(pressure, self.temperature, self._p_grid, self._t_grid, self._mu_o_table)
            else:
                return self._interp(pressure, self._p_grid, self._mu_o_table)
        return torch.full_like(pressure, self.mu_oil)

    # ---- Газовая фаза ----
    def calc_gas_density(self, pressure):
        """Плотность газа ρg(P) с учётом Bg(P,T)."""
        if self._use_pvt and self._bg_table.numel() > 0:
            Bg = self.calc_bg(pressure)
            return self.rho_g_sc / (Bg + 1e-12)
        return self.rho_gas_ref * (1.0 + self.gas_compressibility * (pressure - self.pressure_ref))

    def calc_gas_viscosity(self, pressure):
        """Вязкость газа μg(P[,T])."""
        if self._use_pvt and self._mu_g_table.numel() > 0:
            if self._mu_g_table.dim()==2 and self._use_temp:
                return self._interp2d(pressure, self.temperature, self._p_grid, self._t_grid, self._mu_g_table)
            else:
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

    # ------------------------------------------------------------------
    # 1-D линейная интерполяция – производная d(prop)/dP
    # ------------------------------------------------------------------
    def _interp_derivative(self, p, p_grid, prop_grid):
        """Возвращает производную линейной интерполяции d(prop)/dP."""
        # Гарантируем одинаковое устройство
        p_grid = p_grid.to(p.device)
        prop_grid = prop_grid.to(p.device)

        p_flat = p.view(-1)
        idx_hi = torch.searchsorted(p_grid, p_flat, right=True)
        idx_hi = idx_hi.clamp(1, p_grid.numel() - 1)
        idx_lo = idx_hi - 1

        p_lo = p_grid[idx_lo]
        p_hi = p_grid[idx_hi]
        slope = (prop_grid[idx_hi] - prop_grid[idx_lo]) / (p_hi - p_lo + 1e-12)
        return slope.view_as(p)

    # ------------------------------------------------------------------
    # 2-D (T×P) билинейная интерполяция и её производная по P
    # ------------------------------------------------------------------
    def _interp2d(self, p, t, p_grid, t_grid, prop_grid):
        """Билинейная интерполяция prop(t, p). prop_grid shape=(nT, nP)."""
        # Приводим к тензорам на том же устройстве, что p
        device = p.device
        p_grid = p_grid.to(device)
        t_grid = t_grid.to(device)
        prop_grid = prop_grid.to(device)

        p_flat = p.view(-1)
        t_flat = (t if isinstance(t, torch.Tensor) else torch.tensor(t)).to(device).view(-1).expand_as(p_flat)

        # Индексы по давлению
        idx_p_hi = torch.searchsorted(p_grid, p_flat, right=True).clamp(1, p_grid.numel()-1)
        idx_p_lo = idx_p_hi - 1
        p_lo = p_grid[idx_p_lo]; p_hi = p_grid[idx_p_hi]
        wp = (p_flat - p_lo) / (p_hi - p_lo + 1e-12)

        # Индексы по температуре
        idx_t_hi = torch.searchsorted(t_grid, t_flat, right=True).clamp(1, t_grid.numel()-1)
        idx_t_lo = idx_t_hi - 1
        t_lo = t_grid[idx_t_lo]; t_hi = t_grid[idx_t_hi]
        wt = (t_flat - t_lo) / (t_hi - t_lo + 1e-12)

        # Значения свойства в четырёх узлах
        f_ll = prop_grid[idx_t_lo, idx_p_lo]
        f_lh = prop_grid[idx_t_lo, idx_p_hi]
        f_hl = prop_grid[idx_t_hi, idx_p_lo]
        f_hh = prop_grid[idx_t_hi, idx_p_hi]

        # Интерполяция сначала по P, затем по T
        fp_lo = f_ll + wp * (f_lh - f_ll)
        fp_hi = f_hl + wp * (f_hh - f_hl)
        f = fp_lo + wt * (fp_hi - fp_lo)
        return f.view_as(p)

    def _interp2d_dp(self, p, t, p_grid, t_grid, prop_grid):
        """Производная d(prop)/dP для 2-D таблицы."""
        device = p.device
        p_grid = p_grid.to(device)
        t_grid = t_grid.to(device)
        prop_grid = prop_grid.to(device)

        p_flat = p.view(-1)
        t_flat = (t if isinstance(t, torch.Tensor) else torch.tensor(t)).to(device).view(-1).expand_as(p_flat)

        idx_p_hi = torch.searchsorted(p_grid, p_flat, right=True).clamp(1, p_grid.numel()-1)
        idx_p_lo = idx_p_hi - 1
        p_lo = p_grid[idx_p_lo]; p_hi = p_grid[idx_p_hi]
        inv_dP = 1.0 / (p_hi - p_lo + 1e-12)
        wp = (p_flat - p_lo) * inv_dP

        idx_t_hi = torch.searchsorted(t_grid, t_flat, right=True).clamp(1, t_grid.numel()-1)
        idx_t_lo = idx_t_hi - 1
        t_lo = t_grid[idx_t_lo]; t_hi = t_grid[idx_t_hi]
        wt = (t_flat - t_lo) / (t_hi - t_lo + 1e-12)

        # Слоны вдоль P (склоны на LoT и HiT)
        slope_loT = (prop_grid[idx_t_lo, idx_p_hi] - prop_grid[idx_t_lo, idx_p_lo]) * inv_dP
        slope_hiT = (prop_grid[idx_t_hi, idx_p_hi] - prop_grid[idx_t_hi, idx_p_lo]) * inv_dP

        dfdp = slope_loT * (1 - wt) + slope_hiT * wt
        return dfdp.view_as(p)

    # ------------------------------------------------------------------
    # PVT-производные по давлению
    # ------------------------------------------------------------------
    def calc_dbo_dp(self, pressure):
        if self._use_pvt and self._bo_table.numel() > 0:
            if self._bo_table.dim()==2 and self._use_temp:
                return self._interp2d_dp(pressure, self.temperature, self._p_grid, self._t_grid, self._bo_table)
            else:
                return self._interp_derivative(pressure, self._p_grid, self._bo_table)
        return torch.zeros_like(pressure)

    def calc_dbg_dp(self, pressure):
        if self._use_pvt and self._bg_table.numel() > 0:
            if self._bg_table.dim()==2 and self._use_temp:
                return self._interp2d_dp(pressure, self.temperature, self._p_grid, self._t_grid, self._bg_table)
            else:
                return self._interp_derivative(pressure, self._p_grid, self._bg_table)
        return torch.zeros_like(pressure)

    def calc_dbw_dp(self, pressure):
        if self._use_pvt and self._bw_table.numel() > 0:
            if self._bw_table.dim()==2 and self._use_temp:
                return self._interp2d_dp(pressure, self.temperature, self._p_grid, self._t_grid, self._bw_table)
            else:
                return self._interp_derivative(pressure, self._p_grid, self._bw_table)
        return torch.zeros_like(pressure)

    def calc_drs_dp(self, pressure):
        """dRs/dP (1/Па) – аналитическая производная через PVT-таблицу или линейная модель."""
        if self._use_pvt and self._rs_table.numel() > 0:
            if self._rs_table.dim()==2 and self._use_temp:
                return self._interp2d_dp(pressure, self.temperature, self._p_grid, self._t_grid, self._rs_table)
            else:
                return self._interp_derivative(pressure, self._p_grid, self._rs_table)
        # Fallback: линейная модель ниже pbubble
        pb = self.pbubble
        rs_b = self.rs_bubble
        return torch.where(pressure >= pb,
                           torch.zeros_like(pressure),
                           rs_b / pb)

    def calc_drv_dp(self, pressure):
        """dRv/dP (1/Па)."""
        if self._use_pvt and self._rv_table.numel() > 0:
            if self._rv_table.dim()==2 and self._use_temp:
                return self._interp2d_dp(pressure, self.temperature, self._p_grid, self._t_grid, self._rv_table)
            else:
                return self._interp_derivative(pressure, self._p_grid, self._rv_table)
        # По умолчанию Rv=0 ⇒ производная 0
        return torch.zeros_like(pressure)

    # ------------------------------------------------------------------
    # Вязкости: производные dμ/dP (Па·с / Па)
    # ------------------------------------------------------------------
    def calc_dmu_o_dp(self, pressure):
        if self._use_pvt and self._mu_o_table.numel() > 0:
            if self._mu_o_table.dim()==2 and self._use_temp:
                return self._interp2d_dp(pressure, self.temperature, self._p_grid, self._t_grid, self._mu_o_table)
            else:
                return self._interp_derivative(pressure, self._p_grid, self._mu_o_table)
        return torch.zeros_like(pressure)

    def calc_dmu_w_dp(self, pressure):
        if self._use_pvt and self._mu_w_table.numel() > 0:
            if self._mu_w_table.dim()==2 and self._use_temp:
                return self._interp2d_dp(pressure, self.temperature, self._p_grid, self._t_grid, self._mu_w_table)
            else:
                return self._interp_derivative(pressure, self._p_grid, self._mu_w_table)
        return torch.zeros_like(pressure)

    def calc_dmu_g_dp(self, pressure):
        if self._use_pvt and self._mu_g_table.numel() > 0:
            if self._mu_g_table.dim()==2 and self._use_temp:
                return self._interp2d_dp(pressure, self.temperature, self._p_grid, self._t_grid, self._mu_g_table)
            else:
                return self._interp_derivative(pressure, self._p_grid, self._mu_g_table)
        return torch.zeros_like(pressure)

    # ------------------------------------------------------------------
    # Плотности: dρ/dP (кг·м⁻³ / Па)
    # ------------------------------------------------------------------
    def calc_drho_o_dp(self, pressure):
        if self._use_pvt and self._bo_table.numel() > 0:
            Bo = self.calc_bo(pressure)
            dBo = self.calc_dbo_dp(pressure)
            return -self.rho_o_sc * dBo / (Bo + 1e-12)**2
        # Линейная compressibility
        return self.oil_compressibility * self.rho_oil_ref * torch.ones_like(pressure)

    def calc_drho_w_dp(self, pressure):
        if self._use_pvt and self._bw_table.numel() > 0:
            Bw = self.calc_bw(pressure)
            dBw = self.calc_dbw_dp(pressure)
            return -self.rho_w_sc * dBw / (Bw + 1e-12)**2
        return self.water_compressibility * self.rho_water_ref * torch.ones_like(pressure)

    def calc_drho_g_dp(self, pressure):
        if self._use_pvt and self._bg_table.numel() > 0:
            Bg = self.calc_bg(pressure)
            dBg = self.calc_dbg_dp(pressure)
            return -self.rho_g_sc * dBg / (Bg + 1e-12)**2
        return self.gas_compressibility * self.rho_gas_ref * torch.ones_like(pressure)

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
        """Растворённый газовый фактор Rs(P)."""
        # Табличное значение при наличии PVT
        if self._use_pvt and self._rs_table.numel() > 0:
            if self._rs_table.dim()==2 and self._use_temp:
                return self._interp2d(pressure, self.temperature, self._p_grid, self._t_grid, self._rs_table)
            else:
                return self._interp(pressure, self._p_grid, self._rs_table)

        # Fallback: линейная зависимость от давления
        pb = self.pbubble
        rs_b = self.rs_bubble
        return torch.where(pressure >= pb,
                           torch.full_like(pressure, rs_b),
                           rs_b * pressure / pb)

    def calc_drs_dp_fd(self, pressure):
        """[LEGACY] dRs/dP численно через конечные разности (используется только если явно вызвано)."""
        if self._use_pvt and self._rs_table.numel() > 0:
            # численная производная через центральные разности
            eps = 1e3  # 0.001 МПа
            return (self.calc_rs(pressure + eps) - self.calc_rs(pressure - eps)) / (2 * eps)
        pb = self.pbubble
        rs_b = self.rs_bubble
        return torch.where(pressure >= pb,
                           torch.zeros_like(pressure),
                           rs_b / pb)

    # ------------------------------------------------------------------
    # Масса газа (свободный + растворённый)
    # ------------------------------------------------------------------
    def total_gas_mass(self, s_o, s_g, pressure, porosity):
        """Возвращает суммарную массу газовой фазы в ячейке (кг)."""
        rho_g = self.calc_gas_density(pressure)
        rho_g_sc = self.rho_g_sc
        Rs = self.calc_rs(pressure)
        # m_g =  φ (Sg ρg + So Rs ρg_sc)
        return porosity * (s_g * rho_g + s_o * Rs * rho_g_sc)

    # ---- PVT ------------------------------------------------------------
    def calc_bo(self, pressure):
        if self._use_pvt and self._bo_table.numel() > 0:
            if self._bo_table.dim() == 2 and self._use_temp:
                return self._interp2d(pressure, self.temperature, self._p_grid, self._t_grid, self._bo_table)
            else:
                return self._interp(pressure, self._p_grid, self._bo_table)
        return torch.ones_like(pressure)

    def calc_bg(self, pressure):
        if self._use_pvt and self._bg_table.numel() > 0:
            if self._bg_table.dim()==2 and self._use_temp:
                return self._interp2d(pressure, self.temperature, self._p_grid, self._t_grid, self._bg_table)
            else:
                return self._interp(pressure, self._p_grid, self._bg_table)
        return torch.ones_like(pressure)

    def calc_bw(self, pressure):
        if self._use_pvt and self._bw_table.numel() > 0:
            if self._bw_table.dim()==2 and self._use_temp:
                return self._interp2d(pressure, self.temperature, self._p_grid, self._t_grid, self._bw_table)
            else:
                return self._interp(pressure, self._p_grid, self._bw_table)
        return torch.ones_like(pressure)

    # дублирующее определение calc_rs удалено

    def calc_rv(self, pressure):
        if self._use_pvt and self._rv_table.numel() > 0:
            if self._rv_table.dim()==2 and self._use_temp:
                return self._interp2d(pressure, self.temperature, self._p_grid, self._t_grid, self._rv_table)
            else:
                return self._interp(pressure, self._p_grid, self._rv_table)
        # По умолчанию Rv=0
        return torch.zeros_like(pressure)

    # дублирующие плотности удалены (оставлены единственные версии выше)

    # ------------------------------------------------------------------
    # Helper constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict, reservoir=None, device=None):
        """Создаёт объект Fluid из полной конфигурации симуляции.

        Обёртка предназначена для тестов и высокого уровня API, где передаётся
        полный JSON конфиг, содержащий секции ``reservoir`` и ``fluid``.

        Args:
            cfg: Полный конфиг либо непосредственно словарь параметров флюида.
            reservoir: Опциональный уже созданный объект Reservoir. Если
                отсутствует, будет создан из той же конфигурации.
            device: CPU/GPU устройство.
        """
        # Отложенный импорт, чтобы избежать циклических зависимостей
        from simulator.reservoir import Reservoir  # локальный импорт

        # Извлекаем секцию с параметрами флюида
        fluid_cfg = cfg.get("fluid", cfg)

        # Если резервуар не передан, создаём его из той же конфигурации
        if reservoir is None:
            reservoir = Reservoir.from_config(cfg, device=device)

        return cls(config=fluid_cfg, reservoir=reservoir, device=device)
