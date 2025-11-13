import torch
import numpy as np
from pathlib import Path
from .pvt import PVTTable
from .deck import load_pvt_from_deck, load_relperm_tables_from_deck


class Fluid:
    """
    Класс для моделирования свойств флюидов (нефть, вода и газ).
    Поддерживает трехфазную модель black-oil.
    """
    def __init__(self, config, reservoir, device=None):
        """
        Инициализация флюидов по конфигурации.
        
        Args:
            config: Словарь с параметрами флюидов
            reservoir: Объект пласта
            device: Устройство для вычислений (CPU/GPU)
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Размеры и форма тензоров
        self.dimensions = reservoir.dimensions
        nx, ny, nz = self.dimensions
        
        # Начальные значения
        initial_pressure = config.get('pressure', 20.0) * 1e6  # МПа -> Па
        initial_sw = config.get('s_w', 0.2)
        initial_sg = config.get('s_g', 0.0)  # Начальная газонасыщенность (по умолчанию 0)
        
        # Свойства флюидов (базовые, если нет PVT)
        self.mu_oil = config.get('mu_oil', 1.0) * 1e-3  # сП -> Па*с
        self.mu_water = config.get('mu_water', 0.5) * 1e-3  # сП -> Па*с
        self.mu_gas = config.get('mu_gas', 0.02) * 1e-3  # сП -> Па*с
        
        # Плотности при стандартных условиях (SC)
        self.rho_oil_ref = config.get('rho_oil', 850.0)  # кг/м^3 @SC
        self.rho_water_ref = config.get('rho_water', 1000.0)  # кг/м^3 @SC
        self.rho_gas_ref = config.get('rho_gas', 1.0)  # кг/м^3 @SC
        
        # Сжимаемость (1/Па)
        self.oil_compressibility = config.get('c_oil', 1e-5) / 1e6  # 1/МПа -> 1/Па
        self.water_compressibility = config.get('c_water', 1e-5) / 1e6  # 1/МПа -> 1/Па
        self.gas_compressibility = config.get('c_gas', 1e-3) / 1e6  # 1/МПа -> 1/Па (газ более сжимаем)
        self.rock_compressibility = config.get('c_rock', 1e-5) / 1e6  # 1/МПа -> 1/Па
        
        # Совокупная сжимаемость флюида (используется в IMPES, если нет PVT)
        total_c = (self.oil_compressibility + self.water_compressibility + self.gas_compressibility + self.rock_compressibility) / 3
        self.cf = torch.full(self.dimensions, total_c, device=self.device)

        # PVT таблицы (опционально)
        self.pvt = None
        pvt_path = config.get('pvt_path') or config.get('pvt', {}).get('path')
        deck_relperm_tables = None
        if pvt_path:
            try:
                path_obj = Path(pvt_path)
                if path_obj.exists() and path_obj.suffix.lower() not in {'.json', '.jsn'}:
                    deck_data = load_pvt_from_deck(pvt_path)
                    self.pvt = PVTTable(deck_data)
                    deck_relperm_tables = load_relperm_tables_from_deck(pvt_path)
                    print(f"  PVT: загружен deck {pvt_path}")
                else:
                    self.pvt = PVTTable(pvt_path)
                    print(f"  PVT: загружен {pvt_path}")
            except Exception as e:
                print(f"  PVT: не удалось загрузить '{pvt_path}': {e}")
        
        # Параметры модели относительной проницаемости
        rp_cfg = config.get('relative_permeability', {})
        self.relperm_model = rp_cfg.get('model', 'corey')
        self.relperm_tables = None
        if self.relperm_model == 'table':
            table_path = rp_cfg.get('path')
            tables = None
            if table_path:
                tables = load_relperm_tables_from_deck(table_path)
            elif deck_relperm_tables:
                tables = deck_relperm_tables
            if not tables:
                raise ValueError("relative_permeability 'table' требует указать path или использовать deck вместе с PVT")
            self._init_relperm_tables(tables)
        else:
            self.nw    = rp_cfg.get('nw', 2)
            self.no    = rp_cfg.get('no', 2)
            self.ng    = rp_cfg.get('ng', 2)
            self.ko_end_w = rp_cfg.get('ko_end_w', 1.0)
            self.ko_end_g = rp_cfg.get('ko_end_g', 1.0)
            self.now = rp_cfg.get('now', float(self.no))
            self.nog = rp_cfg.get('nog', float(self.ng))
            self.krw_end = rp_cfg.get('krw_end', 1.0)
            self.krg_end = rp_cfg.get('krg_end', 1.0)
            self.sw_cr = rp_cfg.get('sw_cr', 0.2)
            self.so_r  = rp_cfg.get('so_r', 0.2)
            self.sg_cr = rp_cfg.get('sg_cr', 0.0)
        
        # Инициализация полей
        self.pressure = torch.full(self.dimensions, initial_pressure, device=self.device)
        self.s_w = torch.full(self.dimensions, initial_sw, device=self.device)
        self.s_g = torch.full(self.dimensions, initial_sg, device=self.device)
        self.s_o = 1.0 - self.s_w - self.s_g  # Нефтенасыщенность
        
        # Проверка корректности насыщенностей
        if (self.s_w + self.s_g).max() > 1.0:
            raise ValueError("Сумма начальных водо- и газонасыщенностей не может превышать 1.0")
        
        self.prev_pressure = self.pressure.clone()
        self.prev_sw = self.s_w.clone()
        self.prev_sg = self.s_g.clone()
        
        # Сохраняем предыдущее состояние для неявных расчетов
        self.prev_water_mass = None
        self.prev_oil_mass = None
        self.prev_gas_mass = None
        
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

        # Капиллярное давление нефть-газ (опционально)
        pcg_cfg = config.get('capillary_pressure_og', {})
        self.pc_og_scale = pcg_cfg.get('pc_scale', 0.0)
        self.pc_og_exponent = pcg_cfg.get('pc_exponent', 1.5)
        
        # Выводим информацию об инициализации
        print("Инициализация флюидов и начальных условий...")
        print(f"  Начальное давление: {initial_pressure/1e6:.2f} МПа")
        print(f"  Начальная водонасыщенность: {initial_sw}")
        print(f"  Начальная газонасыщенность: {initial_sg}")
        print(f"  Начальная нефтенасыщенность: {1.0 - initial_sw - initial_sg:.3f}")
        if self.pvt is None:
            print(f"  Вязкость нефти/воды/газа: {self.mu_oil*1e3:.2f}/{self.mu_water*1e3:.2f}/{self.mu_gas*1e3:.3f} сП")
        else:
            print("  Вязкости и B-факторы берутся из PVT-таблиц")
        print(f"  Плотность нефти/воды/газа: {self.rho_oil_ref}/{self.rho_water_ref}/{self.rho_gas_ref} кг/м^3")
        print(f"  Сжимаемость нефти/воды/газа: {self.oil_compressibility*1e6:.1e}/{self.water_compressibility*1e6:.1e}/{self.gas_compressibility*1e6:.1e} 1/МПа")
        print(f"  Капиллярное давление: {self.pc_scale/1e6:.2e} МПа, показатель {self.pc_exponent}")
        print(f"  Связанная водонасыщенность: {self.sw_cr}, остаточная нефтенасыщенность: {self.so_r}")
        print(f"  Тензоры флюидов размещены на: {self.device}")

    def _init_relperm_tables(self, tables):
        """Инициализация таблиц SWOF/SGOF для табличной модели относительной проницаемости."""
        self.relperm_tables = {}

        def _as_np(seq):
            return np.asarray(seq, dtype=np.float64)

        def _compute_slopes(x, y):
            slopes = np.zeros_like(y)
            if len(x) < 2:
                return slopes
            dx = np.diff(x)
            dy = np.diff(y)
            slopes[:-1] = dy / (dx + 1e-12)
            slopes[-1] = slopes[-2] if len(slopes) > 1 else 0.0
            return slopes

        if "swof" in tables:
            tbl = tables["swof"]
            sw = _as_np(tbl["sw"])
            krw = _as_np(tbl["krw"])
            kro = _as_np(tbl["kro"])
            pcow = _as_np(tbl.get("pcow", [0.0] * len(sw))) * 6894.75729  # psi -> Pa
            self.relperm_tables["swof"] = {
                "sw": sw,
                "krw": krw,
                "kro": kro,
                "pcow": pcow,
                "dkrw_dsw": _compute_slopes(sw, krw),
                "dkro_dsw": _compute_slopes(sw, kro),
                "dpcow_dsw": _compute_slopes(sw, pcow),
            }
            self.sw_cr = sw[0]
            self.so_r = max(0.0, 1.0 - sw[-1])
        else:
            self.sw_cr = 0.2
            self.so_r = 0.2

        if "sgof" in tables:
            tbl = tables["sgof"]
            sg = _as_np(tbl["sg"])
            krg = _as_np(tbl["krg"])
            kro = _as_np(tbl["kro"])
            pcog = _as_np(tbl.get("pcog", [0.0] * len(sg))) * 6894.75729
            self.relperm_tables["sgof"] = {
                "sg": sg,
                "krg": krg,
                "kro": kro,
                "pcog": pcog,
                "dkrg_dsg": _compute_slopes(sg, krg),
                "dkro_dsg": _compute_slopes(sg, kro),
                "dpcog_dsg": _compute_slopes(sg, pcog),
            }
            self.sg_cr = sg[0]
        else:
            self.sg_cr = 0.0

        # При табличной модели отключаем степенной Pc по параметрам
        self.pc_scale = 0.0
        self.pc_og_scale = 0.0

    def _interp_table(self, tensor: torch.Tensor, x: np.ndarray, y: np.ndarray) -> torch.Tensor:
        values = np.interp(tensor.detach().cpu().numpy(), x, y, left=y[0], right=y[-1])
        return torch.from_numpy(values).to(self.device, dtype=torch.float32)

    # Свойства для совместимости со старым кодом IMPES
    @property
    def rho_w(self):
        """Плотность воды в пластовых условиях при текущем давлении"""
        if self.pvt is None:
            return self.calc_water_density(self.pressure)
        bw = self._eval_pvt(self.pressure, key='Bw')
        return self.rho_water_ref / (bw + 1e-12)
        
    @property
    def rho_o(self):
        """Плотность нефти в пластовых условиях (учет растворенного газа Rs)"""
        if self.pvt is None:
            return self.calc_oil_density(self.pressure)
        bo = self._eval_pvt(self.pressure, key='Bo')
        rs = self._eval_pvt(self.pressure, key='Rs')
        # ρ_o^res ≈ (ρ_o_sc + Rs * ρ_g_sc) / Bo
        return (self.rho_oil_ref + rs * self.rho_gas_ref) / (bo + 1e-12)
    
    @property
    def rho_g(self):
        """Плотность газа в пластовых условиях"""
        if self.pvt is None:
            return self.calc_gas_density(self.pressure)
        bg = self._eval_pvt(self.pressure, key='Bg')
        # Учет Rv (впаривание нефти в газ): ρ_g^res ≈ (ρ_g_sc + Rv * ρ_o_sc) / Bg
        try:
            rv = self._eval_pvt(self.pressure, key='Rv')
        except KeyError:
            rv = torch.zeros_like(bg)
        return (self.rho_gas_ref + rv * self.rho_oil_ref) / (bg + 1e-12)
        
    @property
    def mu_w(self):
        """Вязкость воды при текущем давлении"""
        if self.pvt is None:
            return self.mu_water
        return self._eval_pvt(self.pressure, key='mu_w') * 1e-3
        
    @property
    def mu_o(self):
        """Вязкость нефти при текущем давлении"""
        if self.pvt is None:
            return self.mu_oil
        return self._eval_pvt(self.pressure, key='mu_o') * 1e-3
    
    @property
    def mu_g(self):
        """Вязкость газа при текущем давлении"""
        if self.pvt is None:
            return self.mu_gas
        return self._eval_pvt(self.pressure, key='mu_g') * 1e-3

    # ---- PVT helpers ----
    def _eval_pvt(self, p_tensor: torch.Tensor, key: str) -> torch.Tensor:
        """Интерполирует таблицу PVT по давлению тензора (в MPa). Возвращает torch-тензор на self.device."""
        if self.pvt is None:
            raise RuntimeError("PVT не загружен")
        p_mpa = (p_tensor.detach().to('cpu').numpy() / 1e6).astype(np.float64)
        vals = self.pvt.eval(p_mpa)[key]
        return torch.from_numpy(vals).to(self.device, dtype=torch.float32).reshape(self.dimensions)

    def calc_total_compressibility(self, pressure: torch.Tensor, s_w: torch.Tensor, s_g: torch.Tensor) -> torch.Tensor:
        """Оценка суммарной сжимаемости ct(P,Sw,Sg) из PVT (пер Па). Fallback на self.cf."""
        if self.pvt is None:
            return self.cf

        dP_mpa = 0.1
        dP_pa = dP_mpa * 1e6

        P = pressure
        P_plus = torch.clamp(P + dP_pa, min=1e3)
        P_minus = torch.clamp(P - dP_pa, min=1e3)

        Bw = self._eval_pvt(P, 'Bw')
        Bo = self._eval_pvt(P, 'Bo')
        Bg = self._eval_pvt(P, 'Bg')

        Bw_p = self._eval_pvt(P_plus, 'Bw')
        Bw_m = self._eval_pvt(P_minus, 'Bw')
        Bo_p = self._eval_pvt(P_plus, 'Bo')
        Bo_m = self._eval_pvt(P_minus, 'Bo')
        Bg_p = self._eval_pvt(P_plus, 'Bg')
        Bg_m = self._eval_pvt(P_minus, 'Bg')

        dBw_dP_mpa = (Bw_p - Bw_m) / (2 * dP_mpa)
        dBo_dP_mpa = (Bo_p - Bo_m) / (2 * dP_mpa)
        dBg_dP_mpa = (Bg_p - Bg_m) / (2 * dP_mpa)

        c_w = -(dBw_dP_mpa / (Bw + 1e-12)) / 1e6
        c_o = -(dBo_dP_mpa / (Bo + 1e-12)) / 1e6
        c_g = -(dBg_dP_mpa / (Bg + 1e-12)) / 1e6

        s_o = 1.0 - s_w - s_g
        ct = self.rock_compressibility + s_w * c_w + s_o * c_o + s_g * c_g
        return ct.to(self.device)

    def _get_normalized_saturation(self, s_w):
        if self.relperm_model == 'table' and 'swof' in self.relperm_tables:
            sw_table = self.relperm_tables['swof']['sw']
            sw_min, sw_max = sw_table[0], sw_table[-1]
            return torch.clamp((s_w - sw_min) / (sw_max - sw_min + 1e-12), 0.0, 1.0)
        s_norm = (s_w - self.sw_cr) / (1 - self.sw_cr - self.so_r)
        return torch.clamp(s_norm, 0.0, 1.0)

    def get_rel_perms(self, s_w, s_g=None):
        kro = self.calc_oil_kr(s_w, s_g)
        krw = self.calc_water_kr(s_w)
        if s_g is None:
            s_g = self.s_g
        krg = self.calc_gas_kr(s_g)
        return kro, krw, krg

    def get_rel_perms_derivatives(self, s_w):
        if self.relperm_model == 'table' and 'swof' in self.relperm_tables:
            table = self.relperm_tables['swof']
            sw = table['sw']
            dkrw = table['dkrw_dsw']
            dkro = table['dkro_dsw']
            dkrw_tensor = self._interp_table(s_w, sw, dkrw)
            dkro_tensor = self._interp_table(s_w, sw, dkro)
            return dkro_tensor, dkrw_tensor
        s_norm = self._get_normalized_saturation(s_w)
        dsw_norm_dsw = 1 / (1 - self.sw_cr - self.so_r)
        dkrw_dsw = self.nw * (s_norm ** (self.nw - 1)) * dsw_norm_dsw
        dkro_dsw = -self.no * ((1 - s_norm) ** (self.no - 1)) * dsw_norm_dsw
        dkrw_dsw = torch.where(s_norm <= 0, torch.zeros_like(dkrw_dsw), dkrw_dsw)
        dkro_dsw = torch.where(s_norm >= 1, torch.zeros_like(dkro_dsw), dkro_dsw)
        return dkro_dsw, dkrw_dsw

    def get_capillary_pressure(self, s_w):
        if self.relperm_model == 'table' and 'swof' in self.relperm_tables:
            table = self.relperm_tables['swof']
            return self._interp_table(s_w, table['sw'], table['pcow'])
        if self.pc_scale == 0.0:
            return torch.zeros_like(s_w)
        s_norm = self._get_normalized_saturation(s_w)
        pc = self.pc_scale * (1.0 - s_norm + 1e-6) ** (-self.pc_exponent)
        return pc

    def get_capillary_pressure_og(self, s_g):
        if self.relperm_model == 'table' and 'sgof' in self.relperm_tables:
            table = self.relperm_tables['sgof']
            return self._interp_table(s_g, table['sg'], table['pcog'])
        if self.pc_og_scale == 0.0:
            return torch.zeros_like(s_g)
        denom = 1 - self.sw_cr - self.so_r
        sg_norm = torch.clamp(s_g / (denom + 1e-12), 0.0, 1.0)
        pcog = self.pc_og_scale * (1.0 - sg_norm + 1e-6) ** (-self.pc_og_exponent)
        return pcog

    def get_capillary_pressure_og_derivative(self, s_g):
        if self.relperm_model == 'table' and 'sgof' in self.relperm_tables:
            table = self.relperm_tables['sgof']
            dpc = table['dpcog_dsg']
            return self._interp_table(s_g, table['sg'], dpc)
        if self.pc_og_scale == 0.0:
            return torch.zeros_like(s_g)
        denom = 1 - self.sw_cr - self.so_r
        sg_norm = torch.clamp(s_g / (denom + 1e-12), 0.0, 1.0)
        dsgn_dsg = 1.0 / (denom + 1e-12)
        dpc_dsgn = self.pc_og_scale * self.pc_og_exponent * (1.0 - sg_norm + 1e-6) ** (-self.pc_og_exponent - 1)
        dpc_dsg = dpc_dsgn * dsgn_dsg
        dpc_dsg = torch.where(sg_norm >= 1, torch.zeros_like(dpc_dsg), dpc_dsg)
        return dpc_dsg

    def get_capillary_pressure_derivative(self, s_w):
        if self.relperm_model == 'table' and 'swof' in self.relperm_tables:
            table = self.relperm_tables['swof']
            dpc = table['dpcow_dsw']
            return self._interp_table(s_w, table['sw'], dpc)
        if self.pc_scale == 0.0:
            return torch.zeros_like(s_w)
        s_norm = self._get_normalized_saturation(s_w)
        dsw_norm_dsw = 1 / (1 - self.sw_cr - self.so_r)
        dpc_dsn = self.pc_scale * self.pc_exponent * (1.0 - s_norm + 1e-6) ** (-self.pc_exponent - 1)
        dpc_dsw = dpc_dsn * dsw_norm_dsw
        dpc_dsw = torch.where(s_norm >= 1, torch.zeros_like(dpc_dsw), dpc_dsw)
        return dpc_dsw

    def calc_water_density(self, pressure):
        """
        Вычисляет плотность воды при заданном давлении.
        
        Args:
            pressure: Тензор давления
            
        Returns:
            Тензор плотности воды
        """
        if self.pvt is not None:
            bw = self._eval_pvt(pressure, key='Bw')
            return self.rho_water_ref / (bw + 1e-12)
        return self.rho_water_ref * (1.0 + self.water_compressibility * (pressure - 1e5))

    def calc_oil_density(self, pressure):
        """
        Вычисляет плотность нефти при заданном давлении.
        
        Args:
            pressure: Тензор давления
            
        Returns:
            Тензор плотности нефти
        """
        if self.pvt is not None:
            bo = self._eval_pvt(pressure, key='Bo')
            rs = self._eval_pvt(pressure, key='Rs')
            return (self.rho_oil_ref + rs * self.rho_gas_ref) / (bo + 1e-12)
        return self.rho_oil_ref * (1.0 + self.oil_compressibility * (pressure - 1e5))
    
    def calc_gas_density(self, pressure):
        """
        Вычисляет плотность газа при заданном давлении.
        Используется упрощенная модель с высокой сжимаемостью.
        
        Args:
            pressure: Тензор давления
            
        Returns:
            Тензор плотности газа
        """
        if self.pvt is not None:
            bg = self._eval_pvt(pressure, key='Bg')
            return self.rho_gas_ref / (bg + 1e-12)
        return self.rho_gas_ref * (1.0 + self.gas_compressibility * (pressure - 1e5))

    def calc_water_kr(self, s_w):
        if self.relperm_model == 'table' and 'swof' in self.relperm_tables:
            table = self.relperm_tables['swof']
            return self._interp_table(s_w, table['sw'], table['krw'])
        s_norm = self._get_normalized_saturation(s_w)
        return s_norm**self.nw

    def calc_oil_kr(self, s_w, s_g=None):
        if self.relperm_model == 'table':
            kro_components = []
            if 'swof' in self.relperm_tables:
                tbl = self.relperm_tables['swof']
                kro_components.append(self._interp_table(s_w, tbl['sw'], tbl['kro']))
            if 'sgof' in self.relperm_tables:
                if s_g is None:
                    s_g = self.s_g
                tbl = self.relperm_tables['sgof']
                kro_components.append(self._interp_table(s_g, tbl['sg'], tbl['kro']))
            if kro_components:
                result = kro_components[0]
                for comp in kro_components[1:]:
                    result = torch.minimum(result, comp)
                return result
        if self.relperm_model == 'stone2':
            if s_g is None:
                s_g = self.s_g
            denom_w = 1.0 - self.sw_cr - self.so_r
            swn = torch.clamp((s_w - self.sw_cr) / (denom_w + 1e-12), 0.0, 1.0)
            denom_g = 1.0 - self.sg_cr - self.so_r
            sgn = torch.clamp((s_g - self.sg_cr) / (denom_g + 1e-12), 0.0, 1.0)
            krow = self.ko_end_w * (1.0 - swn) ** self.now
            krog = self.ko_end_g * (1.0 - sgn) ** self.nog
            kro = krow + krog - min(self.ko_end_w, self.ko_end_g)
            return torch.clamp(kro, min=0.0)
        s_norm = self._get_normalized_saturation(s_w)
        return (1 - s_norm) ** self.no
    
    def calc_gas_kr(self, s_g):
        if self.relperm_model == 'table' and 'sgof' in self.relperm_tables:
            tbl = self.relperm_tables['sgof']
            return self._interp_table(s_g, tbl['sg'], tbl['krg'])
        s_g_norm = s_g / (1.0 - self.sw_cr - self.so_r + 1e-10)
        s_g_norm = torch.clamp(s_g_norm, 0.0, 1.0)
        ng = getattr(self, 'ng', 2.0)
        return s_g_norm**ng

    def calc_dkrw_dsw(self, s_w):
        if self.relperm_model == 'table' and 'swof' in self.relperm_tables:
            tbl = self.relperm_tables['swof']
            return self._interp_table(s_w, tbl['sw'], tbl['dkrw_dsw'])
        s_norm = self._get_normalized_saturation(s_w)
        normalized_range = 1.0 - self.sw_cr - self.so_r
        mask = (s_w > self.sw_cr) & (s_w < 1.0 - self.so_r)
        result = torch.zeros_like(s_w)
        result[mask] = self.nw * s_norm[mask]**(self.nw - 1) / normalized_range
        return result

    def calc_dkro_dsw(self, s_w):
        if self.relperm_model == 'table' and 'swof' in self.relperm_tables:
            tbl = self.relperm_tables['swof']
            return self._interp_table(s_w, tbl['sw'], tbl['dkro_dsw'])
        s_norm = self._get_normalized_saturation(s_w)
        normalized_range = 1.0 - self.sw_cr - self.so_r
        mask = (s_w > self.sw_cr) & (s_w < 1.0 - self.so_r)
        result = torch.zeros_like(s_w)
        result[mask] = -self.no * (1 - s_norm[mask])**(self.no - 1) / normalized_range
        return result
    
    def calc_dkrg_dsg(self, s_g):
        if self.relperm_model == 'table' and 'sgof' in self.relperm_tables:
            tbl = self.relperm_tables['sgof']
            return self._interp_table(s_g, tbl['sg'], tbl['dkrg_dsg'])
        ng = getattr(self, 'ng', 2.0)
        normalized_range = 1.0 - self.sw_cr - self.so_r
        s_g_norm = s_g / (normalized_range + 1e-10)
        s_g_norm = torch.clamp(s_g_norm, 0.0, 1.0)
        mask = (s_g > 0.0) & (s_g < normalized_range)
        result = torch.zeros_like(s_g)
        result[mask] = ng * s_g_norm[mask]**(ng - 1) / normalized_range
        return result

    # ---- Алиасы для обратной совместимости со старым кодом ----
    # (симулятор обращается к этим именам)
    calc_capillary_pressure = get_capillary_pressure
    calc_dpc_dsw            = get_capillary_pressure_derivative
