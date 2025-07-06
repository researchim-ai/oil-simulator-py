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
        
        # Свойства флюидов
        self.mu_oil = config.get('mu_oil', 1.0) * 1e-3  # сП -> Па*с
        self.mu_water = config.get('mu_water', 0.5) * 1e-3  # сП -> Па*с
        
        # Плотности
        self.rho_oil_ref = config.get('rho_oil', 850.0)  # кг/м^3
        self.rho_water_ref = config.get('rho_water', 1000.0)  # кг/м^3
        
        # Сжимаемость (1/Па)
        self.oil_compressibility = config.get('c_oil', 1e-5) / 1e6  # 1/МПа -> 1/Па
        self.water_compressibility = config.get('c_water', 1e-5) / 1e6  # 1/МПа -> 1/Па
        self.rock_compressibility = config.get('c_rock', 1e-5) / 1e6  # 1/МПа -> 1/Па
        
        # 🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: правильное опорное давление для сжимаемости
        self.pressure_ref = getattr(reservoir, 'pressure_ref', 1e5)
        print(f"🔧 Опорное давление для плотности: {self.pressure_ref:.0f} Па ({self.pressure_ref/1e6:.1f} МПа)")
        
        # 🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: правильное опорное давление для сжимаемости
        self.pressure_ref = getattr(reservoir, 'pressure_ref', 1e5)
        print(f"🔧 Опорное давление для плотности: {self.pressure_ref:.0f} Па ({self.pressure_ref/1e6:.1f} МПа)")
        
        # Совокупная сжимаемость флюида (используется в IMPES)
        total_c = (self.oil_compressibility + self.water_compressibility + self.rock_compressibility) / 2
        self.cf = torch.full(self.dimensions, total_c, device=self.device)
        
        # Параметры модели относительной проницаемости
        rp_cfg = config.get('relative_permeability', {})
        self.nw    = rp_cfg.get('nw', 2)           # Показатель Кори для воды
        self.no    = rp_cfg.get('no', 2)           # Показатель Кори для нефти
        self.sw_cr = rp_cfg.get('sw_cr', 0.2)      # Связанная водонасыщенность
        self.so_r  = rp_cfg.get('so_r', 0.2)       # Остаточная нефтенасыщенность
        
        # Инициализация полей
        self.pressure = torch.full(self.dimensions, initial_pressure, device=self.device)
        self.s_w = torch.full(self.dimensions, initial_sw, device=self.device)
        self.s_o = 1.0 - self.s_w
        self.prev_pressure = self.pressure.clone()
        self.prev_sw = self.s_w.clone()
        
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
        print(f"  Вязкость нефти/воды: {self.mu_oil*1e3:.1f}/{self.mu_water*1e3:.1f} сП")
        print(f"  Плотность нефти/воды: {self.rho_oil_ref}/{self.rho_water_ref} кг/м^3")
        print(f"  Сжимаемость: {self.oil_compressibility*1e6:.1e} 1/Па")
        print(f"  Капиллярное давление: {self.pc_scale/1e6:.2e} МПа, показатель {self.pc_exponent}")
        print(f"  Связанная водонасыщенность: {self.sw_cr}, остаточная нефтенасыщенность: {self.so_r}")
        print(f"  Тензоры флюидов размещены на: {self.device}")

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
        """
        Вычисляет плотность воды при заданном давлении.
        
        Args:
            pressure: Тензор давления
            
        Returns:
            Тензор плотности воды
        """
        return self.rho_water_ref * (1.0 + self.water_compressibility * (pressure - self.pressure_ref))

    def calc_oil_density(self, pressure):
        """
        Вычисляет плотность нефти при заданном давлении.
        
        Args:
            pressure: Тензор давления
            
        Returns:
            Тензор плотности нефти
        """
        return self.rho_oil_ref * (1.0 + self.oil_compressibility * (pressure - 1e5))

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

    # ---- Алиасы для обратной совместимости со старым кодом ----
    # (симулятор обращается к этим именам)
    calc_capillary_pressure = get_capillary_pressure
    calc_dpc_dsw            = get_capillary_pressure_derivative
