import torch

class Fluid:
    """
    Класс для хранения свойств флюидов (нефть, вода) и начальных условий в пласте.
    """
    def __init__(self, reservoir, config, device):
        """
        Инициализация свойств флюида.

        :param reservoir: Экземпляр класса Reservoir.
        :param config: Словарь с параметрами флюидов и начальными условиями.
        :param device: Устройство для хранения тензоров.
        """
        self.device = device
        self.nx, self.ny, self.nz = reservoir.dimensions
        
        p_init = config['pressure']
        s_w_init = config['s_w']
        
        # Давление (Па)
        self.pressure = torch.full(reservoir.dimensions, p_init * 1e6, device=self.device)
        print(f"  Начальное давление: {p_init:.2f} МПа")

        # Насыщенность
        self.s_w = torch.full(reservoir.dimensions, s_w_init, device=self.device)
        self.s_o = torch.full(reservoir.dimensions, 1.0 - s_w_init, device=self.device)
        print(f"  Начальная водонасыщенность: {s_w_init}")

        # Вязкость (сП)
        self.mu_o = config['mu_oil']
        self.mu_w = config['mu_water']
        print(f"  Вязкость нефти/воды: {self.mu_o}/{self.mu_w} сП")

        # Плотность (кг/м^3)
        self.rho_o = config['rho_oil']
        self.rho_w = config['rho_water']
        print(f"  Плотность нефти/воды: {self.rho_o}/{self.rho_w} кг/м^3")
        
        # Сжимаемость (1/Па)
        self.c_o = config['c_oil']
        self.c_w = config['c_water']
        self.c_r = config['c_rock']
        self.cf = self.s_w * self.c_w + self.s_o * self.c_o + self.c_r
        print(f"  Сжимаемость: {self.cf.mean().item():.1e} 1/Па")

        # Параметры ОФП (модель Кори)
        rel_perm_params = config['relative_permeability']
        self.sw_cr = rel_perm_params['sw_cr']  # Критическая водонасыщенность
        self.so_r = rel_perm_params['so_r']    # Остаточная нефтенасыщенность
        self.nw = rel_perm_params['nw']        # Степень для воды
        self.no = rel_perm_params['no']        # Степень для нефти
        print(f"  Параметры ОФП: sw_cr={self.sw_cr}, so_r={self.so_r}, nw={self.nw}, no={self.no}")

        # Параметры капиллярного давления
        cap_params = config.get('capillary_pressure', {'pc_scale': 0.0, 'pc_exponent': 1.0})
        self.pc_scale = cap_params.get('pc_scale', 0.0)
        self.pc_exponent = cap_params.get('pc_exponent', 1.0)
        if self.pc_scale > 0:
            print(f"  Параметры капиллярного давления: pc_scale={self.pc_scale}, pc_exponent={self.pc_exponent}")
        
        print(f"  Тензоры размещены на: {self.pressure.device}")

    def _get_normalized_saturation(self, s_w):
        """
        Вычисляет нормализованную водонасыщенность.
        """
        s_norm = (s_w - self.sw_cr) / (1 - self.sw_cr - self.so_r)
        return torch.clamp(s_norm, 0.0, 1.0)

    def get_rel_perms(self, s_w):
        """
        Вычисляет относительные фазовые проницаемости для воды и нефти по модели Кори.
        :param s_w: Тензор текущей водонасыщенности.
        :return: (krw, kro) - кортеж с тензорами ОФП.
        """
        s_norm = self._get_normalized_saturation(s_w)
        # Рассчитываем ОФП по модели Кори
        krw = s_norm ** self.nw
        kro = (1 - s_norm) ** self.no
        
        return kro, krw

    def get_capillary_pressure(self, s_w):
        """
        Вычисляет капиллярное давление по простой степенной модели.
        :param s_w: Тензор текущей водонасыщенности.
        :return: Тензор капиллярного давления (в Па).
        """
        if self.pc_scale == 0.0:
            return torch.zeros_like(s_w)
            
        s_norm = self._get_normalized_saturation(s_w)
        # Простоая степенная модель Pc = scale * s_norm^exponent
        # Добавляем эпсилон для стабильности, если s_norm = 0
        pc = self.pc_scale * (s_norm + 1e-6) ** self.pc_exponent
        return pc
