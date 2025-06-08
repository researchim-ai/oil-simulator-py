import torch

class Fluid:
    """
    Класс для хранения свойств флюидов (нефть, вода) и начальных условий в пласте.
    """
    def __init__(self, reservoir, mu_oil, mu_water, rho_oil, rho_water, 
                 initial_pressure, initial_sw, cf, 
                 sw_cr, so_r, nw, no, krw_end, kro_end):
        """
        Инициализация свойств флюидов и начальных условий.

        :param reservoir: Экземпляр класса Reservoir.
        :param mu_oil: Вязкость нефти.
        :param mu_water: Вязкость воды.
        :param rho_oil: Плотность нефти.
        :param rho_water: Плотность воды.
        :param initial_pressure: Начальное давление в пласте (скаляр, в Паскалях).
        :param initial_sw: Начальная водонасыщенность (доли единицы).
        :param cf: Сжимаемость флюида.
        :param sw_cr: Неснижаемая водонасыщенность.
        :param so_r: Остаточная нефтенасыщенность.
        :param nw: Экспонента Кори для воды.
        :param no: Экспонента Кори для нефти.
        :param krw_end: Конечная ОФП для воды.
        :param kro_end: Конечная ОФП для нефти.
        """
        self.mu_o = mu_oil
        self.mu_w = mu_water
        self.rho_o = rho_oil
        self.rho_w = rho_water
        self.cf = cf # Сжимаемость флюида

        # Параметры для модели ОФП (Кори)
        self.sw_cr = sw_cr # Неснижаемая водонасыщенность
        self.so_r = so_r   # Остаточная нефтенасыщенность
        self.nw = nw       # Экспонента Кори для воды
        self.no = no       # Экспонента Кори для нефти
        self.krw_end = krw_end # Конечная ОФП для воды
        self.kro_end = kro_end # Конечная ОФП для нефти
        
        # Создаем тензоры на том же устройстве, что и пласт
        device = reservoir.device
        nx, ny, nz = reservoir.dimensions

        self.pressure = torch.full((nx, ny, nz), initial_pressure, dtype=torch.float32, device=device)
        self.s_w = torch.full((nx, ny, nz), initial_sw, dtype=torch.float32, device=device)
        self.s_o = torch.full((nx, ny, nz), 1.0 - initial_sw, dtype=torch.float32, device=device)

        print("Инициализация флюидов и начальных условий...")

        print(f"  Начальное давление: {initial_pressure / 1e6:.2f} МПа")
        print(f"  Начальная водонасыщенность: {initial_sw}")
        print(f"  Вязкость нефти/воды: {self.mu_o}/{self.mu_w} сП")
        print(f"  Плотность нефти/воды: {self.rho_o}/{self.rho_w} кг/м^3")
        print(f"  Сжимаемость: {self.cf * 1e5} 1/Па")
        print(f"  Параметры ОФП: sw_cr={self.sw_cr}, so_r={self.so_r}, nw={self.nw}, no={self.no}")
        print(f"  Тензоры размещены на: {self.pressure.device}")

    def get_rel_perms(self, s_w):
        """
        Вычисляет относительные фазовые проницаемости для воды и нефти по модели Кори.
        :param s_w: Тензор текущей водонасыщенности.
        :return: (krw, kro) - кортеж с тензорами ОФП.
        """
        # Нормализация насыщенности
        s_norm = (s_w - self.sw_cr) / (1 - self.sw_cr - self.so_r)
        s_norm = torch.clamp(s_norm, 0.0, 1.0)

        # Расчет ОФП
        krw = self.krw_end * (s_norm ** self.nw)
        kro = self.kro_end * ((1 - s_norm) ** self.no)

        return krw, kro
