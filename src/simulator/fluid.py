import torch

class Fluid:
    """
    Класс для хранения свойств флюидов (нефть, вода) и начальных условий в пласте.
    """
    def __init__(self, reservoir, p_init, s_w_init, mu_oil, mu_water, rho_oil, rho_water, c_oil, c_water, c_rock, sw_cr, so_r, nw, no, device):
        """
        Инициализация свойств флюида.

        :param reservoir: Экземпляр класса Reservoir.
        :param p_init: Начальное давление в пласте (в МПа).
        :param s_w_init: Начальная водонасыщенность (доли единицы).
        :param mu_oil: Вязкость нефти.
        :param mu_water: Вязкость воды.
        :param rho_oil: Плотность нефти.
        :param rho_water: Плотность воды.
        :param c_oil: Сжимаемость нефти.
        :param c_water: Сжимаемость воды.
        :param c_rock: Сжимаемость породы.
        :param sw_cr: Неснижаемая водонасыщенность.
        :param so_r: Остаточная нефтенасыщенность.
        :param nw: Экспонента Кори для воды.
        :param no: Экспонента Кори для нефти.
        :param device: Устройство для хранения тензоров.
        """
        self.device = device
        self.nx, self.ny, self.nz = reservoir.dimensions

        # Давление (Па)
        self.pressure = torch.full(reservoir.dimensions, p_init * 1e6, device=self.device)
        print(f"  Начальное давление: {p_init:.2f} МПа")

        # Насыщенность
        self.s_w = torch.full(reservoir.dimensions, s_w_init, device=self.device)
        self.s_o = torch.full(reservoir.dimensions, 1.0 - s_w_init, device=self.device)
        print(f"  Начальная водонасыщенность: {s_w_init}")

        # Вязкость (сП)
        self.mu_o = mu_oil
        self.mu_w = mu_water
        print(f"  Вязкость нефти/воды: {self.mu_o}/{self.mu_w} сП")

        # Плотность (кг/м^3)
        self.rho_o = rho_oil
        self.rho_w = rho_water
        print(f"  Плотность нефти/воды: {self.rho_o}/{self.rho_w} кг/м^3")
        
        # Сжимаемость (1/Па)
        self.c_o = c_oil
        self.c_w = c_water
        self.c_r = c_rock
        self.cf = self.s_w * self.c_w + self.s_o * self.c_o + self.c_r
        print(f"  Сжимаемость: {self.cf.mean().item():.1e} 1/Па")

        # Параметры ОФП (модель Кори)
        self.sw_cr = sw_cr  # Критическая водонасыщенность
        self.so_r = so_r    # Остаточная нефтенасыщенность
        self.nw = nw        # Степень для воды
        self.no = no        # Степень для нефти
        print(f"  Параметры ОФП: sw_cr={sw_cr}, so_r={so_r}, nw={nw}, no={no}")
        
        print(f"  Тензоры размещены на: {self.pressure.device}")

    def get_rel_perms(self, s_w):
        """
        Вычисляет относительные фазовые проницаемости для воды и нефти по модели Кори.
        :param s_w: Тензор текущей водонасыщенности.
        :return: (krw, kro) - кортеж с тензорами ОФП.
        """
        # Нормализуем насыщенность
        s_norm = (s_w - self.sw_cr) / (1 - self.sw_cr - self.so_r)
        s_norm = torch.clamp(s_norm, 0.0, 1.0) # Ограничиваем значения от 0 до 1

        # Рассчитываем ОФП по модели Кори
        krw = s_norm ** self.nw
        kro = (1 - s_norm) ** self.no
        
        return kro, krw
