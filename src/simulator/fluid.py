import torch

class Fluid:
    """
    Класс для хранения свойств флюидов (нефть, вода) и начальных условий в пласте.
    """
    def __init__(self, reservoir, initial_pressure, initial_sw, fluid_properties):
        """
        Инициализация свойств флюидов и начальных условий.

        :param reservoir: Экземпляр класса Reservoir.
        :param initial_pressure: Начальное давление в пласте (скаляр, в Паскалях).
        :param initial_sw: Начальная водонасыщенность (доли единицы).
        :param fluid_properties: Словарь со свойствами флюидов.
                                 {'oil': {'viscosity': mu_o, 'density': rho_o},
                                  'water': {'viscosity': mu_w, 'density': rho_w}}
        """
        self.reservoir = reservoir
        self.device = reservoir.device
        dims = (reservoir.nx, reservoir.ny, reservoir.nz)

        print("Инициализация флюидов и начальных условий...")

        # Свойства флюидов
        self.mu_oil = fluid_properties['oil']['viscosity']
        self.rho_oil = fluid_properties['oil']['density']
        self.mu_water = fluid_properties['water']['viscosity']
        self.rho_water = fluid_properties['water']['density']
        self.compressibility = fluid_properties.get('compressibility', 1e-5) # Общая сжимаемость, 1/бар

        # Начальные условия - тензоры давления и насыщенности
        self.pressure = torch.full(dims, initial_pressure, device=self.device, dtype=torch.float32)
        self.s_w = torch.full(dims, initial_sw, device=self.device, dtype=torch.float32) # Насыщенность водой
        self.s_o = 1.0 - self.s_w # Насыщенность нефтью

        print(f"  Начальное давление: {initial_pressure / 1e6:.2f} МПа")
        print(f"  Начальная водонасыщенность: {initial_sw}")
        print(f"  Вязкость нефти/воды: {self.mu_oil}/{self.mu_water} сП")
        print(f"  Плотность нефти/воды: {self.rho_oil}/{self.rho_water} кг/м^3")
        print(f"  Сжимаемость: {self.compressibility * 1e5} 1/Па")
        print(f"  Тензоры размещены на: {self.pressure.device}")
