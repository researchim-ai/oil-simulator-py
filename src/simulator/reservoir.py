import torch

class Reservoir:
    """
    Класс для представления модели пласта.
    """
    def __init__(self, config, device):
        """
        Инициализация модели пласта.

        :param config: Словарь с параметрами пласта.
        :param device: Устройство для вычислений ('cpu' или 'cuda').
        """
        print("Создание модели пласта...")
        self.dimensions = tuple(config['dimensions'])
        self.grid_size = tuple(config['grid_size'])
        self.device = device

        self.nx, self.ny, self.nz = self.dimensions
        self.dx, self.dy, self.dz = self.grid_size

        print(f"  Размеры грида: {self.nx}x{self.ny}x{self.nz} ячеек")

        permeability = config['permeability']
        # Если проницаемость задана как скаляр, создаем однородный тензор
        if isinstance(permeability, (int, float)):
            self.perm_h = torch.full(self.dimensions, float(permeability), device=self.device)
        else:
            self.perm_h = permeability.to(self.device)
        
        k_vertical_fraction = config.get('k_vertical_fraction', 1.0)
        self.perm_v = self.perm_h * k_vertical_fraction

        porosity = config['porosity']
        # Если пористость задана как скаляр, создаем однородный тензор
        if isinstance(porosity, (int, float)):
            self.porosity = torch.full(self.dimensions, float(porosity), device=self.device)
        else:
            self.porosity = porosity.to(self.device)

        # Вычисляем объем ячеек
        self.grid_size = torch.tensor(self.grid_size, device=self.device)
        self.cell_volume = self.grid_size.prod()
        self.porous_volume = self.cell_volume * self.porosity

        print(f"  Пористость: {self.porosity.mean().item()}")
        print(f"  Горизонтальная проницаемость: {self.perm_h.mean().item()} мД")
        if k_vertical_fraction != 1.0:
            print(f"  Вертикальная проницаемость: {self.perm_v.mean().item()} мД")
        print(f"  Тензоры размещены на: {self.perm_h.device}")
