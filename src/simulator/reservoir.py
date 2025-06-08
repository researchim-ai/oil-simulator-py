import torch

class Reservoir:
    """
    Класс для представления модели пласта.
    """
    def __init__(self, dimensions, grid_size, porosity, permeability, device):
        """
        Инициализация модели пласта.

        :param dimensions: Кортеж (nx, ny, nz) - количество ячеек по осям X, Y, Z.
        :param grid_size: Кортеж (dx, dy, dz) - физические размеры ячейки в метрах.
        :param porosity: Скалярное значение пористости (доли единицы).
        :param permeability: Проницаемость (скаляр или тензор, в миллидарси).
        :param device: Устройство для вычислений ('cpu' или 'cuda').
        """
        print("Создание модели пласта...")
        self.dimensions = dimensions
        self.grid_size = grid_size
        self.device = device

        self.nx, self.ny, self.nz = self.dimensions
        self.dx, self.dy, self.dz = self.grid_size

        print(f"  Размеры грида: {self.nx}x{self.ny}x{self.nz} ячеек")

        # Если проницаемость задана как скаляр, создаем однородный тензор
        if isinstance(permeability, (int, float)):
            self.permeability = torch.full(dimensions, float(permeability), device=self.device)
        else:
            self.permeability = permeability.to(self.device)

        # Если пористость задана как скаляр, создаем однородный тензор
        if isinstance(porosity, (int, float)):
            self.porosity = torch.full(dimensions, float(porosity), device=self.device)
        else:
            self.porosity = porosity.to(self.device)

        # Вычисляем объем ячеек
        self.grid_size = torch.tensor(grid_size, device=self.device)
        self.cell_volume = self.grid_size.prod()
        self.porous_volume = self.cell_volume * self.porosity

        print(f"  Пористость: {self.porosity.mean().item()}")
        print(f"  Проницаемость: {self.permeability.mean().item()} мД")
        print(f"  Тензоры размещены на: {self.permeability.device}")
