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
        :param permeability: Скалярное значение проницаемости (в миллидарси).
        :param device: Устройство torch (torch.device), на котором будут храниться тензоры.
        """
        self.nx, self.ny, self.nz = dimensions
        self.dx, self.dy, self.dz = grid_size
        self.device = device

        print("Создание модели пласта...")

        # Инициализация тензоров свойств пласта
        # torch.full создает тензор указанной формы, заполненный скалярным значением.
        self.porosity = torch.full(dimensions, porosity, device=self.device, dtype=torch.float32)
        self.permeability = torch.full(dimensions, permeability, device=self.device, dtype=torch.float32)

        print(f"  Размеры грида: {self.nx}x{self.ny}x{self.nz} ячеек")
        print(f"  Пористость: {porosity}")
        print(f"  Проницаемость: {permeability} мД")
        print(f"  Тензоры размещены на: {self.porosity.device}")
