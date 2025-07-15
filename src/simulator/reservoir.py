import torch
import numpy as np

class Reservoir:
    """
    Класс для представления модели пласта.
    """
    def __init__(self, config, device=None):
        """
        Инициализация модели пласта.

        :param config: Словарь с параметрами пласта.
        :param device: Устройство для вычислений ('cpu' или 'cuda').
        """
        self.device = device if device is not None else torch.device('cpu')
        
        # Размеры пласта
        self.dimensions = config.get('dimensions', [10, 10, 1])
        self.nx, self.ny, self.nz = self.dimensions
        
        # Размеры ячеек
        self.grid_size = tuple(config.get('grid_size', [10.0, 10.0, 10.0]))
        self.dx, self.dy, self.dz = self.grid_size
        
        # Пористость (может быть скаляром или массивом)
        porosity_value = config.get('porosity', 0.2)
        self.porosity = torch.full(self.dimensions, porosity_value, device=self.device)
        # Сохраняем исходную (референсную) пористость для расчёта phi(P)
        # Она соответствует пористости при референсном давлении P_ref = 1 атм (≈ 1e5 Па).
        self.porosity_ref = self.porosity.clone()
        
        # Проницаемость
        k_h = config.get('permeability', 100.0)  # мД, горизонтальная проницаемость
        k_v_fraction = config.get('k_vertical_fraction', 0.1)  # доля вертикальной проницаемости
        k_v = k_h * k_v_fraction  # мД, вертикальная проницаемость
        
        # Конвертируем мД -> м^2
        md_to_m2 = 9.869233e-16
        k_h_si = k_h * md_to_m2
        k_v_si = k_v * md_to_m2
        
        # Создаем тензоры проницаемости в СИ
        self.permeability_x = torch.full(self.dimensions, k_h_si, device=self.device)
        self.permeability_y = torch.full(self.dimensions, k_h_si, device=self.device)
        self.permeability_z = torch.full(self.dimensions, k_v_si, device=self.device)
        
        # Сжимаемость породы
        self.rock_compressibility = config.get('c_rock', 1e-5) / 1e6  # 1/МПа -> 1/Па
        
        # Вычисляем объем ячеек
        self.grid_size = torch.tensor(self.grid_size, device=self.device)
        self.cell_volume = self.grid_size.prod()
        self.porous_volume = self.cell_volume * self.porosity

        # Выводим информацию
        print("Создание модели пласта...")
        print(f"  Размеры грида: {self.nx}x{self.ny}x{self.nz} ячеек")
        print(f"  Пористость: {porosity_value}")
        print(f"  Горизонтальная проницаемость: {k_h} мД")
        print(f"  Вертикальная проницаемость: {k_v} мД")
        print(f"  Тензоры размещены на: {self.device}")

    @property
    def permeability_tensors(self):
        """
        Возвращает тензоры проницаемости.
        
        Returns:
            Кортеж из трех тензоров проницаемости (k_x, k_y, k_z) в мД
        """
        return self.permeability_x, self.permeability_y, self.permeability_z
        
    def get_cell_indices(self, i, j, k):
        """
        Возвращает линейный индекс ячейки по трехмерным координатам.
        
        Args:
            i, j, k: Координаты ячейки
            
        Returns:
            Линейный индекс ячейки
        """
        return i + j * self.nx + k * self.nx * self.ny

    @classmethod
    def from_config(cls, cfg: dict, device=None):
        """Создаёт объект Reservoir из полной конфигурации симуляции или из
        отдельного блока `reservoir`.

        Args:
            cfg: Словарь полной конфигурации (может содержать ключ "reservoir")
                 или непосредственно словарь параметров пласта.
            device: PyTorch-устройство (cpu / cuda). По умолчанию CPU.
        """
        # Если нам передали целиком конфиг симулятора – возьмём подпункт
        res_cfg = cfg.get("reservoir", cfg)
        return cls(config=res_cfg, device=device)
