import torch
import numpy as np

class Well:
    """
    Класс для представления одной скважины с учетом модели Писмена.
    """
    def __init__(self, name, well_type, coordinates, radius, control_type, control_value, reservoir_dimensions):
        """
        Инициализация скважины.
        :param name: Имя скважины (str).
        :param well_type: Тип скважины ('producer' или 'injector') (str).
        :param coordinates: Кортеж (i, j, k) с индексами ячейки скважины.
        :param radius: Радиус скважины в метрах (float).
        :param control_type: Тип контроля ('rate' или 'bhp') (str).
        :param control_value: Значение контроля (дебит в м^3/сутки или давление в МПа).
        :param reservoir_dimensions: Кортеж (nx, ny, nz) с размерами пласта.
        """
        self.name = name
        self.type = well_type
        self.coordinates = coordinates
        self.radius = radius
        self.control_type = control_type
        self.control_value = control_value
        
        assert len(coordinates) == 3, "Координаты должны быть в формате (i, j, k)"
        self.i, self.j, self.k = coordinates
        
        nx, ny, nz = reservoir_dimensions
        assert 0 <= self.i < nx and 0 <= self.j < ny and 0 <= self.k < nz, f"Координаты скважины {name} выходят за пределы пласта."

        # Рассчитываем одномерный индекс ячейки
        self.cell_index_flat = self.i * ny * nz + self.j * nz + self.k
        
        self.well_index = None # Будет рассчитан позже

        print(f"  Создана скважина '{self.name}':")
        print(f"    Тип: {self.type}, Расположение: ({self.i}, {self.j}, {self.k})")
        print(f"    Контроль: {self.control_type}, Значение: {self.control_value}")
        print(f"    Индекс скважины '{self.name}': {self.cell_index_flat:.4e}")

    def calculate_well_index(self, reservoir):
        """
        Рассчитывает индекс скважины по модели Писмена.
        """
        perm = reservoir.perm_h # Используем горизонтальную проницаемость
        kx = perm[self.i, self.j, self.k].item()
        ky = perm[self.i, self.j, self.k].item() # Изотропия в плоскости xy
        kv = perm[self.i, self.j, self.k].item()

        dx = reservoir.dx
        dy = reservoir.dy
        dz = reservoir.dz

        # Формула эквивалентного радиуса Писмена для анизотропного случая
        term_x = (ky / kx)**0.25
        term_y = (kx / ky)**0.25
        req = 0.28 * np.sqrt( (dx**2 * term_x**2) + (dy**2 * term_y**2) ) / (term_x + term_y)

        # Индекс скважины. Skin-фактор S принимается равным 0.
        self.well_index = (2 * np.pi * np.sqrt(kx * ky) * dz) / np.log(req / self.radius)
        print(f"    Индекс скважины '{self.name}': {self.well_index:.4e}")


class WellManager:
    """
    Класс для управления всеми скважинами в симуляции.
    """
    def __init__(self, well_configs, reservoir):
        """
        Инициализация и создание скважин из конфигурации.
        """
        self.wells = []
        self.num_wells = len(well_configs)
        self._well_indices_flat = torch.tensor([
            w['coordinates'][0] * reservoir.ny * reservoir.nz + 
            w['coordinates'][1] * reservoir.nz + 
            w['coordinates'][2] 
            for w in well_configs
        ], dtype=torch.long)

        print("Создание менеджера скважин...")
        for well_info in well_configs:
            control = well_info.get('control', {})
            well = Well(
                name=well_info['name'],
                well_type=well_info['type'],
                coordinates=tuple(well_info['coordinates']),
                radius=well_info['radius'],
                control_type=control.get('type'),
                control_value=control.get('value'),
                reservoir_dimensions=reservoir.dimensions
            )
            well.calculate_well_index(reservoir)
            self.wells.append(well)
            print(f"  > Скважина '{well.name}' добавлена в менеджер.")
        print("Менеджер скважин успешно создан.")

    def add_well(self, well):
        # Этот метод больше не нужен, так как вся логика в __init__
        pass

    def get_wells(self):
        return self.wells

    def get_well_indices_flat(self):
        return self._well_indices_flat.to(self.wells[0].well_index.device if self.wells else 'cpu')
