import torch
import numpy as np
import math

class Well:
    """
    Класс для представления скважины в резервуаре.
    """
    
    def __init__(self, name, well_type, i, j, k, radius, control_type, control_value, reservoir_dimensions, injected_phase: str = 'water', rate_type: str = 'reservoir'):
        """
        Инициализирует скважину.
        
        Args:
            name: Имя скважины
            well_type: Тип скважины ('injector' или 'producer')
            i, j, k: Координаты скважины в сетке
            radius: Радиус скважины в метрах
            control_type: Тип контроля ('rate' или 'bhp')
            control_value: Значение контроля (дебит в м³/день или забойное давление в МПа)
            reservoir_dimensions: Размеры резервуара (nx, ny, nz)
        """
        self.name = name
        self.type = well_type
        self.i = i
        self.j = j
        self.k = k
        self.radius = radius
        self.control_type = control_type
        self.control_value = control_value
        self.injected_phase = injected_phase
        self.rate_type = rate_type  # 'reservoir' | 'surface'
        
        nx, ny, nz = reservoir_dimensions
        self.cell_index = (i, j, k)
        self.cell_index_flat = i + j * nx + k * nx * ny
        
        # Определяем реальные размеры ячейки из reservoir_dimensions
        # Для Peaceman нужно знать dx, dy. Полагаем кубические ячейки и берём 1 м, если размеры не переданы.
        dx = dy = 1.0
        
        # Эффективный радиус ячейки для модели Писмана
        ro = 0.28 * ((dx**2 + dy**2)**0.5) / 2
        
        # Индекс скважины по модели Писмана (k заменим позже при гетерогенности)
        k_h_assumed = 100.0  # мД
        self.well_index = 2 * math.pi * k_h_assumed / (math.log(ro / radius) + 1e-12)
        
    def __str__(self):
        return f"Well(name={self.name}, type={self.type}, pos=({self.i},{self.j},{self.k}), control={self.control_type}:{self.control_value})"

class WellManager:
    """
    Класс для управления всеми скважинами в симуляции.
    """
    def __init__(self, well_configs, reservoir):
        """
        Инициализирует менеджер скважин.
        
        Args:
            well_configs: Список конфигураций скважин
            reservoir: Объект пласта
        """
        self.wells = []
        
        print("Создание менеджера скважин...")
        for w in well_configs:
            # Поддержка как старого формата (coordinates), так и нового формата (i,j,k)
            if 'coordinates' in w:
                i, j, k = w['coordinates']
            elif 'i' in w and 'j' in w and 'k' in w:
                i, j, k = w['i'], w['j'], w['k']
            else:
                raise ValueError(f"Неверный формат координат скважины: {w}")
            
            # Поддержка как старого формата (control), так и нового формата (control_type, control_value)
            if 'control' in w:
                control_type = w['control']['type']
                control_value = w['control']['value']
            elif 'control_type' in w and 'control_value' in w:
                control_type = w['control_type']
                control_value = w['control_value']
            else:
                raise ValueError(f"Неверный формат управления скважиной: {w}")
            
            # Создаем и добавляем скважину
            well = Well(
                name=w['name'],
                well_type=w['type'],
                i=i, j=j, k=k,
                radius=w['radius'],
                control_type=control_type,
                control_value=control_value,
                reservoir_dimensions=reservoir.dimensions,
                injected_phase=w.get('injected_phase', 'water'),
                rate_type=w.get('rate_type', 'reservoir')
            )
            self.wells.append(well)
            print(f"  > Скважина '{well.name}' добавлена в менеджер.")
        
        print(f"Менеджер скважин успешно создан. Всего скважин: {len(self.wells)}")

    def add_well(self, well):
        # Этот метод больше не нужен, так как вся логика в __init__
        pass

    def get_wells(self):
        """
        Возвращает список всех скважин.
        
        Returns:
            Список объектов Well
        """
        return self.wells

    def get_well_indices_flat(self):
        """Возвращает тензор с линейными индексами ячеек, в которых расположены скважины."""
        if not self.wells:
            return torch.tensor([], dtype=torch.long)
        idx_list = [w.cell_index_flat for w in self.wells]
        return torch.tensor(idx_list, dtype=torch.long)
