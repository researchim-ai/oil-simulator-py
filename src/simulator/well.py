import torch
import numpy as np
import math

class Well:
    """
    Класс для представления скважины в резервуаре.
    """
    
    def __init__(self, name, well_type, i, j, k, radius, control_type, control_value, reservoir_dimensions, injected_phase: str = 'water', rate_type: str = 'reservoir', surface_phase: str = None, limits: dict = None, bhp_min: float = None,
                 bhp_max: float = None, perforations=None, skin: float = 0.0, well_index: float = None,
                 ramp_days: float = 0.0, ref_depth: float = None):
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
        self.surface_phase = surface_phase  # for producers: 'oil'|'water'|'gas'|'liquid'
        # Лимиты фаз (м3/сутки) и минимальный забой (МПа)
        self.limits = limits or {}
        self.bhp_min = bhp_min  # МПа, если задано
        self.bhp_max = bhp_max  # МПа, предел для нагнетательных
        # Перфорации [(i,j,k), ...]; по умолчанию одна — в ячейке (i,j,k)
        self.perforations = [tuple(int(v) for v in pf) for pf in perforations] if perforations else [(i, j, k)]
        self.skin = float(skin)
        self.wi_override = well_index  # м^3; если не задан, считается по Писмену
        self.connection_wi = None
        self.ref_depth = ref_depth  # м, в координатах reservoir.z_centers; по умолчанию — верхняя перфорация
        
        nx, ny, nz = reservoir_dimensions
        self.cell_index = (i, j, k)
        # Совпадает с tensor(nx, ny, nz).view(-1): k — самая быстрая координата
        self.cell_index_flat = (i * ny + j) * nz + k
        
        # Определяем реальные размеры ячейки из reservoir_dimensions
        # Для Peaceman нужно знать dx, dy. Полагаем кубические ячейки и берём 1 м, если размеры не переданы.
        dx = dy = 1.0
        
        # Эффективный радиус ячейки для модели Писмана
        ro = 0.28 * ((dx**2 + dy**2)**0.5) / 2
        
        # Индекс скважины по модели Писмана (k заменим позже при гетерогенности)
        k_h_assumed = 100.0  # мД
        self.well_index = 2 * math.pi * k_h_assumed / (math.log(ro / radius) + 1e-12)
        # Диагностика последнего шага
        self.last_q_total = None           # м3/с (знак: + инжекция, - отбор)
        self.last_surface_rates = {}       # {'oil','water','gas','liquid'} в м3/сутки
        self.last_mode = None              # 'rate' | 'bhp'
        
        # Мягкий выход на режим (опционально, по умолчанию выключен)
        self.ramping_days = float(ramp_days)
        self.initial_pressure = 20.0  # МПа, используется только при ramping BHP
        
    def get_current_control(self, current_time_days: float):
        """
        Возвращает текущее значение управляющего параметра с учетом Ramping'а.
        
        Args:
            current_time_days: Текущее время симуляции в днях
            
        Returns:
            (control_type, control_value)
        """
        if self.ramping_days <= 0 or current_time_days >= self.ramping_days:
            return self.control_type, self.control_value
            
        # Коэффициент линейного нарастания [0, 1]
        # Начинаем с малого шага (0.05) чтобы избежать деления на 0 или нулевых рейтов
        factor = max(0.05, min(1.0, current_time_days / self.ramping_days))
        
        val = self.control_value
        
        if self.control_type == 'rate':
            # Для дебита просто масштабируем
            val = self.control_value * factor
        elif self.control_type == 'bhp':
            # Для давления интерполируем от начального пластового
            # P_curr = P_init + factor * (P_target - P_init)
            val = self.initial_pressure + factor * (self.control_value - self.initial_pressure)
            
        return self.control_type, val

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
                rate_type=w.get('rate_type', 'reservoir'),
                surface_phase=w.get('surface_phase'),
                limits=w.get('limits'),
                bhp_min=w.get('bhp_min'),
                bhp_max=w.get('bhp_max'),
                perforations=w.get('perforations'),
                skin=w.get('skin', 0.0),
                well_index=w.get('well_index'),
                ramp_days=w.get('ramp_days', 0.0),
                ref_depth=w.get('ref_depth'),
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
