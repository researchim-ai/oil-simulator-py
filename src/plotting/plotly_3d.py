"""
Интерактивная 3D визуализация результатов симуляции через Plotly.
Поддерживает объёмные изоповерхности, срезы, анимацию и экспорт в HTML.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Optional, Dict, Tuple


class Plotly3DVisualizer:
    """
    Класс для создания интерактивных 3D визуализаций результатов симуляции.
    """
    
    def __init__(self, reservoir, device='cpu'):
        """
        Инициализация визуализатора.
        
        Args:
            reservoir: Объект резервуара с информацией о сетке
            device: Устройство для вычислений (для получения координат)
        """
        self.reservoir = reservoir
        self.device = device
        nx, ny, nz = reservoir.dimensions
        
        # Вычисляем координаты центров ячеек
        if hasattr(reservoir, 'grid_size'):
            grid_size = reservoir.grid_size.detach().cpu().numpy() if hasattr(reservoir.grid_size, 'detach') else np.array(reservoir.grid_size)
            dx, dy, dz = grid_size
        else:
            dx = dy = dz = 1.0
        
        # Координаты центров ячеек
        self.x_coords = np.arange(dx/2, nx * dx, dx)
        self.y_coords = np.arange(dy/2, ny * dy, dy)
        self.z_coords = np.arange(dz/2, nz * dz, dz)
        
        # Сетка для объёмной визуализации
        # Для больших сеток используем разреженную выборку для производительности
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        
        # Создаём полную сетку только при необходимости
        self._X = None
        self._Y = None
        self._Z = None
    
    def _get_meshgrid(self, subsample: int = 1):
        """Создаёт или возвращает кэшированную сетку координат."""
        if self._X is None or subsample > 1:
            x_sub = self.x_coords[::subsample]
            y_sub = self.y_coords[::subsample]
            z_sub = self.z_coords[::subsample]
            X, Y, Z = np.meshgrid(x_sub, y_sub, z_sub, indexing='ij')
            if subsample == 1:
                self._X, self._Y, self._Z = X, Y, Z
            return X, Y, Z
        return self._X, self._Y, self._Z
    
    def create_volume_plot(
        self,
        pressure: np.ndarray,
        sw: np.ndarray,
        sg: Optional[np.ndarray] = None,
        iso_value_p: Optional[float] = None,
        iso_value_sw: Optional[float] = None,
        iso_value_sg: Optional[float] = None,
        opacity: float = 0.3,
        show_slices: bool = True,
        title: str = "3D Визуализация резервуара",
        subsample: int = 1
    ) -> go.Figure:
        """
        Создаёт интерактивную 3D визуализацию с изоповерхностями и срезами.
        
        Args:
            pressure: Массив давления (nx, ny, nz)
            sw: Массив водонасыщенности (nx, ny, nz)
            sg: Массив газонасыщенности (nx, ny, nz), опционально
            iso_value_p: Значение изоповерхности для давления (None = автоматически)
            iso_value_sw: Значение изоповерхности для Sw (None = автоматически)
            iso_value_sg: Значение изоповерхности для Sg (None = автоматически)
            opacity: Прозрачность изоповерхностей
            show_slices: Показывать ли срезы
            title: Заголовок графика
            
        Returns:
            go.Figure: Интерактивная фигура Plotly
        """
        nx, ny, nz = pressure.shape
        
        # Для больших сеток используем разреженную выборку
        if subsample > 1:
            pressure_sub = pressure[::subsample, ::subsample, ::subsample]
            sw_sub = sw[::subsample, ::subsample, ::subsample]
            if sg is not None:
                sg_sub = sg[::subsample, ::subsample, ::subsample]
            else:
                sg_sub = None
            X, Y, Z = self._get_meshgrid(subsample)
        else:
            pressure_sub = pressure
            sw_sub = sw
            sg_sub = sg
            X, Y, Z = self._get_meshgrid(1)
        
        # Нормализуем данные для лучшей визуализации
        p_norm = (pressure_sub - pressure_sub.min()) / (pressure_sub.max() - pressure_sub.min() + 1e-12)
        sw_norm = sw_sub
        
        fig = go.Figure()
        
        # Автоматический выбор изоповерхностей
        if iso_value_p is None:
            iso_value_p = np.percentile(p_norm, 50)
        if iso_value_sw is None:
            iso_value_sw = np.percentile(sw_norm, 50)
        
        # Изоповерхность давления
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=p_norm.flatten(),
            isomin=iso_value_p - 0.1,
            isomax=iso_value_p + 0.1,
            surface_count=1,
            colorscale='Jet',
            name='Давление',
            opacity=opacity,
            showscale=True,
            colorbar=dict(title="Давление (норм.)", x=1.1)
        ))
        
        # Изоповерхность водонасыщенности
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=sw_norm.flatten(),
            isomin=iso_value_sw - 0.1,
            isomax=iso_value_sw + 0.1,
            surface_count=1,
            colorscale='Viridis',
            name='Водонасыщенность',
            opacity=opacity,
            showscale=True,
            colorbar=dict(title="Sw", x=1.2)
        ))
        
        # Газонасыщенность (если есть)
        if sg_sub is not None and np.any(sg_sub > 1e-6):
            sg_norm = sg_sub
            if iso_value_sg is None:
                iso_value_sg = np.percentile(sg_norm[sg_norm > 1e-6], 50) if np.any(sg_norm > 1e-6) else 0.1
            
            fig.add_trace(go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=sg_norm.flatten(),
                isomin=max(0.01, iso_value_sg - 0.05),
                isomax=iso_value_sg + 0.05,
                surface_count=1,
                colorscale='Plasma',
                name='Газонасыщенность',
                opacity=opacity * 0.7,
                showscale=True,
                colorbar=dict(title="Sg", x=1.3)
            ))
        
        # Срезы (если включены)
        if show_slices:
            # Срез по середине Z
            z_mid = pressure_sub.shape[2] // 2
            fig.add_trace(go.Surface(
                x=X[:, :, z_mid],
                y=Y[:, :, z_mid],
                z=Z[:, :, z_mid],
                surfacecolor=sw_sub[:, :, z_mid],
                colorscale='Viridis',
                name=f'Срез Z={z_mid}',
                opacity=0.6,
                showscale=False
            ))
        
        # Настройка осей и камеры
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (м)",
                yaxis_title="Y (м)",
                zaxis_title="Z (м)",
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
    
    def create_slice_viewer(
        self,
        pressure: np.ndarray,
        sw: np.ndarray,
        sg: Optional[np.ndarray] = None,
        z_slice: Optional[int] = None,
        title: str = "Срезы резервуара"
    ) -> go.Figure:
        """
        Создаёт интерактивный просмотрщик срезов с возможностью изменения слоя.
        
        Args:
            pressure: Массив давления (nx, ny, nz)
            sw: Массив водонасыщенности (nx, ny, nz)
            sg: Массив газонасыщенности (nx, ny, nz), опционально
            z_slice: Индекс Z-среза (None = средний)
            title: Заголовок
            
        Returns:
            go.Figure: Интерактивная фигура с слайдером
        """
        nx, ny, nz = pressure.shape
        
        if z_slice is None:
            z_slice = nz // 2
        
        has_gas = sg is not None and np.any(sg > 1e-6)
        n_plots = 3 if has_gas else 2
        
        fig = make_subplots(
            rows=1, cols=n_plots,
            subplot_titles=('Давление (МПа)', 'Водонасыщенность', 'Газонасыщенность' if has_gas else None),
            specs=[[{'type': 'surface'}] * n_plots]
        )
        
        # Получаем сетку координат
        X, Y, Z = self._get_meshgrid(1)
        
        # Давление
        fig.add_trace(
            go.Surface(
                x=X[:, :, z_slice],
                y=Y[:, :, z_slice],
                z=pressure[:, :, z_slice] / 1e6,
                colorscale='Jet',
                name='Давление',
                colorbar=dict(title="МПа", x=0.33 if n_plots == 3 else 0.5)
            ),
            row=1, col=1
        )
        
        # Водонасыщенность
        fig.add_trace(
            go.Surface(
                x=X[:, :, z_slice],
                y=Y[:, :, z_slice],
                z=sw[:, :, z_slice],
                colorscale='Viridis',
                name='Sw',
                colorbar=dict(title="Sw", x=0.67 if n_plots == 3 else 1.0)
            ),
            row=1, col=2
        )
        
        # Газонасыщенность
        if has_gas:
            fig.add_trace(
                go.Surface(
                    x=X[:, :, z_slice],
                    y=Y[:, :, z_slice],
                    z=sg[:, :, z_slice],
                    colorscale='Plasma',
                    name='Sg',
                    colorbar=dict(title="Sg", x=1.0)
                ),
                row=1, col=3
            )
        
        # Слайдер для изменения Z-среза
        steps = []
        for k in range(nz):
            z_vals = [
                [pressure[:, :, k] / 1e6],
                [sw[:, :, k]],
            ]
            if has_gas:
                z_vals.append([sg[:, :, k]])
            step = dict(
                method='restyle',
                args=[{'z': z_vals}],
                label=f'Z={k}'
            )
            steps.append(step)
        
        sliders = [dict(
            active=z_slice,
            currentvalue={"prefix": "Слой Z: "},
            pad={"t": 50},
            steps=steps
        )]
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (м)",
                yaxis_title="Y (м)",
                zaxis_title="Значение"
            ),
            scene2=dict(
                xaxis_title="X (м)",
                yaxis_title="Y (м)",
                zaxis_title="Значение"
            ),
            scene3=dict(
                xaxis_title="X (м)",
                yaxis_title="Y (м)",
                zaxis_title="Значение"
            ) if has_gas else {},
            width=1400 if has_gas else 1000,
            height=600,
            sliders=sliders
        )
        
        return fig
    
    def create_animation_frames(
        self,
        data_sequence: list,
        field_name: str = 'pressure',
        title: str = "Анимация симуляции"
    ) -> go.Figure:
        """
        Создаёт анимацию по последовательности шагов.
        
        Args:
            data_sequence: Список словарей с ключами 'pressure', 'sw', 'sg', 'time'
            field_name: Поле для анимации ('pressure', 'sw', 'sg')
            title: Заголовок
            
        Returns:
            go.Figure: Анимированная фигура
        """
        # Выбираем первый кадр для определения формы
        first_frame = data_sequence[0]
        data = first_frame[field_name]
        nx, ny, nz = data.shape
        
        # Для больших сеток используем разреженную выборку
        subsample = max(1, min(4, (nx * ny * nz // 500000) ** (1/3)))  # Примерно 500k точек
        if subsample > 1:
            data_sub = data[::subsample, ::subsample, ::subsample]
            X, Y, Z = self._get_meshgrid(subsample)
        else:
            data_sub = data
            X, Y, Z = self._get_meshgrid(1)
        
        # Создаём фигуру с первым кадром
        fig = go.Figure()
        
        # Изоповерхность для первого кадра
        if field_name == 'pressure':
            data_norm = (data_sub - data_sub.min()) / (data_sub.max() - data_sub.min() + 1e-12)
            colorscale = 'Jet'
            title_suffix = 'Давление (МПа)'
        elif field_name == 'sw':
            data_norm = data_sub
            colorscale = 'Viridis'
            title_suffix = 'Водонасыщенность'
        else:  # sg
            data_norm = data_sub
            colorscale = 'Plasma'
            title_suffix = 'Газонасыщенность'
        
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=data_norm.flatten(),
            isomin=np.percentile(data_norm, 30),
            isomax=np.percentile(data_norm, 70),
            surface_count=1,
            colorscale=colorscale,
            name=title_suffix,
            opacity=0.4
        ))
        
        # Создаём кадры анимации
        frames = []
        for i, frame_data in enumerate(data_sequence):
            data = frame_data[field_name]
            if subsample > 1:
                data = data[::subsample, ::subsample, ::subsample]
            if field_name == 'pressure':
                data_norm = (data - data.min()) / (data.max() - data.min() + 1e-12)
            else:
                data_norm = data
            
            frames.append(go.Frame(
                data=[go.Isosurface(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=data_norm.flatten(),
                    isomin=np.percentile(data_norm, 30),
                    isomax=np.percentile(data_norm, 70),
                    surface_count=1,
                    colorscale=colorscale,
                    opacity=0.4
                )],
                name=f"frame_{i}",
                traces=[0]
            ))
        
        fig.frames = frames
        
        # Кнопки управления анимацией
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (м)",
                yaxis_title="Y (м)",
                zaxis_title="Z (м)",
                aspectmode='data'
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '▶',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300}
                        }]
                    },
                    {
                        'label': '⏸',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            width=1200,
            height=800
        )
        
        return fig
    
    def save_html(
        self,
        fig: go.Figure,
        filepath: str,
        include_plotlyjs: str = 'cdn'
    ):
        """
        Сохраняет фигуру в HTML файл.
        
        Args:
            fig: Фигура Plotly
            filepath: Путь к файлу
            include_plotlyjs: 'cdn' (загрузка из интернета) или 'inline' (встроенный)
        """
        fig.write_html(filepath, include_plotlyjs=include_plotlyjs)
        print(f"3D визуализация сохранена в {filepath}")

