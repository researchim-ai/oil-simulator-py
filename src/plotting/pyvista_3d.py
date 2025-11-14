"""
3D визуализация результатов симуляции через PyVista.
Использует VTK для эффективной работы с большими сетками.
"""

import numpy as np
from typing import Optional
import os

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None


class PyVista3DVisualizer:
    """
    Класс для создания 3D визуализаций через PyVista.
    Работает с VTK данными и может экспортировать в HTML.
    """
    
    def __init__(self, reservoir, device='cpu'):
        """
        Инициализация визуализатора.
        
        Args:
            reservoir: Объект резервуара с информацией о сетке
            device: Устройство для вычислений
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista не установлен. Установите: pip install pyvista")
        
        self.reservoir = reservoir
        self.device = device
        self.nx, self.ny, self.nz = reservoir.dimensions
        
        # Настройка PyVista
        pv.set_plot_theme("dark")  # Тёмная тема для красоты
        
    def create_volume_plot(
        self,
        pressure: np.ndarray,
        sw: np.ndarray,
        sg: Optional[np.ndarray] = None,
        title: str = "3D Визуализация резервуара",
        subsample: int = 1
    ) -> pv.Plotter:
        """
        Создаёт объёмную 3D визуализацию через PyVista.
        
        Args:
            pressure: Массив давления (nx, ny, nz)
            sw: Массив водонасыщенности (nx, ny, nz)
            sg: Массив газонасыщенности (nx, ny, nz), опционально
            title: Заголовок
            subsample: Шаг прореживания для больших сеток
            
        Returns:
            pv.Plotter: Объект Plotter PyVista
        """
        nx, ny, nz = pressure.shape
        
        # Применяем subsample если нужно
        if subsample > 1:
            pressure_sub = pressure[::subsample, ::subsample, ::subsample]
            sw_sub = sw[::subsample, ::subsample, ::subsample]
            if sg is not None:
                sg_sub = sg[::subsample, ::subsample, ::subsample]
            else:
                sg_sub = None
        else:
            pressure_sub = pressure
            sw_sub = sw
            sg_sub = sg
        
        # Создаём структурированную сетку PyVista
        grid = pv.StructuredGrid()
        
        # Получаем координаты центров ячеек
        if hasattr(self.reservoir, 'x_centers') and hasattr(self.reservoir, 'y_centers') and hasattr(self.reservoir, 'z_centers'):
            x_coords = self.reservoir.x_centers.detach().cpu().numpy() if hasattr(self.reservoir.x_centers, 'detach') else np.array(self.reservoir.x_centers)
            y_coords = self.reservoir.y_centers.detach().cpu().numpy() if hasattr(self.reservoir.y_centers, 'detach') else np.array(self.reservoir.y_centers)
            z_coords = self.reservoir.z_centers.detach().cpu().numpy() if hasattr(self.reservoir.z_centers, 'detach') else np.array(self.reservoir.z_centers)
        else:
            # Fallback
            if hasattr(self.reservoir, 'grid_size'):
                grid_size = self.reservoir.grid_size.detach().cpu().numpy() if hasattr(self.reservoir.grid_size, 'detach') else np.array(self.reservoir.grid_size)
                dx, dy, dz = grid_size
            else:
                dx = dy = dz = 1.0
            x_coords = np.arange(dx/2, self.nx * dx, dx)
            y_coords = np.arange(dy/2, self.ny * dy, dy)
            z_coords = np.arange(dz/2, self.nz * dz, dz)
        
        # Применяем subsample к координатам
        if subsample > 1:
            x_coords = x_coords[::subsample]
            y_coords = y_coords[::subsample]
            z_coords = z_coords[::subsample]
        
        # Создаём сетку координат
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        grid.points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        grid.dimensions = pressure_sub.shape
        
        # Добавляем данные
        grid['Давление (МПа)'] = (pressure_sub / 1e6).flatten()
        grid['Водонасыщенность'] = sw_sub.flatten()
        if sg_sub is not None:
            grid['Газонасыщенность'] = sg_sub.flatten()
        
        # Создаём plotter
        plotter = pv.Plotter(title=title)
        
        # Для больших сеток используем срезы вместо объёмного рендеринга
        n_points = len(grid.points)
        if n_points > 200000:
            # Используем несколько срезов
            z_mid = pressure_sub.shape[2] // 2
            z_slices = [
                pressure_sub.shape[2] // 4,
                z_mid,
                3 * pressure_sub.shape[2] // 4
            ]
            
            # Получаем координаты Z для срезов
            if hasattr(self.reservoir, 'z_centers'):
                z_coords = self.reservoir.z_centers.detach().cpu().numpy() if hasattr(self.reservoir.z_centers, 'detach') else np.array(self.reservoir.z_centers)
            else:
                if hasattr(self.reservoir, 'grid_size'):
                    grid_size = self.reservoir.grid_size.detach().cpu().numpy() if hasattr(self.reservoir.grid_size, 'detach') else np.array(self.reservoir.grid_size)
                    dz = grid_size[2]
                else:
                    dz = 1.0
                z_coords = np.arange(dz/2, self.nz * dz, dz)
            
            if subsample > 1:
                z_coords = z_coords[::subsample]
            
            for z_idx in z_slices:
                z_val = float(z_coords[z_idx])
                # Создаём плоскость для среза (origin должен быть кортежем из 3 координат)
                origin = (grid.bounds[0], grid.bounds[2], z_val)  # (x_min, y_min, z_value)
                normal = (0, 0, 1)  # Нормаль вдоль оси Z
                
                # Срез давления
                slice_p = grid.slice(normal=normal, origin=origin)
                plotter.add_mesh(
                    slice_p,
                    scalars='Давление (МПа)',
                    cmap='jet',
                    show_scalar_bar=(z_idx == z_mid),
                    scalar_bar_args={'title': 'Давление (МПа)', 'vertical': True}
                )
                
                # Срез водонасыщенности (смещён немного выше для видимости)
                origin_sw = (grid.bounds[0], grid.bounds[2], z_val + (grid.bounds[5] - grid.bounds[4]) * 0.1)
                slice_sw = grid.slice(normal=normal, origin=origin_sw)
                plotter.add_mesh(
                    slice_sw,
                    scalars='Водонасыщенность',
                    cmap='viridis',
                    show_scalar_bar=(z_idx == z_mid),
                    scalar_bar_args={'title': 'Sw', 'vertical': True}
                )
                
                # Срез газонасыщенности (если есть)
                if sg_sub is not None and np.any(sg_sub > 1e-6):
                    origin_sg = (grid.bounds[0], grid.bounds[2], z_val + (grid.bounds[5] - grid.bounds[4]) * 0.2)
                    slice_sg = grid.slice(normal=normal, origin=origin_sg)
                    plotter.add_mesh(
                        slice_sg,
                        scalars='Газонасыщенность',
                        cmap='plasma',
                        show_scalar_bar=(z_idx == z_mid),
                        scalar_bar_args={'title': 'Sg', 'vertical': True}
                    )
        else:
            # Объёмный рендеринг для меньших сеток
            plotter.add_volume(
                grid,
                scalars='Давление (МПа)',
                cmap='jet',
                opacity='linear',
                show_scalar_bar=True,
                scalar_bar_args={'title': 'Давление (МПа)', 'vertical': True}
            )
            
            plotter.add_volume(
                grid,
                scalars='Водонасыщенность',
                cmap='viridis',
                opacity='linear',
                show_scalar_bar=True,
                scalar_bar_args={'title': 'Sw', 'vertical': True}
            )
            
            if sg_sub is not None and np.any(sg_sub > 1e-6):
                plotter.add_volume(
                    grid,
                    scalars='Газонасыщенность',
                    cmap='plasma',
                    opacity='linear',
                    show_scalar_bar=True,
                    scalar_bar_args={'title': 'Sg', 'vertical': True}
                )
        
        # Настройка камеры
        plotter.camera_position = 'iso'
        plotter.background_color = 'black'
        
        return plotter
    
    def create_slice_viewer(
        self,
        pressure: np.ndarray,
        sw: np.ndarray,
        sg: Optional[np.ndarray] = None,
        title: str = "Срезы резервуара"
    ) -> pv.Plotter:
        """
        Создаёт просмотрщик срезов с возможностью интерактивного изменения слоя.
        
        Args:
            pressure: Массив давления (nx, ny, nz)
            sw: Массив водонасыщенности (nx, ny, nz)
            sg: Массив газонасыщенности (nx, ny, nz), опционально
            title: Заголовок
            
        Returns:
            pv.Plotter: Объект Plotter PyVista
        """
        # Создаём структурированную сетку
        grid = pv.StructuredGrid()
        
        # Получаем координаты
        if hasattr(self.reservoir, 'x_centers') and hasattr(self.reservoir, 'y_centers') and hasattr(self.reservoir, 'z_centers'):
            x_coords = self.reservoir.x_centers.detach().cpu().numpy() if hasattr(self.reservoir.x_centers, 'detach') else np.array(self.reservoir.x_centers)
            y_coords = self.reservoir.y_centers.detach().cpu().numpy() if hasattr(self.reservoir.y_centers, 'detach') else np.array(self.reservoir.y_centers)
            z_coords = self.reservoir.z_centers.detach().cpu().numpy() if hasattr(self.reservoir.z_centers, 'detach') else np.array(self.reservoir.z_centers)
        else:
            if hasattr(self.reservoir, 'grid_size'):
                grid_size = self.reservoir.grid_size.detach().cpu().numpy() if hasattr(self.reservoir.grid_size, 'detach') else np.array(self.reservoir.grid_size)
                dx, dy, dz = grid_size
            else:
                dx = dy = dz = 1.0
            x_coords = np.arange(dx/2, self.nx * dx, dx)
            y_coords = np.arange(dy/2, self.ny * dy, dy)
            z_coords = np.arange(dz/2, self.nz * dz, dz)
        
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        grid.points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        grid.dimensions = pressure.shape
        
        # Добавляем данные
        grid['Давление (МПа)'] = (pressure / 1e6).flatten()
        grid['Водонасыщенность'] = sw.flatten()
        if sg is not None:
            grid['Газонасыщенность'] = sg.flatten()
        
        # Создаём plotter с несколькими окнами
        plotter = pv.Plotter(shape=(1, 3 if sg is not None and np.any(sg > 1e-6) else 2), title=title)
        
        # Срез давления
        plotter.subplot(0, 0)
        z_mid = pressure.shape[2] // 2
        z_val = float(z_coords[z_mid])
        origin = (grid.bounds[0], grid.bounds[2], z_val)
        normal = (0, 0, 1)
        slice_p = grid.slice(normal=normal, origin=origin)
        plotter.add_mesh(slice_p, scalars='Давление (МПа)', cmap='jet', show_scalar_bar=True)
        plotter.add_text('Давление', font_size=12)
        
        # Срез водонасыщенности
        plotter.subplot(0, 1)
        slice_sw = grid.slice(normal=normal, origin=origin)
        plotter.add_mesh(slice_sw, scalars='Водонасыщенность', cmap='viridis', show_scalar_bar=True)
        plotter.add_text('Водонасыщенность', font_size=12)
        
        # Срез газонасыщенности (если есть)
        if sg is not None and np.any(sg > 1e-6):
            plotter.subplot(0, 2)
            slice_sg = grid.slice(normal=normal, origin=origin)
            plotter.add_mesh(slice_sg, scalars='Газонасыщенность', cmap='plasma', show_scalar_bar=True)
            plotter.add_text('Газонасыщенность', font_size=12)
        
        plotter.background_color = 'black'
        
        return plotter
    
    def save_html(self, plotter: pv.Plotter, filepath: str):
        """
        Сохраняет визуализацию в HTML файл.
        
        Args:
            plotter: Объект Plotter PyVista
            filepath: Путь к файлу
        """
        try:
            # PyVista может экспортировать интерактивный HTML через export_html
            # Это создаёт HTML с встроенным VTK.js для интерактивной визуализации
            plotter.export_html(filepath)
            print(f"  ✅ 3D визуализация сохранена в {filepath}")
            print(f"  📖 Откройте файл в браузере для интерактивного просмотра")
        except Exception as e:
            # Fallback: сохраняем как изображение и создаём простой HTML
            try:
                img_path = filepath.replace('.html', '.png')
                plotter.screenshot(img_path, window_size=[1920, 1080])
                
                # Создаём простой HTML с изображением
                html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>3D Визуализация</title>
    <style>
        body {{
            background-color: #000;
            color: #fff;
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 20px;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
        .info {{
            margin-top: 20px;
            color: #888;
        }}
    </style>
</head>
<body>
    <h1>3D Визуализация резервуара</h1>
    <p class="info">Для интерактивной визуализации откройте VTK файл (.vtr) в ParaView</p>
    <p class="info">ParaView можно скачать с <a href="https://www.paraview.org/" style="color: #4CAF50;">paraview.org</a></p>
    <img src="{os.path.basename(img_path)}" alt="3D Visualization">
</body>
</html>
"""
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"  ✅ 3D визуализация сохранена в {filepath} (статическое изображение)")
                print(f"  💡 Для интерактивной визуализации используйте VTK файлы с ParaView")
            except Exception as e2:
                print(f"  ⚠ Не удалось сохранить HTML: {e2}")
                print(f"  💡 Используйте VTK файлы (.vtr) с ParaView для визуализации")
    
    def save_vtk(self, pressure: np.ndarray, sw: np.ndarray, sg: Optional[np.ndarray] = None, filepath: str = None):
        """
        Сохраняет данные в VTK файл для просмотра в ParaView.
        
        Args:
            pressure: Массив давления
            sw: Массив водонасыщенности
            sg: Массив газонасыщенности
            filepath: Путь к файлу
        """
        # Создаём структурированную сетку
        grid = pv.StructuredGrid()
        
        # Получаем координаты
        if hasattr(self.reservoir, 'x_centers') and hasattr(self.reservoir, 'y_centers') and hasattr(self.reservoir, 'z_centers'):
            x_coords = self.reservoir.x_centers.detach().cpu().numpy() if hasattr(self.reservoir.x_centers, 'detach') else np.array(self.reservoir.x_centers)
            y_coords = self.reservoir.y_centers.detach().cpu().numpy() if hasattr(self.reservoir.y_centers, 'detach') else np.array(self.reservoir.y_centers)
            z_coords = self.reservoir.z_centers.detach().cpu().numpy() if hasattr(self.reservoir.z_centers, 'detach') else np.array(self.reservoir.z_centers)
        else:
            if hasattr(self.reservoir, 'grid_size'):
                grid_size = self.reservoir.grid_size.detach().cpu().numpy() if hasattr(self.reservoir.grid_size, 'detach') else np.array(self.reservoir.grid_size)
                dx, dy, dz = grid_size
            else:
                dx = dy = dz = 1.0
            x_coords = np.arange(dx/2, self.nx * dx, dx)
            y_coords = np.arange(dy/2, self.ny * dy, dy)
            z_coords = np.arange(dz/2, self.nz * dz, dz)
        
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        grid.points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        grid.dimensions = pressure.shape
        
        # Добавляем данные
        grid['Давление (МПа)'] = (pressure / 1e6).flatten()
        grid['Водонасыщенность'] = sw.flatten()
        if sg is not None:
            grid['Газонасыщенность'] = sg.flatten()
        
        # Сохраняем
        if filepath:
            grid.save(filepath)
            print(f"VTK файл сохранён: {filepath}")

