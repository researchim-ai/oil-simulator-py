#!/usr/bin/env python3
"""
Профессиональная интерактивная визуализация с полным контролем параметров.
Слайдеры, кнопки, выбор полей, настройка цветов - всё как у профессионалов!

Использование:
    python scripts/view_interactive.py results/mega_3phase_million_*/
"""

import sys
import os
from pathlib import Path
import numpy as np

try:
    import pyvista as pv
    pv.global_theme.allow_empty_mesh = True
    pv.set_plot_theme("dark")
except ImportError:
    print("❌ PyVista не установлен. Установите: pip install pyvista")
    sys.exit(1)


class InteractiveViewer:
    """Профессиональный интерактивный визуализатор с полным контролем."""
    
    def __init__(self, results_dir):
        self.reservoir_dir = Path(results_dir)
        self.vtk_files = []
        self.grids = {}
        self.current_step = 0
        self.load_files()
        
        # Параметры визуализации
        self.params = {
            'field': 'Pressure_MPa',
            'opacity': 0.8,
            'opacity_min': 0.0,
            'opacity_max': 1.0,
            'cmap': 'jet',
            'show_sw': False,
            'show_sg': False,
            'sw_opacity': 0.5,
            'sg_opacity': 0.5,
            'value_min': None,  # Минимальное значение для цветовой шкалы
            'value_max': None,  # Максимальное значение для цветовой шкалы
            'value_min_actual': None,  # Реальное минимальное значение в данных
            'value_max_actual': None,  # Реальное максимальное значение в данных
            'contrast': 1.0,  # Коэффициент контраста (1.0 = нормальный, >1 = усиленный)
            'show_slice': False,
            'slice_z': 0.5,
            'show_diff': False,
            'diff_mode': 'absolute',
        }
        
        # Загружаем первый файл для получения диапазонов
        if self.vtk_files:
            self.update_ranges()
    
    def load_files(self):
        """Загружает все VTK файлы."""
        intermediate_dir = self.reservoir_dir / "intermediate"
        if intermediate_dir.exists():
            self.vtk_files = sorted(intermediate_dir.glob("*.vtr"))
        else:
            self.vtk_files = sorted(self.reservoir_dir.glob("*.vtr"))
        
        if not self.vtk_files:
            print(f"❌ Не найдено VTK файлов в {self.reservoir_dir}")
            return
        
        print(f"📂 Найдено {len(self.vtk_files)} шагов симуляции")
    
    def get_grid(self, step):
        """Загружает сетку для шага (с кешированием)."""
        if step in self.grids:
            return self.grids[step]
        
        if step < 0 or step >= len(self.vtk_files):
            return None
        
        grid = pv.read(str(self.vtk_files[step]))
        self.grids[step] = grid
        return grid
    
    def update_ranges(self):
        """Обновляет диапазоны значений для текущего поля."""
        grid = self.get_grid(self.current_step)
        if grid is None:
            return
        
        field = self.params['field']
        if field in grid.cell_data:
            data = grid.cell_data[field]
            v_min_actual = float(data.min())
            v_max_actual = float(data.max())
            v_mean = float(data.mean())
            v_std = float(data.std())
            
            # Сохраняем реальные значения
            self.params['value_min_actual'] = v_min_actual
            self.params['value_max_actual'] = v_max_actual
            
            # Если диапазоны не заданы вручную, используем реальные
            if self.params['value_min'] is None:
                self.params['value_min'] = v_min_actual
            if self.params['value_max'] is None:
                self.params['value_max'] = v_max_actual
            
            print(f"📊 Диапазон {field}: {v_min_actual:.2f} - {v_max_actual:.2f} (среднее: {v_mean:.2f}, σ: {v_std:.2f})")
    
    def create_visualization(self, plotter=None):
        """Создаёт визуализацию с текущими параметрами."""
        grid = self.get_grid(self.current_step)
        if grid is None:
            return None
        
        # Создаём plotter только если его нет
        if plotter is None:
            plotter = pv.Plotter(
                title=f"3D Визуализация - Шаг {self.current_step+1}/{len(self.vtk_files)}",
                window_size=[1400, 900]
            )
            plotter.set_background('black')
        else:
            # Очищаем существующий plotter
            plotter.clear()
            # Обновляем заголовок
            plotter.title = f"Интерактивная визуализация - Шаг {self.current_step+1}/{len(self.vtk_files)}"
        
        field = self.params['field']
        available_fields = list(grid.cell_data.keys())
        
        if field not in available_fields:
            field = available_fields[0] if available_fields else None
            self.params['field'] = field
        
        if field:
            # Основное поле
            data = grid.cell_data[field]
            
            # Если включён режим сравнения, показываем разницу
            if self.params['show_diff'] and self.current_step > 0:
                prev_grid = self.get_grid(self.current_step - 1)
                if prev_grid and field in prev_grid.cell_data:
                    prev_data = prev_grid.cell_data[field]
                    if self.params['diff_mode'] == 'absolute':
                        diff_data = data - prev_data
                        title = f'Δ{field} (абсолютная)'
                    else:  # relative
                        diff_data = (data - prev_data) / (prev_data + 1e-10) * 100
                        title = f'Δ{field} (%)'
                    
                    # Добавляем разницу как новое поле
                    grid.cell_data[f'Diff_{field}'] = diff_data
                    field = f'Diff_{field}'
                    data = diff_data
                    # Используем другую цветовую схему для разницы
                    cmap = 'coolwarm' if self.params['cmap'] == 'jet' else 'RdBu_r'
                else:
                    cmap = self.params['cmap']
            else:
                cmap = self.params['cmap']
                title = field
            
            # Получаем реальные диапазоны
            v_min_actual = self.params['value_min_actual'] if self.params['value_min_actual'] is not None else float(data.min())
            v_max_actual = self.params['value_max_actual'] if self.params['value_max_actual'] is not None else float(data.max())
            v_mean = float(data.mean())
            
            # Применяем настройки диапазона (clipping)
            v_min = self.params['value_min'] if self.params['value_min'] is not None else v_min_actual
            v_max = self.params['value_max'] if self.params['value_max'] is not None else v_max_actual
            
            # Применяем контраст - сужаем диапазон вокруг среднего для усиления перепадов
            if self.params['contrast'] > 1.0:
                contrast_range = (v_max_actual - v_min_actual) / self.params['contrast']
                v_center = (v_min + v_max) / 2
                v_min = max(v_min_actual, v_center - contrast_range / 2)
                v_max = min(v_max_actual, v_center + contrast_range / 2)
            
            # Создаём кастомную функцию прозрачности
            opacity_points = [v_min, v_max]
            
            volume_actor = plotter.add_volume(
                grid,
                scalars=field,
                cmap=cmap,
                opacity=opacity_points,
                clim=[v_min, v_max],  # Устанавливаем диапазон цветовой шкалы для усиления контраста
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': title,
                    'vertical': True,
                    'title_font_size': 14,
                    'label_font_size': 11,
                    'shadow': True,
                    'n_labels': 5
                }
            )
            
            # Водонасыщенность (если включена)
            if self.params['show_sw'] and 'Water_Saturation' in available_fields:
                sw_data = grid.cell_data['Water_Saturation']
                sw_min, sw_max = float(sw_data.min()), float(sw_data.max())
                plotter.add_volume(
                    grid,
                    scalars='Water_Saturation',
                    cmap='viridis',
                    opacity=[self.params['sw_opacity'] * 0.3, self.params['sw_opacity']],
                    show_scalar_bar=True,
                    scalar_bar_args={
                        'title': 'Водонасыщенность',
                        'vertical': True,
                        'title_font_size': 12
                    }
                )
            
            # Газонасыщенность (если включена)
            if self.params['show_sg'] and 'Gas_Saturation' in available_fields:
                sg_data = grid.cell_data['Gas_Saturation']
                if np.any(sg_data > 1e-6):
                    sg_min, sg_max = float(sg_data.min()), float(sg_data.max())
                    plotter.add_volume(
                        grid,
                        scalars='Gas_Saturation',
                        cmap='plasma',
                        opacity=[self.params['sg_opacity'] * 0.3, self.params['sg_opacity']],
                        show_scalar_bar=True,
                        scalar_bar_args={
                            'title': 'Газонасыщенность',
                            'vertical': True,
                            'title_font_size': 12
                        }
                    )
            
            # Срез (если включен)
            if self.params['show_slice']:
                z_val = grid.bounds[4] + (grid.bounds[5] - grid.bounds[4]) * self.params['slice_z']
                slice_mesh = grid.slice(
                    normal=(0, 0, 1),
                    origin=(grid.bounds[0], grid.bounds[2], z_val)
                )
                plotter.add_mesh(
                    slice_mesh,
                    scalars=field,
                    cmap=self.params['cmap'],
                    opacity=0.9,
                    show_scalar_bar=False
                )
            
            # Информация о шаге - слева вверху, используем нормализованные координаты (0-1)
            base_field = self.params['field']
            
            if self.params['show_diff']:
                info_lines = [
                    f"Шаг {self.current_step+1}/{len(self.vtk_files)} (изменение)",
                    f"Поле: {base_field}",
                    f"Среднее: {float(data.mean()):.4f}",
                    f"Диапазон: {v_min:.4f} - {v_max:.4f}",
                    f"Прозрачность: {self.params['opacity']:.1f}"
                ]
            else:
                info_lines = [
                    f"Шаг {self.current_step+1}/{len(self.vtk_files)}",
                    f"Поле: {base_field}",
                    f"Среднее: {float(data.mean()):.2f}",
                    f"Диапазон: {v_min:.2f} - {v_max:.2f}",
                    f"Прозрачность: {self.params['opacity']:.1f}"
                ]
            
            # Используем нормализованные координаты для корректной работы при изменении размера
            # position='upper_left' использует нормализованные координаты
            info_text = "\n".join(info_lines)
            plotter.add_text(info_text, font_size=10, color='white', 
                           position='upper_left', shadow=True)
        
        # Настраиваем камеру
        plotter.camera_position = 'iso'
        plotter.reset_camera()
        
        return plotter
    
    def add_control_panel(self, plotter):
        """Добавляет панель управления с кнопками и слайдерами."""
        # Сохраняем ссылку на plotter для обновлений
        self.current_plotter = plotter
        
        # Слайдер для выбора шага
        def update_step(value):
            step = int(value)
            if 0 <= step < len(self.vtk_files) and step != self.current_step:
                self.current_step = step
                self.update_ranges()
                # Пересоздаём визуализацию
                self.rebuild_visualization(plotter)
        
        plotter.add_slider_widget(
            update_step,
            value=self.current_step,
            rng=[0, len(self.vtk_files) - 1],
            title=f"Шаг",
            pointa=(0.02, 0.05),
            pointb=(0.15, 0.05),
            style='modern',
            title_height=0.02,
            fmt='%d'
        )
        
        # Слайдер для прозрачности
        def update_opacity(value):
            if abs(self.params['opacity'] - value) > 0.01:  # Избегаем лишних обновлений
                self.params['opacity'] = value
                self.rebuild_visualization(plotter)
        
        plotter.add_slider_widget(
            update_opacity,
            value=self.params['opacity'],
            rng=[0.0, 1.0],
            title="Прозрачность",
            pointa=(0.02, 0.12),
            pointb=(0.15, 0.12),
            style='modern',
            title_height=0.02,
            fmt='%.2f'
        )
        
        # Слайдер для контраста (усиление перепадов)
        def update_contrast(value):
            if abs(self.params['contrast'] - value) > 0.01:
                self.params['contrast'] = value
                self.rebuild_visualization(plotter)
        
        plotter.add_slider_widget(
            update_contrast,
            value=self.params['contrast'],
            rng=[0.5, 5.0],
            title="Контраст",
            pointa=(0.02, 0.19),
            pointb=(0.15, 0.19),
            style='modern',
            title_height=0.02,
            fmt='%.2f'
        )
        
        # Слайдер для минимального значения (clipping)
        def update_value_min(value):
            if self.params['value_min_actual'] is not None:
                v_min_actual = self.params['value_min_actual']
                v_max_actual = self.params['value_max_actual']
                # value - это доля от диапазона (0-1)
                v_min = v_min_actual + (v_max_actual - v_min_actual) * value
                if self.params['value_min'] is None or abs(self.params['value_min'] - v_min) > 0.01:
                    self.params['value_min'] = v_min
                    self.rebuild_visualization(plotter)
        
        v_min_norm = 0.0
        if self.params['value_min_actual'] is not None and self.params['value_max_actual'] is not None:
            if self.params['value_min'] is not None:
                v_min_actual = self.params['value_min_actual']
                v_max_actual = self.params['value_max_actual']
                v_min_norm = (self.params['value_min'] - v_min_actual) / (v_max_actual - v_min_actual) if (v_max_actual - v_min_actual) > 0 else 0.0
        
        plotter.add_slider_widget(
            update_value_min,
            value=v_min_norm,
            rng=[0.0, 1.0],
            title="Min",
            pointa=(0.02, 0.26),
            pointb=(0.15, 0.26),
            style='modern',
            title_height=0.02,
            fmt='%.2f'
        )
        
        # Слайдер для максимального значения (clipping)
        def update_value_max(value):
            if self.params['value_min_actual'] is not None:
                v_min_actual = self.params['value_min_actual']
                v_max_actual = self.params['value_max_actual']
                # value - это доля от диапазона (0-1)
                v_max = v_min_actual + (v_max_actual - v_min_actual) * value
                if self.params['value_max'] is None or abs(self.params['value_max'] - v_max) > 0.01:
                    self.params['value_max'] = v_max
                    self.rebuild_visualization(plotter)
        
        v_max_norm = 1.0
        if self.params['value_min_actual'] is not None and self.params['value_max_actual'] is not None:
            if self.params['value_max'] is not None:
                v_min_actual = self.params['value_min_actual']
                v_max_actual = self.params['value_max_actual']
                v_max_norm = (self.params['value_max'] - v_min_actual) / (v_max_actual - v_min_actual) if (v_max_actual - v_min_actual) > 0 else 1.0
        
        plotter.add_slider_widget(
            update_value_max,
            value=v_max_norm,
            rng=[0.0, 1.0],
            title="Max",
            pointa=(0.02, 0.33),
            pointb=(0.15, 0.33),
            style='modern',
            title_height=0.02,
            fmt='%.2f'
        )
        
        # Кнопки для переключения полей (колбэки принимают значение от виджета)
        def toggle_sw(value):
            self.params['show_sw'] = bool(value)
            self.rebuild_visualization(plotter)
        
        def toggle_sg(value):
            self.params['show_sg'] = bool(value)
            self.rebuild_visualization(plotter)
        
        def change_cmap(value=None):
            cmaps = ['jet', 'viridis', 'plasma', 'coolwarm', 'hot', 'turbo']
            current_idx = cmaps.index(self.params['cmap']) if self.params['cmap'] in cmaps else 0
            self.params['cmap'] = cmaps[(current_idx + 1) % len(cmaps)]
            self.rebuild_visualization(plotter)
        
        # Добавляем панель управления справа внизу
        # Используем нормализованные координаты для корректной работы при изменении размера
        # PyVista использует координаты от 0 до 1, где (0,0) - левый нижний угол
        
        # УПРАВЛЕНИЕ - справа вверху
        plotter.add_text("УПРАВЛЕНИЕ", position=(0.85, 0.95), 
                        font_size=12, color='yellow')
        
        # Кнопки и надписи - справа, вертикально
        plotter.add_checkbox_button_widget(
            toggle_sw,
            value=self.params['show_sw'],
            position=(0.85, 0.88),
            size=20,
            border_size=2,
            color_on='blue',
            color_off='gray',
            background_color='white'
        )
        plotter.add_text("Вода (W)", position=(0.88, 0.88), 
                        font_size=10, color='white')
        
        plotter.add_checkbox_button_widget(
            toggle_sg,
            value=self.params['show_sg'],
            position=(0.85, 0.82),
            size=20,
            border_size=2,
            color_on='purple',
            color_off='gray',
            background_color='white'
        )
        plotter.add_text("Газ (G)", position=(0.88, 0.82), 
                        font_size=10, color='white')
        
        # Информация о цветовой схеме
        plotter.add_text("Цвет: " + self.params['cmap'] + " (C)", 
                        position=(0.85, 0.76), font_size=10, color='white')
    
    def rebuild_visualization(self, plotter):
        """Пересоздаёт визуализацию с текущими параметрами."""
        # Сохраняем виджеты перед очисткой
        widgets = []
        for actor in plotter.renderer.GetActors():
            if hasattr(actor, 'GetProperty'):
                widgets.append(actor)
        
        self.create_visualization(plotter)
        
        # Восстанавливаем виджеты (слайдеры и кнопки)
        self.add_control_panel(plotter)
        
        plotter.render()
    
    def show_interactive(self):
        """Показывает интерактивное окно с настройками."""
        if not self.vtk_files:
            print("❌ Нет файлов для визуализации")
            return
        
        print("\n" + "="*60)
        print("🎨 ИНТЕРАКТИВНАЯ ВИЗУАЛИЗАЦИЯ")
        print("="*60)
        print("\n📋 УПРАВЛЕНИЕ:")
        print("  Клавиатура:")
        print("    ← →     - Переключение шагов")
        print("    ↑ ↓     - Изменение прозрачности")
        print("    +/-     - Изменение диапазона значений")
        print("    F       - Переключение поля (Pressure/Sw/Sg)")
        print("    W       - Включить/выключить водонасыщенность")
        print("    G       - Включить/выключить газонасыщенность")
        print("    C       - Сменить цветовую схему")
        print("    R       - Сбросить настройки (включая диапазоны)")
        print("    Q       - Выход")
        print("\n  Мышь:")
        print("    Левая кнопка + движение - вращение")
        print("    Колесо - масштабирование")
        print("    Средняя кнопка - перемещение")
        print("="*60 + "\n")
        
        # Создаём plotter один раз с явной инициализацией
        plotter = pv.Plotter(
            title=f"3D Визуализация - Шаг {self.current_step+1}/{len(self.vtk_files)}",
            window_size=[1400, 900],
            off_screen=False  # Явно указываем, что окно должно быть видимым
        )
        plotter.set_background('black')
        
        # Создаём начальную визуализацию
        plotter = self.create_visualization(plotter)
        if plotter is None:
            return
        
        # Добавляем панель управления
        self.add_control_panel(plotter)
        
        # Обработчики клавиш
        def next_step():
            if self.current_step < len(self.vtk_files) - 1:
                self.current_step += 1
                self.update_ranges()
                self.rebuild_visualization(plotter)
            return True
        
        def prev_step():
            if self.current_step > 0:
                self.current_step -= 1
                self.update_ranges()
                self.rebuild_visualization(plotter)
            return True
        
        def inc_opacity():
            self.params['opacity'] = min(1.0, self.params['opacity'] + 0.1)
            self.rebuild_visualization(plotter)
            return True
        
        def dec_opacity():
            self.params['opacity'] = max(0.0, self.params['opacity'] - 0.1)
            self.rebuild_visualization(plotter)
            return True
        
        def toggle_sw():
            self.params['show_sw'] = not self.params['show_sw']
            self.rebuild_visualization(plotter)
            return True
        
        def toggle_sg():
            self.params['show_sg'] = not self.params['show_sg']
            self.rebuild_visualization(plotter)
            return True
        
        # Убрали toggle_slice и toggle_diff - они не нужны
        
        def change_field():
            available = list(self.get_grid(self.current_step).cell_data.keys())
            fields_order = ['Pressure_MPa', 'Water_Saturation', 'Gas_Saturation', 
                          'Oil_Saturation', 'Perm_Kh_m2']
            current_idx = 0
            if self.params['field'] in fields_order:
                current_idx = fields_order.index(self.params['field'])
            current_idx = (current_idx + 1) % len(fields_order)
            for f in fields_order[current_idx:]:
                if f in available:
                    self.params['field'] = f
                    self.update_ranges()
                    break
            self.rebuild_visualization(plotter)
            return True
        
        def change_cmap():
            cmaps = ['jet', 'viridis', 'plasma', 'coolwarm', 'hot', 'turbo']
            current_idx = cmaps.index(self.params['cmap']) if self.params['cmap'] in cmaps else 0
            self.params['cmap'] = cmaps[(current_idx + 1) % len(cmaps)]
            self.rebuild_visualization(plotter)
            return True
        
        def reset():
            self.params['opacity'] = 0.8
            self.params['opacity_min'] = 0.0
            self.params['opacity_max'] = 1.0
            self.params['value_min'] = None
            self.params['value_max'] = None
            self.params['contrast'] = 1.0
            self.update_ranges()
            self.rebuild_visualization(plotter)
            return True
        
        # Привязываем клавиши
        plotter.add_key_event('Right', next_step)
        plotter.add_key_event('Left', prev_step)
        plotter.add_key_event('Up', inc_opacity)
        plotter.add_key_event('Down', dec_opacity)
        plotter.add_key_event('f', change_field)
        plotter.add_key_event('w', toggle_sw)
        plotter.add_key_event('g', toggle_sg)
        plotter.add_key_event('c', change_cmap)
        plotter.add_key_event('r', reset)
        
        # Рендерим перед показом для инициализации окна
        plotter.render()
        
        # Показываем окно
        try:
            # Используем show() без параметров
            plotter.show()
        except AttributeError as e:
            if "'NoneType' object has no attribute 'IsCurrent'" in str(e):
                # Пробуем альтернативный способ - создаём окно явно
                print("⚠ Проблема с инициализацией окна, пробуем альтернативный способ...")
                try:
                    # Принудительно инициализируем окно
                    if plotter.render_window is None:
                        plotter.render()
                    plotter.show(interactive=True)
                except Exception as e2:
                    print(f"⚠ Ошибка: {e2}")
                    print("  Возможно, проблема с дисплеем. Попробуйте:")
                    print("  - Убедитесь, что X11 forwarding включен (если используете SSH)")
                    print("  - Проверьте переменную DISPLAY: echo $DISPLAY")
                    print("  - Или используйте ParaView для просмотра VTK файлов")
                    return
            else:
                raise
        except Exception as e:
            print(f"⚠ Ошибка при показе окна: {e}")
            print("  Попробуйте использовать другой дисплей или X11 forwarding")
            return


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Профессиональная интерактивная визуализация с полным контролем',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('results_dir', help='Директория с результатами симуляции')
    
    args = parser.parse_args()
    
    viewer = InteractiveViewer(args.results_dir)
    viewer.show_interactive()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Использование: python scripts/view_interactive.py <results_dir>")
        print("\nПример:")
        print("  python scripts/view_interactive.py results/mega_3phase_million_*/")
        sys.exit(1)
    
    main()

