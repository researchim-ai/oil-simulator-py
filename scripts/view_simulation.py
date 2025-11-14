#!/usr/bin/env python3
"""
Профессиональная интерактивная визуализация всего процесса симуляции.
Показывает все шаги с анимацией, несколькими полями, красивыми цветами.

Использование:
    python scripts/view_simulation.py results/mega_3phase_million_*/
"""

import sys
import os
import glob
import argparse
from pathlib import Path

try:
    import pyvista as pv
    import numpy as np
    # Разрешаем пустые меши для обработки граничных случаев
    pv.global_theme.allow_empty_mesh = True
except ImportError:
    print("❌ PyVista не установлен. Установите: pip install pyvista")
    sys.exit(1)


class SimulationViewer:
    """Профессиональный визуализатор процесса симуляции."""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.vtk_files = []
        self.current_step = 0
        self.grids = []
        self.load_files()
    
    def load_files(self):
        """Загружает все VTK файлы из директории."""
        # Ищем все .vtr файлы
        intermediate_dir = self.results_dir / "intermediate"
        if intermediate_dir.exists():
            self.vtk_files = sorted(intermediate_dir.glob("*.vtr"))
        else:
            self.vtk_files = sorted(self.results_dir.glob("*.vtr"))
        
        if not self.vtk_files:
            print(f"❌ Не найдено VTK файлов в {self.results_dir}")
            return
        
        print(f"📂 Найдено {len(self.vtk_files)} шагов симуляции")
        
        # Загружаем первый файл для получения информации
        first_grid = pv.read(str(self.vtk_files[0]))
        print(f"✅ Размер сетки: {first_grid.dimensions}")
        print(f"📊 Доступные поля: {list(first_grid.cell_data.keys())}")
    
    def get_grid(self, step):
        """Загружает сетку для указанного шага."""
        if step < 0 or step >= len(self.vtk_files):
            return None
        return pv.read(str(self.vtk_files[step]))
    
    def create_volume_visualization(self, grid, step, total_steps):
        """Создаёт красивую объёмную визуализацию."""
        plotter = pv.Plotter(title=f"3D Визуализация резервуара - Шаг {step+1}/{total_steps}")
        plotter.set_background('black')
        
        # Получаем доступные поля
        available_fields = list(grid.cell_data.keys())
        pressure_field = 'Pressure_MPa' if 'Pressure_MPa' in available_fields else None
        sw_field = 'Water_Saturation' if 'Water_Saturation' in available_fields else None
        sg_field = 'Gas_Saturation' if 'Gas_Saturation' in available_fields else None
        
        n_cells = grid.n_cells
        print(f"  📦 Количество ячеек: {n_cells:,}")
        
        # Всегда используем настоящий объёмный рендеринг (не слои!)
        print("  🎨 Создаём объёмную 3D визуализацию...")
        
        if pressure_field:
            # Объёмный рендеринг давления - показывает весь объём с градиентами
            plotter.add_volume(
                grid,
                scalars=pressure_field,
                cmap='jet',
                opacity='linear_r',  # Обратная прозрачность: высокие значения более видимы
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': 'Давление (МПа)',
                    'vertical': True,
                    'title_font_size': 14,
                    'label_font_size': 12,
                    'shadow': True
                }
            )
        
        # Объёмный рендеринг водонасыщенности
        if sw_field:
            plotter.add_volume(
                grid,
                scalars=sw_field,
                cmap='viridis',
                opacity='linear',  # Прямая прозрачность
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': 'Водонасыщенность',
                    'vertical': True,
                    'title_font_size': 14,
                    'label_font_size': 12
                }
            )
        
        # Объёмный рендеринг газонасыщенности
        if sg_field:
            sg_data = grid.cell_data[sg_field]
            if np.any(sg_data > 1e-6):  # Только если есть газ
                plotter.add_volume(
                    grid,
                    scalars=sg_field,
                    cmap='plasma',
                    opacity='linear',
                    show_scalar_bar=True,
                    scalar_bar_args={
                        'title': 'Газонасыщенность',
                        'vertical': True,
                        'title_font_size': 14,
                        'label_font_size': 12
                    }
                )
        
        # Добавляем информацию о шаге
        if pressure_field:
            p_data = grid.cell_data[pressure_field]
            p_min, p_max = float(p_data.min()), float(p_data.max())
            p_mean = float(p_data.mean())
            
            info_text = f"""Шаг {step+1}/{total_steps}
Давление: {p_mean:.2f} МПа
Диапазон: {p_min:.2f} - {p_max:.2f} МПа"""
            
            if sw_field:
                sw_data = grid.cell_data[sw_field]
                sw_mean = float(sw_data.mean())
                info_text += f"\nВодонасыщенность: {sw_mean:.3f}"
            
            plotter.add_text(info_text, font_size=12, color='white', 
                           position='upper_left', shadow=True)
        
        # Настраиваем камеру
        plotter.camera_position = 'iso'
        plotter.reset_camera()
        
        return plotter
    
    def create_multi_field_view(self, grid, step, total_steps):
        """Создаёт вид с несколькими полями одновременно."""
        plotter = pv.Plotter(shape=(2, 2), title=f"Многополевая визуализация - Шаг {step+1}/{total_steps}")
        plotter.set_background('black')
        
        available_fields = list(grid.cell_data.keys())
        pressure_field = 'Pressure_MPa' if 'Pressure_MPa' in available_fields else None
        sw_field = 'Water_Saturation' if 'Water_Saturation' in available_fields else None
        sg_field = 'Gas_Saturation' if 'Gas_Saturation' in available_fields else None
        
        z_mid = grid.bounds[4] + (grid.bounds[5] - grid.bounds[4]) / 2
        origin = (grid.bounds[0], grid.bounds[2], z_mid)
        
        # 1. Давление (левый верхний)
        if pressure_field:
            plotter.subplot(0, 0)
            slice_p = grid.slice(normal=(0, 0, 1), origin=origin)
            plotter.add_mesh(slice_p, scalars=pressure_field, cmap='jet', 
                           show_scalar_bar=True, scalar_bar_args={'title': 'МПа'})
            plotter.add_text('Давление', font_size=14, color='white', position='upper_left')
            plotter.camera_position = 'xy'
        
        # 2. Водонасыщенность (правый верхний)
        if sw_field:
            plotter.subplot(0, 1)
            slice_sw = grid.slice(normal=(0, 0, 1), origin=origin)
            plotter.add_mesh(slice_sw, scalars=sw_field, cmap='viridis', 
                           show_scalar_bar=True, scalar_bar_args={'title': 'Sw'})
            plotter.add_text('Водонасыщенность', font_size=14, color='white', position='upper_left')
            plotter.camera_position = 'xy'
        
        # 3. Газонасыщенность (левый нижний)
        if sg_field:
            plotter.subplot(1, 0)
            slice_sg = grid.slice(normal=(0, 0, 1), origin=origin)
            plotter.add_mesh(slice_sg, scalars=sg_field, cmap='plasma', 
                           show_scalar_bar=True, scalar_bar_args={'title': 'Sg'})
            plotter.add_text('Газонасыщенность', font_size=14, color='white', position='upper_left')
            plotter.camera_position = 'xy'
        
        # 4. Объёмная визуализация (правый нижний)
        plotter.subplot(1, 1)
        if pressure_field:
            grid_points = grid.cell_data_to_point_data()
            p_data = grid_points.point_data[pressure_field]
            p_min, p_max = float(p_data.min()), float(p_data.max())
            iso_value = (p_min + p_max) / 2
            contour = grid_points.contour(scalars=pressure_field, isosurfaces=[iso_value])
            plotter.add_mesh(contour, cmap='jet', opacity=0.7, show_scalar_bar=True)
            plotter.add_text(f'Изоповерхность\n{iso_value:.1f} МПа', 
                           font_size=12, color='white', position='upper_left')
        plotter.camera_position = 'iso'
        
        return plotter
    
    def show_animation(self, mode='volume'):
        """Показывает анимацию всех шагов."""
        if not self.vtk_files:
            print("❌ Нет файлов для анимации")
            return
        
        total_steps = len(self.vtk_files)
        print(f"\n🎬 Запуск анимации {total_steps} шагов...")
        print("   Нажмите 'q' для выхода, стрелки для навигации")
        
        for step in range(total_steps):
            print(f"  📊 Загрузка шага {step+1}/{total_steps}...")
            grid = self.get_grid(step)
            
            if mode == 'volume':
                plotter = self.create_volume_visualization(grid, step, total_steps)
            else:
                plotter = self.create_multi_field_view(grid, step, total_steps)
            
            # Показываем на короткое время
            plotter.show(auto_close=False, interactive_update=True)
            plotter.close()
    
    def show_interactive(self, mode='volume'):
        """Интерактивный просмотр с переключением шагов."""
        if not self.vtk_files:
            print("❌ Нет файлов для просмотра")
            return
        
        total_steps = len(self.vtk_files)
        current_step = 0
        
        def update_visualization(step):
            """Обновляет визуализацию для указанного шага."""
            grid = self.get_grid(step)
            if grid is None:
                return None
            
            if mode == 'volume':
                return self.create_volume_visualization(grid, step, total_steps)
            else:
                return self.create_multi_field_view(grid, step, total_steps)
        
        print(f"\n🖥️  Интерактивный просмотр ({total_steps} шагов)")
        print("   Используйте стрелки ← → для навигации")
        print("   'q' для выхода")
        
        while True:
            grid = self.get_grid(current_step)
            if grid is None:
                break
            
            if mode == 'volume':
                plotter = self.create_volume_visualization(grid, current_step, total_steps)
            else:
                plotter = self.create_multi_field_view(grid, current_step, total_steps)
            
            # Добавляем обработчики клавиш
            def next_step():
                nonlocal current_step
                if current_step < total_steps - 1:
                    current_step += 1
                    plotter.close()
                    grid_new = self.get_grid(current_step)
                    if mode == 'volume':
                        plotter_new = self.create_volume_visualization(grid_new, current_step, total_steps)
                    else:
                        plotter_new = self.create_multi_field_view(grid_new, current_step, total_steps)
                    plotter_new.show()
            
            def prev_step():
                nonlocal current_step
                if current_step > 0:
                    current_step -= 1
                    plotter.close()
                    grid_new = self.get_grid(current_step)
                    if mode == 'volume':
                        plotter_new = self.create_volume_visualization(grid_new, current_step, total_steps)
                    else:
                        plotter_new = self.create_multi_field_view(grid_new, current_step, total_steps)
                    plotter_new.show()
            
            plotter.add_key_event('Right', next_step)
            plotter.add_key_event('Left', prev_step)
            
            plotter.show()
            break  # Выходим после закрытия окна


def main():
    parser = argparse.ArgumentParser(
        description='Профессиональная визуализация процесса симуляции',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Интерактивный просмотр с переключением шагов
  python scripts/view_simulation.py results/mega_3phase_million_*/ --interactive
  
  # Анимация всех шагов
  python scripts/view_simulation.py results/mega_3phase_million_*/ --animate
  
  # Многополевая визуализация
  python scripts/view_simulation.py results/mega_3phase_million_*/ --multi-field
        """
    )
    parser.add_argument('results_dir', help='Директория с результатами симуляции')
    parser.add_argument('--interactive', action='store_true', 
                       help='Интерактивный просмотр с переключением шагов')
    parser.add_argument('--animate', action='store_true', 
                       help='Анимация всех шагов')
    parser.add_argument('--multi-field', action='store_true',
                       help='Многополевая визуализация (4 окна)')
    parser.add_argument('--step', type=int, default=None,
                       help='Показать конкретный шаг (по умолчанию: первый)')
    
    args = parser.parse_args()
    
    viewer = SimulationViewer(args.results_dir)
    
    if args.step is not None:
        grid = viewer.get_grid(args.step - 1)
        if grid:
            mode = 'multi' if args.multi_field else 'volume'
            if mode == 'volume':
                plotter = viewer.create_volume_visualization(grid, args.step - 1, len(viewer.vtk_files))
            else:
                plotter = viewer.create_multi_field_view(grid, args.step - 1, len(viewer.vtk_files))
            plotter.show()
    elif args.animate:
        mode = 'multi' if args.multi_field else 'volume'
        viewer.show_animation(mode)
    elif args.interactive:
        mode = 'multi' if args.multi_field else 'volume'
        viewer.show_interactive(mode)
    else:
        # По умолчанию показываем первый шаг
        grid = viewer.get_grid(0)
        if grid:
            mode = 'multi' if args.multi_field else 'volume'
            if mode == 'volume':
                plotter = viewer.create_volume_visualization(grid, 0, len(viewer.vtk_files))
            else:
                plotter = viewer.create_multi_field_view(grid, 0, len(viewer.vtk_files))
            plotter.show()


if __name__ == '__main__':
    main()

