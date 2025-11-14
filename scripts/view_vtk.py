#!/usr/bin/env python3
"""
Скрипт для интерактивного просмотра VTK файлов через PyVista.
Показывает срезы и объёмную визуализацию в интерактивном окне.

Использование:
    python scripts/view_vtk.py results/.../mega_3phase_million_step_1.vtr
"""

import sys
import os
import argparse

try:
    import pyvista as pv
except ImportError:
    print("❌ PyVista не установлен. Установите: pip install pyvista")
    sys.exit(1)


def view_vtk_file(filepath, show_volume=True):
    """
    Открывает VTK файл в интерактивном окне PyVista с объёмной визуализацией.
    """
    if not os.path.exists(filepath):
        print(f"❌ Файл не найден: {filepath}")
        return
    
    print(f"📂 Загрузка файла: {filepath}")
    
    # Загружаем VTK файл
    grid = pv.read(filepath)
    
    print(f"✅ Файл загружен. Размер сетки: {grid.dimensions}")
    print(f"📊 Доступные поля: {list(grid.cell_data.keys())}")
    
    # Получаем доступные поля
    available_fields = list(grid.cell_data.keys())
    pressure_field = 'Pressure_MPa' if 'Pressure_MPa' in available_fields else available_fields[0]
    sw_field = 'Water_Saturation' if 'Water_Saturation' in available_fields else None
    sg_field = 'Gas_Saturation' if 'Gas_Saturation' in available_fields else None
    
    # Создаём plotter для объёмной визуализации
    plotter = pv.Plotter(title="3D Объёмная визуализация резервуара")
    plotter.set_background('black')
    
    if show_volume:
        # Для больших сеток используем объёмный рендеринг
        n_cells = grid.n_cells
        print(f"📦 Количество ячеек: {n_cells:,}")
        
        if n_cells > 500000:
            print("  ⚠ Сетка очень большая, используем разреженную визуализацию...")
            # Для очень больших сеток показываем несколько изоповерхностей
            grid_points = grid.cell_data_to_point_data()
            p_data = grid_points.point_data[pressure_field]
            p_min, p_max = float(p_data.min()), float(p_data.max())
            
            # Создаём несколько изоповерхностей для объёмной визуализации
            iso_values = [
                p_min + (p_max - p_min) * 0.25,
                p_min + (p_max - p_min) * 0.5,
                p_min + (p_max - p_min) * 0.75
            ]
            
            for i, iso_val in enumerate(iso_values):
                contour = grid_points.contour(scalars=pressure_field, isosurfaces=[iso_val])
                opacity = 0.4 + i * 0.2
                plotter.add_mesh(contour, cmap='jet', opacity=opacity, show_scalar_bar=(i == 1))
            
            plotter.add_text(f'Изоповерхности давления\n({p_min:.1f} - {p_max:.1f} МПа)', 
                           font_size=14, color='white', position='upper_left')
        else:
            # Для меньших сеток используем объёмный рендеринг
            print("  🎨 Создаём объёмную визуализацию...")
            
            # Объёмный рендеринг давления
            plotter.add_volume(
                grid,
                scalars=pressure_field,
                cmap='jet',
                opacity='linear',
                show_scalar_bar=True,
                scalar_bar_args={'title': 'Давление (МПа)', 'vertical': True}
            )
            
            # Объёмный рендеринг водонасыщенности (если есть)
            if sw_field:
                plotter.add_volume(
                    grid,
                    scalars=sw_field,
                    cmap='viridis',
                    opacity='linear',
                    show_scalar_bar=True,
                    scalar_bar_args={'title': 'Водонасыщенность', 'vertical': True}
                )
    else:
        # Альтернатива: показываем воксели (кубики) для каждой ячейки
        # Но это очень медленно для миллиона ячеек, поэтому используем объёмный рендеринг
        print("  🎨 Используем объёмный рендеринг...")
        plotter.add_volume(
            grid,
            scalars=pressure_field,
            cmap='jet',
            opacity='linear',
            show_scalar_bar=True
        )
    
    # Настраиваем камеру
    plotter.camera_position = 'iso'
    plotter.reset_camera()
    
    print("\n🖥️  Открывается интерактивное 3D окно...")
    print("   - Вращайте: зажмите левую кнопку мыши и двигайте")
    print("   - Масштабируйте: колесо мыши")
    print("   - Перемещайте: зажмите среднюю кнопку мыши")
    print("   - Нажмите 'q' или закройте окно для выхода")
    print("   - В панели справа можно изменить opacity и другие параметры")
    
    # Показываем интерактивное окно
    plotter.show()


def main():
    parser = argparse.ArgumentParser(description='Просмотр VTK файлов через PyVista')
    parser.add_argument('file', help='Путь к VTK файлу (.vtr)')
    parser.add_argument('--field', default=None, help='Поле для отображения (по умолчанию: Pressure_MPa)')
    
    args = parser.parse_args()
    
    view_vtk_file(args.file)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Использование: python scripts/view_vtk.py <путь_к_vtr_файлу>")
        print("\nПример:")
        print("  python scripts/view_vtk.py results/mega_3phase_million_*/intermediate/mega_3phase_million_step_1.vtr")
        sys.exit(1)
    
    main()

