#!/usr/bin/env python3
"""
Профессиональный GUI просмотрщик для результатов симуляции.
Требует установки: pip install PyQt5 pyvistaqt

Особенности:
- Лаунчер с выбором проектов
- Полноценный оконный интерфейс (Qt)
- Отдельная панель управления
- Стандартные слайдеры и кнопки
"""

import sys
import os
import glob
import datetime
from pathlib import Path
import numpy as np

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QSlider, QLabel, QPushButton, 
                               QCheckBox, QGroupBox, QComboBox, QSplitter, QFrame,
                               QListWidget, QListWidgetItem, QMessageBox)
    from PyQt5.QtCore import Qt, QTimer, QSize
    from PyQt5.QtGui import QFont, QIcon, QColor
    import pyvista as pv
    from pyvistaqt import QtInteractor
except ImportError as e:
    print(f"Ошибка импорта библиотек GUI: {e}")
    print("Для работы этого скрипта необходимо установить PyQt5 и pyvistaqt:")
    print("pip install PyQt5 pyvistaqt")
    sys.exit(1)

# Настройка темы PyVista
pv.global_theme.allow_empty_mesh = True
pv.set_plot_theme("dark")

class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oil Simulator - Project Manager")
        self.resize(800, 600)
        self.setStyleSheet("QMainWindow { background-color: #2b2b2b; color: white; }")
        
        self.init_ui()
        self.scan_projects()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Заголовок
        header = QLabel("Выберите проект для визуализации")
        header.setStyleSheet("font-size: 18pt; font-weight: bold; color: #ffd700; margin-bottom: 10px;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Список проектов
        self.project_list = QListWidget()
        self.project_list.setStyleSheet("""
            QListWidget {
                background-color: #333;
                border: 1px solid #555;
                color: white;
                font-size: 12pt;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #444;
            }
            QListWidget::item:selected {
                background-color: #0055a5;
            }
        """)
        self.project_list.itemDoubleClicked.connect(self.load_selected_project)
        layout.addWidget(self.project_list)
        
        # Кнопки
        btn_layout = QHBoxLayout()
        
        btn_refresh = QPushButton("Обновить список")
        btn_refresh.setStyleSheet("padding: 8px; font-size: 11pt;")
        btn_refresh.clicked.connect(self.scan_projects)
        btn_layout.addWidget(btn_refresh)
        
        btn_load = QPushButton("Загрузить проект")
        btn_load.setStyleSheet("background-color: #006400; color: white; padding: 8px; font-size: 11pt; font-weight: bold;")
        btn_load.clicked.connect(self.load_selected_project)
        btn_layout.addWidget(btn_load)
        
        layout.addLayout(btn_layout)

    def scan_projects(self):
        self.project_list.clear()
        results_path = Path("results")
        
        if not results_path.exists():
            self.project_list.addItem("Папка results/ не найдена")
            return

        # Ищем папки
        projects = []
        for p in results_path.iterdir():
            if p.is_dir():
                # Проверяем наличие VTK файлов
                vtk_files = list((p / "intermediate").glob("*.vtr"))
                if vtk_files:
                    mtime = p.stat().st_mtime
                    dt = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                    projects.append({
                        'path': p,
                        'name': p.name,
                        'time': mtime,
                        'date_str': dt,
                        'steps': len(vtk_files)
                    })
        
        # Сортируем по дате (новые сверху)
        projects.sort(key=lambda x: x['time'], reverse=True)
        
        for proj in projects:
            item_text = f"{proj['date_str']} | {proj['name']} | Шагов: {proj['steps']}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, str(proj['path']))
            self.project_list.addItem(item)
            
        if not projects:
            self.project_list.addItem("Нет проектов с VTK файлами (.vtr)")

    def load_selected_project(self):
        selected_items = self.project_list.selectedItems()
        if not selected_items:
            return
            
        project_path = selected_items[0].data(Qt.UserRole)
        if not project_path:
            return
            
        # Запускаем viewer
        self.viewer = SimulationGUI(project_path, parent_launcher=self)
        self.viewer.show()
        self.hide()

class SimulationGUI(QMainWindow):
    def __init__(self, results_dir, parent_launcher=None):
        super().__init__()
        self.results_dir = Path(results_dir)
        self.parent_launcher = parent_launcher
        self.vtk_files = sorted(glob.glob(str(self.results_dir / "intermediate" / "*.vtr")), 
                              key=lambda x: int(x.split('_step_')[-1].split('.')[0]))
        
        self.current_step = 0
        self.total_steps = len(self.vtk_files)
        
        # Параметры визуализации
        self.params = {
            'field': 'Pressure',
            'show_sw': False,
            'show_sg': False,
            'cmap': 'jet',
            'opacity': 0.9,
            'show_edges': False
        }
        
        # Кэш для сеток
        self.grid_cache = {}
        self.max_cache_size = 10
        
        self.init_ui()
        # Таймер для отложенной первой отрисовки (чтобы UI прогрузился)
        QTimer.singleShot(100, self.load_current_step)

    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle(f"Oil Simulator Viewer - {self.results_dir.name}")
        self.resize(1600, 1000)
        self.setStyleSheet("QMainWindow { background-color: #1e1e1e; color: white; }")
        
        # Основной виджет и layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Сплиттер
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # === Панель управления ===
        control_panel = QFrame()
        control_panel.setStyleSheet("""
            QFrame { background-color: #2b2b2b; color: white; border: none; }
            QLabel { color: #ddd; }
            QGroupBox { font-weight: bold; border: 1px solid #444; margin-top: 10px; padding-top: 10px; border-radius: 4px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #ffd700; }
            QPushButton { background-color: #444; border: none; padding: 5px; border-radius: 3px; }
            QPushButton:hover { background-color: #555; }
            QPushButton:pressed { background-color: #333; }
            QComboBox { background-color: #333; border: 1px solid #555; padding: 4px; }
        """)
        control_panel.setMinimumWidth(320)
        control_panel.setMaximumWidth(380)
        
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(15, 15, 15, 15)
        
        # Кнопка "Назад к проектам"
        btn_back = QPushButton("← Список проектов")
        btn_back.setStyleSheet("background-color: #0055a5; margin-bottom: 10px; padding: 8px;")
        btn_back.clicked.connect(self.close)
        control_layout.addWidget(btn_back)
        
        # Заголовок проекта
        proj_label = QLabel(self.results_dir.name)
        proj_label.setWordWrap(True)
        proj_label.setStyleSheet("font-size: 10pt; color: #888; margin-bottom: 10px;")
        control_layout.addWidget(proj_label)

        # --- Навигация ---
        nav_group = QGroupBox("Временной шаг")
        nav_layout = QVBoxLayout(nav_group)
        
        self.step_label = QLabel(f"0 / {self.total_steps}")
        self.step_label.setAlignment(Qt.AlignCenter)
        self.step_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        nav_layout.addWidget(self.step_label)
        
        self.step_slider = QSlider(Qt.Horizontal)
        self.step_slider.setMinimum(0)
        self.step_slider.setMaximum(self.total_steps - 1)
        self.step_slider.setValue(0)
        self.step_slider.valueChanged.connect(self.on_step_slider)
        nav_layout.addWidget(self.step_slider)
        
        btns_layout = QHBoxLayout()
        self.btn_prev = QPushButton("←")
        self.btn_prev.clicked.connect(self.prev_step)
        self.btn_next = QPushButton("→")
        self.btn_next.clicked.connect(self.next_step)
        btns_layout.addWidget(self.btn_prev)
        btns_layout.addWidget(self.btn_next)
        nav_layout.addLayout(btns_layout)
        
        control_layout.addWidget(nav_group)
        
        # --- Данные ---
        data_group = QGroupBox("Визуализация")
        data_layout = QVBoxLayout(data_group)
        
        data_layout.addWidget(QLabel("Поле:"))
        self.field_combo = QComboBox()
        self.field_combo.addItems(["Pressure", "Sw (Вода)", "Sg (Газ)"])
        self.field_combo.currentTextChanged.connect(self.update_visualization)
        data_layout.addWidget(self.field_combo)
        
        data_layout.addWidget(QLabel("Цветовая схема:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['jet', 'viridis', 'plasma', 'coolwarm', 'hot', 'turbo'])
        self.cmap_combo.currentTextChanged.connect(self.update_visualization)
        data_layout.addWidget(self.cmap_combo)
        
        self.check_sw = QCheckBox("Изоповерхности воды")
        self.check_sw.stateChanged.connect(self.update_visualization)
        data_layout.addWidget(self.check_sw)
        
        self.check_sg = QCheckBox("Изоповерхности газа")
        self.check_sg.stateChanged.connect(self.update_visualization)
        data_layout.addWidget(self.check_sg)
        
        self.check_edges = QCheckBox("Сетка ячеек")
        self.check_edges.stateChanged.connect(self.update_visualization)
        data_layout.addWidget(self.check_edges)

        data_layout.addWidget(QLabel("Прозрачность:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(90)
        self.opacity_slider.valueChanged.connect(self.update_visualization)
        data_layout.addWidget(self.opacity_slider)
        
        control_layout.addWidget(data_group)
        
        # --- Статистика ---
        self.stats_box = QLabel("Загрузка...")
        self.stats_box.setStyleSheet("background-color: #222; padding: 10px; font-family: monospace; border: 1px solid #444;")
        control_layout.addWidget(self.stats_box)
        
        control_layout.addStretch()
        splitter.addWidget(control_panel)
        
        # === 3D Окно ===
        self.plotter = QtInteractor(central_widget)
        self.plotter.set_background('black')
        splitter.addWidget(self.plotter)
        splitter.setSizes([320, 1280])

    def closeEvent(self, event):
        if self.parent_launcher:
            self.parent_launcher.show()
        event.accept()

    def get_grid(self, step_idx):
        if step_idx in self.grid_cache:
            return self.grid_cache[step_idx]
        
        filepath = self.vtk_files[step_idx]
        grid = pv.read(filepath)
        
        if len(self.grid_cache) >= self.max_cache_size:
            del self.grid_cache[list(self.grid_cache.keys())[0]]
            
        self.grid_cache[step_idx] = grid
        return grid

    def load_current_step(self):
        self.step_label.setText(f"{self.current_step + 1} / {self.total_steps}")
        self.step_slider.setValue(self.current_step)
        self.update_visualization()

    def update_visualization(self):
        # Защита от частых обновлений при инициализации
        if not self.vtk_files:
            return

        grid = self.get_grid(self.current_step)
        self.plotter.clear()
        
        field_map = {
            'Pressure': ('Pressure_MPa', 'Давление (МПа)'),
            'Sw (Вода)': ('Sw', 'Водонасыщенность'),
            'Sg (Газ)': ('Sg', 'Газонасыщенность')
        }
        
        sel_field = self.field_combo.currentText()
        vtk_field, title = field_map[sel_field]
        
        # Проверка полей
        if vtk_field == 'Pressure_MPa' and 'Pressure' in grid.array_names:
            vtk_field = 'Pressure'
            # Не конвертируем тут, просто используем что есть, масштаб может быть разным
        
        if vtk_field not in grid.array_names:
            return

        data = grid[vtk_field]
        
        # Статистика
        stats = (f"ПОЛЕ: {title}\n"
                 f"MIN:  {data.min():.4f}\n"
                 f"MAX:  {data.max():.4f}\n"
                 f"MEAN: {data.mean():.4f}")
        self.stats_box.setText(stats)
        
        # Volume Rendering
        opacity_val = self.opacity_slider.value() / 100.0
        cmap = self.cmap_combo.currentText()
        
        self.plotter.add_volume(
            grid, 
            scalars=vtk_field, 
            cmap=cmap,
            opacity='linear',
            opacity_unit_distance=None,
            show_scalar_bar=True,
            scalar_bar_args={'title': title, 'color': 'white'}
        )
        
        # Изоповерхности
        if self.check_sw.isChecked() and 'Sw' in grid.array_names:
            thresh = grid.threshold(0.3, scalars='Sw')
            if thresh.n_points > 0:
                self.plotter.add_mesh(thresh.contour([0.3, 0.5, 0.7], scalars='Sw'), color='blue', opacity=0.4)
                
        if self.check_sg.isChecked() and 'Sg' in grid.array_names:
            thresh = grid.threshold(0.05, scalars='Sg')
            if thresh.n_points > 0:
                self.plotter.add_mesh(thresh.contour([0.05, 0.1], scalars='Sg'), color='red', opacity=0.4)

        if self.check_edges.isChecked():
            self.plotter.add_mesh(grid.outline(), color='white')

    def on_step_slider(self, value):
        self.current_step = value
        self.step_label.setText(f"{self.current_step + 1} / {self.total_steps}")
        self.update_visualization()

    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.load_current_step()

    def next_step(self):
        if self.current_step < self.total_steps - 1:
            self.current_step += 1
            self.load_current_step()

def main():
    app = QApplication(sys.argv)
    
    # Если передан аргумент командной строки - сразу открываем
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            window = SimulationGUI(path)
            window.show()
        else:
            print(f"Путь не найден: {path}")
            sys.exit(1)
    else:
        # Иначе открываем лаунчер
        window = LauncherWindow()
        window.show()
        
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
