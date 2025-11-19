#!/usr/bin/env python3
"""
Профессиональный GUI просмотрщик для результатов симуляции (v2.0).
Требует установки: pip install PyQt5 pyvistaqt

Особенности:
- Асинхронная загрузка данных (не блокирует UI)
- Интерактивные ортогональные срезы (Slicing)
- Плеер анимации
- Автообнаружение полей
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
                               QListWidget, QListWidgetItem, QMessageBox, QTabWidget,
                               QSpinBox)
    from PyQt5.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal, QObject
    import pyvista as pv
    from pyvistaqt import QtInteractor
except ImportError as e:
    print(f"Ошибка импорта библиотек GUI: {e}")
    sys.exit(1)

# Настройка темы PyVista
pv.global_theme.allow_empty_mesh = True
pv.set_plot_theme("dark")

# --- Рабочий поток для загрузки файлов ---
class FileLoaderWorker(QObject):
    finished = pyqtSignal(object, int, str)  # grid, step_index, filepath

    def load_file(self, filepath, step_index):
        try:
            grid = pv.read(filepath)
            self.finished.emit(grid, step_index, filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            self.finished.emit(None, step_index, filepath)

# --- Основное окно выбора проектов ---
class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oil Simulator - Project Manager")
        self.resize(900, 600)
        self.setStyleSheet("QMainWindow { background-color: #2b2b2b; color: white; }")
        self.init_ui()
        self.scan_projects()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        header = QLabel("Менеджер проектов")
        header.setStyleSheet("font-size: 20pt; font-weight: bold; color: #ffd700; margin: 10px;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        self.project_list = QListWidget()
        self.project_list.setStyleSheet("""
            QListWidget { background-color: #333; border: 1px solid #555; color: white; font-size: 12pt; }
            QListWidget::item { padding: 12px; border-bottom: 1px solid #444; }
            QListWidget::item:selected { background-color: #0055a5; }
        """)
        self.project_list.itemDoubleClicked.connect(self.load_selected_project)
        layout.addWidget(self.project_list)
        
        btn_layout = QHBoxLayout()
        btn_refresh = QPushButton("Обновить список")
        btn_refresh.clicked.connect(self.scan_projects)
        btn_load = QPushButton("Загрузить проект")
        btn_load.setStyleSheet("background-color: #006400; font-weight: bold;")
        btn_load.clicked.connect(self.load_selected_project)
        
        for btn in [btn_refresh, btn_load]:
            btn.setMinimumHeight(40)
            btn.setStyleSheet(btn.styleSheet() + "QPushButton { color: white; border-radius: 5px; padding: 5px 15px; background-color: #444; } QPushButton:hover { background-color: #555; }")
            if "006400" in btn.styleSheet(): # Green button override
                 btn.setStyleSheet("QPushButton { color: white; border-radius: 5px; padding: 5px 15px; background-color: #2e7d32; } QPushButton:hover { background-color: #388e3c; }")

            btn_layout.addWidget(btn)
            
        layout.addLayout(btn_layout)

    def scan_projects(self):
        self.project_list.clear()
        results_path = Path("results")
        if not results_path.exists():
            self.project_list.addItem("Папка results/ не найдена")
            return

        projects = []
        for p in results_path.iterdir():
            if p.is_dir():
                vtk_files = list((p / "intermediate").glob("*.vtr"))
                if vtk_files:
                    mtime = p.stat().st_mtime
                    dt = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                    projects.append({'path': p, 'name': p.name, 'time': mtime, 'date': dt, 'steps': len(vtk_files)})
        
        projects.sort(key=lambda x: x['time'], reverse=True)
        for proj in projects:
            item = QListWidgetItem(f"{proj['date']}  |  {proj['name']}  |  Шагов: {proj['steps']}")
            item.setData(Qt.UserRole, str(proj['path']))
            self.project_list.addItem(item)

    def load_selected_project(self):
        items = self.project_list.selectedItems()
        if not items: return
        path = items[0].data(Qt.UserRole)
        self.viewer = SimulationGUI(path, self)
        self.viewer.show()
        self.hide()

# --- Окно визуализации ---
class SimulationGUI(QMainWindow):
    start_load_signal = pyqtSignal(str, int)

    def __init__(self, results_dir, parent_launcher=None):
        super().__init__()
        self.results_dir = Path(results_dir)
        self.parent_launcher = parent_launcher
        self.vtk_files = sorted(glob.glob(str(self.results_dir / "intermediate" / "*.vtr")), 
                              key=lambda x: int(x.split('_step_')[-1].split('.')[0]))
        
        self.current_step = 0
        self.total_steps = len(self.vtk_files)
        self.grid_cache = {} # Simple LRU could be added
        self.playing = False
        self.current_grid = None
        
        # Threading setup
        self.thread = QThread()
        self.worker = FileLoaderWorker()
        self.worker.moveToThread(self.thread)
        
        # Connect signals properly
        self.worker.finished.connect(self.on_file_loaded)
        self.start_load_signal.connect(self.worker.load_file)
        
        self.thread.start()

        self.init_ui()
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_step)
        
        # Загрузка первого шага
        self.request_step_load(0)

    def init_ui(self):
        self.setWindowTitle(f"Oil Simulator Viewer v2.0 - {self.results_dir.name}")
        self.resize(1600, 1000)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #e0e0e0; }
            QLabel { color: #e0e0e0; }
            QGroupBox { border: 1px solid #555; margin-top: 1.5ex; border-radius: 4px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #4fc3f7; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: #aaa; padding: 8px 20px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #444; color: white; border-bottom: 2px solid #4fc3f7; }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0,0,0,0)
        
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # === Панель управления ===
        panel = QFrame()
        panel.setMinimumWidth(350)
        panel.setMaximumWidth(400)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10,10,10,10)
        
        # Top buttons
        top_btns = QHBoxLayout()
        btn_back = QPushButton("← Меню")
        btn_back.clicked.connect(self.close)
        top_btns.addWidget(btn_back)
        panel_layout.addLayout(top_btns)
        
        # Project Info
        panel_layout.addWidget(QLabel(f"Проект: {self.results_dir.name}"))
        
        # --- Вкладки управления ---
        tabs = QTabWidget()
        
        # TAB 1: Основные настройки
        tab_main = QWidget()
        l_main = QVBoxLayout(tab_main)
        
        # Навигация
        grp_nav = QGroupBox("Временная шкала")
        l_nav = QVBoxLayout(grp_nav)
        
        self.lbl_step = QLabel(f"Шаг: 0 / {self.total_steps}")
        self.lbl_step.setAlignment(Qt.AlignCenter)
        self.lbl_step.setStyleSheet("font-size: 14px; font-weight: bold;")
        l_nav.addWidget(self.lbl_step)
        
        self.slider_step = QSlider(Qt.Horizontal)
        self.slider_step.setMaximum(self.total_steps - 1)
        self.slider_step.valueChanged.connect(self.on_slider_change)
        l_nav.addWidget(self.slider_step)
        
        play_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◄")
        self.btn_prev.clicked.connect(self.prev_step)
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_next = QPushButton("►")
        self.btn_next.clicked.connect(self.next_step)
        play_layout.addWidget(self.btn_prev)
        play_layout.addWidget(self.btn_play)
        play_layout.addWidget(self.btn_next)
        l_nav.addLayout(play_layout)
        l_main.addWidget(grp_nav)
        
        # Данные
        grp_view = QGroupBox("Отображение")
        l_view = QVBoxLayout(grp_view)
        
        l_view.addWidget(QLabel("Поле данных:"))
        self.combo_field = QComboBox()
        self.combo_field.currentTextChanged.connect(self.refresh_view)
        l_view.addWidget(self.combo_field)
        
        l_view.addWidget(QLabel("Цветовая карта:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(['jet', 'viridis', 'plasma', 'magma', 'seismic'])
        self.combo_cmap.currentTextChanged.connect(self.refresh_view)
        l_view.addWidget(self.combo_cmap)
        
        self.chk_edges = QCheckBox("Показать сетку")
        self.chk_edges.stateChanged.connect(self.refresh_view)
        l_view.addWidget(self.chk_edges)
        
        l_main.addWidget(grp_view)
        l_main.addStretch()
        tabs.addTab(tab_main, "Общее")
        
        # TAB 2: Volume & Isosurfaces
        tab_vol = QWidget()
        l_vol = QVBoxLayout(tab_vol)
        
        self.chk_volume = QCheckBox("Volume Rendering (Объем)")
        self.chk_volume.setChecked(True)
        self.chk_volume.stateChanged.connect(self.refresh_view)
        l_vol.addWidget(self.chk_volume)
        
        l_vol.addWidget(QLabel("Прозрачность объема:"))
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(80)
        self.slider_opacity.valueChanged.connect(self.refresh_view)
        l_vol.addWidget(self.slider_opacity)
        
        l_vol.addWidget(QLabel("--- Изоповерхности ---"))
        self.chk_iso_sw = QCheckBox("Sw (Вода > 0.3)")
        self.chk_iso_sw.stateChanged.connect(self.refresh_view)
        l_vol.addWidget(self.chk_iso_sw)
        
        self.chk_iso_sg = QCheckBox("Sg (Газ > 0.05)")
        self.chk_iso_sg.stateChanged.connect(self.refresh_view)
        l_vol.addWidget(self.chk_iso_sg)
        
        l_vol.addStretch()
        tabs.addTab(tab_vol, "3D Объем")
        
        # TAB 3: Slicing (Срезы)
        tab_slice = QWidget()
        l_slice = QVBoxLayout(tab_slice)
        
        self.chk_slices = QCheckBox("Включить срезы")
        self.chk_slices.stateChanged.connect(self.refresh_view)
        l_slice.addWidget(self.chk_slices)
        
        self.slice_widgets = {}
        for axis in ['X', 'Y', 'Z']:
            box = QGroupBox(f"Срез по {axis}")
            bl = QVBoxLayout(box)
            chk = QCheckBox(f"Показать {axis}")
            chk.setChecked(axis == 'Z') # Default Z slice
            chk.stateChanged.connect(self.refresh_view)
            
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 100) # Will update based on grid bounds
            sl.setValue(50)
            sl.valueChanged.connect(self.refresh_view)
            
            bl.addWidget(chk)
            bl.addWidget(sl)
            l_slice.addWidget(box)
            self.slice_widgets[axis] = {'chk': chk, 'slider': sl}
            
        l_slice.addStretch()
        tabs.addTab(tab_slice, "Срезы")
        
        panel_layout.addWidget(tabs)
        
        # Stats box
        self.stats_label = QLabel("Загрузка...")
        self.stats_label.setStyleSheet("background: #222; padding: 5px; border: 1px solid #444; font-family: monospace;")
        panel_layout.addWidget(self.stats_label)
        
        splitter.addWidget(panel)
        
        # === 3D View ===
        self.plotter = QtInteractor(central)
        splitter.addWidget(self.plotter)
        splitter.setSizes([350, 1250])

    def closeEvent(self, event):
        self.thread.quit()
        self.thread.wait()
        if self.parent_launcher:
            self.parent_launcher.show()
        event.accept()

    # --- Logic ---
    def request_step_load(self, idx):
        if 0 <= idx < self.total_steps:
            if idx in self.grid_cache:
                self.on_file_loaded(self.grid_cache[idx], idx, "cache")
            else:
                # Load in background
                # Only load if not already processing that step? 
                # For now, just emit.
                self.start_load_signal.emit(self.vtk_files[idx], idx)

    def on_slider_change(self, val):
        if val != self.current_step:
            self.current_step = val
            self.lbl_step.setText(f"Шаг: {val} / {self.total_steps}")
            self.request_step_load(val)

    def prev_step(self):
        self.slider_step.setValue(max(0, self.current_step - 1))

    def next_step(self):
        next_idx = self.current_step + 1
        if next_idx >= self.total_steps:
            if self.playing:
                next_idx = 0 # Loop
            else:
                return
        self.slider_step.setValue(next_idx)

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.btn_play.setText("|| Stop")
            self.timer.start(500) # 2 FPS
        else:
            self.btn_play.setText("▶ Play")
            self.timer.stop()

    def on_file_loaded(self, grid, step_idx, filepath):
        if grid is None:
            return
            
        # Cache
        if len(self.grid_cache) > 10:
            del self.grid_cache[list(self.grid_cache.keys())[0]]
        self.grid_cache[step_idx] = grid
        
        # Only update if relevant (user might have scrolled away while loading)
        # OR if playing, we want to see frames even if skipped
        if step_idx == self.current_step:
            self.current_grid = grid
            self.update_fields_combo(grid)
            self.refresh_view()

    def update_fields_combo(self, grid):
        current_txt = self.combo_field.currentText()
        self.combo_field.blockSignals(True)
        self.combo_field.clear()
        
        # Prefer nicely named aliases
        display_names = []
        self.field_mapping = {}
        
        for name in grid.array_names:
            pretty = name
            if name == 'Pressure_MPa': pretty = 'Pressure (MPa)'
            elif name == 'Sw': pretty = 'Sw (Water)'
            elif name == 'Sg': pretty = 'Sg (Gas)'
            
            display_names.append(pretty)
            self.field_mapping[pretty] = name
            
        self.combo_field.addItems(display_names)
        
        # Try to restore selection or default to Pressure
        if current_txt in display_names:
            self.combo_field.setCurrentText(current_txt)
        elif 'Pressure (MPa)' in display_names:
            self.combo_field.setCurrentText('Pressure (MPa)')
        elif len(display_names) > 0:
             self.combo_field.setCurrentIndex(0)
            
        self.combo_field.blockSignals(False)

    def refresh_view(self):
        if self.current_grid is None:
            return
            
        grid = self.current_grid
        self.plotter.clear()
        
        # Get Field
        field_display = self.combo_field.currentText()
        if not field_display: return
        field_name = self.field_mapping.get(field_display, field_display)
        
        # Update Slice Sliders Range
        bounds = grid.bounds
        
        # Visualization Logic
        cmap = self.combo_cmap.currentText()
        opacity = self.slider_opacity.value() / 100.0
        
        # 1. Volume Rendering
        if self.chk_volume.isChecked() and not self.chk_slices.isChecked():
            self.plotter.add_volume(grid, scalars=field_name, cmap=cmap, opacity=opacity, 
                                  show_scalar_bar=True, scalar_bar_args={'title': field_display, 'color':'white'})
                                  
        # 2. Slices
        if self.chk_slices.isChecked():
            for ax_idx, axis in enumerate(['X', 'Y', 'Z']):
                w = self.slice_widgets[axis]
                if w['chk'].isChecked():
                    val_pct = w['slider'].value() / 100.0
                    # Calculate coordinate
                    min_b, max_b = bounds[2*ax_idx], bounds[2*ax_idx+1]
                    pos = min_b + (max_b - min_b) * val_pct
                    
                    # Correct origin for slice: needs point on plane.
                    origin = [bounds[0] + (bounds[1]-bounds[0])/2, bounds[2] + (bounds[3]-bounds[2])/2, bounds[4] + (bounds[5]-bounds[4])/2]
                    origin[ax_idx] = pos
                    
                    if axis == 'X': normal = (1,0,0)
                    elif axis == 'Y': normal = (0,1,0)
                    else: normal = (0,0,1)

                    slice_mesh = grid.slice(normal=normal, origin=origin)
                    self.plotter.add_mesh(slice_mesh, scalars=field_name, cmap=cmap, show_scalar_bar=True)
                    
                    # Show plane outline
                    self.plotter.add_mesh(pv.Plane(center=origin, direction=normal, i_size=(bounds[2*1+1]-bounds[2*1]) if axis=='X' else (bounds[1]-bounds[0]), 
                                                 j_size=(bounds[5]-bounds[4]) if axis!='Z' else (bounds[3]-bounds[2])), 
                                        style='wireframe', color='white', opacity=0.2)

        # 3. Isosurfaces (Contours)
        if self.chk_iso_sw.isChecked() and 'Sw' in grid.array_names:
            try:
                cnt = grid.contour([0.3, 0.5, 0.7], scalars='Sw')
                self.plotter.add_mesh(cnt, color='blue', opacity=0.3, label='Sw')
            except: pass
            
        if self.chk_iso_sg.isChecked() and 'Sg' in grid.array_names:
            try:
                cnt = grid.contour([0.05, 0.1], scalars='Sg')
                self.plotter.add_mesh(cnt, color='red', opacity=0.3, label='Sg')
            except: pass
            
        # 4. Edges
        if self.chk_edges.isChecked():
            self.plotter.add_mesh(grid.outline(), color='white')

        # Stats
        data = grid[field_name]
        stats = (f"Файл: {Path(self.vtk_files[self.current_step]).name}\n"
                 f"Поле: {field_display}\n"
                 f"Мин: {data.min():.4f} | Макс: {data.max():.4f}\n"
                 f"Ячеек: {grid.n_cells}")
        self.stats_label.setText(stats)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LauncherWindow()
    win.show()
    sys.exit(app.exec_())
