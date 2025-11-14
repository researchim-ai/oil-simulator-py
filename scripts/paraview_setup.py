"""
Скрипт для автоматической настройки отображения VTK файлов в ParaView.
Использование: paraview --script=scripts/paraview_setup.py -- data.vtr
Или запустите в ParaView: Tools → Python Shell → Execute Script
"""

# Для автоматического запуска в ParaView
try:
    from paraview.simple import *
except ImportError:
    print("Этот скрипт должен запускаться из ParaView")
    print("Использование: paraview --script=paraview_setup.py -- your_file.vtr")
    exit(1)

def setup_visualization():
    """
    Настраивает визуализацию для VTK файлов симулятора.
    """
    # Получаем активный источник (открытый файл)
    source = GetActiveSource()
    
    if source is None:
        print("⚠ Откройте VTK файл сначала (File → Open)")
        return
    
    # Создаём срез по середине Z
    slice = Slice(Input=source)
    slice.SliceType.Origin = source.GetDataInformation().GetBounds()[::2]  # Центр
    slice.SliceType.Normal = [0.0, 0.0, 1.0]  # Срез по Z
    
    # Применяем цветовую схему к срезу
    sliceDisplay = GetDisplayProperties(slice)
    sliceDisplay.Representation = 'Surface'
    sliceDisplay.ColorArrayName = ['CELLS', 'Pressure_MPa']
    sliceDisplay.LookupTable = GetColorTransferFunction('Pressure_MPa')
    
    # Настраиваем цветовую схему для давления
    pressureLUT = GetColorTransferFunction('Pressure_MPa')
    pressureLUT.ApplyPreset('Jet', True)
    pressureLUT.RescaleTransferFunction(15.0, 25.0)  # Диапазон давления в МПа
    
    # Показываем срез
    Show(slice)
    
    # Также создаём изоповерхность для давления
    contour = Contour(Input=source)
    contour.ContourBy = ['CELLS', 'Pressure_MPa']
    contour.Isosurfaces = [20.0]  # Изоповерхность при 20 МПа
    
    contourDisplay = GetDisplayProperties(contour)
    contourDisplay.Representation = 'Surface'
    contourDisplay.ColorArrayName = ['CELLS', 'Pressure_MPa']
    contourDisplay.Opacity = 0.5
    
    Show(contour)
    
    # Настраиваем камеру
    Render()
    ResetCamera()
    
    print("✅ Визуализация настроена!")
    print("   - Срез по Z с цветовым кодированием давления")
    print("   - Изоповерхность давления при 20 МПа")
    print("   - Для изменения поля: выберите источник → Properties → Coloring")

if __name__ == '__main__':
    setup_visualization()

