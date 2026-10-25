# Инструкция по просмотру VTK файлов в ParaView

## Быстрый старт

1. **Установите ParaView**: https://www.paraview.org/download/

2. **Откройте файл**:
   - File → Open
   - Выберите файл `.vtr` из папки `results/.../intermediate/`

3. **Настройте отображение** (если видите коричневый экран):

### Способ 1: Ручная настройка

1. После открытия файла нажмите **Apply** в панели Properties
2. В панели **Coloring** выберите:
   - **Color by**: `Pressure_MPa` (или `Water_Saturation`, `Gas_Saturation`)
   - **Representation**: `Surface` (вместо `Outline`)
3. Нажмите **Apply**

### Способ 2: Создайте срез

1. В меню: **Filters → Alphabetical → Slice**
2. В Properties:
   - **Origin**: установите Z на середину (например, для сетки 100x100x100 это будет ~100 м)
   - **Normal**: `[0, 0, 1]` (срез по Z)
3. Нажмите **Apply**
4. В **Coloring** выберите `Pressure_MPa` или другое поле

### Способ 3: Изоповерхности

1. **Filters → Alphabetical → Contour**
2. **Contour By**: выберите `Pressure_MPa`
3. **Isosurfaces**: укажите значение (например, `20` для 20 МПа)
4. Нажмите **Apply**

## Доступные поля в VTK файлах

- `Pressure_MPa` - давление в МПа
- `Water_Saturation` - водонасыщенность (0-1)
- `Oil_Saturation` - нефтенасыщенность (0-1)
- `Gas_Saturation` - газонасыщенность (0-1)
- `Perm_Kx_m2`, `Perm_Ky_m2`, `Perm_Kz_m2` - проницаемость

## Полезные операции

- **Вращение**: зажмите левую кнопку мыши и двигайте
- **Масштабирование**: колесо мыши
- **Перемещение**: зажмите среднюю кнопку мыши
- **Анимация**: если открыть несколько файлов, можно создать анимацию через **File → Open Data With Time**

## Автоматическая настройка (скрипт)

Запустите ParaView со скриптом:
```bash
paraview --script=scripts/paraview_setup.py -- results/.../mega_3phase_million_step_1.vtr
```

Или в ParaView: **Tools → Python Shell → Execute Script** → выберите `scripts/paraview_setup.py`

