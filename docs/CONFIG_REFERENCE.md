# Справочник параметров конфигурации

Этот раздел описывает формат файлов `configs/*.json`, используемых для запуска симуляций.  
Документ охватывает **все поддерживаемые параметры** на верхнем уровне и внутри основных блоков: `simulation`, `reservoir`, `fluid`, `wells`.

## 1. Общая структура JSON

```json
{
  "description": "Необязательное текстовое описание кейса",
  "output_filename": "имя_выходного_набора",
  "save_vtk": false,

  "simulation": { ... },
  "reservoir": { ... },
  "fluid": { ... },
  "wells": [ ... ]
}
```

| Параметр            | Тип / значение по умолчанию | Описание |
|---------------------|-----------------------------|----------|
| `description`       | `string`, optional          | Свободный комментарий, отображается в логах. |
| `output_filename`   | `string`, optional (default: timestamp) | Базовое имя каталога и отчётных файлов в `results/`. |
| `save_vtk`          | `bool`, default `false`     | Если `true`, финальное состояние записывается в VTK (`.vtr`). |
| `simulation`        | `dict`, required            | Настройки временной схемы, решателей и вывода. |
| `reservoir`         | `dict`, required            | Геометрия, пористость, проницаемость, стохастические поля. |
| `fluid`             | `dict`, required            | Начальные условия и модели PVT/относительных проницаемостей. |
| `wells`             | `list`, optional            | Список скважин; если пустой, рассчитывается без источников. |

## 2. Блок `simulation`

| Параметр | Тип / default | Описание |
|----------|---------------|----------|
| `solver_type` | `"impes"` \| `"fully_implicit"` (default `"impes"`) | Выбор основной схемы. FI требует меньше ограничений по dt, но дороже. |
| `total_time_days` | `float`, default `100.0` | Общая продолжительность расчёта. |
| `time_step_days` | `float`, default `1.0` | Базовый шаг по времени. |
| `max_time_step_attempts` | `int`, default `5` | Количество попыток уменьшить шаг, если решение не сходится. |
| `dt_reduction_factor` | `float`, default `2.0` (`1.5` в FI) | Во сколько раз уменьшать шаг при неудаче. |
| `dt_increase_factor` | `float`, default `1.25` (`1.5` в FI) | Лимит роста шага при успехе. |
| `use_cuda` | `bool`, default `false` | Разрешает перенос данных на GPU (актуально и для FI). |
| `pressure_solver` | `"cg"` \| `"amg"`, default `"cg"` | Решатель для этапа давления. |
| `pressure_float64` | `bool`, default `false` | Переводит систему давления в float64 перед CG. |
| `cg_tolerance` | `float`, default `1e-6` | Допуск CG; также используется как fallback для AMG. |
| `cg_max_iter` | `int`, default `500` | Максимальное число итераций CG. |
| `global_rate_scale` | `float`, default `1.0` | Масштабирование всех заданных дебитов (удобно для чувствительности). |
| `max_substeps` | `int`, default `20` | Верхний лимит подшагов насыщенности (IMPES). |
| `max_saturation_change` | `float`, default `0.05` | Максимальное изменение насыщенности за подшаг (IMPES). |
| `use_capillary_potentials` | `bool`, default `false` | Включает апстрим по полным потенциалам $p \pm p_c \pm ρgΔz$. При `true` рекомендуется уменьшить `time_step_days`. |
| `save_interval` | `int`, default `10` | Шаг сохранения промежуточных PNG в `results/.../intermediate`. |
| `animation_fps` | `int`, default `5` | Частота кадров итоговой анимации (при `save_interval < num_steps`). |
| `write_full_arrays_txt` | `bool`, default `false` | Записывать ли полные 3D-массивы в текстовый отчёт (может быть очень объёмно). |
| `save_npz` | `bool`, default `true` | Сохранять ли финальные поля в `npz`. |
| `debug_component_balance` | `bool`, default `false` | Включает детальный лог компонентного баланса. |
| `debug_log_path` | `string`, optional | Пользовательский путь для debug-лога; если не указан, создаётся `results/component_debug_*.log`. |

### 2.1 Параметры AMG (`simulation.amg`)

| Параметр | Тип / default | Описание |
|----------|---------------|----------|
| `theta` | `float`, default `0.25` | Порог сильных связей (Classical RS). |
| `max_levels` | `int`, default `10` | Максимальная глубина иерархии. |
| `coarsest_size` | `int`, default `200` | Целевой размер самого coarse-уровня. |
| `tol` | `float`, default `1e-6` | Допуск по резидуалу для V-циклов. |
| `max_cycles` | `int`, default `20` | Максимальное число V-циклов. |
| `device` | `"auto"` \| `"cpu"` \| `"cuda"` | Где строить иерархию; `"auto"` выбирает GPU при наличии. |
| `mixed_precision` | `bool`, default `false` | Перевод уровней ≥ `mixed_start_level` в float32. |
| `mixed_start_level` | `int`, default `2` | Уровень, с которого применять float32. |
| `cpu_offload` | `bool`, default `false` | Перенос уровней ≥ `offload_level` на CPU после сборки. |
| `offload_level` | `int`, default `3` | Первый уровень для offload. |

> Если `mixed_precision` или `cpu_offload` включены, параметры остаются опциональными: по умолчанию решение выполняется полностью на GPU в float64.

### 2.2 Дополнительные параметры для fully implicit

| Параметр | Тип / default | Описание |
|----------|---------------|----------|
| `newton_max_iter` | `int`, default `20` | Максимальное число итераций Ньютона. |
| `newton_tolerance` | `float`, default `1e-3` | Целевой относительный резидуал. |
| `damping_factor` | `float`, default `0.7` | Коэффициент демпфирования обновления (`λ ∈ (0,1]`). |
| `jacobian_regularization` | `float`, default `1e-7` | Добавляется на диагональ якобиана для устойчивости. |

## 3. Блок `reservoir`

| Параметр | Тип / default | Описание |
|----------|---------------|----------|
| `dimensions` | `[Nx, Ny, Nz]`, **обязателен** | Число ячеек вдоль каждой оси. |
| `grid_size` | `[Δx, Δy, Δz]`, обязательный базовый шаг (м) | Используется как усреднение и fallback для `grid`. |
| `porosity` | `float` \| массив \| `.npy/.npz`, default `0.2` | Фиксированная или пространственная пористость. |
| `permeability` | `float` \| массив \| `.npy/.npz`, default `100.0` | Горизонтальная проницаемость в мД. |
| `k_vertical_fraction` | `float`, default `0.1` | Отношение $k_v / k_h$ (если нет независимого поля). |
| `c_rock` | `float`, default `1e-5` (1/МПа) | Сжимаемость породы. |

### 3.1 Подблок `reservoir.grid` (опционально)

| Параметр | Тип | Описание |
|----------|-----|----------|
| `dx`, `dy`, `dz` | массив длиной `Nx`, `Ny`, `Nz` | Неравномерный шаг вдоль оси (в метрах). |
| `x_nodes`, `y_nodes`, `z_nodes` | массив длиной `N+1` | Альтернатива: координаты узлов. Разности вычисляются автоматически. |

Можно комбинировать: например, задать `dx`, но использовать `y_nodes`. Если подблок отсутствует, сетка равномерна.

> ⚠️ Fully implicit решатель в текущей версии поддерживает только равномерные сетки (`grid` не задан или все шаги одинаковы).

### 3.2 Подблок `reservoir.stochastic` (опционально)

| Параметр | Тип / default | Описание |
|----------|---------------|----------|
| `seed` | `int`, default `0` | Базовый seed генератора. |
| `porosity` | `dict`, optional | Генерация поля пористости. |
| `permeability` | `dict`, optional | Генерация поля проницаемости (в мД). |

Внутри `porosity`/`permeability` доступны:

| Ключ | Тип / default | Значение |
|------|---------------|----------|
| `mean` | `float`, default текущее среднее | Среднее значение поля. |
| `std` | `float`, default текущее σ или 0.01 | Стандартное отклонение. |
| `corr_length` | `float`, default `0.0` | Корреляционный радиус (в ячейках). При наличии SciPy используется `gaussian_filter`. |
| `distribution` | `"normal"` или `"lognormal"`, default `"normal"` | Форма распределения. |
| `log_mean`, `log_std` | `float`, optional (lognormal) | Параметры лог-нормального распределения. |
| `min`, `max` | `float`, optional | Жёсткие границы (для пористости). |
| `min_md`, `max_md` | `float`, optional | Границы проницаемости в миллидарси. |
| `k_vertical_fraction` | `float`, optional (только для permeability) | Переопределяет вертикальное отношение для сгенерированного поля. |
| `seed` | `int`, optional | Отдельный seed; если не задан, используются `seed`, `seed+1`. |

## 4. Блок `fluid`

| Параметр | Тип / default | Описание |
|----------|---------------|----------|
| `pressure` | `float`, default `20.0` (МПа) | Начальное пластовое давление. |
| `s_w`, `s_g` | `float`, defaults `0.2`, `0.0` | Начальные насыщенности воды и газа. Нефть берётся как `1 - s_w - s_g`. |
| `mu_oil`, `mu_water`, `mu_gas` | `float`, defaults `1.0`, `0.5`, `0.02` (сП) | Вязкости при отсутствии PVT. |
| `rho_oil`, `rho_water`, `rho_gas` | `float`, defaults `850`, `1000`, `1.0` (кг/м³) | Плотности при стандартных условиях. |
| `c_oil`, `c_water`, `c_gas`, `c_rock` | `float`, defaults `1e-5`, `1e-5`, `1e-3`, `1e-5` (1/МПа) | Сжимаемости фаз и породы (используются без PVT). |
| `pvt_path` | `string`, optional | Путь к JSON или Eclipse/CMG deck. При указании автоматически загружаются PVT и таблицы SWOF/SGOF. |
| `pvt` | `dict`, optional | Альтернатива: словарь с полями `pressure_MPa`, `Bo`, `Bw`, `Bg`, `mu_o_cP`, ... (см. `tests/test_deck_loader`). |

### 4.1 Relative permeability (`fluid.relative_permeability`)

| Параметр | Тип / default | Описание |
|----------|---------------|----------|
| `model` | `"corey"` \| `"stone2"` \| `"table"` (default `"corey"`) | Выбор модели. |

**Corey/Stone II параметры (используются по мере необходимости):**

- `sw_cr`, `so_r`, `sg_cr` — критические/остаточные насыщенности (`default sw_cr=0.2, so_r=0.2, sg_cr=0.0`).
- `nw`, `no`, `ng` — степенные показатели кривых (по умолчанию `2.0`).
- `ko_end_w`, `ko_end_g` — конечные значения нефтефазной проницаемости по воде/газу (для Stone II).
- `now`, `nog` — степени нефтефазных ветвей (Stone II; default совпадает с `no`/`ng`).
- `krw_end`, `krg_end` — конечные значения водной и газовой кривой (Corey).

**Табличный режим (`model: "table"`):**

- `path` — путь к deck-файлу или `.json`. Требуются секции **SWOF** (вода–нефть) и/или **SGOF** (газ–нефть).
- Если PVT загружается из того же deck, отдельный `path` можно не указывать.
- Симулятор автоматически строит производные `dkr/dS` и кривые Pc из предоставленных данных.

### 4.2 Capillary pressure

| Блок | Параметры | Описание |
|------|-----------|----------|
| `capillary_pressure` | `pc_scale`, `pc_exponent`, `pc_threshold` (defaults `0`, `1.5`, `0.01`) | Давление между водой и нефтью (степенная зависимость). |
| `capillary_pressure_og` | `pc_scale`, `pc_exponent` (defaults `0`, `1.5`) | Аналогично для пары нефть–газ. |

При ненулевом `pc_scale` рекомендуется включить `simulation.use_capillary_potentials`, чтобы потенциалы учитывали Pc при апстриминге.

## 5. Блок `wells`

Каждый элемент списка описывает отдельную скважину.

| Параметр | Тип / default | Описание |
|----------|---------------|----------|
| `name` | `string` | Уникальное имя. |
| `type` | `"injector"` \| `"producer"` | Направление потока. |
| `i`, `j`, `k` \| `coordinates` | Целочисленные индексы ячейки (оба формата поддерживаются). |
| `radius` | `float` (м) | Радиус ствола (используется в модели Писмана). |
| `control_type` | `"rate"` \| `"bhp"` | Режим управления. В старом формате — `control.type`. |
| `control_value` | `float` | Для `rate`: дебит в м³/сут; для `bhp`: забойное давление в МПа. Старый формат `control.value`. |
| `rate_type` | `"reservoir"` \| `"surface"`, default `"reservoir"` | Относится только к режиму `rate`. |
| `surface_phase` | `"oil"` \| `"water"` \| `"gas"` \| `"liquid"` | Требуется если `rate_type="surface"` и скважина — producer. |
| `injected_phase` | `"water"` \| `"gas"` (default `"water"`) | Для инжекторов c `rate_type` `'surface'`. |
| `limits` | `dict`, optional | Поверхностные лимиты (м³/сут). Поддерживаются ключи `wopr` (oil), `wlpr` (water), `wgpr` (gas), `liqr` (oil+water). |
| `bhp_min` | `float` (МПа), optional | Минимально допустимый BHP (для продакшн-скважин под `rate`). Если давление в ячейке ниже, дебит автоматически тюнингуется. |

> В режиме `surface` симулятор самостоятельно переводит дебит в пластовые условия с использованием текущих $B\_\alpha$, $R_s$, $R_v$.

## 6. Вывод и результаты

- Все результаты сохраняются в `results/<output_filename>_<timestamp>/`.
- Доступные опции:
  - `save_vtk`: финальное состояние в формате `.vtr` (поддерживает ParaView).
  - `simulation.write_full_arrays_txt`: полный дамп массивов в текстовый файл (осторожно, очень большие файлы).
  - `simulation.save_npz`: финальные поля (`pressure`, `sw`, `sg`, `so`) в сжатом `npz`.
  - `simulation.save_interval`, `simulation.animation_fps`: промежуточные PNG и итоговая анимация.
  - `simulation.debug_component_balance` / `debug_log_path`: подробный лог компонентных балансов по шагам, удобен для аудита.

---

### Быстрый шаблон

```json
{
  "description": "3-фазный импульс с управляемой капилляркой и стохастикой",
  "output_filename": "demo_run",
  "save_vtk": true,

  "simulation": {
    "solver_type": "impes",
    "total_time_days": 30,
    "time_step_days": 0.5,
    "max_substeps": 30,
    "max_saturation_change": 0.03,
    "use_capillary_potentials": true,
    "pressure_solver": "amg",
    "amg": {
      "theta": 0.2,
      "max_levels": 15,
      "coarsest_size": 150,
      "mixed_precision": true,
      "cpu_offload": true,
      "offload_level": 4
    }
  },

  "reservoir": {
    "dimensions": [60, 40, 6],
    "grid_size": [20.0, 20.0, 4.0],
    "grid": {
      "x_nodes": [0, 40, 80, 120, 200, 280, 360],
      "dy": [15, 15, 20, 25, 25, 30, 35, 40, 40, 45],
      "dz": [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    },
    "porosity": 0.22,
    "permeability": 150.0,
    "k_vertical_fraction": 0.12,
    "stochastic": {
      "seed": 42,
      "porosity": {"std": 0.04, "corr_length": 1.5, "min": 0.05, "max": 0.35},
      "permeability": {"mean": 180.0, "std": 40.0, "distribution": "lognormal", "corr_length": 2.0, "k_vertical_fraction": 0.08}
    }
  },

  "fluid": {
    "pressure": 22.0,
    "s_w": 0.25,
    "s_g": 0.05,
    "pvt_path": "configs/pvt/pvt_synthetic.json",
    "relative_permeability": {"model": "stone2", "sw_cr": 0.15, "so_r": 0.18, "ko_end_w": 0.95, "ko_end_g": 0.85},
    "capillary_pressure": {"pc_scale": 6e4, "pc_exponent": 1.4},
    "capillary_pressure_og": {"pc_scale": 4e4, "pc_exponent": 1.3}
  },

  "wells": [
    {"name": "INJ-W1", "type": "injector", "i": 5, "j": 5, "k": 0, "radius": 0.15,
     "control_type": "rate", "control_value": 500.0, "injected_phase": "water"},
    {"name": "INJ-G1", "type": "injector", "i": 50, "j": 30, "k": 2, "radius": 0.15,
     "control_type": "rate", "control_value": 300.0, "rate_type": "surface", "injected_phase": "gas"},
    {"name": "PROD-1", "type": "producer", "i": 25, "j": 18, "k": 0, "radius": 0.12,
     "control_type": "rate", "control_value": 2500.0, "rate_type": "surface", "surface_phase": "oil",
     "limits": {"wopr": 1200.0, "liqr": 1800.0}, "bhp_min": 11.0}
  ]
}
```

Используйте этот шаблон как отправную точку и уточняйте параметры под конкретный сценарий.

