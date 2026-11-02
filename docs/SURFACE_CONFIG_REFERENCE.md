# Справочник наземных конфигураций

Этот справочник описывает текущий API; исторические результаты прогонов
хранятся отдельно. Области применимости: [наземное моделирование](SURFACE_MODELING.md).
Все давления абсолютные, температура в K, массовые расходы в кг/с,
длины в м, мощность в Вт. В PVT ось давления задаётся в МПа, вязкость — в сП.

## Жидкостная сеть

`model: incompressible_liquid`, блоки `fluid`, `nodes`, `pipes`,
необязательные `pumps` и `valves`.

- `fluid`: `density_kg_m3`, `viscosity_pa_s`, `vapor_pressure_pa` (по умолчанию 0).
- Узел: `name`, `elevation_m` (0), `source_mass_rate_kg_s` (0),
  необязательное фиксированное `pressure_pa`.
- Труба: `name`, `start`, `end`, `length_m`, `diameter_m`, `roughness_m` (0),
  `minor_loss_coefficient` (0), `open` (true).
- Насос: `name`, `start`, `end`, `shutoff_head_m`,
  `curve_coefficient_s2_m5`, `max_flow_m3_s`, `speed_ratio`,
  `efficiency`, `max_shaft_power_w`, `check_valve`, `open`.
  Табличные карты `head_curve`, `efficiency_curve`, `npsh_curve` содержат
  `flow_m3_s` и `values`. NPSH проверяется при заданных характеристике
  и диаметре входа `inlet_diameter_m`.
- Жидкостный дроссель: `name`, `start`, `end`, `diameter_m`,
  `discharge_coefficient`, `opening`, `open`.

Имена ветвей уникальны; каждая связная часть требует опоры давления.
Закрытая ветвь не связывает давления. Изолированному объёму не назначается
произвольное давление. Рабочие примеры содержат полный набор параметров:
`surface_liquid_booster.json`, `surface_mapped_pump_valve.json`.

```bash
python scripts/run_surface_network.py configs/surface_liquid_ring.json --output /tmp/network.json
python scripts/run_surface_network.py configs/surface_liquid_ring.json --initial-state /tmp/network.json
```

### Управление и станция

Карты управления: `pump_speed_ratios`, `pump_status`, `valve_openings`.
`PumpPressureControl`: `pump`, `node`, `target_pressure_pa`,
`min_speed_ratio`, `max_speed_ratio`, `pressure_tolerance_pa`, `max_evaluations`,
`method` (`bisect`/`newton`), `response_direction` (`increasing`/`decreasing`/`auto`),
`linked_pumps` (общая скорость группы).

В площадке `pump_station_control` задаёт `pumps` (1–8 параллельных агрегатов
с обратными клапанами), `node`, `target_pressure_pa`, `min_speed_ratio`,
`max_speed_ratio`, `min_running`, `max_running`, `max_station_power_w`,
`min_npsh_margin_m`, `min_pump_flow_m3_s`, `switch_saving_w`,
`pressure_tolerance_pa`, `scan_points`, `max_configurations`,
`max_control_evaluations`, `response_direction`.

Перебор ограничен бюджетом и сканированием скоростей; глобальная оптимальность
не гарантируется. Положительный требуемый запас NPSH требует соответствующей
характеристики насоса. Численный отказ не считается отсутствием мощности.

## Сепарация и площадка

`SurfaceFacility.from_dict` принимает `stock_densities`, `feeds`, `pvt`,
`separator_stages`, `water_network`, `water_feed_node`, `limits`,
необязательные `gas_export` и `pump_station_control`.

`stock_densities`: `oil_kg_m3`, `water_kg_m3`, `gas_kg_m3`.
Поток `feeds`: `oil_m3_s`, `water_m3_s`, `gas_m3_s` — стандартные объёмы
консервативных компонентов, включая растворённый газ. Ступень сепарации:
`name`, `pressure_pa`. Недонасыщенная нефть требует покрывающих PVTO;
Rv и унос компонентов не поддержаны.

Ограничение `limits`: `name`, `metric`, `unit`, `minimum` и/или `maximum`.
Неизвестная метрика или несовпадение единиц — ошибка, не успешная проверка.
`FacilityCapacityControl` задаёт `min_fraction`, `max_fraction`,
`fraction_tolerance`, `scan_points`, `max_evaluations` и ограничивает заданную
подачу. Для обратной связи с пластом нужен отдельный блок `coupled_facility`.

```bash
python scripts/run_surface_facility.py configs/surface_facility.json --output /tmp/facility.json
python scripts/run_surface_gathering.py configs/surface_gathering_facility.json --output /tmp/gathering.json
```

## Газовый экспорт

`gas_export`: `header_pressure_pa`, `inlet_temperature_k`, `cooling_temperature_k`,
`standard_pressure_pa`, `standard_temperature_k`, `heat_capacity_ratio`,
`viscosity_pa_s`, `isentropic_efficiency`, `mechanical_efficiency`,
`max_stage_pressure_ratio`, `cooler_pressure_loss_fraction`,
`max_discharge_temperature_k`, `max_stages`, необязательные `export_pipe`, `export_valve`.

`export_pipe`: `name`, `length_m`, `diameter_m`, `roughness_m`.
Газ должен быть сухим и согласованным с идеальным газовым приближением по
плотности на выходе сепарации. Реальный газ и конденсация не подменяются Z=1.

`export_valve`: `name`, `outlet_pressure_pa`, `throat_area_m2`,
`discharge_coefficient` (1), `max_opening` (1). Cd в (0,1], предел открытия в [0,1].
Элемент расположен после экспортной трубы. Требуемое открытие рассчитывается
под фактическую подачу; превышение пропускной способности — физический отказ.
Это сопловой эквивалент Cd A opening. На критической полке обратное давление
не определяется единственным образом.

## Black-oil дерево

`model: homogeneous_blackoil_tree`, `pvt`, `stock_densities`, `nodes`, `pipes`,
необязательный `valves`. Один приёмник фиксированного давления, без разделения
потоков, нулевых/обратных расходов и нисходящих труб.

Узел: `name`, `elevation_m` (0), необязательное `pressure_pa` только у приёмника.
Труба: `name`, `start`, `end`, `length_m`, `diameter_m`, `roughness_m` (0),
`loss_coefficient` (0), `valve_opening` (1). K/opening² распределяется по длине.
Изменение открытия требует K>0. Нулевое открытие не поддержано.

Локальный `valves`: `name`, `start`, `end`, `diameter_m`, `loss_coefficient`,
`vapor_pressure_pa`, `pressure_recovery_factor`, `valve_opening` (1).
Порты на одной отметке; K и давление пара положительны, F_L и opening в (0,1].
Клапан допускает только жидкость; проверка распространяется на сжатое сечение.

`pressure_control`: `valve`, `node`, `target_pressure_pa`, `min_opening` (0.2),
`max_opening` (1), `pressure_tolerance_pa` (1e-4), `max_evaluations` (50).
В конфигурации `GatheringFacility` он задаётся рядом с `gathering`;
`feed_nodes` связывает каждый поток площадки с узлом дерева.
Ручное управление ветвью регулятора отклоняется.

## Расписание площадки

Файл содержит `periods` и необязательный `capacity_control`.
Период: `name`, `duration_s`, `feed_scales`, `pump_speed_ratios`, `valve_openings`,
`gas_header_pressure_pa`, `gas_cooling_temperature_k`, `shutdown`,
`unavailable_pumps`, `station_target_pressure_pa`, `gathering_valve_openings`,
`gathering_target_pressure_pa`, `gas_valve_max_opening`, `gas_valve_outlet_pressure_pa`.

Команды сохраняются до следующего изменения. Отсутствующий `unavailable_pumps`
сохраняет недоступность, пустой список снимает её. На остановке имена и типы
команд всё равно проверяются. Checkpoint привязан к модели и расписанию;
смена исходников/физики может сделать продолжение несовместимым.

```bash
python scripts/run_surface_schedule.py configs/surface_gas_valve_facility.json configs/surface_gas_valve_schedule.json --output /tmp/schedule.json
python scripts/run_surface_schedule.py configs/surface_gas_valve_facility.json configs/surface_gas_valve_schedule.json --stop-after 3 --output /tmp/partial.json
python scripts/run_surface_schedule.py configs/surface_gas_valve_facility.json configs/surface_gas_valve_schedule.json --checkpoint /tmp/partial.json --output /tmp/resumed.json
```

Для `GatheringFacility` добавьте `--gathering`.

## Связь с FIM

`simulation.physical_surface_network` задаёт `network` и `well_nodes`.
Водяной адаптер принимает `controls` и `pressure_control` скорости насоса;
black-oil адаптер — `controls.valve_openings` и регулятор открытия ветви.
Black-oil `network_backend`: `cpu_reference` (по умолчанию) или `torch_device`.
PVT и плотности дерева должны совпадать с пластом, отметки узлов — с опорой BHP.
Поддержка распределённых перфораций и фаз зависит от адаптера; произвольный
скважинный сценарий нельзя подключить без проверки области модели.

### Ресурсы площадки → фактическая добыча

`simulation.coupled_facility`: `facility` (конфигурация выше), `control`:
`min_fraction` (0.001), `fraction_tolerance` (1e-4), `scan_points` (9),
`max_evaluations` (60). Допуск множителя абсолютный. Приёмник дерева и первая
ступень сепарации совпадают по давлению/PVT/PVTO и стандартным плотностям.
Шаблонные `facility.feeds` заменяются фактическими массами скважин.

Контур меняет общий множитель верхних пределов дебитов и повторно решает
пластовый подшаг. Операторские уставки сохраняются, применённые записаны отдельно.
Положительные добывающие скважины без ramp; ограничения существующей физической
black-oil связи сохраняются. Полное управление нагнетательной инфраструктурой
и динамика запасов не реализованы. Число полных повторов видно в отчёте.

```bash
python scripts/run_fim_blackoil_surface.py configs/fim_blackoil_facility_coupled.json --days 5 --report-days 1 --outlet-mpa 1.0 --device cpu --output /tmp/coupled.json
```

Для CUDA: `--device cuda --network-backend torch_device`.
Диагностический runner принимает `--initial-rs` (м³/м³) для начального состояния
без свободного газа; Rs не должен превышать насыщенное значение при начальном
давлении. Локальный жидкостный пример требует `--initial-rs 25` и допустимого
диапазона выходного давления; это указано его командой запуска в конфигурационных примерах.

## Тепловой расчёт чистого флюида

Отдельный файл содержит `fluid` с параметрами `PengRobinsonFluid` и `operations`.
Поля флюида: `name`, `critical_temperature_k`, `critical_pressure_pa`,
`acentric_factor`, `molar_mass_kg_mol`, `ideal_cp_j_mol_k`,
`min_temperature_k`, `max_temperature_k`, необязательные
`reference_temperature_k` (298.15) и `reference_pressure_pa` (100000).
Это явные данные EOS/калорики, а не PVT black-oil.

Операции: `state`, `saturation_temperature`, `saturation_pressure`, `ph`, `ps`,
`letdown`, `nozzle`. В последних двух блок `equipment` задаёт конструкцию;
параметры режима передаются рядом. Пример: `configs/surface_thermal_propane.json`.
Описание уравнений и ограничений: [термодинамика](THERMODYNAMICS.md).


## Тепловой расчёт смеси

В `fluid` вместо полей одного вещества задаются `components` (массив таких же
полных описаний компонентов) и `binary_interactions` (полная симметричная матрица kij).
Пример: `configs/surface_thermal_mixture.json`.

Операции `state`, `stability`, `pt` принимают `pressure_pa`, `temperature_k`,
`mole_fractions`. `ph` принимает `pressure_pa`, `molar_enthalpy_j_mol`,
`mole_fractions`; `ps` — `pressure_pa`, `molar_entropy_j_mol_k`, `mole_fractions`.
Сумма положительных молярных долей должна равняться единице.

Для `letdown`: `equipment` содержит `name` и `outlet_pressure_pa`; режим —
`inlet_pressure_pa`, `inlet_temperature_k`, `mass_rate_kg_s`, `mole_fractions`.
Для `nozzle` `equipment` содержит `name`, `throat_area_m2`,
`minimum_pressure_pa`, необязательные `discharge_coefficient` (1),
`scan_points` (33, целое не меньше 9), `pressure_tolerance_pa` (0.01).
Режим: `inlet_pressure_pa`, `inlet_temperature_k`, `back_pressure_pa`,
`mole_fractions`, необязательное `opening` (1, от 0 до 1).
Пример: `configs/surface_thermal_mixture_nozzle.json`. Уравнения, соглашения по единицам
и ограничения находятся в [термодинамике](THERMODYNAMICS.md).


## Компонентная тепловая схема

Конфигурация `configs/surface_thermal_process.json` содержит `fluid`,
`units`, `periods` и необязательный `resource_limits`. `fluid` — явные `components` и `binary_interactions`, как выше.

Каждый аппарат: `name`, `type`, `inputs` (список имён), `outputs` (список имён),
`parameters` (объект). Типы и параметры:

- `mix`: несколько входов, один выход, пустые `parameters`; давления равны.
- `split`: один вход, два выхода; `fraction` от 0 до 1.
- `letdown`: один вход/выход; `pressure_pa`.
- `heat_exchanger`: один вход/выход; `outlet_temperature_k`,
  `utility_temperature_k`, необязательные `pressure_drop_pa` (0), `max_abs_duty_w`.
- `pressure_raiser`: один вход/выход; `outlet_pressure_pa`, `isentropic_efficiency`,
  необязательные `max_power_w`, `max_outlet_temperature_k`.
- `separator`: один вход, два выхода; `pressure_pa`, необязательный
  `single_phase_outlet` (0 или 1, без значения однофазная сепарация отклоняется).

Имя аппарата не дублируется в `parameters`. Один поток расходуется один раз;
для нескольких потребителей нужен `split`. Непотреблённые потоки — продукты.

Период содержит `duration_s` и `feeds`: отображение имени подачи в
`pressure_pa`, `temperature_k`, `molar_rate_mol_s`, `mole_fractions`.
Нулевая подача допустима. Необязательный `unit_controls` периода — отображение
имени аппарата в переопределяемые поля его `parameters`; команды применяются
к базовому `units` независимо для каждого периода. Подключения потоков менять нельзя.

`resource_limits` корня или периода содержит `max_shaft_power_w`, `max_heating_w`,
`max_cooling_w`: неотрицательные пределы в Вт, null/пропуск — без ограничения.
Блок периода полностью заменяет корневой для этого периода. Нагрев и охлаждение
учитываются раздельно. При наличии блока контроллер выбирает общий множитель
подач с учётом этих пределов и абсолютных лимитов аппаратов, затем пересчитывает
физическую схему. Пример: `configs/surface_thermal_process_controlled.json`.

```bash
python scripts/run_surface_process.py configs/surface_thermal_process.json --output /tmp/process.json
```

`--stop-after N` выполняет не больше N очередных периодов, `--checkpoint PATH`
продолжает проверенный JSON. Схема/расписание/модель должны совпадать с сохранением.
Уравнения и точные границы применимости: [тепловая подготовка](THERMAL_PROCESS.md).


## Динамический сосуд PR-смеси

`configs/surface_dynamic_vessel.json` запускается через `scripts/run_dynamic_vessel.py`.
Поля, единицы, фазовые порты, ограничения шага и перезапуск описаны
в [руководстве динамического сосуда](DYNAMIC_VESSEL.md).
Команды постоянны внутри периода; их интегрирование сохраняет компонентные
запасы и внутреннюю энергию, в отличие от накопления продукции steady-схемы.


## Монолитная динамическая сеть

Пример `configs/surface_vessel_network.json`, команда `scripts/run_vessel_network.py`.
Отдельные `vessels`, `connections` и `initial` задают геометрию, проводимости
и исходные состояния; `periods` — внешние потоки, тепло и `connection_factors`.
Полный формат и ограничения линейной связи: [динамическая сеть](DYNAMIC_NETWORK.md).


Для нелинейной однофазной связи задаётся `model: "pr_single_phase_nozzle"`,
`throat_area_m2`, `discharge_coefficient` (1), `minimum_pressure_ratio` (0.4),
`pressure_tolerance_pa` (0.001) и обязательные предельные перепад/молярный расход.
Общие поля соединения и порты сохраняются; `connection_factors` масштабирует
площадь. Пример: `configs/surface_vessel_network_pr_nozzle.json`.
Старый формат без `model` остаётся линейным; область применимости —
в [описании динамической сети](DYNAMIC_NETWORK.md).


Для двухфазной связи задаётся `model: "pr_hem_nozzle"` с `throat_area_m2`,
`minimum_pressure_pa`, предельными перепадом/молярным расходом и
необязательными `discharge_coefficient` (1), `scan_points` (9),
`pressure_tolerance_pa` (0.01). Порты, reverse и масштаб площади общие.
Пример: `configs/surface_vessel_network_hem.json`; область и численный
якобиан описаны в [HEM-связи](DYNAMIC_NETWORK.md).


## Событийные защиты сети

Корневые `protections` (массив `NetworkTrip`) и `event_time_tolerance_s`
(1e-7 s) необязательны. Периодное `reset_trips` сбрасывает перечисленные
защёлки; при сохраняющемся нарушении они срабатывают снова.
Метрики, единицы численного допуска, действия и приоритеты команд:
[события и защиты](NETWORK_PROTECTIONS.md).
Пример: `configs/surface_vessel_network_protected.json`.

## Стенки динамических аппаратов

Корневой `walls` и начальный `wall_temperature_k` добавляют конечную
теплоёмкость стенок; `ambient_temperatures` задаёт окружение по периодам.
Параметры и воспроизводимый пример: [тепловая инерция](VESSEL_WALL.md).

## Приводы динамических соединений

Корневые `actuators` и `actuator_positions` задают характеристики и начальные
положения. Периодические `connection_factors` становятся командами приводов,
`actuator_modes` задаёт normal/fail_safe/stuck. Параметры, приоритеты защит
и сохранение: [руководство](CONNECTION_ACTUATORS.md).

## Труба с распределёнными запасами и уровень сепаратора

Отдельный CLI `run_transient_pipe.py`: корневые `pipe`, `initial`, `periods`,
опциональный `stepper`; см. [уравнения и единицы](TRANSIENT_PIPE.md).
В PR-сети корневой `geometries` задаёт цилиндры, `withdrawals.phase_0`
принимает объект ограниченного регулятора. Метрика защиты `liquid_level_m`
использует высоту жидкости; см. [геометрический уровень](SEPARATOR_LEVEL.md).

В CLI трубы `model: thermal_homogeneous` выбирает [полную энергию](THERMAL_PIPE.md).
Нужны C0=1, Vd=0, горизонтальная геометрия; температуры открытых границ
задаются явно. `surface_thermal_pipe.json` содержит полный пример.

### Численная диссипация тепловой трубы

Для `ThermalHomogeneousPipe`: `numerical_flux` принимает `entropy_stable`
(по умолчанию) или `rusanov`. `adaptive_entropy_dissipation` — строго boolean,
по умолчанию true: локальный датчик вычисляется из принятого состояния
и фиксируется во время нелинейного решения подшага.
`thermal_entropy_dissipation` — конечное неотрицательное число,
по умолчанию 1.4. Это численный коэффициент, не свойство реального флюида.
Для воспроизведения прежнего постоянного энтропийного потока нужны
`adaptive_entropy_dissipation=false` и `thermal_entropy_dissipation=0.75`.
Подробности и область проверки: [численный метод](THERMAL_NUMERICS.md).

### Вращающийся привод

`configs/surface_rotating_drive.json` задаёт `drive`, `initial`, `periods`.
Основные параметры: `inertia_kg_m2`, `max_speed_rad_s`, `max_motor_torque_nm`,
`max_braking_torque_nm`. Необязательные: `viscous_drag_nm_s` (0),
`motor_efficiency` (1), `regeneration_efficiency` (1),
`max_electrical_draw_w`/`max_electrical_export_w` (null — предел не задан).
Каждый период содержит `duration_s`, `motor_torque_nm` и необязательный
`load_torque_nm` (0). Моменты постоянны внутри периода. Начальное состояние
задаётся `angular_velocity_rad_s` и необязательным `time_s` (оба 0).
Физика, работа ограничений и restart: [руководство привода](ROTATING_DRIVE.md).

### Динамический насос с сетью и PID

`configs/surface_dynamic_pump.json` содержит `system`, `initial`, `periods`.
В `system`: существующий `network` формата LiquidNetwork, `drive` формата
RotatingDrive, `nominal_speed_rad_s`, необязательный `controller` (null без
авторегулирования), `measurement` (pressure/flow/speed), `measurement_node`,
`control_interval_s` (0.1), `maximum_dt_s` (0.02), `minimum_dt_s` (1e-7),
`max_speed_change_rad_s` (2), `no_flow_power_w_at_nominal` (0).

PID: `proportional_gain`, `integral_gain`, `derivative_gain`, `min_output`,
`max_output`; необязательные `derivative_filter_s` (0.1), `tracking_time_s`
(0.5), `bias` (0), `direction` (+1 или −1). Выход — момент в N m,
единицы обратной связи задаются `measurement`.

Период: `duration_s`; необязательные `mode` (auto/manual/off/trip),
`setpoint`, `motor_torque_nm`, `network_options`, `reset_trip` (false),
`max_substeps` (100000). Auto требует контроллер. Пределы насосных карт
не расширяются при запуске. Ограничения привода меняют фактический момент
с сохранением учёта энергии; ошибка регулирования остаётся явной.
Поведение команд, перезапуска и область применимости:
[насос, сеть и PID](DYNAMIC_PUMP.md).

### Компрессор с коллектором и рециркуляцией

`configs/surface_compressor_train.json`: поля `system`, `initial`, `periods`.
В `system`: `compressor` (map, gas, mechanical_efficiency,
max_discharge_temperature_k, max_shaft_power_w), `drive`, `recycle_actuator`
(fail_position=1), `anti_surge_controller` (выход [0,1]), необязательный
`speed_controller`, объём коллектора, давления/температура резервуаров,
площади экспортного/рециркуляционного сопел.

Карта: `corrected_speed_rad_s`, `beta` (0 surge, 1 choke), матрицы
`corrected_mass_flow_kg_s`, `pressure_ratio`, `isentropic_efficiency`;
`reference_temperature_k` (300), `reference_pressure_pa` (1e5).
Матрицы имеют размер скорость × beta. Входные условия относятся к одному
и тому же идеальному газу, скорость задаётся в rad/s, расход в kg/s.

Основные настройки: `target_surge_margin` (0.25), `fast_open_margin` (0.1),
`control_interval_s` (0.05), `maximum_dt_s` (0.01), `minimum_dt_s` (1e-7),
`heat_transfer_w_k` (0), `ambient_temperature_k` (300),
`min_temperature_k` (150), `max_temperature_k` (700), `max_pressure_pa` (1e7),
необязательные пределы `max_recycle_cooling_w`/`max_recycle_heating_w` (null).
Точность состояния: `max_fractional_mass_change` (0.05),
`max_temperature_change_k` (5), `max_speed_change_rad_s` (2).

Период: `duration_s`, `mode` (run/standby/off/trip),
`speed_control_mode` (auto/manual), `speed_setpoint_rad_s`, `motor_torque_nm`,
`export_opening` (1), `anti_surge_mode` (auto/manual),
`manual_recycle_opening` (1), `recycle_actuator_mode` (normal/fail_safe/stuck),
`reset_trip` (false), `max_substeps` (100000). Run ниже диапазона карты
не заменяется экстраполированной характеристикой: запуск требует standby.
Схема нагнетательного trim-клапана, знаки потоков/тепла и ограничения:
[руководство](COMPRESSOR_TRAIN.md).

### Динамическая параллельная насосная станция

`configs/surface_dynamic_station.json`: поля `system`, `initial`, `periods`.
`system.network` — существующая `LiquidNetwork` с 1–8 открытыми
параллельными насосами с обратными клапанами и границами давления.
`drives` — словарь имя насоса → параметры `RotatingDrive`;
`nominal_speeds_rad_s` — имя → номинальная скорость.
`speed_controllers` — имя → PID или null, `pressure_controller` — PID
или null, `pressure_node` — узел обратной связи. Внешний PID требует
локального PID каждого агрегата; его выход — общая относительная скорость
с нижней границей ноль. Выход локального PID — момент в N m.

`dispatch_policy` — параметры существующего `PumpStationControl` или null;
имена насосов и узел должны соответствовать станции. Ограничения количества,
диапазон общей скорости, NPSH/расход и допустимое число конфигураций
относятся к выбору стационарного кандидата. В динамическом блоке его
стоимость пересчитывается в электрическое потребление с потерями приводов.

Настройки: `control_interval_s` (0.2), `maximum_dt_s` (0.05),
`minimum_dt_s` (1e-7), `max_speed_change_rad_s` (2),
`minimum_stage_dwell_s` (5), `dispatch_power_deadband_w` (10),
`max_station_electrical_draw_w`/`max_station_electrical_export_w`
(null без общего ограничения; ноль допустим),
`no_flow_power_w_at_nominal` (словарь, пропущенные агрегаты имеют P0=0 W).

`initial`: необязательные `speeds_rad_s` (словарь, по умолчанию нули),
`time_s` (0), `unavailable` (пустой список), `network_options`.
Период: `duration_s`, `mode` (auto/manual/torque/off/trip, default auto),
`pressure_setpoint_pa` (по умолчанию цель политики), `speed_ratio` (1),
`running_pumps` (список), `motor_torques_nm` (словарь), `unavailable`,
`network_options`, `reset_trip` (false), `redispatch` (false),
`max_substeps` (100000). В torque при пропущенном running_pumps используются
ключи motor_torques_nm; недоступные агрегаты не включаются.
Manual задаёт скорость, torque — моменты. Network options разрешает только
`boundary_pressures_pa` и `valve_openings`.

Пропущенный `unavailable` сохраняет предыдущую недоступность; пустой список
возвращает доступность. Отключённый двигатель не обнуляет скорость вала
или расход выбегающего насоса. Trip имеет защёлку. На рестарте полный
предшествующий журнал воспроизводится один раз с проверкой исходников,
физики, PID и диспетчеризации. Подробнее: [модель станции](DYNAMIC_STATION.md).
