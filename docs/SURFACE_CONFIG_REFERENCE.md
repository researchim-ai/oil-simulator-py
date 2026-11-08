# Справочник наземных конфигураций

Этот справочник описывает текущий API; исторические результаты прогонов
хранятся отдельно. Области применимости: [наземное моделирование](SURFACE_MODELING.md).
Все давления абсолютные, температура в K, массовые расходы в кг/с,
длины в м, мощность в Вт. В PVT ось давления задаётся в МПа, вязкость — в сП.

## Общая компонентно-энергетическая PR-модель

`EquilibriumFluid` / `EquilibriumComponent`, `ComponentEnergyStream`,
`EquilibriumInventory`, `StockComponentAssay`. Все свойства в SI;
молярные расходы — моль/с, потоки полной энергии — Вт.
Порядок компонентов задаётся именованным реестром, а не тремя фазами.

`configs/surface_equilibrium_process.json` и `scripts/run_equilibrium_process.py`:
строгие корневые поля `fluid`, `stock_mass_fractions`, `wells`,
`thermal_conditions`, `transfers`, `mixer`. `fluid` содержит компоненты,
явную матрицу `binary_interactions`, барические пределы и `max_phases`.
Три строки `stock_mass_fractions` относятся к stock-нефти, воде и газу;
каждая содержит массовые доли компонентов и суммируется в единицу.

`wells` содержит принятые `component_mass_rates_kg_s` и `bhp_pa`.
Для каждой скважины в `thermal_conditions` нужны `elevation_m` и ровно
одно условие: `temperature_k` или `total_energy_rate_w`; опциональны
`velocity_m_s` и, для энергетического условия, `temperature_guess_k`.
Это явная тепловая граница; баланс энергии изотермического FIM не добавлен.

`transfers` задаёт конечное давление, высоту/скорость и при необходимости
теплоту и мощность; ненулевая теплота требует `utility_temperature_k`.
`mixer` содержит давление, опциональные высоту, скорость и численную
температурную догадку. `--device cpu/cuda`, `--output PATH`.
Выход публикуется атомарно после проверки состава, энергии, энтропии
и неизменности исходников во время расчёта.

Подробнее: [уравнения, интерфейсы и ограничения](EQUILIBRIUM_THERMODYNAMICS.md).
PR-параметры примера не откалиброваны под реальные PVT Norne или рассол.

## Динамическая трёхфазная сепарация

Отдельный Torch API: `ThreePhaseSeparatorNetwork`,
`configs/surface_three_phase_separator.json` и
`scripts/run_three_phase_separator.py`. Конфигурация содержит строго
`system`, `initial`, `periods`. Фазы в массивах: нефть, вода, газ;
`phase_mass_rates_kg_s` — фактические фазовые, а не stock-расходы.
`system` задаёт свойства, аппараты, патрубки, шаг и допуски;
`initial` — начальные давления, температуры и объёмы обеих жидкостей.
Периоды задают длительность, входы, открытия, тепловые нагрузки и
температуры утилит. Неизвестные поля отвергаются.

`--device cpu/cuda`, `--stop-after N`, `--checkpoint PATH`;
перезапуск привязан к исходникам, окружению, устройству и конфигурации.
Подробные уравнения и ограничения: [трёхфазный сепаратор](THREE_PHASE_SEPARATOR.md).
Модель несмешивающихся фаз и идеального газа не заменяет PR/black-oil flash.

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

### Слабосжимаемая жидкостная сеть с валами

`configs/surface_liquid_transient.json`: `system`, `initial`, `periods`.
`system.network` — формат LiquidNetwork; используются геометрия,
карты и вязкость, но уравнения динамики решает `LiquidTransientNetwork`.
Все высоты и узловые постоянные источники равны нулю. Трубы открыты,
без встроенных обратных клапанов; закрытие/невозвратный режим задаются
отдельными `ThrottleValve`. Насосы параллельные с обратными клапанами.

`storage_volumes_m3`: имя каждого свободного узла → полный объём (m³),
строго положительный. `bulk_modulus_pa`: эффективный K (Pa).
`reference_pressure_pa` (1e6) задаёт плотность и нуль акустической энергии.
`drives` и `nominal_speeds_rad_s` соответствуют всем насосам;
`no_flow_power_w_at_nominal` задаёт их P0 (W).
Общие `max_station_electrical_draw_w`/`max_station_electrical_export_w`
по умолчанию null, ноль допустим. Без насосов словари приводов пустые.

Шаг: `maximum_dt_s` (0.005), `minimum_dt_s` (1e-8),
`acoustic_courant` (0.5, максимум 1), `max_speed_change_rad_s` (2),
`max_pressure_change_pa` (2e5). Область: `max_pressure_pa` (1e8),
`max_compression_fraction` (0.01, максимум 0.05); давление выше vapor_pressure.
Сжимаемость линейная, модель не учитывает кавитацию.

Временная точность: `temporal_error_control` (true),
`relative_tolerance` (1e-5), `pressure_tolerance_pa` (1),
`flow_tolerance_m3_s` (1e-9), `speed_tolerance_rad_s` (1e-5).
Полный и два половинных шага оценивают локальную ошибку; принимаются
консервативные половинные шаги. Эти допуски управляют точностью траектории,
а не заменяют критерий нелинейной сходимости или баланс энергии.

`initial`: `pressures_pa` свободных узлов (по умолчанию p_ref),
`pipe_volume_rates_m3_s` (по умолчанию 0), `speeds_rad_s` (0),
`time_s` (0), `unavailable` (пустой список). Запасы вычисляются из давлений
по линейному закону. Потоки насосов/клапанов алгебраические, не сохраняются
как независимые инерционные состояния.

Период: `duration_s`; `mode` (torque/off/trip, default torque),
`motor_torques_nm` (словарь), `boundary_pressures_pa` (только уже фиксированные
узлы), `valve_openings` ([0,1]), `unavailable`, `reset_trip` (false),
`max_substeps` (100000). Недоступность и trip сохраняются, открытия/границы
при пропуске возвращаются к значениям модели. Низкоуровневый API не включает
PID; отдельный контур приведён ниже. Формулы, баланс механической энергии и restart:
[жидкостная динамика](LIQUID_TRANSIENT.md).

### PID и диспетчеризация динамической жидкостной сети

`configs/surface_controlled_liquid.json`: `system`, `initial`, `periods`.
`system.physics` — полная конфигурация `LiquidTransientNetwork` из предыдущего
раздела. `pressure_node` — свободный узел с конечным объёмом;
`pressure_controller` — PID с min_output=0, выходом является относительная
скорость. `speed_controllers` — имя каждого насоса → PID с выходом в N m;
границы должны укладываться в моторный/тормозной моменты его привода.
`dispatch_policy` — PumpStationControl или null (доступные/прежние агрегаты
без оптимизации). Целевая вершина и набор насосов совпадают с моделью.
Диапазон внешнего PID покрывает диапазон скоростей планировщика.

`control_interval_s` (0.05), `minimum_stage_dwell_s` (0.5),
`dispatch_power_deadband_w` (10). Общие электрические и физические
ограничения задаются внутри physics, не дублируются контроллером.
Номер и начало цифрового такта сохраняются через границы отчётов.

JSON `initial`: необязательный `physical_initial` (поля инициализации
LiquidTransientNetwork), `running_pumps` (пустой список),
`applied_torques_nm` (словарь для уже выбранных агрегатов),
`pressure_setpoint_pa` (для согласованной первоначальной памяти PID).
В Python также можно передать готовый LiquidTransientState через physical;
это создание нового контроллера, не замена полного restart.

Период: `duration_s`, `mode` (auto/manual/torque/off/trip, default auto),
`pressure_setpoint_pa` (default цель политики), `speed_ratio` (1),
`running_pumps`, `motor_torques_nm`, `unavailable`, `boundary_pressures_pa`,
`valve_openings`, `reset_trip` (false), `redispatch` (false),
`max_substeps` (100000), `max_control_samples` (100000).

Auto использует планировщик; manual задаёт общую скорость выбранных насосов;
torque задаёт моменты. Уставки/ручные моменты считываются на такте, а
off/trip и недоступность снимают момент немедленно. Резерв выбирается
на ближайшем такте; положительный момент может появиться на следующем
из-за перехода PID без скачка. Непрочитанная уставка короткого периода не становится
постоянной настройкой. Явный redispatch сохраняется до допустимого отсчёта;
unavailable и защёлка trip сохраняются до явной команды изменения/сброса.

Состояние включает physical, оба уровня PID, held_torques_nm,
torque_integrals_nm_s, feedback_duration_s, selected, clock_origin_s,
next_tick, last_switch_time_s, dispatch_signature/status, pending_redispatch
и model_id. Полное расписание с source fingerprint воспроизводится
при restart, в том числе если checkpoint сделан внутри цифрового окна.
Семантика часов, статусов и замкнутого выхода:
[контур динамической сети](CONTROLLED_LIQUID.md).

### Предохранительный сброс и переходные защиты

В `LiquidTransientNetwork` добавлено необязательное `relief_valves`:
словарь имя существующей ThrottleValve → `cracking_pressure_pa`,
`full_open_pressure_pa`. Это перепады между концами ветви (Pa),
0 ≤ cracking < full_open. Носитель требует open=true,
check_valve=true, opening=0; диаметр и Cd задают конечную площадь.
Динамика определяет открытие сама; обычный valve_openings для такой
ветви отклоняется. При отсутствии relief_valves действуют прежние уравнения.

`configs/surface_protected_liquid.json`: `system`, `initial`, `periods`.
В system: `controller` формата ControlledLiquidStation, `rules`
(непустой список), `protection_interval_s` (0.01).

Правило LiquidTrip:

- name, equipment — уникальное имя правила и имя узла/насоса;
- metric — pressure_pa / pump_flow_m3_s / npsh_margin_m;
- direction — above или below, threshold — положительный порог в единицах метрики;
- stop_pumps — список отключаемых приводов, trip_station — глобальный trip;
  хотя бы одно действие обязательно;
- delay_s (0), startup_grace_s (0), min_speed_ratio (0.6), reset_margin (0).

Давление наблюдается всегда, насосные метрики — при выбранном разрешённом
агрегате с достаточной скоростью. Для NPSH нужны inlet_diameter_m и
npsh_required_m/npsh_curve. Верхний/нижний порог давления должен находиться
строго внутри жёсткой области модели. Пусковая выдержка предшествует
накоплению delay_s; при исчезновении нарушения выдержка сбрасывается.

JSON initial содержит `controller_initial` с полями инициализации
ControlledLiquidStation. Периоды поддерживают его управляющие команды
и дополнительно `reset_trips` (список имён, default пустой),
`max_protection_samples` (100000). max_substeps и max_control_samples
относятся ко всему отчёту. Redispatch сохраняется через внутренние окна,
а не повторно посылается на каждом отсчёте защиты.

reset_trip снимает операторскую защёлку; reset_trips снимает только
названные защиты при безопасном сигнале или невооружённом насосе.
Внешняя unavailable хранится независимо и не исчезает при таком сбросе.
Контроллер, обе шкалы времени, таймеры и источники запретов входят в
source-bound checkpoint с полным replay. Формулы и ограничения:
[сброс давления и защиты](LIQUID_PROTECTION.md).


## Регулируемая рециркуляция однофазной жидкости

`LiquidTransientNetwork.actuated_valves` — mapping имён обычных открытых
ThrottleValve в параметры ConnectionActuator: opening_rate_per_s,
closing_rate_per_s, fail_position, time_constant_s (0), max_position_change (0.1).
Совмещение с relief_valves запрещено. initialize.actuator_positions задаёт
начальное фактическое положение; default — opening исходного клапана.
В периоде valve_openings становится целевым положением приводного клапана;
actuator_modes — normal/fail_safe/stuck. Неприводные клапаны управляются как раньше.

`ControlledLiquidStation.planning_valve_openings` (default {}) задаёт
номинальные открытия только для стационарной диспетчеризации.
Фактическая динамика продолжает использовать команды и запасы приводов.

`RecyclingLiquidStation`: protected (ProtectedLiquidStation), loops,
control_interval_s (0.05). Каждый loop: name, valve, pumps, controller
(PIDController с положительным направлением и выходом [0,1]),
minimum_flow_m3_s, min_speed_ratio (0.6), startup_opening (1), inactive_opening (0).
Насосы и клапаны между контурами не пересекаются. Возвратная ветвь с обратным
клапаном непосредственно соединяет общее нагнетание и всасывание своих насосов;
для неё обязательно planning_valve_openings[valve]=0.

Initial: protected_initial с параметрами initialize защищённой станции.
Периоды: её команды плюс recycle_mode (auto/manual/off), recycle_openings,
actuator_modes, max_recycle_samples (100000). Manual требует всех клапанов
контуров в recycle_openings. Прямой valve_openings для них запрещён.
Управление сторонними клапанами остаётся в valve_openings.

Состояние содержит protected, pids, targets, clock_origin_s, next_tick, model_id.
В отчёте recycle_samples содержит фактическую позицию, команду, вооружённые
насосы, измеренный минимум расхода и дефицит; recycled_volume_m3 интегрируется
по физическим подшагам отдельно от экспортного потока.
[Формулы, ограничения и команды запуска](LIQUID_RECYCLE.md).


## Тепловые запасы жидкостной рециркуляции

`ThermalLiquidStation`: recycling (RecyclingLiquidStation), heat,
temperature_trips (default []), thermostats ([]), control_interval_s (0.05 с),
max_heating_w/max_cooling_w (None). Общие пределы относятся к мощности
контролируемых тепловых утилит, не к электрическому потреблению холодильников.

`heat` использует ту же physics, что recycling; в JSON станции physics
повторно не указывается. Обязательные параметры:

- specific_heat_j_kg_k — постоянная положительная удельная теплоёмкость;
- boundary_temperatures_k — заданные температуры бесконечных границ;
- boundary_reservoirs_kg — начальные массы конечных граничных запасов ({}).

Каждая граница давления должна присутствовать ровно в одном из двух mapping.
Масса конечного резервуара изменяется по фактическому переносу, подпитки нет.

Остальные параметры heat:

- pipe_cells — mapping имени трубы в положительное целое N, default один;
  имена полученных ячеек: pipe_name::0 ... pipe_name::(N-1),
  совпадения с именами гидравлических узлов запрещены;
- walls — mapping конечного жидкостного запаса в параметры VesselWall;
- reference_temperature_k (298.15), min_temperature_k (250), max_temperature_k (450);
- pump_heat_to_fluid_fraction (1), shaft_heat_to_fluid_fraction (0),
  motor_heat_to_fluid_fraction (0), каждая доля в [0,1];
- maximum_dt_s (1), minimum_dt_s (1e-10), positivity_courant (0.8, <=1);
- maximum_temperature_change_k (1), temporal_error_control (true),
  temperature_tolerance_k (1e-5), relative_tolerance (1e-7);
- store_step_temperatures (true): false сохраняет конечный запас и аудиты,
  но исключает полные промежуточные поля из журнала.

`LiquidTemperatureTrip`: name, location, threshold_k; direction above/below
(default above), metric fluid_temperature_k/wall_temperature_k
(default fluid_temperature_k), delay_s (0), reset_margin_k (0),
stop_pumps ([]), trip_station (false). Требуется хотя бы одно действие.
Порог должен находиться строго внутри температурной области датчика.

`LiquidThermostat`: name, location, setpoint_k, controller (PIDController).
Положительное направление PID, выход — тепловая мощность в Вт; отрицательная
означает охлаждение. Два термостата не могут управлять одним жидкостным запасом.

Initial: recycling_initial с параметрами инициализации RecyclingLiquidStation,
temperatures_k и wall_temperatures_k по именам запасов/стенок. По умолчанию
жидкость имеет reference_temperature_k, стенка — начальную температуру жидкости.

Периоды поддерживают команды recycling и дополнительно:

- reset_temperature_trips — список имён, сброс только при безопасной температуре;
- boundary_temperatures_k — команды только бесконечных границ;
- ambient_temperatures_k — температуры окружения по именам стенок;
- heat_duties_w — явно внешние источники/отборы в конечных запасах, не занятых термостатами;
- temperature_setpoints_k — уставки по именам термостатов;
- temperature_control_modes — auto/manual/off по именам термостатов;
- temperature_manual_duties_w — ручные мощности в пределах локального PID;
- max_thermal_samples (100000), max_thermal_substeps (100000).

Все бюджеты нижних регуляторов и физических подшагов действуют на весь
отчётный интервал. Температурные PID/защиты сохраняют свой цифровой такт между
отчётами. Снятие температурной защёлки не снимает операторский trip, внешнюю
unavailable или защиту давления/расхода/NPSH.

Результаты: thermal_steps с локальными энергетическими аудитами,
thermal_audits с совместным механическим/тепловым балансом,
temperature_samples, temperature_events, latched_temperature_trips,
thermal_properties. Состояние хранит отдельные источники запретов,
температуры, тепловые массы, PID и часы.

Температурные зависимости плотности/вязкости и фазовая область здесь
не рассчитываются. [Уравнения, команды запуска и ограничения](THERMAL_LIQUID.md).

## Устройство жидкостной динамики

В `LiquidTransientNetwork` доступны `backend` (`torch` по умолчанию или
`scipy_reference`) и `device` (`cpu` по умолчанию, либо `cuda:N`).
`LiquidHeatNetwork` наследует их от физики; явные значения можно задать
в блоке `heat`. SciPy-эталон поддерживает только CPU.
CLI жидкостных расписаний принимает `--backend` и `--device`. Недоступная
CUDA вызывает ошибку. Модель/checkpoint привязаны к устройству и версиям
численных исходников. Подробнее: [PyTorch](PYTORCH_SURFACE.md).

### Передача теплового состояния

Период `ThermalLiquidStation` принимает `thermal_state_transfer`:
`report` (по умолчанию), `window` или `record`. Этот параметр меняет
частоту сериализации текущих температур/масс, сохраняя физику, такт
регуляторов и формат результата. CLI `run_thermal_liquid.py` может
переопределить его для всех периодов через `--thermal-state-transfer`.
Полный журнал температур управляется отдельно через
`heat.store_step_temperatures`. Подробнее: [резидентное состояние](PYTORCH_SURFACE.md).

### Передача механического состояния

Периоды жидкостных расписаний принимают `hydraulic_state_transfer`:
`interval` (по умолчанию для Torch) либо `solve` (путь для парной проверки).
Давления/расходы/скорости удерживаются на устройстве в режиме interval;
изменение внешнего публичного состояния инвалидирует повторное использование.
Команды/граничные значения и арматура продолжают обновляться по расписанию.
CLI жидкостных блоков поддерживает `--hydraulic-state-transfer`.
При продолжении используйте ту же последовательность параметров периодов.
См. [численное исполнение](PYTORCH_SURFACE.md).

### Прямая связь гидравлики с теплом

Период `ThermalLiquidStation` принимает `hydraulic_heat_transfer`:

- `auto` (по умолчанию): прямой путь для `thermal_state_transfer=report`,
  `hydraulic_state_transfer=interval` и обоих Torch-ядер на одном устройстве;
  иначе используется прежнее CPU-представление записей.
- `device`: требует этих условий; несовместимое исполнение вызывает ошибку.
- `host`: прежний путь для парного сравнения.

CLI: `run_thermal_liquid.py --hydraulic-heat-transfer device`.
В прямом режиме принятые давления, расходы, поагрегатное тепло и переносы
массы передаются тензорами; CPU-журнал и скалярные проверки сохраняются.
Тепло рассчитывается потоково только по принятым физическим шагам.
Параметр входит в привязку checkpoint к расписанию; при продолжении
нужно сохранить исходные параметры периодов.

### Исполнение теплового линейного шага

Период `ThermalLiquidStation` принимает `thermal_execution`:

- `auto` (по умолчанию): повторно используемый CUDA Graph на CUDA,
  обычное eager-исполнение на CPU;
- `eager`: обычные вызовы Torch на выбранном устройстве для парной проверки;
- `graph`: требует CUDA; ошибка захвата не подменяется eager/CPU.

CLI: `run_thermal_liquid.py --thermal-execution graph`.
Параметр входит в fingerprint расписания. Для прямого теплового API
используется `execution`: `advance_record`, `advance_records`, `open_session`.
Режим исполнения не меняет допуски, число итераций, адаптацию времени,
проверки энергетического баланса или формат журналов. Кэш графов ограничен
четырьмя вариантами на экземпляр теплового адаптера; значения текущего
шага, масс, коэффициентов и источников обновляются перед каждым replay.

### Компактный механический журнал

Жидкостные периоды принимают `hydraulic_journal`: `full` (по умолчанию)
или `compact`. `hydraulic_journal_branches` — необязательный список имён
ветвей, чьи средние расходы нужны в компактном журнале. В full-режиме
выгружаются все ветви. Неизвестные и повторяющиеся имена отклоняются.

```json
{
  "duration_s": 0.1,
  "hydraulic_journal": "compact",
  "hydraulic_journal_branches": ["line", "p1"]
}
```

Компактный режим сохраняет каждый принятый подшаг: время/dt, массу и
энергию, максимальные невязки, ограничения мощности, фактические моменты
и мощности насосов, невязки валов, положения приводов арматуры и relief.
Ветви рециркуляции добавляются к выбранным автоматически. Регуляторы,
защиты и интеграл рециркуляции получают прежние данные.

Из промежуточных записей исключаются `midpoint_pressures_pa`,
`final_pressures_pa`, `final_pipe_volume_rates_m3_s`,
`final_speeds_rad_s`, `boundary_mass_transfers_in_kg`,
`dissipation_heat_j`. `midpoint_branch_volume_rates_m3_s` содержит только
выбранные/обязательные ветви. Возвращаемый `state` всегда полный;
полные координаты выгружаются в конце каждого физического `advance`,
включая внутренние вызовы из регуляторов, а не только внешнего отчёта.

Требуется `backend=torch`, `hydraulic_state_transfer=interval`.
У тепловой станции также требуется `thermal_state_transfer=report` и
прямое сопряжение `hydraulic_heat_transfer=auto/device` на одном устройстве.
Несовместимые режимы вызывают ошибку; скрытого перехода на full нет.
На Torch CPU компактный режим поддерживается теми же уравнениями.

CLI всех пяти жидкостных блоков: `--hydraulic-journal compact` и
`--hydraulic-journal-branches line p1`. Экспорт температур управляется
отдельно через `heat.store_step_temperatures`; компактный механический
журнал не отключает тепловую историю. Политика и список ветвей входят
в fingerprint расписания; при restart их нужно сохранить.

## Отдельный компонентный PR-сепаратор

Пример `configs/surface_compositional_separator.json` запускается через
`scripts/run_compositional_separator.py`. Это отдельная схема `system`,
`initial`, `periods`, не неявная опция существующего `coupled_facility`.
Методика и ограничения: [COMPOSITIONAL_SEPARATOR.md](COMPOSITIONAL_SEPARATOR.md).

`system`: `fluid` (схема `EquilibriumFluid`), `volume_m3`, список `outlets`,
необязательная `wall` (схема `VesselWall`), `elevation_m`, `gravity_m_s2`.
Численные поля: `maximum_dt_s` (0.1), `minimum_dt_s` (1e-7),
`relative_tolerance` (1e-4), `temperature_tolerance_k` (0.001),
`temporal_error_control` (true), `maximum_fractional_inventory_change`
(0.1), `maximum_temperature_change_k` (2). Все величины SI.

Выход: обязательные `name`, `back_pressure_pa`,
`volume_conductance_m3_s_pa`; `selection` по умолчанию `bulk`,
также `densest`, `lightest`, `component_rich`. Последний требует `component`
и допускает `minimum_mole_fraction` (0.5). `opening` по умолчанию 1.
`maximum_pressure_drop_fraction` (0.05) ограничивает рабочую область.
Пропорциональная обратная связь: `pressure_setpoint_pa`,
`opening_gain_per_pa` (0), `opening_bias` (0). Это эмпирический закон
отбора, не газовое сопло и не геометрия уровней.

`initial`: `pressure_pa`, `temperature_k`, `mole_fractions`, при стенке
`wall_temperature_k`. Чистый PT точно на линии насыщения не задаёт
фазовые доли; такой начальный запас задаётся через типизированный
Python API `EquilibriumInventory`, а не неоднозначным PT в CLI.

Период: обязательный `duration_s`; необязательные `feeds`, `heat_w`,
`utility_temperature_k`, `openings` (имя выхода → [0,1]), `max_substeps`.
Ненулевое тепло требует температуру внешнего источника. Питание задаётся
объектом `{"from_pt": {...}}` с компонентными молярными расходами, P/T
либо `{"stream": {...}}` с сериализованным `ComponentEnergyStream`.
Для пассивного питания проверяется давление источника.
Неизвестные имена выходов и поля периода отвергаются заранее.

CLI: `--device cpu|cuda`, `--output PATH`, `--stop-after N`,
`--checkpoint PATH`. Checkpoint привязан к исходникам, runtime, устройству
и всему расписанию; физический повтор префикса обязателен.

## Сеть конечных компонентных PR-сосудов

Пример `configs/surface_compositional_network.json`, CLI
`scripts/run_compositional_network.py`; методика и область применения:
[COMPOSITIONAL_NETWORK.md](COMPOSITIONAL_NETWORK.md).

`system.vessels` — словарь имя → конфигурация компонентного сепаратора.
У всех одна EOS, реестр, энергетические референсы, устройство, g и отметка.
`system.connections` — список объектов с обязательными `name`, `source`,
`target`, `volume_conductance_m3_s_pa` (неотрицательная). Поля по умолчанию:
`allow_reverse=true`, `opening=1`, `maximum_pressure_drop_fraction=0.05`,
`source_selection=bulk`, `target_selection=bulk`, `minimum_mole_fraction=0.5`.
При `component_rich` соответствующая сторона требует `source_component`
или `target_component`. Проводимость использует объёмные условия донора.

Численные параметры сети: `maximum_dt_s=0.1`, `minimum_dt_s=1e-7`,
`relative_tolerance=1e-4`, `temperature_tolerance_k=0.001`,
`temporal_error_control=true`, `maximum_branch_attempts=24`.
Ньютон общий для всей сети; собственные временные параметры отдельных
сепараторов не заменяют сетевые. Ограничения изменения их запасов и
температур при этом сохраняются.

`initial`: словарь всех имён сосудов → условия PT/состава/стенки.
Часы сосудов должны совпадать. Период принимает `duration_s`, `feeds`
(сосуд → список потоков), `heat_w`, `utility_temperature_k` (по сосудам),
`outlet_openings` (сосуд → выход → [0,1]), `connection_openings`
(соединение → [0,1]), `max_substeps`. Все величины SI.

CLI поддерживает `--device cpu|cuda`, `--output`, `--stop-after`,
`--checkpoint`. Изменение исходников/устройства/расписания запрещает restart.
Не допускаются разные высоты, скрытая замена GPU на CPU и автоматическое
сведение разных EOS. Связь с FIM или другими типами оборудования этим
конфигом не включается.

## Кинематическая PR-геометрия и уровневое регулирование

[Руководство](PR_PHASE_GEOMETRY.md), примеры
`configs/surface_pr_geometry.json`, `configs/surface_pr_geometry_network.json`.
Это идеальное разбиение фаз на слои при общем давлении и `gravity_m_s2=0`.
Гидростатический PR-профиль, осаждение и капельный унос не рассчитываются.

`CompositionalSeparator.geometry`: `diameter_m`, `length_m`,
`orientation` (`horizontal`/`vertical`); объём должен совпадать с `volume_m3`.
У выхода пара `height_m`, `diameter_m` задаёт положение и круговую площадь
фазового отбора. У соединения независимые пары `source_height_m`,
`source_diameter_m`, `target_height_m`, `target_diameter_m`.
Заданная пара требует геометрии соответствующего сосуда и полного
размещения отверстия по высоте; диаметр не превышает диаметр сосуда.
Без пары остаётся прежнее взвешивание по глобальным фазовым объёмам.

`level_control`: `component`, `setpoint_m`, `opening_gain_per_m`,
`opening_bias` (0), `minimum_mole_fraction` (0.5). Измеряется верхняя
граница наиболее богатой указанным компонентом фазы при выполнении
порога. При отсутствии фазы автоматическое открытие равно нулю.
У соединения `control_side` — `source` (по умолчанию) либо `target`;
выбор измеряемого сосуда не меняется при реверсе. У внешнего выхода
давление и уровень не могут одновременно задавать два автоматических
открытия. Явное открытие периода имеет приоритет.

Это пропорциональная обратная связь без динамики привода и интегратора.
Поля входят в model ID и source-bound restart; старые checkpoint после
изменения исходников намеренно несовместимы.

## Закрытая равновесная PR-колонна с гравитацией

Это отдельный формат: `configs/surface_stratified_pr.json`,
`surface_stratified_pr_3phase.json`, `surface_stratified_pr_gas.json`.
CLI `scripts/run_stratified_pr.py`, [методика](STRATIFIED_PR_COLUMN.md).
Не является периодом `CompositionalNetwork`.

Корень: `system`, `initial`, `samples`. System:
`fluid` (реестр/кij/область PR), `geometry` (цилиндр),
`gravity_m_s2=9.80665`, `elevation_m=0`, `quadrature_order=4`,
`maximum_quadrature_order=16`, `relative_tolerance=1e-6`,
`temperature_tolerance_k=1e-5`. Оба порядка из 2/4/8/16, максимум выше
начального; требуется подтверждение уточнением, не только успех Ньютона.

Initial — либо `pressure_pa`, `temperature_k`, `mole_fractions` для
исходных равномерных фаз с их геометрической потенциальной энергией,
либо `component_moles`, `total_energy_j`, `pressure_guess_pa`,
`temperature_guess_k` для прямого задания сохраняемых N и U+PE.
Равновесную общую температуру находит решение по полной энергии.

Sample: `height_m` либо неотрицательный целый `interface_index` с
обязательным `side=below|above`; необязательный `molar_rate_mol_s`
строит локальный `ComponentEnergyStream`. Он не обновляет запас колонны.
Все величины SI. За пределами геометрии/термодинамики, при вырождении
слоя или неподтверждённом фазовом наборе результат не публикуется.

CLI: `--device cpu|cuda`, `--output PATH`. Выход атомарный и содержит
хеши физических исходников до импорта/после расчёта, начальные запасы,
профиль, интерфейсы, U/PE/S, квадратуру и локальные потоки.

## Конечные PR-запасы трубы и Darcy-гидравлика

Пример: `configs/surface_pr_linepack.json`; runner `scripts/run_pr_linepack.py`.
Корень содержит `model`, `initial` и `schedule`. Описание
`model="quasisteady_pr_linepack_v1"` включает терминальные `vessels`,
список `pipes` и параметры временного интегрирования `integration`.
Начальные состояния задаются отдельно для `terminals` и `pipes`.

У трубы обязательны имя/терминалы, относительные высоты концов, L/D,
целое число ячеек, `dynamic_viscosity_pa_s` (Pa*s) и общая для всей трубы
`wall`. Шероховатость задаётся в m. Теплоёмкость стенки — J/K, оба UA — W/K;
это суммарные величины, которые компилятор делит между ячейками.
Полные границы, поля и аудит: [PR_LINEPACK.md](PR_LINEPACK.md).

`pressure_pa`, `temperature_k` и `mole_fractions` начальной трубы могут
быть общими или заданными массивами по ячейкам. Температура стенки
необязательна и по умолчанию равна заданной температуре жидкости.
Расписание задаёт положительные `duration_s` и необязательные открытия
граней вида `pipe:NAME:face:INDEX`. Для Darcy-грани открытия только 0/1.
`allow_reverse=False` у трубы означает обратный клапан на входной грани,
а не запрет внутреннего перераспределения во всех численных ячейках.
Горизонтальная геометрия и ограничения общего PR-реестра проверяются явно.

## Однофазная динамика импульса PR-трубы

`configs/surface_pr_momentum_pipe.json` и
`configs/surface_pr_momentum_acoustic.json` используют отдельный
`model="single_phase_pr_momentum_v1"`. Обязательны fluid, L/D, число
ячеек и `wave_speed_bound_m_s`; скорость состояния должна укладываться
в `maximum_velocity_m_s`, а физические характеристики — в заданную
численную границу. Вязкость задаётся в Pa*s; нулевое значение отключает
трение. Перепады давления не задаются как фиксированный расход.

Initial содержит PT/состав и `velocity_m_s` (скаляры или массивы ячеек).
Schedule задаёт `duration_s`; для `boundary_mode="ends"` дополнительно
допускаются `left/right`: `"wall"`, отсутствие либо заданные PT/состав.
В периодическом режиме концевые команды запрещены. Подробности,
механические энергетические отсчёты и ограничения:
[PR_MOMENTUM_PIPE.md](PR_MOMENTUM_PIPE.md).

## Конечные сосуды и тепловая PR-труба с импульсом

Конфиг `configs/surface_pr_momentum_vessels.json`, runner
`scripts/run_pr_momentum_vessels.py`, независимый аудитор
`scripts/check_pr_momentum_vessels.py`. Формат, уравнения и ограничения:
[PR_MOMENTUM_VESSELS.md](PR_MOMENTUM_VESSELS.md).
Это однофазная горизонтальная система при g=0; она не подменяет
гравитационный `StratifiedPRNetwork`.

## Гравитационные PR-профили и инерционная труба

Конфиги `configs/surface_stratified_pr_momentum.json` и
`configs/surface_stratified_pr_momentum_reversal.json`, runner
`scripts/run_stratified_pr_momentum.py`, независимый аудитор
`scripts/check_stratified_pr_momentum.py`. API и формат:
[STRATIFIED_PR_MOMENTUM.md](STRATIFIED_PR_MOMENTUM.md).
Одна устойчивая фаза, горизонтальная труба и центр-высотная
аппроксимация патрубков; свободная межфазная граница не поддерживается.

## Однофазная наклонная PR-труба

Модель `inclined_single_phase_pr_momentum_v1`: обычные параметры трубы плюс
`elevation_change_m`, `reference_elevation_m`, `gravity_m_s2`. Состояние
содержит гидростатические реконструкции и U+K+PE. Примеры, якорная
инициализация `hydrostatic`/`equilibrium`/`anchor_height_m`, команды запуска
и ограничения описаны в [PR_INCLINED_PIPE.md](PR_INCLINED_PIPE.md).

### Общий равновесный многофазный транспорт

Модель `pr_equilibrium_transport_graph_v1` и её ограничения описаны в
[PR_EQUILIBRIUM_TRANSPORT.md](PR_EQUILIBRIUM_TRANSPORT.md). Примеры
`surface_pr_equilibrium_transport.json` и `surface_pr_transport_*.json`
задают произвольный граф конечных сосудов и трубных ячеек, фазовое равновесие,
импульс, полную энергию, стенки и бинарную изоляцию. Проскальзывание опциональное
и экспериментальное; геометрические отборы перемешанных сосудов требуют
нулевой тяжести. Боковая слоистая реконструкция трубной ячейки допускает
тяжесть при явном ограничении `rho_max*g*cell_height/P <= 1e-3`.
Гравитационную стратифицированную модель нельзя автоматически
подменять таким сосудом.

В трубной ячейке `left_diameter_m` и `right_diameter_m` задаются вместе;
объём обязан совпадать с объёмом усечённого конуса, а `diameter_m` —
со средним диаметром. Соседние связи используют диаметр соответствующей грани.
`PREquilibriumNozzle` строит такие ячейки из массива диаметров и длин.
`wave_fan`: `fixed_bound` (по умолчанию), `local_single_phase`,
`pressure_single_phase`. `reservoir_ports`: `riemann` (по умолчанию) или
`stagnation_single_phase`. Ограничения и примеры
`surface_pr_resolved_nozzle.json`, `surface_pr_transonic_nozzle.json` описаны
в [PR_RESOLVED_NOZZLE.md](PR_RESOLVED_NOZZLE.md).

`heat_links`: список `name`, `source`, `target`, `conductance_w_k`.
Имена уникальны среди всех соединений; `openings` задаёт их изоляцию.
В транспортном графе стороны — узлы; в гравитационном графе стороны
могут быть также колоннами. См. [тепловые контуры](PR_THERMAL_CIRCUITS.md).
`drift_flux.model="relap7_vertical_pipe"` выбирает
[явное вертикальное режимное замыкание](PR_VERTICAL_DRIFT.md).

`drift_flux.model="dispersed_oil_water"` задаёт
[разбавленную дисперсию](PR_DISPERSED_LIQUIDS.md) с тремя компонентами-маркерами,
диаметром капли, вязкостями, натяжением и `dispersed_liquid: oil|water`.
`pumps`: список `PRPumpUnit` (`node`, `card` и границы применимости).
`compressors`: список `PRCompressorUnit` (`name`, `source`, `target`, `card`,
`speed_rad_s`, `reference_mole_fractions` и границы применимости).
Имена оборудования входят в `openings`; при отсутствии команды используются
дефолты модели на текущий интервал. Подробности, режимы отключения и
учёт `total_work_j`: [оборудование PR-графа](PR_MAPPED_MACHINES.md).

`PREquilibriumLink.source_port/target_port`: `left`, `right` или именованный
`side:<id>`. Боковой порт не занимает осевые торцы. При `source_height_m`
или `target_height_m` для боковой стороны высота относится к низу ячейки;
допустимы только цилиндрические горизонтальные или направленные вверх
вертикальные ячейки. Площадь фазового пересечения считается по круглому
отверстию; боковой поток сохраняет полную энергию и переносит осевой
импульс выбранного фазового донора. Два боковых направления требуют
конечного смешивающего узла.

`PRResolvedSeparator` компилирует вертикальный сосуд из `name`, `diameter_m`,
`height_m`, `cells`, положительного `terminal_response_s`, необязательных
`bottom_elevation_m`, `dynamic_viscosity_pa_s`, `wall`. Это сборщик
`pr_equilibrium_transport_graph_v1`, а не отдельный динамический backend.
`PRLateralNozzle` компилируется в обычное материальное соединение.
`join_pr_transport_graphs` объединяет совместимые графы без скрытых источников.
Формат примеров и область линейного дрейфа:
[PR_RESOLVED_SEPARATOR.md](PR_RESOLVED_SEPARATOR.md).

`PRControlValve.resistance_node` явно выбирает HEM-ячейку сопротивления
на материальном соединении, если её нельзя выбрать однозначно.
`PRSurfacePI.measurement="inventory_level"` требует `component` и
`sensor` — имя непрерывной вертикальной группы постоянного сечения.
Измерение равно выбранному фазовому объёму, делённому на сечение.
Для давления `sensor` необязательно задаёт конкретный узел.

### Гравитационные колонны и инерционный транспорт в общем графе

`pr_gravity_transport_graph_v1` содержит `transport`, именованные `columns`,
список `ports` (`PRColumnPort`) и необязательные `walls` колонн.
`maximum_transverse_head_fraction` ограничивает применимость одномерного входа.
Пример: `configs/surface_pr_gravity_transport.json`.
Команды расписания: `duration_s`, бинарные `openings`, `heat_w`,
`utility_temperature_k`, `loss_coefficients`. Имена тепловых команд относятся
как к колоннам, так и к узлам транспортного графа.
Подробности и ограничения: [PR_GRAVITY_TRANSPORT.md](PR_GRAVITY_TRANSPORT.md).

### Управление отбором в связанном PR-графе

`pr_controlled_surface_v1` содержит `plant: pr_gravity_transport_graph_v1`,
`valves: PRControlValve`, `controllers: PRSurfacePI` и параметры адаптивного
управляющего шага. Положения приводов и интеграторы сохраняются в checkpoint.
Команды расписания дополняются `targets` (ручные положения) и `actuator_modes`
(`normal`, `fail_safe`, `stuck`). Гидравлические изоляция, тепло и сопротивления
передаются в plant. Пример: `configs/surface_pr_controlled_surface.json`.
Модель ограничена малыми перепадами и докритическим HEM-течением; это не
универсальная модель choking/cavitation клапана.
Подробности: [PR_SURFACE_CONTROL.md](PR_SURFACE_CONTROL.md).

### Профиль трассы и коалесценция в общем PR-графе

`PRTerrainPipe` принимает `segment_lengths_m`, `elevations_m`,
`cells_per_segment`, `bend_loss_coefficients` и обычные свойства стенки/трения.
Builder выставляет `hydrostatic_reconstruction=true` трубным ячейкам.
`split_at_levels` и `initialize_from_gravity_profile` обеспечивают согласованную
начальную сетку: [область применимости](PR_TERRAIN_HOLDUP.md).

`PREquilibriumNode.droplet_population` содержит `family`,
`coalescence_kernel_m3_s`, `maximum_number_density_m3` и
`gas_compressibility_threshold`. Требуется `drift_flux.model="dispersed_oil_water"`.
`droplet_capture=true` задаёт явный конечный сборник без собственного числа
капель. В initial число можно задать через `droplet_number`; checkpoint
сохраняет его обязательно. В отчёте — `droplet_populations` по узлам и
`droplet_number_fluxes` по соединениям.
Подробности и ограничения: [капельная модель](PR_DROPLET_POPULATIONS.md).

`drift_flux.model="relap7_vertical_transition"` задаёт равновесный
пузырьково-снарядный переход с явными `surface_tension_n_m` и
`liquid_viscosity_pa_s`: [режимное замыкание](PR_VERTICAL_DRIFT.md).

`hydrostatic_face_guesses` — необязательные начальные приближения двух
торцевых алгебраических состояний трубной ячейки:
`[["left", pressure_pa, mole_fractions], ["right", pressure_pa, mole_fractions]]`.
Поле допускается в initial и сохраняется в checkpoint. Оно не задаёт
граничные условия и не добавляет запасов. Для неподвижного общего
химико-гравитационного профиля приближения строятся автоматически;
старое состояние без поля может быть восстановлено тем же способом.
Подробности: [трёхфазный покой](PR_TERRAIN_HOLDUP.md#трёхфазный-покой-и-перезапуск).

### Конечные приводы и насосные соединения PR

`drives` добавляет `PRMachineDrive` к именованной карте. Скорости хранятся
в `state.rotors`; команды момента передаются через `drive_torque_nm`.
`heat_sink_node` направляет потери в конечный материальный узел.
`pump_connections` задаёт насос между конечными узлами, без искусственного
расхода по средней скорости насосной ячейки. Проверяются PS/PH, обратный
клапан, карта и баланс электричество–вал–ротор–тепло.
Подробности, пределы применимости и примеры: [PR_MACHINE_DRIVES.md](PR_MACHINE_DRIVES.md).

Для продолжения сохранённого расчёта CLI транспорта, гравитационной сети
и регулируемой сети принимает `initial: {"state": <полный checkpoint>}`.
Этот вариант несовместим с одновременным заданием сырых начальных узлов.
Проверяются идентичность модели, реестры запасов, фаз, роторов и регуляторов.
Независимый аудит сверяет начальное состояние с запрошенным checkpoint.

## Критическая арматура, концентрированные дисперсии и дробление

Общий PR-граф принимает `restrictions` с `PRHEMRestriction`; команды
`restriction_openings` меняют площадь горла. Управляемая характеристика
`hem_throat_area` связывает площадь с приводом. Для удалённого датчика
уровня PI указывается `sensor` с именем гравитационной колонны.
Параметры, формулы и ограничения: [критические соединения](PR_CRITICAL_RESTRICTIONS.md).

Замыкание `concentrated_oil_water` использует заданную морфологическую
кривую, стеснённое осаждение и живую вязкость; допустима только объявленная
область сферических капель с неподвижной поверхностью:
[концентрированные дисперсии](PR_CONCENTRATED_EMULSIONS.md).

Популяция `powered_monomoment_breakup` добавляет `breakup_rate_s` и
`agitation_power_w` к параметрам коалесценции. Мощность входит как работа,
а не тепловой поток: [дробление капель](PR_POWERED_BREAKUP.md).
Примеры этих моделей включены в `scripts/validate_pr_surface_matrix.py`;
прохождение функциональной матрицы не заменяет промысловую калибровку.

## Непрерывное газожидкостное замыкание

`drift_flux.model="holmes_gas_liquid"` принимает явные параметры
профиля A/B/Fv и перехода a1/a2, поверхностное натяжение и границы
газосодержания, плотностей, наклона и безразмерного диаметра.
Оно поддержано в `PREquilibriumNode`, `PREquilibriumPipe` и общем
JSON/restart API. [Формулы, проверка однозначности и ограничения](PR_HOLMES_DRIFT.md).
Пример: `configs/surface_pr_holmes_inclined.json`.

## Спектр капель

Популяция `sectional_droplet_population` задаёт опорные объёмы, начальные
объёмные доли, частоты бинарного дробления, ядро коалесценции и мощность
мешалки. По секциям решаются перенос и реакции; крупный хвост имеет
собственные число и объёмный момент. Параметры, энергия, скорости,
пограничные условия и restart: [секционная модель](PR_SECTIONAL_POPULATIONS.md).

## Управление приводами и защиты общего PR-графа

Схемы `motors`, `trips`, измерения `speed`, `temperature`, `pump_flow` и
`compressor_margin`, а также команда `reset_trips` описаны в
[управлении приводами](PR_CONTROL_CHANNELS.md). Примеры: `surface_pr_controlled_motor.json`,
`surface_pr_controlled_minimum_flow.json`, `surface_pr_controlled_antisurge.json`.

## Дополнительные физические области общего графа

- `surface_pr_inclined_inventory.json`: [наклонные гравитационные запасы](PR_INCLINED_INVENTORY.md).
- `surface_pr_stratified_drift.json`: [баланс сил слоистого течения](PR_STRATIFIED_DRIFT.md).
- `surface_pr_sectional_beta_mixer.json`: [статистическое дробление](PR_SECTIONAL_POPULATIONS.md).
- `surface_pr_signed_pump_reversal.json`: [знаковая карта и реверс вала](PR_PUMP_FOUR_QUADRANT.md).
- `surface_pr_signed_pump_connection.json`: [PS/PH-соединение с реверсом](PR_SIGNED_PUMP_CONNECTION.md).
- `surface_pr_signed_compressor.json`: [знаковая газовая карта](PR_SIGNED_COMPRESSOR.md).
- `surface_pr_controlled_signed_pump.json`: отрицательная уставка скорости, защита и изоляция.
- `surface_pr_layered_transport.json`: [наклонные профильные Euler-ячейки](PR_LAYERED_TRANSPORT.md).
- `surface_pr_layered_plant.json`: [единый Ньютон профильной трубы и PR-графа](PR_LAYERED_PLANT.md).
- `surface_pr_profile_graph.json`: [несколько профильных ветвей, общие узлы и изоляция](PR_PROFILE_GRAPH.md).
- `surface_pr_controlled_profile_graph.json`: [PI, защиты и сохраняемая изоляция профильного графа](PR_PROFILE_CONTROLS.md).
- `surface_pr_interphase_relaxation.json`: [локальная фазовая инерция, работа давления и тепло](PR_INTERPHASE_RELAXATION.md).
- `surface_pr_interphase_three_phase.json`: трёхфазный локальный обмен с неподвижной стенкой.
- `surface_pr_interphase_stiff.json`: жёсткий предел межфазного сопротивления.
- `surface_pr_multifluid_counterflow.json`: [пространственные фазовые потоки и локальный обмен](PR_MULTIFLUID_PIPE.md).
- `surface_pr_multifluid_three_phase.json`: три фазовых запаса с раздельными P/T/скоростями.
- `surface_pr_multifluid_walls.json`: отражающие торцы и импульс опор.
- `surface_pr_multifluid_network.json`: [конечные опорные узлы и разветвлённая неравновесная сеть](PR_MULTIFLUID_PIPE.md#конечные-узлы-и-разветвления); векторные реакции опор, общий транспортный Ньютон и restart.

- `surface_pr_multifluid_gravity.json`: наклонная двухфазная сеть v2, отметки портов, вес фаз и потенциальная энергия.
- `surface_pr_multifluid_gravity_three_phase.json`: тот же баланс для трёх фаз; сохраняются все фазы, фазовых переходов нет.

- `surface_pr_multifluid_hydrostatic.json`: v3, проверенный двухфазный изотермический гидростатический профиль без численного дрейфа.
- `surface_pr_multifluid_hydrostatic_three_phase.json`: тот же режим для трёх фаз; требует стационарных локальных источников.
- `surface_pr_multifluid_pump.json`: v4, фазовый жидкостный насос с явной знаковой картой, рабочей плотностью и внешней работой в балансе энергии.

### Пространственная HEM-труба и IMEX-сопряжение оборудования

`pr_equilibrium_euler_pipe_v1`: `lengths_m`, `diameter_m`, `fluid`,
`boundary_mode` (`periodic`/`walls`), акустическая граница и CFL задаются
нормализованным `PREquilibriumEulerPipe.to_dict()`. Начальные данные —
`initial.cells` с компонентами, полной энергией, импульсом и начальными
приближениями P/T либо `initial.state` с полным checkpoint. Необходимые
единицы и точные ключи приведены в четырёх `surface_pr_equilibrium_euler_*.json`.

`pr_equilibrium_euler_plant_v1`: `transport` — полное описание существующей
PR-сети оборудования; `pipes` — именованный реестр с полями `pipe`,
`boundary_nodes` (левый и правый конечные сосуды либо `null` для закрытого
конца), `direction` (горизонтальный единичный вектор). Начало задаётся
`initial.state`, содержащим состояния оборудования и всех труб.

Команды интервала: `duration_s`, `maximum_step_s`, `minimum_step_s`,
`maximum_attempts`; физические команды — `openings`, `pipe_heat_w`,
`pipe_heat_temperatures_k`, `graph_heat_w`, `graph_utility_temperature_k`,
`drive_torque_nm`, `restriction_openings`. Тепло трубы задаётся по ячейкам,
температуры резервуаров обязательны для ненулевой мощности. Примеры связки
сосудов, насоса с ротором и компрессора —
`surface_pr_equilibrium_euler_plant_{cycle,pump,compressor}.json`.

Запуск и независимая проверка используют общие
`scripts/run_pr_multifluid_pipe.py` и `scripts/check_pr_multifluid_pipe.py`.
Модель выполняется на Torch CPU/CUDA, но текущий последовательный UV-контроль
не обеспечивает полностью асинхронную работу GPU.

### Сборка общей неявной PR-сети

Python API: `build_pr_surface_network(fluid, vessels, pipes, ...)`.
`pipes` принимает `PREquilibriumPipe`, `PREquilibriumNozzle` и `PRTerrainPipe`.
Последний поддерживает `drift_flux` как объект замыкания либо его словарь.

`columns` — именованные `StratifiedPRColumn`; `column_terminals` —
`PRColumnTerminal(name, column, height_m)`. Имена терминалов допустимы в
`source`/`target` труб вместо конечного vessel-узла. Терминалы не входят
в запасы сети. `column_walls`, `column_heat_links` задают физические стенки
и тепловые связи. `resolved_separators` и `resolved_nozzles` включают
осевые сепараторы и боковые патрубки. Обычные `links`, `heat_links` и
оборудование транспортной сети сохраняют свои существующие параметры.

`initialize_pr_surface_network(model, transport_initial, column_initial, ...)`
инициализирует общую сеть: трубы можно задавать сгруппированными полями,
колонны — аргументами PTz либо готовыми `StratifiedPRState`.
Нормализованная модель и checkpoint используют `pr_gravity_transport_v1`;
нового несовместимого формата расчётных результатов не добавлено.

Межблочные `links`, `heat_links` и оборудование добавляются после сборки
всех физических узлов, чтобы ссылки на ячейки осевого сепаратора не
отклонялись раньше времени. Допустима сеть из одного `PRResolvedSeparator`
без вспомогательного vessel-узла. Ограничения оборудования на реальные
пленумы, свободные торцы и однородные насосные ячейки при этом сохраняются.

### Термохимический источник и труба с активными фазами

`pr_thermochemical_relaxation_v1`: `fluid`, `relaxation_time_s` (>=0),
`maximum_active_phases` (1..6). Полный checkpoint использует
`PRPhaseEquilibriumState` с идентификатором этого источника; он допускает
разные P/T/u присутствующих фаз. CLI принимает `initial.state` либо
`initial.phases` и `initial.phase_roots` с каноническими фазовыми запасами.

`pr_active_multifluid_pipe_v1`: `fluid`, `phase_slots`, `lengths_m`,
`diameter_m`, `boundary_mode`, `wave_speed_bound_m_s`, `maximum_courant`
(<=0.25), `thermochemical_relaxation_time_s` (`null` отключает источник).
Слот задаёт `name`, `root` (`liquid`/`vapor`), необязательные
`component_marker` и `minimum_marker_fraction`. В `initial.cells` либо
полном `initial.state` перечисляются только присутствующие фазовые запасы.
Пустые фазы не заполняются нулевыми объектами или искусственным остатком.

Обе модели принимают команды `duration_s`, `maximum_step_s`,
`minimum_step_s`, `maximum_attempts`. Данные запаса содержат компоненты,
объём, P/T, полный импульс и полную энергию фазы. Компоненты — моль,
объём — м³, давление — Па, температура — K, импульс — кг·м/с, энергия — Дж.
Постоянная времени — параметр замыкания, а не результат калибровки.
