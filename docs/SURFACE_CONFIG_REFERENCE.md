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

### Активные многофазные трубы в IMEX-сети

Сериализация `PREquilibriumEulerPlant` использует
`model: pr_active_multifluid_plant_v1`, если хотя бы одна запись
`pipes.<name>.pipe` имеет `model: pr_active_multifluid_pipe_v1`.
Остальные поля совпадают с существующим IMEX-форматом: `transport`,
`pipes.<name>.boundary_nodes` (два имени конечных сосудов или `null` для
стенки), `pipes.<name>.direction` (горизонтальный единичный вектор).
Оба конца не могут одновременно быть стенками в связанном адаптере.
Каталог фаз трубы должен представлять фазовые ветви и составы подключённых
сосудов. Неподходящий каталог отклоняется, а не заменяется HEM-проекцией.
Состояния активных труб сохраняются как именованные фазовые запасы в `cells`;
состояния HEM-труб сохраняют прежние `bulk_stocks` и `next_step_s`.
Программный API `from_dict` восстанавливает смешанную сеть на выбранном
Torch-устройстве. Отдельный CLI-аудитор старого HEM-формата не является
аудитором нового смешанного формата.

### Гравитационная активная сеть

`pr_active_multifluid_pipe_v2` добавляет к v1 поля `face_elevations_m`
(число ячеек + 1), `gravity_m_s2` и опциональный `hydrostatic_reference`.
Гидростатический эталон — список ячеек со словарями полных фазовых запасов;
разрешён для отражающих концов без химической релаксации.

`pr_active_surface_v1` содержит:

- `columns`: словарь сериализованных `StratifiedPRColumn`;
- `ports`: записи `{name, column, height_m}`;
- `pipes`: словарь `{pipe, boundary_nodes}`; endpoints — имена патрубков,
  реальных узлов `equipment` или `null`;
- `equipment`: сериализованный `PREquilibriumTransportNetwork` или `null`;
- `maximum_transverse_head_fraction` (по умолчанию 1e-3, максимум 0.01);
- `aperture_relative_tolerance` (по умолчанию 1e-6).

Для runner задаются `system`, `initial`, `duration_s`, необязательные
`commands`, `maximum_attempts`, `minimum_step_s`. `commands.port_openings`
изолирует патрубки; `openings`, `drive_torque_nm`, `restriction_openings`
передаются существующему оборудованию. Единицы SI. Идентичность EOS,
устройства, гравитации и физических отметок проверяется при построении.
Полные примеры: `surface_pr_active_surface_{withdrawal,reverse,interface,isolated}.json`.
Точная область применимости и команды проверки: `PR_ACTIVE_SURFACE.md`.

### Стенки, трение и управление активной ветвью

`pr_active_multifluid_pipe_v3` добавляет опциональный объект:

```json
"wall_friction": {
  "model": "darcy_volume_wetted_v1",
  "viscosity_pa_s": {"gas": 0.00001, "liquid": 0.001},
  "roughness_m": 0.000001
}
```

Имена вязкостей обязаны покрывать весь `phase_slots`; единицы Pa·s и m.
Геометрический профиль v2 остаётся доступен в v3. Это адиабатическое
сопротивление с объёмным распределением смоченного периметра, не общий
автоматический режимный классификатор. Standalone runner/auditor активной
трубы распознают v1/v2/v3.

`pr_active_surface_v2` добавляет `column_walls` — словарь стандартных
сериализованных `VesselWall`. Checkpoint добавляет `wall_temperatures_k`;
Python initialize принимает одноимённый аргумент. Нагрев может менять
число фаз колонки. Полная энергия включает запас стенки.

`commands.port_positions` — словарь долей открытой площади [0,1]. Закрытая
доля сохраняет отражающую стенку и давление; это модель равномерной маски,
а не параметр Cv. `port_openings=false` полностью закрывает проход.
Вместо `duration_s` runner может принимать `schedule` из записей
`{duration_s, commands}`. Команды частично обновляются и сохраняются между
интервалами. Формат исходных v1-конфигов и checkpoint без стенок сохранён.

## Явно выбранная кольцевая плёнка активной PR-трубы

Поле `wall_friction` версии `pr_active_multifluid_pipe_v3`:

```json
{
  "model": "laminar_annular_wallis_interface_v1",
  "gas_slot": "gas",
  "liquid_slot": "liquid",
  "viscosity_pa_s": {"gas": 0.00001, "liquid": 0.001},
  "roughness_m": 0.0
}
```

Имена должны совпадать с единственными двумя слотами vapor/liquid трубы.
Требуются явные `face_elevations_m` и abs(dz)/L>=0.95. Для двухфазной ячейки
проверяются alpha_l<=0.3, rho_l>rho_g и Gamma/mu_l<=400; недопустимый режим
не заменяется молча другим. Исчезнувшая фаза не получает остаточной массы;
однофазные ячейки используют Darcy/Кольбрук. Непрерывная геометрия плёнки
сама по себе не является моделью волн/смачивания/уноса. Модель опциональна,
экспериментальная квалификация этого варианта Wallis пока отрицательная.

В `friction` отчёта дополнительно сохраняются `model`, `interface_work_j`,
`physical_dissipation_j`, `temporal_dissipation_j` и `annular_cells`
(фактическая доля жидкости, eta/beta/c, проводимость, напряжение и скорость
границы, film Re и driving gas Re). В полном балансе работа границы
суммируется с противоположными знаками; нагрев повторно не добавляется
к суммарной энергии. Независимый checker поддерживает эту модель.

Пример связанного запуска: `configs/surface_pr_active_surface_annular.json`.
Вывод об экспериментальной применимости приведён в
[MULTIPHASE_EXPERIMENTAL_VALIDATION.md](MULTIPHASE_EXPERIMENTAL_VALIDATION.md).

## Тепловые команды активных труб

Команды `PRActiveSurfaceNetwork` и `PREquilibriumEulerPlant`:

```json
{
  "pipe_heat_w": {"line": [{"gas": 10.0}, {"gas": -5.0}]},
  "pipe_heat_temperatures_k": {"line": [500.0, 250.0]}
}
```

Число элементов равно числу ячеек трубы. В многофазной ячейке мощность
обязательно задаётся по именам реально присутствующих фаз. Температура
может быть общей для ячейки или словарём фаз. Однофазная ячейка также
принимает скалярную мощность. Отрицательная мощность — охлаждение.
Ненулевая мощность требует положительной конечной температуры резервуара;
непассивная граница, перескок температуры и нагрев отсутствующей фазы
отклоняются. Параметры не означают UA или конечную теплоёмкость стенки.

Для самостоятельной активной трубы в `step/advance` и `schedule` runner
используются `heat_rates_w` и `heat_temperatures_k` без словаря имён труб.
Мощность сохраняется при дроблении шага; энергия — dt*Q фактического подшага.
`external_heat_j` входит в полный энергетический баланс ровно один раз.
`thermal` и `heated_state` позволяют независимо проверить источник до
термохимического превращения. Пример расписания с изоляцией и повторным
открытием: `configs/surface_pr_active_surface_pipe_heat.json`.

### Капли жидкости в газовом носителе

`drift_flux.model="dispersed_gas_liquid"` — отдельное замыкание разреженного
газокапельного течения в общем PR-графе. Оно использует существующие
`droplet_population` либо `sectional_droplet_population`, но носителем служит
газ, а не другая жидкость. Обязателен `gas_viscosity_pa_s`; остальные
материальные параметры задаются явно, как у жидкостной популяции.
`dispersed_liquid` выбирает `oil` или `water`; индексы маркерных компонентов
остаются трёхкомпонентными. Неиспользуемый компонент может иметь ровно
нулевой запас: фиктивная третья фаза не создаётся.

Относительная скорость капли определяется Stokes либо чистой границей
Hadamard–Rybczynski (`interface="clean"`). Скорости фаз удовлетворяют
массовому, а не объёмному барицентру. В секционной модели каждая секция
имеет свою скорость; Brownian-ядро использует вязкость газа. Коалесценция
изменяет поверхность и температуру при сохранении полной энергии.
Конечный `droplet_capture` принимает капли и допускает обратную подачу;
прямое соединение популяций с разными носителями запрещено.

Область ограничена разреженным носителем (доля газа не ниже 0.9), одним
типом жидких капель, малыми Re/Eo и конфайнментом. Для фактического размера
каждой секции проверяются капиллярная поправка давления и
`maximum_knudsen_number` (по умолчанию 0.01, увеличение выше 0.01 запрещено).
Затопление жидкостью не принимается как «газ без капель». Частицы с
разрешённой инерцией, образование капель из плёнки и эффективность
осаждения на стенку этим замыканием **не моделируются**.

Воспроизводимые примеры:
`configs/surface_pr_gas_mist.json`,
`configs/surface_pr_gas_mist_sections.json`,
`configs/surface_pr_gas_mist_capture.json`,
`configs/surface_pr_gas_mist_reinjection.json`.
Матрица: `scripts/validate_pr_surface_matrix.py --case gas_mist
--case gas_mist_sections --case gas_mist_capture --case gas_mist_reinjection
--device cpu --output /tmp/gas-mist-cpu` (команда записана в несколько строк
для чтения; запускать одной командой). Независимый аудитор —
`scripts/check_pr_equilibrium_transport.py`.

### Необязательная проверка полного смачивания

В `wall_friction` модели `laminar_annular_wallis_interface_v1` можно добавить:

```json
"wetting": {
  "model": "hartley_murgatroyd_profile_v1",
  "surface_tension_n_m": 0.02,
  "receding_contact_angle_deg": 20.0,
  "advancing_contact_angle_deg": 80.0,
  "maximum_curvature_ratio": 0.05,
  "require_full_wetting": false
}
```

Числа в примере демонстрационные, не измеренные свойства установки.
Контактные углы и поверхностное натяжение обязательны; скрытых значений
для неизвестного материала нет. Критерий интегрирует силу импульса
фактического радиального Stokes-профиля, учитывает гистерезис углов и
в тонком гравитационном пределе возвращает Hartley–Murgatroyd.
Отчёт `wetting` содержит силу, два порога, кривизну и признаки
`profile_valid`, `full_wetting_supported`, `dry_patch_supported`,
`history_dependent`. Профиль с внутренним реверсом и слишком толстая
плёнка не сертифицируются этим критерием.

`require_full_wetting=true` отклоняет неподтверждённый шаг атомарно.
Это диагностический критерий/защита области применимости, **не** эволюция
сухих пятен, ручейков, контактной линии или волн. При отсутствии `wetting`
прежний расчёт сопротивления не меняется. Критерий основан на
[Hartley & Murgatroyd (1964)](https://doi.org/10.1016/0017-9310(64)90042-0);
использование радиального профиля и двух материальных углов здесь является
явно ограниченным расширением, а не экспериментальной квалификацией.

`configs/surface_pr_active_surface_wetting.json` — связанный пример трубы,
сосуда и конечного патрубка с остановкой и повторным открытием. Углы 20°/80°
в нём заданы только для воспроизводимой демонстрации диагностики;
`require_full_wetting=false`. Они не являются свойствами Hannover.
Начальное состояние переинициализировано для нового идентификатора модели:
checkpoint другой конфигурации не принимается простым изменением флага.

Для газовых капель использованы стандартные предельные законы сопротивления
Stokes/HR; различие подвижной и неподвижной поверхности рассмотрено,
например, в первичной работе
[The transition in settling velocity of surfactant-covered droplets](https://arxiv.org/abs/1712.01567).
Выбор интерфейса — вход модели, а не подбор по выходному давлению.
Применимость терминальной скорости ограничивается перечисленными
безразмерными границами; конечная инерция капель этим не подтверждена.

### Пространственная стеночная плёнка: капиллярность и частичное смачивание

`ThinFilmDynamics` (`periodic_long_wave_film_v1`) — отдельная изотермическая
модель одного/двух координатных направлений периодического плоского участка
стенки. Толщина `h` является распределённым состоянием; динамика включает
капиллярное давление, Navier-slip и взаимодействие жидкости со стенкой.
Классическая градиентная постановка описана в
[Thiele (2010)](https://doi.org/10.1088/0953-8984/22/8/084019).
Численный алгоритм здесь — собственная дискретная градиентная схема с
замороженной положительной подвижностью; это не воспроизведение чужой
теоремы об устойчивости или экспериментальная квалификация.

Параметры задаются в SI: `shape`, `lengths_m`, `surface_tension_n_m`,
`viscosity_pa_s`, `density_kg_m3`, `adsorption_thickness_m`,
`equilibrium_contact_angle_deg`, `slip_length_m`. Для одномерной задачи
`transverse_width_m` задаёт фактическую ширину интегрирования; для двумерной
площадь уже определена двумя длинами. Контактный угол ограничен 20°,
`maximum_slope` — не более 0.2. Границы периодические. Нет входных/выходных
патрубков, турбулентности, инерции жидкости или газовой фазы.

Свободная энергия:

```text
F = integral [sigma*|grad(h)|²/2 + Phi(h) + rho*g_normal*h²/2] dA
Phi(h) = (10/3)*sigma*(1-cos(theta))*[-(he/h)²/2 + (he/h)^5/5]
M(h) = h³/(3*mu) + b*h²/mu
q = -M*grad(chemical_potential) + M*rho*g_tangent
    + [h²/(2*mu) + b*h/mu]*interface_shear
```

`he=adsorption_thickness_m` — явно заданная материальная длина адсорбционного
слоя. Его жидкость входит в полный объём. Это не численный остаточный запас
и не невидимая добавка массы: допустима положительная толщина меньше `he`,
обрезания `h=max(h,he)` нет. Полностью сухое состояние `h=0` эта версия
не представляет. Контактный угол равновесный; динамический гистерезис углов
не реализован. Двумерное поле позволяет развиваться неоднородности толщины
вдоль обеих координат, но не является геометрически точной цилиндрической
поверхностью трубы.

Каждый принятый шаг удовлетворяет дискретному балансу
`F_new-F_old + dissipation = external_work`, с неотрицательной диссипацией.
Объём жидкости сохраняется по каждой ячейке через гранные потоки.
Это баланс **свободной энергии изотермической плёнки**, а не готовый перенос
внутренней энергии PR-смеси. Накопленные `dissipated_energy_j` и
`external_work_j` сохраняются в checkpoint; их повторное суммирование как
нового тепла в другой модели было бы ошибкой.

Капиллярное и смачивающее давление интегрируются неявно. Явный поток от
гравитации/касательного напряжения ограничен Courant <=0.25. Ньютон использует
положительный line search без обрезания толщины; отказ дробит шаг.
`advance` проходит запрошенный интервал полностью. Линейный solve выполнен
через PyTorch; по умолчанию ограничен 512 неизвестными. Это ограничение
реализации плотного solve, не заявленная масштабируемая модель промысла.
Состояния сериализуются через CPU; резидентный интегратор GPU пока не заявлен.

Примеры:

```bash
PYTHONPATH=src:scripts python scripts/run_thin_film_dynamics.py \
  configs/surface_thin_film_dewetting.json --output /tmp/film.json
PYTHONPATH=src:scripts python scripts/check_thin_film_dynamics.py \
  configs/surface_thin_film_dewetting.json /tmp/film.json
```

`surface_thin_film_patch.json` задаёт двумерный участок. Материальные числа
в обоих примерах демонстрационные. Независимый NumPy-аудитор восстанавливает
каждый гранный поток, локальный/глобальный объём, работу, диссипацию и время
каждого подшага. Отдельные тесты сравнивают малые капиллярные и смачивающие
моды с аналитическим законом дисперсии.

**Граница интеграции:** модель доступна как отдельный расчёт стеночного
участка. Она ещё не меняет сопротивление/запасы `PRActiveSurfaceNetwork`, не
создаёт капли и не квалифицирует Hannover. Для полного сопряжения нужны
совместный перенос компонентов и энергии, геометрия поверхности трубы,
работа газового касательного напряжения и физический источник отрыва капель.

### Механическая работа газа и вязкий нагрев стеночной плёнки

Для переноса энергии в PR различаются работа изменения свободной энергии и
полная механическая работа. В отчёте плёнки `dissipation_j` теперь означает
полную вязкую диссипацию Stokes-профиля, включая Navier-slip на стенке.
`interface_work_j` — работа заданного касательного напряжения на фактической
скорости поверхности; `gravity_work_j` — работа тангенциальной гравитации.
Их сумма записывается в `external_work_j`. Для каждой грани возвращаются
`interface_velocity_m_s` и `face_dissipation_j`.

```text
f = rho*g_tangent - grad(chemical_potential)
M = h³/(3*mu) + b*h²/mu
S = h²/(2*mu) + b*h/mu
C = (h+b)/mu
q = M*f + S*tau
u_interface = S*f + C*tau
D = M*f² + 2*S*f*tau + C*tau² >= 0
W = rho*g_tangent*q + tau*u_interface
```

На грани усредняются обе положительные матрицы подвижности соседних
полосок. Положительность диссипации считается в устойчивой форме Schur;
аудитор отдельно интегрирует `mu*(du/dz)²` и потерю Navier-slip.
В частности, однородная плёнка под сдвигом имеет неизменную толщину и
свободную энергию, но ненулевые работу газа и нагрев. Прежняя величина
`M*grad(chemical_potential)²` описывала градиентную диссипацию и не была
полным механическим нагревом при внешнем приводе. При нулевых приводах
оба определения совпадают; прежние неприводные демонстрации не изменены.
Демонстрация постоянного сдвига: `configs/surface_thin_film_shear.json`.

### Консервативный материальный/тепловой перенос плёнки в PR

`remap_phase_volumes(pipe, state, dt_s, face_volume_fluxes_m3_s)` переносит
по периодическим граням донорские компоненты, объём, импульс и полную
энергию одной общей upwind-долей. Грань `i` направлена из ячейки `i` в
`i+1`; сумма пофазных объёмных потоков на каждой грани должна быть нулевой
(встречное вытеснение в ячейках фиксированного объёма). Нельзя отбирать
из отсутствующей фазы или больше её действительного запаса. Отрицательные
остатки не обрезаются: шаг отклоняется. Новые P/T восстанавливаются через PR
по действительным N/V/M/E; проверяются компонентный баланс и энтропия.
Оператор не меняет физические часы — их ведёт вызывающая схема расщепления.
Гравитационная съёмка этим оператором не поддержана и явно отклоняется.

`transfer_film_increment` связывает неприводную одномерную плёнку с этим
переносом: её жидкий объём совпадает с запасом выбранной PR-фазы, газ
вытесняется встречным потоком, поверхностная диссипация возвращается в
жидкость ровно один раз. Сохраняется сумма полной энергии PR и свободной
энергии плёнки. Профиль и PR-состояние возвращаются вместе; при перезапуске
нужно сохранять оба checkpoint. Допускается тонкая плоская полоска
`h/R <= .01` и совпадающие равномерные периодические сетки.

Проверяемый пример материального/теплового оператора:

```bash
PYTHONPATH=src:scripts python scripts/run_pr_film_transfer.py \
  configs/surface_pr_film_transfer.json --output /tmp/pr-film.json
PYTHONPATH=src:scripts python scripts/check_pr_film_transfer.py \
  configs/surface_pr_film_transfer.json /tmp/pr-film.json
```

**Этот оператор не является полным сопряжённым решателем:** он ещё не
передаёт капиллярные силы в инерционное уравнение импульса PR, не включает
отрыв капель и не подключён автоматически к произвольному графу сети.
Сохранение импульса переносимых порций не заменяет капиллярную механику.
Поэтому добавление этого API не меняет расчёты Norne и не закрывает
квалификацию многофазной сети. Постоянные свойства и материальные числа
демонстрационных конфигов не являются калибровкой реальной установки.

### Локальный обмен плёнка–капли с энергетическим ограничением

`PRFilmMistExchange` (`pr_film_mist_exchange_v1`) — отдельный локальный
источник для трёх действительных PR-запасов: `film_slot`, `mist_slot`
(оба `root="liquid"`) и `gas_slot` (`root="vapor"`). Это три морфологии
двух термодинамических фаз; газ и капли имеют отдельный импульс. Источник
не выполняет пространственный транспорт и не переключает термодинамические
корни на основании названия морфологии.

Обязательны `droplet_diameter_m`, `surface_tension_n_m`,
`atomization_efficiency` в (0,1], `heat_to_mist_fraction` в [0,1] и явное
`film_surface_model="planar_complete_wetting"` либо
`film_surface_model="planar_long_wave_wetting"` с явным `wetting`.
Первый вариант означает **идеальный
плоский, энергетически нейтральный резервуар плёнки**: нулевое изменение
суммы энергий стенки и плоской плёнки при обмене (нейтральное растекание,
без кривизны и смачивающего потенциала). Это не универсальное свойство любой
полностью смачиваемой поверхности. Во втором варианте используется общий
`LongWaveWettingPotential`, извлечённый из `ThinFilmDynamics`, а не новая
корреляция. Его параметры: `film_width_m`, `adsorption_thickness_m`,
`equilibrium_contact_angle_deg` (0–20°). Площадь стенки ячейки равна
`film_width_m * dx`, толщина — фактический жидкий объём, делённый на эту
площадь. В энергию входит `A_wall * Phi(h)`. При осаждении в плёнку поступает
`sigma * delta_A_drop - delta_F_wall`; при уносе стоимость равна
`sigma * delta_A_drop + delta_F_wall`. Положительная стоимость оплачивается
относительным движением; отрицательная высвобождается в тепло без
отрицательного сопротивления и без искусственного разгона фаз.
Это локальная плоская геометрия: пространственная энергия наклона не
принадлежит этому источнику и автоматически с ним не складывается.

Метод `apply` получает явно заданные массивы валовых объёмных интенсивностей
`deposition_m3_s` и `entrainment_m3_s`. Сначала выполняется осаждение, затем
унос из полученного запаса плёнки. Объём, компоненты, импульс и полная энергия
порции используют одну донорскую долю. Это заданный порядок первого порядка
по времени, а не неявное решение двух встречных интенсивностей.

Для монодисперсных сфер `A=6*V_mist/d` и `E_surface=sigma*A`. При осаждении
высвободившаяся поверхностная энергия передаётся плёнке. При создании капель
затраты оплачиваются из относительной кинетической энергии газа и капель:

```text
K_relative = 0.5 * (m_g*m_d/(m_g+m_d)) * (u_g-u_d)²
K_removed = max(E_surface_created, 0) / atomization_efficiency
Q_irreversible = K_removed - E_surface_created
```

Равные противоположные импульсы сохраняют суммарный импульс. Кинетическая
энергия уменьшается ровно на `K_removed`; избыток превращается в тепло с
явно заданным распределением газ/капли. PR восстанавливает P/T из новых
N/V/M/E. При эффективности 1 поверхность создаётся без охлаждения фаз:
энергия берётся из движения, а не скрытым вычитанием внутренней энергии.
Если доступного запаса нет, источник отклоняется атомарно. Энергетическое
условие является необходимым ограничением, не достаточным критерием
возникновения уноса. Равные валовые интенсивности не означают нулевых затрат:
осаждение и повторное создание поверхности учитываются раздельно.

Нулевые интенсивности сохраняют исходные состояния точно. В энергетически
нейтральном варианте при исчерпании запаса морфология исчезает без остаточной
массы; при притоке появляется
из действительного донорского состава. Источник не меняет часы — их ведёт
внешнее расщепление. Действительные компоненты/импульс и сумма полной энергии
фаз с поверхностной энергией сохраняются. Число капель в отчёте выводится
из объёма и фиксированного диаметра; это **не** переносимая численная
популяция при сжатии, коалесценции или изменении диаметра.
При ненулевом контактном угле смачивающий потенциал сингулярен в `h=0`.
Полное осушение отклоняется атомарно; остаточная масса не добавляется.
Материальная адсорбционная толщина — параметр энергии, не нижняя граница
объёма. Вариант не реализует рождение сухого пятна или динамику контактной линии.

Примеры `configs/surface_pr_film_mist_cycle.json` и
`configs/surface_pr_film_mist_reverse.json` задают локальное расписание
создания капель, остановки обмена и осаждения, с противоположными начальными
скоростями газа. Числа — демонстрационные, эффективность не калибрована.

```bash
PYTHONPATH=src:scripts python scripts/run_pr_film_mist_exchange.py \
  configs/surface_pr_film_mist_cycle.json --output /tmp/exchange.json
PYTHONPATH=src:scripts python scripts/check_pr_film_mist_exchange.py \
  configs/surface_pr_film_mist_cycle.json /tmp/exchange.json
```

Независимый аудитор использует скалярную PR EOS и заново находит конечные
скорости из сохранённого импульса центра масс и заданного уменьшения
относительной кинетической энергии. Он проверяет составы обеих валовых
порций, энергетические затраты, отсутствие фиктивных фаз и перезапуск.

**Не реализовано этим источником:** закон интенсивности/порог уноса,
криволинейная плёнка и сухие пятна, Laplace-равновесие капель, перенос
числа капель при пространственном движении, капиллярная инерционная механика
и автоматическое включение в граф. Разделение трёх потоков и необходимость
отдельных законов интенсивности/осаждения обсуждаются, например, в
[Utsuno & Kaminaga (1998)](https://www.jstage.jst.go.jp/article/jnst1964/35/9/35_9_643/_pdf).
Их корреляции для высоконапорного пароводяного течения здесь не переносились
на нефтегазовую смесь. Источник не заявляет экспериментальную квалификацию.

Участие задаётся именами трёх морфологий. Другие объявленные фазы проходят
через локальный источник без изменений N/V/M/E/P/T; это позволяет применять
оператор к выбранной группе без потери остальных запасов. Такое сохранение
не добавляет замыканий взаимодействия с этими фазами.

### Совместная инерционная капиллярная механика

`InertialCapillaryFilm` (`inertial_capillary_film_v1`) решает толщину и
импульс в одном нелинейном шаге. Это периодическая одномерная плоская
одножидкостная плёнка с постоянной плотностью, пассивной внешней средой и
нейтральным смачиванием (`equilibrium_contact_angle_deg=0`). Кинетическая
энергия использует plug-профиль; стеночное сопротивление — замыкание
Poiseuille/Navier для средней скорости. Инерция газа, контактная линия,
турбулентность и открытые торцы в этой версии отсутствуют.

Физическая постановка содержит конвективный перенос импульса, капиллярную
силу и, при `viscous_drag=true`, стеночное трение. Важность согласования
капиллярного дисперсионного члена с энергией рассматривается в
[Noble & Vila (2014)](https://arxiv.org/abs/1304.3805). Здесь реализована своя
разнесённая конечно-объёмная схема; теоремы для других дискретизаций ей
автоматически не приписываются.

Толщина хранится в центрах ячеек, скорость — на гранях. Для равномерной
сетки с шагом `dx`, шириной `W` и `A=dx*W`:

```text
h_face = (h_i+h_{i+1})/2
P = rho*A*sum(h_face*u_face)
K = rho*A/2*sum(h_face*u_face²)
F = A*sum[sigma/2*(grad(h))² + rho*g_normal/2*h²]
chemical_potential = -sigma*laplacian(h) + rho*g_normal*h
```

Потоки массы и конвективного импульса и капиллярная работа используют одну
согласованную временную скорость. Совместный Newton решается в PyTorch по
поправкам толщины/скорости: это предотвращает потерю малых изменений при
вычитании близких полных состояний. Положительный line search не обрезает
толщину; отказ дробит шаг. Отчётный интервал проходит полностью, а интервал,
не представимый физическими часами float64, явно отклоняется.

`time_scheme`:

- `roe`: временное среднее скорости с весами квадратных корней из гранной
  толщины. Для нелинейной кинетической энергии выполняется дискретное
  правило цепочки; в невязком режиме нет искусственного энергетического
  затухания. Название относится к весам во временном среднем. Пространственный
  и временной порядок проверены независимо по фазе малой волны.
- `backward`: устойчивый шаг по импульсу с явно выделенной положительной
  временной диссипацией.
- `auto`: использует `roe` для разрешённого режима и `backward`, когда
  локальная оценка `dt*gamma>2`, где `gamma=mu/[rho*h*(h/3+b)]`.
  Этот порог соответствует потере монотонного затухания однородного
  режима при центрированном шаге. Проверяются начальные и средние толщины.

На каждом шаге проверяются объём, импульс с реакцией стенки и энергия:

```text
Delta(K+F) + Q_physical + Q_numerical = 0
```

`physical_heat_j` — дискретизированная работа стеночного сопротивления,
`numerical_heat_j` — отдельно учтённая потеря временной схемы `backward`.
Большая численная доля показывает неразрешённость перехода по времени;
разделение тепла на физическую/численную части тогда требует уточнения
шага. В режиме `roe` численная часть равна нулю.

Возвращаются гранные потоки объёма, материального импульса и механической
энергии, поячеечные силы/нагрев и локальные невязки. Потоки энергии получены
из локальных дискретных правил произведения; они не восстанавливаются
распределением общей невязки по ячейкам. Разнесение исключает неподвижную
шахматную моду толщины. Размер плотной системы ограничен
`maximum_unknowns` (по умолчанию 256); это 2 неизвестные на ячейку.
GPU вычисляет нелинейную алгебру, но состояние пока сериализуется через CPU.
Это не резидентная million-cell реализация.

### Изотермическая связь инерционной плёнки с PR-запасами

`PRIsothermalCapillaryFilm` (`pr_isothermal_capillary_film_v1`) связывает
механику с действительными `PRRelaxationPhase` по каждой ячейке. Обязательны
PR-fluid, совместная механическая модель, `reference_pressure_pa`,
`temperature_k` и единый `mole_fractions`. Механическая плотность должна
совпадать с вычисленной PR-плотностью; произвольная подмена отклоняется.

Это **несжимаемая изотермическая ветвь с PR-калорикой**, а не полный
сжимаемый газожидкостный решатель. Состав однороден, температура поддерживается
термостатом, внешнее давление постоянно. Сохранённое PR-давление является
термодинамическим опорным давлением. Динамическая капиллярная часть входит
в механику отдельно. Проверяется локальная линеаризованная оценка изменения
плотности `(abs(chemical_potential)+rho*u²)/K_T`, где изотермический модуль
`K_T` вычислен из PR. `maximum_density_response` по умолчанию 1e-4, увеличение
выше .01 запрещено. Эта оценка ограничивает область приближения; она не
является экспериментальной оценкой точности жидкости.

Компонентные потоки следуют тому же объёмному потоку, что и механика.
Средний импульс PR-ячейки берётся из двух соседних гранных скоростей.
Кинетическая энергия, потерянная при таком осреднении, сохраняется как
вычисляемая неотрицательная величина:

```text
K_variance_i = rho*A*h_i*(u_right-u_left)²/8
K_mechanical = sum(K_PR_mean) + sum(K_variance_i)
E_joint = sum(E_PR) + sum(K_variance_i) + F + E_thermostat
```

Средняя кинетическая энергия не суммируется повторно. Термостат получает
вязкое и численное тепло по отдельным журналам; оно не прибавляется
повторно к внутренней энергии изотермической PR-фазы. Проверяются локальные
компонентные, импульсные и энергетические потоки, а также общий бюджет.
Checkpoint содержит PR-запасы и механическое состояние.

`face_total_energy_flux_w` использует явно записанную конвенцию
`uniform_exterior_pressure_enthalpy_and_surface_work_cancelled`: работа
постоянного внешнего давления сокращена с соответствующей частью потока
энтальпии. Такой поток нельзя напрямую вставлять в произвольный узел с
меняющимся газовым давлением без восстановления сопряжённой работы.
Старый `transfer_film_increment` явно отклоняет инерционную модель, чтобы
не потерять её кинетическую энергию и не учесть капиллярную работу дважды.

Примеры и общая матрица проверок:

```bash
PYTHONPATH=src:scripts python scripts/validate_pr_surface_matrix.py \
  --case capillary_backward --case capillary_roe \
  --case pr_capillary --case pr_capillary_viscous \
  --device cpu --output /tmp/capillary-cpu
```

Для CUDA следует явно выбрать устройство через `CUDA_VISIBLE_DEVICES`.
Матрица запускает `run_inertial_capillary_film.py` и независимый
`check_inertial_capillary_film.py`. CPU/CUDA-сверка использует единицы полей:
для толщины абсолютный допуск 1e-17 м, для энергии PR — 64 ULP + 1e-22 Дж.
Сравниваются все принятые состояния. Это включение в **матрицу верификации**;
гидравлический адаптер к `PRActiveSurfaceNetwork` с активным газом и
произвольными входами/выходами пока не реализован.

В контролируемом тесте на одной сетке за 80 шагов `backward` сохранил
0.970318 начальной амплитуды, `roe` — практически 1. Оба потребовали
160 итераций Newton. Это сравнение временных схем при фиксированной
пространственной ошибке: частота на сетке из 16 ячеек отличается от
континуального длинноволнового предела примерно на 1.28%. Независимые
проверки уточнения подтверждают второй порядок по времени и пространству.
Это не сравнение с экспериментом или с промышленными моделерами.

Сводка четырёх CPU/CUDA-сценариев, 410 независимо проверенных подшагов,
12943 сравнений значений состояния и хеши:
`benchmarks/results/inertial_capillary_validation.json`.
Открытые задачи — инерция/работа активного газа, неравновесный состав и
теплообмен, физические граничные условия/узлы общего графа и внешняя
квалификация интенсивностей уноса/осаждения.

### Локальный двусторонний обмен газа с инерционной плёнкой

`PRCapillaryGasExchange(film).apply(film_state, gas_phases, dt_s,
conductance_kg_s)` связывает скорости газа с разнесёнными скоростями
`PRIsothermalCapillaryFilm`. Возвращает новое состояние плёнки, газовые
`PRRelaxationPhase` и локальный баланс источника. Газ имеет собственные
N/V/M/E и восстанавливаемые PR-давление/температуру на vapor-ветви.
Составы газа и плёнки могут различаться; обмена веществ в этом операторе нет.

**Геометрический контракт:** один физический запас газа на каждую двойственную
ячейку плёнки. Это не интерполяционная копия газа из основной ячейки сети:
нельзя передавать одни и те же запасы в два оператора и считать их дважды.
Во время источника высоты плёнки и объёмы газа фиксированы. Внешний интегратор
отвечает за консервативный перенос между такими ячейками, нормальную работу
давления и согласование газового объёма при движении поверхности.

Пусть `mg`, `ml` — массы газа и плёнки в двойственной ячейке,
`mr = mg*ml/(mg+ml)`, `a = dt*k/mr`. Для заданной неотрицательной
проводимости `k` [кг/с] неявный импульс равен
`J = mr*a/(1+a)*(ug-ul)`. Газ получает `-J`, плёнка `+J`.
Интерфейс движется с новой plug-скоростью плёнки `ul_new`, работа равна
`W = J*ul_new`. Газовая полная энергия уменьшается на W.
Внутренняя энергия газа возрастает на физическое тепло
`J*(ug-ul)/(1+a)` и отдельно учитываемое численное тепло `J²/(2*mg)`.
Численное тепло плёнки `J²/(2*ml)` поступает в её единственный термостат.
Никакого повторного добавления работы в энергию PR-жидкости нет.

Проверяются локальные тождества кинетической работы, суммарные импульс и
энергия, допустимость PR-ветви и согласованность входных N/V/M/E с P/T.
Недопустимое состояние отклоняется без изменения входов. Коэффициент k
явно задаётся вызывающей стороной: этот API не объявляет его измеренным или
экспериментально подтверждённым. При общем переносе скоростей импульс и
тепло не меняются, а работа преобразуется на J, умноженное на скорость
переноса системы отсчёта.

Оператор **не продвигает часы**. Его нужно вызывать один раз за соответствующий
подшаг расщепления транспортной схемы, а не дополнительно к уже учтённому
межфазному трению. Это локальный касательный источник, а не законченная
газожидкостная труба: нормальная работа давления, теплообмен с изменением
температуры жидкости, испарение/конденсация, открытые граничные условия и
подключение к произвольному графу остаются отдельными задачами.

### Неявная работа PR-газа при движении плёнки

`PRGasLoadedCapillaryFilm` задаёт отдельные замкнутые газовые камеры над
материальными ячейками периодической жидкой плёнки. Объём камеры равен
`Vg_i = cell_area*(ceiling_height_i-h_i)`, число молей и состав в каждой
камере постоянны. Температуры газа и жидкости независимо поддерживаются
термостатами. Камерные перегородки воспринимают горизонтальную реакцию.
Это физический контрольный режим податливого давления, **не проточный
газовый канал** и не замена `PRActiveSurfaceNetwork`.

Газовое давление входит в тот же Newton, который решает толщину и импульс
жидкости. Используется среднее давление по изменению объёма, полученное
аналитическим интегрированием PR при фиксированных N/T:
`p_bar = integral(P(V), V_old, V_new)/(V_new-V_old)`.
При нулевом изменении объёма используется аналитическое продолжение
`log1p(x)/x`, включая производную для автоматического дифференцирования.
Работа не вычисляется вычитанием двух больших абсолютных свободных энергий:
`delta_Fg_i = cell_area*delta_h_i*p_bar_i`.
В дискретном химическом потенциале жидкости добавляется `p_bar_i`;
соответствующая работа включена в локальные потоки энергии и импульса.

После совместного шага газовые N/V/U задают допустимую vapor-ветвь PR.
Газовый термостат получает `delta_Fg-delta_Ug`, что соответствует `-T*delta_Sg`.
Сумма внутренних энергий, кинетической энергии, энергии осреднения,
капиллярной/гравитационной энергии и обоих термостатов сохраняется.
Физическое стеночное и численное тепло жидкости учитываются прежним способом.
Реакция перегородок выдаётся как `support_impulse_kg_m_s`; она не записывается
как вымышленный продольный импульс неподвижного газа.

Контракт `normal_load` низкоуровневого механического шага предназначен для
этого сериализуемого адаптера: один механический checkpoint без модели нагрузки
недостаточен для воспроизведения нагруженной задачи. Полное состояние
`PRGasLoadedCapillaryState` сохраняет геометрию через идентификатор модели,
плёнку, действительные газовые запасы и энергию газового термостата.
Недопустимый объём, PR-ветвь, превышение оценки несжимаемости жидкости или
несходимость приводят к отказу; `advance` дробит шаг и проходит весь отчётный
интервал. Скрытого ограничения объёма или коррекции запасов нет.

```bash
PYTHONPATH=src:scripts python scripts/validate_pr_surface_matrix.py \
  --case pr_gas_loaded_capillary --device cpu --output /tmp/gas-loaded-cpu
```

Конфигурация: `configs/surface_pr_gas_loaded_capillary.json`.
Независимый аудитор использует NumPy, 24-точечную квадратуру PR-давления и
скалярную PR-калорику. Он проверяет перенос объёма, местный импульс, работу,
запасы и полный баланс энергии по траектории. Дополнительно проверены
аналитическая частота малой волны с газовой упругостью, энтропийный эквивалент
тепла термостата, рестарт и отказы при повреждении отчёта.

Камеры используют **основные** ячейки, а касательный оператор
`PRCapillaryGasExchange` — физические запасы в **двойственных** ячейках.
Автоматически объединять эти разные геометрии нельзя: сначала нужен
консервативный газовый транспорт на согласованной сетке. Открытые патрубки,
неизотермический газ, фазовый обмен и экспериментальные законы уноса остаются
за пределами этой ветви.

Сводка текущей проверки: `benchmarks/results/pr_gas_loaded_capillary_validation.json` —
154 теста, пять CPU/CUDA-сценариев, 430 независимо проверенных подшагов
и 16063 сравнения значений состояния. Хеши исходников и тестов привязаны
к этой версии; более ранние сводки остаются свидетельствами своих версий.

### Открытый активный PR-граф: проток, тепло и фазовый обмен

`PRActiveSurfaceNetwork` версии `pr_active_surface_v3` принимает
`reservoirs: dict[str, PRActiveReservoir]`. Колонна больше не обязательна:
допустимы сети активных труб между открытыми границами и реальные конечные
узлы оборудования без фиктивного сепаратора. Старые v1/v2-конфигурации
сохраняют прежнюю сериализацию при отсутствии резервуаров.

`PRActiveReservoir(phases, phase_roots, elevation_m=0)` задаёт поддерживаемое
состояние внешнего источника/приёмника. Каждая фаза имеет действительные
N/V/M/E/P/T, явную PR-ветвь и скорость из своего импульса. Объёмы определяют
входное удержание фаз; они **не входят** в конечные запасы сети. Состояние
проверяется на EOS, допустимую ветвь и совместимость со слотами трубы.
Отметка резервуара должна совпадать с отметкой соответствующего торца трубы.
Один резервуар может питать несколько ветвей; все их расходы суммируются.

Граничный поток вычисляет тот же Riemann-оператор активных фаз, что и
внутренний перенос трубы. Передаются компоненты, импульс, полная энергия
с работой давления и энтропия. Направление не зафиксировано: возможен выход
в резервуар и обратный вход. Это поддерживаемая внешняя Riemann-граница,
а не автоматически неотражающий выход и не универсальная модель скважины.
Заданные P/T/u резервуара поддерживаются внешней инфраструктурой.

`reservoir_exchange[name]` — интегральные входящие в сеть компоненты [моль]
и абсолютная энергия [Дж] за принятый шаг. Положительное значение означает
поступление. `external_component_moles` суммирует компоненты всех открытых
границ; `external_energy_j` включает их энергию и независимые внешние
тепловые/механические воздействия. Потенциальная энергия использует отметку
торца и общий гравитационный датум. При ручном вызове `ledger` открытой сети
нужно передавать оба внешних баланса; `advance` и штатный runner суммируют
их автоматически по всему отчётному интервалу.

Изоляция задаётся `port_openings={reservoir_name: false}`. Тогда торец
использует стеночную границу, источник отключён и его граничный перенос
строго не добавляется в внешний баланс. Дробная `port_positions` остаётся
контрактом конечных патрубков колонн; к резервуарам она не применяется.

Узлы разветвления задаются существующими конечными сосудами
`PREquilibriumTransportNetwork`, подключёнными к нескольким трубам.
Их физический объём и запасы изменяются от суммы потоков всех ветвей.
Безразмерный узел без запасов или искусственный промежуточный объём не
создаётся. Сосуды используют равновесный UV-flash; активные фазы труб
сохраняют собственные температуры, составы и скорости.

Неизотермический фазовый обмен включается у трубы через
`thermochemical_relaxation_time_s`: после переноса и внешнего нагрева
применяется консервативная термохимическая релаксация по N/V/M/E.
Скрытая теплота учитывается PR-калорикой, а не отдельным добавлением энергии.
Появившаяся фаза становится обычным переносимым запасом следующего подшага.
Конечное время релаксации — заданный параметр модели, **не измеренный закон
кинетики испарения/конденсации**. Полная схема использует расщепление
транспорт/нагрев/релаксация, а не монолитный неявный solve всей сети.

При появлении второй фазы ненулевой скалярный нагрев неоднозначен и
отклоняется. Для непрерывного нагрева с фазовым переходом задавайте явные
фазовые мощности и температуры тепловых источников, например:

```json
{
  "pipe_heat_w": {"line": [{"liquid": 100}, {"liquid": 100}]},
  "pipe_heat_temperatures_k": {"line": [{"liquid": 400}, {"liquid": 400}]}
}
```

Это нагрев именно жидкости; он не перераспределяется самопроизвольно после
рождения газа. Исчезновение фазы, для которой задан ненулевой отбор или
подвод тепла, требует изменения управляющей команды.

Контрольные конфигурации:

- `surface_pr_active_surface_open_gas.json`: поддерживаемый проток газа;
- `surface_pr_active_surface_open_evaporation.json`: открытый перенос,
  нагрев жидкости и конечная по времени генерация газа в одном расчёте;
- `surface_pr_active_surface_open_branch.json`: вход, реальный конечный
  сосуд и две выходные ветви с общим внешним приёмником.

Они запускаются через `validate_pr_surface_matrix.py` с именами
`active_surface_open_gas`, `active_surface_open_evaporation` и
`active_surface_open_branch`. Независимый CPU-аудитор пересчитывает
64-точечные фазовые пути, скалярную PR-калорику, фазовые источники и
внешний перенос из проверенных торцевых потоков. Дополнительно проверяются
реверс, изоляция, неравные вход/выход, рестарт и повреждение отчёта.

Эта интеграция относится к **активному PR-графу**. Капиллярная модель
разрешённой плёнки остаётся отдельной: переносить её энергию поверхности и
геометрию в данный граф автоматически нельзя. Наличие открытого газа и
фазового обмена в графе само по себе не подтверждает совместную
капиллярную многофазную модель или экспериментальные коэффициенты обмена.

Сводка этой интеграции: `benchmarks/results/pr_open_surface_validation.json`.
Затронутая регрессия — 133 теста; пять функциональных сценариев — 26
независимо проверенных подшагов CPU/CUDA. Для трёх новых открытых сценариев
сверены начальные, все принятые и конечные состояния: 454 скалярных значения
с допусками по физическим единицам. Проверку всей траектории воспроизводит
`check_pr_open_graph_history.py CPU_DIRECTORY CUDA_DIRECTORY --output FILE`.
Это короткие физические/функциональные проверки, не замер производительности
крупного графа и не экспериментальная квалификация.

### Совместный капиллярный, тепловой и компонентный источник

`PRCapillaryPressure` решает нормальный межфазный обмен непосредственно на
`PRActiveMultifluidState`. Это дополнительный физический оператор, **не
замена осевого транспорта**. Он использует действительные запасы фаз в трубе:
отдельные составы, объёмы, импульсы, полные энергии и температуры.

Для фиксированной покрытой плёнки принимается малый уклон и постоянная σ.
Высота `h_i = V_liquid_i/(width*length_i)` определяет избыточную энергию
поверхности относительно неизменного плоского интерфейса:
`F = sum_edges 0.5*σ*width/distance*(h_j-h_i)^2`.
Постоянная энергия плоского интерфейса опущена только потому, что топология
и площадь его покрытия в этом операторе не меняются. Периодическая сетка
замыкается; непериодическая имеет естественное окончание этой энергии без
заданных внешних капиллярных усилий. Это **не** капиллярная модель патрубка.
Максимальный уклон ограничен явно, по умолчанию 0.2.

Дискретный градиент F берётся в среднем объёме старого и нового состояния.
Поэтому `sum(mu_capillary_i*delta_V_liquid_i) = delta_F` с точностью округления,
включая неравномерную сетку. Неизвестные одного Newton — фазовые объёмы,
температуры и, при включённой кинетике, переносимые количества компонентов.
PR-давления получаются из текущих N/V/T, а не проецируются к общему давлению.

Нормальное движение задаёт положительная подвижность C [м³/(Па·с)]:
`delta_V = dt*C*(p_liquid-p_gas-mu_capillary)`.
Его диссипация `H = delta_V²/(dt*C)` неотрицательна. Жидкость совершает работу
`-p_liquid*delta_V`, газ получает `+p_gas*delta_V`; распределение H между
фазами задано `liquid_heat_fraction`. Суммарная объёмная энергия меняется
ровно на `-delta_F`. Отдельной «компенсации энергии» после решения нет.

Дополнительно поддерживаются:

- `heat_conductance_w_k`: поток `Q = dt*K*(T_gas-T_liquid)`;
- `component_mobility_mol2_k_j_s`: неотрицательные компонентные подвижности;
- `transfer_energy_j_mol`: явно заданные энергии переноса каждого компонента.

Положительный перенос компонента направлен из газа в жидкость. Вместе с
молями передаётся импульс с межфазной скоростью, равной среднему старых
скоростей фаз; полная переносимая энергия задаётся явно. Химическая движущая
сила включает PR-химические потенциалы, температуры и кинетические члены,
согласованные с этим переносом импульса. Это сохраняет компоненты,
суммарный импульс и энергию, включая скрытую теплоту в PR-калорике.
Подвижности и энергии переноса — параметры замыкания, **не калиброванные
константы оборудования или экспериментально подтверждённая кинетика**.

При положительной химической подвижности соответствующий компонент уже
должен присутствовать в обеих фазах. Рождение и исчезновение фаз, зарождение
нового интерфейса и изменение площади покрытия здесь не реализованы.
Искусственные остаточные моли, отсечение объёмов и добавочные термостаты
не используются. Две фазы при нулевом теплообмене могут сохранять разные
температуры. Проверяется рост физической энтропии и его нижняя граница от
нормального, теплового и химического обмена; допуски привязаны к ULP запасов.

На уровне графа доступны `capillary_totals(state, sources)` и
`relax_capillary(state, dt_s, sources)`, где sources — отображение имени трубы
в `PRCapillaryPressure`. Транзакция работает на живых запасах графа, добавляет
F в общий энергетический баланс и не продвигает часы. Параметры источника
сохраняются в его отчёте; для воспроизведения нужны и граф, и эти параметры.
Она **не вызывает** обычный `step` и не закрывает осевой перенос F через торцы
и узлы. Последовательный вызов `relax_capillary` и прежнего `step` нельзя
объявлять готовым капиллярным расчётом сети: фазовый обмен старого step также
ещё не согласован с изменением этой поверхности.

Воспроизводимые операторные проверки:

```bash
PYTHONPATH=src:scripts python scripts/run_pr_capillary_pressure.py \
  configs/surface_pr_capillary_pressure_chemical.json \
  --output /tmp/capillary-source.json
PYTHONPATH=src:scripts python scripts/check_pr_capillary_pressure.py \
  configs/surface_pr_capillary_pressure_chemical.json /tmp/capillary-source.json
```

Вторая конфигурация `surface_pr_capillary_pressure_normal.json` отключает
тепловой и компонентный обмен. Независимый аудитор использует скалярную PR-EOS
и фугитивности: проверяет составы, объёмы, импульсы, работу, энергии,
химические движущие силы и энтропию. Производные химических потенциалов
дополнительно сверены с конечными разностями независимой энергии Гельмгольца.
Это проверка источника на фиксированной двухфазной топологии, не валидация
целой капиллярной трубы и не экспериментальная квалификация.

Сводка источника: `benchmarks/results/pr_capillary_pressure_validation.json`.
Проверены 148 тестов затронутой регрессии с CUDA; два воспроизводимых
операторных сценария прошли независимый аудит на CPU и RTX 3090 Ti
(16 ячеечных транзакций). Сверка устройств охватывает 342 значения
начальных/полученных/конечных состояний с допусками по единицам полей.
Это число сравнений, а не число независимых временных шагов сети.

### Прямые продолжения трубы и общая капиллярная геометрия

`PRActivePipeContinuation(left_pipe, right_pipe)` соединяет выход первой трубы
со входом второй как общую грань сетки. В конфигурации
`pr_active_surface_v4` отображение `continuations` хранит имена этих соединений;
`boundary_nodes` обеих труб ссылаются на одно имя. Дополнительный сосуд,
объём или запас вещества не вводится. Требуются одинаковые сечение, реестр
фаз, граница волновой скорости и совпадающая отметка торцов. Это продолжение
одной линии, не модель тройника, поворота или перехода диаметра.

Обычный `step` использует реальные соседние ячейки, проверяет совпадение
торцевых потоков и неконсервативных путей. Продолжение допускает реверс,
нагрев и рождение фазы обычного PR-оператора. `port_openings` закрывает обе
стороны общей грани. Кольцо допускается без внешнего резервуара.

Для покрытой двухфазной поверхности доступны
`capillary_chain_totals(state, pipe_order, parameters)` и
`relax_capillary_chain(state, dt_s, pipe_order, parameters, port_openings=...)`.
Порядок должен охватывать всю компоненту прямых продолжений. Нормальные,
тепловые и химические неизвестные решаются совместно; производные энергии
поверхности включают стык. Кольцо замыкает последнюю геометрическую грань.
Половина энергии каждой грани относится к каждому соседнему сегменту.
Отчёт содержит противоположные работы стыка и отдельный энергетический
баланс каждого сегмента. Сила в отчёте сопряжена высоте поверхности;
это не осевая сила импульсного уравнения.

Если переданы закрытые продолжения, источник отказывается от операции:
разрыв поверхности требует отдельного граничного геометрического замыкания.
Команды открытия здесь относятся только к продолжениям. По умолчанию они
открыты; вызывающий код должен передавать актуальные команды изоляции.
Источник не меняет часы и не запускается автоматически обычным `step`.

Воспроизводимый запуск с независимым скалярным PR-аудитом:

```bash
PYTHONPATH=src:scripts python scripts/run_pr_capillary_chain.py \
  configs/surface_pr_capillary_continuation_line.json \
  --output /tmp/pr-capillary-line.json
PYTHONPATH=src:scripts python scripts/run_pr_capillary_chain.py \
  configs/surface_pr_capillary_continuation_loop.json \
  --device cuda --output /tmp/pr-capillary-loop.json
```

Аудитор `check_pr_capillary_chain.audit` сверяет реальные запасы графа,
PR-термодинамику источника, геометрию, знаки работ и локальные энергетические
балансы независимо от рабочих функций капиллярной силы. Проверки включают
повреждение отчёта, рестарт и сравнение с неразрезанной трубой.
Это проверка нормального источника на фиксированной покрытой топологии.
Осевые капиллярные напряжения и перенос поверхностной энергии, ветвящиеся
капиллярные узлы, рождение/исчезновение самой поверхности и экспериментальная
квалификация остаются открытыми. Последовательность старого транспортного
шага и этого источника не считается готовым капиллярным интегратором.

Проверка этой версии: 162 теста затронутой регрессии прошли на RTX 3090 Ti;
линия и кольцо отдельно прошли независимый аудит на CPU и CUDA. Сверены
240 скалярных значений начальных и конечных состояний с допусками по единицам
полей. Сводка с хешами исходников и конфигураций:
`benchmarks/results/pr_capillary_continuation_validation.json`.
Это малые операторные проверки; они не измеряют точность полевого прогноза
или скорость крупной сети.

### Дискретные осевые капиллярные порты работы

`PRCapillaryTransportWork(source).evaluate(before, after,
integrated_liquid_flux_m3, volume_source_m3=...)` вычисляет сопряжённые
работы для **заданных** двух состояний и интегральных объёмных потоков.
Это конститутивный оператор для будущего совместного solve; он не меняет
состояние, не решает импульс и не превращает обычный `step` в капиллярный.
Геометрия остаётся покрытой двухфазной поверхностью с малым наклоном.

Для ячейки `i` проверяется фактическое равенство
`delta_V_i = q_left - q_right + s_i`. Источник `s_i` должен быть задан
независимо; он не восстанавливается из невязки. В периодической сетке грань
`i` направлена из ячейки `i` в `i+1`; в непериодической сетке есть `n+1`
граней слева направо, и торцевые потоки должны быть нулевыми. Открытый торец
требует внешней геометрии поверхности и пока отклоняется.

При `h_i=V_i/(width*length_i)` потенциал `mu_i` — производная квадратичной
энергии поверхности по `V_i`, вычисленная на средней геометрии между двумя
состояниями. Для грани `i→j` с весом `k=sigma*width/distance`:

```text
W_bulk_face = -q * (mu_j - mu_i)
W_shape_into_left = k * (h_mid_j-h_mid_i) * (delta_h_i+delta_h_j)/2
J_surface = (mu_i+mu_j)*q/2 - W_shape_into_left
W_normal_i = -mu_i*s_i
```

Половина работы грани относится к каждой соседней ячейке. Половина энергии
геометрической грани также относится к каждой ячейке. Тогда локально:
`delta_F_i + W_axial_i + W_normal_i = J_left - J_right`.
`J_surface` имеет единицы Дж за интервал, а отчётный поток — Вт.
Равенство проверяется до возврата результата; никакая энергия не добавляется
в термодинамические запасы для устранения дефекта.

`network.capillary_chain_transport_work(before, after, pipe_order, parameters,
integrated_liquid_flux_m3, volume_source_m3=..., port_openings=...)`
собирает эти работы на реальных ячейках продолженной линии/кольца и выдаёт
противоположные энергетические порты стыков, плюс баланс каждого сегмента.
Разбиение одной линии на сегменты не меняет результат. Геометрические
вычисления граней векторизованы средствами PyTorch, включая CUDA.

Независимый скалярный аудитор `check_pr_capillary_transport.audit` проверяет
геометрические производные и локальные работы без рабочих функций силы.
Тесты охватывают неравномерную сетку, оба направления потока, независимые
объёмные источники, периодический и закрытый контуры, стыки, конечные
разности энергии и повреждённые отчёты. Это проверка конститутивного
тождества: заданные состояния тестов не являются результатом нового
совместного гидродинамического интегратора.

**Для общего открытого/ветвящегося графа остаются:** уравнения импульса и термодинамическое обновление,
согласованные с этими работами и транспортной диссипацией; внешнюю геометрию
открытых торцов; ветвящиеся поверхности; рождение/исчезновение поверхности.
Капиллярная часть общего динамического графа пока не считается завершённой.

Подтверждение этой версии: 179 тестов затронутой регрессии на RTX 3090 Ti
и отдельная аналитическая проверка предела Лапласа. На сетках 16/32/64
ячейки относительная ошибка потенциала относительно непрерывной гармоники
равна 0.012785/0.003209/0.000803; отношения ошибок 3.985 и 3.996 подтверждают
второй порядок пространственного геометрического оператора. Это пространственная проверка; она не определяет порядок
точности совместного временного интегратора.
Сводка: `benchmarks/results/pr_capillary_transport_validation.json`.

### Совместная осевая PR-капиллярная динамика замкнутой линии

`PRCapillaryAxialPipe(source)` — новый динамический оператор, в отличие от
`PRCapillaryTransportWork`: он действительно обновляет компонентные запасы,
объёмы, импульсы и температуры двух фаз. Область реализации пока ограничена
равномерной периодической линией либо линией с отражающими торцами,
без гравитации, с положительными запасами
обеих фаз и всех компонентов. Порядок слотов — жидкость, газ. Химический
и фрикционный split обычной трубы должен быть выключен. Плотный совместный
Newton ограничен `maximum_unknowns` (по умолчанию 192); при двух компонентах
нужно девять неизвестных на ячейку. Это ограниченная новая ветвь, не замена
масштабируемого пластового FIM и не универсальный наземный граф.

В `step(state, dt_s)` одновременно решаются компонентные, объёмное,
импульсные и энергетические уравнения. Калорика и давление вычисляются
по PR на текущем приближении, геометрия объёмных долей — на среднем между
начальным и конечным состояниями. Консервативные потоки используют явную
акустическую вязкость Rusanov, межфазные пути — BN-замыкание. Совместная
временная схема имеет первый порядок; это не прежний явный `pipe.step`.

Капиллярный потенциал — дискретный градиент поверхностной энергии. Если
`B(alpha_mid)*u_end` задаёт физическую скорость изменения объёма жидкости,
сила определяется сопряжением `F_cap=-B.T*mu`. Тогда работа силы точно
согласована с изменением поверхности. Для равномерного периодического
оператора суммарная капиллярная сила равна нулю. Отдельный вклад акустической
диффузии объёма даёт `Q_cap=-mu*delta_V_diff`; это производное неотрицательное
**численное** тепло, явно записанное в отчёт, а не физическое трение и не
подгонка глобальной невязки. Доля тепла между фазами задаётся `liquid_heat_fraction`.
Неявная временная схема также может давать численную диссипацию; не весь
наблюдаемый нагрев следует трактовать как физический.

Каждый принятый шаг проверяет компоненты, импульс с учётом воздействия
торцевых стенок, сумму объёмной и
поверхностной энергии и неубывание PR-энтропии. Для энтропии применяется
локальный допуск по ULP и абсолютный запас `1e-22 J/K`, а не общий допуск
`1e-7 J/K`, слишком большой для микроскопической контрольной задачи.
Newton использует положительную параметризацию запасов/температур и
ограниченную долю объёма; line search отклоняет нечисловые и недопустимые
PR-состояния. Корень EOS дополнительно подтверждается восстановлением
реальных конечных запасов.

`step_coupled` выполняет осевой шаг, затем `source.apply` на тех же запасах
с тем же функционалом поверхности: нормальный, тепловой и компонентный
обмен, включая латентную калорику PR. Это split первого порядка, а не
единый Newton для обоих операторов. `advance(..., normal_relaxation=True)`
дробит неудачные попытки и достигает отчётного времени точно. Ошибочные
попытки не меняют исходное состояние. При коротком нормальном подшаге
сходимость дополнительно ограничена локальной точностью энергетических
запасов фаз; одной масштабированной общей нормы недостаточно.

`network.advance_capillary_loop(state, duration_s, pipe_order, parameters,
initial_step_s=..., normal_relaxation=True)` переносит этот интегратор на
полную изолированную петлю прямых продолжений: без внешних резервуаров,
сосудов и оборудования. Ячейки лишь собираются в общий solve и возвращаются
в исходные сегменты с новыми запасами и общими часами. Новых физических
объёмов нет. Разрезанная и неразрезанная петли дают одинаковые состояния;
рестарт хранит исходные реальные запасы и конфигурацию модели.

```bash
PYTHONPATH=src:scripts python scripts/run_pr_capillary_axial.py \
  configs/surface_pr_capillary_axial_loop.json \
  --output /tmp/pr-capillary-axial-cpu.json
```

Runner независимо проверяет каждый осевой и нормальный подшаг. Осевой
аудитор использует скалярную PR-EOS, 64-точечную квадратуру BN и силу,
восстановленную через независимую виртуальную работу объёмного оператора.
Проверяются все уравнения и глобальные балансы, не только итоговая энергия.
Контрольный сценарий микроскопический: его секунды и расходы не являются
полевым прогнозом или бенчмарком масштаба сети.

За пределами этой линейной модели остаются неравномерные линии,
появление и исчезновение самой поверхности,
гравитационное сопряжение этой новой ветви и масштабируемая алгебра.
Обычные открытые PR-трубы, оборудование и графы сохраняют свои прежние
возможности, но автоматически капиллярными от этого не становятся.

Сводка динамической ветви: `benchmarks/results/pr_capillary_axial_validation.json`.
В регрессии проверены 186 тестов на RTX 3090 Ti. Воспроизводимый сценарий
на CPU и CUDA проходит три принятых подшага после двух отказов, достигает
`1e-10 s` и проходит независимый аудит каждого осевого и нормального подшага.
Сверка всех начальных/промежуточных/конечных состояний охватывает 456
скалярных значений с допусками по единицам полей. Проверка разрезанной
петли дополнительно подтверждает рестарт и перенос `next_step_s` между
интервалами. Числа относятся к малой контрольной задаче, не к масштабу
промышленной сети и не к экспериментальной валидации замыканий.


### Отражающие торцы и изолированная цепочка сегментов

Совместный `PRCapillaryAxialPipe` теперь поддерживает `boundary_mode="walls"`
с отражающими торцами. У торца отражаются скорость и импульс, а остальные
поля зеркальной ячейки берутся из реальной крайней ячейки. Компонентный
и энергетический потоки через закрытый торец равны нулю. Импульс стенки
вычисляется из тех же численных торцевых потоков, записывается отдельно и
включается в баланс импульса. Геометрическая энергия имеет естественное
нулевое граничное условие на наклон, без периодической замыкающей грани.
Это не замыкание с заданным контактным углом или энергией смачивания торца.

Геометрические производные, сопряжённые силы и численное капиллярное тепло
используют только реальные внутренние грани. Для равномерной линии остаётся
неотрицательным производное капиллярное тепло; полная энергия не уходит
через отражающие торцы. Независимый аудитор строит отражённые состояния
самостоятельно, восстанавливает торцевой импульс и отклоняет повреждённый
отчёт о нём. Проверка массы/энергии не заменяет проверку импульса стенки.

`network.advance_capillary_line` считает всю изолированную равномерную
цепочку либо кольцо. Внешние `boundary_nodes` изолированной цепочки равны
`None`; внутренние соединения — обычные `PRActivePipeContinuation`.
`advance_capillary_loop` сохраняет прежний контракт и требует настоящей
замыкающей грани. Нормальный обмен, точные часы, перенос адаптивного шага
и рестарт используют те же реальные запасы, что и осевой интегратор.

```bash
PYTHONPATH=src:scripts python scripts/run_pr_capillary_axial.py   configs/surface_pr_capillary_axial_reflecting.json   --output /tmp/pr-capillary-reflecting.json
```

Это поддержка неизменной изолированной геометрии. Мгновенное закрытие
внутреннего стыка не реализуется удалением его поверхностной энергии:
команда закрытого продолжения по-прежнему отклоняется капиллярной сборкой.
Для такого переключения ещё нужен согласованный учёт работы привода и
изменения поверхностной геометрии. Для конечных узлов такой учёт теперь
реализован отдельным `PRCapillaryBranchNetwork`, описанным ниже.
Изменение фазовой топологии остаётся незакрытым режимом.

Подтверждение отражающих торцов:
`benchmarks/results/pr_capillary_reflecting_validation.json` — 189 тестов
затронутой регрессии на RTX 3090 Ti. Сценарий с закрытыми торцами на CPU и
CUDA достигает `1e-10 s` за два принятых подшага после одного отказа;
каждый осевой и нормальный подшаг проверен независимо. Сверены 342
скалярных значения состояний. Полный импульс стенки хранится отдельно
для каждого осевого подшага. Это малая контрольная задача с заданными
замыканиями; она не квалифицирует реальные контактные углы, переключение
арматуры или промышленный масштаб сети.

### Открытый капиллярный проток и общий потенциал смачивания

`PRCapillaryAxialPipe(source, boundaries=(left, right))` принимает реальные
PR-состояния обеих фаз на поддерживаемых внешних гранях. Каждая граница —
отображение имён фаз в `PRRelaxationPhase`, либо `None` для отражающего торца.
Внешние состояния задают P/T/состав/скорость и геометрию грани, но не добавляют
внутреннего физического запаса. Объёмная доля зеркальной ячейки экстраполируется
из фиксированного значения на грани и среднего внутреннего значения;
неположительная поддержка отклоняется. Компоненты, энтальпия, импульс,
энтропия и капиллярная граничная работа выводятся явно и независимо проверяются.
В активном графе этот режим доступен через `advance_capillary_open_line`.

Опциональный `wetting` у `PRCapillaryPressure` задаёт
`adsorption_thickness_m` и `equilibrium_contact_angle_deg`. Используется
существующий материальный закон, общий с `ThinFilmDynamics`:

```text
C = (10/3) * sigma * (1 - cos(theta))
Phi(h) = C * (-(h_ads/h)^2/2 + (h_ads/h)^5/5)
F_total = F_strip + sum(A_wall * Phi(h))
mu_wall = [Phi(h_new)-Phi(h_old)] / (h_new-h_old)
```

Разделённая разность вычисляется аналитически без потери точности на
совпадающих толщинах; независимый аудитор интегрирует производную 64-точечной
квадратурой. При `theta=0` потенциал точно нулевой, даже при очень малой
толщине: ненужные большие степени не вычисляются. При `theta>0` нужен
реальный положительный запас плёнки. Формула не задаёт полностью сухое
состояние, нуклеацию или динамический контактный угол.

Потенциал невыпуклый. Для совместного осевого solve нужно явно выбрать
`volume_diffusion="none"`: объёмная акустическая диффузия отключается,
акустическая вязкость N/M/E действует на скачки внутренних фазовых
плотностей, умноженные на среднюю объёмную долю. Энтропийный отказ остаётся
обязательным. Это отдельный вариант первого порядка, без гарантии
устойчивости для произвольного шага. При отсутствии материального потенциала
прежний вариант `volume_diffusion="acoustic"` остаётся по умолчанию.

На фиксированной сетке нелинейный стеночный потенциал даёт ненулевую
суммарную дискретную силу даже на периодической однородной стенке.
`material_wall_impulse_kg_m_s` и `material_wall_resultant_fraction` показывают
этот вклад отдельно; он вычисляется из потенциала и объёмного оператора,
а не из невязки баланса. Его нельзя трактовать как калиброванное закрепление
контактной линии. Для гладкого периодического профиля тест 16/32/64/128
подтверждает уменьшение ошибки примерно в четыре раза при удвоении сетки.
Точный нулевой результант геометрической капиллярной силы не означает
точного нулевого результанта этой материальной дискретизации.

```bash
PYTHONPATH=src:scripts python scripts/run_pr_capillary_axial.py \
  configs/surface_pr_capillary_open_wetting.json --output /tmp/pr-open-wetting.json
PYTHONPATH=src:scripts python scripts/run_pr_film_mist_exchange.py \
  configs/surface_pr_film_mist_wetting.json --output /tmp/pr-mist-wetting.json
PYTHONPATH=src:scripts python scripts/check_pr_film_mist_exchange.py \
  configs/surface_pr_film_mist_wetting.json /tmp/pr-mist-wetting.json
```

### Конечные узлы, рециклы и изоляция капиллярного графа

`PRCapillaryBranchNetwork` принимает `pipes`, `junction_volumes_m3`,
`parameters`, опциональные `sources`, `openings`, `volume_diffusion` и бюджет
системы `maximum_unknowns` и выбор `linear_solver`. Труба задаётся объектом `pipe`, парой
`boundary_nodes` и горизонтальным единичным вектором `axis`. Концы привязаны
к именам конечных камер либо закрыты (`None`). Разветвления, параллельные
пути и рециклы используют один набор реальных N/V/M/E. Сборочное
представление не создаёт дополнительного вещества. Камера имеет собственный
объём и термодинамические запасы; её средний импульс равен нулю — это
стационарный идеально перемешанный узел, а не трёхмерная CFD-модель.

Модель графовой поверхности — явно объявленная энергия полосок на гранях
между соседними реальными объёмами. Реакции опор камер и стенок считаются
в физических векторах с использованием `axis`; сумма локальных осевых
импульсов не заменяет векторный баланс. Независимый аудитор восстанавливает
топологию из конфигурации, PR-калорику, пути, капиллярные силы виртуальной
работой и реакции. Для `sources[node]` разрешены фазовые `inlets` с реальным
PR-состоянием и молярным расходом, `withdrawals` с текущим составом/энтальпией
камеры и `heaters` с мощностью/температурой. Внешние N/E/S входят в балансы.

`switch_ports(state, {"pipe:side": bool})` возвращает новую модель, checkpoint
и событие. При изоляции запас камеры или трубы не исчезает. Изменение
геометрической поверхностной энергии учитывается как явная внешняя работа
`external_actuator_work_j`. Это идеальное обратимое переключение граничного
условия, без характеристики реального клапана, контактного мениска или
диссипации конечного привода. Это также не меняет фазовую топологию.

`PRCapillaryBranchState` сохраняет трубы, реальные камеры, общее время,
переносимый шаг и идентификатор конфигурации. Runner поддерживает расписание
изменений `openings`, проверяет каждый осевой/нормальный шаг и событие,
сохраняет хеши исходников и отказывается перезаписывать готовый результат:

```bash
PYTHONPATH=src:scripts python scripts/run_pr_capillary_branch.py \
  configs/surface_pr_capillary_branch_recycle.json --output /tmp/pr-recycle.json
```

Пример содержит две камеры, два параллельных пути, глухую ветвь и
изоляцию/повторное открытие возврата. Все новые конфиги — микроскопические
синтетические контрольные задачи. Общая новая ветвь пока требует двух фаз
и положительных компонентных запасов, общую секцию/реестр, горизонтальную
геометрию без гравитации. Для больших систем теперь есть разреженный solve,
описанный ниже. Она не заменяет
сразу остальные специализированные модели сепараторов и оборудования.
Отсутствие экспериментальных данных не позволяет назвать параметры
смачивания или уноса калиброванными. Исторические сводки выше относятся
к указанным в них хешам исходников, а не автоматически к текущей версии.

Сводка объединённой ветви:
`benchmarks/results/pr_shared_wetting_validation.json`. Проверены 288 тестов
на RTX 3090 Ti; после уточнения текста квалификации отчётов повторены
затронутые проверки. Три новых сценария воспроизведены на CPU и CUDA:
рецикл с изоляцией, открытый капиллярный проток со смачиванием и 65 шагов
локального обмена плёнка–капли. Независимые проверки прошли на обоих
устройствах; сверены 3543 скалярных значения физических состояний с допусками
по единицам величин. Это не замер ускорения: аудиты входят во время runner,
а нагрузки устройств не изолировались для бенчмарка производительности.
Отдельно прошли 37 проверок существующих распределённых сепараторов и
конечных патрубков (всего 325 уникальных тестов в двух наборах). Это
регрессия самостоятельных моделей; она не подтверждает ещё не реализованное
сопряжение этих сепараторов с новой капиллярной ветвью.

### Согласованная временная геометрия осевого переноса

В актуальной схеме `transport_state="midpoint_intrinsic"` объёмная доля
берётся на середине шага, а внутренние фазовые плотности — на конце:

```text
alpha_bar = (alpha_old + alpha_new)/2
q_transport = alpha_bar * q_new/alpha_new
```

Эта величина используется в конвективном и акустическом переносе N/M/E.
Первичные запасы в невязке остаются действительными конечными N/M/E;
коррекция не усредняет сохранённое состояние и не добавляет вещества.
В варианте без объёмной диффузии вязкость использует скачки внутренних
плотностей `q_new/alpha_new`, умноженные на среднюю геометрию грани.
Граничный поток энтропии использует ту же геометрию. Сила капиллярности
остаётся сопряжённой объёмному оператору с `alpha_bar`, поэтому сохраняются
дискретная работа поверхности и геометрический баланс импульса.

Это устраняет прежнюю несогласованность: объём переносился по среднему
удержанию, а масса — по конечному. На резком контакте она создавала
искусственную сжимаемость и могла вызывать ложный отказ по энтропии.
Проверка материального контакта теперь меняет удержание при сохранении
фазовых плотностей, P и T в обоих направлениях. Настоящий отказ по
положительности при слишком большом шаге по-прежнему дробит интервал.
Уточнение не является новой универсальной гарантией энтропийной устойчивости:
обязательные проверки каждого принятого шага сохраняются.

### Явная конечная энергия сухой стенки

`wetting.dry_extension="c2_half_adsorption"` — опциональное материальное
замыкание, разрешённое для локального `PRFilmMistExchange`. При
`x=2*h/h_ads < 1` оно заменяет сингулярную ветвь полиномом

```text
Phi_dry(h) = C * (1230*x^3 - 2210*x^4 + 1002*x^5)/5
C = (10/3) * sigma * (1-cos(theta))
```

При `h >= h_ads/2` остаётся исходный потенциал. В точке стыка совпадают
энергия и первые две производные. В сухой точке энергия и её первые две
производные равны нулю. Это **явно выбранная феноменологическая модель**
с конечным барьером, а не вывод свойств голой стенки из PVT или измеренных
контактных углов. Исходный сингулярный вариант остаётся по умолчанию.

Разделённая разность устойчива на совпадающих толщинах и при пересечении
стыка; независимый аудитор интегрирует производную по частям. Нулевой
жидкий объём не заменяется адсорбционным запасом. Конфиг
`surface_pr_film_dry_cycle.json` удаляет плёнку полностью и затем создаёт
её из действительных капель, с сохранением компонентов, импульса и суммы
объёмной/поверхностной энергии. Проверены CPU и CUDA.

Ограничение существенно: интенсивности обмена заданы извне. Это конечный
энергетический оператор, а не разрешённая динамика преодоления барьера,
молекулярная нуклеация или движение контактной линии. Он не делает
положительную параметризацию совместного осевого интегратора активным
решателем исчезающих фаз. В осевом solve и в связке ниже смена числа фаз
пока требует отдельной реализации.

### Консервативная связь капиллярной трубы с гравитационным сосудом

`PRCapillaryColumnNetwork` связывает реальный `PRActiveMultifluidPipe` с
существующим `StratifiedPRColumn`. Параметры: `pipe`, `column`, параметры
капиллярного источника `parameters`, `port_side` (0/1), `port_height_m`,
`maximum_unknowns`, `linear_solver`, `volume_diffusion` и
`maximum_transverse_head_fraction` (по умолчанию 1e-3, максимум 1e-2).
Секция трубы горизонтальна, её явно заданные отметки совпадают с абсолютной
отметкой патрубка. Второй торец закрыт. EOS, реестр компонентов и устройство
совпадают. Запасы граничного состояния не добавляются в баланс.

Площади занятых фазами частей круглого патрубка вычисляются по точной
геометрии диска. Для каждой части используется середина её высотного
интервала и фактический PR-профиль сосуда. Нижний/верхний слои связываются
с объявленными жидким/газовым слотами с проверкой действительного корня EOS;
одной строковой метки корня недостаточно при единственном кубическом корне.
Это фазовый midpoint-подход с контролем поперечного напора, а не полная
квадратура многомерного входного течения.

За один шаг труба решается неявно при начальном состоянии сосуда, затем
**те же** принятые компонентные и энтальпийные переносы изменяют конечный
запас сосуда. Учитывается энергия `mass*g*z_port`; сосуд хранит полную
`U+PE`, труба — свои N/M/E и энергию поверхности. После изменения профиля
положение границы поверхности трубы меняется. Его работа не теряется:

```text
E_column_new + F_boundary(pipe_new, column_new)
  = E_column_old - E_port_transfer + Q + F_boundary(pipe_new, column_old)
```

Ограничение энергии решается итерационно на реальных PR-состояниях сосуда.
Работа формы выводится в `boundary_shape_work_from_column_j`, перенос — в
`port_energy_from_column_j`. Это производные физические слагаемые, а не
исправление конечной невязки. Реакция стационарного сосуда записывается
отдельно; поступившая кинетическая энергия остаётся в его энергетическом
запасе. Проверяются общий баланс компонентов, `E+F+PE` и энтропия.

Это консервативная схема IMEX первого порядка. Она не заменяет монолитный
Newton всей сети и не моделирует конечную скорость осаждения внутри
квазистатического `StratifiedPRColumn`. Сам сосуд сохраняет существующую
модель объёмного PR-равновесия с гравитацией: капиллярный скачок давления
и поверхностная энергия границы внутри сосуда в ней не разрешаются.
`F_boundary` относится к поверхности трубы у патрубка. Поэтому тест малых
геометрий проверяет дискретное энергетическое сопряжение этих объявленных
моделей, а не физическую калибровку капиллярного микросепаратора.
В акустической ветви поддерживаются
один/два слоя сосуда, полностью однофазный патрубок и тепловой цикл
исчезновения/восстановления интерфейса. Нулевая площадь означает отсутствие
внешнего фазового состояния, без EOS-вызова или остаточной массы. Внутри
осевых ячеек трубы пока нужны положительные фазовые/компонентные запасы.
Режим `volume_diffusion="none"` с отсутствующей внешней фазой отклоняется:
ему ещё нужен отдельный закон переноса через сухую границу.

Для этой связи используется `boundary_geometry="reservoir_cell"`: расстояние
между центрами запаса трубы и внешнего состояния равно dx. Старый осевой
режим `extrapolated_face` сохраняет расстояние dx/2 и положительную
экстраполяцию обеих фаз. Если фаза снаружи отсутствует, её виртуальная
скорость берётся из ячейки; соответствующая часть капиллярной силы
переносится на внутреннюю скорость, сохраняя дискретную виртуальную работу.
Это проверяется независимым объёмным оператором.

Площади тонких занятых частей диска вычисляются непосредственно как полосы,
без вычитания двух почти полных площадей. Аналитические JVP/VJP равны
ширине хорды на концах полосы и остаются конечными при исчезновении слоя.
Тот же оператор используется в существующих геометрии отборов и PR-патрубках.
Остаются несколько одновременно сопряжённых портов/сосудов и сравнение
полной апертурной квадратуры.

`advance` достигает отчётного времени подшагами и переносит `next_step_s`.
`PRCapillaryColumnState` сохраняет действительные запасы трубы и профиль
сосуда, время, шаг, идентификатор модели и `layer_slots` — привязку каждого
реального слоя к фазовому слоту трубы. Для однофазного начального состояния
привязка задаётся явно. Топология разреженных операторов
переиспользуется между шагами. Runner независимо проверяет каждый шаг,
включая химические профили, геометрию и противоположные переносы:

```bash
PYTHONPATH=src:scripts python scripts/run_pr_capillary_column.py \
  configs/surface_pr_capillary_column.json --output /tmp/pr-column.json
PYTHONPATH=src:scripts python scripts/run_pr_capillary_column.py \
  configs/surface_pr_capillary_column_interface.json --output /tmp/pr-column-interface.json
PYTHONPATH=src:scripts python scripts/run_pr_capillary_column.py \
  configs/surface_pr_capillary_column_phase_cycle.json --output /tmp/pr-column-cycle.json
```

`step`/`advance` принимают физическую мощность `column_heat_w` и температуру
резервуара `heat_temperature_k`. В баланс входит Q=power*dt, в производство
энтропии — вычет Q/T_reservoir. Runner принимает либо `duration_s`, либо
`schedule` с такими командами на каждом интервале. Третий конфиг нагревает
реальный сосуд до одной фазы, затем охлаждает до двух, сохраняя все запасы.
Это синтетическая численная проверка, не экспериментальная калибровка.

Второй конфиг специально делает работу смещения границы существенно
больше погрешности округления и требует итерации энергетического сопряжения.
Независимый проверяющий код использует скалярную PR EOS и существующий
SciPy-верификатор химических профилей. **В расчётном решателе SciPy нет**:
Ньютон, EOS и линейная алгебра исполняются на PyTorch CPU/CUDA.

### Общий граф капиллярных линий и конечных гравитационных сосудов

`PRCapillaryFacility` (`pr_capillary_facility_imex_v1`) задаёт `columns` —
именованные `StratifiedPRColumn` — и `lines`. Каждая линия содержит:

- `pipe`: сериализованный `PRActiveMultifluidPipe`;
- `parameters`: параметры `PRCapillaryPressure` без внешних высот/расстояний;
- `ports`: ровно два элемента, `null` для стенки либо
  `{column: имя_сосуда, height_m: высота_центра_патрубка_в_сосуде}`;
- `maximum_unknowns`, `linear_solver`, `volume_diffusion`: явные настройки solve.

Сосуд является конечной вершиной графа. Несколько патрубков, параллельные
линии и кольца используют **один** его компонентный/энергетический запас.
Допустимы разные горизонтальные отметки линий и разные высоты отборов в
одном сосуде. Отметка каждой линии совпадает с абсолютными отметками её
патрубков; вертикальные участки внутри этой капиллярной ветви ещё не
реализованы. EOS, устройство и фазовый реестр общие; разные сечения и
материалы линий допустимы. Соединённые сосуды имеют одинаковое g.
Топология фиксирована; патрубок `null` закрыт постоянно, не является
динамически переключаемой арматурой. Нулевая камера соединения этим
конечным сосудом не подменяется.

Сначала все линии решаются на одном общем шаге с начальными профилями
сосудов. `PRCapillaryAxialPipe.port_transfers` восстанавливает **по каждому
концу отдельно** интегралы принятых потоков N/M/E и капиллярной работы.
Они не получаются делением суммарной невязки. После этого по всем реальным
патрубкам каждого сосуда суммируются N и энергия, включая mass*g*z порта.
Только эти суммы изменяют его запас. Перенос не проходит последовательно
через копии одного сосуда, порядок строк входного JSON не влияет на расчёт.

Ограничение энергии каждого сосуда включает сумму работ изменения внешней
геометрии всех принадлежащих ему патрубков, при одинаковых конечных запасах
труб. Проверяются общий N/(E+F+PE) и энтропия с Q/T внешних резервуаров.
Импульсы проверяются в локальных уравнениях труб; сосуды стационарны и
воспринимают реакции. Скаляры импульсов разнонаправленных линий не выдаются
за векторный баланс установки.

`PRCapillaryFacilityState` хранит именованные `pipes`, `columns`,
`layer_slots`, общий через трубы clock, `next_step_s` и `model_id`.
`initialize(pipes, columns, layer_slots=..., next_step_s=...)` проверяет
полноту запасов. `step`/`advance` принимают `normal_relaxation` и
`heaters={имя_сосуда: {power_w, temperature_k}}`. Поддерживаются активные
слои сосуда и патрубка; положительная фазовая поддержка внутри осевых ячеек
пока обязательна. Весь шаг графа отклоняется при отказе любой трубы или
сосуда; `advance` дробит шаг, точно достигает отчётного времени и сохраняет
предлагаемый следующий шаг. Изменение кэшей линейной алгебры при отказе не
меняет физических запасов checkpoint.

`maximum_attempts` считает действительные попытки шага. Если последняя
разрешённая попытка достигает отчётного времени, результат принимается
сразу; дополнительная попытка только для проверки clock не требуется.
Незаконченный интервал после исчерпания бюджета остаётся отказом.

Это синхронная консервативная IMEX-схема первого порядка с общими конечными
запасами, **не монолитный Newton всего графа**. Ограничения PR-сосуда и
фазового midpoint-приближения патрубка из предыдущего раздела сохраняются.

#### Согласованное неявное граничное сопряжение

Опция модели `coupling="implicit"` и положительный
`maximum_coupling_iterations` (по умолчанию 12) включают итерацию по
конечным профилям всех сосудов. На каждой итерации трубы рассчитываются
**заново из исходных запасов**, с граничными состояниями текущего
приближения конечных сосудов. Затем суммарные принятые потоки обновляют
конечные запасы сосудов. Останавливаемся, когда целиком совпадают
полученные и использованные состояния сосудов и их реестры слоёв.
Не достаточно только малого суммарного энергетического баланса.

Для этого режима меняется порядок учёта поверхностной работы. Обозначим
состояние трубы через P, профили сосудов через C. Принятый трубный solve
использует C1 и сохраняет `E_pipe + F(P,C1)`. Поэтому сосуды оплачивают
изменение внешней геометрии **при начальном P0**:

```text
shape_work = F_boundary(P0,C1) - F_boundary(P0,C0)
E_column1 = E_column0 - sum(port_energy_to_pipes) + Q - shape_work
```

В исходном IMEX-режиме труба использует C0 и та же работа считается при
конечном P1. Подмена одного порядка другим нарушает суммарный баланс.
В обоих режимах дополнительно учитывается mass*g*z каждого патрубка и
проверяются общий N/(E+F+PE), энтропия и локальные уравнения труб.
В неявном режиме независимый аудитор сверяет реальные граничные EOS-состояния
с **конечными**, а не начальными профилями сосудов, включая смену числа фаз.

Несошедшаяся внешняя итерация является отказом всего физического шага;
`advance` уменьшает dt. Промежуточный сбалансированный IMEX-результат не
принимается как неявный. Кэши разреженной алгебры переиспользуются, запасы
checkpoint не изменяются. Число внешних итераций выводится в
`nonlinear_coupling_iterations`; `coupling_converged` подтверждает приёмку.
Бюджет внешних итераций отделён от `maximum_attempts` физических шагов.

Это сходящееся блочное сопряжение объявленных трубных split-операторов,
а не общая монолитная матрица Ньютона и не устранение временной погрешности
первого порядка. Активный набор фаз внутри осевых ячеек остаётся прежним.
Начальное IMEX-сопряжение сохраняется по умолчанию и не меняет model_id
старых конфигураций. Неявные настройки входят в новый идентификатор модели.

```bash
PYTHONPATH=src:scripts python scripts/run_pr_capillary_facility.py \
  configs/surface_pr_capillary_facility_implicit_cycle.json --device cuda \
  --output /tmp/pr-implicit-facility.json
```

`benchmarks/results/pr_implicit_facility_validation.json` сохраняет
привязанные к исходникам CPU/CUDA-прогоны полного трёхсосудного графа:
два принятых шага, ноль отказов, 5 и 6 внешних итераций, независимые
локальные и глобальные аудиты. Сверены 365 значений checkpoint.
Максимальное отличие T — 2,3e-13 К, скорости — 7,8e-15 м/с.
Для почти нулевых импульсов допуск сверки включает оценки округления
реальных потоков давления обоих backend; локальные уравнения каждого
расчёта независимо проверяются. Допуски Ньютона при этом не меняются.
Целевая регрессия: 48 успешных тестов и отдельный тест режима без нормального
источника. Это численная проверка синтетического режима, не калибровка
на оборудовании; время с дополнительными внешними итерациями не следует
сравнивать с одиночным IMEX-шагом как ускорение алгоритма.

#### Общая редуцированная матрица Ньютона

`coupling="newton"` включает Ньютон по нелинейному редуцированному остатку
`R(C)=C-G(C)`, где C содержит N компонентов и полную энергию всех сосудов.
`G` включает реальные решения труб, химических профилей и ограничения
поверхностной работы. Внутренние неизвестные этих блоков исключены;
производные общей матрицы учитывают их зависимость от C. Это нелинейная
редукция Шура, а не набор независимых поправок сосудов. Неизвестные всех
ячеек труб в одну большую явную матрицу здесь не собираются.

Матрица рассчитывается центральными конечными разностями полного G в
безразмерных координатах. На недопустимой стороне термодинамической области
используется односторонняя разность. Нормирующие масштабы не добавляют
вещества или энергии. Шаг разностей — 1e-4; тест перекрёстных блоков
сравнивает J*d с отдельной направленной разностью с другим шагом.
Матрица не предполагается диагональной или симметричной.

Используется modified/chord Newton: построенная общая матрица переиспользуется
на последующих итерациях этого шага. Решение СЛАУ через PyTorch `solve_ex`
проверяется по исходной линейной невязке; поиск длины поправки требует
уменьшения глобального остатка и допустимости EOS. Последний вычисленный
G кэшируется для повторной проверки той же точки; большой истории состояний
труб в памяти не накапливается. При малом остатке выполняется доведение
полного состояния до точного профильного fixed point, затем прежние
независимые проверки балансов. Малого R(C) для приёмки недостаточно.

`maximum_schur_unknowns` (по умолчанию 96) ограничивает размер этой плотной
матрицы. Число неизвестных равно `число сосудов * (число компонентов+1)`,
а не числу ячеек. Внутренние трубные решения сохраняют свою разреженную
алгебру и бюджеты. Большой плотный fallback не выполняется. Сам Ньютон,
масштабирование и линейная алгебра используют PyTorch на выбранном CPU/CUDA;
SciPy остаётся только в независимых верификаторах.

Полный FD-касательный оператор дорог: режим вводится для согласованного
общего нелинейного решения, не как обещание ускорения относительно IMEX.
Отдельная будущая оптимизация — аналитические/AD-производные исключённых
блоков. Эти настройки входят в model_id только для нового режима.

`schur_newton` в отчёте содержит размер матрицы, число построений и решений,
число вычислений G, попадания в кэш, линейную невязку, устройство матрицы
и факт профильного доведения. Несошедшийся Ньютон отклоняет весь физический
шаг, как и остальные режимы сопряжения.

```bash
PYTHONPATH=src:scripts python scripts/run_pr_capillary_facility.py \
  configs/surface_pr_capillary_facility_newton_cycle.json --device cuda \
  --output /tmp/pr-newton-cycle.json
```

Этот конфиг содержит один конечный сосуд с двумя подключениями одной трубы
и цикл нагрева/охлаждения. Отдельный тест с двумя конечными сосудами проверяет
ненулевые межсосудные блоки общей матрицы. Исчезновение фаз внутри осевых
ячеек этим режимом пока не реализовано.

Runner требует `system`, `initial`, `normal_relaxation`, явный
`maximum_attempts` на интервал и `schedule` с `{duration_s, heaters}`.
Сценарий ниже содержит три сосуда, кольцо, параллельную линию и восемь
реальных портов. Интерфейс одного общего сосуда исчезает и восстанавливается:

```bash
PYTHONPATH=src:scripts python scripts/run_pr_capillary_facility.py \
  configs/surface_pr_capillary_facility_cycle.json --device cuda \
  --output /tmp/pr-facility-cycle.json
```

`check_pr_capillary_facility.py` независимо проверяет скалярную PR EOS,
химические профили, площади патрубков, каждый осевой/нормальный solve,
раздельные портовые потоки, каждый сосуд и суммарные балансы графа.
Эта проверка не вызывает расчётный `step`, обновление сосудов или рабочий
`port_transfers`.

При почти противоположных давлениях на торцах допуск округления импульса
оценивается по абсолютным **реальным потокам и путям давления** и импульсам
ячеек, а не их почти нулевой сумме. Отдельное слагаемое записывается в
`momentum_roundoff_allowance_kg_m_s`. Критерий Ньютона 2e-12 не изменяется;
независимый аудитор продолжает проверять исходные локальные уравнения.

Сводка текущего сценария и CPU/CUDA-сверки хранится в
`benchmarks/results/pr_capillary_facility_validation.json`: 122 теста
затронутых модулей и три отдельные проверки граничного бюджета попыток.
Предыдущие сводки одиночного сосуда и масштабирования сохраняют собственные
SHA256 и относятся к тем версиям исходников, на которых были получены.

### Разреженная алгебра и воспроизводимый замер масштаба

`linear_solver="auto"` у нормального, осевого и узлового интеграторов
использует плотный PyTorch-якобиан до 192 неизвестных и разреженный выше.
Можно явно выбрать `dense` или `sparse`. `maximum_unknowns` остаётся
отдельным бюджетом: его задают для осевого solve и нормального источника
с учётом числа компонентов. Большой плотный fallback запрещён.

Геометрия хранится списком граней O(E+N); `incidence` оставлен только как
лениво создаваемое плотное диагностическое представление. Рабочие давление,
энергия и энергетические порты его не используют. Якобиан строится
окрашенными JVP на физическом графе, с пакетами по восемь направлений.
Соседство включает два перехода по графу в осевом операторе и один в
нормальном. Структура, раскраска и блоки предобуславливателя компилируются
один раз; значения обновляются на каждой итерации Ньютона.

Используются существующие PyTorch COO, блочный Jacobi и GMRES. Статус
читается на границах рестартов, а не для каждого вектора Арнольди. Приёмка
требует истинной невязки через независимый JVP исходного нелинейного
оператора. Пропущенная зависимость в разреженном шаблоне является ошибкой
программы и не маскируется дроблением времени. Невышедшая на требуемую
точность СЛАУ остаётся численным отказом шага.

```bash
PYTHONPATH=src:scripts python scripts/run_pr_capillary_scaling.py \
  configs/surface_pr_capillary_scaling.json --device cuda \
  --output /tmp/pr-capillary-scaling.json
PYTHONPATH=src:scripts python scripts/run_pr_capillary_column_scaling.py \
  configs/surface_pr_capillary_column_scaling.json --device cuda \
  --output /tmp/pr-column-scaling.json
```

Runner увеличивает длину периодического синтетического домена при
неизменных размере ячейки и шаге времени; каждая большая ячейка сверяется
с плотным малым периодическим шаблоном. Замер отделяет построение/прогрев
от тёплого сопряжённого шага, сохраняет число элементов, цветов и рестартов,
невязку и пик памяти PyTorch. Этот тест не подтверждает производительность
большой неоднородной сети с множеством гравитационных сосудов или Norne.

Второй runner увеличивает непериодическую трубу, связанную с одним конечным
гравитационным сосудом. Размер ячейки, шаг и исходный запас сосуда неизменны;
растёт действительный запас трубы. Замер включает её осевой/нормальный
solve и энергетическое согласование сосуда. После таймера каждый размер
проверяется независимыми PR EOS, химическими профилями, объёмными,
импульсными, компонентными и энергетическими уравнениями. Прогрев отделён
от тёплого шага; память относится к аллокациям PyTorch процесса. Это
проверка одного сосуда с одной линией, не произвольного промысла.

Сохранённые измерения и полные параметры:
`benchmarks/results/pr_capillary_sparse_scaling.json` и
`benchmarks/results/pr_capillary_column_scaling.json`. На 1024 ячейках
наблюдались соответственно 1,61 с / 70,1 МиБ и 4,05 с / 71,0 МиБ на
RTX 3090 Ti. Это по одному тёплому шагу на размер, без гарантии изолированной
машины; плотный эталон и независимые физические аудиты приняты.
Переходы и CPU/CUDA-сверка сохранены в `pr_capillary_column_validation.json`
и `pr_dry_wall_validation.json` в той же директории результатов.


### Точно пустые фазы в капиллярной осевой трубе

`PRCapillaryPressure.phase_support` принимает `"positive"` (по умолчанию)
или `"active"`. Поле сериализуется только для активного режима, сохраняя
идентификаторы прежних моделей. В активном режиме отсутствующую фазу
нужно опустить из словаря ячейки. Объекты с нулевыми N/V не создаются.
Оператор осевого шага решает конечные внутренние плотности, скорости,
температуры и объёмную долю в [0,1]. Эти транспортные следы существуют
отдельно от принадлежащих ячейке запасов. EOS пустой фазы с N=V=0
не вызывается; конечный исходящий след проверяется на PR-ветвь.

Активная ветвь применима к физической равномерной линии; перенос её
в специализированный оператор разветвлённых ячеек не объявляется.
Линии могут соединяться конечными сосудами `PRCapillaryFacility`.
Нормальная релаксация учитывает только ячейки с обеими фазами, сохраняя
полную геометрию капиллярной энергии. `active_cells` задаёт индексы
строк её отчёта. При отсутствии интерфейсов возвращается нулевой
источник с пустым списком активных ячеек.

`step(state, dt_s, phase_controls=...)` и `step_coupled` принимают:

```python
phase_controls = {
    0: {
        "inlet_slot": "gas",
        "inlet_phase": actual_pr_gas_phase,
        "target_liquid_fraction": 0.0,
    }
}
```

`inlet_phase` — реальный `PRRelaxationPhase` или его сериализованное
представление. Для жидкостного входа цель должна увеличивать жидкостное
удержание, для газового — уменьшать. Расход решается вместе с уравнениями,
а не назначается после расчёта. Это идеальный управляемый объёмный обмен,
не характеристика произвольного реального клапана. Заданная цель относится
к осевому подшагу; последующая нормальная релаксация может изменить её.
При повторении команды через `advance` цель удерживается каждым подшагом.

Если q_l — подписанный объёмный расход жидкости внутрь ячейки, q_g=-q_l,
а μ — дискретная производная поверхностной энергии по жидкостному объёму,
то внешняя работа привода равна:

```text
W = dt * q_l * (μ - P_liquid + P_gas)
```

Фазовые энергии получают энтальпию потока и работу `-P_phase*q_phase`.
Общий баланс включает входящую/выходящую H и W, состав и импульс каждого
потока; W не вычисляется как остаток невязки. Энтропия потока определяется
PR-состояниями входа и конечного следа отбора. Состояние с отрицательным
производством энтропии отвергается, даже если целевой объём достигнут.

В общем графе `phase_controls` имеет дополнительный внешний ключ имени
линии. Этот аргумент передаётся в IMEX, fixed-point и Schur-Newton режимы;
внешний баланс графа включает состав, H, W, энтропию и потенциальную
энергию бокового потока. В расписании runner поле необязательно:

```json
{"duration_s": 1e-7, "heaters": {}, "phase_controls": {"out": {}}}
```

Пустой словарь допустим только для активной линии. Полный пример с
реальными фазовыми входами: `surface_pr_active_capillary_facility_cycle.json`.
Независимые аудиторы сверяют команды с отчётом, пересчитывают скалярную
PR-термодинамику, 64-точечные пути, сопряжённые силы, объёмные уравнения,
состав, энергию/работу и энтропию. Исчезновение и заполнение при подаче
вещества не заменяют отдельную модель спонтанной нуклеации. Центральный
режим без диффузии не гарантирует допустимость на резком фронте;
для заполнения пустых соседних ячеек проверяется `volume_diffusion="acoustic"`.
Материал стенки без конечного сухого продолжения отвергается в активном
режиме; экспериментальной калибровки этого продолжения нет.

Воспроизведение связанного активного цикла с аудитом каждого принятого шага:

```bash
PYTHONPATH=src:scripts python scripts/run_pr_capillary_facility.py \
  configs/surface_pr_active_capillary_facility_cycle.json --device cpu \
  --output /tmp/pr-active-capillary-cpu.json

CUDA_VISIBLE_DEVICES=GPU-2f9ce359-f3f4-accc-bf9c-31bc88bb1cb7 \
PYTHONPATH=src:scripts python scripts/run_pr_capillary_facility.py \
  configs/surface_pr_active_capillary_facility_cycle.json --device cuda \
  --output /tmp/pr-active-capillary-cuda.json
```

Runner отказывается перезаписывать существующий отчёт, сохраняет запрос,
траекторию, SHA256 исходников, фактические итерации матрицы Шура и результаты
независимых проверок. Малый синтетический цикл не является оценкой
производительности крупной производственной сети или Norne.

Сводка текущего активного цикла:
`benchmarks/results/pr_active_capillary_facility_validation.json`.
Она включает результаты CPU/RTX 3090 Ti, сверку 119 значений состояния,
независимую повторную проверку замкнутого цикла с ненулевым смачиванием,
73 успешных теста, хеши исходников/траекторий и фактическую статистику
Schur-Newton. Сводки предыдущих этапов сохраняют собственные хеши и
не считаются проверкой произвольной последующей версии исходников.

### Тепловое появление и исчезновение капиллярной фазы

`PRCapillaryEquilibrium` — отдельное замыкание мгновенного локального
UV-равновесия с геометрической и материальной поверхностной энергией.
Пример: `configs/surface_pr_capillary_thermal_cycle.json`; вариант с
ненулевым смачиванием — `surface_pr_capillary_wetting_thermal_cycle.json`.

При положительных фазах решаются одновременно равенство температур,
химических потенциалов, `P_l-P_g = dF/dV_l` и сохранение `U+KE+F`.
Объём и компонентные запасы ячейки сохраняются; после мгновенной релаксации
фазы имеют общую скорость при сохранённом суммарном импульсе. Потеря
относительной кинетической энергии остаётся в тепловой энергии.
Однофазные кандидаты имеют ровно один реальный объект запаса. Существующий
UV-flash и состояния насыщения используются только для начального
приближения: принятый двухфазный кандидат решается заново с реальной F.
Проверяются PR-ветвь, устойчивость стационарного решения по Гельмгольцу,
локальный тест появления противоположной фазы и неубывание энтропии.
Многозапусковый поиск не считается доказательством глобального минимума.
Модель использует два слота VLE на физической линии. Если bulk-предиктор
обнаружил третью фазу или два явных жидких корня, расчёт прекращается с
ошибкой неподдерживаемой топологии. Эти фазы не отбрасываются и не
переименовываются в газ. Трёхфазная капиллярная модель с отдельными
межфазными энергиями требует самостоятельного расширения.

`project(state, heat_j=[...], heat_temperature_k=...)` задаёт энергию
каждой ячейки в джоулях. Положительный знак означает подвод тепла.
При ненулевом тепле обязательна температура внешнего резервуара.
Отчётные часы не продвигаются: это термодинамическая транзакция.
Испарение и последующая конденсация могут пройти без материального входа.
Для сухой стенки с ненулевым потенциалом требуется явное конечное
продолжение `c2_half_adsorption`.

**Объём сопряжения:** `project` выполняет один упорядоченный проход
локальных UV-транзакций. В момент обработки ячейки соседние высоты
фиксированы, а изменение полной энергии примыкающих граней учитывается
точно. Это сохраняет глобальные N/M/(E+F) и локальную энтропию, но не
объявляется одновременным сходящимся flash всей линии. Последующая
транзакция соседа может изменить капиллярный градиент предыдущей ячейки.
`equilibrate` повторяет эти проходы до общего критерия 2e-11 по химическим
потенциалам, температуре внутри пары фаз и капиллярному скачку давления,
при стационарной фазовой регистрации и объёмных долях. Тепло прикладывается
только в первом проходе. Нормальный источник `phase_equilibrium=true` и
runner теплового цикла используют именно это доведение, а не одиночный
`project`. `phase_equilibrium_sweeps` задаёт бюджет (по умолчанию 12);
исчерпание бюджета отклоняет всю транзакцию. Разные ячейки могут сохранять
разные температуры: это совместное межфазное равновесие, не выравнивание
температуры или давления по всей пространственной линии. Численная
кинетика нуклеации и экспериментальная скорость межфазного обмена этим
мгновенным замыканием не задаются.

Для применения этого прохода внутри `PRCapillaryAxialPipe.step_coupled`
и `PRCapillaryFacility` у источника задаются:

```json
{"phase_support": "active", "phase_equilibrium": true}
```

`phase_equilibrium=false` сохраняет прежний конечный по скорости источник
и прежние model_id. Мгновенный режим нельзя смешивать с ненулевыми
коэффициентами конечного химического/теплового обмена. Осевой и нормальный
операторы остаются расщеплением первого порядка. Независимый проверяющий
нормального источника распознаёт этот явно объявленный режим и пересчитывает
каждую термодинамическую транзакцию отдельной скалярной PR-реализацией.

```bash
PYTHONPATH=src:scripts python scripts/run_pr_capillary_equilibrium.py \
  configs/surface_pr_capillary_thermal_cycle.json --device cpu \
  --output /tmp/pr-thermal-capillary.json
PYTHONPATH=src:scripts python scripts/check_pr_capillary_equilibrium.py \
  configs/surface_pr_capillary_thermal_cycle.json /tmp/pr-thermal-capillary.json
```

Условия капиллярного равновесия выводятся из полной энергии Гельмгольца,
а не из последующей поправки обычного bulk-flash; первичная работа:
[Sun, Energy Stable Simulation of Two-Phase Equilibria with Capillarity](https://www.iccs-meeting.org/archive/iccs2019/papers/115390520.pdf).
UV-транзакции, разрешённая геометрия и конкретные материальные продолжения
в проекте имеют собственные перечисленные выше ограничения.

### Принятые скважинные дебиты и калорическая подготовка

`FacilityCoupledFIM(..., energy_process=AcceptedWellEnergyProcess(...))`
добавляет термодинамическую обработку после выбора допустимого фактического
FIM-решения, до публикации шага. Требуются PR-fluid, явный
`StockComponentAssay`, тепловые условия всех скважин, `transfers` и `mixer`.
Входом служат `solver.wells.surface_report["wells"]`, а не запрошенные
дебиты или масштабированные продукты отклонённой попытки.

`energy_process` содержит реальные граничные потоки, PR-продукты и
накопленные моли/энергии. В энергии учитываются заданные тепловая граница,
тепло и работа оборудования процесса. Ошибка обработки откатывает поля
пласта, скважины и оба отчёта существующей транзакцией FIM. Продолжение
после pickle сохраняет накопленные значения. При `energy_process=None`
поведение и формат прежнего отчёта не расширяются.

Это калорическая обработка принятого материального решения, а не
термический FIM или динамическая PR-сеть в общем пластовом Якобиане.
Обратная связь производительности остаётся у существующей PVT-установки;
калорический PR-процесс не добавляет собственных ограничений давления.
Модель сериализуется через `AcceptedWellEnergyProcess.to_dict/from_dict` и
задаётся в `simulation.coupled_facility.energy_process`. Текущие команды
передаются отдельно через `energy_controls`: `thermal_conditions`,
`transfers`, `mixer`. Они сохраняются в `applied_controls`, не меняют assay
или EOS и не обнуляют накопленный баланс. Смена уставки температуры
проверяется совместно с изменением производственного ограничения и рестартом.
Существующая связь FIM остаётся producer-only; совместное давление
нагнетательных скважин и новой динамической PR-сети требует отдельного
сопряжения.

Отчёт явно сохраняет `reservoir_energy_equation_solved=false`,
`pressure_feedback_from_caloric_process=false` и
`pvt_eos_characterization_calibrated=false`.

### Цветная касательная матрица и тепловой профиль

Глобальный Schur-Newton группирует возмущения по графу сосудов/линий.
Блок строки зависит от собственного профиля и сосудов на других концах
примыкающих линий. Одновременно возмущаются только столбцы с непересекающимися
наборами строк; это раскраска квадрата графа. На термодинамической границе
группа разбирается на отдельные столбцы и сохраняются односторонние
разности. `jacobian_colours`, `jacobian_sample_groups` и
`jacobian_group_fallbacks` раскрывают фактический путь построения.
`jacobian(..., grouped=False)` оставлен как диагностический эталон.
В контрольной блочной цепочке из 12 узлов с двумя переменными требуется 12
вместо 48 допустимых вычислений карты. Это уменьшение работы построения
касательной, не обещание четырёхкратного ускорения всей физической сети.

Тепловой гравитационный сосуд сначала проверяет найденный совместным
Ньютоном профиль через `_accept` при точных конечных N/E. Сохраняются все
проверки EOS, локальной фазовой устойчивости, границ, геометрии и сходимости
квадратуры. Повторный `_solve` остаётся резервным путём, когда нужна
дополнительная точность. `profile_acceptance` показывает выбранный путь;
критерии физической приёмки не ослабляются.

### Калибровка Torch и отдельная проверочная выборка

`fit_drift_flux` сохраняет прежний интерфейс NumPy и принимает Tensor на
CPU/CUDA. Взвешенная задача с двумя параметрами решается на выбранном
устройстве перебором внутренней точки и четырёх граней прямоугольника
ограничений. Проверка идентифицируемости использует SVD; SciPy в рабочем
модуле больше не требуется. На хост переносится короткий итоговый отчёт.

`fit_validate_drift_flux(..., training_indices=..., validation_indices=...)`
требует полного непересекающегося разбиения наблюдений. Коэффициенты
подбираются только по обучающей части; возвращаются обычные и
нормированные заданными погрешностями ошибки остальных наблюдений.
Экстраполяция проверочной выборки за обученный диапазон по умолчанию
запрещена. Универсальный порог «модель принята» не выдумывается.
Синтетическое восстановление коэффициентов не является калибровкой
реального нефтепромыслового оборудования.

### Набор переходных сценариев

```bash
PYTHONPATH=src:scripts python scripts/validate_pr_surface_matrix.py \
  --suite transition-release --device cpu --jobs 2 \
  --output /tmp/pr-transition-release
```

Набор проверяет реверс, изоляцию, расписание с рестартом, реверс насоса,
его остановку, антипомпаж, открытое испарение и повторное нагнетание тумана.
Каждый сценарий имеет собственный конфиг и независимый аудит. Это серия
отдельных функциональных сценариев, а не доказательство совместимости
всех возможных ограничений в одной произвольной промышленной сети.

### Динамический PR-приёмник с обратной связью в FIM

`simulation.dynamic_receiver` подключает `DynamicReceiverCoupledFIM` к обычному
`Simulator.run_step`. Это альтернативный владелец шага: одновременное задание
`coupled_facility` запрещено. Пример полного конфига —
`configs/fim_dynamic_pr_receiver.json` (поле `simulator` содержит привычный
конфиг пласта, флюида, скважин и симуляции; `intervals` — расписание runner).

Обязательные поля `dynamic_receiver`:

- `model="dynamic_pr_receiver_fim_v1"`;
- `receiver`: сериализованный `StratifiedPRFlow` с пустым списком `feeds`;
- `initial_state`: физический `StratifiedThermalState` этого приёмника;
- `assay`: явные массовые доли PR-компонентов в трёх stock-группах;
- `thermal_conditions`: температура или поток полной энергии каждой скважины;
- `inlet_height_m`: высота материального входа в сосуд;
- `control`: сериализованный `DynamicReceiverControl`.

Высотная отметка входа должна совпадать с существующим приёмным узлом
сборной сети. Тепловые границы скважин используют физические отметки BHP.
Приёмник и пласт работают на одном устройстве. Компонентная характеристика
не выводится из black-oil PVT автоматически; смесь methane/ethane в примере
служит синтетической проверкой переноса, а не характеристикой промысловой
нефти и воды.

Конечное давление в точке входа замыкается внешним скалярным решением.
Каждая попытка повторяет **тот же** шаг пласта и приёмника из исходного
состояния. Фактически полученные stock-дебиты преобразуются в PR-компоненты;
адиабатический перенос сохраняет полную энергию, включая гравитационную.
В приёмнике одновременно меняются запасы, энергия, равновесный
гравитационный профиль, температура стенки и отборы через конечные
круглые патрубки. Его конечное давление возвращается в гидравлический
расчёт скважин. Одного малого изменения давления между итерациями
недостаточно: проверяется физическая невязка «давление профиля минус
давление приёмного узла».

Недостаточное давление подаваемого потока остаётся отказом физической
проверки. `ReceiverFeedPressureError` лишь сообщает найденное давление
неопубликованного кандидата для построения следующей попытки; защита
от неявного появления насоса не отключается. В журнале `pressure_trials`
видны обе стороны поиска и окончательная принятая попытка.

Полный связанный шаг сравнивается с двумя половинными. Оцениваются
давление, температура, компоненты, энергия и интерфейсы сосуда, температура
стенки, давление/насыщенности пласта и растворённые компоненты `rs/rv`.
Принимаются два половинных шага. Их добыча и отборы учитываются один раз;
полный контрольный шаг и отброшенные итерации в накопленные величины не
попадают. Диагностика для следующего шага `Simulator` учитывает изменение
пласта за весь принятый интервал. Превышение ошибки возвращает `False`,
чтобы штатный контроллер дробил время. При любом отказе восстанавливаются
пласт, данные скважин, приёмный узел, запасы сосуда и предыдущий отчёт.

`outlet_openings={имя: открытие}` изменяет команду между шагами, не сбрасывая
запасы и накопленные балансы. Штатные регуляторы патрубков сохраняют свою
семантику `StratifiedPRFlow`; ручное открытие не отменяет активную уставку
давления/уровня. Для точного продолжения сохраняется весь `Simulator`
через существующий checkpoint/pickle: один FIM-решатель остаётся общим с
объектом сопряжения. Начальные данные отдельного нового runner начинаются
с нулевого времени; одного checkpoint сосуда недостаточно для рестарта
пласта.

Воспроизводимый запуск и независимая проверка:

```bash
PYTHONPATH=src:scripts python scripts/run_dynamic_receiver.py \
  configs/fim_dynamic_pr_receiver.json --device cpu \
  --output /tmp/fim-pr-receiver.json
PYTHONPATH=src:scripts python scripts/check_dynamic_receiver.py \
  configs/fim_dynamic_pr_receiver.json /tmp/fim-pr-receiver.json
```

Runner сохраняет исходный запрос, принятые состояния, реальные скважинные
пакеты, давление каждой попытки, времена выполнения и SHA256 Python-кода.
Проверяющий использует скалярную PR-модель и независимую 32-точечную
квадратуру патрубков. Он проверяет профили, конечное давление, stock-assay,
перенос компонентов/энергии, пассивность переноса, последовательность
команд и накопленные суммы. Он не пересчитывает резервуарную СЛАУ и не
подтверждает оценку временной ошибки независимо от парного интегратора;
его скалярный flash потоков ограничен двухфазным равновесием.

Подтверждаемый режим — активные добывающие скважины с положительными
ограничениями дебита, стационарная PVT-сборная сеть и конечный динамический
PR-приёмник. Вход задаётся точечным материальным потоком; конечная площадь
отводов моделируется существующей моделью сосуда. Сжимаемые переходные
процессы в трубах, полный произвольный динамический граф, нагнетательная
обратная связь и тепловое уравнение пласта этим расширением не закрыты.
Поля `reservoir_energy_equation_solved=false` и
`pvt_eos_characterization_calibrated=false` сохраняются;
`pressure_feedback_from_caloric_process=true` относится именно к этому
динамическому приёмнику. Критерий малого перепада существующих клапанов
сохраняется, а экспериментальная калибровка не заявляется.

Полные контрольные CPU/CUDA-прогоны, результаты независимой проверки и
перечень проверенных режимов сохранены в
`benchmarks/results/fim_dynamic_pr_receiver_validation.json`. Для нового
сопряжения ускорение на GPU пока не подтверждено: контрольный малый
интервал медленнее CPU. Это отдельная ветвь от ранее измеренного
ускорения теплового профиля на большой наземной сети.


### Оптимизация термодинамики без изменения физической модели

`EquilibriumFluid.flash_ph/flash_ps` сначала строит однофазный калорический
предиктор по Cp или Cp/T. Результат принимается только после полного
PT-flash и проверки исходного допуска свойства. Если конечное состояние
многофазно, корень вырожден, предиктор не разрешён или нарушен домен,
используется исходный `_flash_property_bracketed`, включая скрытую теплоту.
Состав, параметры EOS, температурный домен и физический `model_id`
не меняются.

TPD сохраняет полный набор исходных детерминированных стартов. Пакетный
поиск использует разности химических потенциалов, собственную касательную
и допустимый спуск каждого старта. Приёмка быстрого стационарного
кандидата требует допустимого EOS и положительной кривизны локального
минимума. Остальные старты сохраняют независимый исходный LBFGS и
Newton-polish. Ошибка пакетной факторизации тоже возвращает исходный
путь; плохой предиктор не означает автоматически устойчивую фазу.
Это по-прежнему многостартовая численная проверка, а не доказательство
глобального минимума Gibbs. Аналитические softmax/tanh-градиенты точно
соответствуют прежним Gibbs–Duhem-целям; EOS в этом поиске не строит
бесполезный обратный AD-граф. Дифференцируемые EOS/равновесные касательные
сохраняются.

`surface.equilibrium_cache.cached_flashes(fluid)` разрешает повторное
использование **точно совпадающих** PT/PH/PS-запросов внутри одного
физического шага. По умолчанию бюджет — 256 записей с вытеснением LRU.
Скалярный Tensor и тот же Python-скаляр нормализуются без округления;
различающиеся на один ULP параметры остаются разными запросами. Ключ
включает параметры/идентичность модели и версии её тензоров. Все
возвращаемые деревья тензоров владеют отдельной памятью. Изменение
полученного отчёта не портит повторный результат. Требующие AD входы
и параметры EOS обходят этот кэш. Ошибки не кэшируются, записи очищаются при выходе из
контекста, включая отказ шага. Вложенный `enabled=False` сохраняет
диагностический путь без мемоизирования.

`DynamicReceiverCoupledFIM.step` использует такой контекст автоматически.
Поле отчёта `property_evaluations` содержит реальные попадания/промахи,
число калорических быстрых решений и резервных брекетирований, количество
пакетных/резервных стартов TPD. Это счётчики выполненной работы, не оценка
числа низкоуровневых PCIe-транзакций. Сохранённый полный шаг служит только
предиктором начального давления двух половинных шагов; ни решение
половинных шагов, ни оценка ошибки не пропускаются.

Проточный сосуд сохраняет `profile_acceptance`:
`verified_joint_flow_iterate` — совместный профиль принят при точных
конечных запасах/энергии с полными исходными проверками;
`refined_fixed_inventory_profile` — потребовалось прежнее уточняющее
решение. Пороги EOS, фазовой устойчивости, квадратуры, энергии, энтропии
и допустимого перепада патрубков не ослабляются. Измерения этой версии
находятся в `benchmarks/results/fim_dynamic_pr_receiver_optimization.json`;
исходная реализация — в предыдущей сводке валидации.

### Пространственные пакеты проверки устойчивости

`EquilibriumFluid.homogeneous_stability_mask(p, t, x, log_phi,
maximum_seed_rows=4096)` проверяет несколько различных P/T/составов за
один пакет. Возвращаемая маска имеет смысл **«устойчивость подтверждена»**:
`False` требует полного индивидуального PT/TPD-flash. Это не классификация
неустойчивой фазы и не разрешение пропустить точку.

Пакет сохраняет все исходные старты TPD для каждой точки, индивидуальные
касательные и поиски шага. Положительный результат требует допустимого
EOS-домена, совпадения фугитивностей с устойчивым кубическим корнем,
разрешения каждого старта и исходного порога TPD. Проверка включает
температурное продолжение PR: множитель перед возведением в квадрат alpha
должен оставаться положительным, как в скалярном `_validate`.

Реестр и точные нулевые компоненты сохраняются. Различающиеся между
точками активные поддержки, недостаточный бюджет, неразрешённые старты,
неустойчивый feed или ошибка пакетной факторизации оставляют точки для
индивидуальной проверки. Большие наборы делятся на порции в пределах
`maximum_seed_rows`; глобальная плотная матрица по всем точкам не строится.
Число строк бюджета относится к стартам TPD, а не числу ячеек пласта.

`StratifiedPRColumn._check_locals` использует эту маску для профилей;
`StratifiedPRFlow` — также для всех квадратурных точек конечных патрубков.
Каждая неподтверждённая точка проходит прежний `_check_local`, включая
точки в сухих частях патрубков. Геометрия, квадратуры, фазы, EOS, допуски
и балансы не меняются. Для строк с разными температурами применяется
явное `vmap` скалярного EOS-контракта, без смешивания координат разных
состояний при матричных умножениях.

В `property_evaluations` добавлены `spatial_stability_batches`,
`spatial_stability_points`, `spatial_stability_confirmed`. Они показывают
реальную работу пакетной валидации; общее число TPD-стартов по-прежнему
учитывается отдельно. Это счётчики вычислений, не число PCIe-транзакций.

Воспроизводимый отдельный замер участка свойств:

```bash
PYTHONPATH=src:scripts python scripts/benchmark_pr_stability_batch.py \
  --device cuda --sizes 1 8 32 128 --repeats 3 \
  --output /tmp/pr-spatial-stability.json
```

Каждая точка имеет свои P/T/состав. Оба пути выполняют одинаковую
физическую проверку; неподтверждённые пакетные точки действительно
проверяются полным индивидуальным flash. Время построения состояний и
прогрев вынесены из измерения. Этот тест участка свойств не измеряет
производительность пласта или всей сети.

`benchmarks/results/pr_spatial_stability_validation.json` хранит два вида
замеров раздельно: участок свойств на 1/8/32/128 точках и реальные связанные
FIM/PR-интервалы на 3/30/300 ячейках. Полные входные конфиги и состояния
включены в сводку. Связанные времена относятся к `Simulator.run_step`,
включая его диагностику; чтение/построение модели и независимый последующий
аудит в это время не входят. Сетки имеют фиксированную ширину 60 м и
явно заданные индексы скважин: это проверка масштабирования и совпадения
версий внутри каждого случая, а не исследование сходимости по сетке.
