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
Операция `nozzle` для смеси явно отклоняется. Уравнения, соглашения по единицам
и ограничения находятся в [термодинамике](THERMODYNAMICS.md).
