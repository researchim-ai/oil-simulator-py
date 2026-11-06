# Секционный спектр капель в общем PR-графе

`PRSectionalDropletPopulation` (`model="sectional_droplet_population"`)
решает числа капель по размерам, коалесценцию, дробление и
поверхностную энергию совместно с PR-запасами, импульсом и температурой.
Модель поддерживается обычной трубной ячейкой и сборщиком распределённого
сепаратора; отдельного интегратора, который подставлял бы готовые расходы
в основной граф, нет.

## Представление размеров

В `reference_volumes_m3` задаются 3…16 строго возрастающих опорных объёмов.
Последняя точка задаёт нижнюю границу крупного хвоста. Для внутренних
секций неизвестны числа капель; хвост имеет две неизвестные: число и
суммарный опорный объём. Поэтому образование капли крупнее последней
точки не удаляет массу и не заменяется её обрезанием до максимального размера.
Средний размер хвоста меняется при коалесценции.

При попадании продукта между двумя точками линейные веса сохраняют число
продуктов и первый объёмный момент. Идея fixed-pivot дискретизации изложена
в [Kumar–Ramkrishna, 1996](https://doi.org/10.1016/0009-2509(96)88489-2).
Двухмоментное продолжение верхнего хвоста — отдельное приближение этой
реализации, а не разрешение полного распределения внутри хвоста.

Физические объёмы капель получаются из опорных общим множителем
`V_dispersed / (sum(N_i*v_i_reference) + V_tail_reference)`.
Таким образом, их сумма всегда совпадает с живым объёмом дисперсной
PR-фазы. Сжатие и равновесный межфазный обмен задают равномерное относительное
изменение объёма капель. Индивидуальная диффузионная кинетика роста здесь
не решается. Все размеры содержат одну жидкую фазу с общими T, P и составом.

## Реакции и мощность

`coalescence_model="constant"` использует заданное эффективное ядро
`coalescence_kernel_m3_s`. Альтернатива `brownian_gravity` вычисляет симметричную
матрицу частот по живым размерам, относительным скоростям и температуре:

`K_B = 2*kB*T*mobility/(3*mu_carrier) * (di+dj)^2/(di*dj)`;
`K_G = pi/4*(di+dj)^2*abs(vi-vj)`;
`K = collision_efficiency*(K_B+K_G)`.

Для чистой подвижной границы множитель mobility согласован с HR-сопротивлением.
Для заданного эффективного постоянного ядра отдельная эффективность должна быть
единичной: двойного учёта коэффициента нет. Сумма механизмов — аддитивное
приближение; гидродинамическое взаимодействие пар и связанная задача
конвективной диффузии здесь не решаются. Базовые диффузионная и геометрическая
формулы приведены в [Arias-Zugasti–Rosner, уравнения 1, 4–5](https://doi.org/10.1103/PhysRevE.84.021401);
эта работа также показывает ограничения простого сложения механизмов.

Каждое успешное столкновение убирает две капли и создаёт одну, сохраняя
первый опорный момент. `section_breakup_rates_s` задаёт частоты дробления
по секциям, включая среднюю каплю хвоста. `daughter_volume_fractions` задаёт
2…16 положительных дочерних объёмных долей с суммой один; по умолчанию
это две равные капли. Для родителей, чья минимальная дочь меньше
первой точки сетки, частота обязана быть нулевой.
Это явная нижняя граница модели; мелкие дочерние капли не исчезают из учёта.

`agitation_power_w` поступает как механическая работа в полную энергию.
Перед шагом проверяется непрерывная потребность в энергии новой поверхности.
На принятом шаге дополнительно проверяется дискретная разность поверхности
с состоянием без источника дробления. Недостаток мощности возвращает отказ.
Коалесценция высвобождает поверхностную энергию в жидкость, не удаляя её из
полного баланса. При нулевых частотах дробления мощность может быть нулевой.

## Перенос по размерам

Каждая секция имеет свою терминальную скорость по выбранному сферическому
Stokes/HR-замыканию. Средняя скорость фаз сохраняет массовый импульс.
Разброс скоростей дополнительно включён в кинетическую энергию и поток
импульса; перенос энергии использует скорости и поверхность каждой секции.
Римановы пакеты разделяют дисперсную фазу по размерам, сохраняя общий EOS.
Числа капель и верхний объёмный момент переносятся тем же потоком.
Характеристики учитывают все эти неизвестные и проверяются на гиперболичность.

Все удерживаемые размеры, включая хвост, проверяются по Re, Eo,
капиллярному перепаду, отношению диаметра капли к трубе и пределу скорости.
Проверка одного среднего диаметра не заменяет этих ограничений.
При закрытом соединении обнуляются и компоненты, и все капельные потоки.

Прямое соединение спектральных ячеек требует одинаковой сетки размеров
и семейства капель. На границе с иной моделью нужен явный конечный объём
`droplet_capture=true`: выходящая поверхность термализуется в его запасах.
Обратное поступление использует явно заданное начальное распределение,
а не восстанавливает неизвестный спектр по одному среднему диаметру.
Это идеализированная граница захвата/инжекции, не измеренная эффективность аппарата.

## Конфигурация, отчёт и restart

Кроме ядра и мощности обязательны `reference_volumes_m3`,
`initial_volume_fractions` и `section_breakup_rates_s` одинаковой длины.
Начальные объёмные доли положительны и суммируются в единицу. При появлении
дисперсной фазы без входящего капельного потока это же заданное распределение
служит начальным спектром; отдельного закона нуклеации здесь нет.
`tail_initial_volume_ratio` задаёт среднюю начальную каплю хвоста относительно
последней точки; `maximum_tail_volume_ratio` ограничивает область замыкания.
Скалярный `breakup_rate_s` одновременно задавать нельзя.

Checkpoint `droplet_sections` хранит числа секций и последним элементом
опорный объём хвоста; `droplet_number` хранит проверяемую сумму чисел.
Отчёт содержит диаметры, объёмные доли, скорости, энергию поверхности,
перенос и реакционные источники по секциям. `droplet_number_fluxes` сохраняет
скалярный суммарный поток числа капель; `droplet_section_fluxes` содержит
потоки чисел секций и последним элементом поток опорного объёма хвоста. Адаптация времени контролирует
все секции, а не только суммарное число капель. Старые монодисперсные
checkpoint не меняют своего формата и физических уравнений.

Примеры:

- `configs/surface_pr_sectional_mixer.json`: закрытый сосуд, коалесценция,
  дробление, работа мешалки и обратная связь с температурой.
- `configs/surface_pr_sectional_physical.json`: живое броуновско-гравитационное
  ядро, дробление и механический бюджет.
- `configs/surface_pr_sectional_transport.json`: две трёхфазные ячейки,
  перенос по размерам, изоляция и повторное открытие.

`scripts/reference_pr_sections.py` независимо пересчитывает реакции
попарным NumPy-алгоритмом, объёмные моменты, скорости, поверхностную энергию
и покоординатные балансы. Основные проверяющие инструменты используют его
для обычного и гравитационного графов. Дополнительно проверено точное решение постоянного ядра Смолуховского
для всего разрешённого спектра и хвоста: контрольный режим переносит в хвост
больше половины объёмного момента без его потери. Численная верификация не является
калибровкой коэффициентов или подтверждением эффективности промышленного сепаратора.

## Statistical binary daughters

Set `daughter_distribution="symmetric_beta"` and `daughter_beta_shape=a`
(an integer from 1 through 8). The default `deterministic` mode retains the
existing 2…16 prescribed daughter fractions. Statistical mode requires the
unused deterministic fractions to remain at their default `(0.5, 0.5)` so
conflicting prescriptions fail explicitly.

For a parent reference volume V, draw X with conditional density proportional
to `[X*(1-X)]^(a-1)` on `[v_min/V, 1-v_min/V]`, then form the paired daughters
`V*X` and `V*(1-X)`. Every realization conserves volume. Shape 1 is uniform;
larger shapes favor equal splits. These are ensemble expected birth sources,
not Monte Carlo sampling. Both daughters are integrated using the symmetric
marginal, whose total number is two and volume moment is V.

The configured sectional frequency is the successful event frequency for
admissible parents, not an unconditioned attempt frequency multiplied by a
survival probability. Parents below `2*v_min` have zero statistical events;
at equality both daughters have volume `v_min`. Conditioning renormalizes the
whole admissible distribution: no sub-grid volume is discarded or individually
clamped. Deterministic runtime events are likewise suppressed if a parent is
too small for its prescribed daughters (the existing initial-grid validation
still applies). No unresolved lower reservoir is implied by this closure.

Runtime integration uses Torch only: eight-point Gauss–Legendre quadrature
on intervals split at every reference pivot. For supported integer shapes,
the density times each linear deposition weight has degree at most 15, so
integration is exact to floating point roundoff. The entire upper interval is
included, even beyond the last pivot: deposited tail number and tail volume
remain separate. Tail parents still use their mean volume; unresolved variance
within the parent tail is not reconstructed. Derivatives exist piecewise;
the admissibility threshold changes the event law discontinuously if its
prescribed rate is nonzero. Use float64 as in the transport solver.

Verification includes an analytical uniform split, independent SciPy adaptive
integration in `reference_pr_sections.py` (never called by runtime), expected
particle gain, finite minimum size, large tails, and Torch source tangents.
The reference's SciPy dependency is confined to statistical audit execution.
These checks establish discrete consistency, not calibration of breakup rates,
shape, collision efficiency, or experimental separator performance.

Scope of the three surface blocks: collision kinetics and powered breakup,
size-dependent transport, and statistical daughter closure are represented in
this implementation. This does not establish complete physical coverage or
industrial validation of those blocks. Turbulence-derived breakup frequencies,
measured daughter spectra, calibration data and experimental validation remain
external work. Reports expose `daughter_distribution` and `daughter_beta_shape`.
Default deterministic fields are omitted from normalized model serialization,
preserving pre-existing deterministic checkpoint identities.

`configs/surface_pr_sectional_beta_mixer.json` is a dedicated configuration
example using shape 3 with the existing mixer geometry and prescribed rates.
Those parameter values are verification inputs, not calibrated predictions.
