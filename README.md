# RU

## Повышение эффективности разработки газовых месторождений за счет перераспределения отборов между скважинами с использованием машинного обучения

### Cуть

В ходе работы над выпускной квалификационной работой, был проведен эксперимент, целью которого стало определить возможно ли с помощью методов машинного обучения выявить целевые параметры, регулирование которых может улучшить технологические и экономические возможности без дополнительных капитальных затрат на производство.

### Результаты

- Были разработаны утилиты по автоматическому информации с workflow выгрузки модели tnavigator/(eclipse) (см. `tools_builder`)
- Написана утилита сбору датасета (см. `tools_builder`)
- Реализованы несколько подходов в алгоритмах машинного обучение, показаны исторические изменение в рамках исследования и до настройки ГДМ модели (см. `validation` и `code models`)
- Проведены различные гипотезы по выбору целевого параметра в выборке (см. `validation` и `datasets`)
- Написано пару интерфейсы под различные алгоритмы (для удобства пользования)

> [!Анотация]
> Задача **оптимального управления разработкой газовых залежей**, в частности **задача регулирования дебитов скважин**, является актуальной.
>
> В статье показано, что регулирование дебитов в период постоянной добычи влияет на коэффициент извлечения газа (КИГ). Для получения взаимосвязи оптимальной величины направляющего дебита с известными параметрами скважины использовано машинное обучение.
> Для формирования обучающей выборки были созданы «синтетические» гидродинамические модели (ГДМ) газовых залежей, на которых воспроизведено 40 различных вариантов разработки, отличающихся числом и расположением скважин.
>
> Для каждого варианта разработки при помощи инструментов оптимизации получены лучшие варианты распределения дебитов, которые вошли в обучающую выборку. Реализованная модель использует алгоритм «Случайный лес» (Random Forest). В тестовом примере при распределении дебитов по ML-модели накопленный дисконтированный прирост добычи газа составил 164 млн.м3 (+0,56% к КИГ) при эталонном распределении (оптимизатор) плюс 255 млн.м3 , что говорит о применимости инструмента в качестве быстродействующей (но менее точной) альтернативы оптимизатору. Сделан вывод о том, что предобученные ML-модели можно использовать внутри оптимизационных алгоритмов для получения решения «первого приближения», что позволяет существенно ускорить дальнейший поиск оптимума.

# ENG

## Improving Gas Field Development Efficiency Through Production Redistribution Between Wells Using Machine Learning

### Core Concept

During the work on the final qualification thesis, an experiment was conducted. Its goal was to determine whether it is possible, using machine learning methods, to identify target parameters whose adjustment could improve technological and economic potential without additional capital expenditures on production.

### Results

- Utilities for automatically extracting information from the workflow of unloading a tnavigator/(eclipse) model were developed (see `tools_builder`).
- A utility for dataset collection was written (see `tools_builder`).
- Several approaches in machine learning algorithms were implemented; historical changes within the research scope and prior to hydrodynamic model tuning are presented (see `validation` and `code_models`).
- Various hypotheses regarding the selection of the target parameter in the sample were tested (see `validation` and `datasets`).
- A couple of interfaces for different algorithms were written (for ease of use).

> [!Abstract]
> The problem of **optimal gas field development management**, specifically the **problem of well production rate regulation**, is relevant.
>
> The article demonstrates that production rate regulation during the period of stable production affects the gas recovery factor (GRF). Machine learning was used to obtain the relationship between the optimal guide production rate and known well parameters.
> To form the training dataset, "synthetic" hydrodynamic models (HDM) of gas reservoirs were created, on which 40 different development scenarios, differing in the number and location of wells, were simulated.
>
> For each development scenario, the best production distribution options were obtained using optimization tools and included in the training dataset. The implemented model uses the "Random Forest" algorithm. In a test example, when distributing production rates according to the ML model, the cumulative discounted increase in gas production amounted to 164 million m³ (+0.56% to GRF) compared to the baseline distribution, and +255 million m³ compared to the reference distribution (optimizer). This indicates the applicability of the tool as a fast-acting (though less accurate) alternative to the optimizer. It is concluded that pre-trained ML models can be used within optimization algorithms to obtain a "first approximation" solution, which significantly speeds up the subsequent search for the optimum.
