# Research Artifacts

Краткий manifest для исследовательских артефактов, которые не должны участвовать в runtime-загрузке пресетов, reference profiles или генерации образцов.

## Исключено из runtime

- `extracted_pages_101_168.txt`
  - Тип: исследовательский extract из литературы.
  - Назначение: ручной анализ и сверка источников.
  - Runtime-статус: исключить из discovery, preset loading и pipeline execution.

- `docs/Literature/Братковский_Шевченко_2017_анализ_отчёт.md`
  - Тип: аналитический research report.
  - Назначение: сводка по источнику, гипотезам и дальнейшим задачам.
  - Runtime-статус: исключить из discovery, preset loading и pipeline execution.

## Правило

Эти файлы считаются `research artifacts` и используются только как вспомогательные материалы для анализа и подготовки данных. Они не являются runtime-конфигурацией и не должны смешиваться с production preset library.
