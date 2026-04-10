# CLAUDE.md — ferro-micro project context

## Миссия проекта
Физически обоснованный генератор микрошлифов Fe-C сталей.
Репозиторий: https://github.com/Neriennaban/material-lab-fe-c-color.git

## Текущий статус
- [ ] Фаза A: физический движок (В РАБОТЕ)
- [ ] Фаза B: интеграция в проект

## Критические правила (НЕ НАРУШАТЬ)
- основной генератор не удалять
- microscope.py — не менять трогать!
- Все физические константы → constants.py с DOI
- seed обязателен везде где есть random

## Стек
Python 3.10+, NumPy, SciPy, Matplotlib, scikit-image, Pillow, tqdm

## Структура папок
текущая
+
ferro-micro/
├── ferro_micro/                    # Основной пакет
│   ├── __init__.py
│   ├── constants.py                # Физические константы + источники (DOI)
│   │
│   ├── physics/                    # Физические модели (чистая математика, без рендера)
│   │   ├── __init__.py
│   │   ├── phase_diagram.py        # Диаграмма Fe-C: правило рычага, фазовые границы
│   │   ├── phase_calculator.py     # Расчёт долей фаз по составу + термопрограмме
│   │   ├── grain_growth.py         # Модель Бека: D(t, T, C)
│   │   ├── pearlite.py             # λ(ΔT), скорость роста, ориентация
│   │   ├── cct_model.py            # CCT/TTT: Аврами, правило Шейла, Ms-temperature
│   │   └── ledeburite.py           # Эвтектика, ледебурит, графитизация
│   │
│   ├── render/                     # Рендер-движок
│   │   ├── __init__.py
│   │   ├── voronoi_engine.py       # Voronoi/Laguerre разбиение с джиттером
│   │   ├── phase_renderer.py       # Растеризация каждой фазы в маску
│   │   ├── texture_generator.py    # Перлит (полосы), мартенсит (иглы), ледебурит
│   │   ├── etching_simulator.py    # Дифференциальный контраст, шум, гамма
│   │   ├── compositor.py           # Финальное смешение фаз + постобработка
│   │   └── tile_renderer.py        # Тайловый рендер для разрешений 4K–16K
│   │
│   ├── api/                        # Публичный интерфейс
│   │   ├── __init__.py
│   │   ├── generator.py            # MicrostructureGenerator — главный класс
│   │   ├── schemas.py              # Датаклассы: CompositionFeC, ThermalProgram, EtchingParams
│   │   ├── presets.py              # Пресеты: Ст3, 45, У8, У12, СЧ20, ХВГ
│   │   └── exceptions.py           # OutOfPhysicalRangeError, CalibrationNeededError
│   │
│   ├── integration/                # Адаптер для подключения к microscope.py
│   │   ├── __init__.py
│   │   ├── microscope_adapter.py   # Мост ferro-micro ↔ microscope.py
│   │   └── interface_contract.py   # Типы и протоколы совместимости
│   │
│   └── utils/
│       ├── __init__.py
│       ├── memory.py               # Управление памятью при 4K–16K генерации
│       ├── cache.py                # LRU-кэш Voronoi-сеток
│       └── profiler.py             # Таймеры и memory profiler
│
├── research/                       # Результаты литературного обзора
│   ├── literature_review.md        # Полный обзор (выход задачи 1)
│   ├── constants_sources.md        # Таблица констант + DOI-источники
│   └── calibration_data/           # CSV с экспериментальными данными для калибровки
│
├── tests/
│   ├── unit/
│   │   ├── test_phase_calculator.py
│   │   ├── test_grain_growth.py
│   │   ├── test_pearlite.py
│   │   └── test_etching.py
│   ├── integration/
│   │   ├── test_full_pipeline.py   # Smoke-тест: каждый из 7 классов
│   │   └── test_microscope_integration.py
│   └── visual/
│       ├── benchmark_images/       # Эталонные микрофотографии (7 классов)
│       └── test_ssim.py            # SSIM-сравнение с эталоном
│
├── examples/
│   ├── basic_usage.py
│   ├── generate_16k.py             # Демо тайловой генерации
│   └── all_steel_grades.py
│
├── docs/
│   ├── physical_models.md          # Документация физических моделей
│   └── api_reference.md
│
├── legacy/                         # Старый код (НЕ УДАЛЯТЬ до завершения интеграции)
│   ├── generator.py                # Оригинальный generator.py (перемещается сюда)
│   └── AUDIT_REPORT.md             # Результат аудита (задача 2)
│
├── pyproject.toml
├── README.md
└── CHANGELOG.md

## Где остановились
[обновляй вручную перед каждой ночной сессией]

## Git-правила для агента
- Коммить после каждого завершённого модуля: git commit -m "feat: physics/grain_growth.py"
- Никогда не делать force push
- Перед началом работы: git pull origin main