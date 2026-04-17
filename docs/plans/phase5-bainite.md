# Sub-plan: Phase 5 — bainite renderer (+ CFB новая стадия)

> **Мастер-план:** `~/.claude/plans/whimsical-wandering-dawn.md`

## Context

Phase 5 подключает бейнитное семейство. Сейчас `bainite_upper` и
`bainite_lower` идут через `_build_bainitic_render_split` (монолит
в `fe_c_unified.py`, `_SPECIALIZED_BAINITIC_STAGES`). `bainite`
(общая стадия) идёт через `_build_martensitic_render`. `carbide_free_bainite`
(§2.7) вообще отсутствует в `SYSTEM_STAGE_ORDER`.

## Область

| Стадия | § | Морфология | Статус |
|---|---|---|---|
| `bainite_upper` | 2.5 | Перистый (feathery) веер пучков 20-100 μm, цементит плёнки ∥ оси | Существует |
| `bainite_lower` | 2.6 | Иглы 1-3 × 20-80 μm, внутренние карбиды под 60° к оси, ОДНО направление в игле | Существует |
| `carbide_free_bainite` | 2.7 | Si≥1.5%, бархатный нанобейнит + блоки γR, БЕЗ точечных карбидов | **Новая стадия** |

## Шаги

- [ ] **5.1** Карточки: `bainite_upper.json` (уже есть с Phase 1, оставляем),
  новые `bainite_lower.json`, `bainite_cfb.json`.

- [ ] **5.2** `tests/renderers/test_bainite_family.py`:
  - 3 стадии рендерятся, маски не пустые.
  - Upper: feathery анизотропия (max grad ≥ 4).
  - Lower: наличие тёмных игл (дискретные dark regions ≥1% pixel ≤60).
  - CFB: отсутствие точечных карбидов (доля ≤50 tone малых областей
    <2% — мягкий критерий), наличие светлых блоков RA.
  - Family strings.
  - Детерминизм.

- [ ] **5.3** Зарегистрировать `carbide_free_bainite` как стадию:
  - `core/generator_phase_map.py:SYSTEM_STAGE_ORDER["fe-c"]` +
    `"carbide_free_bainite"` после `bainite_lower`.
  - `fe_c_unified.py::_STAGE_DEFAULT_FRACTIONS` — BAINITE 0.70 + AUSTENITE 0.25 + MARTENSITE 0.05.
  - `ui_qt/sample_factory_window_v3.py` STRUCTURE_STAGE_LABELS_RU —
    "Безкарбидный бейнит".
  - `phase_orchestrator.py` — условие активации **в будущем** (при
    Si≥1.5% + изотерма 200-400°C); сейчас стадия доступна только при
    явном указании в пресете.

- [ ] **5.4** Реализовать `renderers/bainite.py`:
  - `_render_upper`: анизотропный шум + стрипы-цементит вдоль ориентации,
    feathery packets. Tones matrix (145,130,118), cementite films (70,55,45).
  - `_render_lower`: Poisson иглы ~1-3 × 20-80 px в случайных
    направлениях; внутри каждой — короткие «штрихи» (dash) под 60° к
    оси иглы (внутрипластинчатые карбиды). Tone matrix (95,80,72),
    штрихи (30,22,18), фон (130,115,105).
  - `_render_cfb`: очень мелкий анизотропный шум (scale_long=20,
    scale_trans=2) 10:1, тон 85-110 + блоки γR 0.5-3 μm (lognormal),
    tone (225,220,210). **НЕТ** точечных карбидов.

- [ ] **5.5** Активация: `_PHASE5_ACTIVATED_STAGES` =
  frozenset(_r_bainite.HANDLES_STAGES). _ACTIVATED_RENDERER_STAGES ∪=
  Phase 5. Старый `_build_bainitic_render_split` остаётся как dead code.

- [ ] **5.6** Tone hierarchy: активация для 3 новых карточек,
  расширение `_PHASE_KEYS` для BAINITE.

- [ ] **5.7** Регенерация snapshot baseline — ожидается drift у
  bainite-содержащих пресетов (`steel_45_upper_bainite_klemm_v3` и т.п.).

- [ ] **5.8** Regression + commit + push + PR.

## Acceptance

- 3 bainite стадии маршрутизируются через `renderers/bainite.py`.
- `carbide_free_bainite` зарегистрирован и рендерится.
- `test_bainite_family.py` PASS.
- `test_fe_c_specialized_dispatchers` PASS (старый
  `_build_bainitic_render_split` продолжает тестироваться напрямую —
  dead code, но функция осталась).
