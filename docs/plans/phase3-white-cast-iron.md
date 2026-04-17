# Sub-plan: Phase 3 — white_cast_iron renderer

> **Мастер-план:** `~/.claude/plans/whimsical-wandering-dawn.md`

## Context

После Phase 2 (high_temp_phases активирован через
`_PHASE2_ACTIVATED_STAGES`) переключаем второе семейство — белые чугуны
и ледебурит. Сейчас стадии `white_cast_iron_*` идут через
`_build_white_cast_iron_render` (монолит в `fe_c_unified.py`), а
`ledeburite` — через `_generic_render`.

## Область

Четыре стадии из SYSTEM_STAGE_ORDER:

| Стадия | § | Суть |
|---|---|---|
| `ledeburite` | 1.6 | Ld′ при 20°C — леопардова шкура pearlite+cementite |
| `white_cast_iron_hypoeutectic` | 1.10а | Ld′ + первичные γ-дендриты (→ перлит при 20°C) |
| `white_cast_iron_eutectic` | 1.10б | 100% Ld′ |
| `white_cast_iron_hypereutectic` | 1.10в | Ld′ + первичный Fe₃C_I-«ножи» |

Per §1.6 справочника:
- Cementite matrix: (240,238,230) Nital, контраст ΔR ≈ 140 к перлиту.
- Pearlite islands: (100,100,100), округлые 3-15 μm.
- Эвтектические ячейки 50-300 μm.

Per §1.10в:
- Primary Fe₃C_I: пластины 20-100 μm × 200-2000 μm, квази-параллельные
  внутри зон затвердевания.

## Переиспользуемые утилиты (не переписываем)

- `core/metallography_v3/system_generators/fe_c_textures.py::texture_ledeburite_leopard`
  — уже реализует §1.6: bright cementite matrix (~218) + dark pearlite
  blobs (~60) через двухмасштабный smooth noise + threshold.
- `core/metallography_v3/system_generators/fe_c_dendrites.py::render_fe_c_austenite_dendrites`
  — готовый рендер первичных γ-дендритов.
- `core/metallography_v3/system_generators/fe_c_primary_cementite.py::render_primary_cementite_needles`
  — готовый рендер пластин первичного Fe₃C_I.

Переиспользуем их с тем же API — новый модуль в
`renderers/white_cast_iron.py` оборачивает вызовы, адаптирует к
контракту `RendererOutput`, добавляет поддержку стадии `ledeburite`.

## Шаги

- [ ] **3.1** Четыре структурные карточки в `datasets/structure_cards/`:
  `ledeburite_ld_prime.json`, `white_cast_iron_hypoeutectic.json`,
  `white_cast_iron_eutectic.json`, `white_cast_iron_hypereutectic.json`.

- [ ] **3.2** `tests/renderers/test_white_cast_iron_family.py` —
  падающий семейственный тест: 4 стадии рендерятся, leopard-контраст
  ΔR ≥ 80 (§1.6), hypereutectic содержит яркие области ≥240,
  hypoeutectic — dendrite_mask покрытие ≥1%.

- [ ] **3.3** Реализовать `renderers/white_cast_iron.py`:
  - `render(stage, …)` → диспатчит по stage
  - eutectic / ledeburite → `texture_ledeburite_leopard` as-is
  - hypoeutectic → leopard + `render_fe_c_austenite_dendrites`
  - hypereutectic → leopard + `render_primary_cementite_needles`
  - `morphology_trace["family"]` = `white_cast_iron_*` для совместимости с
    `PresetIntegrationTest`.

- [ ] **3.4** Активация: добавить `ledeburite` и три `white_cast_iron_*`
  в `_PHASE3_ACTIVATED_STAGES` в `fe_c_unified.py`. Расширить существующую
  ветку диспетчера на Phase 2 ∪ Phase 3.

- [ ] **3.5** Активация tone-hierarchy теста: `_ACTIVE_STAGES` включает
  Phase 2 + Phase 3. Карточки `_CARD_TO_STAGE` + runtime defaults.

- [ ] **3.6** Регенерация `presets_baseline_hashes.json` — ожидаем drift
  у пресетов `cast_iron_*_textbook.json`.

- [ ] **3.7** Регрессионный прогон. Существующие
  `tests/test_fe_c_specialized_dispatchers.py::WhiteCastIronDispatcherTest`
  остаются зелёными (тестируют старый `_build_white_cast_iron_render` —
  dead code, но функция сохранена до финальной очистки Phase 8).
  `PresetIntegrationTest` остаётся зелёным (family string совпадает).

- [ ] **3.8** Commit + push + PR (stacked on PR #5).

## Acceptance

- Phase 3 стадии маршрутизируются через новый renderer.
- `test_white_cast_iron_family.py` PASS.
- Tone hierarchy для 4 новых карточек PASS (±55 u8 допуск).
- Snapshot baseline регенерирован.
- `test_fe_c_specialized_dispatchers.py` — PASS без изменений.
