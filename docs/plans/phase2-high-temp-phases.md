# Sub-plan: Phase 2 — high_temp_phases renderer

> **Мастер-план:** `~/.claude/plans/whimsical-wandering-dawn.md` (или
> `docs/plans/master-phase1-infrastructure.md` в будущем).

## Context

После Phase 1 (каркас `renderers/` + диспетчер `_STAGE_TO_RENDERER` +
`StructureCard` loader) подключаем первое семейство — высокотемпературные
и жидкие фазы. Почему первыми:

- Относительно простая морфология (равноосные полиэдры, вермикулярные
  островки, дендриты L-системы) → низкий риск регрессий в пользовательских
  пресетах.
- Стадии редко активируются напрямую — большинство пресетов обращается к
  `alpha_pearlite`/`pearlite`/`martensite`. Поэтому даже при минимальных
  недочётах рендера воздействие ограничено.
- Позволяет отшлифовать паттерн структурных карточек и связку
  «renderer ↔ `_STAGE_TO_RENDERER` ↔ тесты» до более сложных семейств.

## Область

Шесть стадий из SYSTEM_STAGE_ORDER:

| Стадия | § | Морфология | Приоритет |
|---|---|---|---|
| `austenite` | 1.4 | Voronoi 20-100 μm + annealing twins 1-5 μm | High |
| `delta_ferrite` | 1.5 | Анизотропный Simplex-шум stretch_x=3 + threshold | High |
| `liquid` | — | Квази-однородное поле с low-frequency noise | Low (редко используется) |
| `liquid_gamma` | 3.1 | Liquid + дендритный γ-скелет (PDAS/SDAS) | Medium |
| `alpha_gamma` | — | Равноосные γ + тонкая ферритная «оболочка» | Low |
| `gamma_cementite` | — | γ + тёмные сферические Fe₃C on grain boundaries | Low |

**DoD (Definition of Done) по Phase 2:**
- [ ] Каждая карточка `datasets/structure_cards/<stage>.json` заведена с RGB/morphology/composition_triggers из §N.
- [ ] `renderers/high_temp_phases.py::render()` реализован для всех 6 стадий (не `NotImplementedError`).
- [ ] В `fe_c_unified.py::render_fe_c_unified` добавлен ранний dispatch в `_STAGE_TO_RENDERER` для этих 6 стадий (старые пути `_generic_render` остаются для остальных).
- [ ] `tests/renderers/test_tone_hierarchy.py::test_rendered_mean_tones_match_card` активируется для карточек семейства; ±20% допуск.
- [ ] `tests/renderers/test_high_temp_phases_family.py` — 6 smoke-тестов (по одному на стадию) + тест на наличие twins внутри austenite + тест на анизотропию δ-ferrite.
- [ ] `tests/test_v3_presets_snapshot_baseline.py` регенерирован; все остальные регрессии зелёные.
- [ ] Никаких `NotImplementedError` из этих 6 стадий не проходит наружу.

## Последовательность шагов (TDD)

- [ ] **2.1** Написать 6 структурных карточек (datasets/structure_cards/).
  RGB из §1.4 (Нитал 230,228,222 / границы 90,85,80), §1.5 (Нитал 150,145,135)
  и т.п.

- [ ] **2.2** Написать `tests/renderers/test_high_temp_phases_family.py` —
  падающий тест: каждая из 6 стадий должна отрендериться без исключения,
  mean tone в разумном диапазоне [40,250], phase_masks не пустые.

- [ ] **2.3** Запустить тест → FAIL (stub'ы бросают NotImplementedError).

- [ ] **2.4** Реализовать `renderers/high_temp_phases.py::render()`:
  - `austenite` → Voronoi через `_grain_map`, tone (230,228,222), границы
    (90,85,80), twins через `build_twins_from_labels` из `realism_utils`
    (probability 0.4, width 1-5 μm в px).
  - `delta_ferrite` → фон austenite, поверх — анизотропный
    `multiscale_noise` с stretch_x (scales=((26,0.7),(9,0.3)), threshold
    0.65 → маска δ-островков 2-15%). Tone островков (150,145,135).
  - `liquid` → low_frequency_field sigma=24 + multiscale_noise → gradient
    от (245,245,240) (hot) до (210,205,195) (cold), без границ зёрен.
  - `liquid_gamma` → liquid фон + L-система дендритов через
    existing `fe_c_dendrites.py` (SDAS ~ 40.7·Ṙ^(-1/3) μm).
  - `alpha_gamma` → austenite с тонкой ферритной сеткой на границах
    (dilate boundary, tone (225,222,215)).
  - `gamma_cementite` → austenite + Poisson-disk карбиды по границам,
    радиус lognormal(1, 0.3) μm, tone (45,35,25).

- [ ] **2.5** Изменить `fe_c_unified.py::render_fe_c_unified`:
  добавить ранний dispatch на `_STAGE_TO_RENDERER` **только для стадий
  high_temp_phases**. Остальные стадии (мартенсит/бейнит/чугуны/отпуски)
  продолжают идти по старым путям — они будут подключены в Phase 3-8.

  ```python
  _PHASE2_ACTIVATED_STAGES = frozenset({
      "austenite", "delta_ferrite", "alpha_gamma",
      "gamma_cementite", "liquid", "liquid_gamma",
  })
  # ... после _is_pure_iron_like / до старого _SPECIALIZED_* dispatch:
  if stage in _PHASE2_ACTIVATED_STAGES and stage in _STAGE_TO_RENDERER:
      out = _STAGE_TO_RENDERER[stage].render(
          context=context, stage=stage,
          phase_fractions=phase_fractions, seed_split=seed_split,
      )
      image_gray = out.image_gray
      phase_masks = out.phase_masks
      morphology_trace = out.morphology_trace
      rendered_layers = out.rendered_layers or list(phase_masks.keys())
      fragment_area = out.fragment_area or 0
  ```

- [ ] **2.6** Запустить тесты семейства → PASS.

- [ ] **2.7** Регенерировать snapshot baseline:
  `python -m tests.test_v3_presets_snapshot_baseline --regenerate`.
  Ожидается drift только у пресетов, где итоговая `final_stage` попадает
  в шестёрку Phase 2.

- [ ] **2.8** Активировать параметризованный `test_tone_hierarchy` для
  новых карточек: убрать skip, добавить реальную проверку mean tones.

- [ ] **2.9** Регрессионный прогон всех тестов `tests/`.

- [ ] **2.10** Commit + push + PR.

## Acceptance

- Все 6 стадий активно маршрутизируются через новый renderer.
- Старые `_generic_render`/`_build_*_render` для них не вызываются.
- Пресеты, чьи thermal programs приводят к этим стадиям (в основном
  liquid/austenite на высоких T в отдельных тестовых пресетах),
  продолжают рендериться; baseline регенерирован.
- Визуально (headless-проверка): austenite светлый с twins, δ-ferrite
  имеет вытянутые островки вдоль одной оси, остальные — гладкие.
