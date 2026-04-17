# Sub-plan: Phase 8 — widmanstatten + surface_layers + granular_pearlite

> **Мастер-план:** `~/.claude/plans/whimsical-wandering-dawn.md`

## Context

Последняя фаза: 4 новые стадии, каждая регистрируется в
SYSTEM_STAGE_ORDER / _STAGE_DEFAULT_FRACTIONS / UI labels / карточках.

## Область

| Стадия | § | Морфология |
|---|---|---|
| `widmanstatten_ferrite` | 2.10 | Иглы феррита 50-500 × 2-20 μm в направлениях {60°, 120°} из границ PAG + аллотриоморфный феррит на PAG-границах + перлит между иглами |
| `decarburized_layer` | 3.2 | Градиент `C(y) = C_core · erfc(y/depth_eff)` — построчная композиция: FFD (чистый феррит у поверхности) → MAD (частичный) → исходный феррит+перлит |
| `carburized_layer` | 3.3 | `C(y) = C_surface · erfc(y/CD_eff)` — построчная: заэвтектоидный мартенсит на поверхности → эвтектоидный → доэвтектоидный → сердцевина. |
| `granular_pearlite` | 1.9 | Ферритная матрица 5-20 μm + Poisson-disk глобули Fe₃C lognormal(0.5, 0.3) clip [0.1, 2] μm, плотность 0.2-0.6/μm² |

## Шаги

- [ ] **8.1** 4 карточки: `widmanstatten_ferrite.json`,
  `decarburized_layer.json`, `carburized_layer.json`, `granular_pearlite.json`.

- [ ] **8.2** Регистрация 4 новых стадий:
  - `core/generator_phase_map.py::SYSTEM_STAGE_ORDER["fe-c"]`
  - `fe_c_unified.py::_STAGE_DEFAULT_FRACTIONS`
  - `fe_c_unified.py::_TRANSITION_STAGES`
  - `ui_qt/sample_factory_window_v3.py::STRUCTURE_STAGE_LABELS_RU`

- [ ] **8.3** Family tests:
  - `test_widmanstatten_family.py` — иглы с 60°/120° направлением,
    pearlite-фон между иглами.
  - `test_surface_layers_family.py` — decarb: вертикальный градиент
    яркости (верх светлее), carb: обратный градиент + dark martensite
    у поверхности.
  - `test_granular_pearlite_family.py` — светлая ферритная матрица +
    тёмные точки карбидов ≥1% площади.

- [ ] **8.4** Реализация:
  - `renderers/widmanstatten.py`: PAG Voronoi + allotriomorph boundary
    dilate + needle rasterization в 2-3 направлениях 60°-apart +
    pearlite fallback между иглами.
  - `renderers/surface_layers.py`: построчный C(y), разные renderers
    per y-band. decarb: верх=ferrite, середина=alpha_pearlite
    с ростом pearlite fraction, низ=базовый α+P. carb: верх=plate
    martensite, середина=lath martensite, низ=α+P.
  - `renderers/granular_pearlite.py`: `generate_pure_ferrite_micrograph`
    с mean_eq_d_px=12 + Poisson-disk cementite globules.

- [ ] **8.5** Dispatcher: добавить _PHASE8_ACTIVATED_STAGES.

- [ ] **8.6** Tone hierarchy + _CARD_TO_STAGE + runtime defaults.

- [ ] **8.7** Regen baseline + regression + commit + push + PR.

## Acceptance

- 4 новые стадии работают end-to-end через pipeline.
- Family tests зелёные.
- Tone hierarchy тесты активированы.
- Это последний feature-PR перед финальной очисткой.
