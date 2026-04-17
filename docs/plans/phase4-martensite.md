# Sub-plan: Phase 4 — martensite renderer

> **Мастер-план:** `~/.claude/plans/whimsical-wandering-dawn.md`

## Context

После Phase 3 активированы high_temp_phases и white_cast_iron. Phase 4
подключает третье, самое объёмное семейство — мартенситные стадии §2.1–2.4.

Сейчас эти стадии идут через `_build_martensitic_render` (монолит в
`fe_c_unified.py`), общий для 11 стадий (martensite*, troostite*,
sorbite*, bainite, tempered_*). В Phase 4 **только martensite/_tetragonal
/_cubic** переключаются на новый `renderers/martensite.py`; остальные
остаются на `_build_martensitic_render` до своих Phase 5/6/7 sub-plan'ов.

## Область

Три стадии SYSTEM_STAGE_ORDER + retained austenite как под-фаза:

| Стадия | § | Морфология |
|---|---|---|
| `martensite_cubic` | 2.1 | Реечный (C < 0.6%): PAG → packet → block → subblock → lath (0.2-2 μm × 5-50 μm), анизотропия 10:1 |
| `martensite_tetragonal` | 2.2 | Пластинчатый (C > 1.0%): линзы 10-200 μm × 1-10 μm, midrib 0.1-1 μm (40-50 RGB темнее тела), направления {θ₀, θ₀±60°, θ₀±120°, θ₀+90°} |
| `martensite` | 2.3 | Смешанный (C=0.6-1.0%): `plate_fraction = clip((C-0.4)/0.6, 0, 1)` — сначала пластины, остаток реечный |
| `retained_austenite` | 2.4 | Плёнки (inter-lath, 5-30 нм — не разрешаются) + блоки 0.5-10 μm; tone nital (225,220,210); уже инжектится пост-обработкой в `render_fe_c_unified:~1770+`, **renderer не трогает** |

## Переиспользуемые утилиты

- `_grain_map` (Voronoi PAG) — для lath-иерархии
- `build_twins_from_labels` — можно переиспользовать для имитации
  направленных полос внутри зёрен (блочная структура)
- `multiscale_noise` с анизотропным stretch — для бархатного фона
  реек/плит
- `boundary_mask_from_labels` — PAG-границы

## Шаги

- [ ] **4.1** Четыре структурных карточки: `martensite_lath.json` (уже
  есть с Phase 1, дополняю), новые `martensite_plate.json`,
  `martensite_mixed.json`, `retained_austenite.json`.

- [ ] **4.2** `tests/renderers/test_martensite_family.py`:
  - `test_all_stages_render` — 3 стадии рендерятся, mean в [30,180].
  - `test_cubic_is_anisotropic` — реечный мартенсит имеет выраженную
    ориентационную анизотропию: разница stddev проекций вдоль случайных
    осей ≥ 8%.
  - `test_tetragonal_has_needle_structure` — пластинчатый имеет чёткие
    тёмные midrib линии (процент пикселей ≤45 tone ≥ 1%).
  - `test_mixed_darkens_with_carbon` — `martensite @ C=0.8` содержит
    больше midrib-подобных структур чем `martensite_cubic @ C=0.3`.
  - `test_family_trace_strings` — `morphology_trace.family` =
    `martensite_lath` / `martensite_plate` / `martensite_mixed`.
  - `test_determinism`.

- [ ] **4.3** Реализовать `renderers/martensite.py`:
  - `_render_lath` (cubic): Voronoi PAG 50 μm → per-PAG orientation →
    анизотропный `multiscale_noise` (scale_long=80 px, scale_trans=8 px)
    + stripe-pattern вдоль ориентации (sine wave shifted).
    Тоны: laths (70,60,55), boundaries (25,20,18).
  - `_render_plate` (tetragonal): Voronoi PAG → для каждой PAG
    генерация N линзовидных игл по направлениям
    [θ₀, θ₀±60°, θ₀+90°]; первая игла самая длинная, следующие меньше
    (`L_i = max_L × 0.55**(i/N)`); рисование через Bresenham-like
    rasterization лента width W + midrib line 1 px tone −50.
  - `_render_mixed`: гибрид — сначала _render_plate с ограниченным
    plate_fraction = clip((C-0.4)/0.6, 0, 1), остаток заливается
    _render_lath-текстурой с уменьшенными параметрами.

- [ ] **4.4** Добавить `_PHASE4_ACTIVATED_STAGES` =
  `frozenset(_r_martensite.HANDLES_STAGES)` и расширить
  `_ACTIVATED_RENDERER_STAGES`. Важно: `_build_martensitic_render`
  **продолжает** обрабатывать troostite/sorbite/bainite/tempered_*.

- [ ] **4.5** Регенерация snapshot baseline — ожидается drift у всех
  мартенситных пресетов (AISI4140_hardened и т.п.).

- [ ] **4.6** Активация tone-hierarchy и morphology-anisotropy тестов
  для `martensite_*` карточек. Для _tetragonal, _mixed — активация
  morphology anisotropy теста (Fourier или Radon peaks), без строгой
  проверки точных углов.

- [ ] **4.7** Commit + push + PR.

## Acceptance

- Все 3 martensite стадии рендерятся через новый renderer.
- `test_martensite_family.py` PASS.
- `test_fe_c_unified_stage_coverage` PASS.
- Остальные мартенситные стадии (troostite/sorbite/bainite/tempered_*)
  **не затронуты** — идут через `_build_martensitic_render` как раньше.
- RA-инжекция в post-process остаётся рабочей (`renderer` не трогает).
