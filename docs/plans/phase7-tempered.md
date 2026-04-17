# Sub-plan: Phase 7 — tempered renderer

> **Мастер-план:** `~/.claude/plans/whimsical-wandering-dawn.md`

## Context

Phase 7 закрывает 5 стадий отпуска мартенсита (§2.11–2.13):
`tempered_low`, `tempered_medium`, `tempered_high`, и их дубликаты
`troostite_temper` / `sorbite_temper`. После Phase 4-6 на
`_build_martensitic_render` остаются именно эти + `bainite` общий —
после Phase 7 там останется только `bainite` до финальной очистки.

## Область

| Стадия | T, °C | § | Морфология |
|---|---|---|---|
| `tempered_low` | 150-250 | 2.11 | Реечная база × 0.72 яркости + тёплый сдвиг + ε-карбиды под 60° |
| `tempered_medium` / `troostite_temper` | 350-500 | 2.12 | Blur σ=0.3μm поверх реек + высокочастотный Perlin + 60% точек карбидов на бывших границах реек |
| `tempered_high` / `sorbite_temper` | 500-650 | 2.13 | Полигональный феррит 2-10μm + Poisson-карбиды 0.3μm, 70% на границах. НЕТ γост, midrib, реечной геометрии |

## Шаги

- [ ] **7.1** 3 карточки: `tempered_low.json`, `tempered_medium.json`,
  `tempered_high.json`. (`troostite_temper` / `sorbite_temper` используют
  те же карточки по маппингу stage → card.)

- [ ] **7.2** `tests/renderers/test_tempered_family.py`:
  - 5 stage'ей рендерятся.
  - `tempered_low` темнее и теплее чем `martensite_cubic` (как отпуск
    от реечной базы).
  - `tempered_medium` — тёмная «бархатистая» текстура с высокочастотным
    шумом (std≥8).
  - `tempered_high` — светлая полигональная матрица (mean≥130) с
    тёмными точечными карбидами.
  - Family strings; детерминизм.

- [ ] **7.3** Реализовать `renderers/tempered.py`:
  - `_render_tempered_low(C=c_wt)`:
    - Взять реечную базу от `martensite._render_lath` (переиспользование).
    - Затемнить `img *= 0.72`, сдвиг тёплый.
    - Добавить Poisson ε-карбидов (tone ≤30) под углом 60° к per-PAG
      ориентации, плотность ~0.5/μm².
  - `_render_tempered_medium`:
    - Реечная база.
    - `ndimage.gaussian_filter(img, sigma=1.0)` — размытие имитирует
      субμm рекристаллизацию.
    - Поверх: высокочастотный `multiscale_noise` scale=3, oct=4,
      persist=0.6, amp=15.
    - Poisson-карбиды 50-200 нм (≈1-2 px) на бывших границах (60% bias).
  - `_render_tempered_high`:
    - Новая Voronoi d=4-6 px (мелкие ферритные зёрна).
    - Заливка tone ~158 (§2.13 matrix).
    - Границы зёрен tone 75 (§2.13 boundaries).
    - Poisson карбидов: lognormal размер ~0.3 μm, 70% на границах /
      30% внутри. Карбиды tone 30 (§2.13 carbides).
    - Лёгкий Gaussian blur для smooth edges.
  - `_render_troostite_temper` = `_render_tempered_medium` (alias).
  - `_render_sorbite_temper` = `_render_tempered_high` (alias).

- [ ] **7.4** Диспетчер: `_PHASE7_ACTIVATED_STAGES` =
  frozenset(_r_tempered.HANDLES_STAGES). _ACTIVATED_RENDERER_STAGES ∪=.

- [ ] **7.5** Tone hierarchy: добавить 3 карточки в runtime defaults,
  расширить _PHASE_KEYS (TROOSTITE_TEMPER / SORBITE_TEMPER / FERRITE
  ключи).

- [ ] **7.6** Regen baseline + regression + commit + push + PR.

## Acceptance

- 5 tempered-stage'ей (tempered_low/medium/high + troostite_temper +
  sorbite_temper) идут через `renderers/tempered.py`.
- Существующие тесты остаются зелёными (`_build_martensitic_render`
  продолжает обрабатывать только `bainite` — последний stub до
  финальной очистки Phase 8).
