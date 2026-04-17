# Sub-plan: Phase 6 — quench_products renderer

> **Мастер-план:** `~/.claude/plans/whimsical-wandering-dawn.md`

## Context

Phase 6 подключает «закалочные» продукты мелкопластинчатого перлита
(§2.8-2.9). Сейчас `troostite_quench` и `sorbite_quench` идут через
`_build_martensitic_render` (монолит, общий с мартенситом/отпуском).

**Важно:** troostite и sorbite *закалки* принципиально отличаются от
их *отпускных* аналогов (§2.12-2.13) — разная кинетика (перегиб
С-кривой ТТТ vs. разложение мартенсита). Поэтому в Phase 6 —
именно `*_quench`; `*_temper` пойдут в Phase 7.

## Область

| Стадия | § | Морфология |
|---|---|---|
| `troostite_quench` | 2.8 | S₀≈0.1μm — **не разрешается** оптически. «Чёрные кляксы», колонии 2-10μm, часто кольца вдоль PAG. Isotropic Perlin без анизотропии. |
| `sorbite_quench` | 2.9 | S₀=0.2-0.3μm — **различима** штриховка. Колонии 5-20μm, каждой случайный θ и период p=0.25μm. |

## Шаги

- [ ] **6.1** 2 карточки: `troostite_quench.json`, `sorbite_quench.json`.

- [ ] **6.2** `tests/renderers/test_quench_products_family.py`:
  - Обе стадии рендерятся.
  - Troostite: **изотропна** — `abs(grad_x - grad_y) < 3.0` (низкая
    анизотропия); среднее очень тёмное (mean<100).
  - Sorbite: **анизотропные колонии** — большая доля зон с единообразной
    ориентацией градиента (`unimodal gradient histogram per colony`).
    Упростим: mean в диапазоне 100-160, присутствует текстурная
    вариация (std≥15).
  - Family strings.
  - Детерминизм.

- [ ] **6.3** Реализовать `renderers/quench_products.py`:
  - `_render_troostite_quench`: Poisson «ядра» с биасом к PAG-границам
    (0.05 на границах / 0.02 в объёме per μm²), radial Gaussian
    falloff σ=2-4μm. Общий фон — средне-серый (~85), клячсы тёмные
    (~40). Isotropic multiscale_noise. Tone Nital (55,45,40).
  - `_render_sorbite_quench`: Voronoi колоний 5-20μm, per-colony
    случайный θ и period p. Паттерн `I = 0.5 + 0.5·sin(2π/p · proj +
    φ)` с Perlin в φ. Tone α (190,180,165), Fe3C (80,65,55).

- [ ] **6.4** Активация `_PHASE6_ACTIVATED_STAGES` =
  frozenset(_r_quench_products.HANDLES_STAGES). Старый
  `_build_martensitic_render` продолжает обрабатывать
  martensite/tempered/bainite (только `*_quench` уходят отсюда).

- [ ] **6.5** Tone hierarchy активация для 2 карточек. Runtime defaults.

- [ ] **6.6** Regen baseline + regression + commit + push + PR.

## Acceptance

- 2 quench-стадии идут через новый renderer.
- `test_quench_products_family.py` PASS.
- `_build_martensitic_render` остаётся функциональным для
  tempered_*, bainite, generic martensite. (Те стадии уже через
  новый renderer идут martensite/_tetragonal/_cubic из Phase 4.)
