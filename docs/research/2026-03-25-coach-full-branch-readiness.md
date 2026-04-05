# COACH Full Branch Readiness

## Что уже подготовлено
- Общая vocabulary для optics/PSF profiles:
  - `standard`
  - `bessel_extended_dof`
  - `airy_push_pull`
  - `self_rotating`
  - `stir_sectioning`
  - `lens_axicon_hybrid`
- Разделение live optics surrogate и static `pro` render surrogate.
- Metadata contract для axial shaping и sectioning.
- Research-only preset/profile path для безопасного opt-in использования.

## Чего ещё не хватает до full COACH branch
- IPSF library recording workflow
- correlation reconstruction branch over recorded library
- LRRA/INDIA runtime path
- volumetric slice selector / scene stack UI
- explicit capture/reconstruction session model for research experiments

## Почему branch пока не делается
- Текущий проект уже получил high-value optics improvements без тяжёлого architectural jump.
- Full COACH branch — это отдельный продуктовый и вычислительный контур, а не расширение текущего blur/focus layer.
