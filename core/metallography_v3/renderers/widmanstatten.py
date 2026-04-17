"""Видманштеттов феррит (§2.10 справочника).

Обслуживает: widmanstatten_ferrite — иглы 50–500 × 2–20 μm в направлениях
{60°, 120°} из границ PAG, на фоне перлита.

Планируется переиспользовать ``widmanstatten_field`` из
``core/metallography_pro/morphology_fe_c.py`` — в Phase 8 первым шагом
smoke-тест совместимости с новым ``_grain_map``.

Phase 1 stub: регистрируется в диспетчере, но не подключается в runtime.
Реализация — отдельный sub-plan (Phase 8).
"""
from __future__ import annotations

from core.metallography_v3.renderers._common import RendererOutput
from core.metallography_v3.system_generators.base import SystemGenerationContext

HANDLES_STAGES: frozenset[str] = frozenset({"widmanstatten_ferrite"})


def render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> RendererOutput:
    raise NotImplementedError(
        f"widmanstatten renderer for stage {stage!r} not implemented yet — "
        "see whimsical-wandering-dawn.md Phase 8"
    )
