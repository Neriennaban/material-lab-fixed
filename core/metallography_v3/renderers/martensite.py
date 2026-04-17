"""Мартенситное семейство (§2.1–2.4 справочника).

Обслуживает: lath (§2.1), plate/lenticular с midrib (§2.2),
mixed (§2.3), retained austenite как под-фаза (§2.4).

Phase 1 stub: регистрируется в диспетчере, но не подключается в runtime.
Реализация — отдельный sub-plan (Phase 4).
"""
from __future__ import annotations

from core.metallography_v3.renderers._common import RendererOutput
from core.metallography_v3.system_generators.base import SystemGenerationContext

HANDLES_STAGES: frozenset[str] = frozenset(
    {
        "martensite",
        "martensite_tetragonal",
        "martensite_cubic",
    }
)


def render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> RendererOutput:
    raise NotImplementedError(
        f"martensite renderer for stage {stage!r} not implemented yet — "
        "see whimsical-wandering-dawn.md Phase 4"
    )
