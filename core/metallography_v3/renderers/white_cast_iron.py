"""Белые чугуны + ледебурит (§1.6, §1.10 справочника).

Обслуживает:
  * ledeburite (Ld′ — леопардова шкура при 20°C)
  * white_cast_iron_hypoeutectic
  * white_cast_iron_eutectic (100% Ld′)
  * white_cast_iron_hypereutectic (+ первичный Fe₃C_I "ножи")

Phase 1 stub: регистрируется в диспетчере, но не подключается в runtime.
Реализация — отдельный sub-plan (Phase 3).
"""
from __future__ import annotations

from core.metallography_v3.renderers._common import RendererOutput
from core.metallography_v3.system_generators.base import SystemGenerationContext

HANDLES_STAGES: frozenset[str] = frozenset(
    {
        "ledeburite",
        "white_cast_iron_hypoeutectic",
        "white_cast_iron_eutectic",
        "white_cast_iron_hypereutectic",
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
        f"white_cast_iron renderer for stage {stage!r} not implemented yet — "
        "see whimsical-wandering-dawn.md Phase 3"
    )
