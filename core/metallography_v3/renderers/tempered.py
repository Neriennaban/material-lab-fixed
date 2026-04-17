"""Отпуск мартенсита (§2.11–2.13 справочника).

Обслуживает:
  * tempered_low (§2.11, 150–250°C, ε-карбиды внутри реек)
  * tempered_medium / troostite_temper (§2.12, 350–500°C)
  * tempered_high / sorbite_temper (§2.13, 500–650°C, Q+T "улучшение")

Phase 1 stub: регистрируется в диспетчере, но не подключается в runtime.
Реализация — отдельный sub-plan (Phase 7).
"""
from __future__ import annotations

from core.metallography_v3.renderers._common import RendererOutput
from core.metallography_v3.system_generators.base import SystemGenerationContext

HANDLES_STAGES: frozenset[str] = frozenset(
    {
        "tempered_low",
        "tempered_medium",
        "tempered_high",
        "troostite_temper",
        "sorbite_temper",
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
        f"tempered renderer for stage {stage!r} not implemented yet — "
        "see whimsical-wandering-dawn.md Phase 7"
    )
